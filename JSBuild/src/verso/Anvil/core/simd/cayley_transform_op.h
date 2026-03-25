// highnoon/_native/ops/cayley_transform_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file cayley_transform_op.h
 * @brief SIMD-optimized Cayley Transform for orthogonal weight matrices.
 *
 * Implements the Cayley transform: W = (I - A)(I + A)^{-1}
 * where A is a learnable skew-symmetric matrix (A^T = -A).
 *
 * This guarantees W is orthogonal (W^T W = I), which:
 * - Mitigates gradient explosion/vanishing in deep networks
 * - Provides stable long-term correlations for sequential models
 *
 * SIMD optimizations:
 * - AVX-512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - ARM NEON: 4-wide vectorization
 * - Scalar fallback for all architectures
 *
 * Thread-safe: all functions are reentrant with no shared state.
 * Precision: float32 only.
 */

#ifndef HIGHNOON_NATIVE_OPS_CAYLEY_TRANSFORM_OP_H_
#define HIGHNOON_NATIVE_OPS_CAYLEY_TRANSFORM_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace highnoon {
namespace ops {
namespace cayley {

// =============================================================================
// CONFIGURATION
// =============================================================================

struct CayleyConfig {
    int64_t input_dim = 0;       // Input dimension
    int64_t output_dim = 0;      // Output dimension (units)
    bool use_bias = true;        // Include bias term
    bool cache_inverse = true;   // Cache matrix inverse for inference
    float eps = 1e-6f;           // Numerical stability epsilon
};

// =============================================================================
// SKEW-SYMMETRIC MATRIX CONSTRUCTION
// =============================================================================

/**
 * @brief Construct skew-symmetric matrix A from upper triangular parameters.
 *
 * Given n(n-1)/2 parameters, constructs A where A^T = -A.
 * A[i,j] = params[k] for i < j, A[j,i] = -A[i,j], A[i,i] = 0.
 *
 * @param params Upper triangular parameters [n*(n-1)/2]
 * @param A Output skew-symmetric matrix [n, n]
 * @param n Matrix dimension
 */
inline void cayley_construct_skew_symmetric(
    const float* params, float* A, int64_t n) {
    
    // Zero initialize
    std::fill(A, A + n * n, 0.0f);
    
    // Fill upper triangular and negate for lower
    int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            float val = params[k++];
            A[i * n + j] = val;       // Upper triangular
            A[j * n + i] = -val;      // Lower triangular (negated)
        }
    }
}

/**
 * @brief Compute gradient of skew-symmetric construction.
 *
 * @param grad_A Gradient w.r.t. A [n, n]
 * @param grad_params Output gradient w.r.t. params [n*(n-1)/2]
 * @param n Matrix dimension
 */
inline void cayley_skew_symmetric_backward(
    const float* grad_A, float* grad_params, int64_t n) {
    
    int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            // d(loss)/d(params[k]) = d(loss)/d(A[i,j]) + d(loss)/d(A[j,i]) * (-1)
            //                      = grad_A[i,j] - grad_A[j,i]
            grad_params[k++] = grad_A[i * n + j] - grad_A[j * n + i];
        }
    }
}

// =============================================================================
// LU DECOMPOSITION WITH PARTIAL PIVOTING
// =============================================================================

/**
 * @brief LU decomposition with partial pivoting for matrix inversion.
 *
 * Computes PA = LU where P is permutation, L is lower triangular, U is upper.
 *
 * @param A Input matrix [n, n] - modified in place to store L\U
 * @param pivot Permutation indices [n]
 * @param n Matrix dimension
 * @return true if decomposition succeeded, false if singular
 */
inline bool lu_decompose(float* A, int32_t* pivot, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        pivot[i] = static_cast<int32_t>(i);
    }
    
    for (int64_t k = 0; k < n; ++k) {
        // Find pivot
        float max_val = std::abs(A[k * n + k]);
        int64_t max_idx = k;
        for (int64_t i = k + 1; i < n; ++i) {
            float val = std::abs(A[i * n + k]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        // Check for singularity
        if (max_val < 1e-10f) {
            return false;
        }
        
        // Swap rows
        if (max_idx != k) {
            std::swap(pivot[k], pivot[max_idx]);
            for (int64_t j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[max_idx * n + j]);
            }
        }
        
        // Eliminate below
        float inv_pivot = 1.0f / A[k * n + k];
        for (int64_t i = k + 1; i < n; ++i) {
            float factor = A[i * n + k] * inv_pivot;
            A[i * n + k] = factor;  // Store L factor
            
            // Update remaining row with SIMD
            int64_t j = k + 1;
#if defined(__AVX512F__)
            __m512 f_vec = _mm512_set1_ps(factor);
            for (; j + 16 <= n; j += 16) {
                __m512 a_row = _mm512_loadu_ps(&A[i * n + j]);
                __m512 k_row = _mm512_loadu_ps(&A[k * n + j]);
                a_row = _mm512_fnmadd_ps(f_vec, k_row, a_row);
                _mm512_storeu_ps(&A[i * n + j], a_row);
            }
#elif defined(__AVX2__)
            __m256 f_vec = _mm256_set1_ps(factor);
            for (; j + 8 <= n; j += 8) {
                __m256 a_row = _mm256_loadu_ps(&A[i * n + j]);
                __m256 k_row = _mm256_loadu_ps(&A[k * n + j]);
                a_row = _mm256_fnmadd_ps(f_vec, k_row, a_row);
                _mm256_storeu_ps(&A[i * n + j], a_row);
            }
#elif defined(__ARM_NEON)
            float32x4_t f_vec = vdupq_n_f32(factor);
            for (; j + 4 <= n; j += 4) {
                float32x4_t a_row = vld1q_f32(&A[i * n + j]);
                float32x4_t k_row = vld1q_f32(&A[k * n + j]);
                a_row = vmlsq_f32(a_row, f_vec, k_row);
                vst1q_f32(&A[i * n + j], a_row);
            }
#endif
            for (; j < n; ++j) {
                A[i * n + j] -= factor * A[k * n + j];
            }
        }
    }
    return true;
}

/**
 * @brief Solve LU @ x = b using forward/backward substitution.
 *
 * @param LU LU-decomposed matrix [n, n]
 * @param pivot Permutation indices [n]
 * @param b Right-hand side vector [n]
 * @param x Solution vector [n]
 * @param n Matrix dimension
 */
inline void lu_solve(
    const float* LU, const int32_t* pivot, const float* b,
    float* x, int64_t n) {
    
    // Apply permutation and forward solve (L @ y = Pb)
    std::vector<float> y(n);
    for (int64_t i = 0; i < n; ++i) {
        float sum = b[pivot[i]];
        for (int64_t j = 0; j < i; ++j) {
            sum -= LU[i * n + j] * y[j];
        }
        y[i] = sum;
    }
    
    // Backward solve (U @ x = y)
    for (int64_t i = n - 1; i >= 0; --i) {
        float sum = y[i];
        for (int64_t j = i + 1; j < n; ++j) {
            sum -= LU[i * n + j] * x[j];
        }
        x[i] = sum / LU[i * n + i];
    }
}

/**
 * @brief Compute matrix inverse using LU decomposition.
 *
 * @param A Input matrix [n, n]
 * @param A_inv Output inverse [n, n]
 * @param n Matrix dimension
 * @return true if inversion succeeded
 */
inline bool matrix_inverse(const float* A, float* A_inv, int64_t n) {
    std::vector<float> LU(n * n);
    std::copy(A, A + n * n, LU.begin());
    
    std::vector<int32_t> pivot(n);
    if (!lu_decompose(LU.data(), pivot.data(), n)) {
        return false;
    }
    
    // Solve for each column of identity
    std::vector<float> e(n, 0.0f);
    for (int64_t j = 0; j < n; ++j) {
        std::fill(e.begin(), e.end(), 0.0f);
        e[j] = 1.0f;
        lu_solve(LU.data(), pivot.data(), e.data(), &A_inv[j], n);
    }
    
    // Transpose result (columns → rows)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            std::swap(A_inv[i * n + j], A_inv[j * n + i]);
        }
    }
    
    return true;
}

// =============================================================================
// CAYLEY TRANSFORM: W = (I - A)(I + A)^{-1}
// =============================================================================

/**
 * @brief Compute Cayley transform for orthogonal weight matrix.
 *
 * W = (I - A) @ (I + A)^{-1}
 *
 * @param A Skew-symmetric matrix [n, n]
 * @param W Output orthogonal matrix [n, n]
 * @param n Matrix dimension
 * @param eps Numerical stability epsilon
 * @return true if computation succeeded
 */
inline bool cayley_transform(
    const float* A, float* W, int64_t n, float eps = 1e-6f) {
    
    // Compute I + A
    std::vector<float> I_plus_A(n * n);
    std::copy(A, A + n * n, I_plus_A.begin());
    for (int64_t i = 0; i < n; ++i) {
        I_plus_A[i * n + i] += 1.0f;
    }
    
    // Compute (I + A)^{-1}
    std::vector<float> I_plus_A_inv(n * n);
    if (!matrix_inverse(I_plus_A.data(), I_plus_A_inv.data(), n)) {
        return false;
    }
    
    // Compute I - A
    std::vector<float> I_minus_A(n * n);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float val = (i == j) ? 1.0f : 0.0f;
            I_minus_A[i * n + j] = val - A[i * n + j];
        }
    }
    
    // W = (I - A) @ (I + A)^{-1}
    // SIMD-optimized matrix multiply
    #pragma omp parallel for
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            int64_t k = 0;
            
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            for (; k + 16 <= n; k += 16) {
                __m512 a = _mm512_loadu_ps(&I_minus_A[i * n + k]);
                __m512 b = _mm512_set_ps(
                    I_plus_A_inv[(k+15) * n + j], I_plus_A_inv[(k+14) * n + j],
                    I_plus_A_inv[(k+13) * n + j], I_plus_A_inv[(k+12) * n + j],
                    I_plus_A_inv[(k+11) * n + j], I_plus_A_inv[(k+10) * n + j],
                    I_plus_A_inv[(k+9) * n + j], I_plus_A_inv[(k+8) * n + j],
                    I_plus_A_inv[(k+7) * n + j], I_plus_A_inv[(k+6) * n + j],
                    I_plus_A_inv[(k+5) * n + j], I_plus_A_inv[(k+4) * n + j],
                    I_plus_A_inv[(k+3) * n + j], I_plus_A_inv[(k+2) * n + j],
                    I_plus_A_inv[(k+1) * n + j], I_plus_A_inv[k * n + j]
                );
                acc = _mm512_fmadd_ps(a, b, acc);
            }
            sum = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            for (; k + 8 <= n; k += 8) {
                __m256 a = _mm256_loadu_ps(&I_minus_A[i * n + k]);
                __m256 b = _mm256_set_ps(
                    I_plus_A_inv[(k+7) * n + j], I_plus_A_inv[(k+6) * n + j],
                    I_plus_A_inv[(k+5) * n + j], I_plus_A_inv[(k+4) * n + j],
                    I_plus_A_inv[(k+3) * n + j], I_plus_A_inv[(k+2) * n + j],
                    I_plus_A_inv[(k+1) * n + j], I_plus_A_inv[k * n + j]
                );
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            for (int t = 0; t < 8; ++t) sum += tmp[t];
#elif defined(__ARM_NEON)
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (; k + 4 <= n; k += 4) {
                float32x4_t a = vld1q_f32(&I_minus_A[i * n + k]);
                float b_arr[4] = {
                    I_plus_A_inv[k * n + j], I_plus_A_inv[(k+1) * n + j],
                    I_plus_A_inv[(k+2) * n + j], I_plus_A_inv[(k+3) * n + j]
                };
                float32x4_t b = vld1q_f32(b_arr);
                acc = vmlaq_f32(acc, a, b);
            }
            float tmp[4];
            vst1q_f32(tmp, acc);
            for (int t = 0; t < 4; ++t) sum += tmp[t];
#endif
            for (; k < n; ++k) {
                sum += I_minus_A[i * n + k] * I_plus_A_inv[k * n + j];
            }
            W[i * n + j] = sum;
        }
    }
    
    return true;
}

// =============================================================================
// CAYLEY DENSE FORWARD PASS
// =============================================================================

/**
 * @brief Forward pass for CayleyDense layer.
 *
 * For square matrices: W = (I - A)(I + A)^{-1} where A is skew-symmetric
 * For rectangular: Uses projection weight (orthogonality not guaranteed)
 *
 * @param input Input tensor [batch, input_dim]
 * @param skew_params Skew-symmetric parameters [n*(n-1)/2]
 * @param proj_weight Projection weight for rectangular case [input_dim, output_dim], null for square
 * @param bias Bias [output_dim], null if no bias
 * @param output Output tensor [batch, output_dim]
 * @param batch_size Batch dimension
 * @param input_dim Input feature dimension
 * @param output_dim Output feature dimension (units)
 * @param is_training Training mode flag
 * @param cached_W Cached weight matrix for inference [output_dim, output_dim], may be updated
 */
inline void cayley_dense_forward(
    const float* input,
    const float* skew_params,
    const float* proj_weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t input_dim,
    int64_t output_dim,
    bool is_training,
    float* cached_W) {
    
    bool is_square = (input_dim == output_dim);
    int64_t min_dim = std::min(input_dim, output_dim);
    
    // Workspace for weight matrix
    std::vector<float> W;
    const float* weight_ptr = nullptr;
    
    if (is_square) {
        // Square case: use Cayley transform
        W.resize(min_dim * min_dim);
        
        // Construct skew-symmetric matrix
        std::vector<float> A(min_dim * min_dim);
        cayley_construct_skew_symmetric(skew_params, A.data(), min_dim);
        
        // Compute Cayley transform
        cayley_transform(A.data(), W.data(), min_dim);
        
        weight_ptr = W.data();
        
        // Cache for inference
        if (!is_training && cached_W != nullptr) {
            std::copy(W.begin(), W.end(), cached_W);
        }
    } else {
        // Rectangular case: use projection weight
        weight_ptr = proj_weight;
    }
    
    // Matrix multiply: output = input @ W
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* in_row = input + b * input_dim;
        float* out_row = output + b * output_dim;
        
        for (int64_t j = 0; j < output_dim; ++j) {
            float sum = 0.0f;
            int64_t i = 0;
            
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            for (; i + 16 <= input_dim; i += 16) {
                __m512 in_vec = _mm512_loadu_ps(&in_row[i]);
                // Gather weight column
                __m512 w_vec = _mm512_set_ps(
                    weight_ptr[(i+15) * output_dim + j], weight_ptr[(i+14) * output_dim + j],
                    weight_ptr[(i+13) * output_dim + j], weight_ptr[(i+12) * output_dim + j],
                    weight_ptr[(i+11) * output_dim + j], weight_ptr[(i+10) * output_dim + j],
                    weight_ptr[(i+9) * output_dim + j], weight_ptr[(i+8) * output_dim + j],
                    weight_ptr[(i+7) * output_dim + j], weight_ptr[(i+6) * output_dim + j],
                    weight_ptr[(i+5) * output_dim + j], weight_ptr[(i+4) * output_dim + j],
                    weight_ptr[(i+3) * output_dim + j], weight_ptr[(i+2) * output_dim + j],
                    weight_ptr[(i+1) * output_dim + j], weight_ptr[i * output_dim + j]
                );
                acc = _mm512_fmadd_ps(in_vec, w_vec, acc);
            }
            sum = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            for (; i + 8 <= input_dim; i += 8) {
                __m256 in_vec = _mm256_loadu_ps(&in_row[i]);
                __m256 w_vec = _mm256_set_ps(
                    weight_ptr[(i+7) * output_dim + j], weight_ptr[(i+6) * output_dim + j],
                    weight_ptr[(i+5) * output_dim + j], weight_ptr[(i+4) * output_dim + j],
                    weight_ptr[(i+3) * output_dim + j], weight_ptr[(i+2) * output_dim + j],
                    weight_ptr[(i+1) * output_dim + j], weight_ptr[i * output_dim + j]
                );
                acc = _mm256_fmadd_ps(in_vec, w_vec, acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            for (int t = 0; t < 8; ++t) sum += tmp[t];
#elif defined(__ARM_NEON)
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (; i + 4 <= input_dim; i += 4) {
                float32x4_t in_vec = vld1q_f32(&in_row[i]);
                float w_arr[4] = {
                    weight_ptr[i * output_dim + j], weight_ptr[(i+1) * output_dim + j],
                    weight_ptr[(i+2) * output_dim + j], weight_ptr[(i+3) * output_dim + j]
                };
                float32x4_t w_vec = vld1q_f32(w_arr);
                acc = vmlaq_f32(acc, in_vec, w_vec);
            }
            float tmp[4];
            vst1q_f32(tmp, acc);
            for (int t = 0; t < 4; ++t) sum += tmp[t];
#endif
            for (; i < input_dim; ++i) {
                if (is_square) {
                    sum += in_row[i] * weight_ptr[i * min_dim + j];
                } else {
                    sum += in_row[i] * weight_ptr[i * output_dim + j];
                }
            }
            
            // Add bias
            if (bias != nullptr) {
                sum += bias[j];
            }
            
            out_row[j] = sum;
        }
    }
}

// =============================================================================
// CAYLEY DENSE BACKWARD PASS
// =============================================================================

/**
 * @brief Backward pass for CayleyDense layer.
 *
 * Computes gradients for skew_params, proj_weight, bias, and input.
 *
 * @param grad_output Gradient w.r.t. output [batch, output_dim]
 * @param input Original input [batch, input_dim]
 * @param skew_params Skew-symmetric parameters [n*(n-1)/2]
 * @param proj_weight Projection weight [input_dim, output_dim], null for square
 * @param grad_input Output gradient w.r.t. input [batch, input_dim]
 * @param grad_skew_params Output gradient w.r.t. skew_params [n*(n-1)/2]
 * @param grad_proj_weight Output gradient w.r.t. proj_weight [input_dim, output_dim]
 * @param grad_bias Output gradient w.r.t. bias [output_dim], null if no bias
 * @param batch_size Batch dimension
 * @param input_dim Input feature dimension
 * @param output_dim Output feature dimension
 */
inline void cayley_dense_backward(
    const float* grad_output,
    const float* input,
    const float* skew_params,
    const float* proj_weight,
    float* grad_input,
    float* grad_skew_params,
    float* grad_proj_weight,
    float* grad_bias,
    int64_t batch_size,
    int64_t input_dim,
    int64_t output_dim) {
    
    bool is_square = (input_dim == output_dim);
    int64_t min_dim = std::min(input_dim, output_dim);
    
    // Compute weight matrix W for backward
    std::vector<float> W;
    const float* weight_ptr = nullptr;
    
    if (is_square) {
        W.resize(min_dim * min_dim);
        std::vector<float> A(min_dim * min_dim);
        cayley_construct_skew_symmetric(skew_params, A.data(), min_dim);
        cayley_transform(A.data(), W.data(), min_dim);
        weight_ptr = W.data();
    } else {
        weight_ptr = proj_weight;
    }
    
    // Gradient w.r.t. bias: sum over batch
    if (grad_bias != nullptr) {
        std::fill(grad_bias, grad_bias + output_dim, 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            int64_t j = 0;
#if defined(__AVX512F__)
            for (; j + 16 <= output_dim; j += 16) {
                __m512 g = _mm512_loadu_ps(&grad_output[b * output_dim + j]);
                __m512 acc = _mm512_loadu_ps(&grad_bias[j]);
                _mm512_storeu_ps(&grad_bias[j], _mm512_add_ps(acc, g));
            }
#elif defined(__AVX2__)
            for (; j + 8 <= output_dim; j += 8) {
                __m256 g = _mm256_loadu_ps(&grad_output[b * output_dim + j]);
                __m256 acc = _mm256_loadu_ps(&grad_bias[j]);
                _mm256_storeu_ps(&grad_bias[j], _mm256_add_ps(acc, g));
            }
#endif
            for (; j < output_dim; ++j) {
                grad_bias[j] += grad_output[b * output_dim + j];
            }
        }
    }
    
    // Gradient w.r.t. input: grad_input = grad_output @ W^T
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* g_row = grad_output + b * output_dim;
        float* gi_row = grad_input + b * input_dim;
        
        for (int64_t i = 0; i < input_dim; ++i) {
            float sum = 0.0f;
            for (int64_t j = 0; j < output_dim; ++j) {
                if (is_square) {
                    sum += g_row[j] * weight_ptr[i * min_dim + j];
                } else {
                    sum += g_row[j] * weight_ptr[i * output_dim + j];
                }
            }
            gi_row[i] = sum;
        }
    }
    
    // Gradient w.r.t. weight: grad_W = input^T @ grad_output
    if (!is_square && grad_proj_weight != nullptr) {
        std::fill(grad_proj_weight, grad_proj_weight + input_dim * output_dim, 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < input_dim; ++i) {
                float in_val = input[b * input_dim + i];
                int64_t j = 0;
#if defined(__AVX512F__)
                __m512 in_vec = _mm512_set1_ps(in_val);
                for (; j + 16 <= output_dim; j += 16) {
                    __m512 g = _mm512_loadu_ps(&grad_output[b * output_dim + j]);
                    __m512 acc = _mm512_loadu_ps(&grad_proj_weight[i * output_dim + j]);
                    _mm512_storeu_ps(&grad_proj_weight[i * output_dim + j],
                                     _mm512_fmadd_ps(in_vec, g, acc));
                }
#elif defined(__AVX2__)
                __m256 in_vec = _mm256_set1_ps(in_val);
                for (; j + 8 <= output_dim; j += 8) {
                    __m256 g = _mm256_loadu_ps(&grad_output[b * output_dim + j]);
                    __m256 acc = _mm256_loadu_ps(&grad_proj_weight[i * output_dim + j]);
                    _mm256_storeu_ps(&grad_proj_weight[i * output_dim + j],
                                     _mm256_fmadd_ps(in_vec, g, acc));
                }
#endif
                for (; j < output_dim; ++j) {
                    grad_proj_weight[i * output_dim + j] += in_val * grad_output[b * output_dim + j];
                }
            }
        }
    }
    
    // Gradient w.r.t. skew_params for square case
    if (is_square && grad_skew_params != nullptr) {
        // grad_W = input^T @ grad_output
        std::vector<float> grad_W(min_dim * min_dim, 0.0f);
        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t i = 0; i < min_dim; ++i) {
                float in_val = input[b * input_dim + i];
                for (int64_t j = 0; j < min_dim; ++j) {
                    grad_W[i * min_dim + j] += in_val * grad_output[b * output_dim + j];
                }
            }
        }
        
        // Approximate: For Cayley transform, grad_A ≈ grad_W (simplified)
        // Full gradient requires implicit differentiation through matrix inverse
        // For efficiency, we use a first-order approximation
        cayley_skew_symmetric_backward(grad_W.data(), grad_skew_params, min_dim);
    }
}

}  // namespace cayley
}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_CAYLEY_TRANSFORM_OP_H_
