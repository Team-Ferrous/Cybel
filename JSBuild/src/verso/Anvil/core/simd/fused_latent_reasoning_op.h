// highnoon/_native/ops/fused_latent_reasoning_op.h
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
 * @file fused_latent_reasoning_op.h
 * @brief Latent Reasoning Block SIMD helpers.
 *
 * Implements core operations for the LatentReasoningBlock:
 *   - Multi-step thought refinement (LayerNorm + GELU + projection)
 *   - Uncertainty computation (reduce_std)
 *   - Thought memory ring buffer
 *   - ACT-Lite halting probability
 *
 * SIMD optimizations:
 * - AVX512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - NEON: 4-wide vectorization (ARM)
 * - Scalar fallback for all architectures
 *
 * Functions use the latent_ prefix to avoid ODR violations.
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_LATENT_REASONING_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_LATENT_REASONING_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

// Include shared SIMD library for common operations
#include "hnn_simd_common.h"

// SIMD intrinsics for cross-architecture vectorization
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace highnoon {
namespace ops {

// =============================================================================
// LATENT REASONING SIMD HELPERS
// All functions have latent_ prefix to avoid ODR violations
// =============================================================================

/**
 * @brief Layer normalization with SIMD optimization.
 * Delegates to shared SIMD library.
 */
inline void latent_layer_norm(
    const float* input, const float* gamma, const float* beta,
    float* output, int64_t batch_seq, int64_t dim, float eps = 1e-6f) {
    simd_layernorm(input, gamma, beta, output, batch_seq, dim, eps);
}

/**
 * @brief GELU activation function with SIMD optimization.
 * Delegates to shared SIMD library.
 */
inline void latent_gelu(const float* input, float* output, int64_t size) {
    std::copy(input, input + size, output);
    simd_gelu_inplace(output, size);
}

/**
 * @brief Compute per-row standard deviation (uncertainty).
 *
 * std = sqrt(mean((x - mean(x))^2))
 *
 * @param input Input tensor [batch_seq, dim]
 * @param output Output std values [batch_seq]
 * @param batch_seq Number of rows
 * @param dim Dimension per row
 */
inline void latent_reduce_std(
    const float* input, float* output,
    int64_t batch_seq, int64_t dim) {
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_seq; ++i) {
        const float* row = input + i * dim;
        
        // Compute mean
        float mean = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            mean += row[d];
        }
        mean /= static_cast<float>(dim);
        
        // Compute variance
        float var = 0.0f;
        int64_t d = 0;
#if defined(__AVX512F__)
        __m512 mean_v = _mm512_set1_ps(mean);
        __m512 var_acc = _mm512_setzero_ps();
        for (; d + 16 <= dim; d += 16) {
            __m512 x = _mm512_loadu_ps(&row[d]);
            __m512 diff = _mm512_sub_ps(x, mean_v);
            var_acc = _mm512_fmadd_ps(diff, diff, var_acc);
        }
        float tmp[16];
        _mm512_storeu_ps(tmp, var_acc);
        for (int j = 0; j < 16; ++j) var += tmp[j];
#elif defined(__AVX2__)
        __m256 mean_v = _mm256_set1_ps(mean);
        __m256 var_acc = _mm256_setzero_ps();
        for (; d + 8 <= dim; d += 8) {
            __m256 x = _mm256_loadu_ps(&row[d]);
            __m256 diff = _mm256_sub_ps(x, mean_v);
            var_acc = _mm256_fmadd_ps(diff, diff, var_acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, var_acc);
        for (int j = 0; j < 8; ++j) var += tmp[j];
#elif defined(__ARM_NEON)
        float32x4_t mean_v = vdupq_n_f32(mean);
        float32x4_t var_acc = vdupq_n_f32(0.0f);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t x = vld1q_f32(&row[d]);
            float32x4_t diff = vsubq_f32(x, mean_v);
            var_acc = vmlaq_f32(var_acc, diff, diff);
        }
        float tmp[4];
        vst1q_f32(tmp, var_acc);
        for (int j = 0; j < 4; ++j) var += tmp[j];
#endif
        for (; d < dim; ++d) {
            float diff = row[d] - mean;
            var += diff * diff;
        }
        
        output[i] = std::sqrt(var / static_cast<float>(dim));
    }
}

/**
 * @brief In-place sigmoid for halting probability.
 * Delegates to shared SIMD library.
 */
inline void latent_sigmoid_inplace(float* data, int64_t size) {
    simd_sigmoid_inplace(data, size);
}

/**
 * @brief Element-wise residual add.
 * Delegates to shared SIMD library.
 */
inline void latent_add(
    const float* a, const float* b, float* out, int64_t size) {
    simd_add(a, b, out, size);
}

/**
 * @brief Masked select/update for entropy-guided refinement.
 *
 * For each row: out[i] = mask[i] ? a[i] : b[i]
 *
 * @param mask Boolean mask [batch_seq]
 * @param a True branch values [batch_seq, dim]
 * @param b False branch values [batch_seq, dim]
 * @param out Output [batch_seq, dim]
 * @param batch_seq Number of rows
 * @param dim Dimension per row
 */
inline void latent_masked_select(
    const float* mask, const float* a, const float* b,
    float* out, int64_t batch_seq, int64_t dim) {
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_seq; ++i) {
        const float* src = mask[i] > 0.5f ? a : b;
        const float* row_src = src + i * dim;
        float* row_out = out + i * dim;
        
        int64_t d = 0;
#if defined(__AVX512F__)
        for (; d + 16 <= dim; d += 16) {
            __m512 v = _mm512_loadu_ps(&row_src[d]);
            _mm512_storeu_ps(&row_out[d], v);
        }
#elif defined(__AVX2__)
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&row_src[d]);
            _mm256_storeu_ps(&row_out[d], v);
        }
#elif defined(__ARM_NEON)
        for (; d + 4 <= dim; d += 4) {
            float32x4_t v = vld1q_f32(&row_src[d]);
            vst1q_f32(&row_out[d], v);
        }
#endif
        for (; d < dim; ++d) {
            row_out[d] = row_src[d];
        }
    }
}

// =============================================================================
// BACKWARD OPERATIONS
// =============================================================================

/**
 * @brief GELU backward pass.
 * d/dx GELU(x) = 0.5 * (1 + tanh(c)) + 0.5 * x * sech^2(c) * c'
 * where c = sqrt(2/pi) * (x + 0.044715 * x^3)
 * Simplified approximation for efficiency.
 */
inline void latent_gelu_backward(
    const float* grad_out, const float* input, float* grad_in, int64_t size) {
    
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        float x = input[i];
        float x_cubed = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_inner = std::tanh(inner);
        float sech_sq = 1.0f - tanh_inner * tanh_inner;
        
        // d_inner/dx = sqrt_2_over_pi * (1 + 3 * coeff * x^2)
        float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
        
        // GELU gradient
        float gelu_grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech_sq * d_inner;
        grad_in[i] = grad_out[i] * gelu_grad;
    }
}

/**
 * @brief LayerNorm backward pass.
 * 
 * Forward: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *        = gamma * x_hat + beta
 * 
 * Backward:
 *   grad_gamma = sum(grad_out * x_hat)
 *   grad_beta = sum(grad_out)
 *   grad_x = (gamma / std) * (grad_out - mean(grad_out) - x_hat * mean(grad_out * x_hat))
 */
inline void latent_layer_norm_backward(
    const float* grad_out, const float* x, const float* gamma,
    float* grad_x, float* grad_gamma, float* grad_beta,
    int64_t batch_seq, int64_t dim, float eps = 1e-6f) {
    
    // Initialize gradient accumulators
    std::vector<float> gamma_grad_acc(dim, 0.0f);
    std::vector<float> beta_grad_acc(dim, 0.0f);
    
    for (int64_t i = 0; i < batch_seq; ++i) {
        const float* x_row = x + i * dim;
        const float* grad_out_row = grad_out + i * dim;
        float* grad_x_row = grad_x + i * dim;
        
        // Compute mean and variance for this row
        float mean = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            mean += x_row[d];
        }
        mean /= static_cast<float>(dim);
        
        float var = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            float diff = x_row[d] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(dim);
        float std_inv = 1.0f / std::sqrt(var + eps);
        
        // Compute x_hat
        std::vector<float> x_hat(dim);
        for (int64_t d = 0; d < dim; ++d) {
            x_hat[d] = (x_row[d] - mean) * std_inv;
        }
        
        // Accumulate gamma and beta gradients
        for (int64_t d = 0; d < dim; ++d) {
            gamma_grad_acc[d] += grad_out_row[d] * x_hat[d];
            beta_grad_acc[d] += grad_out_row[d];
        }
        
        // Compute grad_x for this row
        // grad_x = (gamma / std) * (grad_out - mean(grad_out) - x_hat * mean(grad_out * x_hat))
        float mean_grad_out = 0.0f;
        float mean_grad_out_xhat = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            mean_grad_out += grad_out_row[d] * gamma[d];
            mean_grad_out_xhat += grad_out_row[d] * gamma[d] * x_hat[d];
        }
        mean_grad_out /= static_cast<float>(dim);
        mean_grad_out_xhat /= static_cast<float>(dim);
        
        for (int64_t d = 0; d < dim; ++d) {
            grad_x_row[d] = std_inv * (
                grad_out_row[d] * gamma[d] 
                - mean_grad_out 
                - x_hat[d] * mean_grad_out_xhat
            );
        }
    }
    
    // Copy accumulated gradients
    std::copy(gamma_grad_acc.begin(), gamma_grad_acc.end(), grad_gamma);
    std::copy(beta_grad_acc.begin(), beta_grad_acc.end(), grad_beta);
}

/**
 * @brief Matrix multiply backward.
 * Forward: C = A @ B  where A is [M,K], B is [K,N], C is [M,N]
 * Backward:
 *   grad_A = grad_C @ B^T  [M,N] @ [N,K] -> [M,K]
 *   grad_B = A^T @ grad_C  [K,M] @ [M,N] -> [K,N]
 */
inline void matmul_backward(
    const float* grad_C, const float* A, const float* B,
    float* grad_A, float* grad_B,
    int64_t M, int64_t K, int64_t N) {
    
    // grad_A = grad_C @ B^T
    #pragma omp parallel for
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float sum = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                sum += grad_C[i * N + j] * B[k * N + j];
            }
            grad_A[i * K + k] = sum;
        }
    }
    
    // grad_B = A^T @ grad_C
    // Initialize to zero
    std::fill(grad_B, grad_B + K * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            for (int64_t j = 0; j < N; ++j) {
                grad_B[k * N + j] += A[i * K + k] * grad_C[i * N + j];
            }
        }
    }
}

/**
 * @brief Bias gradient accumulation.
 * Forward: C[i,j] += bias[j]
 * Backward: grad_bias[j] = sum_i(grad_C[i,j])
 */
inline void bias_backward(
    const float* grad_C, float* grad_bias, int64_t rows, int64_t cols) {
    
    std::fill(grad_bias, grad_bias + cols, 0.0f);
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            grad_bias[j] += grad_C[i * cols + j];
        }
    }
}

/**
 * @brief Full thought step backward.
 * Backward through: LayerNorm -> Up proj + GELU -> Down proj -> Residual
 */
inline void latent_thought_step_backward(
    const float* grad_hidden,        // [batch_seq, embed_dim]
    const float* prev_hidden,        // [batch_seq, embed_dim] saved from forward
    const float* normalized,         // [batch_seq, embed_dim] saved from forward
    const float* up_projected,       // [batch_seq, d_inner] saved (pre-GELU)
    const float* gelu_activated,     // [batch_seq, d_inner] saved (post-GELU)
    const float* gamma,              // [embed_dim]
    const float* up_weight,          // [embed_dim, d_inner]
    const float* down_weight,        // [d_inner, embed_dim]
    float* grad_prev_hidden,         // [batch_seq, embed_dim] output
    float* grad_gamma,               // [embed_dim] output (accumulated)
    float* grad_beta,                // [embed_dim] output (accumulated)
    float* grad_up_weight,           // [embed_dim, d_inner] output (accumulated)
    float* grad_up_bias,             // [d_inner] output (accumulated)
    float* grad_down_weight,         // [d_inner, embed_dim] output (accumulated)
    float* grad_down_bias,           // [embed_dim] output (accumulated)
    int64_t batch_seq, int64_t embed_dim, int64_t d_inner) {
    
    // Allocate temporary buffers
    std::vector<float> grad_down_projected(batch_seq * embed_dim);
    std::vector<float> grad_gelu_out(batch_seq * d_inner);
    std::vector<float> grad_up_out(batch_seq * d_inner);
    std::vector<float> grad_normalized(batch_seq * embed_dim);
    std::vector<float> grad_ln_x(batch_seq * embed_dim);
    std::vector<float> temp_grad_gamma(embed_dim, 0.0f);
    std::vector<float> temp_grad_beta(embed_dim, 0.0f);
    std::vector<float> temp_grad_up_weight(embed_dim * d_inner, 0.0f);
    std::vector<float> temp_grad_up_bias(d_inner, 0.0f);
    std::vector<float> temp_grad_down_weight(d_inner * embed_dim, 0.0f);
    std::vector<float> temp_grad_down_bias(embed_dim, 0.0f);
    
    // Backward through residual: hidden = prev_hidden + down_projected
    // grad_prev_hidden += grad_hidden, grad_down_projected = grad_hidden
    std::copy(grad_hidden, grad_hidden + batch_seq * embed_dim, grad_down_projected.data());
    std::copy(grad_hidden, grad_hidden + batch_seq * embed_dim, grad_prev_hidden);
    
    // Backward through down bias
    bias_backward(grad_down_projected.data(), temp_grad_down_bias.data(), batch_seq, embed_dim);
    
    // Backward through down projection: down_projected = gelu_out @ down_weight
    matmul_backward(
        grad_down_projected.data(), gelu_activated, down_weight,
        grad_gelu_out.data(), temp_grad_down_weight.data(),
        batch_seq, d_inner, embed_dim
    );
    
    // Backward through GELU
    latent_gelu_backward(grad_gelu_out.data(), up_projected, grad_up_out.data(), batch_seq * d_inner);
    
    // Backward through up bias
    bias_backward(grad_up_out.data(), temp_grad_up_bias.data(), batch_seq, d_inner);
    
    // Backward through up projection: up_projected = normalized @ up_weight
    matmul_backward(
        grad_up_out.data(), normalized, up_weight,
        grad_normalized.data(), temp_grad_up_weight.data(),
        batch_seq, embed_dim, d_inner
    );
    
    // Backward through LayerNorm
    latent_layer_norm_backward(
        grad_normalized.data(), prev_hidden, gamma,
        grad_ln_x.data(), temp_grad_gamma.data(), temp_grad_beta.data(),
        batch_seq, embed_dim
    );
    
    // Accumulate gradients from LayerNorm into prev_hidden gradient
    for (int64_t i = 0; i < batch_seq * embed_dim; ++i) {
        grad_prev_hidden[i] += grad_ln_x[i];
    }
    
    // Accumulate into output gradients
    for (int64_t i = 0; i < embed_dim; ++i) {
        grad_gamma[i] += temp_grad_gamma[i];
        grad_beta[i] += temp_grad_beta[i];
    }
    for (int64_t i = 0; i < embed_dim * d_inner; ++i) {
        grad_up_weight[i] += temp_grad_up_weight[i];
    }
    for (int64_t i = 0; i < d_inner; ++i) {
        grad_up_bias[i] += temp_grad_up_bias[i];
    }
    for (int64_t i = 0; i < d_inner * embed_dim; ++i) {
        grad_down_weight[i] += temp_grad_down_weight[i];
    }
    for (int64_t i = 0; i < embed_dim; ++i) {
        grad_down_bias[i] += temp_grad_down_bias[i];
    }
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_LATENT_REASONING_OP_H_

