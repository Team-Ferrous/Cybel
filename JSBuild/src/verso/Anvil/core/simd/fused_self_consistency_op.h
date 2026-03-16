// highnoon/_native/ops/fused_self_consistency_op.h
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
 * @file fused_self_consistency_op.h
 * @brief Self-Consistency Verification SIMD helpers.
 *
 * Implements core operations for the SelfConsistencyVerifier:
 *   - Pairwise cosine similarity computation
 *   - Consistency score aggregation
 *   - Multi-head verification projections
 *   - Threshold-based gating
 *
 * SIMD optimizations:
 * - AVX512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - NEON: 4-wide vectorization (ARM)
 * - Scalar fallback for all architectures
 *
 * Functions use the self_consistency_ prefix to avoid ODR violations.
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_SELF_CONSISTENCY_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_SELF_CONSISTENCY_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

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
// SELF-CONSISTENCY SIMD HELPERS
// All functions have self_consistency_ prefix to avoid ODR violations
// =============================================================================

/**
 * @brief L2 normalize vectors in-place for cosine similarity.
 *
 * Computes: v[i] = v[i] / ||v||_2
 *
 * @param data Input/output tensor [batch_seq, dim]
 * @param batch_seq Number of vectors
 * @param dim Dimension of each vector
 */
inline void self_consistency_l2_normalize(
    float* data, int64_t batch_seq, int64_t dim) {
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_seq; ++i) {
        float* row = data + i * dim;
        
        // Compute L2 norm
        float norm_sq = 0.0f;
        int64_t d = 0;
#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        for (; d + 16 <= dim; d += 16) {
            __m512 v = _mm512_loadu_ps(&row[d]);
            acc = _mm512_fmadd_ps(v, v, acc);
        }
        float tmp[16];
        _mm512_storeu_ps(tmp, acc);
        for (int j = 0; j < 16; ++j) norm_sq += tmp[j];
#elif defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&row[d]);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int j = 0; j < 8; ++j) norm_sq += tmp[j];
#elif defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t v = vld1q_f32(&row[d]);
            acc = vmlaq_f32(acc, v, v);
        }
        float tmp[4];
        vst1q_f32(tmp, acc);
        for (int j = 0; j < 4; ++j) norm_sq += tmp[j];
#endif
        for (; d < dim; ++d) {
            norm_sq += row[d] * row[d];
        }
        
        // Normalize
        float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-8f);
        d = 0;
#if defined(__AVX512F__)
        __m512 inv_v = _mm512_set1_ps(inv_norm);
        for (; d + 16 <= dim; d += 16) {
            __m512 v = _mm512_loadu_ps(&row[d]);
            _mm512_storeu_ps(&row[d], _mm512_mul_ps(v, inv_v));
        }
#elif defined(__AVX2__)
        __m256 inv_v = _mm256_set1_ps(inv_norm);
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&row[d]);
            _mm256_storeu_ps(&row[d], _mm256_mul_ps(v, inv_v));
        }
#elif defined(__ARM_NEON)
        float32x4_t inv_v = vdupq_n_f32(inv_norm);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t v = vld1q_f32(&row[d]);
            vst1q_f32(&row[d], vmulq_f32(v, inv_v));
        }
#endif
        for (; d < dim; ++d) {
            row[d] *= inv_norm;
        }
    }
}

/**
 * @brief Compute dot product between two vectors.
 *
 * @param a First vector [dim]
 * @param b Second vector [dim]
 * @param dim Vector dimension
 * @return Dot product value
 */
inline float self_consistency_dot_product(
    const float* a, const float* b, int64_t dim) {
    
    float result = 0.0f;
    int64_t i = 0;
    
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m512 av = _mm512_loadu_ps(&a[i]);
        __m512 bv = _mm512_loadu_ps(&b[i]);
        acc = _mm512_fmadd_ps(av, bv, acc);
    }
    float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    for (int j = 0; j < 16; ++j) result += tmp[j];
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    for (int j = 0; j < 8; ++j) result += tmp[j];
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= dim; i += 4) {
        float32x4_t av = vld1q_f32(&a[i]);
        float32x4_t bv = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, av, bv);
    }
    float tmp[4];
    vst1q_f32(tmp, acc);
    for (int j = 0; j < 4; ++j) result += tmp[j];
#endif
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

/**
 * @brief Compute pairwise agreement (cosine similarity) between paths.
 *
 * Assumes paths are already L2 normalized.
 *
 * @param paths Normalized paths [batch, seq_len, num_paths, dim]
 * @param agreement Output agreement matrix [batch, seq_len, num_paths, num_paths]
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param num_paths Number of paths
 * @param dim Embedding dimension
 */
inline void self_consistency_pairwise_agreement(
    const float* paths, float* agreement,
    int64_t batch_size, int64_t seq_len, int64_t num_paths, int64_t dim) {
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            // Base offset for this (batch, seq) position
            const int64_t path_base = ((b * seq_len + s) * num_paths) * dim;
            const int64_t agree_base = (b * seq_len + s) * num_paths * num_paths;
            
            for (int64_t i = 0; i < num_paths; ++i) {
                const float* path_i = paths + path_base + i * dim;
                for (int64_t j = 0; j < num_paths; ++j) {
                    const float* path_j = paths + path_base + j * dim;
                    agreement[agree_base + i * num_paths + j] = 
                        self_consistency_dot_product(path_i, path_j, dim);
                }
            }
        }
    }
}

/**
 * @brief Compute consistency score from agreement matrix.
 *
 * Computes mean of off-diagonal elements.
 *
 * @param agreement Agreement matrix [batch, seq_len, num_paths, num_paths]
 * @param consistency Output consistency scores [batch, seq_len]
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param num_paths Number of paths
 */
inline void self_consistency_compute_score(
    const float* agreement, float* consistency,
    int64_t batch_size, int64_t seq_len, int64_t num_paths) {
    
    const float count = static_cast<float>(num_paths * (num_paths - 1));
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t agree_base = (b * seq_len + s) * num_paths * num_paths;
            
            float sum = 0.0f;
            for (int64_t i = 0; i < num_paths; ++i) {
                for (int64_t j = 0; j < num_paths; ++j) {
                    if (i != j) {
                        sum += agreement[agree_base + i * num_paths + j];
                    }
                }
            }
            
            consistency[b * seq_len + s] = sum / (count + 1e-8f);
        }
    }
}

/**
 * @brief Softmax for path weights.
 *
 * @param input Input scores [batch_seq, num_paths]
 * @param output Output probabilities [batch_seq, num_paths]
 * @param batch_seq Number of rows
 * @param num_paths Number of paths
 */
inline void self_consistency_softmax(
    const float* input, float* output,
    int64_t batch_seq, int64_t num_paths) {
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_seq; ++i) {
        const float* in_row = input + i * num_paths;
        float* out_row = output + i * num_paths;
        
        // Find max for numerical stability
        float max_val = in_row[0];
        for (int64_t j = 1; j < num_paths; ++j) {
            max_val = std::max(max_val, in_row[j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int64_t j = 0; j < num_paths; ++j) {
            out_row[j] = std::exp(in_row[j] - max_val);
            sum += out_row[j];
        }
        
        // Normalize
        float inv_sum = 1.0f / (sum + 1e-8f);
        for (int64_t j = 0; j < num_paths; ++j) {
            out_row[j] *= inv_sum;
        }
    }
}

/**
 * @brief Weighted combination of paths.
 *
 * output[i] = sum_j(weights[j] * paths[j][i])
 *
 * @param paths Input paths [batch, seq_len, num_paths, dim]
 * @param weights Path weights [batch, seq_len, num_paths]
 * @param output Output tensor [batch, seq_len, dim]
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param num_paths Number of paths
 * @param dim Embedding dimension
 */
inline void self_consistency_weighted_combine(
    const float* paths, const float* weights, float* output,
    int64_t batch_size, int64_t seq_len, int64_t num_paths, int64_t dim) {
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t path_base = ((b * seq_len + s) * num_paths) * dim;
            const int64_t weight_base = (b * seq_len + s) * num_paths;
            float* out_row = output + (b * seq_len + s) * dim;
            
            // Initialize output to zero
            for (int64_t d = 0; d < dim; ++d) {
                out_row[d] = 0.0f;
            }
            
            // Weighted sum over paths
            for (int64_t p = 0; p < num_paths; ++p) {
                const float w = weights[weight_base + p];
                const float* path = paths + path_base + p * dim;
                
                int64_t d = 0;
#if defined(__AVX512F__)
                __m512 wv = _mm512_set1_ps(w);
                for (; d + 16 <= dim; d += 16) {
                    __m512 out_v = _mm512_loadu_ps(&out_row[d]);
                    __m512 path_v = _mm512_loadu_ps(&path[d]);
                    _mm512_storeu_ps(&out_row[d], _mm512_fmadd_ps(wv, path_v, out_v));
                }
#elif defined(__AVX2__)
                __m256 wv = _mm256_set1_ps(w);
                for (; d + 8 <= dim; d += 8) {
                    __m256 out_v = _mm256_loadu_ps(&out_row[d]);
                    __m256 path_v = _mm256_loadu_ps(&path[d]);
                    _mm256_storeu_ps(&out_row[d], _mm256_fmadd_ps(wv, path_v, out_v));
                }
#elif defined(__ARM_NEON)
                float32x4_t wv = vdupq_n_f32(w);
                for (; d + 4 <= dim; d += 4) {
                    float32x4_t out_v = vld1q_f32(&out_row[d]);
                    float32x4_t path_v = vld1q_f32(&path[d]);
                    vst1q_f32(&out_row[d], vmlaq_f32(out_v, wv, path_v));
                }
#endif
                for (; d < dim; ++d) {
                    out_row[d] += w * path[d];
                }
            }
        }
    }
}

/**
 * @brief Layer normalization.
 */
inline void self_consistency_layer_norm(
    const float* input, const float* gamma, const float* beta,
    float* output, int64_t batch_seq, int64_t dim, float eps = 1e-6f) {
    
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_seq; ++i) {
        const float* x_row = input + i * dim;
        float* out_row = output + i * dim;
        
        // Compute mean
        float mean = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            mean += x_row[d];
        }
        mean /= static_cast<float>(dim);
        
        // Compute variance
        float var = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            float diff = x_row[d] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(dim);
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int64_t d = 0; d < dim; ++d) {
            out_row[d] = gamma[d] * (x_row[d] - mean) * inv_std + beta[d];
        }
    }
}

/**
 * @brief Layer normalization backward pass.
 *
 * Computes gradients through layer norm for proper backpropagation.
 *
 * @param grad_output Upstream gradient [batch_seq, dim]
 * @param input Original input before layer norm [batch_seq, dim]
 * @param gamma Scale parameters [dim]
 * @param grad_input Output gradient w.r.t. input [batch_seq, dim]
 * @param grad_gamma Output gradient w.r.t. gamma [dim] (accumulated)
 * @param grad_beta Output gradient w.r.t. beta [dim] (accumulated)
 * @param batch_seq Number of vectors
 * @param dim Dimension of each vector
 * @param eps Layer norm epsilon
 */
inline void self_consistency_layer_norm_backward(
    const float* grad_output, const float* input, const float* gamma,
    float* grad_input, float* grad_gamma, float* grad_beta,
    int64_t batch_seq, int64_t dim, float eps = 1e-6f) {
    
    // First pass: compute grad_gamma and grad_beta
    #pragma omp parallel for
    for (int64_t d = 0; d < dim; ++d) {
        float sum_grad_gamma = 0.0f;
        float sum_grad_beta = 0.0f;
        
        for (int64_t i = 0; i < batch_seq; ++i) {
            const float* x_row = input + i * dim;
            
            // Recompute mean and variance for this row
            float mean = 0.0f;
            for (int64_t dd = 0; dd < dim; ++dd) {
                mean += x_row[dd];
            }
            mean /= static_cast<float>(dim);
            
            float var = 0.0f;
            for (int64_t dd = 0; dd < dim; ++dd) {
                float diff = x_row[dd] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(dim);
            
            float inv_std = 1.0f / std::sqrt(var + eps);
            float x_hat = (x_row[d] - mean) * inv_std;
            
            sum_grad_gamma += grad_output[i * dim + d] * x_hat;
            sum_grad_beta += grad_output[i * dim + d];
        }
        
        grad_gamma[d] += sum_grad_gamma;
        grad_beta[d] += sum_grad_beta;
    }
    
    // Second pass: compute grad_input
    #pragma omp parallel for
    for (int64_t i = 0; i < batch_seq; ++i) {
        const float* x_row = input + i * dim;
        const float* g_row = grad_output + i * dim;
        float* gi_row = grad_input + i * dim;
        
        // Recompute mean and variance
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
        
        float inv_std = 1.0f / std::sqrt(var + eps);
        float dim_f = static_cast<float>(dim);
        
        // Compute intermediate terms
        float sum_g_gamma = 0.0f;
        float sum_g_gamma_xhat = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            float x_hat = (x_row[d] - mean) * inv_std;
            sum_g_gamma += g_row[d] * gamma[d];
            sum_g_gamma_xhat += g_row[d] * gamma[d] * x_hat;
        }
        
        // Compute gradient w.r.t. input
        for (int64_t d = 0; d < dim; ++d) {
            float x_hat = (x_row[d] - mean) * inv_std;
            gi_row[d] = inv_std / dim_f * (
                dim_f * g_row[d] * gamma[d] -
                sum_g_gamma -
                x_hat * sum_g_gamma_xhat
            );
        }
    }
}

/**
 * @brief Threshold-based confidence gating.
 *
 * gated[i] = consistency[i] >= threshold ? consistency[i] : consistency[i] * 0.5
 */
inline void self_consistency_threshold_gate(
    const float* consistency, float* gated,
    int64_t size, float threshold) {
    
    for (int64_t i = 0; i < size; ++i) {
        float c = consistency[i];
        // Clamp to [0, 1] range first
        c = std::max(0.0f, std::min(1.0f, c));
        gated[i] = c >= threshold ? c : c * 0.5f;
    }
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_SELF_CONSISTENCY_OP_H_
