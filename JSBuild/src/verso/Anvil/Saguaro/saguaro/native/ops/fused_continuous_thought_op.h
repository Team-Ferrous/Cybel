// saguaro.native/ops/fused_continuous_thought_op.h
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
 * @file fused_continuous_thought_op.h
 * @brief COCONUT Continuous Thought Block SIMD helpers.
 *
 * Implements core operations for the ContinuousThoughtBlock:
 *   - Mean pooling over sequence (thought extraction)
 *   - Iterative thought refinement (LayerNorm + MLP + residual)
 *   - Broadcast thought back to sequence
 *   - Gated residual connection
 *
 * SIMD optimizations:
 * - AVX512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - NEON: 4-wide vectorization (ARM)
 * - Scalar fallback for all architectures
 *
 * Functions use the continuous_thought_ prefix to avoid ODR violations.
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef SAGUARO_NATIVE_OPS_FUSED_CONTINUOUS_THOUGHT_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_CONTINUOUS_THOUGHT_OP_H_

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

namespace saguaro {
namespace ops {

// =============================================================================
// CONTINUOUS THOUGHT SIMD HELPERS
// All functions have continuous_thought_ prefix to avoid ODR violations
// =============================================================================

/**
 * @brief Layer normalization with SIMD optimization.
 * Delegates to shared SIMD library.
 */
inline void continuous_thought_layer_norm(
    const float* input, const float* gamma, const float* beta,
    float* output, int64_t batch_seq, int64_t dim, float eps = 1e-6f) {
    simd_layernorm(input, gamma, beta, output, batch_seq, dim, eps);
}

/**
 * @brief Mean pooling over sequence dimension.
 *
 * For each batch element, computes mean over seq_len positions.
 * output[b, d] = mean(input[b, :, d])
 *
 * @param input Input tensor [batch, seq_len, dim]
 * @param output Output tensor [batch, dim]
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param dim Hidden dimension
 */
inline void continuous_thought_mean_pool(
    const float* input, float* output,
    int64_t batch_size, int64_t seq_len, int64_t dim) {
    
    const float inv_seq = 1.0f / static_cast<float>(seq_len);
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out_row = output + b * dim;
        
        // Initialize to zero
        for (int64_t d = 0; d < dim; ++d) {
            out_row[d] = 0.0f;
        }
        
        // Accumulate over sequence
        for (int64_t s = 0; s < seq_len; ++s) {
            const float* in_row = input + (b * seq_len + s) * dim;
            
            int64_t d = 0;
#if defined(__AVX512F__)
            for (; d + 16 <= dim; d += 16) {
                __m512 acc = _mm512_loadu_ps(&out_row[d]);
                __m512 x = _mm512_loadu_ps(&in_row[d]);
                _mm512_storeu_ps(&out_row[d], _mm512_add_ps(acc, x));
            }
#elif defined(__AVX2__)
            for (; d + 8 <= dim; d += 8) {
                __m256 acc = _mm256_loadu_ps(&out_row[d]);
                __m256 x = _mm256_loadu_ps(&in_row[d]);
                _mm256_storeu_ps(&out_row[d], _mm256_add_ps(acc, x));
            }
#elif defined(__ARM_NEON)
            for (; d + 4 <= dim; d += 4) {
                float32x4_t acc = vld1q_f32(&out_row[d]);
                float32x4_t x = vld1q_f32(&in_row[d]);
                vst1q_f32(&out_row[d], vaddq_f32(acc, x));
            }
#endif
            for (; d < dim; ++d) {
                out_row[d] += in_row[d];
            }
        }
        
        // Divide by seq_len
        int64_t d = 0;
#if defined(__AVX512F__)
        __m512 inv_seq_v = _mm512_set1_ps(inv_seq);
        for (; d + 16 <= dim; d += 16) {
            __m512 v = _mm512_loadu_ps(&out_row[d]);
            _mm512_storeu_ps(&out_row[d], _mm512_mul_ps(v, inv_seq_v));
        }
#elif defined(__AVX2__)
        __m256 inv_seq_v = _mm256_set1_ps(inv_seq);
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&out_row[d]);
            _mm256_storeu_ps(&out_row[d], _mm256_mul_ps(v, inv_seq_v));
        }
#elif defined(__ARM_NEON)
        float32x4_t inv_seq_v = vdupq_n_f32(inv_seq);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t v = vld1q_f32(&out_row[d]);
            vst1q_f32(&out_row[d], vmulq_f32(v, inv_seq_v));
        }
#endif
        for (; d < dim; ++d) {
            out_row[d] *= inv_seq;
        }
    }
}

/**
 * @brief GELU activation function with fast approximation.
 * Delegates to shared SIMD library.
 */
inline void continuous_thought_gelu(const float* input, float* output, int64_t size) {
    std::copy(input, input + size, output);
    simd_gelu_inplace(output, size);
}

/**
 * @brief Matrix-vector product: output = input @ weight + bias
 *
 * @param input Input vector [batch, in_dim]
 * @param weight Weight matrix [in_dim, out_dim]
 * @param bias Bias vector [out_dim]
 * @param output Output vector [batch, out_dim]
 * @param batch_size Batch dimension
 * @param in_dim Input dimension
 * @param out_dim Output dimension
 */
inline void continuous_thought_dense(
    const float* input, const float* weight, const float* bias,
    float* output, int64_t batch_size, int64_t in_dim, int64_t out_dim) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* in_row = input + b * in_dim;
        float* out_row = output + b * out_dim;
        
        for (int64_t o = 0; o < out_dim; ++o) {
            float sum = bias[o];
            
            int64_t i = 0;
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            for (; i + 16 <= in_dim; i += 16) {
                __m512 x = _mm512_loadu_ps(&in_row[i]);
                __m512 w = _mm512_loadu_ps(&weight[i * out_dim + o]);
                // Note: weight layout is [in_dim, out_dim], stride by out_dim
                // For efficient SIMD, use strided load or different layout
                // Fallback to scalar for now
            }
            // Simplified: scalar accumulation
            for (i = 0; i < in_dim; ++i) {
                sum += in_row[i] * weight[i * out_dim + o];
            }
#elif defined(__AVX2__)
            for (i = 0; i < in_dim; ++i) {
                sum += in_row[i] * weight[i * out_dim + o];
            }
#else
            for (i = 0; i < in_dim; ++i) {
                sum += in_row[i] * weight[i * out_dim + o];
            }
#endif
            out_row[o] = sum;
        }
    }
}

/**
 * @brief Broadcast thought to sequence and add with gating.
 *
 * output[b, s, d] = x[b, s, d] + gate[b, s, d] * thought[b, d]
 *
 * @param x Input hidden states [batch, seq_len, dim]
 * @param thought Final thought state [batch, dim]
 * @param gate Gating values [batch, seq_len, dim]
 * @param output Output tensor [batch, seq_len, dim]
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param dim Hidden dimension
 */
inline void continuous_thought_gated_broadcast(
    const float* x, const float* thought, const float* gate,
    float* output, int64_t batch_size, int64_t seq_len, int64_t dim) {
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t seq_idx = (b * seq_len + s) * dim;
            const float* x_row = x + seq_idx;
            const float* gate_row = gate + seq_idx;
            const float* thought_row = thought + b * dim;
            float* out_row = output + seq_idx;
            
            int64_t d = 0;
#if defined(__AVX512F__)
            for (; d + 16 <= dim; d += 16) {
                __m512 x_v = _mm512_loadu_ps(&x_row[d]);
                __m512 g_v = _mm512_loadu_ps(&gate_row[d]);
                __m512 t_v = _mm512_loadu_ps(&thought_row[d]);
                // out = x + gate * thought
                __m512 result = _mm512_fmadd_ps(g_v, t_v, x_v);
                _mm512_storeu_ps(&out_row[d], result);
            }
#elif defined(__AVX2__)
            for (; d + 8 <= dim; d += 8) {
                __m256 x_v = _mm256_loadu_ps(&x_row[d]);
                __m256 g_v = _mm256_loadu_ps(&gate_row[d]);
                __m256 t_v = _mm256_loadu_ps(&thought_row[d]);
                __m256 result = _mm256_fmadd_ps(g_v, t_v, x_v);
                _mm256_storeu_ps(&out_row[d], result);
            }
#elif defined(__ARM_NEON)
            for (; d + 4 <= dim; d += 4) {
                float32x4_t x_v = vld1q_f32(&x_row[d]);
                float32x4_t g_v = vld1q_f32(&gate_row[d]);
                float32x4_t t_v = vld1q_f32(&thought_row[d]);
                float32x4_t result = vmlaq_f32(x_v, g_v, t_v);
                vst1q_f32(&out_row[d], result);
            }
#endif
            for (; d < dim; ++d) {
                out_row[d] = x_row[d] + gate_row[d] * thought_row[d];
            }
        }
    }
}

/**
 * @brief Simple broadcast add (no gating).
 *
 * output[b, s, d] = x[b, s, d] + thought[b, d]
 */
inline void continuous_thought_broadcast_add(
    const float* x, const float* thought,
    float* output, int64_t batch_size, int64_t seq_len, int64_t dim) {
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t s = 0; s < seq_len; ++s) {
            const int64_t seq_idx = (b * seq_len + s) * dim;
            const float* x_row = x + seq_idx;
            const float* thought_row = thought + b * dim;
            float* out_row = output + seq_idx;
            
            int64_t d = 0;
#if defined(__AVX512F__)
            for (; d + 16 <= dim; d += 16) {
                __m512 x_v = _mm512_loadu_ps(&x_row[d]);
                __m512 t_v = _mm512_loadu_ps(&thought_row[d]);
                _mm512_storeu_ps(&out_row[d], _mm512_add_ps(x_v, t_v));
            }
#elif defined(__AVX2__)
            for (; d + 8 <= dim; d += 8) {
                __m256 x_v = _mm256_loadu_ps(&x_row[d]);
                __m256 t_v = _mm256_loadu_ps(&thought_row[d]);
                _mm256_storeu_ps(&out_row[d], _mm256_add_ps(x_v, t_v));
            }
#elif defined(__ARM_NEON)
            for (; d + 4 <= dim; d += 4) {
                float32x4_t x_v = vld1q_f32(&x_row[d]);
                float32x4_t t_v = vld1q_f32(&thought_row[d]);
                vst1q_f32(&out_row[d], vaddq_f32(x_v, t_v));
            }
#endif
            for (; d < dim; ++d) {
                out_row[d] = x_row[d] + thought_row[d];
            }
        }
    }
}

/**
 * @brief Sigmoid activation for gating.
 * Delegates to shared SIMD library.
 */
inline void continuous_thought_sigmoid(const float* input, float* output, int64_t size) {
    std::copy(input, input + size, output);
    simd_sigmoid_inplace(output, size);
}

/**
 * @brief Element-wise addition with SIMD.
 * Delegates to shared SIMD library.
 */
inline void continuous_thought_add(
    const float* a, const float* b, float* out, int64_t size) {
    simd_add(a, b, out, size);
}

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_FUSED_CONTINUOUS_THOUGHT_OP_H_
