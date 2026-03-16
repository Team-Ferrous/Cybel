// highnoon/_native/ops/fused_collapse_op.h
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
 * @file fused_collapse_op.h
 * @brief Fused Contextual Gating Collapse C++ kernel with SIMD optimization.
 *
 * Implements the collapse mechanism for superposition states using cross-attention.
 * This layer collapses quantum-inspired superposed representations into deterministic
 * outputs based on contextual relevance.
 *
 * Features:
 * - SIMD-optimized Q/K/V projections (AVX2/AVX512/NEON)
 * - Gumbel-Softmax sampling for unified train/inference
 * - Optional kernel attention linearization (ELU+1, ReLU²)
 * - Multi-head attention with parallel head processing
 * - OpenMP parallelization across batch dimension
 *
 * Migration from: highnoon/models/layers/collapse.py
 * Phase 16: Contextual Gating Collapse Enhancement
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_COLLAPSE_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_COLLAPSE_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <random>

// SIMD architecture detection
#if defined(__AVX512F__)
#include <immintrin.h>
#define HN_COLLAPSE_SIMD_AVX512 1
#define HN_COLLAPSE_SIMD_WIDTH 16
#elif defined(__AVX2__)
#include <immintrin.h>
#define HN_COLLAPSE_SIMD_AVX2 1
#define HN_COLLAPSE_SIMD_WIDTH 8
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define HN_COLLAPSE_SIMD_NEON 1
#define HN_COLLAPSE_SIMD_WIDTH 4
#else
#define HN_COLLAPSE_SIMD_SCALAR 1
#define HN_COLLAPSE_SIMD_WIDTH 1
#endif

namespace highnoon {
namespace ops {

// =============================================================================
// GUMBEL NOISE GENERATION
// =============================================================================

/**
 * @brief Thread-local random number generator for Gumbel noise.
 */
inline std::mt19937& get_thread_local_rng() {
    thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

/**
 * @brief Generate Gumbel noise in-place: -log(-log(U)) where U ~ Uniform(0,1).
 *
 * Uses thread-local RNG for thread safety.
 *
 * @param data Array to fill with Gumbel noise.
 * @param size Number of elements.
 */
template <typename T>
inline void generate_gumbel_noise(T* data, int64_t size) {
    std::mt19937& rng = get_thread_local_rng();
    std::uniform_real_distribution<T> dist(static_cast<T>(1e-9), static_cast<T>(1.0 - 1e-9));
    
    for (int64_t i = 0; i < size; ++i) {
        T u = dist(rng);
        data[i] = -std::log(-std::log(u));
    }
}

// =============================================================================
// GUMBEL-SOFTMAX SAMPLING
// =============================================================================

/**
 * @brief Vectorized Gumbel-Softmax with temperature and optional hard mode.
 *
 * Computes: softmax((logits + gumbel_noise) / temperature)
 * If hard=true, uses straight-through estimator for one-hot output.
 *
 * @param logits Input logits array [size].
 * @param gumbel_noise Pre-generated Gumbel noise [size].
 * @param output Output probabilities [size].
 * @param size Number of elements.
 * @param temperature Softmax temperature (lower = sharper).
 * @param hard If true, return one-hot with straight-through gradient.
 */
template <typename T>
inline void simd_gumbel_softmax(
    const T* logits,
    const T* gumbel_noise,
    T* output,
    int64_t size,
    float temperature,
    bool hard) {
    
    // Step 1: Add Gumbel noise and divide by temperature
    T inv_temp = static_cast<T>(1.0 / temperature);
    T max_val = -std::numeric_limits<T>::infinity();
    
    // Find max for numerical stability
    for (int64_t i = 0; i < size; ++i) {
        output[i] = (logits[i] + gumbel_noise[i]) * inv_temp;
        max_val = std::max(max_val, output[i]);
    }
    
    // Compute exp(x - max) and sum
    T sum = static_cast<T>(0);
    int64_t i = 0;
    
#if defined(HN_COLLAPSE_SIMD_AVX512)
    if (std::is_same<T, float>::value) {
        const __m512 max_vec = _mm512_set1_ps(static_cast<float>(max_val));
        __m512 sum_vec = _mm512_setzero_ps();
        
        for (; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(reinterpret_cast<const float*>(&output[i]));
            v = _mm512_sub_ps(v, max_vec);
            // exp approximation: 1 + x + x²/2 + x³/6
            __m512 v2 = _mm512_mul_ps(v, v);
            __m512 v3 = _mm512_mul_ps(v2, v);
            __m512 exp_v = _mm512_add_ps(_mm512_set1_ps(1.0f), v);
            exp_v = _mm512_fmadd_ps(v2, _mm512_set1_ps(0.5f), exp_v);
            exp_v = _mm512_fmadd_ps(v3, _mm512_set1_ps(0.16666667f), exp_v);
            // Clamp to positive
            exp_v = _mm512_max_ps(exp_v, _mm512_set1_ps(1e-9f));
            _mm512_storeu_ps(reinterpret_cast<float*>(&output[i]), exp_v);
            sum_vec = _mm512_add_ps(sum_vec, exp_v);
        }
        sum = static_cast<T>(_mm512_reduce_add_ps(sum_vec));
    }
#elif defined(HN_COLLAPSE_SIMD_AVX2)
    if (std::is_same<T, float>::value) {
        const __m256 max_vec = _mm256_set1_ps(static_cast<float>(max_val));
        __m256 sum_vec = _mm256_setzero_ps();
        
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_loadu_ps(reinterpret_cast<const float*>(&output[i]));
            v = _mm256_sub_ps(v, max_vec);
            __m256 v2 = _mm256_mul_ps(v, v);
            __m256 v3 = _mm256_mul_ps(v2, v);
            __m256 exp_v = _mm256_add_ps(_mm256_set1_ps(1.0f), v);
            exp_v = _mm256_fmadd_ps(v2, _mm256_set1_ps(0.5f), exp_v);
            exp_v = _mm256_fmadd_ps(v3, _mm256_set1_ps(0.16666667f), exp_v);
            exp_v = _mm256_max_ps(exp_v, _mm256_set1_ps(1e-9f));
            _mm256_storeu_ps(reinterpret_cast<float*>(&output[i]), exp_v);
            sum_vec = _mm256_add_ps(sum_vec, exp_v);
        }
        // Horizontal sum for AVX2
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum = static_cast<T>(_mm_cvtss_f32(sum128));
    }
#elif defined(HN_COLLAPSE_SIMD_NEON)
    if (std::is_same<T, float>::value) {
        const float32x4_t max_vec = vdupq_n_f32(static_cast<float>(max_val));
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        
        for (; i + 4 <= size; i += 4) {
            float32x4_t v = vld1q_f32(reinterpret_cast<const float*>(&output[i]));
            v = vsubq_f32(v, max_vec);
            float32x4_t v2 = vmulq_f32(v, v);
            float32x4_t v3 = vmulq_f32(v2, v);
            float32x4_t exp_v = vaddq_f32(vdupq_n_f32(1.0f), v);
            exp_v = vmlaq_n_f32(exp_v, v2, 0.5f);
            exp_v = vmlaq_n_f32(exp_v, v3, 0.16666667f);
            exp_v = vmaxq_f32(exp_v, vdupq_n_f32(1e-9f));
            vst1q_f32(reinterpret_cast<float*>(&output[i]), exp_v);
            sum_vec = vaddq_f32(sum_vec, exp_v);
        }
        sum = static_cast<T>(vaddvq_f32(sum_vec));
    }
#endif
    
    // Scalar remainder (or full pass if T is not float)
    for (; i < size; ++i) {
        T v = output[i] - max_val;
        T exp_v = std::exp(v);
        output[i] = exp_v;
        sum += exp_v;
    }
    
    // Normalize
    T inv_sum = (sum > static_cast<T>(0)) ? (static_cast<T>(1) / sum) : static_cast<T>(0);
    i = 0;
    
#if defined(HN_COLLAPSE_SIMD_AVX512)
    if (std::is_same<T, float>::value) {
        const __m512 inv_sum_vec = _mm512_set1_ps(static_cast<float>(inv_sum));
        for (; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(reinterpret_cast<float*>(&output[i]));
            _mm512_storeu_ps(reinterpret_cast<float*>(&output[i]), _mm512_mul_ps(v, inv_sum_vec));
        }
    }
#elif defined(HN_COLLAPSE_SIMD_AVX2)
    if (std::is_same<T, float>::value) {
        const __m256 inv_sum_vec8 = _mm256_set1_ps(static_cast<float>(inv_sum));
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_loadu_ps(reinterpret_cast<float*>(&output[i]));
            _mm256_storeu_ps(reinterpret_cast<float*>(&output[i]), _mm256_mul_ps(v, inv_sum_vec8));
        }
    }
#elif defined(HN_COLLAPSE_SIMD_NEON)
    if (std::is_same<T, float>::value) {
        const float32x4_t inv_sum_vec4 = vdupq_n_f32(static_cast<float>(inv_sum));
        for (; i + 4 <= size; i += 4) {
            float32x4_t v = vld1q_f32(reinterpret_cast<float*>(&output[i]));
            vst1q_f32(reinterpret_cast<float*>(&output[i]), vmulq_f32(v, inv_sum_vec4));
        }
    }
#endif
    
    for (; i < size; ++i) {
        output[i] *= inv_sum;
    }
    
    // Hard mode: straight-through one-hot
    if (hard) {
        int64_t max_idx = 0;
        T max_prob = output[0];
        for (int64_t j = 1; j < size; ++j) {
            if (output[j] > max_prob) {
                max_prob = output[j];
                max_idx = j;
            }
        }
        // Store soft probs temporarily, then overwrite with hard
        // The gradient flows through the soft values (straight-through)
        for (int64_t j = 0; j < size; ++j) {
            output[j] = (j == max_idx) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
}

// =============================================================================
// KERNEL ATTENTION FEATURE MAPS
// =============================================================================

/**
 * @brief ReLU² feature map: φ(x) = relu(x)².
 *
 * @param data Float array to apply feature map in-place.
 * @param size Number of elements.
 */
inline void simd_relu_squared_inplace(float* data, int64_t size) {
    int64_t i = 0;
    
#if defined(HN_COLLAPSE_SIMD_AVX512)
    const __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        v = _mm512_max_ps(v, zero);  // ReLU
        v = _mm512_mul_ps(v, v);      // Square
        _mm512_storeu_ps(&data[i], v);
    }
#elif defined(HN_COLLAPSE_SIMD_AVX2)
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = _mm256_max_ps(v, zero);
        v = _mm256_mul_ps(v, v);
        _mm256_storeu_ps(&data[i], v);
    }
#elif defined(HN_COLLAPSE_SIMD_NEON)
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        v = vmaxq_f32(v, zero);
        v = vmulq_f32(v, v);
        vst1q_f32(&data[i], v);
    }
#endif
    
    for (; i < size; ++i) {
        float v = std::max(0.0f, data[i]);
        data[i] = v * v;
    }
}

/**
 * @brief ELU+1 feature map: φ(x) = elu(x) + 1.
 *
 * For x > 0: φ(x) = x + 1
 * For x ≤ 0: φ(x) = exp(x)
 *
 * @param data Float array to apply feature map in-place.
 * @param size Number of elements.
 */
inline void simd_elu_plus_one_inplace(float* data, int64_t size) {
    int64_t i = 0;
    
#if defined(HN_COLLAPSE_SIMD_AVX512)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        __mmask16 pos_mask = _mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ);
        // x > 0: x + 1
        __m512 pos_result = _mm512_add_ps(v, one);
        // x <= 0: exp(x) approximation
        __m512 v2 = _mm512_mul_ps(v, v);
        __m512 v3 = _mm512_mul_ps(v2, v);
        __m512 neg_result = _mm512_add_ps(one, v);
        neg_result = _mm512_fmadd_ps(v2, _mm512_set1_ps(0.5f), neg_result);
        neg_result = _mm512_fmadd_ps(v3, _mm512_set1_ps(0.16666667f), neg_result);
        __m512 result = _mm512_mask_blend_ps(pos_mask, neg_result, pos_result);
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(HN_COLLAPSE_SIMD_AVX2)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 pos_mask = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
        __m256 pos_result = _mm256_add_ps(v, one);
        __m256 v2 = _mm256_mul_ps(v, v);
        __m256 v3 = _mm256_mul_ps(v2, v);
        __m256 neg_result = _mm256_add_ps(one, v);
        neg_result = _mm256_fmadd_ps(v2, _mm256_set1_ps(0.5f), neg_result);
        neg_result = _mm256_fmadd_ps(v3, _mm256_set1_ps(0.16666667f), neg_result);
        __m256 result = _mm256_blendv_ps(neg_result, pos_result, pos_mask);
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(HN_COLLAPSE_SIMD_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        uint32x4_t pos_mask = vcgtq_f32(v, zero);
        float32x4_t pos_result = vaddq_f32(v, one);
        float32x4_t v2 = vmulq_f32(v, v);
        float32x4_t v3 = vmulq_f32(v2, v);
        float32x4_t neg_result = vaddq_f32(one, v);
        neg_result = vmlaq_n_f32(neg_result, v2, 0.5f);
        neg_result = vmlaq_n_f32(neg_result, v3, 0.16666667f);
        float32x4_t result = vbslq_f32(pos_mask, pos_result, neg_result);
        vst1q_f32(&data[i], result);
    }
#endif
    
    for (; i < size; ++i) {
        float x = data[i];
        data[i] = (x > 0.0f) ? (x + 1.0f) : std::exp(x);
    }
}

// =============================================================================
// SIMD MATRIX-VECTOR MULTIPLICATION
// =============================================================================

/**
 * @brief SIMD-optimized matrix-vector multiply: y = W @ x + b.
 *
 * @param weights Weight matrix [out_dim, in_dim] row-major.
 * @param input Input vector [in_dim].
 * @param bias Bias vector [out_dim].
 * @param output Output vector [out_dim].
 * @param out_dim Output dimension.
 * @param in_dim Input dimension.
 */
inline void simd_matvec_add(
    const float* weights,
    const float* input,
    const float* bias,
    float* output,
    int64_t out_dim,
    int64_t in_dim) {
    
    #pragma omp parallel for
    for (int64_t o = 0; o < out_dim; ++o) {
        const float* row = weights + o * in_dim;
        float sum = 0.0f;
        int64_t i = 0;
        
#if defined(HN_COLLAPSE_SIMD_AVX512)
        __m512 acc = _mm512_setzero_ps();
        for (; i + 16 <= in_dim; i += 16) {
            __m512 w = _mm512_loadu_ps(&row[i]);
            __m512 x = _mm512_loadu_ps(&input[i]);
            acc = _mm512_fmadd_ps(w, x, acc);
        }
        sum = _mm512_reduce_add_ps(acc);
#elif defined(HN_COLLAPSE_SIMD_AVX2)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= in_dim; i += 8) {
            __m256 w = _mm256_loadu_ps(&row[i]);
            __m256 x = _mm256_loadu_ps(&input[i]);
            acc = _mm256_fmadd_ps(w, x, acc);
        }
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum = _mm_cvtss_f32(sum128);
#elif defined(HN_COLLAPSE_SIMD_NEON)
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; i + 4 <= in_dim; i += 4) {
            float32x4_t w = vld1q_f32(&row[i]);
            float32x4_t x = vld1q_f32(&input[i]);
            acc = vmlaq_f32(acc, w, x);
        }
        sum = vaddvq_f32(acc);
#endif
        
        for (; i < in_dim; ++i) {
            sum += row[i] * input[i];
        }
        
        output[o] = sum + bias[o];
    }
}

// =============================================================================
// FORWARD/BACKWARD KERNEL DECLARATIONS
// =============================================================================

/**
 * @brief Fused collapse forward pass.
 *
 * Implements: y = collapse(context, superposed) using multi-head cross-attention
 * with Gumbel-Softmax for unified training/inference.
 *
 * @param context Context tensor [batch, d_in] for query generation.
 * @param superposed Superposed states [batch, superposition_dim, d_out].
 * @param q_weights Query projection weights [d_in, d_out].
 * @param k_weights Key projection weights [d_out, d_out].
 * @param v_weights Value projection weights [d_out, d_out].
 * @param o_weights Output projection weights [d_out, d_out].
 * @param q_bias Query bias [d_out].
 * @param k_bias Key bias [d_out].
 * @param v_bias Value bias [d_out].
 * @param o_bias Output bias [d_out].
 * @param output Output tensor [batch, d_out].
 * @param attention_cache Cached attention weights for backward [batch, num_heads, superposition_dim].
 * @param batch Batch size.
 * @param superposition_dim Superposition dimension (S).
 * @param d_in Context input dimension.
 * @param d_out Model dimension.
 * @param num_heads Number of attention heads.
 * @param temperature Gumbel-Softmax temperature.
 * @param training Whether in training mode.
 * @param use_kernel_attention Use kernel attention feature map.
 * @param feature_map Feature map type: 0=softmax, 1=elu_plus_one, 2=relu_squared.
 */
template <typename T>
void FusedCollapseForward(
    const T* context,
    const T* superposed,
    const T* q_weights,
    const T* k_weights,
    const T* v_weights,
    const T* o_weights,
    const T* q_bias,
    const T* k_bias,
    const T* v_bias,
    const T* o_bias,
    T* output,
    T* attention_cache,
    int batch,
    int superposition_dim,
    int d_in,
    int d_out,
    int num_heads,
    float temperature,
    bool training,
    bool use_kernel_attention,
    int feature_map);

/**
 * @brief Fused collapse backward pass.
 *
 * Computes gradients for context, superposed, and all projection weights/biases.
 *
 * @param grad_output Gradient of output [batch, d_out].
 * @param context Context tensor [batch, d_in].
 * @param superposed Superposed states [batch, superposition_dim, d_out].
 * @param attention_cache Cached attention weights [batch, num_heads, superposition_dim].
 * @param q_weights Query weights [d_in, d_out].
 * @param k_weights Key weights [d_out, d_out].
 * @param v_weights Value weights [d_out, d_out].
 * @param o_weights Output weights [d_out, d_out].
 * @param grad_context Output: gradient for context [batch, d_in].
 * @param grad_superposed Output: gradient for superposed [batch, S, d_out].
 * @param grad_q_weights Output: gradient for Q weights.
 * @param grad_k_weights Output: gradient for K weights.
 * @param grad_v_weights Output: gradient for V weights.
 * @param grad_o_weights Output: gradient for O weights.
 * @param grad_q_bias Output: gradient for Q bias.
 * @param grad_k_bias Output: gradient for K bias.
 * @param grad_v_bias Output: gradient for V bias.
 * @param grad_o_bias Output: gradient for O bias.
 * @param batch Batch size.
 * @param superposition_dim Superposition dimension.
 * @param d_in Context dimension.
 * @param d_out Model dimension.
 * @param num_heads Number of heads.
 */
template <typename T>
void FusedCollapseBackward(
    const T* grad_output,
    const T* context,
    const T* superposed,
    const T* attention_cache,
    const T* q_weights,
    const T* k_weights,
    const T* v_weights,
    const T* o_weights,
    T* grad_context,
    T* grad_superposed,
    T* grad_q_weights,
    T* grad_k_weights,
    T* grad_v_weights,
    T* grad_o_weights,
    T* grad_q_bias,
    T* grad_k_bias,
    T* grad_v_bias,
    T* grad_o_bias,
    int batch,
    int superposition_dim,
    int d_in,
    int d_out,
    int num_heads);

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_COLLAPSE_OP_H_
