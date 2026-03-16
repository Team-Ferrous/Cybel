// saguaro.native/ops/fused_linear_gqa_op.h
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
 * @file fused_linear_gqa_op.h
 * @brief Linear Grouped-Query Attention (Linear GQA) SIMD helpers.
 *
 * Implements O(n) linear GQA using kernel feature maps to approximate softmax:
 *   - FAVOR# random features (trigonometric basis)
 *   - ELU feature map (ELU(x) + 1)
 *   - EXP feature map (bounded exponential)
 *
 * Combined with GQA's KV head sharing for both memory and compute efficiency.
 *
 * Complexity: O(n) instead of O(n²) for standard attention.
 *
 * SIMD optimizations:
 * - AVX512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - NEON: 4-wide vectorization (ARM)
 * - Scalar fallback for all architectures
 *
 * Functions use the linear_gqa_ prefix to avoid ODR violations.
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef SAGUARO_NATIVE_OPS_FUSED_LINEAR_GQA_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_LINEAR_GQA_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <random>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define SAGUARO_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define SAGUARO_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SAGUARO_NEON 1
#endif

namespace saguaro {
namespace ops {

// =============================================================================
// FEATURE MAP TYPES
// =============================================================================

enum class FeatureMapType : int {
    ELU = 0,     // ELU(x) + 1
    EXP = 1,     // Bounded exponential
    FAVOR = 2   // FAVOR# random features
};

// =============================================================================
// SIMD FEATURE MAP IMPLEMENTATIONS
// =============================================================================

/**
 * @brief Apply ELU feature map: φ(x) = ELU(x) + 1 = max(x, 0) + alpha*(exp(min(x,0)) - 1) + 1
 *
 * Simple and effective feature map that ensures non-negative outputs.
 * Uses scalar fallback for simplicity and compatibility.
 */
inline void linear_gqa_feature_map_elu(
    const float* input, float* output,
    int64_t size) {
    
    // Use scalar implementation for maximum compatibility
    // SIMD version can be added after exp functions are validated
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        float x = input[i];
        float elu = (x > 0.0f) ? x : (std::exp(x) - 1.0f);
        output[i] = elu + 1.0f;
    }
}


/**
 * @brief Apply bounded exponential feature map: φ(x) = exp(clamp(x, -10, 10))
 *
 * Closer to softmax behavior but with bounded output to prevent overflow.
 * Uses scalar implementation for compatibility.
 */
inline void linear_gqa_feature_map_exp(
    const float* input, float* output,
    int64_t size) {
    
    const float min_val = -10.0f;
    const float max_val = 10.0f;

    // Use scalar implementation for maximum compatibility
    #pragma omp parallel for
    for (int64_t i = 0; i < size; ++i) {
        float x = std::max(min_val, std::min(max_val, input[i]));
        output[i] = std::exp(x);
    }
}

/**
 * @brief Apply FAVOR# random features: φ(x) = exp(-||x||²/2) * [cos(Ωx), sin(Ωx)] / √m
 *
 * Random features using orthogonal projection matrix Ω.
 * Provides better approximation of softmax attention than ELU.
 *
 * @param input Input tensor [size]
 * @param output Output features [size * 2] (cos and sin components)
 * @param random_features Random projection matrix [head_dim, num_random_features]
 * @param head_dim Dimension of each head
 * @param num_random_features Number of random features (output dimension = 2 * this)
 */
inline void linear_gqa_feature_map_favor(
    const float* input, float* output,
    const float* random_features,
    int64_t num_positions,
    int64_t head_dim,
    int64_t num_random_features) {
    
    const float scale = 1.0f / std::sqrt(static_cast<float>(num_random_features));
    
    for (int64_t p = 0; p < num_positions; ++p) {
        const float* x = input + p * head_dim;
        float* out = output + p * (2 * num_random_features);
        
        // Compute ||x||²
        float norm_sq = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
            norm_sq += x[d] * x[d];
        }
        float normalizer = std::exp(-norm_sq / 2.0f) * scale;
        
        // Compute projections: x @ Ω
        for (int64_t r = 0; r < num_random_features; ++r) {
            float projection = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                projection += x[d] * random_features[d * num_random_features + r];
            }
            // Trigonometric features
            out[r] = normalizer * std::cos(projection);
            out[num_random_features + r] = normalizer * std::sin(projection);
        }
    }
}

// =============================================================================
// APPROXIMATE EXP FUNCTIONS FOR SIMD
// =============================================================================

#if defined(SAGUARO_AVX512)
/**
 * @brief Fast approximate exp for AVX512.
 * Uses polynomial approximation: exp(x) ≈ 2^(x * log2(e))
 */
inline __m512 highnoon_mm512_exp_ps(__m512 x) {
    const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
    const __m512 c1 = _mm512_set1_ps(0.693147180559945f);
    const __m512 c2 = _mm512_set1_ps(0.240226506959101f);
    const __m512 c3 = _mm512_set1_ps(0.0558263180532956f);
    const __m512 one = _mm512_set1_ps(1.0f);
    
    __m512 t = _mm512_mul_ps(x, log2e);
    __m512 ti = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT);
    __m512 tf = _mm512_sub_ps(t, ti);
    
    // Polynomial approximation for 2^tf
    __m512 p = _mm512_fmadd_ps(c3, tf, c2);
    p = _mm512_fmadd_ps(p, tf, c1);
    p = _mm512_fmadd_ps(p, tf, one);
    
    // Scale by 2^ti
    __m512i ti_int = _mm512_cvtps_epi32(ti);
    ti_int = _mm512_add_epi32(ti_int, _mm512_set1_epi32(127));
    ti_int = _mm512_slli_epi32(ti_int, 23);
    __m512 scale = _mm512_castsi512_ps(ti_int);
    
    return _mm512_mul_ps(p, scale);
}
#endif

#if defined(SAGUARO_AVX2)
/**
 * @brief Fast approximate exp for AVX2.
 */
inline __m256 highnoon_mm256_exp_ps(__m256 x) {
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c1 = _mm256_set1_ps(0.693147180559945f);
    const __m256 c2 = _mm256_set1_ps(0.240226506959101f);
    const __m256 c3 = _mm256_set1_ps(0.0558263180532956f);
    const __m256 one = _mm256_set1_ps(1.0f);
    
    __m256 t = _mm256_mul_ps(x, log2e);
    __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT);
    __m256 tf = _mm256_sub_ps(t, ti);
    
    // Polynomial approximation
    __m256 p = _mm256_fmadd_ps(c3, tf, c2);
    p = _mm256_fmadd_ps(p, tf, c1);
    p = _mm256_fmadd_ps(p, tf, one);
    
    // Scale by 2^ti
    __m256i ti_int = _mm256_cvtps_epi32(ti);
    ti_int = _mm256_add_epi32(ti_int, _mm256_set1_epi32(127));
    ti_int = _mm256_slli_epi32(ti_int, 23);
    __m256 scale = _mm256_castsi256_ps(ti_int);
    
    return _mm256_mul_ps(p, scale);
}
#endif

#if defined(SAGUARO_NEON)
/**
 * @brief Fast approximate exp for NEON.
 */
inline float32x4_t highnoon_vexpq_f32(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    const float32x4_t c1 = vdupq_n_f32(0.693147180559945f);
    const float32x4_t c2 = vdupq_n_f32(0.240226506959101f);
    const float32x4_t c3 = vdupq_n_f32(0.0558263180532956f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    
    float32x4_t t = vmulq_f32(x, log2e);
    float32x4_t ti = vrndnq_f32(t);
    float32x4_t tf = vsubq_f32(t, ti);
    
    // Polynomial
    float32x4_t p = vfmaq_f32(c2, c3, tf);
    p = vfmaq_f32(c1, p, tf);
    p = vfmaq_f32(one, p, tf);
    
    // Scale
    int32x4_t ti_int = vcvtq_s32_f32(ti);
    ti_int = vaddq_s32(ti_int, vdupq_n_s32(127));
    ti_int = vshlq_n_s32(ti_int, 23);
    float32x4_t scale = vreinterpretq_f32_s32(ti_int);
    
    return vmulq_f32(p, scale);
}
#endif

// =============================================================================
// GQA KV HEAD EXPANSION
// =============================================================================

/**
 * @brief Expand KV heads to match query heads via repetition.
 *
 * If num_heads = 8 and num_kv_heads = 2, each KV head is repeated 4 times.
 *
 * @param input Input KV tensor [batch, num_kv_heads, seq, head_dim]
 * @param output Output tensor [batch, num_heads, seq, head_dim]
 * @param batch_size Batch size
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of KV heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 */
inline void linear_gqa_expand_kv_heads(
    const float* input, float* output,
    int64_t batch_size,
    int64_t num_kv_heads,
    int64_t num_queries_per_kv,
    int64_t seq_len,
    int64_t head_dim) {
    
    const int64_t num_heads = num_kv_heads * num_queries_per_kv;
    const int64_t kv_stride = seq_len * head_dim;
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t kv_h = 0; kv_h < num_kv_heads; ++kv_h) {
            const float* kv_head = input + b * num_kv_heads * kv_stride + kv_h * kv_stride;
            
            // Copy to each query head that shares this KV head
            for (int64_t q = 0; q < num_queries_per_kv; ++q) {
                int64_t q_head = kv_h * num_queries_per_kv + q;
                float* out_head = output + b * num_heads * kv_stride + q_head * kv_stride;
                
                std::copy(kv_head, kv_head + kv_stride, out_head);
            }
        }
    }
}

// =============================================================================
// CAUSAL LINEAR ATTENTION (O(n) COMPLEXITY)
// =============================================================================

/**
 * @brief Causal linear attention forward pass with cumulative sum.
 *
 * For autoregressive models where each position only attends to past.
 * Uses cumulative sum formulation for O(n) complexity:
 *
 *   S_t = Σ_{i≤t} φ(K_i)^T ⊗ V_i  (cumsum of outer products)
 *   Z_t = Σ_{i≤t} φ(K_i)          (cumsum of keys for normalization)
 *   O_t = φ(Q_t) @ S_t / (φ(Q_t) · Z_t + ε)
 *
 * @param q_features Query with feature map applied [batch, heads, seq, feature_dim]
 * @param k_features Key with feature map applied [batch, heads, seq, feature_dim]
 * @param v Values [batch, heads, seq, head_dim]
 * @param output Output tensor [batch, heads, seq, head_dim]
 * @param kv_state Running KV state [batch, heads, feature_dim, head_dim]
 * @param k_sum_state Running K sum [batch, heads, feature_dim]
 */
template<typename T>
void linear_gqa_causal_attention_forward(
    const T* q_features,
    const T* k_features,
    const T* v,
    T* output,
    T* kv_state,        // [B, H, F, D] - F = feature_dim
    T* k_sum_state,     // [B, H, F]
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t feature_dim,
    int64_t head_dim,
    T eps) {
    
    const int64_t bhld_stride_b = num_heads * seq_len * head_dim;
    const int64_t bhld_stride_h = seq_len * head_dim;
    const int64_t bhlf_stride_b = num_heads * seq_len * feature_dim;
    const int64_t bhlf_stride_h = seq_len * feature_dim;
    
    const int64_t bhfd_stride_b = num_heads * feature_dim * head_dim;
    const int64_t bhfd_stride_h = feature_dim * head_dim;
    const int64_t bhf_stride_b = num_heads * feature_dim;
    const int64_t bhf_stride_h = feature_dim;
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            // Pointers for this batch/head
            const T* q_f = q_features + b * bhlf_stride_b + h * bhlf_stride_h;
            const T* k_f = k_features + b * bhlf_stride_b + h * bhlf_stride_h;
            const T* v_ptr = v + b * bhld_stride_b + h * bhld_stride_h;
            T* out_ptr = output + b * bhld_stride_b + h * bhld_stride_h;
            T* kv = kv_state + b * bhfd_stride_b + h * bhfd_stride_h;
            T* k_sum = k_sum_state + b * bhf_stride_b + h * bhf_stride_h;
            
            // Initialize state to zero
            std::fill(kv, kv + feature_dim * head_dim, T(0));
            std::fill(k_sum, k_sum + feature_dim, T(0));
            
            // Process sequence causally
            for (int64_t t = 0; t < seq_len; ++t) {
                const T* q_t = q_f + t * feature_dim;
                const T* k_t = k_f + t * feature_dim;
                const T* v_t = v_ptr + t * head_dim;
                T* o_t = out_ptr + t * head_dim;
                
                // Update running state: kv += k_t ⊗ v_t, k_sum += k_t
                for (int64_t f = 0; f < feature_dim; ++f) {
                    k_sum[f] += k_t[f];
                    for (int64_t d = 0; d < head_dim; ++d) {
                        kv[f * head_dim + d] += k_t[f] * v_t[d];
                    }
                }
                
                // Compute output: o_t = q_t @ kv / (q_t · k_sum + eps)
                T denom = eps;
                for (int64_t f = 0; f < feature_dim; ++f) {
                    denom += q_t[f] * k_sum[f];
                }
                
                for (int64_t d = 0; d < head_dim; ++d) {
                    T num = T(0);
                    for (int64_t f = 0; f < feature_dim; ++f) {
                        num += q_t[f] * kv[f * head_dim + d];
                    }
                    o_t[d] = num / denom;
                }
            }
        }
    }
}

/**
 * @brief Non-causal linear attention forward pass.
 *
 * Full bidirectional attention but with O(n) complexity.
 */
template<typename T>
void linear_gqa_bidirectional_attention_forward(
    const T* q_features,
    const T* k_features,
    const T* v,
    T* output,
    T* kv_cache,        // [B, H, F, D]
    T* k_sum_cache,     // [B, H, F]
    int64_t batch_size,
    int64_t num_heads,
    int64_t seq_len,
    int64_t feature_dim,
    int64_t head_dim,
    T eps) {
    
    const int64_t bhld_stride_b = num_heads * seq_len * head_dim;
    const int64_t bhld_stride_h = seq_len * head_dim;
    const int64_t bhlf_stride_b = num_heads * seq_len * feature_dim;
    const int64_t bhlf_stride_h = seq_len * feature_dim;
    
    const int64_t bhfd_stride_b = num_heads * feature_dim * head_dim;
    const int64_t bhfd_stride_h = feature_dim * head_dim;
    const int64_t bhf_stride_b = num_heads * feature_dim;
    const int64_t bhf_stride_h = feature_dim;
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t h = 0; h < num_heads; ++h) {
            const T* q_f = q_features + b * bhlf_stride_b + h * bhlf_stride_h;
            const T* k_f = k_features + b * bhlf_stride_b + h * bhlf_stride_h;
            const T* v_ptr = v + b * bhld_stride_b + h * bhld_stride_h;
            T* out_ptr = output + b * bhld_stride_b + h * bhld_stride_h;
            T* kv = kv_cache + b * bhfd_stride_b + h * bhfd_stride_h;
            T* k_sum = k_sum_cache + b * bhf_stride_b + h * bhf_stride_h;
            
            // Step 1: Compute KV = sum_t (k_t ⊗ v_t) and K_sum = sum_t k_t
            std::fill(kv, kv + feature_dim * head_dim, T(0));
            std::fill(k_sum, k_sum + feature_dim, T(0));
            
            for (int64_t t = 0; t < seq_len; ++t) {
                const T* k_t = k_f + t * feature_dim;
                const T* v_t = v_ptr + t * head_dim;
                
                for (int64_t f = 0; f < feature_dim; ++f) {
                    k_sum[f] += k_t[f];
                    for (int64_t d = 0; d < head_dim; ++d) {
                        kv[f * head_dim + d] += k_t[f] * v_t[d];
                    }
                }
            }
            
            // Step 2: Compute output for each position
            for (int64_t t = 0; t < seq_len; ++t) {
                const T* q_t = q_f + t * feature_dim;
                T* o_t = out_ptr + t * head_dim;
                
                T denom = eps;
                for (int64_t f = 0; f < feature_dim; ++f) {
                    denom += q_t[f] * k_sum[f];
                }
                
                for (int64_t d = 0; d < head_dim; ++d) {
                    T num = T(0);
                    for (int64_t f = 0; f < feature_dim; ++f) {
                        num += q_t[f] * kv[f * head_dim + d];
                    }
                    o_t[d] = num / denom;
                }
            }
        }
    }
}

// =============================================================================
// RANDOM FEATURE GENERATION FOR FAVOR#
// =============================================================================

/**
 * @brief Generate orthogonal random features using QR decomposition.
 *
 * FAVOR# uses orthonormal random features which have lower variance
 * than standard random features (FAVOR+).
 *
 * @param output Output matrix [head_dim, num_features]
 * @param head_dim Input dimension
 * @param num_features Number of random features
 * @param seed Random seed for reproducibility
 */
inline void linear_gqa_generate_random_features(
    float* output,
    int64_t head_dim,
    int64_t num_features,
    uint64_t seed = 42) {
    
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Generate random Gaussian matrix
    for (int64_t i = 0; i < head_dim * num_features; ++i) {
        output[i] = dist(rng);
    }
    
    // Simplified Gram-Schmidt orthogonalization for FAVOR#
    // For production, use proper QR decomposition
    const float scale = std::sqrt(static_cast<float>(num_features));
    
    for (int64_t r = 0; r < num_features; ++r) {
        float* col = output + r;
        
        // Orthogonalize against previous columns
        for (int64_t p = 0; p < r; ++p) {
            float* prev_col = output + p;
            
            // Compute dot product
            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += col[d * num_features] * prev_col[d * num_features];
            }
            
            // Subtract projection
            for (int64_t d = 0; d < head_dim; ++d) {
                col[d * num_features] -= dot * prev_col[d * num_features];
            }
        }
        
        // Normalize
        float norm = 0.0f;
        for (int64_t d = 0; d < head_dim; ++d) {
            norm += col[d * num_features] * col[d * num_features];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (int64_t d = 0; d < head_dim; ++d) {
                col[d * num_features] = col[d * num_features] / norm * scale;
            }
        }
    }
}

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_FUSED_LINEAR_GQA_OP_H_
