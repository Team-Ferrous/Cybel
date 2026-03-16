// highnoon/_native/ops/unified_attention_op.h
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file unified_attention_op.h
 * @brief Unified Attention Operations for HighNoon Framework
 *
 * This header consolidates all 11 attention mechanisms into a single unified
 * interface with shared SIMD kernels. This reduces code duplication, binary
 * size, and maintenance overhead while maintaining full functionality.
 *
 * Supported Attention Modes:
 * - FLASH:           Full O(n²) with memory-efficient tiling
 * - LINEAR:          O(n) via ELU+1 feature map approximation  
 * - LOCAL_WINDOWED:  O(n×w) with window size w (Griffin-style)
 * - DIFFERENTIAL:    λ₁A₁ - λ₂A₂ style (ICLR 2025 Differential Transformer)
 * - SPARSE_NSA:      Native Sparse Attention O(n log n)
 * - GQA:             Grouped-Query Attention
 * - LINEAR_GQA:      Linear + GQA combined
 * - SLIDING_GQA:     Sliding window + GQA
 * - LATENT_KV:       Latent KV compression
 * - QASA:            Quantum Adaptive Self-Attention
 * - LMWT:            Learnable Multi-Scale Wavelet Transformer
 *
 * SIMD Support:
 * - AVX-512: 16-wide vectorization (Intel Xeon, newer Core)
 * - AVX2:    8-wide vectorization (Primary target: AMD Ryzen/EPYC, Intel 6th+)
 * - NEON:    4-wide vectorization (Apple M1/M2/M3, AWS Graviton)
 * - Scalar fallback for all architectures
 *
 * Thread Safety: All functions are reentrant with no shared state.
 * Precision: float32 only (maintains O(n) linear complexity)
 *
 * @note This is part of Phase 2 of the V2 Performance Optimization effort.
 *       See V2_PERFORMANCE_OPTIMIZATION_ANALYSIS.md for details.
 */

#ifndef HIGHNOON_NATIVE_OPS_UNIFIED_ATTENTION_OP_H_
#define HIGHNOON_NATIVE_OPS_UNIFIED_ATTENTION_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

// Include shared SIMD utilities from Phase 1
#include "hnn_simd_common.h"
#include "common/tensor_stream_pool.h"  // Phase 4: Zero-copy KV buffer streaming

// SIMD intrinsics for cross-architecture vectorization
#if defined(__AVX512F__)
#include <immintrin.h>
#define UA_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define UA_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define UA_NEON 1
#endif

namespace hsmn {
namespace attention {

// =============================================================================
// ATTENTION MODE ENUMERATION
// =============================================================================

/**
 * @brief Attention mechanism types supported by the unified attention system.
 *
 * Each mode corresponds to a previously separate attention implementation.
 * The mode selector allows runtime or compile-time dispatch to the appropriate
 * kernel while sharing common SIMD infrastructure.
 */
enum class AttentionMode : int {
    // Standard Attention Modes
    FLASH = 0,            ///< Full O(n²) flash attention with tiling
    LINEAR = 1,           ///< O(n) linear attention via feature maps
    LOCAL_WINDOWED = 2,   ///< O(n×w) local windowed attention
    
    // Enhanced Attention Modes
    DIFFERENTIAL = 3,     ///< Differential Transformer: softmax(Q₁K₁ᵀ) - λ·softmax(Q₂K₂ᵀ)
    SPARSE_NSA = 4,       ///< Native Sparse Attention with O(n log n)
    
    // Grouped-Query Attention Variants
    GQA = 5,              ///< Standard Grouped-Query Attention
    LINEAR_GQA = 6,       ///< Linear attention + GQA
    SLIDING_GQA = 7,      ///< Sliding window + GQA
    
    // Advanced Attention Modes
    LATENT_KV = 8,        ///< Latent KV compression (DeepSeek-style)
    QASA = 9,             ///< Quantum Adaptive Self-Attention
    LMWT = 10,            ///< Learnable Multi-Scale Wavelet Transformer
    
    // Count for validation
    NUM_MODES = 11
};

/**
 * @brief Convert AttentionMode to human-readable string.
 */
inline const char* AttentionModeToString(AttentionMode mode) {
    switch (mode) {
        case AttentionMode::FLASH:           return "FLASH";
        case AttentionMode::LINEAR:          return "LINEAR";
        case AttentionMode::LOCAL_WINDOWED:  return "LOCAL_WINDOWED";
        case AttentionMode::DIFFERENTIAL:    return "DIFFERENTIAL";
        case AttentionMode::SPARSE_NSA:      return "SPARSE_NSA";
        case AttentionMode::GQA:             return "GQA";
        case AttentionMode::LINEAR_GQA:      return "LINEAR_GQA";
        case AttentionMode::SLIDING_GQA:     return "SLIDING_GQA";
        case AttentionMode::LATENT_KV:       return "LATENT_KV";
        case AttentionMode::QASA:            return "QASA";
        case AttentionMode::LMWT:            return "LMWT";
        default:                             return "UNKNOWN";
    }
}

// =============================================================================
// UNIFIED ATTENTION CONFIGURATION
// =============================================================================

/**
 * @brief Unified configuration for all attention modes.
 *
 * This struct contains all parameters needed for any attention mode.
 * Unused parameters for a given mode are simply ignored, allowing a
 * single configuration type to drive all attention variants.
 */
struct UnifiedAttentionConfig {
    // === Core Parameters (all modes) ===
    AttentionMode mode = AttentionMode::FLASH;
    int batch_size = 1;
    int num_heads = 8;
    int head_dim = 64;
    int seq_len = 512;           ///< Query sequence length
    int kv_seq_len = 512;        ///< Key/Value sequence length (can differ from seq_len)
    float scale = 0.0f;          ///< Attention scale (0 = auto: 1/sqrt(head_dim))
    float dropout_rate = 0.0f;   ///< Dropout probability (0 = no dropout)
    float epsilon = 1e-6f;       ///< Numerical stability epsilon
    bool causal = true;          ///< Apply causal masking
    
    // === Grouped-Query Attention (GQA mode) ===
    int num_kv_heads = 1;        ///< Number of KV heads (< num_heads enables GQA)
    
    // === Local/Windowed Attention (LOCAL_WINDOWED, SLIDING_GQA) ===
    int window_size = 256;       ///< Local attention window size
    bool use_multiscale = false; ///< Enable per-head variable windows
    
    // === Differential Attention (DIFFERENTIAL mode) ===
    float lambda_init = 0.8f;    ///< Initial lambda for differential: A₁ - λ·A₂
    float lambda_min = 0.0f;     ///< Minimum lambda clamp
    float lambda_max = 2.0f;     ///< Maximum lambda clamp
    bool normalize_diff = true;  ///< Apply (1-λ_init) normalization
    
    // === Native Sparse Attention (SPARSE_NSA mode) ===
    int block_size = 64;         ///< Token compression block size
    int num_selected_blocks = 8; ///< Number of blocks to select
    int tokens_per_block = 8;    ///< Fine-grained tokens per selected block
    bool use_global_tokens = true;
    int num_global_tokens = 1;   ///< CLS/special tokens for global attention
    float temperature = 1.0f;    ///< Attention temperature
    
    // === Linear Attention (LINEAR, LINEAR_GQA modes) ===
    bool use_elu_feature_map = true;  ///< Use ELU+1 feature map (ignored if use_holographic_features=true)
    bool use_gla_gates = false;       ///< Enable Gated Linear Attention
    int rff_dim = 64;                 ///< Random Fourier Features dimension
    
    // === Holographic Linear Attention (extends LINEAR mode for HD embeddings) ===
    // When enabled, uses FFT-based feature maps for O(n) linear attention that
    // meshes with hyperdimensional embeddings. Replaces ELU+1 with:
    //   φ(x) = [Re(FFT(x)), Im(FFT(x))]  (complex FFT features)
    // Complexity: O(n × d log d) for FFT, O(n × d) for attention = O(n × d log d) total
    bool use_holographic_features = false;  ///< Use FFT feature maps for HD embedding mesh
    
    // === Quantum Attention (QASA mode) ===
    int num_qubits = 4;               ///< VQC qubit count
    int vqc_layers = 2;               ///< VQC circuit depth
    float entanglement_strength = 0.5f;
    bool use_residual_projection = true;
    int residual_proj_dim = 32;
    
    // === Latent KV (LATENT_KV mode) ===
    int latent_dim = 256;         ///< Latent compression dimension
    int num_latents = 64;         ///< Number of latent vectors
    bool use_cross_attention = true;
    
    // === Wavelet Attention (LMWT mode) ===
    int num_wavelet_scales = 4;       ///< Decomposition levels
    bool learn_wavelet_filters = true;
    float alpha_init = 0.7071067811865476f;  ///< 1/√2
    float beta_init = 0.7071067811865476f;   ///< 1/√2
    
    /**
     * @brief Compute the attention scaling factor.
     * @return 1/sqrt(head_dim) if scale==0, otherwise the configured scale.
     */
    float ComputeScale() const {
        if (scale == 0.0f) {
            return 1.0f / std::sqrt(static_cast<float>(head_dim));
        }
        return scale;
    }
    
    /**
     * @brief Compute the number of queries per KV head for GQA.
     * @return num_heads / num_kv_heads
     */
    int QueriesPerKVHead() const {
        return (num_kv_heads > 0) ? (num_heads / num_kv_heads) : num_heads;
    }
    
    /**
     * @brief Validate configuration parameters.
     * @return true if configuration is valid, false otherwise.
     */
    bool Validate() const {
        if (static_cast<int>(mode) < 0 || static_cast<int>(mode) >= static_cast<int>(AttentionMode::NUM_MODES)) {
            return false;
        }
        if (batch_size < 1 || num_heads < 1 || head_dim < 1) {
            return false;
        }
        if (seq_len < 1 || kv_seq_len < 1) {
            return false;
        }
        if (num_kv_heads > num_heads || (num_heads % num_kv_heads) != 0) {
            return false;
        }
        return true;
    }
};

// =============================================================================
// SHARED SIMD UTILITIES FOR ATTENTION
// =============================================================================

namespace simd {

/**
 * @brief SIMD-optimized dot product between two vectors.
 *
 * @param a First vector
 * @param b Second vector
 * @param size Vector length
 * @return Dot product result
 */
inline float dot_product(const float* a, const float* b, int64_t size) {
    float result = 0.0f;
    int64_t i = 0;
    
#if defined(UA_AVX512)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    result = _mm512_reduce_add_ps(acc);
#elif defined(UA_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    result = _mm_cvtss_f32(sum128);
#elif defined(UA_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, va, vb);
    }
    result = vaddvq_f32(acc);
#endif
    // Scalar tail
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief SIMD-optimized scaled dot product: scale * (a · b)
 */
inline float scaled_dot_product(const float* a, const float* b, float scale, int64_t size) {
    return scale * dot_product(a, b, size);
}

/**
 * @brief SIMD-optimized vector addition: out = a + b
 */
inline void add(const float* a, const float* b, float* out, int64_t size) {
    int64_t i = 0;
    
#if defined(UA_AVX512)
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_add_ps(va, vb));
    }
#elif defined(UA_AVX2)
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_add_ps(va, vb));
    }
#elif defined(UA_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vaddq_f32(va, vb));
    }
#endif
    for (; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief SIMD-optimized scaled addition: out = a + scale * b
 */
inline void add_scaled(const float* a, const float* b, float scale, float* out, int64_t size) {
    int64_t i = 0;
    
#if defined(UA_AVX512)
    __m512 vscale = _mm512_set1_ps(scale);
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_fmadd_ps(vb, vscale, va));
    }
#elif defined(UA_AVX2)
    __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_fmadd_ps(vb, vscale, va));
    }
#elif defined(UA_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vmlaq_f32(va, vb, vscale));
    }
#endif
    for (; i < size; ++i) {
        out[i] = a[i] + scale * b[i];
    }
}

/**
 * @brief SIMD-optimized vector scaling: out = scale * a
 */
inline void scale(const float* a, float scale_val, float* out, int64_t size) {
    int64_t i = 0;
    
#if defined(UA_AVX512)
    __m512 vscale = _mm512_set1_ps(scale_val);
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        _mm512_storeu_ps(&out[i], _mm512_mul_ps(va, vscale));
    }
#elif defined(UA_AVX2)
    __m256 vscale = _mm256_set1_ps(scale_val);
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(va, vscale));
    }
#elif defined(UA_NEON)
    float32x4_t vscale = vdupq_n_f32(scale_val);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        vst1q_f32(&out[i], vmulq_f32(va, vscale));
    }
#endif
    for (; i < size; ++i) {
        out[i] = scale_val * a[i];
    }
}

/**
 * @brief SIMD-optimized max reduction.
 */
inline float reduce_max(const float* data, int64_t size) {
    float max_val = -std::numeric_limits<float>::infinity();
    int64_t i = 0;
    
#if defined(UA_AVX512)
    if (size >= 16) {
        __m512 max_vec = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        for (; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(&data[i]);
            max_vec = _mm512_max_ps(max_vec, v);
        }
        max_val = _mm512_reduce_max_ps(max_vec);
    }
#elif defined(UA_AVX2)
    if (size >= 8) {
        __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);
            max_vec = _mm256_max_ps(max_vec, v);
        }
        __m128 lo = _mm256_castps256_ps128(max_vec);
        __m128 hi = _mm256_extractf128_ps(max_vec, 1);
        __m128 max128 = _mm_max_ps(lo, hi);
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
        max_val = _mm_cvtss_f32(max128);
    }
#elif defined(UA_NEON)
    if (size >= 4) {
        float32x4_t max_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        for (; i + 4 <= size; i += 4) {
            float32x4_t v = vld1q_f32(&data[i]);
            max_vec = vmaxq_f32(max_vec, v);
        }
        max_val = vmaxvq_f32(max_vec);
    }
#endif
    for (; i < size; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    return max_val;
}

/**
 * @brief SIMD-optimized sum reduction.
 */
inline float reduce_sum(const float* data, int64_t size) {
    float sum = 0.0f;
    int64_t i = 0;
    
#if defined(UA_AVX512)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        acc = _mm512_add_ps(acc, v);
    }
    sum = _mm512_reduce_add_ps(acc);
#elif defined(UA_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        acc = _mm256_add_ps(acc, v);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum = _mm_cvtss_f32(sum128);
#elif defined(UA_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        acc = vaddq_f32(acc, v);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

/**
 * @brief SIMD-optimized numerically stable softmax (in-place).
 *
 * @param data Input/output array
 * @param size Number of elements
 * @param eps Epsilon for numerical stability
 */
inline void softmax_inplace(float* data, int64_t size, float eps = 1e-6f) {
    // Step 1: Find max for numerical stability
    float max_val = reduce_max(data, size);
    
    // Step 2: Subtract max and compute exp
    int64_t i = 0;
#if defined(UA_AVX512)
    __m512 max_vec = _mm512_set1_ps(max_val);
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        _mm512_storeu_ps(&data[i], _mm512_sub_ps(v, max_vec));
    }
#elif defined(UA_AVX2)
    __m256 max_vec = _mm256_set1_ps(max_val);
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_sub_ps(v, max_vec));
    }
#elif defined(UA_NEON)
    float32x4_t max_vec = vdupq_n_f32(max_val);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vst1q_f32(&data[i], vsubq_f32(v, max_vec));
    }
#endif
    for (; i < size; ++i) {
        data[i] -= max_val;
    }
    
    // Step 3: Apply exp (use highnoon::ops::simd_exp_inplace from hnn_simd_common.h)
    highnoon::ops::simd_exp_inplace(data, size);
    
    // Step 4: Normalize by sum
    float sum = reduce_sum(data, size);
    float inv_sum = (sum > eps) ? (1.0f / sum) : 0.0f;
    
    i = 0;
#if defined(UA_AVX512)
    __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        _mm512_storeu_ps(&data[i], _mm512_mul_ps(v, inv_sum_vec));
    }
#elif defined(UA_AVX2)
    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, inv_sum_vec));
    }
#elif defined(UA_NEON)
    float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vst1q_f32(&data[i], vmulq_f32(v, inv_sum_vec));
    }
#endif
    for (; i < size; ++i) {
        data[i] *= inv_sum;
    }
}

/**
 * @brief Apply causal mask to attention scores (set future positions to -inf).
 *
 * @param scores Attention scores [seq_q, seq_k]
 * @param seq_q Query sequence length
 * @param seq_k Key sequence length
 * @param offset Offset for autoregressive generation (q_pos + offset)
 */
inline void apply_causal_mask(float* scores, int seq_q, int seq_k, int offset = 0) {
    constexpr float NEG_INF = -1e9f;  // Use finite value to avoid NaN in softmax
    
    for (int q = 0; q < seq_q; ++q) {
        int max_k = q + offset + 1;  // Can attend up to position q (inclusive)
        for (int k = max_k; k < seq_k; ++k) {
            scores[q * seq_k + k] = NEG_INF;
        }
    }
}

/**
 * @brief Apply local window mask (only attend within window_size).
 *
 * @param scores Attention scores [seq_q, seq_k]
 * @param seq_q Query sequence length
 * @param seq_k Key sequence length
 * @param window_size Local attention window size
 * @param causal Also apply causal masking
 */
inline void apply_local_window_mask(float* scores, int seq_q, int seq_k, 
                                     int window_size, bool causal) {
    constexpr float NEG_INF = -1e9f;
    
    for (int q = 0; q < seq_q; ++q) {
        int window_start = std::max(0, q - window_size / 2);
        int window_end = std::min(seq_k, q + window_size / 2 + 1);
        
        if (causal) {
            window_end = std::min(window_end, q + 1);
        }
        
        for (int k = 0; k < window_start; ++k) {
            scores[q * seq_k + k] = NEG_INF;
        }
        for (int k = window_end; k < seq_k; ++k) {
            scores[q * seq_k + k] = NEG_INF;
        }
    }
}

/**
 * @brief ELU+1 feature map for linear attention: φ(x) = elu(x) + 1
 *
 * @param data Input/output array
 * @param size Number of elements
 */
inline void elu_plus_one_inplace(float* data, int64_t size) {
    int64_t i = 0;
    
#if defined(UA_AVX2)
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 zero = _mm256_setzero_ps();
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        
        // ELU: x if x > 0, else exp(x) - 1
        __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GT_OS);
        
        // For x <= 0: exp(x) - 1 + 1 = exp(x)
        // We'll compute this in scalar for numerical stability
        _mm256_storeu_ps(&data[i], x);  // Store temporarily
    }
#endif
    
    // Scalar loop (handles all or remainder)
    for (; i < size; ++i) {
        float x = data[i];
        if (x > 0.0f) {
            data[i] = x + 1.0f;
        } else {
            data[i] = std::exp(x);  // exp(x) - 1 + 1 = exp(x)
        }
    }
}

/**
 * @brief Weighted sum: output[d] = sum_j(weights[j] * values[j,d])
 *
 * @param weights Attention weights [len]
 * @param values Value vectors [len, dim]
 * @param output Output vector [dim]
 * @param len Number of values
 * @param dim Value dimension
 */
inline void weighted_sum(const float* weights, const float* values,
                         float* output, int len, int dim) {
    // Initialize output to zero
    std::fill(output, output + dim, 0.0f);
    
    for (int j = 0; j < len; ++j) {
        float w = weights[j];
        const float* v = values + j * dim;
        
        int d = 0;
#if defined(UA_AVX512)
        __m512 vw = _mm512_set1_ps(w);
        for (; d + 16 <= dim; d += 16) {
            __m512 vo = _mm512_loadu_ps(&output[d]);
            __m512 vv = _mm512_loadu_ps(&v[d]);
            _mm512_storeu_ps(&output[d], _mm512_fmadd_ps(vw, vv, vo));
        }
#elif defined(UA_AVX2)
        __m256 vw = _mm256_set1_ps(w);
        for (; d + 8 <= dim; d += 8) {
            __m256 vo = _mm256_loadu_ps(&output[d]);
            __m256 vv = _mm256_loadu_ps(&v[d]);
            _mm256_storeu_ps(&output[d], _mm256_fmadd_ps(vw, vv, vo));
        }
#elif defined(UA_NEON)
        float32x4_t vw = vdupq_n_f32(w);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t vo = vld1q_f32(&output[d]);
            float32x4_t vv = vld1q_f32(&v[d]);
            vst1q_f32(&output[d], vmlaq_f32(vo, vw, vv));
        }
#endif
        for (; d < dim; ++d) {
            output[d] += w * v[d];
        }
    }
}

}  // namespace simd

// =============================================================================
// FORWARD DECLARATIONS FOR ATTENTION KERNELS
// =============================================================================

// Forward declarations - implementations in unified_attention_op.cc
void FlashAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output,
    float* workspace = nullptr);

void LinearAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output,
    float* kv_state = nullptr,
    float* k_sum = nullptr);

/**
 * @brief Apply holographic FFT feature map to input tensor.
 *
 * Computes FFT-based feature representation for linear attention:
 *   φ(x) = [Re(FFT(x)), Im(FFT(x))]  (concatenated real and imaginary parts)
 *
 * This meshes with hyperdimensional embeddings which use circular convolution
 * (FFT-based) for binding. Complexity: O(n × d log d).
 *
 * @param input Input tensor [batch, heads, seq, head_dim]
 * @param features_re Real part of FFT features [batch, heads, seq, head_dim]
 * @param features_im Imaginary part of FFT features [batch, heads, seq, head_dim]
 * @param batch Batch size
 * @param heads Number of heads
 * @param seq Sequence length
 * @param head_dim Head dimension (should be power of 2 for efficient FFT)
 */
void HolographicFeatureMap(
    const float* input,
    float* features_re,
    float* features_im,
    int batch, int heads, int seq, int head_dim);

/**
 * @brief O(n) Linear Attention with Holographic FFT Features.
 *
 * Uses FFT-based feature maps instead of ELU+1 for linear attention that
 * meshes with hyperdimensional embeddings. The kernel trick formulation:
 *
 *   KV_state = Σⱼ [φ_re(K[j]) + i·φ_im(K[j])] ⊗ V[j]   O(n × d)
 *   output[i] = Re([φ_re(Q[i]) + i·φ_im(Q[i])] · KV_state) / norm   O(n × d)
 *
 * Total complexity: O(n × d log d) for FFT + O(n × d) for attention.
 *
 * @param Q Query tensor [batch, heads, seq, head_dim]
 * @param K Key tensor [batch, heads, seq, head_dim]
 * @param V Value tensor [batch, heads, seq, head_dim]
 * @param config Unified attention configuration (use_holographic_features must be true)
 * @param output Output tensor [batch, heads, seq, head_dim]
 */
void HolographicLinearAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output);

void LocalWindowedAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output);

void DifferentialAttentionForward(
    const float* Q, const float* K, const float* V,
    float lambda,
    const UnifiedAttentionConfig& config,
    float* output);

void SparseNSAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output);

void GQAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output);

void LinearGQAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output);

void SlidingGQAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output);

void LatentKVForward(
    const float* Q, const float* K, const float* V,
    const float* latent_keys, const float* latent_values,
    const UnifiedAttentionConfig& config,
    float* output);

void QASAForward(
    const float* Q, const float* K, const float* V,
    const float* vqc_params,
    const UnifiedAttentionConfig& config,
    float* output);

void LMWTForward(
    const float* Q, const float* K, const float* V,
    const float* alpha, const float* beta,
    const UnifiedAttentionConfig& config,
    float* output);

// =============================================================================
// UNIFIED ATTENTION DISPATCHER
// =============================================================================

/**
 * @brief Unified attention forward pass dispatcher.
 *
 * Routes to the appropriate kernel based on config.mode.
 * All inputs and outputs are in [batch, heads, seq, head_dim] layout.
 *
 * @param Q Query tensor [batch, num_heads, seq_len, head_dim]
 * @param K Key tensor [batch, num_kv_heads, kv_seq_len, head_dim]
 * @param V Value tensor [batch, num_kv_heads, kv_seq_len, head_dim]
 * @param config Unified attention configuration
 * @param output Output tensor [batch, num_heads, seq_len, head_dim]
 * @param extra_inputs Optional mode-specific inputs (VQC params, latents, etc.)
 * @param workspace Optional scratch space for intermediate computations
 */
inline void UnifiedAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output,
    const float* extra_inputs = nullptr,
    float* workspace = nullptr) {
    
    switch (config.mode) {
        case AttentionMode::FLASH:
            FlashAttentionForward(Q, K, V, config, output, workspace);
            break;
            
        case AttentionMode::LINEAR:
            // Use holographic FFT features when enabled (HD embedding mesh)
            if (config.use_holographic_features) {
                HolographicLinearAttentionForward(Q, K, V, config, output);
            } else {
                LinearAttentionForward(Q, K, V, config, output);
            }
            break;
            
        case AttentionMode::LOCAL_WINDOWED:
            LocalWindowedAttentionForward(Q, K, V, config, output);
            break;
            
        case AttentionMode::DIFFERENTIAL:
            DifferentialAttentionForward(Q, K, V, config.lambda_init, config, output);
            break;
            
        case AttentionMode::SPARSE_NSA:
            SparseNSAForward(Q, K, V, config, output);
            break;
            
        case AttentionMode::GQA:
            GQAForward(Q, K, V, config, output);
            break;
            
        case AttentionMode::LINEAR_GQA:
            LinearGQAForward(Q, K, V, config, output);
            break;
            
        case AttentionMode::SLIDING_GQA:
            SlidingGQAForward(Q, K, V, config, output);
            break;
            
        case AttentionMode::LATENT_KV:
            // Latent KV requires extra_inputs to contain [latent_keys, latent_values]
            // Layout: extra_inputs[0:num_latents*head_dim] = latent_keys
            //         extra_inputs[num_latents*head_dim:] = latent_values
            if (extra_inputs) {
                const float* latent_keys = extra_inputs;
                const float* latent_values = extra_inputs + config.num_latents * config.head_dim;
                LatentKVForward(Q, K, V, latent_keys, latent_values, config, output);
            }
            break;
            
        case AttentionMode::QASA:
            // QASA requires vqc_params in extra_inputs
            if (extra_inputs) {
                QASAForward(Q, K, V, extra_inputs, config, output);
            }
            break;
            
        case AttentionMode::LMWT:
            // LMWT requires [alpha, beta] wavelet parameters in extra_inputs
            if (extra_inputs) {
                const float* alpha = extra_inputs;
                const float* beta = extra_inputs + config.num_wavelet_scales;
                LMWTForward(Q, K, V, alpha, beta, config, output);
            }
            break;
            
        default:
            // Fallback to flash attention for unknown modes
            FlashAttentionForward(Q, K, V, config, output, workspace);
            break;
    }
}

// =============================================================================
// PHASE 4: STREAMING UNIFIED ATTENTION (TensorStreamPool Integration)
// =============================================================================
// Zero-copy streaming variant with KV buffer pool management.
// Enables direct buffer handoff between QKV projection and attention kernels.

/**
 * @brief Streaming unified attention with TensorStreamPool KV buffer management.
 *
 * Manages KV projection buffers via pool for zero-copy inter-layer streaming.
 * When use_streaming=true, acquires KV buffers from pool and hands off to next layer.
 *
 * @param Q Query tensor [batch, num_heads, seq_len, head_dim]
 * @param K Key tensor (or nullptr to acquire from pool)
 * @param V Value tensor (or nullptr to acquire from pool)
 * @param config Attention configuration
 * @param output Output tensor [batch, num_heads, seq_len, head_dim]
 * @param pool_K If K is nullptr, returns pool-acquired K buffer
 * @param pool_V If V is nullptr, returns pool-acquired V buffer
 * @param use_streaming Enable TensorStreamPool (default: true)
 * @param consumer_hint Next kernel hint for handoff telemetry
 */
inline void UnifiedAttentionForwardStreaming(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output,
    float** pool_K = nullptr,
    float** pool_V = nullptr,
    const float* extra_inputs = nullptr,
    float* workspace = nullptr,
    bool use_streaming = true,
    const char* consumer_hint = "NextAttentionLayer"
) {
    using namespace hsmn::ops;
    
    bool k_from_pool = (K == nullptr && pool_K != nullptr);
    bool v_from_pool = (V == nullptr && pool_V != nullptr);
    
    size_t kv_size = static_cast<size_t>(config.batch_size) * 
                     config.num_kv_heads * config.kv_seq_len * config.head_dim * sizeof(float);
    
    float* k_buffer = const_cast<float*>(K);
    float* v_buffer = const_cast<float*>(V);
    
    if (use_streaming && k_from_pool) {
        k_buffer = GetTensorStreamPool().Acquire(kv_size, "attention_K");
        *pool_K = k_buffer;
    }
    
    if (use_streaming && v_from_pool) {
        v_buffer = GetTensorStreamPool().Acquire(kv_size, "attention_V");
        *pool_V = v_buffer;
    }
    
    // Dispatch to appropriate attention kernel
    UnifiedAttentionForward(Q, k_buffer, v_buffer, config, output, extra_inputs, workspace);
    
    // Handoff KV buffers to next layer if streaming
    if (use_streaming && k_from_pool) {
        GetTensorStreamPool().Handoff(k_buffer, consumer_hint);
    }
    if (use_streaming && v_from_pool) {
        GetTensorStreamPool().Handoff(v_buffer, consumer_hint);
    }
}

/**
 * @brief Release KV buffers acquired via UnifiedAttentionForwardStreaming.
 *
 * Call when KV buffers are no longer needed (e.g., at end of attention stack).
 *
 * @param pool_K Pool-acquired K buffer (or nullptr)
 * @param pool_V Pool-acquired V buffer (or nullptr)
 */
inline void ReleasePooledKVBuffers(float* pool_K, float* pool_V) {
    using namespace hsmn::ops;
    if (pool_K) {
        GetTensorStreamPool().Release(pool_K);
    }
    if (pool_V) {
        GetTensorStreamPool().Release(pool_V);
    }
}

}  // namespace attention
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_UNIFIED_ATTENTION_OP_H_
