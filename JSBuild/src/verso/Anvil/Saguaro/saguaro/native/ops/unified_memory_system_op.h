// saguaro.native/ops/unified_memory_system_op.h
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
 * @file unified_memory_system_op.h
 * @brief Unified Memory System Operations for HighNoon Framework
 *
 * This header consolidates all 5 memory mechanisms into a single unified
 * interface with shared primitives and SIMD optimization.
 *
 * Supported Memory Types:
 * - CONTENT_ADDRESSED:    Standard attention-based memory (cosine similarity)
 * - PRODUCT_KEY:          Sub-linear O(√M) lookup via product codebooks
 * - HOPFIELD:             Energy-based associative memory (exponential capacity)
 * - ADAPTIVE:             Learned gating with surprise-based writes
 * - HIERARCHICAL:         Multi-level memory with CTQW traversal
 *
 * Shared Primitives:
 * - memory_read: Cosine similarity with softmax attention
 * - memory_write: Gated update with decay
 * - top_k_select: Fast top-k for sparse retrieval
 * - surprise_gate: Titans-style novelty detection
 *
 * SIMD Support:
 * - AVX-512: 16-wide vectorization
 * - AVX2:    8-wide vectorization
 * - NEON:    4-wide vectorization
 * - Scalar fallback
 *
 * Thread Safety: All functions are reentrant with no shared state.
 *
 * @note Phase 4 of V2 Performance Optimization.
 *       See V2_PERFORMANCE_OPTIMIZATION_ANALYSIS.md for details.
 */

#ifndef SAGUARO_NATIVE_OPS_UNIFIED_MEMORY_SYSTEM_OP_H_
#define SAGUARO_NATIVE_OPS_UNIFIED_MEMORY_SYSTEM_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

// Include shared SIMD utilities from Phase 1
#include "hnn_simd_common.h"

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define MEM_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define MEM_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define MEM_NEON 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace saguaro {
namespace memory {

// =============================================================================
// MEMORY TYPE ENUMERATION
// =============================================================================

/**
 * @brief Memory types supported by the unified memory system.
 */
enum class MemoryType : int {
    CONTENT_ADDRESSED = 0,   ///< Standard attention-based memory
    PRODUCT_KEY = 1,         ///< Sub-linear O(√M) lookup
    HOPFIELD = 2,            ///< Energy-based associative memory
    ADAPTIVE = 3,            ///< Learned gating with surprise
    HIERARCHICAL = 4,        ///< Multi-level with CTQW
    
    NUM_TYPES = 5
};

/**
 * @brief Convert MemoryType to string.
 */
inline const char* MemoryTypeToString(MemoryType type) {
    switch (type) {
        case MemoryType::CONTENT_ADDRESSED: return "CONTENT_ADDRESSED";
        case MemoryType::PRODUCT_KEY:       return "PRODUCT_KEY";
        case MemoryType::HOPFIELD:          return "HOPFIELD";
        case MemoryType::ADAPTIVE:          return "ADAPTIVE";
        case MemoryType::HIERARCHICAL:      return "HIERARCHICAL";
        default:                            return "UNKNOWN";
    }
}

// =============================================================================
// UNIFIED MEMORY CONFIGURATION
// =============================================================================

/**
 * @brief Unified configuration for all memory operations.
 */
struct MemoryConfig {
    // === Core Parameters ===
    MemoryType mem_type = MemoryType::CONTENT_ADDRESSED;
    int batch_size = 1;
    int num_slots = 256;             ///< Number of memory slots
    int slot_dim = 512;              ///< Dimension per slot
    int query_dim = 512;             ///< Query dimension
    float epsilon = 1e-6f;
    
    // === Content-Addressed Parameters ===
    float temperature = 1.0f;        ///< Softmax temperature
    bool normalize_keys = true;      ///< L2 normalize keys
    
    // === Product-Key Parameters ===
    int codebook_size = 64;          ///< √M entries per codebook
    int subkey_dim = 256;            ///< Sub-key dimension
    int product_k = 8;               ///< Top-k per sub-codebook
    
    // === Hopfield Parameters ===
    float beta = 1.0f;               ///< Inverse temperature
    int num_iterations = 1;          ///< Hopfield update iterations
    
    // === Adaptive Parameters ===
    float surprise_threshold = 0.5f; ///< Surprise gating threshold
    float decay_rate = 0.99f;        ///< Memory decay rate
    float write_strength = 0.1f;     ///< Default write strength
    
    // === Hierarchical Parameters ===
    int num_levels = 3;              ///< Memory hierarchy levels
    float ctqw_gamma = 0.1f;         ///< CTQW hopping parameter
    int slots_per_level[8] = {64, 32, 16, 8, 4, 2, 1, 1}; ///< Slots per level
    
    /**
     * @brief Validate configuration.
     */
    bool Validate() const {
        if (static_cast<int>(mem_type) < 0 || 
            static_cast<int>(mem_type) >= static_cast<int>(MemoryType::NUM_TYPES)) {
            return false;
        }
        if (num_slots < 1 || slot_dim < 1) {
            return false;
        }
        return true;
    }
};

// =============================================================================
// SHARED MEMORY PRIMITIVES
// =============================================================================

namespace primitives {

/**
 * @brief SIMD-optimized dot product.
 */
inline float simd_dot(const float* a, const float* b, int64_t size) {
    float sum = 0.0f;
    int64_t i = 0;

#if defined(MEM_AVX512)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    sum = _mm512_reduce_add_ps(acc);
#elif defined(MEM_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum = _mm_cvtss_f32(sum128);
#elif defined(MEM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, va, vb);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * @brief Compute L2 norm of a vector.
 */
inline float l2_norm(const float* x, int64_t size) {
    return std::sqrt(simd_dot(x, x, size));
}

/**
 * @brief Compute cosine similarity between query and key.
 */
inline float cosine_similarity(const float* query, const float* key, int64_t dim, float epsilon = 1e-6f) {
    float dot = simd_dot(query, key, dim);
    float norm_q = l2_norm(query, dim);
    float norm_k = l2_norm(key, dim);
    return dot / (norm_q * norm_k + epsilon);
}

/**
 * @brief Compute cosine similarity between query and all keys.
 *
 * @param query Query vector [dim]
 * @param keys Key matrix [num_slots, dim]
 * @param similarity Output similarities [num_slots]
 * @param num_slots Number of memory slots
 * @param dim Dimension
 */
inline void batch_cosine_similarity(
    const float* query,
    const float* keys,
    float* similarity,
    int64_t num_slots,
    int64_t dim,
    float epsilon = 1e-6f) {
    
    float norm_q = l2_norm(query, dim);
    
    #pragma omp parallel for
    for (int64_t s = 0; s < num_slots; ++s) {
        const float* key = keys + s * dim;
        float dot = simd_dot(query, key, dim);
        float norm_k = l2_norm(key, dim);
        similarity[s] = dot / (norm_q * norm_k + epsilon);
    }
}

/**
 * @brief In-place softmax with temperature scaling.
 */
inline void softmax_inplace(float* scores, int64_t size, float temperature = 1.0f) {
    float inv_temp = 1.0f / temperature;
    
    // Find max for numerical stability
    float max_val = scores[0];
    for (int64_t i = 1; i < size; ++i) {
        max_val = std::max(max_val, scores[i]);
    }
    
    // Exp and sum
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        scores[i] = std::exp((scores[i] - max_val) * inv_temp);
        sum += scores[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-10f);
    int64_t i = 0;
#if defined(MEM_AVX2)
    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&scores[i]);
        _mm256_storeu_ps(&scores[i], _mm256_mul_ps(v, inv_sum_vec));
    }
#endif
    for (; i < size; ++i) {
        scores[i] *= inv_sum;
    }
}

/**
 * @brief Weighted read from memory using attention weights.
 *
 * @param weights Attention weights [num_slots]
 * @param memory Memory matrix [num_slots, slot_dim]
 * @param output Output vector [slot_dim]
 * @param num_slots Number of slots
 * @param slot_dim Slot dimension
 */
inline void weighted_read(
    const float* weights,
    const float* memory,
    float* output,
    int64_t num_slots,
    int64_t slot_dim) {
    
    std::fill(output, output + slot_dim, 0.0f);
    
    for (int64_t s = 0; s < num_slots; ++s) {
        float w = weights[s];
        if (w < 1e-8f) continue;  // Skip near-zero weights
        
        const float* slot = memory + s * slot_dim;
        int64_t d = 0;
        
#if defined(MEM_AVX2)
        __m256 w_vec = _mm256_set1_ps(w);
        for (; d + 8 <= slot_dim; d += 8) {
            __m256 out_v = _mm256_loadu_ps(&output[d]);
            __m256 slot_v = _mm256_loadu_ps(&slot[d]);
            out_v = _mm256_fmadd_ps(w_vec, slot_v, out_v);
            _mm256_storeu_ps(&output[d], out_v);
        }
#endif
        for (; d < slot_dim; ++d) {
            output[d] += w * slot[d];
        }
    }
}

/**
 * @brief Gated memory write.
 *
 * memory_new = gate * memory_old + (1 - gate) * value
 *
 * @param memory Memory matrix [num_slots, slot_dim]
 * @param slot_idx Slot index to write
 * @param value Value to write [slot_dim]
 * @param gate Write gate [0, 1]
 * @param slot_dim Slot dimension
 */
inline void gated_write(
    float* memory,
    int64_t slot_idx,
    const float* value,
    float gate,
    int64_t slot_dim) {
    
    float* slot = memory + slot_idx * slot_dim;
    float one_minus_gate = 1.0f - gate;
    
    int64_t d = 0;
#if defined(MEM_AVX2)
    __m256 gate_vec = _mm256_set1_ps(gate);
    __m256 omg_vec = _mm256_set1_ps(one_minus_gate);
    for (; d + 8 <= slot_dim; d += 8) {
        __m256 old_v = _mm256_loadu_ps(&slot[d]);
        __m256 new_v = _mm256_loadu_ps(&value[d]);
        __m256 result = _mm256_fmadd_ps(gate_vec, old_v,
                                        _mm256_mul_ps(omg_vec, new_v));
        _mm256_storeu_ps(&slot[d], result);
    }
#endif
    for (; d < slot_dim; ++d) {
        slot[d] = gate * slot[d] + one_minus_gate * value[d];
    }
}

/**
 * @brief Compute surprise signal for adaptive write gating.
 *
 * Surprise = ||x - predicted||² / dim
 * Gate = sigmoid(surprise - threshold)
 *
 * @param input Input vector [dim]
 * @param predicted Prediction from memory [dim]
 * @param dim Vector dimension
 * @param threshold Surprise threshold
 * @return Write gate value in [0, 1]
 */
inline float surprise_gate(
    const float* input,
    const float* predicted,
    int64_t dim,
    float threshold) {
    
    float mse = 0.0f;
    int64_t d = 0;
    
#if defined(MEM_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; d + 8 <= dim; d += 8) {
        __m256 x = _mm256_loadu_ps(&input[d]);
        __m256 p = _mm256_loadu_ps(&predicted[d]);
        __m256 diff = _mm256_sub_ps(x, p);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    for (int i = 0; i < 8; ++i) mse += tmp[i];
#endif
    for (; d < dim; ++d) {
        float diff = input[d] - predicted[d];
        mse += diff * diff;
    }
    
    mse /= dim;
    float surprise = mse - threshold;
    return 1.0f / (1.0f + std::exp(-surprise));
}

/**
 * @brief Fast top-k selection.
 */
inline void top_k_select(
    const float* scores,
    int32_t* indices,
    float* values,
    int64_t size,
    int64_t k) {
    
    // Create index-value pairs
    std::vector<std::pair<float, int32_t>> pairs(size);
    for (int64_t i = 0; i < size; ++i) {
        pairs[i] = {scores[i], static_cast<int32_t>(i)};
    }
    
    // Partial sort
    k = std::min(k, size);
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.first > b.first;
                      });
    
    // Extract results
    for (int64_t i = 0; i < k; ++i) {
        values[i] = pairs[i].first;
        indices[i] = pairs[i].second;
    }
}

}  // namespace primitives

// =============================================================================
// MEMORY KERNEL FORWARD DECLARATIONS
// =============================================================================

void ContentAddressedMemoryRead(
    const float* query, const float* keys, const float* values,
    float* output, float* attention_weights, const MemoryConfig& config);

void ContentAddressedMemoryWrite(
    float* keys, float* values, const float* key, const float* value,
    int slot_idx, float gate, const MemoryConfig& config);

void ProductKeyMemoryRead(
    const float* query, const float* codebook_a, const float* codebook_b,
    const float* memory, float* output, float* attention_weights,
    const MemoryConfig& config);

void HopfieldMemoryRead(
    const float* query, const float* patterns, float* output,
    const MemoryConfig& config);

void HopfieldMemoryEnergy(
    const float* state, const float* patterns, float* energy,
    const MemoryConfig& config);

void AdaptiveMemoryReadWrite(
    const float* input, float* memory, float* output, float* surprise,
    bool write_enabled, const MemoryConfig& config);

void HierarchicalMemoryRead(
    const float* query, float* const* level_memory, float* output,
    const MemoryConfig& config);

// =============================================================================
// UNIFIED MEMORY DISPATCHER
// =============================================================================

/**
 * @brief Unified memory read dispatcher.
 */
inline void UnifiedMemoryRead(
    const float* query,
    const float* memory,
    float* output,
    float* attention_weights,
    const MemoryConfig& config,
    const float* aux_data = nullptr) {
    
    switch (config.mem_type) {
        case MemoryType::CONTENT_ADDRESSED:
            ContentAddressedMemoryRead(query, memory, memory + config.num_slots * config.slot_dim,
                                       output, attention_weights, config);
            break;
            
        case MemoryType::HOPFIELD:
            HopfieldMemoryRead(query, memory, output, config);
            break;
            
        case MemoryType::PRODUCT_KEY:
            if (aux_data != nullptr) {
                // aux_data contains codebook_a, codebook_b
                int codebook_size = config.codebook_size * config.subkey_dim;
                ProductKeyMemoryRead(query, aux_data, aux_data + codebook_size,
                                     memory, output, attention_weights, config);
            }
            break;
            
        default:
            // Fall back to content-addressed
            ContentAddressedMemoryRead(query, memory, memory + config.num_slots * config.slot_dim,
                                       output, attention_weights, config);
            break;
    }
}

}  // namespace memory
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_UNIFIED_MEMORY_SYSTEM_OP_H_
