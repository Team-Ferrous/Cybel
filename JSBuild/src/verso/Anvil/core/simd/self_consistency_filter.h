// highnoon/_native/ops/self_consistency_filter.h
// Copyright 2026 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Self-consistency filtering for ensemble QSG with coherence propagation.
// This implements Phase 3.2 of the QSG Enterprise Optimization Roadmap.
//
// Key insight: Instead of verifying against AR (which breaks parallelism),
// we check cross-position consistency within the ensemble output. Positions
// that "agree" with their neighbors get higher confidence; positions with
// low agreement get their logits softened for Jacobi refinement.
//
// This is a quantum-inspired approach where we treat each position as a
// measurement that should be consistent with its local context.

#ifndef HIGHNOON_NATIVE_OPS_SELF_CONSISTENCY_FILTER_H_
#define HIGHNOON_NATIVE_OPS_SELF_CONSISTENCY_FILTER_H_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#define CONSISTENCY_SIMD_WIDTH 16
#define CONSISTENCY_USE_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define CONSISTENCY_SIMD_WIDTH 8
#define CONSISTENCY_USE_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define CONSISTENCY_SIMD_WIDTH 4
#define CONSISTENCY_USE_NEON 1
#else
#define CONSISTENCY_SIMD_WIDTH 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace highnoon {
namespace ops {
namespace qsg {

/**
 * @brief Self-consistency filtering using cross-position coherence.
 *
 * After ensemble path combination, this filter checks for cross-position
 * consistency by examining whether neighboring positions "agree" on the
 * semantic direction of the generated tokens.
 *
 * Algorithm:
 * 1. For each position i, compute cosine similarity with neighbors j ∈ [i-w, i+w]
 * 2. Weight similarities by distance: closer neighbors matter more
 * 3. Aggregate into a consistency score ∈ [0, 1]
 * 4. Adjust logits: low consistency → soften (increase entropy)
 *
 * Complexity: O(N × W × V) where W is window size, V is vocab size
 *   - With W=3 and sparse early exit: effectively O(N × V)
 *
 * The softening operation:
 *   adjusted_logits[i] = logits[i] / (1 + strength × (1 - consistency[i]))
 *
 * This has the effect of making uncertain positions more open to correction
 * by the Jacobi refinement pass, while preserving high-confidence positions.
 *
 * @param ensemble_logits Combined logits from ensemble [seq_len, vocab_size]
 * @param consistency_scores Output consistency per position [seq_len]
 * @param adjusted_logits Output adjusted logits [seq_len, vocab_size]
 * @param seq_len Number of positions
 * @param vocab_size Vocabulary size
 * @param window_size Consistency window (typically 3-5)
 * @param softening_strength How much to soften inconsistent positions (0-1)
 */
inline void SelfConsistencyFilter(
    const float* ensemble_logits,
    float* consistency_scores,
    float* adjusted_logits,
    int seq_len,
    int vocab_size,
    int window_size = 3,
    float softening_strength = 0.5f) {
    
    // Pre-compute L2 norms for all positions to avoid redundant computation
    std::vector<float> norms(seq_len);
    
    #pragma omp parallel for
    for (int i = 0; i < seq_len; ++i) {
        const float* logits_i = ensemble_logits + i * vocab_size;
        float norm_sq = 0.0f;
        
        int v = 0;
#if defined(CONSISTENCY_USE_AVX512)
        __m512 norm_vec = _mm512_setzero_ps();
        for (; v + 16 <= vocab_size; v += 16) {
            __m512 l = _mm512_loadu_ps(logits_i + v);
            norm_vec = _mm512_fmadd_ps(l, l, norm_vec);
        }
        norm_sq = _mm512_reduce_add_ps(norm_vec);
#elif defined(CONSISTENCY_USE_AVX2)
        __m256 norm_vec = _mm256_setzero_ps();
        for (; v + 8 <= vocab_size; v += 8) {
            __m256 l = _mm256_loadu_ps(logits_i + v);
            norm_vec = _mm256_fmadd_ps(l, l, norm_vec);
        }
        __m128 hi = _mm256_extractf128_ps(norm_vec, 1);
        __m128 lo = _mm256_castps256_ps128(norm_vec);
        __m128 sum4 = _mm_add_ps(lo, hi);
        sum4 = _mm_hadd_ps(sum4, sum4);
        sum4 = _mm_hadd_ps(sum4, sum4);
        norm_sq = _mm_cvtss_f32(sum4);
#elif defined(CONSISTENCY_USE_NEON)
        float32x4_t norm_vec = vdupq_n_f32(0.0f);
        for (; v + 4 <= vocab_size; v += 4) {
            float32x4_t l = vld1q_f32(logits_i + v);
            norm_vec = vmlaq_f32(norm_vec, l, l);
        }
        norm_sq = vaddvq_f32(norm_vec);
#endif
        for (; v < vocab_size; ++v) {
            norm_sq += logits_i[v] * logits_i[v];
        }
        
        norms[i] = std::sqrt(norm_sq) + 1e-8f;
    }
    
    // Step 1: Compute cross-position consistency scores
    #pragma omp parallel for
    for (int i = 0; i < seq_len; ++i) {
        const float* logits_i = ensemble_logits + i * vocab_size;
        float norm_i = norms[i];
        
        float total_sim = 0.0f;
        float total_weight = 0.0f;
        
        for (int offset = -window_size; offset <= window_size; ++offset) {
            if (offset == 0) continue;  // Skip self
            int j = i + offset;
            if (j < 0 || j >= seq_len) continue;
            
            const float* logits_j = ensemble_logits + j * vocab_size;
            
            // Compute dot product
            float dot = 0.0f;
            int v = 0;
            
#if defined(CONSISTENCY_USE_AVX512)
            __m512 dot_vec = _mm512_setzero_ps();
            for (; v + 16 <= vocab_size; v += 16) {
                __m512 li = _mm512_loadu_ps(logits_i + v);
                __m512 lj = _mm512_loadu_ps(logits_j + v);
                dot_vec = _mm512_fmadd_ps(li, lj, dot_vec);
            }
            dot = _mm512_reduce_add_ps(dot_vec);
#elif defined(CONSISTENCY_USE_AVX2)
            __m256 dot_vec = _mm256_setzero_ps();
            for (; v + 8 <= vocab_size; v += 8) {
                __m256 li = _mm256_loadu_ps(logits_i + v);
                __m256 lj = _mm256_loadu_ps(logits_j + v);
                dot_vec = _mm256_fmadd_ps(li, lj, dot_vec);
            }
            __m128 dot_hi = _mm256_extractf128_ps(dot_vec, 1);
            __m128 dot_lo = _mm256_castps256_ps128(dot_vec);
            __m128 d4 = _mm_add_ps(dot_lo, dot_hi);
            d4 = _mm_hadd_ps(d4, d4);
            d4 = _mm_hadd_ps(d4, d4);
            dot = _mm_cvtss_f32(d4);
#elif defined(CONSISTENCY_USE_NEON)
            float32x4_t dot_vec = vdupq_n_f32(0.0f);
            for (; v + 4 <= vocab_size; v += 4) {
                float32x4_t li = vld1q_f32(logits_i + v);
                float32x4_t lj = vld1q_f32(logits_j + v);
                dot_vec = vmlaq_f32(dot_vec, li, lj);
            }
            dot = vaddvq_f32(dot_vec);
#endif
            for (; v < vocab_size; ++v) {
                dot += logits_i[v] * logits_j[v];
            }
            
            float norm_j = norms[j];
            float cos_sim = dot / (norm_i * norm_j);
            
            // Weight by distance: closer neighbors are more important
            // Weight = 1 / (1 + |offset|) normalized
            float weight = 1.0f / (1.0f + std::abs(offset));
            
            // Transform cosine similarity from [-1, 1] to [0, 1]
            float sim_01 = (cos_sim + 1.0f) * 0.5f;
            
            total_sim += weight * sim_01;
            total_weight += weight;
        }
        
        // Normalize consistency score
        consistency_scores[i] = (total_weight > 0.0f) ? 
            total_sim / total_weight : 1.0f;
    }
    
    // Step 2: Adjust logits based on consistency
    #pragma omp parallel for
    for (int i = 0; i < seq_len; ++i) {
        const float* logits_i = ensemble_logits + i * vocab_size;
        float* adj_logits_i = adjusted_logits + i * vocab_size;
        
        float consistency = consistency_scores[i];
        
        // Softening factor increases entropy for low-consistency positions
        // softening_factor = 1 + strength * (1 - consistency)
        // Range: [1, 1 + strength] when consistency ∈ [0, 1]
        float softening_factor = 1.0f + softening_strength * (1.0f - consistency);
        
        // Apply as temperature scaling (divide by factor)
        int v = 0;
#if defined(CONSISTENCY_USE_AVX512)
        __m512 sf_vec = _mm512_set1_ps(softening_factor);
        for (; v + 16 <= vocab_size; v += 16) {
            __m512 l = _mm512_loadu_ps(logits_i + v);
            __m512 adj = _mm512_div_ps(l, sf_vec);
            _mm512_storeu_ps(adj_logits_i + v, adj);
        }
#elif defined(CONSISTENCY_USE_AVX2)
        __m256 sf_vec = _mm256_set1_ps(softening_factor);
        for (; v + 8 <= vocab_size; v += 8) {
            __m256 l = _mm256_loadu_ps(logits_i + v);
            __m256 adj = _mm256_div_ps(l, sf_vec);
            _mm256_storeu_ps(adj_logits_i + v, adj);
        }
#elif defined(CONSISTENCY_USE_NEON)
        float32x4_t sf_vec = vdupq_n_f32(softening_factor);
        for (; v + 4 <= vocab_size; v += 4) {
            float32x4_t l = vld1q_f32(logits_i + v);
            float32x4_t adj = vdivq_f32(l, sf_vec);
            vst1q_f32(adj_logits_i + v, adj);
        }
#endif
        for (; v < vocab_size; ++v) {
            adj_logits_i[v] = logits_i[v] / softening_factor;
        }
    }
}


/**
 * @brief Grover-enhanced quality boosting for ensemble outputs.
 *
 * Applies Grover-inspired amplitude amplification to boost high-quality
 * solutions (those with high consistency) while suppressing low-quality ones.
 *
 * The Grover diffusion operator reflects about the mean:
 *   logits' = 2 * mean(logits) - logits
 *
 * We modulate this by consistency:
 *   output = consistency * diffused + (1 - consistency) * original
 *
 * This has the effect of:
 * - High consistency: Strong amplification toward high-probability tokens
 * - Low consistency: Keep original distribution for Jacobi refinement
 *
 * Mathematical connection to quantum computing:
 * - The diffusion operator is the key component of Grover's algorithm
 * - It amplifies marked states (high-probability tokens)
 * - Multiple iterations further concentrate probability mass
 *
 * Complexity: O(seq_len × vocab_size × iterations)
 *
 * @param ensemble_logits Combined ensemble logits [seq_len, vocab_size]
 * @param consistency_scores Per-position consistency [seq_len]
 * @param output Boosted logits [seq_len, vocab_size]
 * @param seq_len Sequence length
 * @param vocab_size Vocabulary size
 * @param iterations Grover iterations (typically 2-3)
 */
inline void GroverQualityBoost(
    const float* ensemble_logits,
    const float* consistency_scores,
    float* output,
    int seq_len,
    int vocab_size,
    int iterations = 2) {
    
    // Copy input to output for in-place iteration
    std::memcpy(output, ensemble_logits, seq_len * vocab_size * sizeof(float));
    
    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel for
        for (int pos = 0; pos < seq_len; ++pos) {
            float* logits = output + pos * vocab_size;
            float consistency = consistency_scores[pos];
            
            // Skip very low consistency positions (let Jacobi handle them)
            if (consistency < 0.3f) continue;
            
            // Compute mean logit
            float sum = 0.0f;
            int v = 0;
            
#if defined(CONSISTENCY_USE_AVX512)
            __m512 sum_vec = _mm512_setzero_ps();
            for (; v + 16 <= vocab_size; v += 16) {
                sum_vec = _mm512_add_ps(sum_vec, _mm512_loadu_ps(logits + v));
            }
            sum = _mm512_reduce_add_ps(sum_vec);
#elif defined(CONSISTENCY_USE_AVX2)
            __m256 sum_vec = _mm256_setzero_ps();
            for (; v + 8 <= vocab_size; v += 8) {
                sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(logits + v));
            }
            __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
            __m128 lo = _mm256_castps256_ps128(sum_vec);
            __m128 s4 = _mm_add_ps(lo, hi);
            s4 = _mm_hadd_ps(s4, s4);
            s4 = _mm_hadd_ps(s4, s4);
            sum = _mm_cvtss_f32(s4);
#elif defined(CONSISTENCY_USE_NEON)
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            for (; v + 4 <= vocab_size; v += 4) {
                sum_vec = vaddq_f32(sum_vec, vld1q_f32(logits + v));
            }
            sum = vaddvq_f32(sum_vec);
#endif
            for (; v < vocab_size; ++v) {
                sum += logits[v];
            }
            
            float mean = sum / vocab_size;
            
            // Grover diffusion: reflect about mean, modulated by consistency
            // output[v] = consistency * (2*mean - logits[v]) + (1-consistency) * logits[v]
            //           = 2*consistency*mean + (1 - 2*consistency) * logits[v]
            float alpha = 2.0f * consistency * mean;
            float beta = 1.0f - 2.0f * consistency;
            
            v = 0;
#if defined(CONSISTENCY_USE_AVX512)
            __m512 alpha_vec = _mm512_set1_ps(alpha);
            __m512 beta_vec = _mm512_set1_ps(beta);
            for (; v + 16 <= vocab_size; v += 16) {
                __m512 l = _mm512_loadu_ps(logits + v);
                __m512 out = _mm512_fmadd_ps(beta_vec, l, alpha_vec);
                _mm512_storeu_ps(logits + v, out);
            }
#elif defined(CONSISTENCY_USE_AVX2)
            __m256 alpha_vec = _mm256_set1_ps(alpha);
            __m256 beta_vec = _mm256_set1_ps(beta);
            for (; v + 8 <= vocab_size; v += 8) {
                __m256 l = _mm256_loadu_ps(logits + v);
                __m256 out = _mm256_fmadd_ps(beta_vec, l, alpha_vec);
                _mm256_storeu_ps(logits + v, out);
            }
#elif defined(CONSISTENCY_USE_NEON)
            float32x4_t alpha_vec = vdupq_n_f32(alpha);
            float32x4_t beta_vec = vdupq_n_f32(beta);
            for (; v + 4 <= vocab_size; v += 4) {
                float32x4_t l = vld1q_f32(logits + v);
                float32x4_t out = vmlaq_f32(alpha_vec, beta_vec, l);
                vst1q_f32(logits + v, out);
            }
#endif
            for (; v < vocab_size; ++v) {
                logits[v] = alpha + beta * logits[v];
            }
        }
    }
}


/**
 * @brief Combined self-consistency filter with Grover boost.
 *
 * Convenience function that chains:
 * 1. SelfConsistencyFilter: Compute consistency and adjust logits
 * 2. GroverQualityBoost: Amplify high-consistency solutions
 *
 * @param input_logits Input ensemble logits [seq_len, vocab_size]
 * @param output_logits Output refined logits [seq_len, vocab_size]
 * @param consistency_scores Output consistency scores [seq_len]
 * @param seq_len Sequence length
 * @param vocab_size Vocabulary size
 * @param window_size Consistency window size
 * @param softening_strength Softening strength for low consistency
 * @param grover_iterations Number of Grover boost iterations
 */
inline void SelfConsistencyWithGroverBoost(
    const float* input_logits,
    float* output_logits,
    float* consistency_scores,
    int seq_len,
    int vocab_size,
    int window_size = 3,
    float softening_strength = 0.5f,
    int grover_iterations = 2) {
    
    // Allocate intermediate buffer for adjusted logits
    std::vector<float> adjusted(seq_len * vocab_size);
    
    // Step 1: Self-consistency filter
    SelfConsistencyFilter(
        input_logits,
        consistency_scores,
        adjusted.data(),
        seq_len,
        vocab_size,
        window_size,
        softening_strength
    );
    
    // Step 2: Grover quality boost
    GroverQualityBoost(
        adjusted.data(),
        consistency_scores,
        output_logits,
        seq_len,
        vocab_size,
        grover_iterations
    );
}

}  // namespace qsg
}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_SELF_CONSISTENCY_FILTER_H_
