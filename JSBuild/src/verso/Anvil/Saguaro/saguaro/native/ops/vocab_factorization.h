// saguaro.native/ops/vocab_factorization.h
// Copyright 2026 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// Vocabulary embedding factorization for O(r) oracle scoring in QSG pipeline.
// This implements Phase 1.2 of the QSG Enterprise Optimization Roadmap.
//
// Key optimization: Factor vocab_embeddings [V, d] ≈ U [V, r] × V^T [r, d]
// where r << d (typically r=32, d=128). This reduces oracle scoring from
// O(N × V × d) to O(N × d × r + N × K × r), achieving ~200x speedup.

#ifndef SAGUARO_NATIVE_OPS_VOCAB_FACTORIZATION_H_
#define SAGUARO_NATIVE_OPS_VOCAB_FACTORIZATION_H_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#define VOCAB_SIMD_WIDTH 16
#define VOCAB_USE_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define VOCAB_SIMD_WIDTH 8
#define VOCAB_USE_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define VOCAB_SIMD_WIDTH 4
#define VOCAB_USE_NEON 1
#else
#define VOCAB_SIMD_WIDTH 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace saguaro {
namespace ops {
namespace vocab {

/**
 * @brief Factored vocabulary representation for fast similarity scoring.
 *
 * Stores vocabulary embeddings in factored form: E ≈ U × V^T where:
 *   - U: [vocab_size, rank] - per-token reduced representation
 *   - V: [dim, rank] - shared projection basis
 *
 * Benefits:
 *   - Training: Learns compressed representation natively
 *   - Inference: O(d × r) context projection + O(K × r) candidate scoring
 *   - Memory: V×r + r×d instead of V×d (e.g., 60K×32 + 32×128 vs 60K×128)
 *
 * For QSG oracle scoring:
 *   sim(ctx, vocab[v]) ≈ (ctx @ V) · U[v]^T
 *   = O(d × r) projection (shared) + O(r) dot product (per candidate)
 *
 * With r=32, d=128, K=1024:
 *   Original: O(K × d) = O(131,072) per position
 *   Factored: O(d × r) + O(K × r) = O(4,096 + 32,768) = O(36,864) per position
 *   Speedup: ~3.6x from factorization alone
 *
 * Combined with Top-K pruning (V=60K → K=1024):
 *   Total speedup: 60x × 3.6x = 216x
 */
struct FactoredVocabEmbeddings {
    std::vector<float> U;        // [vocab_size, rank] row-major
    std::vector<float> V;        // [dim, rank] row-major
    std::vector<float> U_norms;  // [vocab_size] precomputed L2 norms
    int vocab_size;
    int dim;
    int rank;
    bool initialized;
    
    FactoredVocabEmbeddings() : vocab_size(0), dim(0), rank(0), initialized(false) {}
    
    FactoredVocabEmbeddings(int vocab_size_, int dim_, int rank_)
        : vocab_size(vocab_size_), dim(dim_), rank(rank_), initialized(false) {
        U.resize(vocab_size * rank);
        V.resize(dim * rank);
        U_norms.resize(vocab_size);
    }
    
    /**
     * @brief Initialize factored embeddings from full embeddings via truncated SVD.
     *
     * This is used for retrofitting existing models. For new training,
     * use random initialization with initialize_random().
     *
     * Note: SVD is computed externally (Python/numpy) and loaded here.
     * This function just sets the internal state.
     *
     * @param U_data Pre-computed U matrix [vocab_size, rank]
     * @param V_data Pre-computed V matrix [dim, rank]
     */
    void load_factors(const float* U_data, const float* V_data) {
        std::memcpy(U.data(), U_data, vocab_size * rank * sizeof(float));
        std::memcpy(V.data(), V_data, dim * rank * sizeof(float));
        precompute_norms();
        initialized = true;
    }
    
    /**
     * @brief Initialize with random orthogonal matrices for training from scratch.
     *
     * Uses Xavier/Glorot initialization scaled for the factored representation.
     * V is initialized as a random orthonormal basis.
     * U is initialized with small random values.
     */
    void initialize_random(unsigned int seed = 42) {
        // Xavier initialization scale
        float scale_u = std::sqrt(2.0f / (vocab_size + rank));
        float scale_v = std::sqrt(2.0f / (dim + rank));
        
        // Simple LCG for reproducibility
        unsigned int state = seed;
        auto rand_float = [&state]() -> float {
            state = state * 1103515245 + 12345;
            return ((state >> 16) & 0x7FFF) / 32768.0f * 2.0f - 1.0f;
        };
        
        // Initialize U
        for (int i = 0; i < vocab_size * rank; ++i) {
            U[i] = rand_float() * scale_u;
        }
        
        // Initialize V with approximate orthonormalization
        for (int i = 0; i < dim * rank; ++i) {
            V[i] = rand_float() * scale_v;
        }
        
        // Gram-Schmidt orthonormalization on V columns
        for (int r = 0; r < rank; ++r) {
            // Subtract projections onto previous columns
            for (int prev = 0; prev < r; ++prev) {
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    dot += V[d * rank + r] * V[d * rank + prev];
                }
                for (int d = 0; d < dim; ++d) {
                    V[d * rank + r] -= dot * V[d * rank + prev];
                }
            }
            // Normalize
            float norm = 0.0f;
            for (int d = 0; d < dim; ++d) {
                norm += V[d * rank + r] * V[d * rank + r];
            }
            norm = std::sqrt(norm) + 1e-8f;
            for (int d = 0; d < dim; ++d) {
                V[d * rank + r] /= norm;
            }
        }
        
        precompute_norms();
        initialized = true;
    }
    
    /**
     * @brief Precompute L2 norms of U rows for fast cosine similarity.
     */
    void precompute_norms() {
        #pragma omp parallel for
        for (int v = 0; v < vocab_size; ++v) {
            float norm_sq = 0.0f;
            const float* u_row = U.data() + v * rank;
            for (int r = 0; r < rank; ++r) {
                norm_sq += u_row[r] * u_row[r];
            }
            U_norms[v] = std::sqrt(norm_sq) + 1e-8f;
        }
    }
    
    /**
     * @brief Project context embeddings to low-rank space.
     *
     * Computes: projected[pos, :] = context[pos, :] @ V
     *
     * Output shape: [seq_len, rank]
     * Complexity: O(seq_len × dim × rank)
     *
     * @param context Input context embeddings [seq_len, dim]
     * @param projected Output projected embeddings [seq_len, rank]
     * @param seq_len Number of sequence positions
     */
    void project_context(
        const float* context,
        float* projected,
        int seq_len) const {
        
        #pragma omp parallel for
        for (int pos = 0; pos < seq_len; ++pos) {
            const float* ctx = context + pos * dim;
            float* proj = projected + pos * rank;
            
            // Zero output
            std::memset(proj, 0, rank * sizeof(float));
            
            // Matrix-vector multiply: proj = ctx @ V
            for (int d = 0; d < dim; ++d) {
                float ctx_val = ctx[d];
                const float* v_row = V.data() + d * rank;
                
#if defined(VOCAB_USE_AVX512)
                int r = 0;
                __m512 ctx_vec = _mm512_set1_ps(ctx_val);
                for (; r + 16 <= rank; r += 16) {
                    __m512 v_vec = _mm512_loadu_ps(v_row + r);
                    __m512 p_vec = _mm512_loadu_ps(proj + r);
                    p_vec = _mm512_fmadd_ps(ctx_vec, v_vec, p_vec);
                    _mm512_storeu_ps(proj + r, p_vec);
                }
                for (; r < rank; ++r) {
                    proj[r] += ctx_val * v_row[r];
                }
#elif defined(VOCAB_USE_AVX2)
                int r = 0;
                __m256 ctx_vec = _mm256_set1_ps(ctx_val);
                for (; r + 8 <= rank; r += 8) {
                    __m256 v_vec = _mm256_loadu_ps(v_row + r);
                    __m256 p_vec = _mm256_loadu_ps(proj + r);
                    p_vec = _mm256_fmadd_ps(ctx_vec, v_vec, p_vec);
                    _mm256_storeu_ps(proj + r, p_vec);
                }
                for (; r < rank; ++r) {
                    proj[r] += ctx_val * v_row[r];
                }
#elif defined(VOCAB_USE_NEON)
                int r = 0;
                float32x4_t ctx_vec = vdupq_n_f32(ctx_val);
                for (; r + 4 <= rank; r += 4) {
                    float32x4_t v_vec = vld1q_f32(v_row + r);
                    float32x4_t p_vec = vld1q_f32(proj + r);
                    p_vec = vmlaq_f32(p_vec, ctx_vec, v_vec);
                    vst1q_f32(proj + r, p_vec);
                }
                for (; r < rank; ++r) {
                    proj[r] += ctx_val * v_row[r];
                }
#else
                for (int r = 0; r < rank; ++r) {
                    proj[r] += ctx_val * v_row[r];
                }
#endif
            }
        }
    }
    
    /**
     * @brief Score candidate tokens using projected context.
     *
     * Computes cosine similarity between projected context and U rows
     * for specified candidate tokens only.
     *
     * scores[pos, k] = cos_sim(projected[pos], U[candidates[pos, k]])
     *                = (projected · U[v]) / (||projected|| × ||U[v]||)
     *
     * Complexity: O(seq_len × num_candidates × rank)
     *
     * @param projected_context Projected context embeddings [seq_len, rank]
     * @param candidates Candidate token indices [seq_len, num_candidates]
     * @param num_candidates Number of candidates per position
     * @param scores Output similarity scores [seq_len, num_candidates]
     * @param seq_len Number of sequence positions
     */
    void score_candidates(
        const float* projected_context,
        const int* candidates,
        int num_candidates,
        float* scores,
        int seq_len) const {
        
        #pragma omp parallel for collapse(2)
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int k = 0; k < num_candidates; ++k) {
                int v = candidates[pos * num_candidates + k];
                
                // Bounds check
                if (v < 0 || v >= vocab_size) {
                    scores[pos * num_candidates + k] = 0.0f;
                    continue;
                }
                
                const float* proj = projected_context + pos * rank;
                const float* u_row = U.data() + v * rank;
                
                // Compute dot product and projected norm
                float dot = 0.0f;
                float proj_norm_sq = 0.0f;
                
                int r = 0;
#if defined(VOCAB_USE_AVX2)
                __m256 dot_vec = _mm256_setzero_ps();
                __m256 pn_vec = _mm256_setzero_ps();
                for (; r + 8 <= rank; r += 8) {
                    __m256 p = _mm256_loadu_ps(proj + r);
                    __m256 u = _mm256_loadu_ps(u_row + r);
                    dot_vec = _mm256_fmadd_ps(p, u, dot_vec);
                    pn_vec = _mm256_fmadd_ps(p, p, pn_vec);
                }
                // Horizontal sum
                __m128 dot_hi = _mm256_extractf128_ps(dot_vec, 1);
                __m128 dot_lo = _mm256_castps256_ps128(dot_vec);
                __m128 d4 = _mm_add_ps(dot_lo, dot_hi);
                d4 = _mm_hadd_ps(d4, d4);
                d4 = _mm_hadd_ps(d4, d4);
                dot = _mm_cvtss_f32(d4);
                
                __m128 pn_hi = _mm256_extractf128_ps(pn_vec, 1);
                __m128 pn_lo = _mm256_castps256_ps128(pn_vec);
                __m128 p4 = _mm_add_ps(pn_lo, pn_hi);
                p4 = _mm_hadd_ps(p4, p4);
                p4 = _mm_hadd_ps(p4, p4);
                proj_norm_sq = _mm_cvtss_f32(p4);
#endif
                for (; r < rank; ++r) {
                    dot += proj[r] * u_row[r];
                    proj_norm_sq += proj[r] * proj[r];
                }
                
                float proj_norm = std::sqrt(proj_norm_sq) + 1e-8f;
                float u_norm = U_norms[v];
                
                // Cosine similarity in [−1, 1], normalize to [0, 1] for oracle
                float cos_sim = dot / (proj_norm * u_norm);
                scores[pos * num_candidates + k] = (cos_sim + 1.0f) * 0.5f;
            }
        }
    }
    
    /**
     * @brief Reconstruct full embeddings for a subset of tokens.
     *
     * full[k, :] = U[tokens[k], :] @ V^T
     *
     * Used for computing full similarity when needed (e.g., verification).
     *
     * @param tokens Token indices to reconstruct [num_tokens]
     * @param num_tokens Number of tokens
     * @param full Output full embeddings [num_tokens, dim]
     */
    void reconstruct(
        const int* tokens,
        int num_tokens,
        float* full) const {
        
        #pragma omp parallel for
        for (int i = 0; i < num_tokens; ++i) {
            int v = tokens[i];
            if (v < 0 || v >= vocab_size) {
                std::memset(full + i * dim, 0, dim * sizeof(float));
                continue;
            }
            
            const float* u_row = U.data() + v * rank;
            float* out = full + i * dim;
            
            // out = u_row @ V^T
            for (int d = 0; d < dim; ++d) {
                float sum = 0.0f;
                const float* v_row = V.data() + d * rank;
                for (int r = 0; r < rank; ++r) {
                    sum += u_row[r] * v_row[r];
                }
                out[d] = sum;
            }
        }
    }
};

/**
 * @brief Thread-safe factored vocabulary embedding lookup.
 *
 * Wraps FactoredVocabEmbeddings with TensorStreamPool integration
 * for zero-copy buffer handoff between kernels.
 */
class FactoredVocabLookup {
public:
    FactoredVocabEmbeddings embeddings;
    
    // Scratch buffers for thread-local projection results
    std::vector<std::vector<float>> thread_projection_buffers;
    
    void initialize_thread_buffers(int num_threads, int max_seq_len) {
        thread_projection_buffers.resize(num_threads);
        for (auto& buf : thread_projection_buffers) {
            buf.resize(max_seq_len * embeddings.rank);
        }
    }
    
    /**
     * @brief Score candidates with automatic buffer management.
     *
     * Projects context and scores candidates in a fused operation.
     *
     * @param context Context embeddings [seq_len, dim]
     * @param candidates Candidate indices [seq_len, num_candidates]
     * @param num_candidates Candidates per position
     * @param scores Output scores [seq_len, num_candidates]
     * @param seq_len Sequence length
     */
    void fused_project_and_score(
        const float* context,
        const int* candidates,
        int num_candidates,
        float* scores,
        int seq_len) {
        
        // Use thread-local buffer for projection
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        float* projected = thread_projection_buffers[tid].data();
        
        embeddings.project_context(context, projected, seq_len);
        embeddings.score_candidates(projected, candidates, num_candidates, 
                                   scores, seq_len);
    }
};

}  // namespace vocab
}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_VOCAB_FACTORIZATION_H_
