// highnoon/_native/ops/memory_builder_enhancements.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Memory Builder Enhancement Kernels for hierarchical memory aggregation.
// Implements Enhancements 3-7 from the Memory Builder roadmap.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef HIGHNOON_NATIVE_OPS_MEMORY_BUILDER_ENHANCEMENTS_H_
#define HIGHNOON_NATIVE_OPS_MEMORY_BUILDER_ENHANCEMENTS_H_

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define MEMBUILDER_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define MEMBUILDER_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define MEMBUILDER_NEON 1
#endif

namespace highnoon {
namespace ops {

// =============================================================================
// SIMD HELPER FUNCTIONS (with unique membuilder_ prefix)
// =============================================================================

/**
 * @brief SIMD-optimized dot product.
 */
inline float membuilder_dot_product(const float* a, const float* b, int64_t size) {
    float sum = 0.0f;
    int64_t i = 0;
    
#if defined(MEMBUILDER_AVX512)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    sum = _mm512_reduce_add_ps(acc);
#elif defined(MEMBUILDER_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum = _mm_cvtss_f32(sum128);
#elif defined(MEMBUILDER_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    sum = vaddvq_f32(acc);
#endif
    
    // Scalar remainder
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

/**
 * @brief SIMD-optimized vector norm.
 */
inline float membuilder_norm(const float* x, int64_t size) {
    return std::sqrt(membuilder_dot_product(x, x, size) + 1e-8f);
}

/**
 * @brief SIMD-optimized softmax in-place.
 */
inline void membuilder_softmax_inplace(float* x, int64_t size) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int64_t i = 1; i < size; ++i) {
        max_val = std::max(max_val, x[i]);
    }
    
    // Exp and sum
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int64_t i = 0; i < size; ++i) {
        x[i] *= inv_sum;
    }
}

// =============================================================================
// ENHANCEMENT 3: CTQW-ENHANCED AGGREGATION
// Continuous-Time Quantum Walk for computing aggregation weights
// =============================================================================

/**
 * @brief CTQW Aggregator using Cayley-approximated matrix exponential.
 * 
 * Computes aggregation weights via exp(-itL) where L is the graph Laplacian.
 * Uses Cayley transform: (I - itL/2)(I + itL/2)^-1 ≈ exp(-itL)
 * 
 * Reference: CTQW-GraphSAGE (2024), GQWformer (2024)
 */
class CTQWAggregator {
public:
    /**
     * @brief Construct CTQW aggregator.
     * 
     * @param time Initial quantum walk time (learnable)
     * @param use_cayley Use Cayley approximation (faster) vs direct exponential
     */
    explicit CTQWAggregator(float time = 1.0f, bool use_cayley = true)
        : time_(time), use_cayley_(use_cayley) {}
    
    /**
     * @brief Compute CTQW aggregation weights from node representations.
     * 
     * Algorithm:
     * 1. Compute adjacency A[i,j] = exp(-||x[i] - x[j]||^2 / sigma^2)
     * 2. Compute degree D[i] = sum_j A[i,j]
     * 3. Compute Laplacian L = D - A
     * 4. Apply Cayley evolution: weights = |exp(-itL)|^2
     * 
     * @param x Input representations [num_nodes, embed_dim]
     * @param weights Output aggregation weights [num_nodes, num_nodes]
     * @param num_nodes Number of nodes
     * @param embed_dim Embedding dimension
     * @param sigma Kernel bandwidth (default: sqrt(embed_dim))
     */
    void compute_weights(
        const float* x,
        float* weights,
        int64_t num_nodes,
        int64_t embed_dim,
        float sigma = -1.0f
    ) const {
        if (sigma < 0) {
            sigma = std::sqrt(static_cast<float>(embed_dim));
        }
        float inv_sigma2 = 1.0f / (sigma * sigma + 1e-8f);
        
        // Step 1: Compute similarity-based adjacency matrix
        std::vector<float> adjacency(num_nodes * num_nodes);
        std::vector<float> degree(num_nodes, 0.0f);
        
        for (int64_t i = 0; i < num_nodes; ++i) {
            for (int64_t j = 0; j < num_nodes; ++j) {
                if (i == j) {
                    adjacency[i * num_nodes + j] = 0.0f;
                    continue;
                }
                
                // Compute squared distance
                float dist2 = 0.0f;
                for (int64_t d = 0; d < embed_dim; ++d) {
                    float diff = x[i * embed_dim + d] - x[j * embed_dim + d];
                    dist2 += diff * diff;
                }
                
                // RBF kernel
                float a_ij = std::exp(-dist2 * inv_sigma2);
                adjacency[i * num_nodes + j] = a_ij;
                degree[i] += a_ij;
            }
        }
        
        // Step 2: Compute Laplacian L = D - A
        std::vector<float> laplacian(num_nodes * num_nodes);
        for (int64_t i = 0; i < num_nodes; ++i) {
            for (int64_t j = 0; j < num_nodes; ++j) {
                if (i == j) {
                    laplacian[i * num_nodes + j] = degree[i];
                } else {
                    laplacian[i * num_nodes + j] = -adjacency[i * num_nodes + j];
                }
            }
        }
        
        // Step 3: Apply Cayley evolution exp(-itL)
        if (use_cayley_) {
            apply_cayley_evolution(laplacian.data(), weights, num_nodes);
        } else {
            // Direct Taylor approximation for small matrices
            apply_taylor_evolution(laplacian.data(), weights, num_nodes);
        }
        
        // Step 4: Take absolute values squared (probability amplitudes)
        // and normalize rows to get proper weights
        for (int64_t i = 0; i < num_nodes; ++i) {
            float row_sum = 0.0f;
            for (int64_t j = 0; j < num_nodes; ++j) {
                float val = weights[i * num_nodes + j];
                weights[i * num_nodes + j] = val * val;  // |amplitude|^2
                row_sum += weights[i * num_nodes + j];
            }
            // Normalize
            if (row_sum > 1e-8f) {
                float inv_sum = 1.0f / row_sum;
                for (int64_t j = 0; j < num_nodes; ++j) {
                    weights[i * num_nodes + j] *= inv_sum;
                }
            }
        }
    }
    
    /**
     * @brief Compute gradient of CTQW w.r.t. time parameter.
     */
    float grad_time(const float* x, const float* grad_weights,
                    int64_t num_nodes, int64_t embed_dim) const {
        // Numerical gradient for now (analytical is complex)
        const float epsilon = 1e-4f;
        
        std::vector<float> weights_plus(num_nodes * num_nodes);
        std::vector<float> weights_minus(num_nodes * num_nodes);
        
        CTQWAggregator agg_plus(time_ + epsilon, use_cayley_);
        CTQWAggregator agg_minus(time_ - epsilon, use_cayley_);
        
        agg_plus.compute_weights(x, weights_plus.data(), num_nodes, embed_dim);
        agg_minus.compute_weights(x, weights_minus.data(), num_nodes, embed_dim);
        
        float grad = 0.0f;
        for (int64_t i = 0; i < num_nodes * num_nodes; ++i) {
            grad += grad_weights[i] * (weights_plus[i] - weights_minus[i]) / (2.0f * epsilon);
        }
        return grad;
    }
    
    void set_time(float t) { time_ = t; }
    float get_time() const { return time_; }

private:
    float time_;
    bool use_cayley_;
    
    /**
     * @brief Cayley approximation: (I - A)(I + A)^-1 where A = itL/2
     */
    void apply_cayley_evolution(
        const float* laplacian,
        float* output,
        int64_t n
    ) const {
        float alpha = time_ * 0.5f;  // itL/2 coefficient
        
        std::vector<float> I_plus_A(n * n);
        std::vector<float> I_minus_A(n * n);
        
        // Build I + α*L and I - α*L
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float L_ij = laplacian[i * n + j];
                float identity = (i == j) ? 1.0f : 0.0f;
                I_plus_A[i * n + j] = identity + alpha * L_ij;
                I_minus_A[i * n + j] = identity - alpha * L_ij;
            }
        }
        
        // Solve (I + A)^-1 via Gauss-Jordan for small matrices
        std::vector<float> inv(n * n);
        invert_matrix(I_plus_A.data(), inv.data(), n);
        
        // Multiply: (I - A) @ inv
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < n; ++k) {
                    sum += I_minus_A[i * n + k] * inv[k * n + j];
                }
                output[i * n + j] = sum;
            }
        }
    }
    
    /**
     * @brief Taylor approximation: exp(-itL) ≈ I - itL + (itL)^2/2 - ...
     */
    void apply_taylor_evolution(
        const float* laplacian,
        float* output,
        int64_t n
    ) const {
        // Initialize output as identity
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                output[i * n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        
        // Taylor terms up to order 4
        std::vector<float> term(n * n);
        std::vector<float> L_power(n * n);
        
        // Initialize L_power as identity
        for (int64_t i = 0; i < n * n; ++i) {
            L_power[i] = (i % (n + 1) == 0) ? 1.0f : 0.0f;
        }
        
        float coeff = 1.0f;
        float t = -time_;  // -it (imaginary, but treat as real for approximation)
        
        for (int order = 1; order <= 4; ++order) {
            coeff *= t / static_cast<float>(order);
            
            // L_power = L_power @ L
            std::vector<float> temp(n * n, 0.0f);
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t k = 0; k < n; ++k) {
                        temp[i * n + j] += L_power[i * n + k] * laplacian[k * n + j];
                    }
                }
            }
            std::copy(temp.begin(), temp.end(), L_power.begin());
            
            // output += coeff * L_power
            for (int64_t i = 0; i < n * n; ++i) {
                output[i] += coeff * L_power[i];
            }
        }
    }
    
    /**
     * @brief Simple matrix inversion via Gauss-Jordan elimination.
     */
    void invert_matrix(const float* A, float* inv, int64_t n) const {
        std::vector<float> work(n * 2 * n);
        
        // Build augmented matrix [A | I]
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                work[i * 2 * n + j] = A[i * n + j];
                work[i * 2 * n + n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        
        // Forward elimination with partial pivoting
        for (int64_t col = 0; col < n; ++col) {
            // Find pivot
            int64_t max_row = col;
            float max_val = std::abs(work[col * 2 * n + col]);
            for (int64_t row = col + 1; row < n; ++row) {
                float val = std::abs(work[row * 2 * n + col]);
                if (val > max_val) {
                    max_val = val;
                    max_row = row;
                }
            }
            
            // Swap rows
            if (max_row != col) {
                for (int64_t j = 0; j < 2 * n; ++j) {
                    std::swap(work[col * 2 * n + j], work[max_row * 2 * n + j]);
                }
            }
            
            // Scale pivot row
            float pivot = work[col * 2 * n + col];
            if (std::abs(pivot) < 1e-10f) {
                pivot = 1e-10f;  // Avoid division by zero
            }
            float inv_pivot = 1.0f / pivot;
            for (int64_t j = 0; j < 2 * n; ++j) {
                work[col * 2 * n + j] *= inv_pivot;
            }
            
            // Eliminate column
            for (int64_t row = 0; row < n; ++row) {
                if (row != col) {
                    float factor = work[row * 2 * n + col];
                    for (int64_t j = 0; j < 2 * n; ++j) {
                        work[row * 2 * n + j] -= factor * work[col * 2 * n + j];
                    }
                }
            }
        }
        
        // Extract inverse from right half
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                inv[i * n + j] = work[i * 2 * n + n + j];
            }
        }
    }
};

// =============================================================================
// ENHANCEMENT 4: MULTI-RATE HIERARCHICAL PROCESSING
// Different hierarchy levels operate at different temporal rates
// =============================================================================

/**
 * @brief Multi-rate EMA processor for hierarchical memory.
 * 
 * Implements: memory_new = α * memory + (1-α) * aggregated
 * where α = base_rate * (decay ^ level)
 * 
 * Lower levels (fine): low α = fast update
 * Higher levels (coarse): high α = slow update
 * 
 * Reference: HMNet (CVPR 2024), Titans (Google 2024)
 */
class MultiRateProcessor {
public:
    /**
     * @brief Construct multi-rate processor.
     * 
     * @param base_rate Initial rate for level 0 (typically 0.1-0.5)
     * @param level_decay Decay factor per level (typically 0.5-0.9)
     */
    explicit MultiRateProcessor(float base_rate = 0.1f, float level_decay = 0.5f)
        : base_rate_(base_rate), level_decay_(level_decay) {}
    
    /**
     * @brief Apply multi-rate EMA update.
     * 
     * @param memory Current memory [num_tokens, embed_dim]
     * @param aggregated New aggregated values [num_tokens, embed_dim]
     * @param output Updated memory [num_tokens, embed_dim]
     * @param num_tokens Number of tokens
     * @param embed_dim Embedding dimension
     * @param level Current hierarchy level (0 = finest)
     */
    void apply(
        const float* memory,
        const float* aggregated,
        float* output,
        int64_t num_tokens,
        int64_t embed_dim,
        int level
    ) const {
        // Compute level-dependent rate: higher level = slower update
        float alpha = compute_rate(level);
        float one_minus_alpha = 1.0f - alpha;
        
        int64_t total = num_tokens * embed_dim;
        int64_t i = 0;
        
#if defined(MEMBUILDER_AVX512)
        __m512 valpha = _mm512_set1_ps(alpha);
        __m512 v1ma = _mm512_set1_ps(one_minus_alpha);
        for (; i + 16 <= total; i += 16) {
            __m512 vm = _mm512_loadu_ps(memory + i);
            __m512 va = _mm512_loadu_ps(aggregated + i);
            __m512 result = _mm512_fmadd_ps(valpha, vm, _mm512_mul_ps(v1ma, va));
            _mm512_storeu_ps(output + i, result);
        }
#elif defined(MEMBUILDER_AVX2)
        __m256 valpha = _mm256_set1_ps(alpha);
        __m256 v1ma = _mm256_set1_ps(one_minus_alpha);
        for (; i + 8 <= total; i += 8) {
            __m256 vm = _mm256_loadu_ps(memory + i);
            __m256 va = _mm256_loadu_ps(aggregated + i);
            __m256 result = _mm256_fmadd_ps(valpha, vm, _mm256_mul_ps(v1ma, va));
            _mm256_storeu_ps(output + i, result);
        }
#elif defined(MEMBUILDER_NEON)
        float32x4_t valpha = vdupq_n_f32(alpha);
        float32x4_t v1ma = vdupq_n_f32(one_minus_alpha);
        for (; i + 4 <= total; i += 4) {
            float32x4_t vm = vld1q_f32(memory + i);
            float32x4_t va = vld1q_f32(aggregated + i);
            float32x4_t result = vmlaq_f32(vmulq_f32(v1ma, va), valpha, vm);
            vst1q_f32(output + i, result);
        }
#endif
        
        // Scalar remainder
        for (; i < total; ++i) {
            output[i] = alpha * memory[i] + one_minus_alpha * aggregated[i];
        }
    }
    
    /**
     * @brief Compute gradient w.r.t. inputs.
     */
    void grad(
        const float* grad_output,
        float* grad_memory,
        float* grad_aggregated,
        int64_t num_tokens,
        int64_t embed_dim,
        int level
    ) const {
        float alpha = compute_rate(level);
        float one_minus_alpha = 1.0f - alpha;
        
        int64_t total = num_tokens * embed_dim;
        for (int64_t i = 0; i < total; ++i) {
            grad_memory[i] = alpha * grad_output[i];
            grad_aggregated[i] = one_minus_alpha * grad_output[i];
        }
    }
    
    float compute_rate(int level) const {
        return base_rate_ * std::pow(level_decay_, static_cast<float>(level));
    }
    
    void set_base_rate(float rate) { base_rate_ = rate; }
    void set_level_decay(float decay) { level_decay_ = decay; }

private:
    float base_rate_;
    float level_decay_;
};

// =============================================================================
// ENHANCEMENT 5: CROSS-LEVEL ATTENTION BRIDGES
// Bidirectional information flow across hierarchy levels
// =============================================================================

/**
 * @brief O(n) linear cross-level attention using ELU+1 kernel.
 * 
 * Enables:
 * - Fine-to-coarse: details inform summaries
 * - Coarse-to-fine: context guides local interpretation
 * 
 * Uses kernel attention: K(q,k) = (elu(q) + 1) · (elu(k) + 1)
 * For O(n) complexity via associative property.
 * 
 * Reference: Cross-Layer Attention (MIT 2025), Titans Memory Module
 */
class CrossLevelAttention {
public:
    /**
     * @brief Construct cross-level attention.
     * 
     * @param num_heads Number of attention heads
     * @param head_dim Dimension per head
     * @param shared_kv Whether to share KV across levels (CLA-style)
     */
    CrossLevelAttention(int num_heads = 4, int head_dim = 32, bool shared_kv = true)
        : num_heads_(num_heads), head_dim_(head_dim), shared_kv_(shared_kv) {}
    
    /**
     * @brief Apply cross-level attention.
     * 
     * @param query Query from one level [num_query, embed_dim]
     * @param key Key from another level [num_kv, embed_dim]
     * @param value Value from another level [num_kv, embed_dim]
     * @param output Attended output [num_query, embed_dim]
     * @param num_query Number of query tokens
     * @param num_kv Number of key/value tokens
     * @param embed_dim Embedding dimension
     */
    void apply(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        int64_t num_query,
        int64_t num_kv,
        int64_t embed_dim
    ) const {
        // For simplicity, assume embed_dim = num_heads * head_dim
        // Use linear attention with ELU+1 kernel
        
        // Step 1: Apply ELU+1 kernel to queries and keys
        std::vector<float> q_elu(num_query * embed_dim);
        std::vector<float> k_elu(num_kv * embed_dim);
        
        apply_elu_plus_1(query, q_elu.data(), num_query * embed_dim);
        apply_elu_plus_1(key, k_elu.data(), num_kv * embed_dim);
        
        // Step 2: Compute KV summary: S = sum_j k_elu[j] ⊗ v[j]
        // For each head, this is [head_dim, head_dim]
        std::vector<float> kv_sum(embed_dim * embed_dim, 0.0f);
        std::vector<float> k_sum(embed_dim, 0.0f);
        
        for (int64_t j = 0; j < num_kv; ++j) {
            for (int64_t d1 = 0; d1 < embed_dim; ++d1) {
                k_sum[d1] += k_elu[j * embed_dim + d1];
                for (int64_t d2 = 0; d2 < embed_dim; ++d2) {
                    kv_sum[d1 * embed_dim + d2] += 
                        k_elu[j * embed_dim + d1] * value[j * embed_dim + d2];
                }
            }
        }
        
        // Step 3: For each query, compute output
        // output[i] = (q_elu[i] @ S) / (q_elu[i] @ k_sum)
        for (int64_t i = 0; i < num_query; ++i) {
            // Compute normalizer: q @ k_sum
            float normalizer = 0.0f;
            for (int64_t d = 0; d < embed_dim; ++d) {
                normalizer += q_elu[i * embed_dim + d] * k_sum[d];
            }
            normalizer = std::max(normalizer, 1e-6f);
            
            // Compute output: q @ S / normalizer
            for (int64_t d2 = 0; d2 < embed_dim; ++d2) {
                float sum = 0.0f;
                for (int64_t d1 = 0; d1 < embed_dim; ++d1) {
                    sum += q_elu[i * embed_dim + d1] * kv_sum[d1 * embed_dim + d2];
                }
                output[i * embed_dim + d2] = sum / normalizer;
            }
        }
    }
    
    /**
     * @brief Apply with residual connection.
     */
    void apply_residual(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        int64_t num_query,
        int64_t num_kv,
        int64_t embed_dim,
        float residual_scale = 1.0f
    ) const {
        std::vector<float> attended(num_query * embed_dim);
        apply(query, key, value, attended.data(), num_query, num_kv, embed_dim);
        
        // output = query + scale * attended
        for (int64_t i = 0; i < num_query * embed_dim; ++i) {
            output[i] = query[i] + residual_scale * attended[i];
        }
    }

private:
    int num_heads_;
    int head_dim_;
    bool shared_kv_;
    
    /**
     * @brief Apply ELU+1 activation: f(x) = elu(x) + 1 = max(0,x) + min(0, exp(x)-1) + 1
     */
    void apply_elu_plus_1(const float* input, float* output, int64_t size) const {
        for (int64_t i = 0; i < size; ++i) {
            float x = input[i];
            float elu = (x >= 0) ? x : (std::exp(x) - 1.0f);
            output[i] = elu + 1.0f;
        }
    }
};

// =============================================================================
// ENHANCEMENT 6: ADAPTIVE CONTENT-BASED CHUNKING
// Replace fixed chunking with semantic boundary detection
// =============================================================================

/**
 * @brief Adaptive chunker based on representation discontinuity.
 * 
 * Identifies chunk boundaries at local minima of similarity between
 * adjacent representations.
 * 
 * Reference: HSNN (Meta 2024), Hierarchical Document Processing
 */
class AdaptiveChunker {
public:
    /**
     * @brief Construct adaptive chunker.
     * 
     * @param min_chunk_size Minimum tokens per chunk
     * @param max_chunk_size Maximum tokens per chunk
     * @param boundary_threshold Similarity threshold for boundary detection
     */
    AdaptiveChunker(int min_chunk_size = 2, int max_chunk_size = 16, 
                    float boundary_threshold = 0.5f)
        : min_chunk_size_(min_chunk_size)
        , max_chunk_size_(max_chunk_size)
        , boundary_threshold_(boundary_threshold) {}
    
    /**
     * @brief Compute chunk assignments.
     * 
     * @param x Input representations [num_tokens, embed_dim]
     * @param chunk_ids Output chunk assignment per token [num_tokens]
     * @param num_tokens Number of tokens
     * @param embed_dim Embedding dimension
     * @return Number of chunks created
     */
    int compute_chunks(
        const float* x,
        int* chunk_ids,
        int64_t num_tokens,
        int64_t embed_dim
    ) const {
        if (num_tokens <= min_chunk_size_) {
            // Single chunk
            for (int64_t i = 0; i < num_tokens; ++i) {
                chunk_ids[i] = 0;
            }
            return 1;
        }
        
        // Step 1: Compute local similarity scores
        std::vector<float> similarity(num_tokens - 1);
        for (int64_t i = 0; i < num_tokens - 1; ++i) {
            similarity[i] = cosine_similarity(
                x + i * embed_dim,
                x + (i + 1) * embed_dim,
                embed_dim
            );
        }
        
        // Step 2: Find local minima as potential boundaries
        std::vector<bool> is_boundary(num_tokens, false);
        for (int64_t i = 1; i < num_tokens - 2; ++i) {
            if (similarity[i] < similarity[i-1] && 
                similarity[i] < similarity[i+1] &&
                similarity[i] < boundary_threshold_) {
                is_boundary[i + 1] = true;  // Boundary after token i
            }
        }
        
        // Step 3: Enforce min/max chunk size constraints
        int current_chunk = 0;
        int chunk_start = 0;
        
        for (int64_t i = 0; i < num_tokens; ++i) {
            chunk_ids[i] = current_chunk;
            
            int chunk_size = static_cast<int>(i - chunk_start + 1);
            
            // Force boundary if max size reached
            if (chunk_size >= max_chunk_size_ && i < num_tokens - 1) {
                current_chunk++;
                chunk_start = i + 1;
            }
            // Allow natural boundary if min size satisfied
            else if (is_boundary[i] && chunk_size >= min_chunk_size_ && 
                     i < num_tokens - 1) {
                current_chunk++;
                chunk_start = i + 1;
            }
        }
        
        return current_chunk + 1;
    }
    
    /**
     * @brief Pool within chunks using mean aggregation.
     * 
     * @param x Input representations [num_tokens, embed_dim]
     * @param chunk_ids Chunk assignments [num_tokens]
     * @param output Pooled representations [num_chunks, embed_dim]
     * @param num_tokens Number of input tokens
     * @param num_chunks Number of chunks
     * @param embed_dim Embedding dimension
     */
    void pool_chunks(
        const float* x,
        const int* chunk_ids,
        float* output,
        int64_t num_tokens,
        int num_chunks,
        int64_t embed_dim
    ) const {
        // Initialize output and counts
        std::vector<int> counts(num_chunks, 0);
        std::fill(output, output + num_chunks * embed_dim, 0.0f);
        
        // Accumulate
        for (int64_t i = 0; i < num_tokens; ++i) {
            int chunk = chunk_ids[i];
            counts[chunk]++;
            for (int64_t d = 0; d < embed_dim; ++d) {
                output[chunk * embed_dim + d] += x[i * embed_dim + d];
            }
        }
        
        // Average
        for (int c = 0; c < num_chunks; ++c) {
            if (counts[c] > 0) {
                float inv_count = 1.0f / static_cast<float>(counts[c]);
                for (int64_t d = 0; d < embed_dim; ++d) {
                    output[c * embed_dim + d] *= inv_count;
                }
            }
        }
    }

private:
    int min_chunk_size_;
    int max_chunk_size_;
    float boundary_threshold_;
    
    float cosine_similarity(const float* a, const float* b, int64_t size) const {
        float dot = membuilder_dot_product(a, b, size);
        float norm_a = membuilder_norm(a, size);
        float norm_b = membuilder_norm(b, size);
        return dot / (norm_a * norm_b + 1e-8f);
    }
};

// =============================================================================
// ENHANCEMENT 7: ENHANCED QGAN WITH QUANTUM NOISE
// Structured quantum noise for improved QGAN training
// =============================================================================

/**
 * @brief Quantum noise generator using rotation matrix samples.
 * 
 * Replaces Gaussian noise with structured quantum-inspired noise
 * that maintains correlations via rotation matrices.
 * 
 * Reference: Quantum Noise Improves GANs (arXiv 2024)
 */
class QuantumNoiseGenerator {
public:
    /**
     * @brief Construct quantum noise generator.
     * 
     * @param entanglement_strength Correlation strength between dimensions
     * @param seed Random seed for reproducibility
     */
    explicit QuantumNoiseGenerator(float entanglement_strength = 0.1f, 
                                    uint64_t seed = 42)
        : entanglement_strength_(entanglement_strength)
        , rng_(seed)
        , normal_dist_(0.0f, 1.0f)
        , uniform_dist_(0.0f, 2.0f * 3.14159265358979f) {}
    
    /**
     * @brief Generate structured quantum noise.
     * 
     * Algorithm:
     * 1. Sample base Gaussian noise
     * 2. Apply random rotation matrices (quantum-inspired unitary)
     * 3. Add entanglement correlations between pairs
     * 
     * @param output Output noise tensor [batch, dim]
     * @param batch Batch size
     * @param dim Noise dimension
     */
    void generate(float* output, int64_t batch, int64_t dim) {
        // Step 1: Base Gaussian noise
        for (int64_t i = 0; i < batch * dim; ++i) {
            output[i] = normal_dist_(rng_);
        }
        
        // Step 2: Apply 2D rotations to pairs of dimensions
        for (int64_t b = 0; b < batch; ++b) {
            float* row = output + b * dim;
            
            // Rotate pairs
            for (int64_t d = 0; d + 1 < dim; d += 2) {
                float theta = uniform_dist_(rng_) * entanglement_strength_;
                float cos_t = std::cos(theta);
                float sin_t = std::sin(theta);
                
                float x = row[d];
                float y = row[d + 1];
                row[d] = cos_t * x - sin_t * y;
                row[d + 1] = sin_t * x + cos_t * y;
            }
        }
        
        // Step 3: Add entanglement correlations between adjacent batch samples
        if (batch > 1 && entanglement_strength_ > 0) {
            for (int64_t b = 1; b < batch; ++b) {
                float* curr = output + b * dim;
                float* prev = output + (b - 1) * dim;
                
                for (int64_t d = 0; d < dim; ++d) {
                    curr[d] = (1.0f - entanglement_strength_) * curr[d] + 
                              entanglement_strength_ * prev[d];
                }
            }
        }
    }
    
    /**
     * @brief Compute entanglement regularization loss.
     * 
     * Encourages structured correlations in generated samples.
     * 
     * @param samples Generated noise [batch, dim]
     * @param batch Batch size
     * @param dim Dimension
     * @return Regularization loss value
     */
    float entanglement_loss(const float* samples, int64_t batch, int64_t dim) const {
        if (batch < 2) return 0.0f;
        
        // Target: adjacent samples should have correlation ~entanglement_strength
        float loss = 0.0f;
        
        for (int64_t b = 1; b < batch; ++b) {
            const float* curr = samples + b * dim;
            const float* prev = samples + (b - 1) * dim;
            
            // Compute empirical correlation
            float dot = membuilder_dot_product(curr, prev, dim);
            float norm_curr = membuilder_norm(curr, dim);
            float norm_prev = membuilder_norm(prev, dim);
            float corr = dot / (norm_curr * norm_prev + 1e-8f);
            
            // L2 loss towards target correlation
            float diff = corr - entanglement_strength_;
            loss += diff * diff;
        }
        
        return loss / static_cast<float>(batch - 1);
    }
    
    void set_entanglement_strength(float strength) { 
        entanglement_strength_ = strength; 
    }
    void set_seed(uint64_t seed) { rng_.seed(seed); }

private:
    float entanglement_strength_;
    std::mt19937_64 rng_;
    std::normal_distribution<float> normal_dist_;
    std::uniform_real_distribution<float> uniform_dist_;
};

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_MEMORY_BUILDER_ENHANCEMENTS_H_
