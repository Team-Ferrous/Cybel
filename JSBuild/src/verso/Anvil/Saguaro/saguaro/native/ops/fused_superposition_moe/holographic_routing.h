// saguaro.native/ops/fused_superposition_moe/holographic_routing.h
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
 * @file holographic_routing.h
 * @brief SIMD-optimized holographic circular correlation routing for unified
 *        HD-SuperposedExpert architecture.
 *
 * This replaces attention-based superposition collapse with holographic routing:
 *   - O(D log D) via FFT-based circular correlation (future)
 *   - O(D) via simplified cosine similarity (current optimized path)
 *
 * Key Innovation: Circular correlation enables geometric routing in HD space
 * where path selection is based on holographic similarity rather than learned
 * attention weights, maintaining algebraic closure in hyperdimensional algebra.
 *
 * SIMD Support:
 *   - AVX-512: 16-wide float32 vectorization
 *   - AVX2: 8-wide float32 vectorization with FMA
 *   - ARM NEON: 4-wide float32 vectorization
 *   - Scalar fallback for all architectures
 */

#ifndef SAGUARO_NATIVE_OPS_HOLOGRAPHIC_ROUTING_H_
#define SAGUARO_NATIVE_OPS_HOLOGRAPHIC_ROUTING_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

// Import shared SIMD utilities
#include "../hnn_simd_common.h"

namespace saguaro {
namespace hd_routing {

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * @brief Configuration for holographic superposition routing.
 */
struct HolographicRoutingConfig {
    int hd_dim = 4096;           // Hyperdimensional embedding dimension
    int superposition_dim = 4;   // Number of parallel superposition paths (K)
    float temperature = 1.0f;    // Softmax temperature for path selection
    bool use_fft = false;        // Use FFT-based correlation (future)
};

// =============================================================================
// SIMD HELPER: DOT PRODUCT
// =============================================================================

/**
 * @brief SIMD-optimized dot product of two float vectors.
 *
 * Uses horizontal sum reduction for final accumulation.
 *
 * @param a First vector [dim]
 * @param b Second vector [dim]
 * @param dim Vector dimension
 * @return Dot product a·b
 */
inline float simd_dot_product(const float* a, const float* b, int64_t dim) {
    float result = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    result = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // Horizontal sum: [a0,a1,a2,a3,a4,a5,a6,a7] -> single float
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    result = _mm_cvtss_f32(sum);
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, va, vb);
    }
    // Horizontal sum
    float32x2_t sum = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum = vpadd_f32(sum, sum);
    result = vget_lane_f32(sum, 0);
#endif

    // Scalar remainder
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// =============================================================================
// SIMD HELPER: VECTOR NORM
// =============================================================================

/**
 * @brief SIMD-optimized L2 norm of a float vector.
 *
 * @param v Vector [dim]
 * @param dim Vector dimension
 * @return L2 norm ||v||
 */
inline float simd_vector_norm(const float* v, int64_t dim) {
    float sum_sq = simd_dot_product(v, v, dim);
    return std::sqrt(sum_sq);
}

// =============================================================================
// SIMD HELPER: ELEMENTWISE MULTIPLY-ACCUMULATE
// =============================================================================

/**
 * @brief SIMD-optimized weighted accumulation: output += weight * input
 *
 * @param output Output vector [dim] (accumulated into)
 * @param input Input vector [dim]
 * @param weight Scalar weight
 * @param dim Vector dimension
 */
inline void simd_weighted_accumulate(
    float* output, const float* input, float weight, int64_t dim
) {
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 vw = _mm512_set1_ps(weight);
    for (; i + 16 <= dim; i += 16) {
        __m512 vo = _mm512_loadu_ps(&output[i]);
        __m512 vi = _mm512_loadu_ps(&input[i]);
        __m512 result = _mm512_fmadd_ps(vw, vi, vo);
        _mm512_storeu_ps(&output[i], result);
    }
#elif defined(__AVX2__)
    __m256 vw = _mm256_set1_ps(weight);
    for (; i + 8 <= dim; i += 8) {
        __m256 vo = _mm256_loadu_ps(&output[i]);
        __m256 vi = _mm256_loadu_ps(&input[i]);
        __m256 result = _mm256_fmadd_ps(vw, vi, vo);
        _mm256_storeu_ps(&output[i], result);
    }
#elif defined(__ARM_NEON)
    float32x4_t vw = vdupq_n_f32(weight);
    for (; i + 4 <= dim; i += 4) {
        float32x4_t vo = vld1q_f32(&output[i]);
        float32x4_t vi = vld1q_f32(&input[i]);
        float32x4_t result = vmlaq_f32(vo, vw, vi);
        vst1q_f32(&output[i], result);
    }
#endif

    // Scalar remainder
    for (; i < dim; ++i) {
        output[i] += weight * input[i];
    }
}

// =============================================================================
// SIMD HELPER: ELEMENTWISE HADAMARD (BINDING)
// =============================================================================

/**
 * @brief SIMD-optimized elementwise multiply (HD binding): output = a ⊗ b
 *
 * In holographic algebra, binding is circular convolution. For efficiency,
 * we use elementwise multiply as a simplified binding operation.
 *
 * @param output Output vector [dim]
 * @param a First vector [dim]
 * @param b Second vector [dim]
 * @param dim Vector dimension
 */
inline void simd_hadamard_product(
    float* output, const float* a, const float* b, int64_t dim
) {
    int64_t i = 0;

#if defined(__AVX512F__)
    for (; i + 16 <= dim; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&output[i], _mm512_mul_ps(va, vb));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&output[i], _mm256_mul_ps(va, vb));
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&output[i], vmulq_f32(va, vb));
    }
#endif

    // Scalar remainder
    for (; i < dim; ++i) {
        output[i] = a[i] * b[i];
    }
}

// =============================================================================
// HOLOGRAPHIC SIMILARITY
// =============================================================================

/**
 * @brief Compute holographic similarity via normalized dot product.
 *
 * Full holographic similarity uses circular correlation:
 *   Similarity(a, b) = IFFT(FFT(a) × conj(FFT(b)))
 *
 * This simplified version uses cosine similarity:
 *   Similarity(a, b) = (a · b) / (||a|| × ||b||)
 *
 * For normalized HD vectors, both are equivalent up to rotation.
 *
 * @param a First HD vector [hd_dim]
 * @param b Second HD vector [hd_dim]
 * @param hd_dim Dimension
 * @return Holographic similarity score in [-1, 1]
 */
inline float holographic_similarity(
    const float* a, const float* b, int64_t hd_dim
) {
    float dot = simd_dot_product(a, b, hd_dim);
    float norm_a = simd_vector_norm(a, hd_dim);
    float norm_b = simd_vector_norm(b, hd_dim);
    
    // STABILITY FIX: Use max() with epsilon to prevent inf from near-zero division
    constexpr float kEps = 1e-6f;
    float norm_product = std::max(norm_a * norm_b, kEps);
    return dot / norm_product;
}

// =============================================================================
// PATH ROUTING SCORES
// =============================================================================

/**
 * @brief Compute routing scores for K superposition paths.
 *
 * Each path k has a learned "path_base" vector. The routing score
 * is the holographic similarity between the superposed output Y_k
 * and the path_base_k.
 *
 * @param y_superposed Superposed outputs [K, hd_dim]
 * @param path_bases Path base vectors [K, hd_dim]
 * @param scores Output routing scores [K]
 * @param K Number of superposition paths
 * @param hd_dim HD dimension
 */
inline void compute_path_routing_scores(
    const float* y_superposed,
    const float* path_bases,
    float* scores,
    int K,
    int64_t hd_dim
) {
    for (int k = 0; k < K; ++k) {
        const float* y_k = y_superposed + k * hd_dim;
        const float* base_k = path_bases + k * hd_dim;
        scores[k] = holographic_similarity(y_k, base_k, hd_dim);
    }
}

// =============================================================================
// SOFTMAX OVER PATH SCORES
// =============================================================================

/**
 * @brief Apply softmax with temperature to path scores.
 *
 * weights[k] = exp(scores[k] / T) / Σ_j exp(scores[j] / T)
 *
 * Uses numerically stable softmax with max subtraction.
 *
 * @param scores Input routing scores [K]
 * @param weights Output softmax weights [K]
 * @param K Number of paths
 * @param temperature Softmax temperature
 */
inline void softmax_path_scores(
    const float* scores,
    float* weights,
    int K,
    float temperature
) {
    // Find max for numerical stability
    float max_score = scores[0];
    for (int k = 1; k < K; ++k) {
        max_score = std::max(max_score, scores[k]);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int k = 0; k < K; ++k) {
        weights[k] = std::exp((scores[k] - max_score) / temperature);
        sum_exp += weights[k];
    }
    
    // Normalize
    if (sum_exp > 1e-8f) {
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < K; ++k) {
            weights[k] *= inv_sum;
        }
    } else {
        // Fallback to uniform
        float uniform = 1.0f / static_cast<float>(K);
        for (int k = 0; k < K; ++k) {
            weights[k] = uniform;
        }
    }
}

// =============================================================================
// HOLOGRAPHIC PATH COLLAPSE
// =============================================================================

/**
 * @brief Collapse superposition paths via holographic routing.
 *
 * Replaces attention-based collapse with holographic similarity routing:
 *   1. Compute holographic similarity between each Y_k and path_base_k
 *   2. Apply softmax to get path weights
 *   3. Compute weighted combination with HD binding
 *
 * output = Σ_k weights[k] × bind(Y_superposed[k], path_weights[k])
 *
 * @param y_superposed Superposed outputs [K, hd_dim]
 * @param path_bases Path base vectors for routing [K, hd_dim]
 * @param path_weights Path transformation weights [K, hd_dim]
 * @param output Collapsed output [hd_dim]
 * @param config Routing configuration
 */
inline void holographic_collapse(
    const float* y_superposed,
    const float* path_bases,
    const float* path_weights,
    float* output,
    const HolographicRoutingConfig& config
) {
    const int K = config.superposition_dim;
    const int64_t hd_dim = config.hd_dim;
    
    // Step 1: Compute routing scores
    std::vector<float> scores(K);
    compute_path_routing_scores(y_superposed, path_bases, scores.data(), K, hd_dim);
    
    // Step 2: Apply softmax
    std::vector<float> weights(K);
    softmax_path_scores(scores.data(), weights.data(), K, config.temperature);
    
    // Step 3: Compute weighted combination with binding
    // Initialize output to zero
    std::fill(output, output + hd_dim, 0.0f);
    
    std::vector<float> bound_output(hd_dim);
    for (int k = 0; k < K; ++k) {
        const float* y_k = y_superposed + k * hd_dim;
        const float* pw_k = path_weights + k * hd_dim;
        
        // Bind Y_k with path_weights_k
        simd_hadamard_product(bound_output.data(), y_k, pw_k, hd_dim);
        
        // Accumulate weighted result
        simd_weighted_accumulate(output, bound_output.data(), weights[k], hd_dim);
    }
}

// =============================================================================
// HOLOGRAPHIC COLLAPSE WITH GRADIENT SUPPORT
// =============================================================================

/**
 * @brief Forward pass of holographic collapse with intermediate storage.
 *
 * Stores intermediates needed for backward pass:
 *   - scores, weights for routing gradient
 *   - bound outputs for weight gradients
 *
 * @param y_superposed Superposed outputs [K, hd_dim]
 * @param path_bases Path base vectors [K, hd_dim]
 * @param path_weights Path transformation weights [K, hd_dim]
 * @param output Collapsed output [hd_dim]
 * @param routing_weights_out Output: routing weights [K] (for backward)
 * @param config Routing configuration
 */
inline void holographic_collapse_forward(
    const float* y_superposed,
    const float* path_bases,
    const float* path_weights,
    float* output,
    float* routing_weights_out,
    const HolographicRoutingConfig& config
) {
    const int K = config.superposition_dim;
    const int64_t hd_dim = config.hd_dim;
    
    // Step 1: Compute routing scores
    std::vector<float> scores(K);
    compute_path_routing_scores(y_superposed, path_bases, scores.data(), K, hd_dim);
    
    // Step 2: Apply softmax
    softmax_path_scores(scores.data(), routing_weights_out, K, config.temperature);
    
    // Step 3: Compute weighted combination with binding
    std::fill(output, output + hd_dim, 0.0f);
    
    std::vector<float> bound_output(hd_dim);
    for (int k = 0; k < K; ++k) {
        const float* y_k = y_superposed + k * hd_dim;
        const float* pw_k = path_weights + k * hd_dim;
        
        simd_hadamard_product(bound_output.data(), y_k, pw_k, hd_dim);
        simd_weighted_accumulate(output, bound_output.data(), routing_weights_out[k], hd_dim);
    }
}

/**
 * @brief Backward pass of holographic collapse.
 *
 * Computes gradients for:
 *   - y_superposed: grad_y_superposed [K, hd_dim]
 *   - path_weights: grad_path_weights [K, hd_dim]
 *   - path_bases: grad_path_bases [K, hd_dim] (via routing gradient)
 *
 * @param grad_output Gradient w.r.t output [hd_dim]
 * @param y_superposed Superposed outputs from forward [K, hd_dim]
 * @param path_bases Path base vectors [K, hd_dim]
 * @param path_weights Path transformation weights [K, hd_dim]
 * @param routing_weights Routing weights from forward [K]
 * @param grad_y_superposed Output: gradient w.r.t y_superposed [K, hd_dim]
 * @param grad_path_weights Output: gradient w.r.t path_weights [K, hd_dim]
 * @param grad_path_bases Output: gradient w.r.t path_bases [K, hd_dim]
 * @param config Routing configuration
 */
inline void holographic_collapse_backward(
    const float* grad_output,
    const float* y_superposed,
    const float* path_bases,
    const float* path_weights,
    const float* routing_weights,
    float* grad_y_superposed,
    float* grad_path_weights,
    float* grad_path_bases,
    const HolographicRoutingConfig& config
) {
    const int K = config.superposition_dim;
    const int64_t hd_dim = config.hd_dim;
    const float temperature = config.temperature;
    
    // Zero initialize output gradient only (grad_y_superposed is output)
    // NOTE: grad_path_weights and grad_path_bases are ACCUMULATORS - do NOT zero them!
    // The caller is responsible for managing accumulator initialization.
    std::fill(grad_y_superposed, grad_y_superposed + K * hd_dim, 0.0f);
    
    // =========================================================================
    // Part 1: Gradient through weighted sum
    // output = Σ_k w_k * (y_k ⊗ pw_k)
    // =========================================================================
    
    // Compute bound outputs for routing gradient (needed below)
    std::vector<float> bound_outputs(K * hd_dim);
    for (int k = 0; k < K; ++k) {
        const float* y_k = y_superposed + k * hd_dim;
        const float* pw_k = path_weights + k * hd_dim;
        float* bound_k = bound_outputs.data() + k * hd_dim;
        simd_hadamard_product(bound_k, y_k, pw_k, hd_dim);
    }
    
    for (int k = 0; k < K; ++k) {
        const float* y_k = y_superposed + k * hd_dim;
        const float* pw_k = path_weights + k * hd_dim;
        float w_k = routing_weights[k];
        
        float* grad_y_k = grad_y_superposed + k * hd_dim;
        float* grad_pw_k = grad_path_weights + k * hd_dim;
        
        for (int64_t d = 0; d < hd_dim; ++d) {
            // Chain rule: grad_output * d(bound)/d(y) * d(weighted)/d(bound)
            float grad_bound = grad_output[d] * w_k;
            
            // d(y_k * pw_k)/d(y_k) = pw_k
            grad_y_k[d] += grad_bound * pw_k[d];
            
            // d(y_k * pw_k)/d(pw_k) = y_k
            grad_pw_k[d] += grad_bound * y_k[d];
        }
    }
    
    // =========================================================================
    // Part 2: Gradient through softmax routing -> scores -> path_bases
    //
    // output = Σ_k softmax_k * bound_k
    // softmax_k = exp(score_k / T) / Σ_j exp(score_j / T)
    // score_k = cosine_similarity(y_k, path_bases_k)
    //         = (y_k · path_bases_k) / (||y_k|| * ||path_bases_k||)
    //
    // Gradient chain:
    //   d(output)/d(path_bases_k) = Σ_d grad_output[d] * d(output[d])/d(w_k) 
    //                             * d(w_k)/d(score_k) * d(score_k)/d(path_bases_k)
    // =========================================================================
    
    // Step 2a: Compute d(output)/d(w_k) = bound_k (dot with grad_output)
    std::vector<float> grad_weights(K, 0.0f);
    for (int k = 0; k < K; ++k) {
        const float* bound_k = bound_outputs.data() + k * hd_dim;
        for (int64_t d = 0; d < hd_dim; ++d) {
            grad_weights[k] += grad_output[d] * bound_k[d];
        }
    }
    
    // Step 2b: Compute d(w_k)/d(score_j) via softmax Jacobian
    // Jacobian: dw_i/ds_j = w_i * (delta_ij - w_j) / T
    // grad_score_k = Σ_i grad_w_i * dw_i/ds_k
    //              = Σ_i grad_w_i * w_i * (delta_ik - w_k) / T
    //              = (grad_w_k * w_k - w_k * Σ_i grad_w_i * w_i) / T
    float weighted_grad_sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        weighted_grad_sum += grad_weights[i] * routing_weights[i];
    }
    
    std::vector<float> grad_scores(K);
    for (int k = 0; k < K; ++k) {
        float w_k = routing_weights[k];
        grad_scores[k] = (grad_weights[k] * w_k - w_k * weighted_grad_sum) / temperature;
    }
    
    // Step 2c: Compute d(score_k)/d(path_bases_k) for cosine similarity
    // score_k = (y_k · b_k) / (||y_k|| * ||b_k||)
    // d(score_k)/d(b_k[d]) = y_k[d] / (||y_k|| * ||b_k||) 
    //                        - (y_k · b_k) * b_k[d] / (||y_k|| * ||b_k||^3)
    //                      = (y_k[d] - score_k * b_k[d] / ||b_k||^2) / (||y_k|| * ||b_k||)
    //
    // GRADIENT FIX v2: More conservative scaling to prevent vanishing
    // Previous: 1/sqrt(max(8, K)) was too aggressive (~0.35)
    // New: 1/sqrt(K) gives ~0.5 for K=4, better gradient flow
    const float grad_scale = 1.0f / std::sqrt(static_cast<float>(K) + 1e-6f);
    
    // Minimum gradient threshold to prevent complete vanishing
    const float min_grad = 1e-7f;
    
    for (int k = 0; k < K; ++k) {
        const float* y_k = y_superposed + k * hd_dim;
        const float* b_k = path_bases + k * hd_dim;
        float* grad_b_k = grad_path_bases + k * hd_dim;
        
        // Compute norms with stability guards
        float norm_y = simd_vector_norm(y_k, hd_dim);
        float norm_b = simd_vector_norm(b_k, hd_dim);
        
        // STABILITY FIX: Use larger epsilon with max() to prevent inf
        constexpr float kEps = 1e-6f;
        float norm_product = std::max(norm_y * norm_b, kEps);
        
        // Skip only if truly degenerate (both norms essentially zero)
        if (norm_y < 1e-12f && norm_b < 1e-12f) {
            continue;
        }
        
        float dot_yb = simd_dot_product(y_k, b_k, hd_dim);
        float score_k = dot_yb / norm_product;
        
        // STABILITY FIX: Guard norm_b_sq with epsilon to prevent inf
        float norm_b_sq = std::max(norm_b * norm_b, kEps);
        float inv_norm_product = 1.0f / norm_product;
        
        // Gradient contribution
        float grad_s_k = grad_scores[k] * grad_scale;
        
        for (int64_t d = 0; d < hd_dim; ++d) {
            // d(score)/d(b[d]) = (y[d] - score * b[d] / ||b||^2) / (||y|| * ||b||)
            float dscore_db = (y_k[d] - score_k * b_k[d] / norm_b_sq) * inv_norm_product;
            float grad_val = grad_s_k * dscore_db;
            
            // Apply minimum threshold to prevent complete vanishing
            if (std::abs(grad_val) > min_grad) {
                grad_b_k[d] += grad_val;
            } else if (std::abs(grad_s_k) > min_grad) {
                // If score gradient is non-trivial, use small perturbation
                grad_b_k[d] += std::copysign(min_grad, grad_val);
            }
        }
    }
}

}  // namespace hd_routing
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_HOLOGRAPHIC_ROUTING_H_
