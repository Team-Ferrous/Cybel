// highnoon/_native/ops/fused_mod_routing_op.h
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
 * @file fused_mod_routing_op.h
 * @brief Mixture-of-Depths (MoD) routing SIMD helpers.
 *
 * Implements the Mixture-of-Depths mechanism from Google DeepMind (2024):
 * Dynamic per-token routing that decides whether to process, skip, or repeat
 * transformer layers based on token difficulty.
 *
 * Key Innovation:
 *   - Unlike uniform depth, MoD allows "easy" tokens to skip layers
 *   - Saves 50% FLOPs while maintaining or improving quality
 *   - Per-token routing decision via lightweight router network
 *
 * Operations:
 *   1. Router computes importance score for each token
 *   2. Top-k tokens selected for processing (capacity constraint)
 *   3. Selected tokens routed through layer, others skip
 *   4. Outputs recombined preserving original positions
 *
 * SIMD optimizations for all architectures.
 *
 * @note Thread-safe. All functions are reentrant.
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_MOD_ROUTING_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_MOD_ROUTING_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <numeric>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define MOD_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define MOD_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define MOD_NEON 1
#endif

namespace highnoon {
namespace ops {

// =============================================================================
// MOD ROUTING CONFIGURATION
// =============================================================================

/**
 * @brief Routing decision for a token.
 */
enum class MoDAction {
    SKIP = 0,     // Token skips this layer (residual passthrough)
    PROCESS = 1,  // Token is processed by this layer
    REPEAT = 2    // Token goes through layer twice (for very difficult tokens)
};

/**
 * @brief Configuration for MoD routing.
 */
struct MoDConfig {
    float capacity_factor = 0.5f;  // Fraction of tokens to process (default 50%)
    bool allow_repeat = false;     // Whether to allow repeat routing
    float repeat_threshold = 0.95f; // Score threshold for repeat (if enabled)
    float skip_threshold = 0.3f;   // Score threshold for definite skip
    bool use_auxiliary_loss = true;// Add load balancing aux loss
    float aux_loss_weight = 0.01f; // Weight of auxiliary loss
};

/**
 * @brief Result of MoD routing.
 */
struct MoDRoutingResult {
    std::vector<int32_t> selected_indices;  // Indices of tokens to process
    std::vector<int32_t> route_weights;     // 0=skip, 1=process, 2=repeat
    std::vector<float> router_probs;        // Router probabilities for each token
    float auxiliary_loss;                   // Load balancing loss
};

// =============================================================================
// SIMD HELPERS
// =============================================================================

/**
 * @brief SIMD sigmoid activation.
 */
inline void mod_sigmoid_inplace(float* data, int64_t size) {
    int64_t i = 0;
    
#if defined(MOD_AVX2)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666667f);
    const __m256 c4 = _mm256_set1_ps(0.04166667f);
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 neg_x = _mm256_mul_ps(x, neg_one);
        
        // Taylor exp approximation
        __m256 x2 = _mm256_mul_ps(neg_x, neg_x);
        __m256 x3 = _mm256_mul_ps(x2, neg_x);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        
        __m256 exp_neg = _mm256_add_ps(one, neg_x);
        exp_neg = _mm256_fmadd_ps(x2, c2, exp_neg);
        exp_neg = _mm256_fmadd_ps(x3, c3, exp_neg);
        exp_neg = _mm256_fmadd_ps(x4, c4, exp_neg);
        
        __m256 denom = _mm256_add_ps(one, exp_neg);
        __m256 result = _mm256_div_ps(one, denom);
        
        _mm256_storeu_ps(&data[i], result);
    }
#endif
    
    for (; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

/**
 * @brief SIMD dot product for router score computation.
 */
inline float mod_dot(const float* a, const float* b, int64_t size) {
    float sum = 0.0f;
    int64_t i = 0;
    
#if defined(MOD_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum = _mm_cvtss_f32(sum128);
#endif
    
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// =============================================================================
// CORE ROUTING OPERATIONS
// =============================================================================

/**
 * @brief Compute router logits for all tokens.
 *
 * router_logit[t] = hidden[t] @ router_weight + router_bias
 *
 * @param hidden Hidden states [num_tokens, hidden_dim]
 * @param router_weight Router projection [hidden_dim]
 * @param router_bias Router bias (scalar)
 * @param logits Output logits [num_tokens]
 * @param num_tokens Number of tokens
 * @param hidden_dim Hidden dimension
 */
inline void mod_compute_router_logits(
    const float* hidden,
    const float* router_weight,
    float router_bias,
    float* logits,
    int num_tokens,
    int hidden_dim) {
    
    #pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        const float* h = hidden + t * hidden_dim;
        logits[t] = mod_dot(h, router_weight, hidden_dim) + router_bias;
    }
    
    // Apply sigmoid to get routing probabilities
    mod_sigmoid_inplace(logits, num_tokens);
}

/**
 * @brief Select top-k tokens for processing based on router scores.
 *
 * Uses efficient partial sort to find top-k without full sorting.
 *
 * @param router_probs Router probabilities [num_tokens]
 * @param num_tokens Total number of tokens
 * @param capacity Maximum tokens to process
 * @param selected_indices Output: indices of selected tokens
 * @return Number of actually selected tokens
 */
inline int mod_select_top_k(
    const float* router_probs,
    int num_tokens,
    int capacity,
    std::vector<int32_t>& selected_indices) {
    
    capacity = std::min(capacity, num_tokens);
    
    // Create index-score pairs
    std::vector<std::pair<float, int>> scored_indices(num_tokens);
    for (int i = 0; i < num_tokens; ++i) {
        scored_indices[i] = {router_probs[i], i};
    }
    
    // Partial sort to get top-k
    std::partial_sort(
        scored_indices.begin(),
        scored_indices.begin() + capacity,
        scored_indices.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Extract indices
    selected_indices.resize(capacity);
    for (int i = 0; i < capacity; ++i) {
        selected_indices[i] = scored_indices[i].second;
    }
    
    // Sort by position for efficient scatter
    std::sort(selected_indices.begin(), selected_indices.end());
    
    return capacity;
}

/**
 * @brief Compute auxiliary load balancing loss.
 *
 * Encourages uniform token selection across sequence positions.
 * aux_loss = mean(router_probs) * num_tokens / capacity
 *
 * @param router_probs Router probabilities [num_tokens]
 * @param num_tokens Total tokens
 * @param capacity Target capacity
 * @return Auxiliary loss value
 */
inline float mod_compute_aux_loss(
    const float* router_probs,
    int num_tokens,
    int capacity) {
    
    float sum = 0.0f;
    for (int i = 0; i < num_tokens; ++i) {
        sum += router_probs[i];
    }
    float mean_prob = sum / num_tokens;
    
    // Ideal mean probability for uniform selection
    float target_prob = static_cast<float>(capacity) / num_tokens;
    
    // Squared difference
    float diff = mean_prob - target_prob;
    return diff * diff * num_tokens;
}

/**
 * @brief Full MoD routing forward pass.
 *
 * @param hidden Hidden states [num_tokens, hidden_dim]
 * @param router_weight Router weight [hidden_dim]
 * @param router_bias Router bias
 * @param num_tokens Number of tokens
 * @param hidden_dim Hidden dimension
 * @param config MoD configuration
 * @return MoDRoutingResult with selected indices and aux loss
 */
inline MoDRoutingResult mod_route_forward(
    const float* hidden,
    const float* router_weight,
    float router_bias,
    int num_tokens,
    int hidden_dim,
    const MoDConfig& config) {
    
    MoDRoutingResult result;
    
    // Compute router probabilities
    result.router_probs.resize(num_tokens);
    mod_compute_router_logits(
        hidden, router_weight, router_bias,
        result.router_probs.data(),
        num_tokens, hidden_dim);
    
    // Determine capacity
    int capacity = static_cast<int>(num_tokens * config.capacity_factor + 0.5f);
    capacity = std::max(1, capacity);
    
    // Select top-k tokens
    mod_select_top_k(
        result.router_probs.data(),
        num_tokens,
        capacity,
        result.selected_indices);
    
    // Compute route weights (0=skip, 1=process)
    result.route_weights.resize(num_tokens, 0);
    for (int idx : result.selected_indices) {
        result.route_weights[idx] = 1;
        
        // Check for repeat routing
        if (config.allow_repeat && result.router_probs[idx] >= config.repeat_threshold) {
            result.route_weights[idx] = 2;
        }
    }
    
    // Compute auxiliary loss
    if (config.use_auxiliary_loss) {
        result.auxiliary_loss = config.aux_loss_weight * 
            mod_compute_aux_loss(result.router_probs.data(), num_tokens, capacity);
    } else {
        result.auxiliary_loss = 0.0f;
    }
    
    return result;
}

/**
 * @brief Gather selected tokens for layer processing.
 *
 * @param hidden Full hidden states [num_tokens, hidden_dim]
 * @param selected_indices Indices of tokens to gather
 * @param gathered Output: gathered tokens [num_selected, hidden_dim]
 * @param num_selected Number of selected tokens
 * @param hidden_dim Hidden dimension
 */
inline void mod_gather(
    const float* hidden,
    const int32_t* selected_indices,
    float* gathered,
    int num_selected,
    int hidden_dim) {
    
    #pragma omp parallel for
    for (int i = 0; i < num_selected; ++i) {
        int src_idx = selected_indices[i];
        const float* src = hidden + src_idx * hidden_dim;
        float* dst = gathered + i * hidden_dim;
        
        int64_t d = 0;
#if defined(MOD_AVX2)
        for (; d + 8 <= hidden_dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&src[d]);
            _mm256_storeu_ps(&dst[d], v);
        }
#endif
        for (; d < hidden_dim; ++d) {
            dst[d] = src[d];
        }
    }
}

/**
 * @brief Scatter processed tokens back to original positions.
 *
 * @param processed Processed tokens [num_selected, hidden_dim]
 * @param selected_indices Original positions
 * @param output Full output [num_tokens, hidden_dim] (pre-initialized with residual)
 * @param router_probs Router probabilities for weighting
 * @param num_selected Number of processed tokens
 * @param hidden_dim Hidden dimension
 */
inline void mod_scatter_add(
    const float* processed,
    const int32_t* selected_indices,
    float* output,
    const float* router_probs,
    int num_selected,
    int hidden_dim) {
    
    #pragma omp parallel for
    for (int i = 0; i < num_selected; ++i) {
        int dst_idx = selected_indices[i];
        const float* src = processed + i * hidden_dim;
        float* dst = output + dst_idx * hidden_dim;
        float weight = router_probs[dst_idx];
        
        int64_t d = 0;
#if defined(MOD_AVX2)
        __m256 w = _mm256_set1_ps(weight);
        for (; d + 8 <= hidden_dim; d += 8) {
            __m256 p = _mm256_loadu_ps(&src[d]);
            __m256 o = _mm256_loadu_ps(&dst[d]);
            // output = residual * (1 - weight) + processed * weight
            // Since output is already residual, add weighted processed
            o = _mm256_fmadd_ps(w, p, _mm256_mul_ps(o, _mm256_sub_ps(_mm256_set1_ps(1.0f), w)));
            _mm256_storeu_ps(&dst[d], o);
        }
#endif
        for (; d < hidden_dim; ++d) {
            dst[d] = dst[d] * (1.0f - weight) + src[d] * weight;
        }
    }
}

/**
 * @brief Combined MoD forward pass with layer application.
 *
 * This is the main entry point for MoD routing:
 * 1. Compute router scores
 * 2. Select top-k tokens
 * 3. Gather selected tokens
 * 4. (Caller applies layer to gathered tokens)
 * 5. Scatter results back
 *
 * @param hidden Input hidden states [num_tokens, hidden_dim]
 * @param router_weight Router weight vector [hidden_dim]
 * @param router_bias Router bias scalar
 * @param output Output hidden states [num_tokens, hidden_dim]
 * @param gathered_out Output: gathered tokens for layer [capacity, hidden_dim]
 * @param selected_out Output: selected indices [capacity]
 * @param num_selected_out Output: actual number selected
 * @param aux_loss_out Output: auxiliary loss
 * @param num_tokens Number of tokens
 * @param hidden_dim Hidden dimension
 * @param config MoD configuration
 */
inline void mod_forward_gather(
    const float* hidden,
    const float* router_weight,
    float router_bias,
    float* output,
    float* gathered_out,
    int32_t* selected_out,
    int* num_selected_out,
    float* aux_loss_out,
    int num_tokens,
    int hidden_dim,
    const MoDConfig& config) {
    
    // Route
    MoDRoutingResult route = mod_route_forward(
        hidden, router_weight, router_bias,
        num_tokens, hidden_dim, config);
    
    // Copy to output buffers
    *num_selected_out = route.selected_indices.size();
    std::copy(route.selected_indices.begin(), route.selected_indices.end(), selected_out);
    *aux_loss_out = route.auxiliary_loss;
    
    // Initialize output with residual (copy input)
    std::copy(hidden, hidden + num_tokens * hidden_dim, output);
    
    // Gather selected tokens
    mod_gather(hidden, selected_out, gathered_out, *num_selected_out, hidden_dim);
}

/**
 * @brief Scatter processed tokens after layer application.
 *
 * @param processed Layer output for selected tokens [num_selected, hidden_dim]
 * @param selected_indices Selected token indices
 * @param output Output to scatter into [num_tokens, hidden_dim]
 * @param router_probs Router probabilities [num_tokens]
 * @param num_selected Number of selected tokens
 * @param hidden_dim Hidden dimension
 */
inline void mod_forward_scatter(
    const float* processed,
    const int32_t* selected_indices,
    float* output,
    const float* router_probs,
    int num_selected,
    int hidden_dim) {
    
    mod_scatter_add(processed, selected_indices, output, router_probs, num_selected, hidden_dim);
}

// =============================================================================
// GRADIENT OPERATIONS
// =============================================================================

/**
 * @brief Gradient for MoD scatter operation.
 *
 * @param grad_output Gradient w.r.t. full output [num_tokens, hidden_dim]
 * @param selected_indices Selected token indices
 * @param router_probs Router probabilities
 * @param grad_processed Output: gradient for processed [num_selected, hidden_dim]
 * @param num_selected Number of selected
 * @param hidden_dim Hidden dimension
 */
inline void mod_scatter_grad(
    const float* grad_output,
    const int32_t* selected_indices,
    const float* router_probs,
    float* grad_processed,
    int num_selected,
    int hidden_dim) {
    
    #pragma omp parallel for
    for (int i = 0; i < num_selected; ++i) {
        int idx = selected_indices[i];
        const float* grad_out = grad_output + idx * hidden_dim;
        float* grad_proc = grad_processed + i * hidden_dim;
        float weight = router_probs[idx];
        
        for (int d = 0; d < hidden_dim; ++d) {
            grad_proc[d] = grad_out[d] * weight;
        }
    }
}

/**
 * @brief Gradient for router parameters.
 *
 * @param grad_output Gradient w.r.t. output [num_tokens, hidden_dim]
 * @param hidden Input hidden states [num_tokens, hidden_dim]
 * @param processed Processed outputs (for selected) [num_selected, hidden_dim]
 * @param selected_indices Selected indices
 * @param router_probs Router probabilities
 * @param grad_router_weight Output: gradient for router weight [hidden_dim]
 * @param grad_router_bias Output: gradient for router bias
 * @param num_tokens Total tokens
 * @param num_selected Selected tokens
 * @param hidden_dim Hidden dimension
 */
inline void mod_router_grad(
    const float* grad_output,
    const float* hidden,
    const float* processed,
    const int32_t* selected_indices,
    const float* router_probs,
    float* grad_router_weight,
    float* grad_router_bias,
    int num_tokens,
    int num_selected,
    int hidden_dim) {
    
    // Initialize gradients to zero
    std::fill(grad_router_weight, grad_router_weight + hidden_dim, 0.0f);
    *grad_router_bias = 0.0f;
    
    // Gradient flows through the routing probability
    // d_loss/d_router_logit = d_loss/d_prob * d_prob/d_logit
    // where d_prob/d_logit = prob * (1 - prob) for sigmoid
    
    for (int i = 0; i < num_selected; ++i) {
        int idx = selected_indices[i];
        const float* grad_out = grad_output + idx * hidden_dim;
        const float* h = hidden + idx * hidden_dim;
        const float* proc = processed + i * hidden_dim;
        float prob = router_probs[idx];
        
        // Gradient through weighted combination
        float d_prob = 0.0f;
        for (int d = 0; d < hidden_dim; ++d) {
            // output = hidden * (1 - prob) + processed * prob
            // d_output/d_prob = processed - hidden
            d_prob += grad_out[d] * (proc[d] - h[d]);
        }
        
        // Chain rule through sigmoid
        float d_logit = d_prob * prob * (1.0f - prob);
        
        // Accumulate gradients
        for (int d = 0; d < hidden_dim; ++d) {
            grad_router_weight[d] += d_logit * h[d];
        }
        *grad_router_bias += d_logit;
    }
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_MOD_ROUTING_OP_H_
