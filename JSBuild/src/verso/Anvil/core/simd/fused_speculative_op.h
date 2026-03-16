// highnoon/_native/ops/fused_speculative_op.h
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
 * @file fused_speculative_op.h
 * @brief Speculative Decoding SIMD helpers.
 *
 * Core operations for speculative decoding acceleration:
 *   - Temperature-scaled softmax
 *   - Top-k filtering
 *   - Probability gathering
 *   - Acceptance ratio computation
 *   - Token rejection sampling
 *
 * SIMD optimizations for probability computations.
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_SPECULATIVE_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_SPECULATIVE_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace highnoon {
namespace ops {

/**
 * @brief Apply temperature scaling to logits.
 *
 * logits[i] = logits[i] / temperature
 */
inline void speculative_temperature_scale(
    float* logits, int64_t size, float temperature) {
    
    float inv_temp = 1.0f / std::max(temperature, 1e-8f);
    int64_t i = 0;
    
#if defined(__AVX512F__)
    __m512 scale = _mm512_set1_ps(inv_temp);
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&logits[i]);
        _mm512_storeu_ps(&logits[i], _mm512_mul_ps(v, scale));
    }
#elif defined(__AVX2__)
    __m256 scale = _mm256_set1_ps(inv_temp);
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&logits[i]);
        _mm256_storeu_ps(&logits[i], _mm256_mul_ps(v, scale));
    }
#elif defined(__ARM_NEON)
    float32x4_t scale = vdupq_n_f32(inv_temp);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&logits[i]);
        vst1q_f32(&logits[i], vmulq_f32(v, scale));
    }
#endif
    for (; i < size; ++i) {
        logits[i] *= inv_temp;
    }
}

/**
 * @brief Apply top-k filtering to logits.
 *
 * Sets all logits below the k-th largest to -inf.
 */
inline void speculative_top_k_filter(
    float* logits, int64_t size, int64_t k) {
    
    if (k <= 0 || k >= size) return;
    
    // Find k-th largest value
    std::vector<float> sorted(logits, logits + size);
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                      std::greater<float>());
    float threshold = sorted[k - 1];
    
    // Mask values below threshold
    for (int64_t i = 0; i < size; ++i) {
        if (logits[i] < threshold) {
            logits[i] = -1e9f;
        }
    }
}

/**
 * @brief Compute softmax probabilities from logits.
 */
inline void speculative_softmax(
    const float* logits, float* probs, int64_t size) {
    
    // Find max for numerical stability
    float max_val = logits[0];
    for (int64_t i = 1; i < size; ++i) {
        max_val = std::max(max_val, logits[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-8f);
    int64_t i = 0;
#if defined(__AVX512F__)
    __m512 scale = _mm512_set1_ps(inv_sum);
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&probs[i]);
        _mm512_storeu_ps(&probs[i], _mm512_mul_ps(v, scale));
    }
#elif defined(__AVX2__)
    __m256 scale = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&probs[i]);
        _mm256_storeu_ps(&probs[i], _mm256_mul_ps(v, scale));
    }
#elif defined(__ARM_NEON)
    float32x4_t scale = vdupq_n_f32(inv_sum);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&probs[i]);
        vst1q_f32(&probs[i], vmulq_f32(v, scale));
    }
#endif
    for (; i < size; ++i) {
        probs[i] *= inv_sum;
    }
}

/**
 * @brief Gather probabilities for specific token indices.
 *
 * @param probs Probability distributions [batch, vocab]
 * @param tokens Token indices [batch]
 * @param output Output probabilities [batch]
 * @param batch_size Number of samples
 * @param vocab_size Vocabulary size
 */
inline void speculative_gather_probs(
    const float* probs, const int32_t* tokens, float* output,
    int64_t batch_size, int64_t vocab_size) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        int32_t token = tokens[b];
        if (token >= 0 && token < vocab_size) {
            output[b] = probs[b * vocab_size + token];
        } else {
            output[b] = 0.0f;
        }
    }
}

/**
 * @brief Compute acceptance ratios for rejection sampling.
 *
 * accept_ratio = min(1, p_target / p_draft)
 */
inline void speculative_acceptance_ratio(
    const float* p_target, const float* p_draft, float* ratios,
    int64_t size) {
    
    for (int64_t i = 0; i < size; ++i) {
        float ratio = p_target[i] / (p_draft[i] + 1e-8f);
        ratios[i] = std::min(1.0f, ratio);
    }
}

/**
 * @brief Perform rejection sampling to determine accepted tokens.
 *
 * @param accept_ratios Acceptance ratios [num_speculative]
 * @param num_speculative Number of speculated tokens
 * @return Number of accepted tokens (0 to num_speculative)
 */
inline int speculative_reject_sample(
    const float* accept_ratios, int64_t num_speculative,
    std::mt19937& rng) {
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int64_t i = 0; i < num_speculative; ++i) {
        float u = dist(rng);
        if (u >= accept_ratios[i]) {
            // Reject this and all subsequent tokens
            return static_cast<int>(i);
        }
    }
    
    // All tokens accepted
    return static_cast<int>(num_speculative);
}

/**
 * @brief Sample a token from a probability distribution.
 *
 * @param probs Probability distribution [vocab_size]
 * @param vocab_size Size of vocabulary
 * @return Sampled token index
 */
inline int32_t speculative_sample_token(
    const float* probs, int64_t vocab_size, std::mt19937& rng) {
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    
    float cumsum = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (u <= cumsum) {
            return static_cast<int32_t>(i);
        }
    }
    
    // Fallback to last token
    return static_cast<int32_t>(vocab_size - 1);
}

/**
 * @brief Argmax over probability/logits.
 */
inline int32_t speculative_argmax(
    const float* values, int64_t size) {
    
    int32_t max_idx = 0;
    float max_val = values[0];
    
    for (int64_t i = 1; i < size; ++i) {
        if (values[i] > max_val) {
            max_val = values[i];
            max_idx = static_cast<int32_t>(i);
        }
    }
    
    return max_idx;
}

/**
 * @brief Batch speculative verification.
 *
 * Verifies K draft tokens against target probabilities using
 * rejection sampling.
 *
 * @param target_probs Target model probs [num_spec, vocab]
 * @param draft_probs Draft model probs [num_spec, vocab]
 * @param draft_tokens Draft token indices [num_spec]
 * @param num_speculative Number of speculated tokens
 * @param vocab_size Vocabulary size
 * @return Number of accepted tokens
 */
inline int speculative_batch_verify(
    const float* target_probs, const float* draft_probs,
    const int32_t* draft_tokens,
    int64_t num_speculative, int64_t vocab_size,
    std::mt19937& rng) {
    
    for (int64_t i = 0; i < num_speculative; ++i) {
        int32_t token = draft_tokens[i];
        
        float p_target = target_probs[i * vocab_size + token];
        float p_draft = draft_probs[i * vocab_size + token];
        
        float accept_ratio = std::min(1.0f, p_target / (p_draft + 1e-8f));
        
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float u = dist(rng);
        
        if (u >= accept_ratio) {
            return static_cast<int>(i);
        }
    }
    
    return static_cast<int>(num_speculative);
}

// =============================================================================
// EAGLE-STYLE ENHANCEMENTS (ICLR 2024, EMNLP 2024, NeurIPS 2025)
// =============================================================================

/**
 * @brief Configuration for EAGLE-style speculative decoding.
 */
struct EAGLEConfig {
    int draft_depth = 4;             // Number of draft tokens to generate
    float acceptance_threshold = 0.5f; // Min acceptance rate for adaptation
    bool use_dynamic_tree = true;    // Context-aware draft tree (EAGLE-2)
    int num_feature_layers = 3;      // Layers to fuse features from (EAGLE-3)
    int max_tree_width = 4;          // Maximum tree branching factor
    float temperature = 1.0f;        // Sampling temperature
    bool greedy_verification = true; // Use argmax for verification target
};

/**
 * @brief Draft tree node for dynamic expansion.
 */
struct DraftTreeNode {
    int32_t token;
    float prob;
    int parent_idx;
    int depth;
    bool accepted;
};

/**
 * @brief Fuse features from multiple transformer layers (EAGLE-3).
 *
 * EAGLE-3 extracts and combines features from multiple layers, not just
 * the second-to-top layer like EAGLE-1, for better draft prediction.
 *
 * @param layer_features Features from each layer [num_layers, seq_len, hidden_dim]
 * @param layer_weights Learnable weights for each layer [num_layers]
 * @param fused_output Output fused features [seq_len, hidden_dim]
 * @param num_layers Number of layers to fuse
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 */
inline void eagle_fuse_layer_features(
    const float* layer_features,
    const float* layer_weights,
    float* fused_output,
    int num_layers,
    int seq_len,
    int hidden_dim) {
    
    // Initialize output to zero
    std::fill_n(fused_output, seq_len * hidden_dim, 0.0f);
    
    // Compute weighted sum of layer features
    #pragma omp parallel for
    for (int s = 0; s < seq_len; ++s) {
        for (int l = 0; l < num_layers; ++l) {
            float weight = layer_weights[l];
            const float* layer_feat = layer_features + l * seq_len * hidden_dim + s * hidden_dim;
            float* out = fused_output + s * hidden_dim;
            
            int64_t d = 0;
#if defined(__AVX2__)
            __m256 w_vec = _mm256_set1_ps(weight);
            for (; d + 8 <= hidden_dim; d += 8) {
                __m256 f = _mm256_loadu_ps(&layer_feat[d]);
                __m256 o = _mm256_loadu_ps(&out[d]);
                o = _mm256_fmadd_ps(w_vec, f, o);
                _mm256_storeu_ps(&out[d], o);
            }
#endif
            for (; d < hidden_dim; ++d) {
                out[d] += weight * layer_feat[d];
            }
        }
    }
}

/**
 * @brief Compute context-aware acceptance threshold (EAGLE-2).
 *
 * The acceptance rate varies based on context - some contexts are
 * easier to predict than others. This adapts the threshold dynamically.
 *
 * @param hidden_state Last hidden state [hidden_dim]
 * @param threshold_weights Learned threshold network weights [hidden_dim]
 * @param hidden_dim Hidden dimension
 * @param default_threshold Default acceptance threshold
 * @return Adapted threshold for this context
 */
inline float eagle_context_aware_threshold(
    const float* hidden_state,
    const float* threshold_weights,
    int hidden_dim,
    float default_threshold = 0.5f) {
    
    // Simple learned threshold: sigmoid(dot(hidden, weights))
    float dot = 0.0f;
    int64_t i = 0;
    
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= hidden_dim; i += 8) {
        __m256 h = _mm256_loadu_ps(&hidden_state[i]);
        __m256 w = _mm256_loadu_ps(&threshold_weights[i]);
        acc = _mm256_fmadd_ps(h, w, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    dot = _mm_cvtss_f32(sum128);
#endif
    for (; i < hidden_dim; ++i) {
        dot += hidden_state[i] * threshold_weights[i];
    }
    
    // Sigmoid to get threshold
    float threshold = 1.0f / (1.0f + std::exp(-dot));
    
    // Blend with default (learned adjustment)
    return 0.5f * threshold + 0.5f * default_threshold;
}

/**
 * @brief Build dynamic draft tree based on context confidence.
 *
 * EAGLE-2 uses a context-aware tree where high-confidence predictions
 * branch more widely, while uncertain predictions branch less.
 *
 * @param draft_logits Logits from draft model [max_depth, vocab_size]
 * @param config EAGLE configuration
 * @param vocab_size Vocabulary size
 * @param tree Output tree nodes
 * @param rng Random number generator
 */
inline void eagle_build_dynamic_tree(
    const float* draft_logits,
    const EAGLEConfig& config,
    int64_t vocab_size,
    std::vector<DraftTreeNode>& tree,
    std::mt19937& rng) {
    
    tree.clear();
    std::vector<float> probs(vocab_size);
    
    // Add root (depth 0)
    for (int depth = 0; depth < config.draft_depth; ++depth) {
        const float* logits = draft_logits + depth * vocab_size;
        speculative_softmax(logits, probs.data(), vocab_size);
        
        // Find confidence (entropy-based)
        float entropy = 0.0f;
        for (int64_t v = 0; v < vocab_size; ++v) {
            if (probs[v] > 1e-8f) {
                entropy -= probs[v] * std::log(probs[v]);
            }
        }
        float max_entropy = std::log(static_cast<float>(vocab_size));
        float confidence = 1.0f - entropy / max_entropy;
        
        // Adaptive branching: more branches for confident predictions
        int width = 1;
        if (config.use_dynamic_tree) {
            width = std::max(1, static_cast<int>(confidence * config.max_tree_width));
        }
        
        // Sample top-k tokens based on width
        std::vector<std::pair<float, int32_t>> ranked;
        for (int64_t v = 0; v < vocab_size; ++v) {
            ranked.push_back({probs[v], static_cast<int32_t>(v)});
        }
        std::partial_sort(ranked.begin(), ranked.begin() + width, ranked.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Add nodes
        int parent = depth == 0 ? -1 : static_cast<int>(tree.size()) - 1;
        for (int w = 0; w < width; ++w) {
            DraftTreeNode node;
            node.token = ranked[w].second;
            node.prob = ranked[w].first;
            node.parent_idx = parent;
            node.depth = depth;
            node.accepted = false;
            tree.push_back(node);
        }
    }
}

/**
 * @brief EAGLE-style batch verification with tree structure.
 *
 * Verifies draft tree against target model, accepting the longest
 * valid prefix in the tree.
 *
 * @param target_probs Target model probs for each tree node
 * @param tree Draft tree structure
 * @param config EAGLE configuration
 * @param vocab_size Vocabulary size
 * @param rng Random generator
 * @return Number of accepted tokens from root
 */
inline int eagle_verify_tree(
    const float* target_probs,
    std::vector<DraftTreeNode>& tree,
    const EAGLEConfig& config,
    int64_t vocab_size,
    std::mt19937& rng) {
    
    if (tree.empty()) return 0;
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int max_accepted = 0;
    
    for (size_t i = 0; i < tree.size(); ++i) {
        DraftTreeNode& node = tree[i];
        
        // Get target probability for this token
        float p_target = target_probs[i * vocab_size + node.token];
        
        // Rejection sampling
        float accept_ratio = std::min(1.0f, p_target / (node.prob + 1e-8f));
        float u = dist(rng);
        
        if (u < accept_ratio) {
            node.accepted = true;
            max_accepted = std::max(max_accepted, node.depth + 1);
        } else {
            // Rejection: don't accept this or descendants
            node.accepted = false;
        }
    }
    
    return max_accepted;
}

/**
 * @brief Compute residual correction for draft head (EAGLE-3).
 *
 * EAGLE-3 uses a residual connection that allows the draft head to
 * make small corrections to feature vectors rather than generating
 * from scratch.
 *
 * @param features Input features [seq_len, hidden_dim]
 * @param residual_weight Residual projection [hidden_dim, hidden_dim]
 * @param residual_bias Residual bias [hidden_dim]
 * @param output Output with residual [seq_len, hidden_dim]
 * @param seq_len Sequence length
 * @param hidden_dim Hidden dimension
 */
inline void eagle_apply_residual(
    const float* features,
    const float* residual_weight,
    const float* residual_bias,
    float* output,
    int seq_len,
    int hidden_dim) {
    
    #pragma omp parallel for
    for (int s = 0; s < seq_len; ++s) {
        const float* in = features + s * hidden_dim;
        float* out = output + s * hidden_dim;
        
        // Compute residual projection
        for (int o = 0; o < hidden_dim; ++o) {
            float sum = residual_bias ? residual_bias[o] : 0.0f;
            for (int i = 0; i < hidden_dim; ++i) {
                sum += in[i] * residual_weight[i * hidden_dim + o];
            }
            // Add residual connection
            out[o] = in[o] + sum;
        }
    }
}

/**
 * @brief Compute masked self-attention for draft head.
 *
 * Lightweight self-attention for the draft head to maintain context.
 *
 * @param Q Query [seq_len, head_dim]
 * @param K Key [seq_len, head_dim]
 * @param V Value [seq_len, head_dim]
 * @param output Output [seq_len, head_dim]
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 */
inline void eagle_draft_attention(
    const float* Q, const float* K, const float* V,
    float* output,
    int seq_len, int head_dim) {
    
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);
    
    for (int q = 0; q < seq_len; ++q) {
        const float* q_vec = Q + q * head_dim;
        float* out = output + q * head_dim;
        
        // Compute attention scores (causal: only attend to past)
        float max_score = -1e9f;
        for (int k = 0; k <= q; ++k) {
            const float* k_vec = K + k * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_vec[d] * k_vec[d];
            }
            scores[k] = score * scale;
            max_score = std::max(max_score, scores[k]);
        }
        
        // Softmax
        float sum = 0.0f;
        for (int k = 0; k <= q; ++k) {
            scores[k] = std::exp(scores[k] - max_score);
            sum += scores[k];
        }
        for (int k = 0; k <= q; ++k) {
            scores[k] /= (sum + 1e-8f);
        }
        
        // Apply to values
        std::fill_n(out, head_dim, 0.0f);
        for (int k = 0; k <= q; ++k) {
            const float* v_vec = V + k * head_dim;
            float w = scores[k];
            for (int d = 0; d < head_dim; ++d) {
                out[d] += w * v_vec[d];
            }
        }
    }
}

/**
 * @brief Complete EAGLE-3 speculative generation pass.
 *
 * Generates draft tokens using fused features and dynamic tree expansion.
 *
 * @param layer_features Features from transformer layers [num_layers, 1, hidden_dim]
 * @param layer_weights Layer fusion weights [num_layers]
 * @param draft_weight Draft head projection [hidden_dim, vocab_size]
 * @param draft_bias Draft head bias [vocab_size]
 * @param draft_tokens Output: draft token sequence [max_draft]
 * @param draft_probs Output: draft probabilities [max_draft]
 * @param num_layers Number of feature layers
 * @param hidden_dim Hidden dimension
 * @param vocab_size Vocabulary size
 * @param config EAGLE configuration
 * @param rng Random generator
 * @return Number of draft tokens generated
 */
inline int eagle_generate_drafts(
    const float* layer_features,
    const float* layer_weights,
    const float* draft_weight,
    const float* draft_bias,
    int32_t* draft_tokens,
    float* draft_probs,
    int num_layers,
    int hidden_dim,
    int64_t vocab_size,
    const EAGLEConfig& config,
    std::mt19937& rng) {
    
    // Fuse features from multiple layers
    std::vector<float> fused(hidden_dim);
    eagle_fuse_layer_features(
        layer_features, layer_weights, fused.data(),
        num_layers, 1, hidden_dim);
    
    // Generate drafts autoregressively
    std::vector<float> logits(vocab_size);
    std::vector<float> probs(vocab_size);
    
    for (int d = 0; d < config.draft_depth; ++d) {
        // Project to vocab logits
        for (int64_t v = 0; v < vocab_size; ++v) {
            float sum = draft_bias ? draft_bias[v] : 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                sum += fused[h] * draft_weight[h * vocab_size + v];
            }
            logits[v] = sum / config.temperature;
        }
        
        // Softmax
        speculative_softmax(logits.data(), probs.data(), vocab_size);
        
        // Sample
        draft_tokens[d] = speculative_sample_token(probs.data(), vocab_size, rng);
        draft_probs[d] = probs[draft_tokens[d]];
    }
    
    return config.draft_depth;
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_SPECULATIVE_OP_H_

