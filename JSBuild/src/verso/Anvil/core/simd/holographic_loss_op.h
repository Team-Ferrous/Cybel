// highnoon/_native/ops/holographic_loss_op.h
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
 * @file holographic_loss_op.h
 * @brief Phase 200+: Holographic Cross-Entropy Loss in HD space.
 *
 * HIGHNOON_UPGRADE_ROADMAP.md Phase 3.2 - QULS Native Ops.
 *
 * This op computes cross-entropy loss directly in HD space, avoiding the
 * projection back to vocab-space for memory efficiency. The key insight is
 * that softmax probabilities can be approximated via holographic similarity:
 *
 *   P(token_i | hd_state) ∝ exp(similarity(hd_state, token_base_i))
 *
 * This enables O(D) loss computation instead of O(V) where D << V typically.
 *
 * For large vocabularies (V > 100k), this provides significant memory savings
 * by never materializing the full [batch, seq, vocab] logits tensor.
 */

#ifndef HIGHNOON_NATIVE_OPS_HOLOGRAPHIC_LOSS_OP_H_
#define HIGHNOON_NATIVE_OPS_HOLOGRAPHIC_LOSS_OP_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

namespace hsmn {
namespace holographic_loss {

// =============================================================================
// NUMERICAL STABILITY CONSTANTS (Enterprise-level guards)
// =============================================================================

// STABILITY FIX: Use 1e-6 for norm epsilon to prevent gradient explosion.
// With 1e-8, product of two norms = 1e-16 → division produces 1e16 → INF.
constexpr float kHDNormEpsilon = 1e-6f;

// Minimum log-sum-exp term to prevent -inf in log.
constexpr float kMinLogSumExp = 1e-6f;

/**
 * Holographic Loss Configuration.
 */
struct HolographicLossConfig {
    int hd_dim = 4096;           // Hyperdimensional embedding dimension
    int vocab_size = 50000;      // Vocabulary size
    float label_smoothing = 0.1f; // Label smoothing factor
    float temperature = 1.0f;    // Softmax temperature
    int num_negatives = 64;      // Number of negative samples for NCE approximation
    bool use_nce = true;         // Use NCE for large vocab
};

/**
 * Compute holographic log-probability for a single token.
 *
 * Uses cosine similarity in HD space:
 *   log P(token_i) ≈ similarity(hd_state, base_i) - log(Z)
 *
 * @param hd_state HD state vector [hd_dim]
 * @param token_base Token's HD base vector [hd_dim]
 * @param hd_dim Dimension
 * @param temperature Softmax temperature
 * @return Log-probability
 */
inline float holographic_log_prob(
    const float* hd_state,
    const float* token_base,
    int hd_dim,
    float temperature
) {
    // Compute cosine similarity
    float dot = 0.0f;
    float norm_state = 0.0f;
    float norm_token = 0.0f;

    for (int d = 0; d < hd_dim; ++d) {
        dot += hd_state[d] * token_base[d];
        norm_state += hd_state[d] * hd_state[d];
        norm_token += token_base[d] * token_base[d];
    }

    // STABILITY FIX: Use max() with kHDNormEpsilon to prevent sqrt(0) issues.
    float sqrt_norm_state = std::sqrt(std::max(norm_state, kHDNormEpsilon));
    float sqrt_norm_token = std::sqrt(std::max(norm_token, kHDNormEpsilon));

    // STABILITY FIX: Use max() on product to prevent near-zero division.
    // With individual epsilons of 1e-6, product is at least 1e-12 which is safe.
    float norm = std::max(sqrt_norm_state * sqrt_norm_token, kHDNormEpsilon);
    float similarity = dot / norm;

    // Clamp similarity to [-1, 1] for numerical stability.
    similarity = std::clamp(similarity, -1.0f, 1.0f);

    // Scale by temperature and convert to log-probability
    return similarity / std::max(temperature, kHDNormEpsilon);
}

/**
 * Compute NCE (Noise Contrastive Estimation) loss.
 *
 * Approximates full softmax by sampling negative examples:
 *   L_NCE = -log(σ(s_pos)) - sum_k log(σ(-s_neg_k))
 *
 * This reduces complexity from O(V) to O(k) where k << V.
 *
 * @param hd_state HD state vector [hd_dim]
 * @param token_bases All token HD base vectors [vocab_size, hd_dim]
 * @param target_idx Target token index
 * @param negative_indices Sampled negative indices [num_negatives]
 * @param config Configuration
 * @return NCE loss value
 */
inline float nce_loss(
    const float* hd_state,
    const float* token_bases,
    int target_idx,
    const int* negative_indices,
    const HolographicLossConfig& config
) {
    const int hd_dim = config.hd_dim;
    const int num_neg = config.num_negatives;
    const float temperature = config.temperature;

    // Positive sample score
    const float* pos_base = token_bases + target_idx * hd_dim;
    float pos_score = holographic_log_prob(hd_state, pos_base, hd_dim, temperature);

    // Sigmoid of positive: -log(σ(s)) = log(1 + exp(-s))
    float pos_loss = std::log(1.0f + std::exp(-pos_score));

    // Negative samples
    float neg_loss = 0.0f;
    for (int k = 0; k < num_neg; ++k) {
        int neg_idx = negative_indices[k];
        const float* neg_base = token_bases + neg_idx * hd_dim;
        float neg_score = holographic_log_prob(hd_state, neg_base, hd_dim, temperature);

        // -log(σ(-s)) = log(1 + exp(s))
        neg_loss += std::log(1.0f + std::exp(neg_score));
    }

    return pos_loss + neg_loss / static_cast<float>(num_neg);
}

/**
 * Compute full holographic cross-entropy (for small vocab).
 *
 * Computes exact softmax over all tokens using HD similarity.
 *
 * @param hd_state HD state vector [hd_dim]
 * @param token_bases All token HD base vectors [vocab_size, hd_dim]
 * @param target_idx Target token index
 * @param config Configuration
 * @return Cross-entropy loss value
 */
inline float full_cross_entropy(
    const float* hd_state,
    const float* token_bases,
    int target_idx,
    const HolographicLossConfig& config
) {
    const int hd_dim = config.hd_dim;
    const int vocab_size = config.vocab_size;
    const float temperature = config.temperature;
    const float smoothing = config.label_smoothing;

    // Compute all scores
    std::vector<float> scores(vocab_size);
    float max_score = -1e9f;

    for (int v = 0; v < vocab_size; ++v) {
        const float* base = token_bases + v * hd_dim;
        scores[v] = holographic_log_prob(hd_state, base, hd_dim, temperature);
        max_score = std::max(max_score, scores[v]);
    }

    // Compute log-sum-exp for normalization
    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
        sum_exp += std::exp(scores[v] - max_score);
    }
    // STABILITY FIX: Use max() to prevent log(0) = -inf.
    float log_z = max_score + std::log(std::max(sum_exp, kMinLogSumExp));

    // Cross-entropy with label smoothing
    // L = -(1 - ε) * log P(target) - ε * (1/V) * sum_v log P(v)
    float target_log_prob = scores[target_idx] - log_z;
    float uniform_log_prob = -std::log(static_cast<float>(vocab_size));

    float loss = -(1.0f - smoothing) * target_log_prob - smoothing * uniform_log_prob;

    return loss;
}

/**
 * Holographic Loss Forward Pass.
 *
 * Computes cross-entropy loss directly in HD space.
 *
 * @param hd_states HD state vectors [batch, seq_len, hd_dim]
 * @param token_bases Token HD base vectors [vocab_size, hd_dim]
 * @param targets Target token indices [batch, seq_len]
 * @param negative_samples Pre-sampled negatives [batch, seq_len, num_negatives]
 * @param losses Per-token losses [batch, seq_len]
 * @param config Configuration
 * @param batch_size Batch size
 * @param seq_len Sequence length
 */
inline void HolographicLossForward(
    const float* hd_states,
    const float* token_bases,
    const int* targets,
    const int* negative_samples,
    float* losses,
    const HolographicLossConfig& config,
    int batch_size,
    int seq_len
) {
    const int hd_dim = config.hd_dim;
    const int num_neg = config.num_negatives;
    const bool use_nce = config.use_nce;

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int idx = b * seq_len + t;
            const float* state = hd_states + idx * hd_dim;
            int target = targets[idx];

            float loss;
            if (use_nce && negative_samples != nullptr) {
                const int* neg_samples = negative_samples + idx * num_neg;
                loss = nce_loss(state, token_bases, target, neg_samples, config);
            } else {
                loss = full_cross_entropy(state, token_bases, target, config);
            }

            losses[idx] = loss;
        }
    }
}

/**
 * Holographic Loss Backward Pass.
 *
 * Computes gradients for HD states and token bases.
 *
 * @param hd_states Forward HD states [batch, seq_len, hd_dim]
 * @param token_bases Token HD base vectors [vocab_size, hd_dim]
 * @param targets Target token indices [batch, seq_len]
 * @param negative_samples Pre-sampled negatives [batch, seq_len, num_negatives]
 * @param grad_states Gradient w.r.t. HD states [batch, seq_len, hd_dim]
 * @param grad_bases Gradient w.r.t. token bases [vocab_size, hd_dim]
 * @param config Configuration
 * @param batch_size Batch size
 * @param seq_len Sequence length
 */
inline void HolographicLossBackward(
    const float* hd_states,
    const float* token_bases,
    const int* targets,
    const int* negative_samples,
    float* grad_states,
    float* grad_bases,
    const HolographicLossConfig& config,
    int batch_size,
    int seq_len
) {
    const int hd_dim = config.hd_dim;
    const int vocab_size = config.vocab_size;
    const int num_neg = config.num_negatives;
    const float temperature = config.temperature;
    const bool use_nce = config.use_nce;

    // Zero-initialize gradients
    std::memset(grad_bases, 0, vocab_size * hd_dim * sizeof(float));

    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int idx = b * seq_len + t;
            const float* state = hd_states + idx * hd_dim;
            float* g_state = grad_states + idx * hd_dim;
            int target = targets[idx];

            // Initialize state gradient to zero
            std::memset(g_state, 0, hd_dim * sizeof(float));

            if (use_nce && negative_samples != nullptr) {
                // NCE gradient
                const int* neg_samples = negative_samples + idx * num_neg;

                // Gradient from positive sample
                const float* pos_base = token_bases + target * hd_dim;
                float* g_pos_base = grad_bases + target * hd_dim;
                float pos_score = holographic_log_prob(state, pos_base, hd_dim, temperature);
                float pos_prob = 1.0f / (1.0f + std::exp(-pos_score));

                // d(L)/d(score) = σ(s) - 1 = -exp(-s)/(1+exp(-s))
                float pos_grad_coef = pos_prob - 1.0f;

                // Gradient through similarity
                float norm_state_sq = 0.0f, norm_pos_sq = 0.0f;
                for (int d = 0; d < hd_dim; ++d) {
                    norm_state_sq += state[d] * state[d];
                    norm_pos_sq += pos_base[d] * pos_base[d];
                }
                // STABILITY FIX: Use max() with kHDNormEpsilon before sqrt.
                float norm_state = std::sqrt(std::max(norm_state_sq, kHDNormEpsilon));
                float norm_pos = std::sqrt(std::max(norm_pos_sq, kHDNormEpsilon));

                // STABILITY FIX: Use max() on product and temperature.
                float norm_product = std::max(norm_state * norm_pos, kHDNormEpsilon);
                float safe_temp = std::max(temperature, kHDNormEpsilon);
                float inv_norm_temp = 1.0f / (norm_product * safe_temp);

                for (int d = 0; d < hd_dim; ++d) {
                    // d(sim)/d(state) ≈ base / (||state|| * ||base||)
                    float grad_sim_state = pos_base[d] * inv_norm_temp;
                    float grad_sim_base = state[d] * inv_norm_temp;

                    float g_s = pos_grad_coef * grad_sim_state;
                    float g_b = pos_grad_coef * grad_sim_base;

                    g_state[d] += std::isfinite(g_s) ? g_s : 0.0f;
                    g_pos_base[d] += std::isfinite(g_b) ? g_b : 0.0f;
                }

                // Gradient from negative samples
                for (int k = 0; k < num_neg; ++k) {
                    int neg_idx = neg_samples[k];
                    const float* neg_base = token_bases + neg_idx * hd_dim;
                    float* g_neg_base = grad_bases + neg_idx * hd_dim;
                    float neg_score = holographic_log_prob(state, neg_base, hd_dim, temperature);
                    float neg_prob = 1.0f / (1.0f + std::exp(-neg_score));

                    // d(L)/d(score) for negative = σ(s) = exp(s)/(1+exp(s))
                    float neg_grad_coef = neg_prob / static_cast<float>(num_neg);

                    float norm_neg_sq = 0.0f;
                    for (int d = 0; d < hd_dim; ++d) {
                        norm_neg_sq += neg_base[d] * neg_base[d];
                    }
                    // STABILITY FIX: Use max() with kHDNormEpsilon before sqrt.
                    float norm_neg = std::sqrt(std::max(norm_neg_sq, kHDNormEpsilon));

                    // STABILITY FIX: Use max() on product and temperature.
                    float norm_product_neg = std::max(norm_state * norm_neg, kHDNormEpsilon);
                    float inv_norm_temp_neg = 1.0f / (norm_product_neg * safe_temp);

                    for (int d = 0; d < hd_dim; ++d) {
                        float grad_sim_state = neg_base[d] * inv_norm_temp_neg;
                        float grad_sim_base = state[d] * inv_norm_temp_neg;

                        float g_s = neg_grad_coef * grad_sim_state;
                        float g_b = neg_grad_coef * grad_sim_base;

                        g_state[d] += std::isfinite(g_s) ? g_s : 0.0f;
                        g_neg_base[d] += std::isfinite(g_b) ? g_b : 0.0f;
                    }
                }
            } else {
                // Full softmax gradient (simplified)
                // For each token v: grad = P(v) - 1{v=target}
                std::vector<float> probs(vocab_size);
                float max_score = -1e9f;

                for (int v = 0; v < vocab_size; ++v) {
                    const float* base = token_bases + v * hd_dim;
                    probs[v] = holographic_log_prob(state, base, hd_dim, temperature);
                    max_score = std::max(max_score, probs[v]);
                }

                float sum_exp = 0.0f;
                for (int v = 0; v < vocab_size; ++v) {
                    probs[v] = std::exp(probs[v] - max_score);
                    sum_exp += probs[v];
                }
                // STABILITY FIX: Prevent division by zero.
                float inv_sum = 1.0f / std::max(sum_exp, kMinLogSumExp);
                for (int v = 0; v < vocab_size; ++v) {
                    probs[v] *= inv_sum;
                }

                // Pre-compute state norm for all gradient computations.
                float norm_s_sq = 0.0f;
                for (int d = 0; d < hd_dim; ++d) {
                    norm_s_sq += state[d] * state[d];
                }
                float norm_s = std::sqrt(std::max(norm_s_sq, kHDNormEpsilon));
                float safe_temp = std::max(temperature, kHDNormEpsilon);

                // Gradient: P(v) - 1{v=target}
                for (int v = 0; v < vocab_size; ++v) {
                    float grad_coef = probs[v] - (v == target ? 1.0f : 0.0f);
                    const float* base = token_bases + v * hd_dim;
                    float* g_base = grad_bases + v * hd_dim;

                    float norm_base_sq = 0.0f;
                    for (int d = 0; d < hd_dim; ++d) {
                        norm_base_sq += base[d] * base[d];
                    }
                    // STABILITY FIX: Use max() with kHDNormEpsilon before sqrt.
                    float norm_base = std::sqrt(std::max(norm_base_sq, kHDNormEpsilon));

                    // STABILITY FIX: Use max() on product and temperature.
                    float norm_product = std::max(norm_s * norm_base, kHDNormEpsilon);
                    float inv_norm_temp = 1.0f / (norm_product * safe_temp);

                    for (int d = 0; d < hd_dim; ++d) {
                        float grad_sim_state = base[d] * inv_norm_temp;
                        float grad_sim_base = state[d] * inv_norm_temp;

                        float g_s = grad_coef * grad_sim_state;
                        float g_b = grad_coef * grad_sim_base;

                        g_state[d] += std::isfinite(g_s) ? g_s : 0.0f;
                        g_base[d] += std::isfinite(g_b) ? g_b : 0.0f;
                    }
                }
            }
        }
    }
}

}  // namespace holographic_loss
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_HOLOGRAPHIC_LOSS_OP_H_
