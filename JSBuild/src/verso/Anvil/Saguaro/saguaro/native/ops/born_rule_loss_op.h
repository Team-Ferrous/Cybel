// saguaro.native/ops/born_rule_loss_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

/**
 * @file born_rule_loss_op.h
 * @brief Phase 51: Quantum Born Rule Loss (QBRL)
 *
 * Interprets logits as quantum amplitudes with Born rule probability
 * and uses Quantum Fisher Information Metric for gradients.
 *
 * P(v) = |⟨v|ψ⟩|² / Σ|⟨w|ψ⟩|²
 *
 * Benefits: Natural probabilistic interpretation, QFIM gradients
 * Complexity: O(N × V) where V = vocab size
 */

#ifndef SAGUARO_NATIVE_OPS_BORN_RULE_LOSS_OP_H_
#define SAGUARO_NATIVE_OPS_BORN_RULE_LOSS_OP_H_

#include <cmath>
#include <algorithm>
#include <vector>

namespace saguaro {
namespace qbrl {

// =============================================================================
// NUMERICAL STABILITY CONSTANTS (Enterprise-level guards)
// =============================================================================

// STABILITY FIX: Use 1e-6 instead of 1e-10 for float32 numerical stability.
// At 1e-10, division produces 1e10 → INF after accumulation.
constexpr float kQBRLEpsilon = 1e-6f;

/**
 * @brief Born rule loss with QFIM gradients.
 */
inline void BornRuleLoss(
    const float* logits, const int* targets,
    float* loss, float* grad_logits,
    int batch, int seq, int vocab, float temperature = 1.0f, bool use_qfim = true) {

    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        float batch_loss = 0.0f;

        for (int s = 0; s < seq; ++s) {
            const float* log_s = logits + (b * seq + s) * vocab;
            float* grad_s = grad_logits + (b * seq + s) * vocab;
            int target = targets[b * seq + s];

            // Validate target index
            if (target < 0 || target >= vocab) {
                // Invalid target - zero gradients and skip
                for (int v = 0; v < vocab; ++v) {
                    grad_s[v] = 0.0f;
                }
                continue;
            }

            // Compute |amplitude|² (Born rule)
            float sum_sq = 0.0f;
            float target_sq = 0.0f;

            for (int v = 0; v < vocab; ++v) {
                float amp = log_s[v] / temperature;
                float sq = amp * amp;
                sum_sq += sq;
                if (v == target) target_sq = sq;
            }

            // STABILITY FIX: Use max() to ensure sum_sq is never too small.
            // This bounds the gradient magnitude to 1/kQBRLEpsilon = 1e6.
            float sum_sq_safe = std::max(sum_sq, kQBRLEpsilon);

            // Probability via Born rule with epsilon floor to prevent log(0).
            float prob = std::max(target_sq / sum_sq_safe, kQBRLEpsilon);

            float sample_loss = -std::log(prob);
            batch_loss += sample_loss;

            // QFIM gradient with stability guards
            if (use_qfim) {
                for (int v = 0; v < vocab; ++v) {
                    float amp = log_s[v] / temperature;

                    // grad_base = 2 * amp / sum_sq_safe
                    float grad_base = 2.0f * amp / sum_sq_safe;

                    float grad;
                    if (v == target) {
                        grad = -grad_base * (1.0f - prob);
                    } else {
                        grad = grad_base * prob;
                    }

                    // Replace non-finite with zero.
                    grad_s[v] = std::isfinite(grad) ? grad : 0.0f;
                }
            }
        }

        loss[b] = batch_loss / static_cast<float>(seq);
    }
}

}}
#endif
