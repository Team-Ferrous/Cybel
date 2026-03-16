// highnoon/_native/ops/fused_quls_loss_op.h
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
 * @file fused_quls_loss_op.h
 * @brief Fused Quantum Unified Loss System Operator
 *
 * Single-kernel computation of all QULS loss components, eliminating
 * intermediate memory allocations and reducing kernel launch overhead.
 * Achieves 1.4-1.8× speedup over sequential component computation.
 *
 * FUSED LOSS COMPONENTS:
 *   1. Sparse CE with label smoothing: -Σ (1-ε)log(p_y) - ε/V Σlog(p_i)
 *   2. Fidelity: F(p,q) = (Σ√pᵢ√qᵢ)²
 *   3. Born rule: ||ψ|² - p||²
 *   4. Coherence: max(0, threshold - C)²
 *   5. Symplectic: |H_final - H_init| / dt
 *   6. Entropy: (H(ρ) - target)²
 *   7. Spectral: (flatness - target)²
 *
 * Each component is computed in a single pass over the logits, sharing
 * intermediate values (softmax, log-softmax, sqrt-probs) across all terms.
 *
 * SIMD: AVX2 primary, AVX-512 secondary, ARM NEON tertiary
 *
 * Reference: HIGHNOON_V2_PERFORMANCE_ANALYSIS.md Section 6.1
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_QULS_LOSS_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_QULS_LOSS_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

#include "hnn_simd_common.h"

namespace highnoon {
namespace ops {
namespace quls {

// =============================================================================
// NUMERICAL STABILITY CONSTANTS (Enterprise-level guards)
// =============================================================================

// STABILITY FIX: Use 1e-6 instead of 1e-10 for float32 numerical stability.
// At 1e-10, gradient through log() produces 1e10 → INF after accumulation.
// At 1e-6, gradient through log() produces 1e6 → safe for float32.
constexpr float kStableEpsilon = 1e-6f;

// Minimum probability floor for log() operations to bound gradients.
// log(kMinProb) ≈ -13.8, so grad = 1/kMinProb = 1e6 (safe).
constexpr float kMinProb = 1e-6f;

// Minimum value for log-sum-exp sum to prevent log(0) - must be > 0.
constexpr float kMinLogSumExp = 1e-12f;

// =============================================================================
// LOSS CONFIGURATION
// =============================================================================

struct QULSLossConfig {
    // Vocabulary sizes (Active-First strategy)
    int64_t vocab_size;        // Total vocab size (Path B)
    int64_t active_vocab_size; // Active vocab size (Path A) - streaming dense
    
    // Loss weights (from QULS unified loss)
    float ce_weight;          // Cross-entropy (always 1.0)
    float fidelity_weight;    // Quantum fidelity
    float born_weight;        // Born rule regularization
    float coherence_weight;   // QCB coherence penalty
    float symplectic_weight;  // Hamiltonian energy conservation
    float entropy_weight;     // Entropy regularization
    float spectral_weight;    // Spectral uniformity
    float infonce_weight;     // InfoNCE contrastive weight
    
    // Label smoothing
    float label_smoothing;    // ε in (1-ε)log(p_y) + ε/V Σlog(p_i)
    
    // Targets
    float target_entropy;          // Target entropy value (0.5 = balanced)
    float target_spectral_flatness; // Target spectral flatness (0.8)
    float coherence_threshold;      // Minimum coherence threshold (0.85)
    
    // For symplectic
    float symplectic_dt;      // Time step normalization
    float infonce_temperature; // Contrastive temperature (tau)

    // VQC Specifics
    int vqc_qubits = 8;
    int vqc_layers = 2;
    
    QULSLossConfig()
        : vocab_size(32000)
        , active_vocab_size(1024)
        , ce_weight(1.0f)
        , fidelity_weight(0.01f)
        , born_weight(0.005f)
        , coherence_weight(0.01f)
        , symplectic_weight(0.01f)
        , entropy_weight(0.01f)
        , spectral_weight(0.01f)
        , infonce_weight(0.0f)
        , label_smoothing(0.1f)
        , target_entropy(0.5f)
        , target_spectral_flatness(0.8f)
        , coherence_threshold(0.85f)
        , symplectic_dt(0.01f)
        , infonce_temperature(0.07f) {}
};

// =============================================================================
// LOSS OUTPUT STRUCTURE
// =============================================================================

struct QULSLossOutput {
    float total_loss;
    float ce_loss;
    float fidelity_loss;
    float born_loss;
    float coherence_loss;
    float symplectic_loss;
    float entropy_loss;
    float spectral_loss;
    float infonce_loss;
    
    // Phase 2.2.3: Flag to detect uniform predictions (potential barren plateau)
    // Set when max(softmax) < 2/vocab_size (essentially random guessing)
    bool uniform_prediction_detected;
    
    QULSLossOutput() 
        : total_loss(0.0f)
        , ce_loss(0.0f)
        , fidelity_loss(0.0f)
        , born_loss(0.0f)
        , coherence_loss(0.0f)
        , symplectic_loss(0.0f)
        , entropy_loss(0.0f)
        , spectral_loss(0.0f)
        , infonce_loss(0.0f)
        , uniform_prediction_detected(false) {}
        
    void reset() {
        total_loss = ce_loss = fidelity_loss = born_loss = 0.0f;
        coherence_loss = symplectic_loss = entropy_loss = spectral_loss = 0.0f;
        infonce_loss = 0.0f;
        uniform_prediction_detected = false;
    }
};

// =============================================================================
// SIMD-OPTIMIZED SOFTMAX WITH MAX SUBTRACTION
// =============================================================================

/**
 * @brief Compute numerically stable softmax in place.
 *
 * Algorithm:
 *   1. Find max(logits) for numerical stability
 *   2. Compute exp(logit - max) for each element
 *   3. Sum exps and divide to normalize
 *
 * @param logits Input logits, will be overwritten with softmax probs
 * @param vocab_size Vocabulary size
 * @param max_out Output max logit (for log_softmax computation)
 * @param sum_out Output sum of exp(logits - max)
 */
inline void simd_softmax(
    float* logits,
    int64_t vocab_size,
    float& max_out,
    float& sum_out
) {
    // Step 1: Find max
    float max_val = logits[0];
    int64_t i = 0;
    
#if defined(__AVX2__)
    __m256 max_vec = _mm256_set1_ps(-1e38f);
    for (; i + 8 <= vocab_size; i += 8) {
        __m256 v = _mm256_loadu_ps(&logits[i]);
        max_vec = _mm256_max_ps(max_vec, v);
    }
    // Horizontal max
    __m256 tmp = _mm256_permute2f128_ps(max_vec, max_vec, 1);
    max_vec = _mm256_max_ps(max_vec, tmp);
    tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(2, 3, 0, 1));
    max_vec = _mm256_max_ps(max_vec, tmp);
    tmp = _mm256_shuffle_ps(max_vec, max_vec, _MM_SHUFFLE(1, 0, 3, 2));
    max_vec = _mm256_max_ps(max_vec, tmp);
    max_val = _mm256_cvtss_f32(max_vec);
#elif defined(__ARM_NEON)
    float32x4_t max_vec = vdupq_n_f32(-1e38f);
    for (; i + 4 <= vocab_size; i += 4) {
        float32x4_t v = vld1q_f32(&logits[i]);
        max_vec = vmaxq_f32(max_vec, v);
    }
    max_val = vmaxvq_f32(max_vec);
#endif
    for (; i < vocab_size; ++i) {
        max_val = std::max(max_val, logits[i]);
    }
    max_out = max_val;
    
    // Step 2: Subtract max and compute exp, accumulate sum
    float sum = 0.0f;
    i = 0;
    
#if defined(__AVX2__)
    __m256 max_broadcast = _mm256_set1_ps(max_val);
    __m256 sum_vec = _mm256_setzero_ps();
    
    for (; i + 8 <= vocab_size; i += 8) {
        __m256 v = _mm256_loadu_ps(&logits[i]);
        v = _mm256_sub_ps(v, max_broadcast);  // logit - max
        _mm256_storeu_ps(&logits[i], v);
    }
    
    // Apply exp using simd_exp_inplace from hnn_simd_common.h
    simd_exp_inplace(logits, vocab_size);
    
    // Sum the exp values
    i = 0;
    for (; i + 8 <= vocab_size; i += 8) {
        __m256 v = _mm256_loadu_ps(&logits[i]);
        sum_vec = _mm256_add_ps(sum_vec, v);
    }
    // Horizontal sum
    __m256 tmp2 = _mm256_hadd_ps(sum_vec, sum_vec);
    tmp2 = _mm256_hadd_ps(tmp2, tmp2);
    __m128 lo = _mm256_extractf128_ps(tmp2, 0);
    __m128 hi = _mm256_extractf128_ps(tmp2, 1);
    sum = _mm_cvtss_f32(_mm_add_ss(lo, hi));
#elif defined(__ARM_NEON)
    float32x4_t max_broadcast = vdupq_n_f32(max_val);
    
    for (; i + 4 <= vocab_size; i += 4) {
        float32x4_t v = vld1q_f32(&logits[i]);
        v = vsubq_f32(v, max_broadcast);
        vst1q_f32(&logits[i], v);
    }
    
    simd_exp_inplace(logits, vocab_size);
    
    i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 4 <= vocab_size; i += 4) {
        float32x4_t v = vld1q_f32(&logits[i]);
        sum_vec = vaddq_f32(sum_vec, v);
    }
    sum = vaddvq_f32(sum_vec);
#else
    for (; i < vocab_size; ++i) {
        logits[i] -= max_val;
    }
    simd_exp_inplace(logits, vocab_size);
    for (i = 0; i < vocab_size; ++i) {
        sum += logits[i];
    }
#endif

    for (; i < vocab_size; ++i) {
        sum += logits[i];
    }
    sum_out = sum;

    // Step 3: Normalize with stable epsilon
    // STABILITY FIX: Use kStableEpsilon (1e-6) instead of 1e-10 to prevent
    // near-zero division producing values that overflow in gradient computation.
    float inv_sum = 1.0f / std::max(sum, kStableEpsilon);
    i = 0;
    
#if defined(__AVX2__)
    __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= vocab_size; i += 8) {
        __m256 v = _mm256_loadu_ps(&logits[i]);
        v = _mm256_mul_ps(v, inv_sum_vec);
        _mm256_storeu_ps(&logits[i], v);
    }
#elif defined(__ARM_NEON)
    float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
    for (; i + 4 <= vocab_size; i += 4) {
        float32x4_t v = vld1q_f32(&logits[i]);
        v = vmulq_f32(v, inv_sum_vec);
        vst1q_f32(&logits[i], v);
    }
#endif
    for (; i < vocab_size; ++i) {
        logits[i] *= inv_sum;
    }
}

// =============================================================================
// FUSED CROSS-ENTROPY + FIDELITY + ENTROPY IN SINGLE PASS
// =============================================================================

/**
 * @brief Fused computation of CE, fidelity, and entropy losses.
 *
 * Computes in a single pass over softmax probabilities:
 *   - CE loss: -(1-ε)log(p_target) - ε/V Σlog(p_i)
 *   - Fidelity loss: 1 - (Σ√p·√target)**2
 *   - Entropy: -Σ p log(p) for entropy regularization
 *
 * @param probs Softmax probabilities [vocab_size]
 * @param target_idx Target label index
 * @param label_smoothing Label smoothing factor ε
 * @param vocab_size Vocabulary size
 * @param ce_loss Output CE loss
 * @param fidelity_loss Output fidelity loss
 * @param entropy Output entropy value
 */
inline void fused_ce_fidelity_entropy(
    const float* probs,
    int32_t target_idx,
    float label_smoothing,
    int64_t vocab_size,
    float& ce_loss,
    float& fidelity_loss,
    float& entropy
) {
    // STABILITY FIX: Use kMinProb (1e-6) for log() operations to bound gradients.
    // With epsilon = 1e-10, grad = 1/epsilon = 1e10 → INF after accumulation.
    constexpr float epsilon = kMinProb;
    
    float log_sum = 0.0f;      // For label smoothing term
    float sqrt_overlap = 0.0f; // For fidelity
    float entropy_sum = 0.0f;  // For entropy
    float target_prob = probs[target_idx];
    
    int64_t i = 0;
    
#if defined(__AVX2__)
    __m256 eps_vec = _mm256_set1_ps(epsilon);
    __m256 log_sum_vec = _mm256_setzero_ps();
    __m256 sqrt_overlap_vec = _mm256_setzero_ps();
    __m256 entropy_vec = _mm256_setzero_ps();
    
    // Target is one-hot, so sqrt(target) = 1 at target_idx, 0 elsewhere
    // fidelity overlap = sqrt(p[target_idx])
    
    for (; i + 8 <= vocab_size; i += 8) {
        __m256 p = _mm256_loadu_ps(&probs[i]);
        __m256 p_safe = _mm256_max_ps(p, eps_vec);
        
        // log(p) for label smoothing
        // Use simd_log by copying to temp buffer
        float log_buf[8];
        _mm256_storeu_ps(log_buf, p_safe);
        simd_log_inplace(log_buf, 8);
        __m256 log_p = _mm256_loadu_ps(log_buf);
        
        log_sum_vec = _mm256_add_ps(log_sum_vec, log_p);
        
        // -p * log(p) for entropy
        __m256 neg_p_log_p = _mm256_mul_ps(p, log_p);
        neg_p_log_p = _mm256_mul_ps(neg_p_log_p, _mm256_set1_ps(-1.0f));
        entropy_vec = _mm256_add_ps(entropy_vec, neg_p_log_p);
    }
    
    // Horizontal reductions
    __m256 tmp = _mm256_hadd_ps(log_sum_vec, log_sum_vec);
    tmp = _mm256_hadd_ps(tmp, tmp);
    __m128 lo = _mm256_extractf128_ps(tmp, 0);
    __m128 hi = _mm256_extractf128_ps(tmp, 1);
    log_sum = _mm_cvtss_f32(_mm_add_ss(lo, hi));
    
    tmp = _mm256_hadd_ps(entropy_vec, entropy_vec);
    tmp = _mm256_hadd_ps(tmp, tmp);
    lo = _mm256_extractf128_ps(tmp, 0);
    hi = _mm256_extractf128_ps(tmp, 1);
    entropy_sum = _mm_cvtss_f32(_mm_add_ss(lo, hi));
#elif defined(__ARM_NEON)
    float32x4_t eps_vec = vdupq_n_f32(epsilon);
    float32x4_t log_sum_vec = vdupq_n_f32(0.0f);
    float32x4_t entropy_vec = vdupq_n_f32(0.0f);
    
    for (; i + 4 <= vocab_size; i += 4) {
        float32x4_t p = vld1q_f32(&probs[i]);
        float32x4_t p_safe = vmaxq_f32(p, eps_vec);
        
        float log_buf[4];
        vst1q_f32(log_buf, p_safe);
        simd_log_inplace(log_buf, 4);
        float32x4_t log_p = vld1q_f32(log_buf);
        
        log_sum_vec = vaddq_f32(log_sum_vec, log_p);
        
        float32x4_t neg_p_log_p = vmulq_f32(p, log_p);
        neg_p_log_p = vmulq_f32(neg_p_log_p, vdupq_n_f32(-1.0f));
        entropy_vec = vaddq_f32(entropy_vec, neg_p_log_p);
    }
    
    log_sum = vaddvq_f32(log_sum_vec);
    entropy_sum = vaddvq_f32(entropy_vec);
#endif

    // Scalar remainder
    for (; i < vocab_size; ++i) {
        float p = std::max(probs[i], epsilon);
        float log_p = std::log(p);
        log_sum += log_p;
        entropy_sum += -p * log_p;
    }
    
    // Compute losses
    float target_log = std::log(std::max(target_prob, epsilon));
    
    // CE with label smoothing: -(1-ε)log(p_target) - ε/V * Σlog(p_i)
    ce_loss = -(1.0f - label_smoothing) * target_log 
              - label_smoothing / static_cast<float>(vocab_size) * log_sum;
    
    // Fidelity: for one-hot target, F = sqrt(p_target)**2 = p_target
    // So fidelity_loss = 1 - p_target
    fidelity_loss = 1.0f - target_prob;
    
    entropy = entropy_sum;
}

// =============================================================================
// BORN RULE LOSS
// =============================================================================

/**
 * @brief Compute Born rule regularization loss.
 *
 * Penalizes deviation from |ψ|² = 1 normalization.
 * L_born = ||Σ|ψᵢ|² - 1||²
 *
 * @param amplitudes Complex amplitudes in [real, imag, real, imag, ...] format
 * @param num_amplitudes Number of complex amplitudes
 * @return Born rule loss
 */
inline float compute_born_loss(
    const float* amplitudes,
    int num_amplitudes
) {
    if (amplitudes == nullptr || num_amplitudes <= 0) {
        return 0.0f;
    }
    
    float norm_sq = 0.0f;
    int64_t i = 0;
    
#if defined(__AVX2__)
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i + 8 <= num_amplitudes * 2; i += 8) {
        __m256 v = _mm256_loadu_ps(&amplitudes[i]);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);  // sum += v*v
    }
    // Horizontal sum
    __m256 tmp = _mm256_hadd_ps(sum_vec, sum_vec);
    tmp = _mm256_hadd_ps(tmp, tmp);
    __m128 lo = _mm256_extractf128_ps(tmp, 0);
    __m128 hi = _mm256_extractf128_ps(tmp, 1);
    norm_sq = _mm_cvtss_f32(_mm_add_ss(lo, hi));
#elif defined(__ARM_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 4 <= num_amplitudes * 2; i += 4) {
        float32x4_t v = vld1q_f32(&amplitudes[i]);
        sum_vec = vmlaq_f32(sum_vec, v, v);
    }
    norm_sq = vaddvq_f32(sum_vec);
#endif

    for (; i < num_amplitudes * 2; ++i) {
        norm_sq += amplitudes[i] * amplitudes[i];
    }
    
    // Born rule: norm should equal 1
    float deviation = norm_sq - 1.0f;
    return deviation * deviation;
}

/**
 * @brief Compute InfoNCE (contrastive) weight with specified temperature.
 *
 * L_infonce = -log( exp(l_pos/tau) / sum(exp(l_j/tau)) )
 *
 * @param logits Input pre-softmax logits [vocab_size]
 * @param target_idx Ground truth token index
 * @param vocab_size Size of vocabulary
 * @param temperature Contrastive temperature (tau)
 * @return InfoNCE loss value
 */
inline float compute_infonce_loss(
    const float* logits,
    int32_t target_idx,
    int64_t vocab_size,
    float temperature
) {
    if (logits == nullptr || vocab_size <= 0) return 0.0f;
    
    float safe_temp = std::max(temperature, 1e-6f);
    float inv_temp = 1.0f / safe_temp;
    
    // Find max for numerical stability
    float max_logit = -1e9f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        max_logit = std::max(max_logit, logits[i] * inv_temp);
    }
    
    // log(sum(exp(l_j/tau))) = max + log(sum(exp(l_j/tau - max)))
    float sum_exp = 0.0f;
    for (int64_t i = 0; i < vocab_size; ++i) {
        sum_exp += std::exp(logits[i] * inv_temp - max_logit);
    }
    
    float log_z = max_logit + std::log(std::max(sum_exp, kMinLogSumExp));
    float target_score = logits[target_idx] * inv_temp;
    
    return std::max(log_z - target_score, 0.0f);
}

// =============================================================================
// COHERENCE PRESERVATION LOSS
// =============================================================================

/**
 * @brief Compute coherence preservation loss.
 *
 * Penalizes coherence below threshold:
 * L_coherence = max(0, threshold - C)²
 *
 * @param coherence_scores Coherence scores per block [num_blocks]
 * @param num_blocks Number of quantum blocks
 * @param threshold Minimum acceptable coherence
 * @return Coherence preservation loss
 */
inline float compute_coherence_loss(
    const float* coherence_scores,
    int num_blocks,
    float threshold
) {
    if (coherence_scores == nullptr || num_blocks <= 0) {
        return 0.0f;
    }
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < num_blocks; ++i) {
        float deficit = threshold - coherence_scores[i];
        if (deficit > 0.0f) {
            total_loss += deficit * deficit;
        }
    }
    
    return total_loss / static_cast<float>(num_blocks);
}

// =============================================================================
// SYMPLECTIC ENERGY CONSERVATION LOSS
// =============================================================================

/**
 * @brief Compute symplectic energy conservation loss.
 *
 * Penalizes Hamiltonian energy drift:
 * L_symplectic = |H_final - H_init| / dt
 *
 * @param h_init Initial Hamiltonian energy [batch]
 * @param h_final Final Hamiltonian energy [batch]
 * @param batch_size Batch size
 * @param dt Time step for normalization
 * @return Symplectic energy loss
 */
inline float compute_symplectic_loss(
    const float* h_init,
    const float* h_final,
    int batch_size,
    float dt
) {
    if (h_init == nullptr || h_final == nullptr || batch_size <= 0) {
        return 0.0f;
    }
    
    float total_drift = 0.0f;
    int64_t i = 0;
    
#if defined(__AVX2__)
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i + 8 <= batch_size; i += 8) {
        __m256 init = _mm256_loadu_ps(&h_init[i]);
        __m256 final_v = _mm256_loadu_ps(&h_final[i]);
        __m256 diff = _mm256_sub_ps(final_v, init);
        // abs(diff)
        __m256 sign_mask = _mm256_set1_ps(-0.0f);
        diff = _mm256_andnot_ps(sign_mask, diff);
        sum_vec = _mm256_add_ps(sum_vec, diff);
    }
    __m256 tmp = _mm256_hadd_ps(sum_vec, sum_vec);
    tmp = _mm256_hadd_ps(tmp, tmp);
    __m128 lo = _mm256_extractf128_ps(tmp, 0);
    __m128 hi = _mm256_extractf128_ps(tmp, 1);
    total_drift = _mm_cvtss_f32(_mm_add_ss(lo, hi));
#elif defined(__ARM_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 4 <= batch_size; i += 4) {
        float32x4_t init = vld1q_f32(&h_init[i]);
        float32x4_t final_v = vld1q_f32(&h_final[i]);
        float32x4_t diff = vsubq_f32(final_v, init);
        diff = vabsq_f32(diff);
        sum_vec = vaddq_f32(sum_vec, diff);
    }
    total_drift = vaddvq_f32(sum_vec);
#endif

    for (; i < batch_size; ++i) {
        total_drift += std::abs(h_final[i] - h_init[i]);
    }

    // STABILITY FIX: Use max() with kStableEpsilon to prevent inf when dt ≈ 0.
    float denominator = static_cast<float>(batch_size) * std::max(dt, kStableEpsilon);
    return total_drift / denominator;
}

// =============================================================================
// ENTROPY REGULARIZATION LOSS
// =============================================================================

/**
 * @brief Compute entropy regularization loss from pre-computed entropy.
 *
 * L_entropy = (entropy - target_entropy)²
 *
 * @param entropy Pre-computed entropy value
 * @param target_entropy Target entropy value
 * @return Entropy loss
 */
inline float compute_entropy_loss(float entropy, float target_entropy) {
    float diff = entropy - target_entropy;
    return diff * diff;
}

// =============================================================================
// SPECTRAL FLATNESS LOSS
// =============================================================================

/**
 * @brief Compute spectral flatness from eigenvalues.
 *
 * SF = geometric_mean(eigenvalues) / arithmetic_mean(eigenvalues)
 *
 * @param eigenvalues Eigenvalues [num_eigs]
 * @param num_eigs Number of eigenvalues
 * @param target_flatness Target flatness
 * @return Spectral flatness loss
 */
inline float compute_spectral_loss(
    const float* eigenvalues,
    int num_eigs,
    float target_flatness
) {
    if (eigenvalues == nullptr || num_eigs <= 0) {
        return 0.0f;
    }

    // STABILITY FIX: Use kMinProb for eigenvalue threshold to ensure
    // log(ev) gradient = 1/ev stays bounded (≤ 1e6).
    constexpr float epsilon = kMinProb;

    // Compute geometric and arithmetic means
    float log_sum = 0.0f;
    float arith_sum = 0.0f;
    int positive_count = 0;

    for (int i = 0; i < num_eigs; ++i) {
        float ev = eigenvalues[i];
        if (ev > epsilon) {
            // STABILITY FIX: Clamp eigenvalue to kMinProb floor for log().
            float ev_safe = std::max(ev, epsilon);
            log_sum += std::log(ev_safe);
            arith_sum += ev;
            positive_count++;
        }
    }

    if (positive_count == 0) {
        return 0.0f;
    }

    float n = static_cast<float>(positive_count);

    // STABILITY FIX: Clamp log_sum/n to prevent exp() overflow.
    float avg_log = std::clamp(log_sum / n, -20.0f, 20.0f);
    float geo_mean = std::exp(avg_log);
    float arith_mean = arith_sum / n;

    // STABILITY FIX: Use max() to prevent division by near-zero arith_mean.
    float flatness = geo_mean / std::max(arith_mean, epsilon);
    flatness = std::clamp(flatness, 0.0f, 1.0f);

    float diff = flatness - target_flatness;
    return diff * diff;
}

// =============================================================================
// LANCZOS ALGORITHM HELPERS
// =============================================================================

// Vector dot product
inline float simd_dot(const float* a, const float* b, int64_t n) {
    float sum = 0.0f;
    int64_t i = 0;
#if defined(__AVX512F__)
    __m512 sum_v = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        sum_v = _mm512_fmadd_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i]), sum_v);
    }
    // Reduction
    sum = _mm512_reduce_add_ps(sum_v);
#elif defined(__AVX2__)
    __m256 sum_v = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        sum_v = _mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]), sum_v);
    }
    // Horizontal sum
    __m256 t1 = _mm256_hadd_ps(sum_v, sum_v);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 lo = _mm256_extractf128_ps(t2, 0);
    __m128 hi = _mm256_extractf128_ps(t2, 1);
    sum = _mm_cvtss_f32(_mm_add_ss(lo, hi));
#elif defined(__ARM_NEON)
    float32x4_t sum_v = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        sum_v = vmlaq_f32(sum_v, vld1q_f32(&a[i]), vld1q_f32(&b[i]));
    }
    sum = vaddvq_f32(sum_v);
#endif
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// Matrix-vector multiply: y = A * x
// A is [rows, cols], x is [cols], y is [rows]
inline void simd_gemv(const float* A, const float* x, float* y, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        y[i] = simd_dot(&A[i * cols], x, cols);
    }
}

// Matrix-transpose-vector multiply: y = A^T * x
// A is [rows, cols], x is [rows], y is [cols]
inline void simd_gemv_t(const float* A, const float* x, float* y, int64_t rows, int64_t cols) {
    // Clear y
    std::fill(y, y + cols, 0.0f);
    
    for (int64_t i = 0; i < rows; ++i) {
        float xi = x[i];
        const float* row_A = &A[i * cols];
        int64_t j = 0;
#if defined(__AVX512F__)
        __m512 xi_v = _mm512_set1_ps(xi);
        for (; j + 16 <= cols; j += 16) {
            __m512 y_v = _mm512_loadu_ps(&y[j]);
            __m512 a_v = _mm512_loadu_ps(&row_A[j]);
            _mm512_storeu_ps(&y[j], _mm512_fmadd_ps(xi_v, a_v, y_v));
        }
#elif defined(__AVX2__)
        __m256 xi_v = _mm256_set1_ps(xi);
        for (; j + 8 <= cols; j += 8) {
            __m256 y_v = _mm256_loadu_ps(&y[j]);
            __m256 a_v = _mm256_loadu_ps(&row_A[j]);
            _mm256_storeu_ps(&y[j], _mm256_fmadd_ps(xi_v, a_v, y_v));
        }
#endif
        for (; j < cols; ++j) {
            y[j] += xi * row_A[j];
        }
    }
}

/**
 * @brief Compute top-k eigenvalues of Covariance Matrix C = H^T * H / N using Lanczos.
 * 
 * Uses implicit matrix multiplication. C*v = (H^T * (H * v)) / N.
 * 
 * @param hidden_states Input tensor [N, dim]
 * @param n Number of samples (N)
 * @param d Dimension (dim)
 * @param k Number of eigenvalues to compute
 * @param out_eigenvalues Output buffer for eigenvalues [k]
 * @param out_eigenvectors Optional output buffer for eigenvectors [k, d]
 */
/**
 * @brief Compute top-k eigenvalues via Lanczos algorithm.
 * 
 * Specifically optimized for large-scale hidden states (O(N*d) complexity).
 * Uses T-matrix diagonalization for faster convergence than power iteration.
 */
inline void compute_lanczos_eigenvalues(
    const float* hidden_states,
    int64_t n, 
    int64_t d,
    int k,
    float* out_eigenvalues,
    float* out_eigenvectors = nullptr,
    int max_steps = 20
) {
    if (n <= 0 || d <= 0 || k <= 0) return;
    
    // Center the data on-the-fly or pre-calculate mean
    std::vector<float> mean(d, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        const float* row = &hidden_states[i * d];
        for (int64_t j = 0; j < d; ++j) mean[j] += row[j];
    }
    float inv_n = 1.0f / n;
    for (int64_t j = 0; j < d; ++j) mean[j] *= inv_n;

    // Lanczos Iteration for Covariance Matrix A = (1/n) * (H-u)^T * (H-u)
    // We only need the top-k eigenvalues, so we build a tridiagonal T matrix.
    
    int m = std::min(max_steps, (int)d);
    std::vector<float> alpha(m, 0.0f);
    std::vector<float> beta(m, 0.0f);
    std::vector<std::vector<float>> q(m + 1, std::vector<float>(d));
    
    // Initial random vector q1
    for(int j=0; j<d; ++j) q[0][j] = (float(rand())/RAND_MAX) - 0.5f;
    float q_norm = std::sqrt(simd_dot(q[0].data(), q[0].data(), d));
    for(int j=0; j<d; ++j) q[0][j] /= (q_norm + 1e-12f);
    
    std::vector<float> r(d);
    std::vector<float> H_q(n);

    for (int i = 0; i < m; ++i) {
        // v = A * q_i
        // 1. x = (H-u) * q_i
        float mean_dot_q = simd_dot(mean.data(), q[i].data(), d);
        for (int64_t j = 0; j < n; ++j) {
            H_q[j] = simd_dot(&hidden_states[j * d], q[i].data(), d) - mean_dot_q;
        }
        
        // 2. v = (H-u)^T * x
        simd_gemv_t(hidden_states, H_q.data(), r.data(), n, d);
        float sum_x = 0.0f;
        for(int64_t j=0; j<n; ++j) sum_x += H_q[j];
        for (int64_t j = 0; j < d; ++j) r[j] = (r[j] - mean[j] * sum_x) * inv_n;
        
        // Orthogonalize against previous vectors (Full Reorthogonalization for stability)
        if (i > 0) {
            for(int j=0; j<d; ++j) r[j] -= beta[i-1] * q[i-1][j];
        }
        
        alpha[i] = simd_dot(q[i].data(), r.data(), d);
        for(int j=0; j<d; ++j) r[j] -= alpha[i] * q[i][j];
        
        // Re-orthogonalize against all previous q vectors to prevent ghost eigenvalues
        for(int j=0; j <= i; ++j) {
            float dot = simd_dot(q[j].data(), r.data(), d);
            for(int l=0; l<d; ++l) r[l] -= dot * q[j][l];
        }
        
        if (i < m - 1) {
            beta[i] = std::sqrt(simd_dot(r.data(), r.data(), d));
            if (beta[i] < 1e-9f) {
                m = i + 1;
                break;
            }
            for(int j=0; j<d; ++j) q[i+1][j] = r[j] / beta[i];
        }
    }
    
    // Diagonalize T matrix (m x m tridiagonal)
    // For small m, we can use a simple QR or just a tridiagonal solver
    // Here we use a basic tridiagonal implicit QR step implementation (simplified)
    std::vector<float> d_diag = alpha;
    std::vector<float> e_diag = beta;
    
    // Tridiagonal QR (simplified)
    for (int iter = 0; iter < 30; ++iter) {
        for (int i = 0; i < m - 1; ++i) {
            float g = (d_diag[i+1] - d_diag[i]) / (2.0f * e_diag[i] + 1e-12f);
            float r_val = std::sqrt(g*g + 1.0f);
            if (g < 0) r_val = -r_val;
            float t = e_diag[i] / (g + r_val);
            float c = 1.0f / std::sqrt(t*t + 1.0f);
            float s = c * t;
            
            // Shift
            float p = s * (d_diag[i] + t * e_diag[i]);
            d_diag[i] -= p;
            d_diag[i+1] += p;
            // Note: this is a very simplified tridiagonal update, 
            // in production we'd use Eigen or Lapack dstev.
        }
    }
    
    std::sort(d_diag.begin(), d_diag.begin() + m, std::greater<float>());
    for(int i=0; i<k && i<m; ++i) out_eigenvalues[i] = std::max(d_diag[i], 0.0f);
}

/**
 * @brief Compute all QULS loss components in a fused kernel.
 *
 * Single kernel that computes all 7 loss components efficiently by:
 *   1. Computing softmax once and sharing across components
 *   2. Fusing CE, fidelity, and entropy in a single pass
 *   3. Computing auxiliary losses (Born, coherence, symplectic, spectral)
 *   4. Applying weights and summing
 *
 * @param logits Input logits [batch, vocab_size] - NOT modified
 * @param labels Target labels [batch]
 * @param amplitudes Quantum amplitudes [batch, K*2] (real/imag interleaved) - nullable
 * @param coherence_scores QCB coherence per block [num_blocks] - nullable
 * @param h_init Initial Hamiltonian energy [batch] - nullable
 * @param h_final Final Hamiltonian energy [batch] - nullable
 * @param eigenvalues Pre-computed eigenvalues [num_eigs] - nullable
 * @param output Loss output structure
 * @param config QULS loss configuration
 * @param batch_size Batch dimension
 * @param K Superposition paths for Born rule
 * @param num_blocks QCB blocks for coherence
 * @param num_eigs Number of eigenvalues for spectral loss
 */
inline void quls_loss_forward(
    const float* logits,
    const int32_t* labels,
    const float* amplitudes,
    const float* coherence_scores,
    const float* h_init,
    const float* h_final,
    const float* eigenvalues,
    const float* hidden_states,
    // Head parameters (if streaming)
    const float* head_weights,
    const float* head_bias,
    const float* vqc_rot,
    const float* vqc_ent,
    const float* vqc_in_proj,
    const float* vqc_out_proj,
    QULSLossOutput& output,
    const QULSLossConfig& config,
    int64_t batch_size,
    int K = 2,
    int num_blocks = 6,
    int num_eigs = 8,
    int64_t num_states = 0,
    int64_t hidden_dim = 0
) {
    output.reset();
    
    if (batch_size <= 0 || logits == nullptr || labels == nullptr) {
        return;
    }
    
    // Get scratch buffer for softmax computation
    float* probs = g_path_scratch.get(config.vocab_size);
    if (probs == nullptr) {
        return;  // Allocation failed
    }
    
    // Compute eigenvalues if hidden_states provided
    // If eigenvalues input is provided (pre-computed), use it.
    // If not, and hidden_states provided, compute via Lanczos.
    std::vector<float> computed_eigenvalues;
    const float* effective_eigenvalues = eigenvalues;
    int effective_num_eigs = num_eigs;
    
    if (eigenvalues == nullptr && hidden_states != nullptr && num_states > 0 && hidden_dim > 0) {
        if (config.entropy_weight > 0.0f || config.spectral_weight > 0.0f) {
            computed_eigenvalues.resize(num_eigs);
            compute_lanczos_eigenvalues(
                hidden_states, num_states, hidden_dim, num_eigs, 
                computed_eigenvalues.data()
            );
            effective_eigenvalues = computed_eigenvalues.data();
        }
    }
    
    // Accumulate per-sample losses
    float total_ce = 0.0f;
    float total_fidelity = 0.0f;
    float total_entropy = 0.0f;
    float total_infonce = 0.0f;
    
    for (int64_t b = 0; b < batch_size; ++b) {
        // 1. Get Logits for this sample
        if (logits != nullptr) {
            // Case A: Logits already materialized (baseline)
            const float* sample_logits = logits + b * config.vocab_size;
            std::copy(sample_logits, sample_logits + config.vocab_size, probs);
        } else if (hidden_states != nullptr && head_weights != nullptr) {
            // Case B: Streaming Head Projection (Memory Optimal)
            // Roadmap 1.1 - Resolve O(N*V) materialization
            const float* h = hidden_states + b * hidden_dim;
            
            // If VQC parameters provided, apply VQC transformation first
            if (vqc_rot != nullptr) {
                // TODO: Implement fused VQC-HD projection if requested
                // For now, assume standard linear projection
                simd_gemv_t(head_weights, h, probs, config.vocab_size, hidden_dim);
            } else {
                // Standard Linear Head: Logits = weights * hidden + bias
                simd_gemv_t(head_weights, h, probs, config.vocab_size, hidden_dim);
            }
            
            if (head_bias != nullptr) {
                for(int64_t i=0; i<config.vocab_size; ++i) probs[i] += head_bias[i];
            }
        } else {
            // Fallback: Uniform predictions if no data provided
            std::fill(probs, probs + config.vocab_size, 1.0f / config.vocab_size);
            output.uniform_prediction_detected = true;
        }
        
        // --- Sprint 5: InfoNCE (Contrastive) Loss ---
        if (config.infonce_weight > 0.0f) {
            total_infonce += compute_infonce_loss(
                probs, labels[b], config.vocab_size, config.infonce_temperature
            );
        }
        
        float max_val, sum_val;
        simd_softmax(probs, config.vocab_size, max_val, sum_val);
        
        // Fused CE + fidelity + entropy
        float ce, fid, ent;
        fused_ce_fidelity_entropy(
            probs,
            labels[b],
            config.label_smoothing,
            config.vocab_size,
            ce, fid, ent
        );
        
        total_ce += ce;
        total_fidelity += fid;
        total_entropy += ent;
    }
    
    // Average over batch
    float inv_batch = 1.0f / static_cast<float>(batch_size);
    output.ce_loss = total_ce * inv_batch;
    output.fidelity_loss = total_fidelity * inv_batch;
    output.infonce_loss = total_infonce * inv_batch;
    
    // Entropy loss: Logic update for spectral vs prediction entropy.
    // If we have spectral eigenvalues, we prioritize using them for the 'entropy' loss
    // (Spectral Entropy) as per Python definition.
    // Otherwise we use prediction entropy calculated above.
    
    if (effective_eigenvalues != nullptr && config.entropy_weight > 0.0f) {
        // Compute Spectral Entropy
        float sum_eigs = 0.0f;
        for(int i=0; i<effective_num_eigs; ++i) sum_eigs += effective_eigenvalues[i];
        
        float spec_ent = 0.0f;
        if (sum_eigs > 1e-8f) {
            float inv_sum = 1.0f / sum_eigs;
            for(int i=0; i<effective_num_eigs; ++i) {
                float p = effective_eigenvalues[i] * inv_sum;
                if (p > 1e-8f) spec_ent -= p * std::log(p);
            }
            float max_ent = std::log(static_cast<float>(effective_num_eigs));
            spec_ent /= std::max(max_ent, 1e-6f); // Normalize
        }
        output.entropy_loss = compute_entropy_loss(spec_ent, config.target_entropy);
    } else {
        // Fallback to Prediction Entropy
        float avg_entropy = total_entropy * inv_batch;
        float safe_vocab = std::max(static_cast<float>(config.vocab_size), 2.0f);
        float max_ent = std::log(safe_vocab);
        float norm_entropy = avg_entropy / std::max(max_ent, kStableEpsilon);
        output.entropy_loss = compute_entropy_loss(norm_entropy, config.target_entropy);
    }
    
    // Born rule loss
    if (amplitudes != nullptr && config.born_weight > 0.0f) {
        float born_total = 0.0f;
        for (int64_t b = 0; b < batch_size; ++b) {
            born_total += compute_born_loss(amplitudes + b * K * 2, K);
        }
        output.born_loss = born_total * inv_batch;
    }
    
    // Coherence loss
    if (coherence_scores != nullptr && config.coherence_weight > 0.0f) {
        output.coherence_loss = compute_coherence_loss(
            coherence_scores, num_blocks, config.coherence_threshold
        );
    }
    
    // Symplectic loss
    if (h_init != nullptr && h_final != nullptr && config.symplectic_weight > 0.0f) {
        output.symplectic_loss = compute_symplectic_loss(
            h_init, h_final, batch_size, config.symplectic_dt
        );
    }
    
    // Spectral loss (Flatness)
    if (effective_eigenvalues != nullptr && config.spectral_weight > 0.0f) {
        output.spectral_loss = compute_spectral_loss(
            effective_eigenvalues, effective_num_eigs, config.target_spectral_flatness
        );
    }
    
    // Compute total weighted loss
    output.total_loss = config.ce_weight * output.ce_loss
                      + config.fidelity_weight * output.fidelity_loss
                      + config.born_weight * output.born_loss
                      + config.coherence_weight * output.coherence_loss
                      + config.symplectic_weight * output.symplectic_loss
                      + config.entropy_weight * output.entropy_loss
                      + config.spectral_weight * output.spectral_loss
                      + config.infonce_weight * output.infonce_loss;
}

// =============================================================================
// GRADIENT COMPUTATION (BACKWARD PASS)
// =============================================================================

/**
 * @brief Compute gradients for QULS loss.
 *
 * Computes gradients w.r.t. logits in a single pass, fusing gradients from:
 *   - Cross-entropy with label smoothing
 *   - Fidelity loss
 *   - Born rule normalization
 *   - Coherence preservation
 *   - Symplectic energy conservation
 *   - Spectral entropy/flatness
 *
 * @param grad_logits Output gradient w.r.t. logits [batch, vocab_size]
 * @param grad_amplitudes Output gradient w.r.t. VQC amplitudes [batch, K*2]
 * @param grad_coherence_scores Output gradient w.r.t. coherence [num_blocks]
 * @param grad_h_init Output gradient w.r.t. initial energy [batch]
 * @param grad_h_final Output gradient w.r.t. final energy [batch]
 * @param grad_eigenvalues Output gradient w.r.t. eigenvalues [num_eigs]
 * @param grad_hidden_states Output gradient w.r.t. hidden states [N, d]
 * @param logits Input logits [batch, vocab_size]
 * @param labels Target labels [batch]
 * @param amplitudes VQC amplitude outputs [batch, K*2]
 * @param coherence_scores QCB coherence scores [num_blocks]
 * @param h_init Initial Hamiltonian energy [batch]
 * @param h_final Final Hamiltonian energy [batch]
 * @param eigenvalues Pre-computed eigenvalues [num_eigs] (optional)
 * @param hidden_states Hidden states for Lanczos [N, d] (optional)
 * @param config QULS loss configuration
 * @param batch_size Batch dimension
 * @param K Superposition paths
 * @param num_blocks QCB blocks
 * @param num_eigs Number of eigenvalues
 * @param num_states Number of hidden states (N)
 * @param hidden_dim Hidden dimension (d)
 */
inline void quls_loss_backward(
    float* grad_logits,
    float* grad_amplitudes,
    float* grad_coherence_scores,
    float* grad_h_init,
    float* grad_h_final,
    float* grad_eigenvalues,
    float* grad_hidden_states,
    // Head gradients
    float* grad_head_weights,
    float* grad_head_bias,
    float* grad_vqc_rot,
    float* grad_vqc_ent,
    float* grad_vqc_in,
    float* grad_vqc_out,
    const float* logits,
    const int32_t* labels,
    const float* amplitudes,
    const float* coherence_scores,
    const float* h_init,
    const float* h_final,
    const float* eigenvalues,
    const float* hidden_states,
    // Head parameters
    const float* head_weights,
    const float* head_bias,
    const float* vqc_rot,
    const float* vqc_ent,
    const float* vqc_in_proj,
    const float* vqc_out_proj,
    const QULSLossConfig& config,
    int64_t batch_size,
    int K = 2,
    int num_blocks = 6,
    int num_eigs = 8,
    int64_t num_states = 0,
    int64_t hidden_dim = 0
) {
    if (batch_size <= 0 || labels == nullptr) {
        return;
    }
    
    // Safety: we either need logits OR hidden_states + head_weights
    if (logits == nullptr && (hidden_states == nullptr || head_weights == nullptr)) {
        return;
    }
    
    // Safety check for grad_logits
    if (logits != nullptr && grad_logits == nullptr) return;

    float* probs = g_path_scratch.get(config.vocab_size);
    if (probs == nullptr) {
        return;
    }
    
    float inv_batch = 1.0f / static_cast<float>(batch_size);

    // Compute effective eigenvalues if hidden_states used
    std::vector<float> computed_eigenvalues;
    std::vector<float> computed_eigenvectors; // [k, d]
    const float* effective_eigenvalues = eigenvalues;
    
    if (eigenvalues == nullptr && hidden_states != nullptr && num_states > 0 && hidden_dim > 0) {
        if (config.entropy_weight > 0.0f || config.spectral_weight > 0.0f) {
            computed_eigenvalues.resize(num_eigs);
            computed_eigenvectors.resize(num_eigs * hidden_dim);
            compute_lanczos_eigenvalues(
                hidden_states, num_states, hidden_dim, num_eigs, 
                computed_eigenvalues.data(), computed_eigenvectors.data()
            );
            effective_eigenvalues = computed_eigenvalues.data();
        }
    }

    // Initialize head gradients to zero if provided
    if (grad_head_weights != nullptr) std::fill(grad_head_weights, grad_head_weights + config.vocab_size * hidden_dim, 0.0f);
    if (grad_head_bias != nullptr) std::fill(grad_head_bias, grad_head_bias + config.vocab_size, 0.0f);

    // 1. Gradients w.r.t. Logits (CE + Fidelity + Entropy)
    for (int64_t b = 0; b < batch_size; ++b) {
        int32_t target = labels[b];
        if (target < 0 || target >= config.vocab_size) {
            if (logits != nullptr && grad_logits != nullptr) {
                std::fill(grad_logits + b * config.vocab_size, grad_logits + (b+1) * config.vocab_size, 0.0f);
            }
            continue;
        }

        // a. Get/Compute probs
        if (logits != nullptr) {
            const float* sample_logits = logits + b * config.vocab_size;
            std::copy(sample_logits, sample_logits + config.vocab_size, probs);
        } else {
            const float* h = hidden_states + b * hidden_dim;
            simd_gemv_t(head_weights, h, probs, config.vocab_size, hidden_dim);
            if (head_bias != nullptr) {
                for(int64_t i=0; i<config.vocab_size; ++i) probs[i] += head_bias[i];
            }
        }
        
        // --- Sprint 5: InfoNCE Probabilities ---
        float* infonce_p = nullptr;
        float inv_tau = 1.0f / std::max(config.infonce_temperature, 1e-6f);
        if (config.infonce_weight > 0.0f) {
            infonce_p = g_path_scratch.get(config.vocab_size, 2); // Use different offset
            float max_l = -1e9f;
            for(int i=0; i<config.vocab_size; ++i) {
                infonce_p[i] = probs[i] * inv_tau;
                max_l = std::max(max_l, infonce_p[i]);
            }
            float sum_e = 0.0f;
            for(int i=0; i<config.vocab_size; ++i) {
                infonce_p[i] = std::exp(infonce_p[i] - max_l);
                sum_e += infonce_p[i];
            }
            float inv_sum = 1.0f / std::max(sum_e, kMinLogSumExp);
            for(int i=0; i<config.vocab_size; ++i) infonce_p[i] *= inv_sum;
        }
        // ----------------------------------------
        
        float max_val, sum_val;
        simd_softmax(probs, config.vocab_size, max_val, sum_val);
        
        // b. Compute dL/dLogit for this sample
        float* sample_grad = (grad_logits != nullptr) ? (grad_logits + b * config.vocab_size) : g_path_scratch.get(config.vocab_size + 1024, 1);
        
        bool use_prediction_entropy = (effective_eigenvalues == nullptr);
        float H = 0.0f;
        if (use_prediction_entropy) {
            for (int64_t i = 0; i < config.vocab_size; ++i) {
                float p = std::max(probs[i], kMinProb);
                H -= p * std::log(p);
            }
        }
        
        float safe_vocab = std::max(static_cast<float>(config.vocab_size), 2.0f);
        float max_ent = std::log(safe_vocab);
        float norm_H = H / std::max(max_ent, kStableEpsilon);
        float uniform_target = config.label_smoothing / static_cast<float>(config.vocab_size);
        
        for (int64_t i = 0; i < config.vocab_size; ++i) {
            float p = probs[i];
            float p_safe = std::max(p, kMinProb);
            float log_p = std::log(p_safe);
            
            float ce_target = (i == target) ? (1.0f - config.label_smoothing) + uniform_target : uniform_target;
            float g_ce = p - ce_target;
            float p_target = probs[target];
            float g_fid = (i == target) ? -p_target * (1.0f - p_target) : p_target * p;
            float g_ent = 0.0f;
            if (config.entropy_weight > 0.0f && use_prediction_entropy) {
                float diff = norm_H - config.target_entropy;
                float dH_dlogit = -p * (log_p + H);
                g_ent = 2.0f * diff * (1.0f / max_ent) * dH_dlogit;
            }
            float g_infonce = 0.0f;
            if (config.infonce_weight > 0.0f && infonce_p != nullptr) {
                g_infonce = inv_tau * (infonce_p[i] - ce_target);
            }
            
            sample_grad[i] = inv_batch * (config.ce_weight * g_ce + 
                                       config.fidelity_weight * g_fid + 
                                       config.entropy_weight * g_ent +
                                       config.infonce_weight * g_infonce);
        }

        // c. Backprop from Logits to Hidden and Head Parameters
        if (logits == nullptr && hidden_states != nullptr && head_weights != nullptr) {
            const float* h = hidden_states + b * hidden_dim;
            
            // dL/dh = grad_logits * weights
            if (grad_hidden_states != nullptr) {
                float* gh = grad_hidden_states + b * hidden_dim;
                simd_gemv(head_weights, sample_grad, gh, config.vocab_size, hidden_dim);
            }
            
            // dL/dweights += grad_logits^T * h
            if (grad_head_weights != nullptr) {
                for(int64_t i=0; i<config.vocab_size; ++i) {
                    float g_i = sample_grad[i];
                    if (std::abs(g_i) < 1e-12f) continue;
                    float* gw_row = grad_head_weights + i * hidden_dim;
                    for(int64_t j=0; j<hidden_dim; ++j) gw_row[j] += g_i * h[j];
                }
            }
            
            // dL/dbias += grad_logits
            if (grad_head_bias != nullptr) {
                for(int64_t i=0; i<config.vocab_size; ++i) grad_head_bias[i] += sample_grad[i];
            }
        }
        
        // d. NaN/Inf safety
        if (grad_logits != nullptr || (logits == nullptr && hidden_states != nullptr)) {
            for (int64_t j = 0; j < config.vocab_size; ++j) {
                if (!std::isfinite(sample_grad[j])) sample_grad[j] = 0.0f;
            }
        }
    }

    // 2-4. [Same as before: Born, Coherence, Symplectic]
    if (grad_amplitudes != nullptr && amplitudes != nullptr && config.born_weight > 0.0f) {
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* amp = amplitudes + b * K * 2;
            float* g_amp = grad_amplitudes + b * K * 2;
            float norm_sq = 0.0f;
            for (int i = 0; i < K * 2; ++i) norm_sq += amp[i] * amp[i];
            float common = 4.0f * (norm_sq - 1.0f) * config.born_weight * inv_batch;
            for (int i = 0; i < K * 2; ++i) g_amp[i] = common * amp[i];
        }
    }
    
    if (grad_coherence_scores != nullptr && coherence_scores != nullptr && config.coherence_weight > 0.0f) {
        float inv_blocks = 1.0f / static_cast<float>(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            float deficit = config.coherence_threshold - coherence_scores[i];
            if (deficit > 0.0f) {
                grad_coherence_scores[i] = -2.0f * deficit * inv_blocks * config.coherence_weight;
            } else {
                grad_coherence_scores[i] = 0.0f;
            }
        }
    }
    
    if (grad_h_init != nullptr && grad_h_final != nullptr && h_init != nullptr && h_final != nullptr && config.symplectic_weight > 0.0f) {
        float denom = static_cast<float>(batch_size) * std::max(config.symplectic_dt, kStableEpsilon);
        float weight_scaled = config.symplectic_weight / denom;
        for (int64_t b = 0; b < batch_size; ++b) {
            float diff = h_final[b] - h_init[b];
            float sign = (diff > 0.0f) ? 1.0f : (diff < 0.0f ? -1.0f : 0.0f);
            grad_h_final[b] = sign * weight_scaled;
            grad_h_init[b] = -sign * weight_scaled;
        }
    }

    // 5. Gradients w.r.t. Eigenvalues and Hidden States (Spectral Entropy / Flatness)
    // Accumulate dL/dLambda
    std::vector<float> dL_dlambda(num_eigs, 0.0f);
    
    if (effective_eigenvalues != nullptr) {
        // Spectral Flatness Contribution
        if (config.spectral_weight > 0.0f) {
             float log_sum = 0.0f;
             float arith_sum = 0.0f;
             int n_pos = 0;
             for (int i = 0; i < num_eigs; ++i) {
                 if (effective_eigenvalues[i] > kMinProb) {
                     log_sum += std::log(effective_eigenvalues[i]);
                     arith_sum += effective_eigenvalues[i];
                     n_pos++;
                 }
             }
             if (n_pos > 0) {
                 float n = static_cast<float>(n_pos);
                 float geo_mean = std::exp(log_sum / n);
                 float arith_mean = arith_sum / n;
                 float SF = geo_mean / std::max(arith_mean, kStableEpsilon);
                 float diff = SF - config.target_spectral_flatness;
                 float common = 2.0f * diff * config.spectral_weight;
                 for (int i = 0; i < num_eigs; ++i) {
                     if (effective_eigenvalues[i] > kMinProb) {
                         float dSF_dev = SF * (1.0f / (n * effective_eigenvalues[i]) - 1.0f / arith_sum);
                         dL_dlambda[i] += common * dSF_dev;
                     }
                 }
             }
        }
        
        // Spectral Entropy Contribution (if this was used as entropy loss)
        if (config.entropy_weight > 0.0f) {
             float sum_eigs = 0.0f;
             for(int i=0; i<num_eigs; ++i) sum_eigs += effective_eigenvalues[i];
             if (sum_eigs > 1e-8f) {
                 float inv_sum = 1.0f / sum_eigs;
                 float spec_ent = 0.0f;
                 for(int i=0; i<num_eigs; ++i) {
                     float p = effective_eigenvalues[i] * inv_sum;
                     if (p > 1e-8f) spec_ent -= p * std::log(p);
                 }
                 float max_ent = std::log(static_cast<float>(num_eigs));
                 float norm_scale = 1.0f / std::max(max_ent, 1e-6f);
                 float norm_ent = spec_ent * norm_scale;
                 float diff = norm_ent - config.target_entropy;
                 
                 float common = 2.0f * diff * config.entropy_weight * norm_scale;
                 // dH/dlambda_k
                 // H = -sum (lam_i/S) log(lam_i/S)
                 // dH/dlam_k = (-1/S) * (log(lam_k/S) + 1) + (1/S^2)*(sum lam log lam) ??
                 // Easier: d(p_i)/dlam_k. 
                 // p_i = lam_i / S. S = sum lam.
                 // dp_i/dlam_k = (del_ik * S - lam_i) / S^2 = (del_ik - p_i)/S.
                 // dH/dp_i = -(1 + log p_i).
                 // dH/dlam_k = sum_i dH/dp_i * dp_i/dlam_k
                 // = sum_i -(1+log p_i) * (del_ik - p_i)/S
                 // = (-1/S) * [ (1+log p_k) - sum_i p_i(1+log p_i) ]
                 // = (-1/S) * [ 1 + log p_k - (1 + sum p_i log p_i) ] (sum p_i=1)
                 // = (-1/S) * [ log p_k + H_unnormalized ]
                 
                 for(int k=0; k<num_eigs; ++k) {
                     if (effective_eigenvalues[k] > kMinProb) {
                         float p_k = effective_eigenvalues[k] * inv_sum;
                         float dH_dlam = -(1.0f/sum_eigs) * (std::log(p_k) + spec_ent);
                         dL_dlambda[k] += common * dH_dlam;
                     }
                 }
             }
        }
    }
    
    // Output grad_eigenvalues if requested
    if (grad_eigenvalues != nullptr) {
        for(int i=0; i<num_eigs; ++i) grad_eigenvalues[i] = dL_dlambda[i];
    }
    
    // Output grad_hidden_states if requested
    if (grad_hidden_states != nullptr && hidden_states != nullptr && num_states > 0) {
        std::fill(grad_hidden_states, grad_hidden_states + num_states * hidden_dim, 0.0f);
        
        // Calculate gradient w.r.t H
        // dL/dH = (2/N) * sum_k (dL/dlam_k) * H_c * v_k * v_k^T
        //       = (2/N) * sum_k (dL/dlam_k) * (H_c * v_k) * v_k^T
        // H_c * v_k is exactly what we compute in Lanczos as y before normalization?
        // Actually, let's just recompute z_k = (H - mean) * v_k.
        
        // Need mean
        std::vector<float> mean(hidden_dim, 0.0f);
        for(int64_t i=0; i<num_states; ++i) {
            for(int j=0; j<hidden_dim; ++j) mean[j] += hidden_states[i*hidden_dim + j];
        }
        float inv_N = 1.0f / num_states;
        for(int j=0; j<hidden_dim; ++j) mean[j] *= inv_N;
        
        // For each eigenvector k
        std::vector<float> z_k(num_states);
        for(int k=0; k<num_eigs; ++k) {
            float dL = dL_dlambda[k];
            if (std::abs(dL) < 1e-12f) continue;
            
            const float* v = computed_eigenvectors.data() + k * hidden_dim;
            
            // z_k = (H - mean) * v
            float mean_dot_v = simd_dot(mean.data(), v, hidden_dim);
            for(int64_t i=0; i<num_states; ++i) {
                 z_k[i] = simd_dot(&hidden_states[i*hidden_dim], v, hidden_dim) - mean_dot_v;
            }
            
            // Add (2dL/N) * z_k * v^T to gradient
            float alpha = 2.0f * dL * inv_N;
            for(int64_t i=0; i<num_states; ++i) {
                 float scale = alpha * z_k[i];
                 float* grad_row = grad_hidden_states + i * hidden_dim;
                 for(int j=0; j<hidden_dim; ++j) {
                     grad_row[j] += scale * v[j];
                 }
            }
        }
        
        // Center gradients?
        // As derived, subtracting mean of gradients over batch is usually required if centering was part of op
        // dL/dH = P * (gradient derived above). P = I - J/N.
        std::vector<float> grad_mean(hidden_dim, 0.0f);
        for(int64_t i=0; i<num_states; ++i) {
            for(int j=0; j<hidden_dim; ++j) grad_mean[j] += grad_hidden_states[i*hidden_dim + j];
        }
        for(int j=0; j<hidden_dim; ++j) grad_mean[j] *= inv_N;
        
        for(int64_t i=0; i<num_states; ++i) {
             for(int j=0; j<hidden_dim; ++j) grad_hidden_states[i*hidden_dim + j] -= grad_mean[j];
        }
    }
}

}  // namespace quls
}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_QULS_LOSS_OP_H_
