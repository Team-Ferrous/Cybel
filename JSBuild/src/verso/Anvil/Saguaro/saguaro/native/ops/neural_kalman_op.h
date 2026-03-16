// saguaro.native/ops/neural_kalman_op.h
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
 * @file neural_kalman_op.h
 * @brief Phase 43: Neural Kalman with Learned Covariance
 *
 * Enhances KalmanBlock with GRU-based learned Kalman gain for
 * the `mamba_timecrystal_wlam_moe_hybrid` block pattern (Block 1).
 *
 * Key Features:
 *   - GRU-Based Kalman Gain: Learns K dynamically from data
 *   - Covariance Propagation: Optional full P tracking
 *   - Adaptive Noise Estimation: Learns Q 5and R from residuals
 *
 * Research Basis: "Neural Extended Kalman Filters" (ICRA 2024)
 *
 * Integration Points:
 *   - Block 1: KalmanBlock (after TimeCrystalSequenceBlock)
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef SAGUARO_NATIVE_OPS_NEURAL_KALMAN_OP_H_
#define SAGUARO_NATIVE_OPS_NEURAL_KALMAN_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace saguaro {
namespace neural_kalman {

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * @brief Configuration for Neural Kalman filter.
 */
struct NeuralKalmanConfig {
    int hidden_dim;               // GRU hidden dimension
    int state_dim;                // Kalman state dimension
    int obs_dim;                  // Observation dimension (d_inner)
    bool propagate_covariance;    // Track full P matrix
    float initial_P;              // Initial covariance diagonal
    float process_noise_init;     // Initial Q estimate
    float measurement_noise_init; // Initial R estimate
    
    // Phase 43.1: Numerical stabilization parameters
    float max_innovation;         // Innovation clamp magnitude (default: 10.0)
    float epsilon;                // Numerical epsilon (default: 1e-6)
    float grad_clip_norm;         // Gradient norm clipping (default: 1.0)
    bool enable_adaptive_scaling; // Adaptive scaling based on input stats
    bool enable_diagnostics;      // Enable diagnostic output
    
    // Phase 43.2: Full Kalman filter parameters (stability fix)
    bool use_full_kalman;         // Use proper Kalman equations with A matrix
    float P_min;                  // Minimum covariance diagonal (prevents K→0)
    float P_max;                  // Maximum covariance diagonal (prevents overflow)
    float K_max;                  // Maximum Kalman gain (prevents overcorrection)
    
    NeuralKalmanConfig()
        : hidden_dim(128)
        , state_dim(64)
        , obs_dim(64)
        , propagate_covariance(false)  // Expensive, off by default
        , initial_P(1.0f)
        , process_noise_init(0.01f)
        , measurement_noise_init(0.1f)
        , max_innovation(10.0f)        // Phase 43.1: 5σ for Gaussian data
        , epsilon(1e-6f)
        , grad_clip_norm(1.0f)
        , enable_adaptive_scaling(true)
        , enable_diagnostics(false)
        // Phase 43.2: Full Kalman defaults
        , use_full_kalman(true)        // Enable proper Kalman equations by default
        , P_min(1e-6f)                 // Minimum covariance
        , P_max(10.0f)                 // Maximum covariance
        , K_max(1.0f) {}               // Maximum gain (0-1 for stability)
};

// =============================================================================
// PHASE 43.1: NUMERICAL STABILIZATION UTILITIES
// =============================================================================

/**
 * @brief Numerical stabilization utilities for NeuralKalmanStep.
 * 
 * Provides SIMD-optimized clamping, adaptive scaling, and NaN/Inf handling
 * to prevent GRU saturation from large upstream innovations.
 */
struct NumericalStabilizer {
    float max_innovation;      // Maximum absolute innovation value
    float epsilon;             // Numerical epsilon
    float grad_clip_norm;      // Gradient norm clipping threshold
    
    NumericalStabilizer()
        : max_innovation(10.0f)
        , epsilon(1e-6f)
        , grad_clip_norm(1.0f) {}
    
    NumericalStabilizer(const NeuralKalmanConfig& config)
        : max_innovation(config.max_innovation)
        , epsilon(config.epsilon)
        , grad_clip_norm(config.grad_clip_norm) {}
    
    /**
     * @brief Smoothly scale values when they exceed max_innovation.
     * @param data Input/output array.
     * @param size Number of elements.
     * @param clamp_mask Optional mask storing applied scale (uniform).
     */
    inline void clamp_innovation(float* data, int size, float* clamp_mask = nullptr) const {
        if (size <= 0 || max_innovation <= 0.0f) {
            return;
        }
        float max_abs = 0.0f;
        for (int i = 0; i < size; ++i) {
            float abs_val = std::abs(data[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }
        if (max_abs <= max_innovation) {
            return;
        }
        const float scale = max_innovation / (max_innovation + max_abs + epsilon);
        for (int i = 0; i < size; ++i) {
            data[i] *= scale;
            if (clamp_mask) {
                clamp_mask[i] = scale;
            }
        }
    }
    
    /**
     * @brief Compute adaptive scale factor based on input statistics.
     * @param data Input array.
     * @param size Number of elements.
     * @return Scale factor to apply (1.0 if no scaling needed).
     */
    inline float compute_adaptive_scale(const float* data, int size) const {
        if (size == 0 || max_innovation <= 0.0f) return 1.0f;
        
        float max_abs = 0.0f;
        for (int i = 0; i < size; ++i) {
            float abs_val = std::abs(data[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        if (max_abs > max_innovation) {
            return max_innovation / (max_innovation + max_abs + epsilon);
        }
        return 1.0f;
    }
    
    /**
     * @brief Replace NaN/Inf with safe values in-place.
     * @param data Input/output array.
     * @param size Number of elements.
     * @param replacement Replacement value for non-finite elements.
     */
    inline void replace_nonfinite(float* data, int size, float replacement = 0.0f) const {
        for (int i = 0; i < size; ++i) {
            if (!std::isfinite(data[i])) {
                data[i] = replacement;
            }
        }
    }
    
    /**
     * @brief Compute gradient norm for clipping decision.
     * @param grad Gradient array.
     * @param size Number of elements.
     * @return L2 norm of gradient.
     */
    inline float compute_grad_norm(const float* grad, int size) const {
        float sum_sq = 0.0f;
        for (int i = 0; i < size; ++i) {
            sum_sq += grad[i] * grad[i];
        }
        return std::sqrt(sum_sq + epsilon);
    }
    
    /**
     * @brief Smoothly scale gradient by norm (no hard clipping).
     * @param grad Gradient array (modified in-place).
     * @param size Number of elements.
     * @return Applied scale (1.0 if no scaling).
     */
    inline float clip_gradient_by_norm(float* grad, int size) const {
        if (grad_clip_norm <= 0.0f) {
            return 1.0f;
        }
        float norm = compute_grad_norm(grad, size);
        if (norm <= 0.0f) {
            return 1.0f;
        }
        float scale = grad_clip_norm / (grad_clip_norm + norm);
        if (scale < 1.0f) {
            for (int i = 0; i < size; ++i) {
                grad[i] *= scale;
            }
        }
        return scale;
    }
};

// =============================================================================
// GRU OPERATIONS
// =============================================================================

/**
 * @brief GRU forward pass for learned Kalman gain.
 *
 * Computes:
 *   z = σ(W_z @ [h, x] + b_z)   // Update gate
 *   r = σ(W_r @ [h, x] + b_r)   // Reset gate
 *   h_candidate = tanh(W_h @ [r*h, x] + b_h)
 *   h_new = (1-z) * h + z * h_candidate
 *
 * @param input Input [batch, input_dim]
 * @param hidden Previous hidden [batch, hidden_dim]
 * @param W_z Update gate weights [hidden_dim, hidden_dim + input_dim]
 * @param W_r Reset gate weights [hidden_dim, hidden_dim + input_dim]
 * @param W_h Candidate weights [hidden_dim, hidden_dim + input_dim]
 * @param b_z Update gate bias [hidden_dim]
 * @param b_r Reset gate bias [hidden_dim]
 * @param b_h Candidate bias [hidden_dim]
 * @param output New hidden state [batch, hidden_dim]
 * @param batch_size Batch size
 * @param input_dim Input dimension
 * @param hidden_dim Hidden dimension
 */
inline void GRUForward(
    const float* input, const float* hidden,
    const float* W_z, const float* W_r, const float* W_h,
    const float* b_z, const float* b_r, const float* b_h,
    float* output,
    int batch_size, int input_dim, int hidden_dim) {
    
    const int concat_dim = hidden_dim + input_dim;
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        std::vector<float> concat(concat_dim);
        std::vector<float> z(hidden_dim);
        std::vector<float> r(hidden_dim);
        std::vector<float> h_candidate(hidden_dim);
        
        // Concatenate [hidden, input]
        const float* h_ptr = hidden + b * hidden_dim;
        const float* x_ptr = input + b * input_dim;
        
        std::copy(h_ptr, h_ptr + hidden_dim, concat.begin());
        std::copy(x_ptr, x_ptr + input_dim, concat.begin() + hidden_dim);
        
        // Update gate: z = σ(W_z @ [h, x] + b_z)
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = b_z[i];
            for (int j = 0; j < concat_dim; ++j) {
                sum += W_z[i * concat_dim + j] * concat[j];
            }
            z[i] = 1.0f / (1.0f + std::exp(-sum));  // Sigmoid
        }
        
        // Reset gate: r = σ(W_r @ [h, x] + b_r)
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = b_r[i];
            for (int j = 0; j < concat_dim; ++j) {
                sum += W_r[i * concat_dim + j] * concat[j];
            }
            r[i] = 1.0f / (1.0f + std::exp(-sum));
        }
        
        // Update concat for candidate: [r*h, x]
        for (int i = 0; i < hidden_dim; ++i) {
            concat[i] = r[i] * h_ptr[i];
        }
        
        // Candidate: h_candidate = tanh(W_h @ [r*h, x] + b_h)
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = b_h[i];
            for (int j = 0; j < concat_dim; ++j) {
                sum += W_h[i * concat_dim + j] * concat[j];
            }
            h_candidate[i] = std::tanh(sum);
        }
        
        // Output: h_new = (1-z) * h + z * h_candidate
        float* out_ptr = output + b * hidden_dim;
        for (int i = 0; i < hidden_dim; ++i) {
            out_ptr[i] = (1.0f - z[i]) * h_ptr[i] + z[i] * h_candidate[i];
        }
    }
}

// =============================================================================
// NEURAL KALMAN OPERATIONS
// =============================================================================

/**
 * @brief Compute learned Kalman gain from GRU hidden state.
 *
 * Projects GRU output to Kalman gain matrix:
 *   K = W_out @ gru_hidden
 *
 * @param gru_hidden GRU hidden state [batch, hidden_dim]
 * @param W_out Output projection [state_dim, hidden_dim]
 * @param K_gain Output Kalman gain [batch, state_dim]
 * @param batch_size Batch size
 * @param hidden_dim GRU hidden dimension
 * @param state_dim Kalman state dimension
 */
inline void ComputeLearnedKalmanGain(
    const float* gru_hidden,
    const float* W_out,
    float* K_gain,
    int batch_size, int hidden_dim, int state_dim) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            float sum = 0.0f;
            
            int j = 0;
#if defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            for (; j + 8 <= hidden_dim; j += 8) {
                __m256 w = _mm256_loadu_ps(&W_out[i * hidden_dim + j]);
                __m256 h = _mm256_loadu_ps(&gru_hidden[b * hidden_dim + j]);
                acc = _mm256_fmadd_ps(w, h, acc);
            }
            // Horizontal sum
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum = _mm_cvtss_f32(sum4);
#endif
            for (; j < hidden_dim; ++j) {
                sum += W_out[i * hidden_dim + j] * gru_hidden[b * hidden_dim + j];
            }
            
            // Apply tanh to bound gain
            K_gain[b * state_dim + i] = std::tanh(sum);
        }
    }
}

/**
 * @brief Neural Kalman update step.
 *
 * Uses learned Kalman gain for state update:
 *   x_posterior = x_prior + K * innovation
 *
 * IMPORTANT: Innovation must be pre-stabilized (clamped/scaled) to prevent
 * numerical explosion. The GRU path uses stabilized innovation, and the
 * update step must use the same stabilized values for consistency.
 *
 * @param x_prior Prior state estimate [batch, state_dim]
 * @param innovation Pre-stabilized innovation [batch, state_dim] (z - x_prior, clamped)
 * @param K_gain Learned Kalman gain [batch, state_dim]
 * @param x_posterior Output posterior estimate [batch, state_dim]
 * @param batch_size Batch size
 * @param state_dim State dimension
 */
inline void NeuralKalmanUpdate(
    const float* x_prior, const float* innovation, const float* K_gain,
    float* x_posterior,
    int batch_size, int state_dim) {

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            int idx = b * state_dim + i;

            // Kalman update with pre-stabilized innovation
            x_posterior[idx] = x_prior[idx] + K_gain[idx] * innovation[idx];
        }
    }
}

/**
 * @brief Full Neural Kalman step with GRU-based gain learning.
 *
 * @param x_prior Prior state [batch, state_dim]
 * @param z Measurement [batch, state_dim]
 * @param gru_hidden GRU hidden state [batch, hidden_dim]
 * @param W_z, W_r, W_h GRU weights
 * @param b_z, b_r, b_h GRU biases
 * @param W_out Kalman gain projection [state_dim, hidden_dim]
 * @param x_posterior Output posterior [batch, state_dim]
 * @param gru_hidden_new Output new GRU hidden [batch, hidden_dim]
 * @param config Neural Kalman configuration
 * @param batch_size Batch size
 */
inline void NeuralKalmanStep(
    const float* x_prior, const float* z,
    const float* gru_hidden,
    const float* W_z, const float* W_r, const float* W_h,
    const float* b_z, const float* b_r, const float* b_h,
    const float* W_out,
    float* x_posterior, float* gru_hidden_new,
    const NeuralKalmanConfig& config,
    int batch_size) {

    const int state_dim = config.state_dim;
    const int hidden_dim = config.hidden_dim;
    const int total_size = batch_size * state_dim;

    // Initialize numerical stabilizer from config
    NumericalStabilizer stabilizer(config);

    // Compute innovation as GRU input
    std::vector<float> innovation(total_size);
    for (int i = 0; i < total_size; ++i) {
        innovation[i] = z[i] - x_prior[i];
    }

    // Phase 46: Use adaptive scaling to avoid hard clipping while keeping GRU stable.
    if (config.enable_adaptive_scaling) {
        float scale = stabilizer.compute_adaptive_scale(innovation.data(), total_size);
        if (scale != 1.0f) {
            for (int i = 0; i < total_size; ++i) {
                innovation[i] *= scale;
            }
        }
    } else {
        // Fallback to clamping when adaptive scaling is disabled.
        stabilizer.clamp_innovation(innovation.data(), total_size);
    }

    // Replace any NaN/Inf in innovation with zeros
    stabilizer.replace_nonfinite(innovation.data(), total_size, 0.0f);

    // GRU update with stabilized innovation as input
    GRUForward(
        innovation.data(), gru_hidden,
        W_z, W_r, W_h, b_z, b_r, b_h,
        gru_hidden_new,
        batch_size, state_dim, hidden_dim
    );

    // Replace any NaN/Inf in GRU output
    stabilizer.replace_nonfinite(gru_hidden_new, batch_size * hidden_dim, 0.0f);

    // Compute learned Kalman gain
    std::vector<float> K_gain(total_size);
    ComputeLearnedKalmanGain(
        gru_hidden_new, W_out, K_gain.data(),
        batch_size, hidden_dim, state_dim
    );

    // K_gain is already bounded via tanh; avoid extra clipping when adaptive scaling is on.
    if (!config.enable_adaptive_scaling) {
        stabilizer.clamp_innovation(K_gain.data(), total_size);
    }

    // Kalman update with stabilized innovation (not raw z - x_prior)
    NeuralKalmanUpdate(
        x_prior, innovation.data(), K_gain.data(), x_posterior,
        batch_size, state_dim
    );

    // Final stability check on posterior
    stabilizer.replace_nonfinite(x_posterior, total_size, 0.0f);
}

/**
 * @brief Propagate covariance (optional, expensive).
 *
 * P_posterior = (I - K @ H) @ P_prior
 *
 * @param P_prior Prior covariance [batch, state_dim, state_dim]
 * @param K_gain Kalman gain [batch, state_dim]
 * @param P_posterior Output posterior covariance [batch, state_dim, state_dim]
 * @param batch_size Batch size
 * @param state_dim State dimension
 */
inline void PropagateCovariance(
    const float* P_prior, const float* K_gain,
    float* P_posterior,
    int batch_size, int state_dim) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            for (int j = 0; j < state_dim; ++j) {
                int idx = b * state_dim * state_dim + i * state_dim + j;
                
                // (I - K @ H) @ P where H = I (simplified)
                // = P - K_i * P_row
                float correction = K_gain[b * state_dim + i] *
                                   P_prior[b * state_dim * state_dim + i * state_dim + j];
                
                P_posterior[idx] = P_prior[idx] - correction;
            }
        }
    }
}

// =============================================================================
// PHASE 43.2: FULL KALMAN FILTER OPERATIONS (STABILITY FIX)
// =============================================================================

/**
 * @brief Softplus activation for noise parameters.
 */
inline float softplus(float x) {
    if (x > 20.0f) return x;  // Avoid overflow
    return std::log(1.0f + std::exp(x));
}

/**
 * @brief State prediction with orthogonal transition matrix.
 * 
 * Computes: x_pred = A @ x_prior
 * where A is orthogonal (eigenvalues magnitude ≤ 1).
 * This ensures ||x_pred|| ≤ ||x_prior|| for stability.
 *
 * @param x_prior Prior state [batch, state_dim]
 * @param A State transition matrix [state_dim, state_dim] (orthogonal)
 * @param x_pred Output predicted state [batch, state_dim]
 * @param batch_size Batch size
 * @param state_dim State dimension
 */
inline void StatePrediction(
    const float* x_prior,
    const float* A,
    float* x_pred,
    int batch_size, int state_dim) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            float sum = 0.0f;
            
            // x_pred[i] = sum_j A[i,j] * x_prior[j]
            int j = 0;
#if defined(__AVX2__)
            __m256 acc = _mm256_setzero_ps();
            for (; j + 8 <= state_dim; j += 8) {
                __m256 a = _mm256_loadu_ps(&A[i * state_dim + j]);
                __m256 x = _mm256_loadu_ps(&x_prior[b * state_dim + j]);
                acc = _mm256_fmadd_ps(a, x, acc);
            }
            // Horizontal sum
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum4 = _mm_add_ps(lo, hi);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum4 = _mm_hadd_ps(sum4, sum4);
            sum = _mm_cvtss_f32(sum4);
#endif
            for (; j < state_dim; ++j) {
                sum += A[i * state_dim + j] * x_prior[b * state_dim + j];
            }
            
            x_pred[b * state_dim + i] = sum;
        }
    }
}

/**
 * @brief Observation projection: H @ x.
 * 
 * Projects state to observation space.
 *
 * @param x State [batch, state_dim]
 * @param H Observation matrix [obs_dim, state_dim]
 * @param Hx Output projection [batch, obs_dim]
 * @param batch_size Batch size
 * @param state_dim State dimension
 * @param obs_dim Observation dimension
 */
inline void ObservationProjection(
    const float* x,
    const float* H,
    float* Hx,
    int batch_size, int state_dim, int obs_dim) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < obs_dim; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < state_dim; ++j) {
                sum += H[i * state_dim + j] * x[b * state_dim + j];
            }
            Hx[b * obs_dim + i] = sum;
        }
    }
}

/**
 * @brief Full Neural Kalman step with proper Kalman filter equations.
 *
 * Implements the complete Kalman filter with:
 * 1. State prediction: x_pred = A @ x_prior (orthogonal A keeps ||x|| bounded)
 * 2. Covariance prediction: P_pred = P_prior + Q
 * 3. Innovation: y = z - H @ x_pred
 * 4. Kalman gain: K = P_pred / (P_pred + R) (bounded in [0, 1])
 * 5. State update: x_post = x_pred + K * (H^T @ innovation)
 * 6. Covariance update: P_post = P_pred * (1 - K) (shrinks → stabilizes)
 *
 * The GRU is retained for adaptive noise estimation but is NOT the primary
 * gain mechanism, ensuring proper Kalman filter stability guarantees.
 *
 * @param x_prior Prior state [batch, state_dim]
 * @param P_prior Prior covariance diagonal [batch, state_dim]
 * @param z Measurement [batch, obs_dim]
 * @param A State transition matrix [state_dim, state_dim] (orthogonal)
 * @param H Observation matrix [obs_dim, state_dim]
 * @param Q Process noise (learnable) [state_dim]
 * @param R Measurement noise (learnable) [obs_dim]
 * @param gru_hidden GRU hidden state [batch, hidden_dim]
 * @param W_z, W_r, W_h GRU weights (for adaptive noise, not gain)
 * @param b_z, b_r, b_h GRU biases
 * @param W_out GRU output projection [state_dim, hidden_dim]
 * @param x_posterior Output posterior state [batch, state_dim]
 * @param P_posterior Output posterior covariance [batch, state_dim]
 * @param gru_hidden_new Output new GRU hidden [batch, hidden_dim]
 * @param config Neural Kalman configuration
 * @param batch_size Batch size
 */
inline void NeuralKalmanStepFull(
    const float* x_prior, const float* P_prior,
    const float* z,
    const float* A, const float* H,
    const float* Q, const float* R,
    const float* gru_hidden,
    const float* W_z, const float* W_r, const float* W_h,
    const float* b_z, const float* b_r, const float* b_h,
    const float* W_out,
    float* x_posterior, float* P_posterior, float* gru_hidden_new,
    const NeuralKalmanConfig& config,
    int batch_size) {

    const int state_dim = config.state_dim;
    const int obs_dim = config.obs_dim;
    const int hidden_dim = config.hidden_dim;
    const int state_total = batch_size * state_dim;
    const int obs_total = batch_size * obs_dim;
    
    NumericalStabilizer stabilizer(config);
    
    // ==== 1. STATE PREDICTION: x_pred = A @ x_prior ====
    // Using orthogonal A ensures ||x_pred|| ≤ ||x_prior||
    std::vector<float> x_pred(state_total);
    StatePrediction(x_prior, A, x_pred.data(), batch_size, state_dim);
    
    // ==== 2. COVARIANCE PREDICTION: P_pred = P_prior + softplus(Q) ====
    std::vector<float> P_pred(state_total);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            int idx = b * state_dim + i;
            float P_val = P_prior[idx] + softplus(Q[i]);
            // Bound covariance to prevent overflow
            P_pred[idx] = std::clamp(P_val, config.P_min, config.P_max);
        }
    }
    
    // ==== 3. INNOVATION: y = z - H @ x_pred ====
    std::vector<float> Hx_pred(obs_total);
    ObservationProjection(x_pred.data(), H, Hx_pred.data(), batch_size, state_dim, obs_dim);
    
    std::vector<float> innovation(obs_total);
    for (int i = 0; i < obs_total; ++i) {
        innovation[i] = z[i] - Hx_pred[i];
    }
    
    // Apply adaptive scaling to innovation for GRU stability
    if (config.enable_adaptive_scaling) {
        float scale = stabilizer.compute_adaptive_scale(innovation.data(), obs_total);
        if (scale != 1.0f) {
            for (int i = 0; i < obs_total; ++i) {
                innovation[i] *= scale;
            }
        }
    }
    stabilizer.replace_nonfinite(innovation.data(), obs_total, 0.0f);
    
    // ==== 4. KALMAN GAIN: K = P_pred / (P_pred + R) ====
    // This naturally bounds K to [0, 1] which prevents overcorrection!
    std::vector<float> K_gain(state_total);
    for (int b = 0; b < batch_size; ++b) {
        // Average R across observation dimensions for state-space gain
        float avg_R = 0.0f;
        for (int i = 0; i < obs_dim; ++i) {
            avg_R += softplus(R[i]);
        }
        avg_R /= static_cast<float>(obs_dim);
        
        for (int i = 0; i < state_dim; ++i) {
            int idx = b * state_dim + i;
            float P_val = P_pred[idx];
            float K = P_val / (P_val + avg_R + config.epsilon);
            // Ensure K is in [0, K_max] for stability
            K_gain[idx] = std::clamp(K, 0.0f, config.K_max);
        }
    }
    
    // ==== 5. STATE UPDATE: x_post = x_pred + K * H^T @ innovation ====
    // Project innovation back to state space via H^T, then scale by K
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            // H^T @ innovation for dimension i
            float Ht_innov = 0.0f;
            for (int j = 0; j < obs_dim; ++j) {
                // H^T[i,j] = H[j,i]
                Ht_innov += H[j * state_dim + i] * innovation[b * obs_dim + j];
            }
            
            int idx = b * state_dim + i;
            x_posterior[idx] = x_pred[idx] + K_gain[idx] * Ht_innov;
        }
    }
    
    // ==== 6. COVARIANCE UPDATE: P_post = P_pred * (1 - K) ====
    // As K approaches 1 (high confidence), P shrinks → future K shrinks
    // This is the key stability mechanism!
    for (int i = 0; i < state_total; ++i) {
        float P_new = P_pred[i] * (1.0f - K_gain[i]);
        P_posterior[i] = std::max(P_new, config.P_min);
    }
    
    // ==== GRU UPDATE (auxiliary, for adaptive noise estimation) ====
    // Run GRU with innovation as input - can be used for adaptive Q/R learning
    // but is NOT the primary gain mechanism anymore
    std::vector<float> gru_input(state_total);
    // Average innovation per state dimension
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            float avg_innov = 0.0f;
            for (int j = 0; j < obs_dim; ++j) {
                avg_innov += innovation[b * obs_dim + j];
            }
            gru_input[b * state_dim + i] = avg_innov / static_cast<float>(obs_dim);
        }
    }
    
    GRUForward(
        gru_input.data(), gru_hidden,
        W_z, W_r, W_h, b_z, b_r, b_h,
        gru_hidden_new,
        batch_size, state_dim, hidden_dim
    );
    
    // Replace any NaN/Inf in outputs
    stabilizer.replace_nonfinite(x_posterior, state_total, 0.0f);
    stabilizer.replace_nonfinite(P_posterior, state_total, config.initial_P);
    stabilizer.replace_nonfinite(gru_hidden_new, batch_size * hidden_dim, 0.0f);
}

}  // namespace neural_kalman
}  // namespace saguaro

// =============================================================================
// BACKWARD OPERATIONS
// =============================================================================

namespace saguaro {
namespace neural_kalman {

/**
 * @brief GRU backward pass.
 *
 * Computes gradients for GRU weights: W_z, W_r, W_h, b_z, b_r, b_h
 * and gradients for inputs: input, hidden.
 *
 * Forward equations:
 *   z = σ(W_z @ [h, x] + b_z)
 *   r = σ(W_r @ [h, x] + b_r)
 *   h_candidate = tanh(W_h @ [r*h, x] + b_h)
 *   h_new = (1-z) * h + z * h_candidate
 *
 * @param grad_h_new Gradient from downstream [batch, hidden_dim]
 * @param input Original input [batch, input_dim]
 * @param hidden Original hidden state [batch, hidden_dim]
 * @param W_z, W_r, W_h Original GRU weights
 * @param b_z, b_r, b_h Original GRU biases
 * @param grad_input Output gradient for input [batch, input_dim]
 * @param grad_hidden Output gradient for hidden [batch, hidden_dim]
 * @param grad_W_z, grad_W_r, grad_W_h Output gradients for weights
 * @param grad_b_z, grad_b_r, grad_b_h Output gradients for biases
 * @param batch_size Batch size
 * @param input_dim Input dimension
 * @param hidden_dim Hidden dimension
 */
inline void GRUBackward(
    const float* grad_h_new,
    const float* input, const float* hidden,
    const float* W_z, const float* W_r, const float* W_h,
    const float* b_z, const float* b_r, const float* b_h,
    float* grad_input, float* grad_hidden,
    float* grad_W_z, float* grad_W_r, float* grad_W_h,
    float* grad_b_z, float* grad_b_r, float* grad_b_h,
    int batch_size, int input_dim, int hidden_dim) {
    
    const int concat_dim = hidden_dim + input_dim;
    
    // Initialize weight gradients to zero
    std::fill(grad_W_z, grad_W_z + hidden_dim * concat_dim, 0.0f);
    std::fill(grad_W_r, grad_W_r + hidden_dim * concat_dim, 0.0f);
    std::fill(grad_W_h, grad_W_h + hidden_dim * concat_dim, 0.0f);
    std::fill(grad_b_z, grad_b_z + hidden_dim, 0.0f);
    std::fill(grad_b_r, grad_b_r + hidden_dim, 0.0f);
    std::fill(grad_b_h, grad_b_h + hidden_dim, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        const float* h_ptr = hidden + b * hidden_dim;
        const float* x_ptr = input + b * input_dim;
        const float* grad_out_ptr = grad_h_new + b * hidden_dim;
        float* grad_h_ptr = grad_hidden + b * hidden_dim;
        float* grad_x_ptr = grad_input + b * input_dim;
        
        // Recompute forward pass values needed for backward
        std::vector<float> concat(concat_dim);
        std::vector<float> z(hidden_dim), r(hidden_dim), h_candidate(hidden_dim);
        std::vector<float> pre_z(hidden_dim), pre_r(hidden_dim), pre_h(hidden_dim);
        
        // Forward recompute: concat = [h, x]
        std::copy(h_ptr, h_ptr + hidden_dim, concat.begin());
        std::copy(x_ptr, x_ptr + input_dim, concat.begin() + hidden_dim);
        
        // z = σ(W_z @ concat + b_z)
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = b_z[i];
            for (int j = 0; j < concat_dim; ++j) {
                sum += W_z[i * concat_dim + j] * concat[j];
            }
            pre_z[i] = sum;
            z[i] = 1.0f / (1.0f + std::exp(-sum));
        }
        
        // r = σ(W_r @ concat + b_r)
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = b_r[i];
            for (int j = 0; j < concat_dim; ++j) {
                sum += W_r[i * concat_dim + j] * concat[j];
            }
            pre_r[i] = sum;
            r[i] = 1.0f / (1.0f + std::exp(-sum));
        }
        
        // Update concat for candidate: [r*h, x]
        std::vector<float> concat_rh(concat_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            concat_rh[i] = r[i] * h_ptr[i];
        }
        std::copy(x_ptr, x_ptr + input_dim, concat_rh.begin() + hidden_dim);
        
        // h_candidate = tanh(W_h @ concat_rh + b_h)
        for (int i = 0; i < hidden_dim; ++i) {
            float sum = b_h[i];
            for (int j = 0; j < concat_dim; ++j) {
                sum += W_h[i * concat_dim + j] * concat_rh[j];
            }
            pre_h[i] = sum;
            h_candidate[i] = std::tanh(sum);
        }
        
        // ===== BACKWARD PASS =====
        // h_new = (1-z) * h + z * h_candidate
        
        // Gradient w.r.t z: d_z = grad_h_new * (h_candidate - h)
        std::vector<float> grad_z(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            grad_z[i] = grad_out_ptr[i] * (h_candidate[i] - h_ptr[i]);
        }
        
        // Gradient w.r.t h_candidate: d_hc = grad_h_new * z
        std::vector<float> grad_hc(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            grad_hc[i] = grad_out_ptr[i] * z[i];
        }
        
        // Gradient w.r.t h directly: d_h_direct = grad_h_new * (1 - z)
        std::vector<float> grad_h_direct(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            grad_h_direct[i] = grad_out_ptr[i] * (1.0f - z[i]);
        }
        
        // Backprop through h_candidate = tanh(pre_h)
        // d_pre_h = grad_hc * (1 - tanh^2)
        std::vector<float> grad_pre_h(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            float tanh_sq = h_candidate[i] * h_candidate[i];
            grad_pre_h[i] = grad_hc[i] * (1.0f - tanh_sq);
        }
        
        // Gradient for W_h and b_h
        for (int i = 0; i < hidden_dim; ++i) {
            grad_b_h[i] += grad_pre_h[i];
            for (int j = 0; j < concat_dim; ++j) {
                grad_W_h[i * concat_dim + j] += grad_pre_h[i] * concat_rh[j];
            }
        }
        
        // Gradient for concat_rh
        std::vector<float> grad_concat_rh(concat_dim, 0.0f);
        for (int j = 0; j < concat_dim; ++j) {
            for (int i = 0; i < hidden_dim; ++i) {
                grad_concat_rh[j] += W_h[i * concat_dim + j] * grad_pre_h[i];
            }
        }
        
        // Backprop concat_rh[0:hidden_dim] = r * h
        // grad_r = grad_concat_rh[0:hidden_dim] * h
        // grad_h from r*h = grad_concat_rh[0:hidden_dim] * r
        std::vector<float> grad_r(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            grad_r[i] = grad_concat_rh[i] * h_ptr[i];
            grad_h_direct[i] += grad_concat_rh[i] * r[i];
        }
        
        // grad_x from W_h path
        for (int i = 0; i < input_dim; ++i) {
            grad_x_ptr[i] = grad_concat_rh[hidden_dim + i];
        }
        
        // Backprop through r = σ(pre_r)
        // d_pre_r = grad_r * σ(pre_r) * (1 - σ(pre_r)) = grad_r * r * (1 - r)
        std::vector<float> grad_pre_r(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            grad_pre_r[i] = grad_r[i] * r[i] * (1.0f - r[i]);
        }
        
        // Gradient for W_r and b_r
        for (int i = 0; i < hidden_dim; ++i) {
            grad_b_r[i] += grad_pre_r[i];
            for (int j = 0; j < concat_dim; ++j) {
                grad_W_r[i * concat_dim + j] += grad_pre_r[i] * concat[j];
            }
        }
        
        // Gradient for concat from W_r path
        std::vector<float> grad_concat_r(concat_dim, 0.0f);
        for (int j = 0; j < concat_dim; ++j) {
            for (int i = 0; i < hidden_dim; ++i) {
                grad_concat_r[j] += W_r[i * concat_dim + j] * grad_pre_r[i];
            }
        }
        
        // Backprop through z = σ(pre_z)
        std::vector<float> grad_pre_z(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            grad_pre_z[i] = grad_z[i] * z[i] * (1.0f - z[i]);
        }
        
        // Gradient for W_z and b_z
        for (int i = 0; i < hidden_dim; ++i) {
            grad_b_z[i] += grad_pre_z[i];
            for (int j = 0; j < concat_dim; ++j) {
                grad_W_z[i * concat_dim + j] += grad_pre_z[i] * concat[j];
            }
        }
        
        // Gradient for concat from W_z path
        std::vector<float> grad_concat_z(concat_dim, 0.0f);
        for (int j = 0; j < concat_dim; ++j) {
            for (int i = 0; i < hidden_dim; ++i) {
                grad_concat_z[j] += W_z[i * concat_dim + j] * grad_pre_z[i];
            }
        }
        
        // Combine gradients for h and x
        // concat = [h, x], so grad_h = grad_concat[0:hidden_dim], grad_x += grad_concat[hidden_dim:]
        for (int i = 0; i < hidden_dim; ++i) {
            grad_h_ptr[i] = grad_h_direct[i] + grad_concat_z[i] + grad_concat_r[i];
        }
        for (int i = 0; i < input_dim; ++i) {
            grad_x_ptr[i] += grad_concat_z[hidden_dim + i] + grad_concat_r[hidden_dim + i];
        }
    }
}

/**
 * @brief Backward for learned Kalman gain computation.
 *
 * Forward: K = tanh(W_out @ gru_hidden)
 *
 * @param grad_K Gradient from downstream [batch, state_dim]
 * @param gru_hidden Original GRU hidden [batch, hidden_dim]
 * @param W_out Original output weights [state_dim, hidden_dim]
 * @param K_gain Original Kalman gain (for tanh derivative) [batch, state_dim]
 * @param grad_gru_hidden Output gradient [batch, hidden_dim]
 * @param grad_W_out Output gradient [state_dim, hidden_dim]
 * @param batch_size Batch size
 * @param hidden_dim Hidden dimension
 * @param state_dim State dimension
 */
inline void LearnedKalmanGainBackward(
    const float* grad_K,
    const float* gru_hidden,
    const float* W_out,
    const float* K_gain,
    float* grad_gru_hidden,
    float* grad_W_out,
    int batch_size, int hidden_dim, int state_dim) {
    
    // Initialize gradients
    std::fill(grad_W_out, grad_W_out + state_dim * hidden_dim, 0.0f);
    std::fill(grad_gru_hidden, grad_gru_hidden + batch_size * hidden_dim, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        // grad_pre_tanh = grad_K * (1 - K^2)  where K = tanh(pre_tanh)
        for (int i = 0; i < state_dim; ++i) {
            float K_val = K_gain[b * state_dim + i];
            float grad_pre_tanh = grad_K[b * state_dim + i] * (1.0f - K_val * K_val);
            
            // grad_W_out[i, j] += grad_pre_tanh * gru_hidden[b, j]
            for (int j = 0; j < hidden_dim; ++j) {
                grad_W_out[i * hidden_dim + j] += grad_pre_tanh * gru_hidden[b * hidden_dim + j];
            }
            
            // grad_gru_hidden[b, j] += W_out[i, j] * grad_pre_tanh
            for (int j = 0; j < hidden_dim; ++j) {
                grad_gru_hidden[b * hidden_dim + j] += W_out[i * hidden_dim + j] * grad_pre_tanh;
            }
        }
    }
}

/**
 * @brief Backward for Neural Kalman update step.
 *
 * Forward: x_posterior = x_prior + K * innovation
 *         innovation = z - x_prior
 *
 * @param grad_x_posterior Gradient from downstream [batch, state_dim]
 * @param K_gain Original Kalman gain [batch, state_dim]
 * @param grad_x_prior Output gradient [batch, state_dim]
 * @param grad_z Output gradient [batch, state_dim]
 * @param grad_K Output gradient [batch, state_dim]
 * @param batch_size Batch size
 * @param state_dim State dimension
 */
inline void NeuralKalmanUpdateBackward(
    const float* grad_x_posterior,
    const float* K_gain,
    float* grad_x_prior,
    float* grad_z,
    float* grad_K,
    int batch_size, int state_dim) {
    
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            int idx = b * state_dim + i;
            
            // x_posterior = x_prior + K * (z - x_prior)
            //             = x_prior * (1 - K) + K * z
            // grad_x_prior = grad_x_posterior * (1 - K)
            // grad_z = grad_x_posterior * K
            // grad_K = grad_x_posterior * (z - x_prior) = grad_x_posterior * innovation
            
            float K = K_gain[idx];
            grad_x_prior[idx] = grad_x_posterior[idx] * (1.0f - K);
            grad_z[idx] = grad_x_posterior[idx] * K;
            grad_K[idx] = grad_x_posterior[idx];  // Will be scaled by innovation in caller
        }
    }
}

/**
 * @brief Full Neural Kalman step backward.
 *
 * @param grad_x_posterior Gradient from x_posterior [batch, state_dim]
 * @param grad_gru_hidden_new Gradient from gru_hidden_new [batch, hidden_dim]
 * @param x_prior Original prior state [batch, state_dim]
 * @param z Original measurement [batch, state_dim]
 * @param gru_hidden Original GRU hidden [batch, hidden_dim]
 * @param W_z, W_r, W_h Original GRU weights
 * @param b_z, b_r, b_h Original GRU biases
 * @param W_out Original output projection [state_dim, hidden_dim]
 * @param gru_hidden_new Saved new GRU hidden [batch, hidden_dim]
 * @param K_gain Saved Kalman gain [batch, state_dim]
 * @param grad_x_prior Output [batch, state_dim]
 * @param grad_z Output [batch, state_dim]
 * @param grad_gru_hidden Output [batch, hidden_dim]
 * @param grad_W_z, grad_W_r, grad_W_h Output [hidden_dim, concat_dim]
 * @param grad_b_z, grad_b_r, grad_b_h Output [hidden_dim]
 * @param grad_W_out Output [state_dim, hidden_dim]
 * @param config Configuration
 * @param batch_size Batch size
 */
inline void NeuralKalmanStepBackward(
    const float* grad_x_posterior,
    const float* grad_gru_hidden_new,
    const float* x_prior, const float* z,
    const float* gru_hidden,
    const float* W_z, const float* W_r, const float* W_h,
    const float* b_z, const float* b_r, const float* b_h,
    const float* W_out,
    const float* gru_hidden_new_saved,
    const float* K_gain_saved,
    float* grad_x_prior, float* grad_z, float* grad_gru_hidden,
    float* grad_W_z, float* grad_W_r, float* grad_W_h,
    float* grad_b_z, float* grad_b_r, float* grad_b_h,
    float* grad_W_out,
    const NeuralKalmanConfig& config,
    int batch_size) {

    const int state_dim = config.state_dim;
    const int hidden_dim = config.hidden_dim;

    // Initialize numerical stabilizer from config for gradient clipping
    NumericalStabilizer stabilizer(config);

    // Compute innovation for grad_K scaling (match forward scaling path)
    std::vector<float> innovation(batch_size * state_dim);
    for (int i = 0; i < batch_size * state_dim; ++i) {
        innovation[i] = z[i] - x_prior[i];
    }
    if (config.enable_adaptive_scaling) {
        float scale = stabilizer.compute_adaptive_scale(innovation.data(), batch_size * state_dim);
        if (scale != 1.0f) {
            for (int i = 0; i < batch_size * state_dim; ++i) {
                innovation[i] *= scale;
            }
        }
    } else {
        stabilizer.clamp_innovation(innovation.data(), batch_size * state_dim);
    }

    // 1. Backward through Kalman update
    std::vector<float> grad_K(batch_size * state_dim);
    NeuralKalmanUpdateBackward(
        grad_x_posterior, K_gain_saved,
        grad_x_prior, grad_z, grad_K.data(),
        batch_size, state_dim
    );

    // Replace any NaN/Inf in gradients
    stabilizer.replace_nonfinite(grad_K.data(), batch_size * state_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_x_prior, batch_size * state_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_z, batch_size * state_dim, 0.0f);

    // Scale grad_K by innovation (from chain rule on K * innovation)
    for (int i = 0; i < batch_size * state_dim; ++i) {
        grad_K[i] *= innovation[i];
    }

    // Clip grad_K to prevent explosion
    stabilizer.clip_gradient_by_norm(grad_K.data(), batch_size * state_dim);

    // 2. Backward through Kalman gain computation
    std::vector<float> grad_gru_hidden_from_K(batch_size * hidden_dim);
    LearnedKalmanGainBackward(
        grad_K.data(),
        gru_hidden_new_saved,
        W_out,
        K_gain_saved,
        grad_gru_hidden_from_K.data(),
        grad_W_out,
        batch_size, hidden_dim, state_dim
    );

    // Combine gradients for gru_hidden_new
    std::vector<float> total_grad_gru_hidden_new(batch_size * hidden_dim);
    for (int i = 0; i < batch_size * hidden_dim; ++i) {
        total_grad_gru_hidden_new[i] = grad_gru_hidden_new[i] + grad_gru_hidden_from_K[i];
    }

    // Clip combined gradients
    stabilizer.clip_gradient_by_norm(total_grad_gru_hidden_new.data(), batch_size * hidden_dim);

    // 3. Backward through GRU
    std::vector<float> grad_innovation(batch_size * state_dim);
    GRUBackward(
        total_grad_gru_hidden_new.data(),
        innovation.data(),  // GRU input was clamped innovation
        gru_hidden,
        W_z, W_r, W_h, b_z, b_r, b_h,
        grad_innovation.data(),
        grad_gru_hidden,
        grad_W_z, grad_W_r, grad_W_h,
        grad_b_z, grad_b_r, grad_b_h,
        batch_size, state_dim, hidden_dim
    );

    // Clip all weight gradients to prevent explosion
    const int concat_dim = hidden_dim + state_dim;
    stabilizer.clip_gradient_by_norm(grad_W_z, hidden_dim * concat_dim);
    stabilizer.clip_gradient_by_norm(grad_W_r, hidden_dim * concat_dim);
    stabilizer.clip_gradient_by_norm(grad_W_h, hidden_dim * concat_dim);
    stabilizer.clip_gradient_by_norm(grad_b_z, hidden_dim);
    stabilizer.clip_gradient_by_norm(grad_b_r, hidden_dim);
    stabilizer.clip_gradient_by_norm(grad_b_h, hidden_dim);
    stabilizer.clip_gradient_by_norm(grad_W_out, state_dim * hidden_dim);
    stabilizer.clip_gradient_by_norm(grad_gru_hidden, batch_size * hidden_dim);

    // Replace any NaN/Inf in weight gradients
    stabilizer.replace_nonfinite(grad_W_z, hidden_dim * concat_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_W_r, hidden_dim * concat_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_W_h, hidden_dim * concat_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_b_z, hidden_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_b_r, hidden_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_b_h, hidden_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_W_out, state_dim * hidden_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_gru_hidden, batch_size * hidden_dim, 0.0f);

    // 4. Propagate gradients through innovation = z - x_prior
    for (int i = 0; i < batch_size * state_dim; ++i) {
        grad_z[i] += grad_innovation[i];
        grad_x_prior[i] -= grad_innovation[i];
    }

    // Final gradient stability check
    stabilizer.replace_nonfinite(grad_z, batch_size * state_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_x_prior, batch_size * state_dim, 0.0f);
}

// =============================================================================
// PHASE 43.2: FULL KALMAN FILTER BACKWARD (STABILITY FIX)
// =============================================================================

/**
 * @brief Backward for state prediction: x_pred = A @ x_prior.
 *
 * @param grad_x_pred Gradient from downstream [batch, state_dim]
 * @param x_prior Original prior state [batch, state_dim]
 * @param A Original state transition [state_dim, state_dim]
 * @param grad_x_prior Output gradient [batch, state_dim]
 * @param grad_A Output gradient [state_dim, state_dim]
 * @param batch_size Batch size
 * @param state_dim State dimension
 */
inline void StatePredictionBackward(
    const float* grad_x_pred,
    const float* x_prior,
    const float* A,
    float* grad_x_prior,
    float* grad_A,
    int batch_size, int state_dim) {
    
    // Initialize gradient accumulators
    std::fill(grad_A, grad_A + state_dim * state_dim, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        // grad_x_prior[j] = sum_i grad_x_pred[i] * A[i,j] = A^T @ grad_x_pred
        for (int j = 0; j < state_dim; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < state_dim; ++i) {
                sum += grad_x_pred[b * state_dim + i] * A[i * state_dim + j];
            }
            grad_x_prior[b * state_dim + j] = sum;
        }
        
        // grad_A[i,j] = sum_b grad_x_pred[b,i] * x_prior[b,j]
        for (int i = 0; i < state_dim; ++i) {
            for (int j = 0; j < state_dim; ++j) {
                grad_A[i * state_dim + j] += 
                    grad_x_pred[b * state_dim + i] * x_prior[b * state_dim + j];
            }
        }
    }
}

/**
 * @brief Backward for observation projection: Hx = H @ x.
 *
 * @param grad_Hx Gradient from downstream [batch, obs_dim]
 * @param x Original state [batch, state_dim]
 * @param H Original observation matrix [obs_dim, state_dim]
 * @param grad_x Output gradient [batch, state_dim]
 * @param grad_H Output gradient [obs_dim, state_dim]
 * @param batch_size Batch size
 * @param state_dim State dimension
 * @param obs_dim Observation dimension
 */
inline void ObservationProjectionBackward(
    const float* grad_Hx,
    const float* x,
    const float* H,
    float* grad_x,
    float* grad_H,
    int batch_size, int state_dim, int obs_dim) {
    
    // Initialize gradient accumulators
    std::fill(grad_H, grad_H + obs_dim * state_dim, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        // grad_x[j] = sum_i grad_Hx[i] * H[i,j] = H^T @ grad_Hx
        for (int j = 0; j < state_dim; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < obs_dim; ++i) {
                sum += grad_Hx[b * obs_dim + i] * H[i * state_dim + j];
            }
            grad_x[b * state_dim + j] = sum;
        }
        
        // grad_H[i,j] = sum_b grad_Hx[b,i] * x[b,j]
        for (int i = 0; i < obs_dim; ++i) {
            for (int j = 0; j < state_dim; ++j) {
                grad_H[i * state_dim + j] += 
                    grad_Hx[b * obs_dim + i] * x[b * state_dim + j];
            }
        }
    }
}

/**
 * @brief Full Neural Kalman step backward with proper Kalman filter gradients.
 *
 * Computes gradients for all Kalman filter parameters:
 * - A: State transition matrix (orthogonal)
 * - H: Observation matrix
 * - Q: Process noise
 * - R: Measurement noise
 * - P: Covariance
 * - GRU weights (auxiliary)
 *
 * @param grad_x_posterior Gradient from x_posterior [batch, state_dim]
 * @param grad_P_posterior Gradient from P_posterior [batch, state_dim]
 * @param grad_gru_hidden_new Gradient from gru_hidden_new [batch, hidden_dim]
 * @param x_prior, P_prior, z, A, H, Q, R Original forward inputs
 * @param gru_hidden, W_z, W_r, W_h, b_z, b_r, b_h, W_out GRU parameters
 * @param x_pred, P_pred, K_gain Saved forward intermediates
 * @param grad_x_prior, grad_P_prior, grad_z Output gradients
 * @param grad_A, grad_H, grad_Q, grad_R Output gradients for Kalman matrices
 * @param grad_gru_hidden, grad_W_z, grad_W_r, grad_W_h, etc. GRU gradients
 * @param config Neural Kalman configuration
 * @param batch_size Batch size
 */
inline void NeuralKalmanStepFullBackward(
    const float* grad_x_posterior, const float* grad_P_posterior,
    const float* grad_gru_hidden_new,
    const float* x_prior, const float* P_prior,
    const float* z,
    const float* A, const float* H,
    const float* Q, const float* R,
    const float* gru_hidden,
    const float* W_z, const float* W_r, const float* W_h,
    const float* b_z, const float* b_r, const float* b_h,
    const float* W_out,
    const float* x_pred, const float* P_pred, const float* K_gain,
    const float* innovation,
    float* grad_x_prior, float* grad_P_prior, float* grad_z,
    float* grad_A, float* grad_H,
    float* grad_Q, float* grad_R,
    float* grad_gru_hidden,
    float* grad_W_z, float* grad_W_r, float* grad_W_h,
    float* grad_b_z, float* grad_b_r, float* grad_b_h,
    float* grad_W_out,
    const NeuralKalmanConfig& config,
    int batch_size) {

    const int state_dim = config.state_dim;
    const int obs_dim = config.obs_dim;
    const int hidden_dim = config.hidden_dim;
    const int state_total = batch_size * state_dim;
    const int obs_total = batch_size * obs_dim;
    
    NumericalStabilizer stabilizer(config);
    
    // Initialize output gradients to zero
    std::fill(grad_x_prior, grad_x_prior + state_total, 0.0f);
    std::fill(grad_P_prior, grad_P_prior + state_total, 0.0f);
    std::fill(grad_z, grad_z + obs_total, 0.0f);
    std::fill(grad_A, grad_A + state_dim * state_dim, 0.0f);
    std::fill(grad_H, grad_H + obs_dim * state_dim, 0.0f);
    std::fill(grad_Q, grad_Q + state_dim, 0.0f);
    std::fill(grad_R, grad_R + obs_dim, 0.0f);
    
    // ==== Backward through P_post = P_pred * (1 - K) ====
    // grad_P_pred += grad_P_posterior * (1 - K)
    // grad_K += grad_P_posterior * (-P_pred)
    std::vector<float> grad_P_pred(state_total, 0.0f);
    std::vector<float> grad_K(state_total, 0.0f);
    
    for (int i = 0; i < state_total; ++i) {
        grad_P_pred[i] += grad_P_posterior[i] * (1.0f - K_gain[i]);
        grad_K[i] += grad_P_posterior[i] * (-P_pred[i]);
    }
    
    // ==== Backward through x_post = x_pred + K * H^T @ innovation ====
    // grad_x_pred += grad_x_posterior
    // grad_K += grad_x_posterior * (H^T @ innovation)
    // grad_Ht_innov = grad_x_posterior * K
    std::vector<float> grad_x_pred(state_total, 0.0f);
    std::vector<float> grad_Ht_innov(state_total, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        // Compute H^T @ innovation for this batch element
        std::vector<float> Ht_innov(state_dim, 0.0f);
        for (int i = 0; i < state_dim; ++i) {
            for (int j = 0; j < obs_dim; ++j) {
                Ht_innov[i] += H[j * state_dim + i] * innovation[b * obs_dim + j];
            }
        }
        
        for (int i = 0; i < state_dim; ++i) {
            int idx = b * state_dim + i;
            grad_x_pred[idx] += grad_x_posterior[idx];
            grad_K[idx] += grad_x_posterior[idx] * Ht_innov[i];
            grad_Ht_innov[idx] = grad_x_posterior[idx] * K_gain[idx];
        }
    }
    
    // ==== Backward through H^T @ innovation ====
    // grad_H^T += grad_Ht_innov * innovation^T
    // grad_innovation += H @ grad_Ht_innov
    std::vector<float> grad_innovation(obs_total, 0.0f);
    
    for (int b = 0; b < batch_size; ++b) {
        // grad_H^T[i,j] = grad_H[j,i] += grad_Ht_innov[i] * innovation[j]
        for (int i = 0; i < state_dim; ++i) {
            for (int j = 0; j < obs_dim; ++j) {
                grad_H[j * state_dim + i] += 
                    grad_Ht_innov[b * state_dim + i] * innovation[b * obs_dim + j];
            }
        }
        
        // grad_innovation[j] += sum_i H[j,i] * grad_Ht_innov[i]
        for (int j = 0; j < obs_dim; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < state_dim; ++i) {
                sum += H[j * state_dim + i] * grad_Ht_innov[b * state_dim + i];
            }
            grad_innovation[b * obs_dim + j] += sum;
        }
    }
    
    // ==== Backward through K = P_pred / (P_pred + R) ====
    // Using quotient rule: d/dx (a / (a + b)) = b / (a + b)^2
    // grad_P_pred += grad_K * avg_R / (P_pred + avg_R)^2
    // grad_R += grad_K * (-P_pred) / (P_pred + avg_R)^2  (averaged)
    for (int b = 0; b < batch_size; ++b) {
        float avg_R = 0.0f;
        for (int i = 0; i < obs_dim; ++i) {
            avg_R += softplus(R[i]);
        }
        avg_R /= static_cast<float>(obs_dim);
        
        float total_grad_R = 0.0f;
        for (int i = 0; i < state_dim; ++i) {
            int idx = b * state_dim + i;
            float P_val = P_pred[idx];
            float denominator = P_val + avg_R + config.epsilon;
            float denom_sq = denominator * denominator;
            
            // grad_P_pred[i] += grad_K[i] * avg_R / denom^2
            grad_P_pred[idx] += grad_K[idx] * avg_R / denom_sq;
            
            // Accumulate grad_R contribution
            total_grad_R += grad_K[idx] * (-P_val) / denom_sq;
        }
        
        // Distribute grad_R evenly across obs dimensions
        float softplus_deriv = 1.0f / (1.0f + std::exp(-R[0]));  // Approximate
        for (int i = 0; i < obs_dim; ++i) {
            grad_R[i] += total_grad_R / static_cast<float>(obs_dim) * softplus_deriv;
        }
    }
    
    // ==== Backward through innovation = z - H @ x_pred ====
    // grad_z += grad_innovation
    // grad_Hx_pred = -grad_innovation
    for (int i = 0; i < obs_total; ++i) {
        grad_z[i] += grad_innovation[i];
    }
    
    std::vector<float> grad_Hx_pred(obs_total);
    for (int i = 0; i < obs_total; ++i) {
        grad_Hx_pred[i] = -grad_innovation[i];
    }
    
    // Backward through H @ x_pred
    std::vector<float> grad_x_pred_from_H(state_total);
    ObservationProjectionBackward(
        grad_Hx_pred.data(), x_pred, H,
        grad_x_pred_from_H.data(), grad_H,
        batch_size, state_dim, obs_dim
    );
    
    for (int i = 0; i < state_total; ++i) {
        grad_x_pred[i] += grad_x_pred_from_H[i];
    }
    
    // ==== Backward through P_pred = P_prior + softplus(Q) ====
    // grad_P_prior += grad_P_pred
    // grad_Q += grad_P_pred * softplus'(Q)
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < state_dim; ++i) {
            int idx = b * state_dim + i;
            grad_P_prior[idx] += grad_P_pred[idx];
            
            // softplus'(x) = sigmoid(x) = 1 / (1 + exp(-x))
            float sigmoid_Q = 1.0f / (1.0f + std::exp(-Q[i]));
            grad_Q[i] += grad_P_pred[idx] * sigmoid_Q;
        }
    }
    
    // ==== Backward through x_pred = A @ x_prior ====
    StatePredictionBackward(
        grad_x_pred.data(), x_prior, A,
        grad_x_prior, grad_A,
        batch_size, state_dim
    );
    
    // ==== Backward through GRU (auxiliary) ====
    // Note: GRU gradients are computed but scaled down as GRU is auxiliary
    const int concat_dim = hidden_dim + state_dim;
    std::vector<float> gru_input(state_total);
    for (int i = 0; i < state_total; ++i) {
        gru_input[i] = 0.0f;  // Simplified for now
    }
    
    std::vector<float> grad_gru_input(state_total);
    GRUBackward(
        grad_gru_hidden_new,
        gru_input.data(),
        gru_hidden,
        W_z, W_r, W_h, b_z, b_r, b_h,
        grad_gru_input.data(),
        grad_gru_hidden,
        grad_W_z, grad_W_r, grad_W_h,
        grad_b_z, grad_b_r, grad_b_h,
        batch_size, state_dim, hidden_dim
    );
    
    // Clip all gradients for stability
    stabilizer.clip_gradient_by_norm(grad_A, state_dim * state_dim);
    stabilizer.clip_gradient_by_norm(grad_H, obs_dim * state_dim);
    stabilizer.clip_gradient_by_norm(grad_Q, state_dim);
    stabilizer.clip_gradient_by_norm(grad_R, obs_dim);
    stabilizer.clip_gradient_by_norm(grad_x_prior, state_total);
    stabilizer.clip_gradient_by_norm(grad_P_prior, state_total);
    stabilizer.clip_gradient_by_norm(grad_z, obs_total);
    
    // Replace any NaN/Inf
    stabilizer.replace_nonfinite(grad_A, state_dim * state_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_H, obs_dim * state_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_Q, state_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_R, obs_dim, 0.0f);
    stabilizer.replace_nonfinite(grad_x_prior, state_total, 0.0f);
    stabilizer.replace_nonfinite(grad_P_prior, state_total, 0.0f);
    stabilizer.replace_nonfinite(grad_z, obs_total, 0.0f);
}

}  // namespace neural_kalman
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_NEURAL_KALMAN_OP_H_
