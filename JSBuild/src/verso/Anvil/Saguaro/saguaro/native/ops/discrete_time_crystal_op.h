// saguaro.native/ops/discrete_time_crystal_op.h
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
 * @file discrete_time_crystal_op.h
 * @brief Phase 38: Discrete Time Crystal State Protection
 *
 * Stabilizes TimeCrystalSequenceBlock hidden states using DTC principles
 * for the `mamba_timecrystal_wlam_moe_hybrid` block pattern (Block 1).
 *
 * Key Features:
 *   - Floquet Driving: Periodic kicks that induce period-doubled response
 *   - Many-Body Localization: Prevents thermalization via disorder
 *   - Phase Coherence: Maintains quantum correlations longer
 *
 * Research Basis: "Phase Transitions in DTCs" (Nature Physics 2024)
 *
 * Integration Points:
 *   - Block 1: TimeCrystalSequenceBlock + KalmanBlock
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef SAGUARO_NATIVE_OPS_DISCRETE_TIME_CRYSTAL_OP_H_
#define SAGUARO_NATIVE_OPS_DISCRETE_TIME_CRYSTAL_OP_H_

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

namespace saguaro {
namespace dtc {

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * @brief Configuration for Discrete Time Crystal protection.
 */
struct DTCConfig {
    int floquet_period;           // T = period of driving (in steps)
    float coupling_J;             // Heisenberg coupling strength
    float disorder_W;             // Disorder strength for MBL
    float pi_pulse_error;         // Imperfection in π rotations (ε)
    bool use_prethermal_phase;    // Enable prethermal DTC regime
    int num_floquet_cycles;       // Number of Floquet cycles per step
    uint32_t seed;                // Random seed for disorder
    
    DTCConfig()
        : floquet_period(4)
        , coupling_J(1.0f)
        , disorder_W(0.5f)
        , pi_pulse_error(0.01f)
        , use_prethermal_phase(true)
        , num_floquet_cycles(1)
        , seed(42) {}
};

// =============================================================================
// CORE DTC OPERATIONS
// =============================================================================

/**
 * @brief Apply Hamiltonian evolution for half period.
 *
 * Simulates exp(-i H t/2) using first-order approximation:
 *   state ← state + (t/2) * H @ state
 *
 * For real-valued states, this becomes:
 *   state ← (I + (t/2) * H_eff) @ state
 *
 * @param hidden_state Hidden state [batch, seq, state_dim]
 * @param H_evolution Effective Hamiltonian [state_dim, state_dim]
 * @param time Evolution time
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param state_dim State dimension
 */
inline void ApplyHamiltonianEvolution(
    float* hidden_state,
    const float* H_evolution,
    float time,
    int batch_size, int seq_len, int state_dim) {
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int base_idx = b * seq_len * state_dim + t * state_dim;
            
            // Allocate temporary for matrix-vector product
            std::vector<float> evolved(state_dim, 0.0f);
            
            // Compute H @ state
            for (int i = 0; i < state_dim; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < state_dim; ++j) {
                    sum += H_evolution[i * state_dim + j] * 
                           hidden_state[base_idx + j];
                }
                // Apply time evolution: state + time * H @ state
                evolved[i] = hidden_state[base_idx + i] + time * sum;
            }
            
            // Copy back
            for (int i = 0; i < state_dim; ++i) {
                hidden_state[base_idx + i] = evolved[i];
            }
        }
    }
}

/**
 * @brief Apply π-pulse rotation (global spin flip) for DTC dynamics.
 *
 * The π-pulse creates the period-doubling characteristic of DTCs:
 *   R_x(π + ε) ≈ cos(π + ε) * I + sin(π + ε) * X
 *
 * For real states, this approximates a reflection with small error ε.
 *
 * @param state Hidden state [batch, seq, state_dim]
 * @param error π-pulse imperfection ε
 * @param batch_size Batch size
 * @param seq_len Sequence length  
 * @param state_dim State dimension
 */
inline void ApplyPiPulse(
    float* state,
    float error,
    int batch_size, int seq_len, int state_dim) {
    
    const float angle = M_PI + error;
    const float cos_angle = std::cos(angle);
    const float sin_angle = std::sin(angle);
    
    const int total_size = batch_size * seq_len * state_dim;
    
    int i = 0;
    
#if defined(__AVX512F__)
    const __m512 cos_v = _mm512_set1_ps(cos_angle);
    const __m512 sin_v = _mm512_set1_ps(sin_angle);
    
    for (; i + 16 <= total_size; i += 16) {
        __m512 val = _mm512_loadu_ps(&state[i]);
        
        // R(π + ε): approximately flips sign with small perturbation
        // state ← cos(angle) * state + sin(angle) * tanh(state)
        // Simplified: state ← -state + 2*ε*state = -(1-2ε)*state
        __m512 result = _mm512_mul_ps(cos_v, val);
        // Add sin term as perturbation
        __m512 perturb = _mm512_mul_ps(sin_v, val);
        result = _mm512_add_ps(result, perturb);
        
        _mm512_storeu_ps(&state[i], result);
    }
#elif defined(__AVX2__)
    const __m256 cos_v = _mm256_set1_ps(cos_angle);
    const __m256 sin_v = _mm256_set1_ps(sin_angle);
    
    for (; i + 8 <= total_size; i += 8) {
        __m256 val = _mm256_loadu_ps(&state[i]);
        __m256 result = _mm256_mul_ps(cos_v, val);
        __m256 perturb = _mm256_mul_ps(sin_v, val);
        result = _mm256_add_ps(result, perturb);
        _mm256_storeu_ps(&state[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t cos_v = vdupq_n_f32(cos_angle);
    const float32x4_t sin_v = vdupq_n_f32(sin_angle);
    
    for (; i + 4 <= total_size; i += 4) {
        float32x4_t val = vld1q_f32(&state[i]);
        float32x4_t result = vmulq_f32(cos_v, val);
        float32x4_t perturb = vmulq_f32(sin_v, val);
        result = vaddq_f32(result, perturb);
        vst1q_f32(&state[i], result);
    }
#endif
    
    // Scalar fallback
    for (; i < total_size; ++i) {
        state[i] = cos_angle * state[i] + sin_angle * state[i];
    }
}

/**
 * @brief Apply Many-Body Localization disorder for thermalization prevention.
 *
 * MBL introduces random on-site potentials that prevent energy diffusion:
 *   state_i ← state_i + W * h_i * state_i
 * where h_i ~ Uniform(-1, 1) is quenched disorder.
 *
 * @param state Hidden state [batch, seq, state_dim]
 * @param disorder_W Disorder strength W
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param state_dim State dimension
 * @param seed Random seed for reproducible disorder
 */
inline void ApplyMBLDisorder(
    float* state,
    float disorder_W,
    int batch_size, int seq_len, int state_dim,
    uint32_t seed = 42) {
    
    // Generate quenched disorder (same for all timesteps, varies by position)
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> disorder(-1.0f, 1.0f);
    
    std::vector<float> h_disorder(state_dim);
    for (int i = 0; i < state_dim; ++i) {
        h_disorder[i] = disorder(rng);
    }
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int base_idx = b * seq_len * state_dim + t * state_dim;
            
            for (int i = 0; i < state_dim; ++i) {
                // On-site disorder term
                state[base_idx + i] += disorder_W * h_disorder[i] * state[base_idx + i];
            }
        }
    }
}

/**
 * @brief Apply nearest-neighbor Heisenberg coupling.
 *
 * Implements the interaction term of the Heisenberg Hamiltonian:
 *   H_int = J * Σ_i (σ_i · σ_{i+1})
 *
 * For real states, this becomes a correlation-based update:
 *   state_i ← state_i + J * (state_{i-1} + state_{i+1}) * state_i
 *
 * @param state Hidden state [batch, seq, state_dim]
 * @param coupling_J Heisenberg coupling strength
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param state_dim State dimension
 */
inline void ApplyHeisenbergCoupling(
    float* state,
    float coupling_J,
    int batch_size, int seq_len, int state_dim) {
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int base_idx = b * seq_len * state_dim + t * state_dim;
            
            std::vector<float> coupling_contrib(state_dim, 0.0f);
            
            for (int i = 0; i < state_dim; ++i) {
                float neighbor_sum = 0.0f;
                
                // Left neighbor (periodic boundary)
                int left = (i - 1 + state_dim) % state_dim;
                neighbor_sum += state[base_idx + left];
                
                // Right neighbor (periodic boundary)
                int right = (i + 1) % state_dim;
                neighbor_sum += state[base_idx + right];
                
                coupling_contrib[i] = coupling_J * neighbor_sum * state[base_idx + i];
            }
            
            // Apply coupling
            for (int i = 0; i < state_dim; ++i) {
                state[base_idx + i] += 0.1f * coupling_contrib[i];  // Scaled coupling
            }
        }
    }
}

/**
 * @brief Full DTC-stabilized evolution step.
 *
 * Implements the Floquet operator:
 *   U_F = exp(-i H T/2) * R_x(π + ε) * exp(-i H T/2)
 *
 * @param hidden_state Hidden state [batch, seq, state_dim]
 * @param H_evolution Effective Hamiltonian [state_dim, state_dim]
 * @param config DTC configuration
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param state_dim State dimension
 */
inline void DTCStabilizedEvolution(
    float* hidden_state,
    const float* H_evolution,
    const DTCConfig& config,
    int batch_size, int seq_len, int state_dim) {
    
    const float half_period = config.floquet_period / 2.0f;
    
    for (int cycle = 0; cycle < config.num_floquet_cycles; ++cycle) {
        // Step 1: First half-evolution under Hamiltonian
        ApplyHamiltonianEvolution(
            hidden_state, H_evolution, half_period,
            batch_size, seq_len, state_dim
        );
        
        // Step 2: Apply Heisenberg coupling
        if (config.coupling_J > 0) {
            ApplyHeisenbergCoupling(
                hidden_state, config.coupling_J,
                batch_size, seq_len, state_dim
            );
        }
        
        // Step 3: π-pulse with controlled error for DTC dynamics
        ApplyPiPulse(
            hidden_state, config.pi_pulse_error,
            batch_size, seq_len, state_dim
        );
        
        // Step 4: Second half-evolution
        ApplyHamiltonianEvolution(
            hidden_state, H_evolution, half_period,
            batch_size, seq_len, state_dim
        );
        
        // Step 5: Many-body localization disorder (prevents thermalization)
        if (config.disorder_W > 0) {
            ApplyMBLDisorder(
                hidden_state, config.disorder_W,
                batch_size, seq_len, state_dim,
                config.seed + cycle
            );
        }
    }
}

/**
 * @brief Compute DTC order parameter (magnetization oscillation).
 *
 * The DTC phase is characterized by period-doubled oscillations:
 *   M(t) = ⟨Σ_i σ_i^z(t)⟩
 *
 * Returns the magnetization at each timestep.
 *
 * @param state Hidden state [batch, seq, state_dim]
 * @param magnetization Output magnetization [batch, seq]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param state_dim State dimension
 */
inline void ComputeDTCOrderParameter(
    const float* state,
    float* magnetization,
    int batch_size, int seq_len, int state_dim) {
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < seq_len; ++t) {
            int base_idx = b * seq_len * state_dim + t * state_dim;
            
            // Sum over all "spins" (state dimensions)
            float mag = 0.0f;
            for (int i = 0; i < state_dim; ++i) {
                mag += state[base_idx + i];
            }
            
            magnetization[b * seq_len + t] = mag / state_dim;
        }
    }
}

/**
 * @brief Check if system is in DTC phase (period-doubling detected).
 *
 * Computes the Fourier component at frequency ω/2 (half the drive frequency).
 * Strong peak indicates DTC order.
 *
 * @param magnetization Magnetization time series [seq_len]
 * @param seq_len Sequence length
 * @param floquet_period Floquet driving period
 * @return DTC order strength (0 to 1)
 */
inline float ComputeDTCPhaseOrder(
    const float* magnetization,
    int seq_len,
    int floquet_period) {
    
    // Compute Fourier component at period-2 frequency
    float cos_sum = 0.0f, sin_sum = 0.0f;
    const float omega_half = M_PI / floquet_period;  // ω/2
    
    for (int t = 0; t < seq_len; ++t) {
        float phase = omega_half * t;
        cos_sum += magnetization[t] * std::cos(phase);
        sin_sum += magnetization[t] * std::sin(phase);
    }
    
    // Fourier amplitude at ω/2
    float amplitude = std::sqrt(cos_sum * cos_sum + sin_sum * sin_sum) / seq_len;
    
    // Normalize to [0, 1]
    return std::min(1.0f, amplitude * 2.0f);
}

}  // namespace dtc
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_DISCRETE_TIME_CRYSTAL_OP_H_
