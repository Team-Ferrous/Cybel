// highnoon/_native/ops/qmamba_op.h
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
 * @file qmamba_op.h
 * @brief Phase 37: QMamba - Quantum-Enhanced Selective State Space Model
 *
 * Extends Mamba SSM with quantum-enhanced state transitions for the
 * `mamba_timecrystal_wlam_moe_hybrid` block pattern (Blocks 0 and 3).
 *
 * Key Features:
 *   - Quantum State Superposition: K parallel state paths exist simultaneously
 *   - Entanglement-Aware Updates: VQC encodes inter-position correlations
 *   - Amplitude-Weighted Selection: Born rule for selective scanning
 *
 * Research Basis: QMamba (Koelle et al., ICAART 2025), Q-SSM (arXiv 2025)
 *
 * Integration Points:
 *   - Block 0: SpatialBlock (Mamba SSM) primary instance
 *   - Block 3: SpatialBlock (Mamba SSM alt) secondary instance
 *
 * Complexity: O(n * K * state_dim) where K is num_superposition_states
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef HIGHNOON_NATIVE_OPS_QMAMBA_OP_H_
#define HIGHNOON_NATIVE_OPS_QMAMBA_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

// SIMD intrinsics for cross-architecture vectorization
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace hsmn {
namespace qmamba {

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * @brief Configuration for QMamba quantum-enhanced SSM.
 */
struct QMambaConfig {
    int num_superposition_states;   // K parallel quantum states (default: 4)
    int entanglement_depth;         // VQC entanglement layers (default: 2)
    float entanglement_strength;    // α ∈ [0,1] for quantum mixing (default: 0.5)
    bool use_amplitude_selection;   // Born rule vs softmax (default: true)
    float gumbel_temperature;       // Temperature for Gumbel-softmax collapse
    uint32_t seed;                  // Random seed for reproducibility
    
    QMambaConfig()
        : num_superposition_states(4)
        , entanglement_depth(2)
        , entanglement_strength(0.5f)
        , use_amplitude_selection(true)
        , gumbel_temperature(1.0f)
        , seed(42) {}
};

// =============================================================================
// CORE QMAMBA OPERATIONS
// =============================================================================

/**
 * @brief Initialize superposition states via Hadamard-like distribution.
 *
 * Creates K parallel state paths with equal initial amplitudes.
 * Each path gets a slight perturbation for diversity.
 *
 * @param h_super Output superposition states [batch, K, state_dim]
 * @param batch_size Batch size
 * @param num_states Number of superposition states K
 * @param state_dim State dimension
 * @param seed Random seed
 */
inline void InitSuperposition(
    float* h_super,
    int batch_size, int num_states, int state_dim,
    uint32_t seed = 42) {
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 0.01f);
    
    // Initialize with small random values for symmetry breaking
    const float base_amplitude = 1.0f / std::sqrt(static_cast<float>(num_states));
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int k = 0; k < num_states; ++k) {
            std::mt19937 local_rng(seed + b * num_states + k);
            std::normal_distribution<float> local_normal(0.0f, 0.01f);
            
            for (int n = 0; n < state_dim; ++n) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                h_super[idx] = base_amplitude + local_normal(local_rng);
            }
        }
    }
}

/**
 * @brief Apply VQC-inspired entanglement layer between state paths.
 *
 * Uses parameterized rotations to create correlations between paths:
 *   RY(θ) rotation followed by CNOT-like correlation
 *
 * @param states State paths [batch, K, state_dim]
 * @param rotation_angles VQC angles [entanglement_depth, K]
 * @param entanglement_strength Mixing strength α ∈ [0,1]
 * @param batch_size Batch size
 * @param num_states Number of superposition states
 * @param state_dim State dimension
 * @param layer Current entanglement layer index
 */
inline void ApplyEntanglementLayer(
    float* states,
    const float* rotation_angles,
    float entanglement_strength,
    int batch_size, int num_states, int state_dim,
    int layer) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // First: Apply parameterized RY rotations to each path
        for (int k = 0; k < num_states; ++k) {
            float theta = rotation_angles[layer * num_states + k];
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            
            for (int n = 0; n < state_dim; ++n) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                // RY rotation: [cos(θ/2), sin(θ/2)] applied to amplitude
                float val = states[idx];
                states[idx] = cos_t * val + sin_t * std::tanh(val);
            }
        }
        
        // Second: CNOT-like entanglement between adjacent paths
        // Creates correlations: |k, k+1⟩ → α|k⟩|k+1⟩ + (1-α)|k+1⟩|k⟩
        for (int k = 0; k < num_states - 1; ++k) {
            int idx_k = b * num_states * state_dim + k * state_dim;
            int idx_k1 = b * num_states * state_dim + (k + 1) * state_dim;
            
            for (int n = 0; n < state_dim; ++n) {
                float val_k = states[idx_k + n];
                float val_k1 = states[idx_k1 + n];
                
                // Controlled phase interaction
                float mix = entanglement_strength * val_k * val_k1;
                states[idx_k + n] = val_k + mix;
                states[idx_k1 + n] = val_k1 - mix;
            }
        }
        
        // Third: Ring closure - entangle last with first for cyclic coherence
        if (num_states > 2) {
            int idx_last = b * num_states * state_dim + (num_states - 1) * state_dim;
            int idx_first = b * num_states * state_dim;
            
            for (int n = 0; n < state_dim; ++n) {
                float val_last = states[idx_last + n];
                float val_first = states[idx_first + n];
                
                float mix = 0.5f * entanglement_strength * val_last * val_first;
                states[idx_last + n] = val_last + mix;
                states[idx_first + n] = val_first + mix;
            }
        }
    }
}

/**
 * @brief Collapse superposition paths via Born rule (amplitude-weighted selection).
 *
 * Computes path probabilities from squared amplitudes and selects via
 * weighted combination:
 *   prob_k = |ψ_k|² / Σ|ψ_k|²
 *   h_out = Σ prob_k * h_k
 *
 * @param h_super Superposition states [batch, K, state_dim]
 * @param h_collapsed Output collapsed state [batch, state_dim]
 * @param batch_size Batch size
 * @param num_states Number of superposition states
 * @param state_dim State dimension
 */
inline void BornRuleCollapse(
    const float* h_super,
    float* h_collapsed,
    int batch_size, int num_states, int state_dim) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // Compute amplitudes (sum of squares per path)
        std::vector<float> amplitudes(num_states, 0.0f);
        
        for (int k = 0; k < num_states; ++k) {
            float amp_sq = 0.0f;
            for (int n = 0; n < state_dim; ++n) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                amp_sq += h_super[idx] * h_super[idx];
            }
            amplitudes[k] = amp_sq;
        }
        
        // Normalize to probabilities (Born rule)
        float total = 0.0f;
        for (int k = 0; k < num_states; ++k) {
            total += amplitudes[k];
        }
        total = std::max(total, 1e-10f);
        
        for (int k = 0; k < num_states; ++k) {
            amplitudes[k] /= total;
        }
        
        // Weighted collapse
        for (int n = 0; n < state_dim; ++n) {
            float collapsed_val = 0.0f;
            for (int k = 0; k < num_states; ++k) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                collapsed_val += amplitudes[k] * h_super[idx];
            }
            h_collapsed[b * state_dim + n] = collapsed_val;
        }
    }
}

/**
 * @brief Collapse superposition paths via Gumbel-Softmax (differentiable).
 *
 * Uses Gumbel-Softmax trick for gradient-friendly path selection:
 *   logits_k = log(|ψ_k|²) + Gumbel_noise
 *   weights_k = softmax(logits_k / τ)
 *
 * @param h_super Superposition states [batch, K, state_dim]
 * @param path_logits Path selection logits [batch, K]
 * @param h_collapsed Output collapsed state [batch, state_dim]
 * @param batch_size Batch size
 * @param num_states Number of superposition states
 * @param state_dim State dimension
 * @param temperature Gumbel-Softmax temperature τ
 * @param seed Random seed for Gumbel noise
 */
inline void GumbelSoftmaxCollapse(
    const float* h_super,
    const float* path_logits,
    float* h_collapsed,
    int batch_size, int num_states, int state_dim,
    float temperature = 1.0f,
    uint32_t seed = 42) {
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        std::mt19937 local_rng(seed + b);
        std::uniform_real_distribution<float> local_uniform(0.0f, 1.0f);
        
        // Compute Gumbel-perturbed logits
        std::vector<float> gumbel_logits(num_states);
        float max_logit = -1e10f;
        
        for (int k = 0; k < num_states; ++k) {
            // Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
            float u = local_uniform(local_rng);
            u = std::max(u, 1e-10f);
            float gumbel_noise = -std::log(-std::log(u));
            
            gumbel_logits[k] = path_logits[b * num_states + k] + gumbel_noise;
            max_logit = std::max(max_logit, gumbel_logits[k]);
        }
        
        // Softmax with temperature
        float sum_exp = 0.0f;
        for (int k = 0; k < num_states; ++k) {
            gumbel_logits[k] = std::exp((gumbel_logits[k] - max_logit) / temperature);
            sum_exp += gumbel_logits[k];
        }
        
        for (int k = 0; k < num_states; ++k) {
            gumbel_logits[k] /= sum_exp;
        }
        
        // Weighted collapse
        for (int n = 0; n < state_dim; ++n) {
            float collapsed_val = 0.0f;
            for (int k = 0; k < num_states; ++k) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                collapsed_val += gumbel_logits[k] * h_super[idx];
            }
            h_collapsed[b * state_dim + n] = collapsed_val;
        }
    }
}

/**
 * @brief Full QMamba selective scan with quantum superposition.
 *
 * Extends standard Mamba scan with K parallel state paths:
 *   1. Initialize K superposition states
 *   2. Apply entanglement layers for correlations
 *   3. Run parallel SSM scans on each path
 *   4. Collapse to single output via Born rule/Gumbel
 *
 * @param x Input sequence [batch, seq_len, d_inner]
 * @param h_super Superposed states [batch, K, d_inner, state_dim]
 * @param A_log Log of decay rates [d_inner, state_dim]
 * @param B B projections [batch, seq_len, state_dim]
 * @param C C projections [batch, seq_len, state_dim]
 * @param dt Delta timesteps [batch, seq_len, d_inner]
 * @param rotation_angles VQC angles [entanglement_depth, K]
 * @param config QMamba configuration
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param d_inner Inner dimension
 * @param state_dim State dimension
 */
inline void QMambaSelectiveScan(
    const float* x,
    float* h_super,
    const float* A_log,
    const float* B, const float* C, const float* dt,
    const float* rotation_angles,
    const QMambaConfig& config,
    int batch_size, int seq_len, int d_inner, int state_dim) {
    
    const int K = config.num_superposition_states;
    
    // Initialize superposition states
    InitSuperposition(h_super, batch_size, K, d_inner * state_dim, config.seed);
    
    // Apply entanglement layers
    for (int layer = 0; layer < config.entanglement_depth; ++layer) {
        ApplyEntanglementLayer(
            h_super, rotation_angles, config.entanglement_strength,
            batch_size, K, d_inner * state_dim, layer
        );
    }
    
    // Process sequence with parallel SSM on each superposition path
    for (int t = 0; t < seq_len; ++t) {
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int k = 0; k < K; ++k) {
                // Each path has slightly different effective dt for exploration
                float path_scale = 1.0f + 0.1f * (k - K / 2.0f) / K;
                
                for (int d = 0; d < d_inner; ++d) {
                    float dt_val = dt[b * seq_len * d_inner + t * d_inner + d] * path_scale;
                    float x_val = x[b * seq_len * d_inner + t * d_inner + d];
                    
                    for (int n = 0; n < state_dim; ++n) {
                        int h_idx = b * K * d_inner * state_dim +
                                    k * d_inner * state_dim +
                                    d * state_dim + n;
                        
                        // Discretize A: A_disc = exp(dt * A_log)
                        float A_disc = std::exp(dt_val * A_log[d * state_dim + n]);
                        float B_val = B[b * seq_len * state_dim + t * state_dim + n];
                        
                        // SSM update
                        h_super[h_idx] = A_disc * h_super[h_idx] + B_val * x_val;
                    }
                }
            }
        }
    }
}

/**
 * @brief Compute output from QMamba superposition states.
 *
 * @param h_super Superposed states [batch, K, d_inner, state_dim]
 * @param C C projections [batch, state_dim]
 * @param D Skip connection [d_inner]
 * @param x Input [batch, d_inner]
 * @param output Output [batch, d_inner]
 * @param config QMamba configuration
 * @param batch_size Batch size
 * @param d_inner Inner dimension
 * @param state_dim State dimension
 */
inline void QMambaOutput(
    const float* h_super,
    const float* C, const float* D, const float* x,
    float* output,
    const QMambaConfig& config,
    int batch_size, int d_inner, int state_dim) {
    
    const int K = config.num_superposition_states;
    
    // Workspace for collapsed states
    std::vector<float> h_collapsed(batch_size * d_inner * state_dim);
    
    // Collapse superposition to single state per (batch, d_inner)
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < d_inner; ++d) {
            // Compute Born rule weights for this (b, d) position
            std::vector<float> amplitudes(K);
            float total_amp = 0.0f;
            
            for (int k = 0; k < K; ++k) {
                float amp_sq = 0.0f;
                for (int n = 0; n < state_dim; ++n) {
                    int h_idx = b * K * d_inner * state_dim +
                                k * d_inner * state_dim +
                                d * state_dim + n;
                    amp_sq += h_super[h_idx] * h_super[h_idx];
                }
                amplitudes[k] = amp_sq;
                total_amp += amp_sq;
            }
            
            total_amp = std::max(total_amp, 1e-10f);
            
            // Collapse with normalized weights
            for (int n = 0; n < state_dim; ++n) {
                float collapsed = 0.0f;
                for (int k = 0; k < K; ++k) {
                    int h_idx = b * K * d_inner * state_dim +
                                k * d_inner * state_dim +
                                d * state_dim + n;
                    collapsed += (amplitudes[k] / total_amp) * h_super[h_idx];
                }
                h_collapsed[b * d_inner * state_dim + d * state_dim + n] = collapsed;
            }
        }
    }
    
    // Compute output: y = C @ h_collapsed + D * x
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int d = 0; d < d_inner; ++d) {
            float y_val = 0.0f;
            
            for (int n = 0; n < state_dim; ++n) {
                y_val += C[b * state_dim + n] *
                         h_collapsed[b * d_inner * state_dim + d * state_dim + n];
            }
            
            // Skip connection
            y_val += D[d] * x[b * d_inner + d];
            
            output[b * d_inner + d] = y_val;
        }
    }
}

// =============================================================================
// SIMD-OPTIMIZED VARIANTS
// =============================================================================

#if defined(__AVX2__)
/**
 * @brief AVX2-optimized entanglement layer application.
 */
inline void ApplyEntanglementLayerAVX2(
    float* states,
    const float* rotation_angles,
    float entanglement_strength,
    int batch_size, int num_states, int state_dim,
    int layer) {
    
    const __m256 alpha = _mm256_set1_ps(entanglement_strength);
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // Apply RY rotations
        for (int k = 0; k < num_states; ++k) {
            float theta = rotation_angles[layer * num_states + k];
            __m256 cos_t = _mm256_set1_ps(std::cos(theta));
            __m256 sin_t = _mm256_set1_ps(std::sin(theta));
            
            int base_idx = b * num_states * state_dim + k * state_dim;
            int n = 0;
            
            for (; n + 8 <= state_dim; n += 8) {
                __m256 val = _mm256_loadu_ps(&states[base_idx + n]);
                
                // Approximate tanh with fast approximation
                __m256 tanh_val = val;  // Simplified: tanh(x) ≈ x for small x
                
                __m256 result = _mm256_add_ps(
                    _mm256_mul_ps(cos_t, val),
                    _mm256_mul_ps(sin_t, tanh_val)
                );
                
                _mm256_storeu_ps(&states[base_idx + n], result);
            }
            
            // Scalar remainder
            for (; n < state_dim; ++n) {
                float val = states[base_idx + n];
                states[base_idx + n] = std::cos(theta) * val + std::sin(theta) * std::tanh(val);
            }
        }
        
        // CNOT-like entanglement
        for (int k = 0; k < num_states - 1; ++k) {
            int idx_k = b * num_states * state_dim + k * state_dim;
            int idx_k1 = b * num_states * state_dim + (k + 1) * state_dim;
            
            int n = 0;
            for (; n + 8 <= state_dim; n += 8) {
                __m256 val_k = _mm256_loadu_ps(&states[idx_k + n]);
                __m256 val_k1 = _mm256_loadu_ps(&states[idx_k1 + n]);
                
                __m256 mix = _mm256_mul_ps(alpha, _mm256_mul_ps(val_k, val_k1));
                
                _mm256_storeu_ps(&states[idx_k + n], _mm256_add_ps(val_k, mix));
                _mm256_storeu_ps(&states[idx_k1 + n], _mm256_sub_ps(val_k1, mix));
            }
            
            for (; n < state_dim; ++n) {
                float val_k = states[idx_k + n];
                float val_k1 = states[idx_k1 + n];
                float mix = entanglement_strength * val_k * val_k1;
                states[idx_k + n] = val_k + mix;
                states[idx_k1 + n] = val_k1 - mix;
            }
        }
    }
}
#endif  // __AVX2__

#if defined(__AVX512F__)
/**
 * @brief AVX-512 optimized Born rule collapse.
 */
inline void BornRuleCollapseAVX512(
    const float* h_super,
    float* h_collapsed,
    int batch_size, int num_states, int state_dim) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        // Compute amplitudes
        std::vector<float> amplitudes(num_states, 0.0f);
        
        for (int k = 0; k < num_states; ++k) {
            __m512 sum = _mm512_setzero_ps();
            int n = 0;
            
            for (; n + 16 <= state_dim; n += 16) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                __m512 val = _mm512_loadu_ps(&h_super[idx]);
                sum = _mm512_fmadd_ps(val, val, sum);
            }
            
            amplitudes[k] = _mm512_reduce_add_ps(sum);
            
            // Scalar remainder
            for (; n < state_dim; ++n) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                amplitudes[k] += h_super[idx] * h_super[idx];
            }
        }
        
        // Normalize
        float total = 0.0f;
        for (int k = 0; k < num_states; ++k) total += amplitudes[k];
        total = std::max(total, 1e-10f);
        for (int k = 0; k < num_states; ++k) amplitudes[k] /= total;
        
        // Weighted collapse with AVX-512
        int n = 0;
        for (; n + 16 <= state_dim; n += 16) {
            __m512 collapsed = _mm512_setzero_ps();
            
            for (int k = 0; k < num_states; ++k) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                __m512 h_val = _mm512_loadu_ps(&h_super[idx]);
                __m512 weight = _mm512_set1_ps(amplitudes[k]);
                collapsed = _mm512_fmadd_ps(weight, h_val, collapsed);
            }
            
            _mm512_storeu_ps(&h_collapsed[b * state_dim + n], collapsed);
        }
        
        // Scalar remainder
        for (; n < state_dim; ++n) {
            float val = 0.0f;
            for (int k = 0; k < num_states; ++k) {
                int idx = b * num_states * state_dim + k * state_dim + n;
                val += amplitudes[k] * h_super[idx];
            }
            h_collapsed[b * state_dim + n] = val;
        }
    }
}
#endif  // __AVX512F__

}  // namespace qmamba
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_QMAMBA_OP_H_
