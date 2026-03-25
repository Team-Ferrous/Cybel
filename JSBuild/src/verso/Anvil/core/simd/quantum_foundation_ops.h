// highnoon/_native/ops/quantum_foundation_ops.h
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file quantum_foundation_ops.h
 * @brief Unified Quantum Foundation Operations for HighNoon Framework
 *
 * This header consolidates all 15 quantum mechanisms into a single unified
 * interface with shared VQC primitives, SIMD optimization, and common
 * quantum computing abstractions.
 *
 * Supported Quantum Operation Types:
 * - EMBEDDING:          Holographic token embeddings (FFT-bind)
 * - POSITION_ENCODING:  Floquet/SU(2) position encoding
 * - LM_HEAD:           VQC-based language model head (Born rule)
 * - EXPERT:            Unitary expert networks (Cayley transform)
 * - NORM:              Unitary/Stiefel normalization
 * - RESIDUAL:          Unitary residual connections
 * - MEASUREMENT:       Born rule measurement with collapse
 * - COHERENCE_BUS:     Phase-coherent state transport
 * - TELEPORT_BUS:      Quantum teleportation-inspired state transfer
 * - VQC:               Variational Quantum Circuit simulation
 * - TENSOR_RING_VQC:   Tensor ring VQC with neural BP mitigation
 * - CRYSTALLIZATION:   Quantum state crystallization
 * - FIDELITY_LOSS:     Quantum fidelity regularization
 * - DROPOUT:           Quantum measurement dropout
 * - CURRICULUM:        Spectral-aware quantum curriculum
 *
 * Shared Primitives:
 * - VQC gates: RY, RZ, CNOT (simulated)
 * - Born rule: probability extraction from quantum states
 * - Partial trace: density matrix operations
 * - FFT: Holographic binding/unbinding
 * - Cayley transform: Skew-symmetric → Unitary mapping
 *
 * SIMD Support:
 * - AVX-512: 16-wide vectorization
 * - AVX2:    8-wide vectorization
 * - NEON:    4-wide vectorization
 * - Scalar fallback
 *
 * Thread Safety: All functions are reentrant with no shared state.
 *
 * @note Phase 3 of V2 Performance Optimization.
 *       See V2_PERFORMANCE_OPTIMIZATION_ANALYSIS.md for details.
 */

#ifndef HIGHNOON_NATIVE_OPS_QUANTUM_FOUNDATION_OPS_H_
#define HIGHNOON_NATIVE_OPS_QUANTUM_FOUNDATION_OPS_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <complex>
#include <random>

// Include shared SIMD utilities from Phase 1
#include "hnn_simd_common.h"

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define QF_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define QF_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define QF_NEON 1
#endif

namespace hsmn {
namespace quantum {

// =============================================================================
// QUANTUM OPERATION TYPE ENUMERATION
// =============================================================================

/**
 * @brief Quantum operation types supported by the unified quantum system.
 */
enum class QuantumOpType : int {
    // Core Quantum Operations
    EMBEDDING = 0,           ///< Holographic token embeddings
    POSITION_ENCODING = 1,   ///< Floquet/SU(2) position encoding
    LM_HEAD = 2,             ///< VQC-based language model head
    EXPERT = 3,              ///< Unitary expert networks
    NORM = 4,                ///< Unitary/Stiefel normalization
    RESIDUAL = 5,            ///< Unitary residual connections
    
    // Quantum Bus Operations
    COHERENCE_BUS = 6,       ///< Phase-coherent state transport
    TELEPORT_BUS = 7,        ///< Quantum teleportation state transfer
    
    // VQC Operations
    VQC = 8,                 ///< Variational Quantum Circuit
    TENSOR_RING_VQC = 9,     ///< Tensor ring VQC simulation
    
    // Quantum Training Operations
    MEASUREMENT = 10,        ///< Born rule measurement
    CRYSTALLIZATION = 11,    ///< Quantum state crystallization
    FIDELITY_LOSS = 12,      ///< Quantum fidelity regularization
    DROPOUT = 13,            ///< Quantum measurement dropout
    CURRICULUM = 14,         ///< Spectral-aware quantum curriculum
    
    NUM_TYPES = 15
};

/**
 * @brief Convert QuantumOpType to string.
 */
inline const char* QuantumOpTypeToString(QuantumOpType type) {
    switch (type) {
        case QuantumOpType::EMBEDDING:         return "EMBEDDING";
        case QuantumOpType::POSITION_ENCODING: return "POSITION_ENCODING";
        case QuantumOpType::LM_HEAD:           return "LM_HEAD";
        case QuantumOpType::EXPERT:            return "EXPERT";
        case QuantumOpType::NORM:              return "NORM";
        case QuantumOpType::RESIDUAL:          return "RESIDUAL";
        case QuantumOpType::COHERENCE_BUS:     return "COHERENCE_BUS";
        case QuantumOpType::TELEPORT_BUS:      return "TELEPORT_BUS";
        case QuantumOpType::VQC:               return "VQC";
        case QuantumOpType::TENSOR_RING_VQC:   return "TENSOR_RING_VQC";
        case QuantumOpType::MEASUREMENT:       return "MEASUREMENT";
        case QuantumOpType::CRYSTALLIZATION:   return "CRYSTALLIZATION";
        case QuantumOpType::FIDELITY_LOSS:     return "FIDELITY_LOSS";
        case QuantumOpType::DROPOUT:           return "DROPOUT";
        case QuantumOpType::CURRICULUM:        return "CURRICULUM";
        default:                               return "UNKNOWN";
    }
}

// =============================================================================
// UNIFIED QUANTUM CONFIGURATION
// =============================================================================

/**
 * @brief Unified configuration for all quantum operations.
 */
struct QuantumConfig {
    // === Core Parameters ===
    QuantumOpType op_type = QuantumOpType::VQC;
    int batch_size = 1;
    int seq_len = 512;
    int d_model = 512;
    int vocab_size = 32000;
    float epsilon = 1e-6f;
    
    // === VQC Parameters ===
    int num_qubits = 4;
    int vqc_layers = 2;
    float entanglement_strength = 0.5f;
    int neumann_terms = 6;           ///< For Cayley transform
    
    // === Embedding Parameters ===
    int embedding_dim = 512;
    int num_bundles = 4;             ///< Holographic bundles
    
    // === Position Encoding Parameters ===
    float floquet_omega = 1.0f;      ///< Floquet driving frequency
    float floquet_amplitude = 0.1f;
    int su2_components = 3;          ///< SU(2) basis components
    
    // === Expert Parameters ===
    int d_ff = 2048;
    float activation_angle = 0.5f;
    
    // === Norm Parameters ===
    bool use_bias = true;
    
    // === Bus Parameters ===
    int num_channels = 8;
    float coherence_threshold = 0.9f;
    
    // === Dropout Parameters ===
    float dropout_rate = 0.1f;
    float collapse_probability = 0.5f;
    
    // === Curriculum Parameters ===
    float spectral_complexity_threshold = 0.5f;
    bool use_fft_analysis = true;
    
    // === Tensor Ring Parameters ===
    int tr_rank = 8;
    int tr_cores = 4;
    float bp_mitigation_strength = 0.1f;
    
    // === Crystallization Parameters ===
    float crystallization_rate = 0.1f;
    int memory_slots = 64;
    
    /**
     * @brief Validate configuration.
     */
    bool Validate() const {
        if (static_cast<int>(op_type) < 0 || 
            static_cast<int>(op_type) >= static_cast<int>(QuantumOpType::NUM_TYPES)) {
            return false;
        }
        if (batch_size < 1 || d_model < 1) {
            return false;
        }
        if (num_qubits < 1 || vqc_layers < 1) {
            return false;
        }
        return true;
    }
};

// =============================================================================
// VQC PRIMITIVE OPERATIONS
// =============================================================================

namespace vqc {

/**
 * @brief Apply RY rotation to qubit amplitude pair.
 *
 * RY(θ) = [[cos(θ/2), -sin(θ/2)],
 *          [sin(θ/2),  cos(θ/2)]]
 *
 * @param amp Two-element amplitude vector [a, b]
 * @param theta Rotation angle
 */
inline void ry_rotation(float* amp, float theta) {
    float half = theta * 0.5f;
    float c = std::cos(half);
    float s = std::sin(half);
    float a0 = amp[0], a1 = amp[1];
    amp[0] = c * a0 - s * a1;
    amp[1] = s * a0 + c * a1;
}

/**
 * @brief Apply RZ rotation to qubit amplitude pair.
 *
 * For real simulation: RZ just scales by cos(θ/2).
 *
 * @param amp Two-element amplitude vector
 * @param theta Rotation angle
 */
inline void rz_rotation(float* amp, float theta) {
    float half = theta * 0.5f;
    float c = std::cos(half);
    amp[0] *= c;
    amp[1] *= c;
}

/**
 * @brief Apply CNOT gate (simulated for classical).
 *
 * Conditionally flips target based on control amplitude.
 *
 * @param control Control qubit [2]
 * @param target Target qubit [2]
 */
inline void cnot(const float* control, float* target) {
    // If control is mostly |1>, flip target
    float control_prob_1 = control[1] * control[1];
    if (control_prob_1 > 0.5f) {
        std::swap(target[0], target[1]);
    } else {
        // Partial flip: entanglement
        float flip_amp = std::sqrt(control_prob_1);
        float t0 = target[0], t1 = target[1];
        target[0] = (1.0f - flip_amp) * t0 + flip_amp * t1;
        target[1] = (1.0f - flip_amp) * t1 + flip_amp * t0;
    }
}

/**
 * @brief Apply Hadamard gate to qubit.
 *
 * H = 1/√2 [[1, 1], [1, -1]]
 *
 * @param amp Two-element amplitude vector
 */
inline void hadamard(float* amp) {
    constexpr float INV_SQRT2 = 0.7071067811865476f;
    float a0 = amp[0], a1 = amp[1];
    amp[0] = INV_SQRT2 * (a0 + a1);
    amp[1] = INV_SQRT2 * (a0 - a1);
}

/**
 * @brief Initialize qubit in |0⟩ state.
 */
inline void init_zero(float* amp) {
    amp[0] = 1.0f;
    amp[1] = 0.0f;
}

/**
 * @brief Initialize qubit in |+⟩ = H|0⟩ state.
 */
inline void init_plus(float* amp) {
    init_zero(amp);
    hadamard(amp);
}

/**
 * @brief Apply full VQC layer: RY → RZ → CNOT entanglement.
 *
 * @param state Quantum state [2 * num_qubits]
 * @param params Layer parameters [2 * num_qubits] (RY, RZ angles)
 * @param num_qubits Number of qubits
 */
inline void apply_vqc_layer(float* state, const float* params, int num_qubits) {
    // Single-qubit rotations
    for (int q = 0; q < num_qubits; ++q) {
        float* qubit = state + 2 * q;
        ry_rotation(qubit, params[q]);
        rz_rotation(qubit, params[num_qubits + q]);
    }
    
    // Entangling gates (ring topology)
    for (int q = 0; q < num_qubits - 1; ++q) {
        const float* control = state + 2 * q;
        float* target = state + 2 * (q + 1);
        cnot(control, target);
    }
}

}  // namespace vqc

// =============================================================================
// BORN RULE MEASUREMENT
// =============================================================================

namespace measurement {

/**
 * @brief Extract probabilities from quantum state via Born rule.
 *
 * P(outcome) = |⟨outcome|ψ⟩|²
 *
 * @param state Quantum state [2 * num_qubits]
 * @param probabilities Output probabilities [2^num_qubits or vocab_size]
 * @param num_qubits Number of qubits
 * @param output_dim Output dimension
 * @param weights Optional output weights for mapping [output_dim, num_amplitude_pairs]
 */
inline void born_rule(
    const float* state,
    float* probabilities,
    int num_qubits,
    int output_dim,
    const float* weights = nullptr) {
    
    int num_amps = 2 * num_qubits;
    
    if (weights != nullptr) {
        // Map amplitudes to output via weights
        #pragma omp parallel for
        for (int o = 0; o < output_dim; ++o) {
            float prob = 0.0f;
            for (int a = 0; a < num_amps; ++a) {
                float amp = state[a] * weights[o * num_amps + a];
                prob += amp * amp;
            }
            probabilities[o] = prob;
        }
    } else {
        // Direct amplitude squared
        for (int a = 0; a < num_amps; ++a) {
            probabilities[a] = state[a] * state[a];
        }
    }
}

/**
 * @brief Apply measurement collapse (project to basis state).
 *
 * @param state Quantum state to collapse [2 * num_qubits]
 * @param outcome Measurement outcome
 * @param qubit_idx Which qubit was measured
 * @param num_qubits Total qubits
 */
inline void collapse(float* state, int outcome, int qubit_idx, int num_qubits) {
    float* qubit = state + 2 * qubit_idx;
    
    // Project to |outcome⟩
    float norm = qubit[outcome];
    if (std::abs(norm) > 1e-10f) {
        float inv_norm = 1.0f / std::abs(norm);
        qubit[outcome] = (norm >= 0) ? 1.0f : -1.0f;
        qubit[1 - outcome] = 0.0f;
    } else {
        // Zero amplitude - this shouldn't happen in valid simulation
        qubit[outcome] = 1.0f;
        qubit[1 - outcome] = 0.0f;
    }
}

/**
 * @brief Apply measurement dropout (randomly collapse qubits).
 *
 * @param state Quantum state [2 * num_qubits]
 * @param num_qubits Number of qubits
 * @param dropout_rate Probability of measuring each qubit
 * @param seed Random seed
 */
inline void measurement_dropout(float* state, int num_qubits, float dropout_rate, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int q = 0; q < num_qubits; ++q) {
        if (dist(rng) < dropout_rate) {
            float* qubit = state + 2 * q;
            float p0 = qubit[0] * qubit[0];
            float p1 = qubit[1] * qubit[1];
            
            // Probabilistic collapse
            int outcome = (dist(rng) < p0 / (p0 + p1 + 1e-10f)) ? 0 : 1;
            collapse(state, outcome, q, num_qubits);
        }
    }
}

}  // namespace measurement

// =============================================================================
// DENSITY MATRIX OPERATIONS
// =============================================================================

namespace density {

/**
 * @brief Compute partial trace of density matrix.
 *
 * Given ρ_AB, compute ρ_A = Tr_B(ρ_AB)
 *
 * @param rho Input density matrix [dim_total, dim_total]
 * @param rho_reduced Output reduced density matrix [dim_keep, dim_keep]
 * @param dim_keep Dimension of subsystem to keep
 * @param dim_trace Dimension of subsystem to trace out
 */
inline void partial_trace(
    const float* rho,
    float* rho_reduced,
    int dim_keep,
    int dim_trace) {
    
    int dim_total = dim_keep * dim_trace;
    
    // Initialize to zero
    std::fill(rho_reduced, rho_reduced + dim_keep * dim_keep, 0.0f);
    
    // Sum over traced-out subsystem
    for (int i = 0; i < dim_keep; ++i) {
        for (int j = 0; j < dim_keep; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dim_trace; ++k) {
                int row = i * dim_trace + k;
                int col = j * dim_trace + k;
                sum += rho[row * dim_total + col];
            }
            rho_reduced[i * dim_keep + j] = sum;
        }
    }
}

/**
 * @brief Compute von Neumann entropy: S = -Tr(ρ log ρ)
 *
 * @param eigenvalues Eigenvalues of density matrix [dim]
 * @param dim Dimension
 * @return Entropy value
 */
inline float von_neumann_entropy(const float* eigenvalues, int dim) {
    float entropy = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float p = eigenvalues[i];
        if (p > 1e-10f) {
            entropy -= p * std::log(p);
        }
    }
    return entropy;
}

/**
 * @brief Compute quantum fidelity: F(ρ, σ) = Tr(√(√ρ σ √ρ))²
 *
 * Simplified for pure states: F = |⟨ψ|φ⟩|²
 *
 * @param state1 First state amplitudes [dim]
 * @param state2 Second state amplitudes [dim]
 * @param dim State dimension
 * @return Fidelity value [0, 1]
 */
inline float fidelity(const float* state1, const float* state2, int dim) {
    float overlap = 0.0f;
    for (int i = 0; i < dim; ++i) {
        overlap += state1[i] * state2[i];
    }
    return overlap * overlap;
}

}  // namespace density

// =============================================================================
// FFT FOR HOLOGRAPHIC OPERATIONS
// =============================================================================

namespace fft {

/**
 * @brief In-place radix-2 FFT.
 *
 * @param real Real part [n]
 * @param imag Imaginary part [n]
 * @param n Size (must be power of 2)
 * @param inverse If true, compute inverse FFT
 */
inline void radix2(float* real, float* imag, int n, bool inverse = false) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    
    // Cooley-Tukey FFT
    for (int len = 2; len <= n; len <<= 1) {
        float angle = 2.0f * static_cast<float>(M_PI) / len;
        if (inverse) angle = -angle;
        float wlen_r = std::cos(angle);
        float wlen_i = std::sin(angle);
        
        for (int i = 0; i < n; i += len) {
            float w_r = 1.0f, w_i = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                int u_idx = i + j;
                int v_idx = i + j + len / 2;
                
                float u_r = real[u_idx];
                float u_i = imag[u_idx];
                float v_r = real[v_idx] * w_r - imag[v_idx] * w_i;
                float v_i = real[v_idx] * w_i + imag[v_idx] * w_r;
                
                real[u_idx] = u_r + v_r;
                imag[u_idx] = u_i + v_i;
                real[v_idx] = u_r - v_r;
                imag[v_idx] = u_i - v_i;
                
                float new_w_r = w_r * wlen_r - w_i * wlen_i;
                float new_w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = new_w_r;
                w_i = new_w_i;
            }
        }
    }
    
    // Scale for inverse
    if (inverse) {
        float inv_n = 1.0f / n;
        for (int i = 0; i < n; ++i) {
            real[i] *= inv_n;
            imag[i] *= inv_n;
        }
    }
}

/**
 * @brief Holographic bind: circular convolution via FFT.
 *
 * bind(a, b) = IFFT(FFT(a) ⊙ FFT(b))
 *
 * @param a First vector [dim]
 * @param b Second vector [dim]
 * @param result Output [dim]
 * @param dim Dimension (power of 2)
 */
inline void holographic_bind(const float* a, const float* b, float* result, int dim) {
    std::vector<float> a_real(dim), a_imag(dim, 0.0f);
    std::vector<float> b_real(dim), b_imag(dim, 0.0f);
    
    std::copy(a, a + dim, a_real.data());
    std::copy(b, b + dim, b_real.data());
    
    // Forward FFT
    radix2(a_real.data(), a_imag.data(), dim, false);
    radix2(b_real.data(), b_imag.data(), dim, false);
    
    // Element-wise complex multiplication
    std::vector<float> c_real(dim), c_imag(dim);
    for (int i = 0; i < dim; ++i) {
        c_real[i] = a_real[i] * b_real[i] - a_imag[i] * b_imag[i];
        c_imag[i] = a_real[i] * b_imag[i] + a_imag[i] * b_real[i];
    }
    
    // Inverse FFT
    radix2(c_real.data(), c_imag.data(), dim, true);
    
    std::copy(c_real.begin(), c_real.end(), result);
}

/**
 * @brief Holographic unbind (inverse of bind).
 *
 * @param bound Bound representation [dim]
 * @param key Key to unbind [dim]
 * @param result Unbound value [dim]
 * @param dim Dimension
 */
inline void holographic_unbind(const float* bound, const float* key, float* result, int dim) {
    std::vector<float> b_real(dim), b_imag(dim, 0.0f);
    std::vector<float> k_real(dim), k_imag(dim, 0.0f);
    
    std::copy(bound, bound + dim, b_real.data());
    std::copy(key, key + dim, k_real.data());
    
    // Forward FFT
    radix2(b_real.data(), b_imag.data(), dim, false);
    radix2(k_real.data(), k_imag.data(), dim, false);
    
    // Element-wise complex multiplication with conjugate of key FFT
    std::vector<float> c_real(dim), c_imag(dim);
    for (int i = 0; i < dim; ++i) {
        // conj(k) = (k_real, -k_imag)
        c_real[i] = b_real[i] * k_real[i] + b_imag[i] * k_imag[i];
        c_imag[i] = -b_real[i] * k_imag[i] + b_imag[i] * k_real[i];
    }
    
    // Inverse FFT
    radix2(c_real.data(), c_imag.data(), dim, true);
    
    std::copy(c_real.begin(), c_real.end(), result);
}

}  // namespace fft

// =============================================================================
// CAYLEY TRANSFORM (SKEW-SYMMETRIC → UNITARY)
// =============================================================================

namespace cayley {

/**
 * @brief Compute (I + A)^{-1} using Neumann series.
 *
 * For small ||A||: (I + A)^{-1} ≈ I - A + A² - A³ + ...
 *
 * @param A Skew-symmetric matrix [dim, dim]
 * @param result Output inverse [dim, dim]
 * @param dim Matrix dimension
 * @param num_terms Neumann series terms
 */
inline void neumann_inverse(const float* A, float* result, int dim, int num_terms = 6) {
    std::vector<float> A_power(dim * dim);
    std::vector<float> temp(dim * dim);
    
    // Initialize result = I
    std::fill(result, result + dim * dim, 0.0f);
    for (int i = 0; i < dim; ++i) {
        result[i * dim + i] = 1.0f;
    }
    
    // Initialize A_power = I
    std::fill(A_power.begin(), A_power.end(), 0.0f);
    for (int i = 0; i < dim; ++i) {
        A_power[i * dim + i] = 1.0f;
    }
    
    // Accumulate: I - A + A² - A³ + ...
    for (int k = 1; k < num_terms; ++k) {
        // A_power = A_power @ A
        std::fill(temp.begin(), temp.end(), 0.0f);
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                for (int l = 0; l < dim; ++l) {
                    temp[i * dim + j] += A_power[i * dim + l] * A[l * dim + j];
                }
            }
        }
        std::copy(temp.begin(), temp.end(), A_power.begin());
        
        // result += (-1)^k * A_power
        float sign = (k % 2 == 0) ? 1.0f : -1.0f;
        for (int i = 0; i < dim * dim; ++i) {
            result[i] += sign * A_power[i];
        }
    }
}

/**
 * @brief Cayley transform: U = (I - A)(I + A)^{-1}
 *
 * Maps skew-symmetric A to orthogonal U.
 *
 * @param A_skew Skew-symmetric matrix [dim, dim]
 * @param U Output unitary matrix [dim, dim]
 * @param dim Dimension
 * @param num_terms Neumann series terms
 */
inline void transform(const float* A_skew, float* U, int dim, int num_terms = 6) {
    std::vector<float> I_plus_A_inv(dim * dim);
    std::vector<float> I_minus_A(dim * dim);
    
    // Compute (I + A)^{-1}
    neumann_inverse(A_skew, I_plus_A_inv.data(), dim, num_terms);
    
    // Compute I - A
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float delta = (i == j) ? 1.0f : 0.0f;
            I_minus_A[i * dim + j] = delta - A_skew[i * dim + j];
        }
    }
    
    // U = (I - A) @ (I + A)^{-1}
    std::fill(U, U + dim * dim, 0.0f);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                U[i * dim + j] += I_minus_A[i * dim + k] * I_plus_A_inv[k * dim + j];
            }
        }
    }
}

}  // namespace cayley

// =============================================================================
// FORWARD DECLARATIONS FOR QUANTUM OPERATIONS
// =============================================================================

// Each operation has a forward kernel - implementations in quantum_foundation_ops.cc
void QuantumEmbeddingForward(
    const int32_t* token_ids, const float* holographic_store, const float* token_keys,
    float* output, const QuantumConfig& config);

void QuantumPositionEncodingForward(
    const float* input, float* output, int position_offset, const QuantumConfig& config);

void QuantumLMHeadForward(
    const float* hidden, const float* vqc_params, const float* output_weights,
    float* logits, const QuantumConfig& config);

void QuantumExpertForward(
    const float* input, const float* U_skew, float* output, const QuantumConfig& config);

void QuantumExpertBackward(
    const float* grad_output, const float* input, const float* U_skew,
    float* grad_input, float* grad_U_skew, const QuantumConfig& config);

void QuantumNormForward(
    const float* input, const float* scale, const float* bias,
    float* output, const QuantumConfig& config);

void QuantumResidualForward(
    const float* x, const float* fx, const float* alpha, float* output, const QuantumConfig& config);

void QuantumCoherenceBusForward(
    const float* state, const float* phase_keys, float* transported, const QuantumConfig& config);

void QuantumTeleportForward(
    const float* source, const float* bell_state, float* destination, const QuantumConfig& config);

void VQCForward(
    const float* input, const float* params, float* output, const QuantumConfig& config);

void TensorRingVQCForward(
    const float* input, const float* core_params, float* output, const QuantumConfig& config);

void QuantumMeasurementForward(
    const float* state, float* probabilities, float* collapsed, const QuantumConfig& config);

void QuantumCrystallizationForward(
    const float* state, float* crystallized, float* memory, const QuantumConfig& config);

void QuantumFidelityLoss(
    const float* state, const float* target, float* loss, const QuantumConfig& config);

void QuantumDropoutForward(
    const float* state, float* output, bool training, uint64_t seed, const QuantumConfig& config);

void QuantumCurriculumScore(
    const float* input, float* score, const QuantumConfig& config);

// =============================================================================
// UNIFIED QUANTUM DISPATCHER
// =============================================================================

/**
 * @brief Unified quantum operation dispatcher.
 *
 * Routes to appropriate kernel based on config.op_type.
 *
 * @param input Input tensor (interpretation depends on op_type)
 * @param params Operation-specific parameters
 * @param output Output tensor
 * @param config Quantum configuration
 * @param aux_input Auxiliary inputs (weights, keys, etc.)
 * @param training Training mode flag
 */
void UnifiedQuantumForward(
    const float* input,
    const float* params,
    float* output,
    const QuantumConfig& config,
    const float* aux_input = nullptr,
    bool training = false);

/**
 * @brief Unified quantum operation backward pass dispatcher.
 */
void UnifiedQuantumBackward(
    const float* grad_output,
    const float* input,
    const float* params,
    float* grad_input,
    float* grad_params,
    const QuantumConfig& config,
    const float* aux_input = nullptr);

}  // namespace quantum
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_QUANTUM_FOUNDATION_OPS_H_
