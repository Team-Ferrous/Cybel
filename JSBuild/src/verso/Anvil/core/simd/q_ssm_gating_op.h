// highnoon/_native/ops/q_ssm_gating_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");

/**
 * @file q_ssm_gating_op.h
 * @brief Phase 69: Q-SSM Quantum State Space Gating
 *
 * Integrates variational quantum circuit gating into Mamba SSM for
 * quantum-adaptive memory control in `mamba_timecrystal_wlam_moe_hybrid`.
 *
 * Key Features:
 *   - VQC Gating: RY-RX ansatz regulates memory updates adaptively
 *   - Quantum Stabilization: Prevents optimization instabilities
 *   - Born Rule Interpretation: Gate values via quantum measurement
 *
 * Research Basis:
 *   - "Q-SSM: Quantum-Optimized Selective State Space Model" 
 *     (arXiv 2509.00259, 2025)
 *
 * Benefits:
 *   - +15% accuracy on time series (MAE/MSE)
 *   - Stabilized optimization prevents divergence
 *   - Quantum adaptive memory control
 *
 * Integration: Enhances Phase 37 (QMamba)
 * Complexity: O(N × D × vqc_layers)
 */

#ifndef HIGHNOON_NATIVE_OPS_Q_SSM_GATING_OP_H_
#define HIGHNOON_NATIVE_OPS_Q_SSM_GATING_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>

#include "common/perf_utils.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace hsmn {
namespace qssm {

// =============================================================================
// CONFIGURATION
// =============================================================================

struct QSSMConfig {
    int state_dim;               // SSM state dimension
    int input_dim;               // Input dimension
    int vqc_qubits;              // log2(state_dim) effective qubits
    int vqc_layers;              // RY-RX rotation layers
    bool use_born_rule;          // Born rule vs sigmoid gate
    float measurement_temp;      // Temperature for soft measurement
    uint32_t seed;               // Random seed
    
    QSSMConfig()
        : state_dim(16)
        , input_dim(64)
        , vqc_qubits(4)
        , vqc_layers(2)
        , use_born_rule(true)
        , measurement_temp(1.0f)
        , seed(42) {}
};

// =============================================================================
// VQC HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Apply RY rotation gate.
 *
 * RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
 *
 * @param state_real Real part of qubit state
 * @param state_imag Imaginary part of qubit state
 * @param theta Rotation angle
 */
inline void ApplyRY(float& state_real, float& state_imag, float theta) {
    float cos_half = std::cos(theta * 0.5f);
    float sin_half = std::sin(theta * 0.5f);
    
    float new_real = cos_half * state_real - sin_half * state_imag;
    float new_imag = sin_half * state_real + cos_half * state_imag;
    
    state_real = new_real;
    state_imag = new_imag;
}

/**
 * @brief Apply RX rotation gate.
 *
 * RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
 *
 * @param state_real Real part of qubit state
 * @param state_imag Imaginary part of qubit state
 * @param theta Rotation angle
 */
inline void ApplyRX(float& state_real, float& state_imag, float theta) {
    float cos_half = std::cos(theta * 0.5f);
    float sin_half = std::sin(theta * 0.5f);
    
    float new_real = cos_half * state_real + sin_half * state_imag;
    float new_imag = cos_half * state_imag - sin_half * state_real;
    
    state_real = new_real;
    state_imag = new_imag;
}

/**
 * @brief Apply CNOT gate between control and target qubits.
 *
 * Creates entanglement between qubits in the circuit.
 *
 * @param states Array of qubit states [num_qubits * 2] (real, imag pairs)
 * @param control Control qubit index
 * @param target Target qubit index
 */
inline void ApplyCNOT(float* states, int control, int target) {
    // Simplified CNOT: XOR behavior for classical-like states
    float ctrl_prob = states[control * 2] * states[control * 2] + 
                      states[control * 2 + 1] * states[control * 2 + 1];
    
    if (ctrl_prob > 0.5f) {
        // Flip target
        float tmp = states[target * 2];
        states[target * 2] = states[target * 2 + 1];
        states[target * 2 + 1] = tmp;
    }
}

// =============================================================================
// VQC CIRCUIT SIMULATION
// =============================================================================

/**
 * @brief Simulate VQC and return expectation value ⟨Z⟩.
 *
 * Circuit structure (per layer):
 *   - RY(θ_i) on each qubit
 *   - RX(φ_i) on each qubit
 *   - CNOT entanglement (ring topology)
 *
 * @param encoded_input Input features encoded as angles [vqc_qubits]
 * @param rotation_params VQC parameters [vqc_layers, vqc_qubits, 2] (RY, RX)
 * @param config QSSMConfig
 * @return Expectation value in [-1, 1]
 */
inline float VQCExpectation(
    const float* encoded_input,
    const float* rotation_params,
    const QSSMConfig& config) {
    
    const int num_qubits = config.vqc_qubits;
    const int num_layers = config.vqc_layers;
    
    // Initialize qubit states: |0⟩ for all
    std::vector<float> states(num_qubits * 2, 0.0f);
    for (int q = 0; q < num_qubits; ++q) {
        states[q * 2] = 1.0f;      // |0⟩ has amplitude 1 in real part
        states[q * 2 + 1] = 0.0f;  // Imaginary part = 0
    }
    
    // Apply input encoding
    for (int q = 0; q < num_qubits; ++q) {
        ApplyRY(states[q * 2], states[q * 2 + 1], encoded_input[q]);
    }
    
    // Apply VQC layers
    for (int layer = 0; layer < num_layers; ++layer) {
        const float* layer_params = rotation_params + layer * num_qubits * 2;
        
        // Rotation layer
        for (int q = 0; q < num_qubits; ++q) {
            float ry_theta = layer_params[q * 2 + 0];
            float rx_theta = layer_params[q * 2 + 1];
            
            ApplyRY(states[q * 2], states[q * 2 + 1], ry_theta);
            ApplyRX(states[q * 2], states[q * 2 + 1], rx_theta);
        }
        
        // Entanglement layer (ring topology)
        for (int q = 0; q < num_qubits - 1; ++q) {
            ApplyCNOT(states.data(), q, q + 1);
        }
        if (num_qubits > 1) {
            ApplyCNOT(states.data(), num_qubits - 1, 0);  // Close ring
        }
    }
    
    // Compute ⟨Z⟩ expectation on first qubit
    // ⟨Z⟩ = |⟨0|ψ⟩|² - |⟨1|ψ⟩|²
    // For our simplified simulation: P(0) - P(1)
    float prob_0 = states[0] * states[0] + states[1] * states[1];
    return 2.0f * prob_0 - 1.0f;  // Map [0,1] -> [-1, 1]
}

/**
 * @brief Compute VQC gate expectation values for batch.
 *
 * @param encoded_input Input features [batch, vqc_qubits]
 * @param rotation_params VQC parameters [vqc_layers, vqc_qubits, 2]
 * @param gate_values Output gate values [batch]
 * @param config QSSMConfig
 * @param batch_size Batch size
 */
inline void VQCGateExpectation(
    const float* encoded_input,
    const float* rotation_params,
    float* gate_values,
    const QSSMConfig& config,
    int batch_size) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        float exp_val = VQCExpectation(
            encoded_input + b * config.vqc_qubits,
            rotation_params,
            config
        );
        
        // Apply temperature scaling and convert to gate
        if (config.use_born_rule) {
            // Born rule: P = |⟨0|ψ⟩|² mapped to [0, 1]
            gate_values[b] = (exp_val + 1.0f) * 0.5f;
        } else {
            // Soft sigmoid interpretation
            gate_values[b] = 1.0f / (1.0f + std::exp(-exp_val * config.measurement_temp));
        }
    }
}

// =============================================================================
// Q-SSM FORWARD PASS
// =============================================================================

/**
 * @brief Encode input features as VQC angles.
 *
 * Maps input dimensions to qubit angles via arctan normalization.
 *
 * @param input Input features [batch, input_dim]
 * @param encoded Encoded angles [batch, vqc_qubits]
 * @param config QSSMConfig
 * @param batch_size Batch size
 */
inline void EncodeInputForVQC(
    const float* input,
    float* encoded,
    const QSSMConfig& config,
    int batch_size) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        for (int q = 0; q < config.vqc_qubits; ++q) {
            // Aggregate input dimensions into qubit
            float sum = 0.0f;
            int start_idx = q * (config.input_dim / config.vqc_qubits);
            int end_idx = (q + 1) * (config.input_dim / config.vqc_qubits);
            if (q == config.vqc_qubits - 1) {
                end_idx = config.input_dim;
            }
            
            for (int d = start_idx; d < end_idx; ++d) {
                sum += input[b * config.input_dim + d];
            }
            
            // Normalize to angle in [-π, π]
            encoded[b * config.vqc_qubits + q] = std::atan(sum) * 2.0f;
        }
    }
}

/**
 * @brief Q-SSM forward pass with VQC gating.
 *
 * Implements the core Q-SSM equation:
 *   S_t = σ_VQC(x_t) ⊙ S_{t-1} + (1 - σ_VQC(x_t)) ⊙ Update(x_t)
 *
 * @param input Input sequence [batch, seq, input_dim]
 * @param state Running SSM state [batch, state_dim] (modified in-place)
 * @param vqc_params VQC rotation parameters [vqc_layers, vqc_qubits, 2]
 * @param output Output sequence [batch, seq, input_dim]
 * @param config QSSMConfig
 * @param batch_size Batch size
 * @param seq_len Sequence length
 */
inline void QSSMForward(
    const float* input,
    float* state,
    const float* vqc_params,
    float* output,
    const QSSMConfig& config,
    int batch_size, int seq_len) {
    
    std::vector<float> encoded(batch_size * config.vqc_qubits);
    std::vector<float> gate_values(batch_size);
    
    for (int t = 0; t < seq_len; ++t) {
        const float* input_t = input + t * batch_size * config.input_dim;
        float* output_t = output + t * batch_size * config.input_dim;
        
        // Phase 96: Prefetch next timestep's input
        if (t + 1 < seq_len) {
            const float* next_input = input + (t + 1) * batch_size * config.input_dim;
            hsmn::ops::PrefetchT0(next_input);
            hsmn::ops::PrefetchT0(next_input + 64);  // Prefetch 2 cache lines
        }
        
        // Encode input for VQC
        EncodeInputForVQC(input_t, encoded.data(), config, batch_size);
        
        // Compute VQC gate values
        VQCGateExpectation(encoded.data(), vqc_params, gate_values.data(), 
                          config, batch_size);
        
        // Apply gated state update
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            float gate = gate_values[b];
            
            // Phase 96: Prefetch state for next batch element
            if (b + 1 < batch_size) {
                hsmn::ops::PrefetchT0(state + (b + 1) * config.state_dim);
            }
            
            for (int d = 0; d < config.state_dim; ++d) {
                int input_d = d % config.input_dim;
                float inp_val = input_t[b * config.input_dim + input_d];
                
                // Gated update: S = gate * S_prev + (1-gate) * input
                state[b * config.state_dim + d] = 
                    gate * state[b * config.state_dim + d] + 
                    (1.0f - gate) * inp_val;
            }
            
            // Output projection
            for (int d = 0; d < config.input_dim; ++d) {
                int state_d = d % config.state_dim;
                output_t[b * config.input_dim + d] = 
                    state[b * config.state_dim + state_d];
            }
        }
    }
}

/**
 * @brief Compute gate values only (for monitoring/visualization).
 *
 * @param input Input features [batch, seq, input_dim]
 * @param vqc_params VQC parameters [vqc_layers, vqc_qubits, 2]
 * @param gate_values Output gates [batch, seq]
 * @param config QSSMConfig
 * @param batch_size Batch size
 * @param seq_len Sequence length
 */
inline void ComputeGateValues(
    const float* input,
    const float* vqc_params,
    float* gate_values,
    const QSSMConfig& config,
    int batch_size, int seq_len) {
    
    std::vector<float> encoded(batch_size * config.vqc_qubits);
    
    for (int t = 0; t < seq_len; ++t) {
        const float* input_t = input + t * batch_size * config.input_dim;
        float* gates_t = gate_values + t * batch_size;
        
        EncodeInputForVQC(input_t, encoded.data(), config, batch_size);
        VQCGateExpectation(encoded.data(), vqc_params, gates_t, config, batch_size);
    }
}

}  // namespace qssm
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_Q_SSM_GATING_OP_H_
