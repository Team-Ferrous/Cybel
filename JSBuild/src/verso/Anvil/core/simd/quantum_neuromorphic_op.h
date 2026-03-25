// highnoon/_native/ops/quantum_neuromorphic_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
/**
 * @file quantum_neuromorphic_op.h
 * @brief Phase 68: Quantum Neuromorphic Memory with spike-timing plasticity
 */
#ifndef HIGHNOON_NATIVE_OPS_QUANTUM_NEUROMORPHIC_OP_H_
#define HIGHNOON_NATIVE_OPS_QUANTUM_NEUROMORPHIC_OP_H_
#include <cmath>
#include <vector>

namespace hsmn {
namespace neuromorphic {

inline void SpikingQuantumNeuron(
    const float* input, const float* membrane_potential,
    float* spikes, float* new_potential,
    float threshold, float tau, int batch, int neurons) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < neurons; ++n) {
            int idx = b * neurons + n;
            float mp = membrane_potential[idx] * (1.0f - 1.0f/tau) + input[idx];
            if (mp > threshold) {
                spikes[idx] = 1.0f;
                new_potential[idx] = 0.0f;  // Reset
            } else {
                spikes[idx] = 0.0f;
                new_potential[idx] = mp;
            }
        }
    }
}

inline void STDP(
    const float* pre_spikes, const float* post_spikes,
    float* weights, float a_plus, float a_minus,
    float tau_plus, float tau_minus, int pre, int post) {
    
    for (int i = 0; i < pre; ++i) {
        for (int j = 0; j < post; ++j) {
            int idx = i * post + j;
            float delta_w = 0.0f;
            if (pre_spikes[i] > 0.5f && post_spikes[j] > 0.5f) {
                delta_w = a_plus * std::exp(-std::abs(pre_spikes[i] - post_spikes[j]) / tau_plus);
            } else if (pre_spikes[i] > 0.5f) {
                delta_w = -a_minus * std::exp(-1.0f / tau_minus);
            }
            weights[idx] = std::max(0.0f, std::min(1.0f, weights[idx] + delta_w));
        }
    }
}
}}
#endif
