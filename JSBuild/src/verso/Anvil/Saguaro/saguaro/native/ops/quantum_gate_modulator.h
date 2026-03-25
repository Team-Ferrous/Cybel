// saguaro.native/ops/quantum_gate_modulator.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Quantum-enhanced gate modulation for xLSTM.
// Enhancement 2: Uses VQC-inspired rotation before exponential gating.
//
// Reference: xLSTM roadmap - Quantum-Enhanced Exponential Gating

#ifndef SAGUARO_NATIVE_OPS_QUANTUM_GATE_MODULATOR_H_
#define SAGUARO_NATIVE_OPS_QUANTUM_GATE_MODULATOR_H_

#include <cmath>
#include <vector>

namespace saguaro {
namespace ops {

/**
 * @brief Quantum-inspired gate modulation using Cayley transform.
 * 
 * Applies a learned rotation before exponential gating:
 *   gate_out = exp(cayley_transform(gate_in, theta))
 * 
 * The Cayley transform provides a smooth unitary approximation:
 *   C(A) = (I - A)(I + A)^-1 for skew-symmetric A
 * 
 * For scalar gates, this simplifies to:
 *   C(x, theta) = (1 - tan(theta)*x) / (1 + tan(theta)*x)
 * 
 * This adds learned non-linearity before exp() while maintaining
 * O(L) complexity.
 */
template <typename T>
class QuantumGateModulator {
public:
    /**
     * @brief Construct gate modulator with learnable rotation angle.
     * 
     * @param theta Initial rotation angle (typically 0, learned during training)
     */
    explicit QuantumGateModulator(T theta = 0) : theta_(theta) {}
    
    /**
     * @brief Apply Cayley-modulated exponential gate.
     * 
     * Computes: exp(cayley(x, theta))
     * 
     * @param x Input gate pre-activation
     * @return Modulated gate value
     */
    inline T modulate(T x) const {
        // Skip modulation if theta is zero (default)
        if (std::abs(theta_) < static_cast<T>(1e-8)) {
            return std::exp(x);
        }
        
        // Cayley transform: (1 - tan(theta)*x) / (1 + tan(theta)*x)
        T tan_theta = std::tan(theta_);
        T denominator = static_cast<T>(1) + tan_theta * x;
        
        // Numerical stability
        if (std::abs(denominator) < static_cast<T>(1e-8)) {
            denominator = static_cast<T>(1e-8) * (denominator >= 0 ? 1 : -1);
        }
        
        T cayley_x = (static_cast<T>(1) - tan_theta * x) / denominator;
        return std::exp(cayley_x);
    }
    
    /**
     * @brief Batch modulate gates.
     * 
     * @param gates Input gate pre-activations [size]
     * @param output Output modulated gates [size]
     * @param size Number of gates
     */
    void modulate_batch(const T* gates, T* output, int size) const {
        if (std::abs(theta_) < static_cast<T>(1e-8)) {
            // Fast path: no modulation
            for (int i = 0; i < size; ++i) {
                output[i] = std::exp(gates[i]);
            }
        } else {
            // Modulated path
            for (int i = 0; i < size; ++i) {
                output[i] = modulate(gates[i]);
            }
        }
    }
    
    /**
     * @brief Update rotation angle (for training).
     */
    void set_theta(T theta) { theta_ = theta; }
    T get_theta() const { return theta_; }
    
    /**
     * @brief Compute gradient w.r.t. theta.
     * 
     * d(modulate)/d(theta) for backpropagation.
     * 
     * @param x Input gate pre-activation
     * @param grad_out Gradient from output
     * @return Gradient w.r.t. theta
     */
    T grad_theta(T x, T grad_out) const {
        if (std::abs(theta_) < static_cast<T>(1e-8)) {
            return static_cast<T>(0);
        }
        
        T tan_theta = std::tan(theta_);
        T sec2_theta = static_cast<T>(1) + tan_theta * tan_theta;
        T denom = static_cast<T>(1) + tan_theta * x;
        T denom2 = denom * denom;
        
        // d(cayley)/d(theta) = -2*x*sec^2(theta) / (1 + tan(theta)*x)^2
        T dcayley_dtheta = -static_cast<T>(2) * x * sec2_theta / denom2;
        
        // Chain rule: d(exp(cayley))/d(theta) = exp(cayley) * d(cayley)/d(theta)
        T cayley_x = (static_cast<T>(1) - tan_theta * x) / denom;
        return grad_out * std::exp(cayley_x) * dcayley_dtheta;
    }

private:
    T theta_;  // Learnable rotation angle
};

/**
 * @brief Log-space parallel prefix sum for numerical stability.
 * 
 * Enhancement 5: O(log L) parallel sequence evaluation.
 * 
 * For gates in log-space (log_f = log(exp(f)) = f), we can use
 * associative parallel scan:
 *   log_cumsum[t] = log(sum_{s<=t} exp(log_f[s]))
 *                 = log_f[t] + log(1 + exp(log_cumsum[t-1] - log_f[t]))
 *                 = log_f[t] + softplus(log_cumsum[t-1] - log_f[t])
 */
template <typename T>
void log_space_parallel_scan(
    const T* log_gates,   // [seq_len] log-space gates (= pre-activation)
    T* log_cumsum,        // [seq_len] output cumulative sum
    int seq_len) {
    
    if (seq_len == 0) return;
    
    log_cumsum[0] = log_gates[0];
    
    // For now, sequential scan (full O(log n) parallel impl requires GPU)
    // On CPU, this is already efficient due to cache locality
    for (int t = 1; t < seq_len; ++t) {
        // log(a + b) = log(a) + log(1 + b/a) = log(a) + softplus(log(b) - log(a))
        // Here: log(f_t + cumsum_{t-1}) when f_t is dominant
        T diff = log_cumsum[t-1] - log_gates[t];
        T softplus_diff;
        if (diff > static_cast<T>(20)) {
            // Avoid overflow: softplus(x) ≈ x for large x
            softplus_diff = diff;
        } else if (diff < static_cast<T>(-20)) {
            // softplus(x) ≈ exp(x) for small x
            softplus_diff = std::exp(diff);
        } else {
            softplus_diff = std::log(static_cast<T>(1) + std::exp(diff));
        }
        log_cumsum[t] = log_gates[t] + softplus_diff;
    }
}

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_QUANTUM_GATE_MODULATOR_H_
