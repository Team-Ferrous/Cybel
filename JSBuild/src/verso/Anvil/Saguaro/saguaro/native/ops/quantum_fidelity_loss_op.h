// saguaro.native/ops/quantum_fidelity_loss_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

/**
 * @file quantum_fidelity_loss_op.h
 * @brief Phase 52: Quantum Fidelity Regularization
 *
 * Sequence-level loss using quantum fidelity between predicted and
 * target density matrices.
 *
 * F(ρ_pred, ρ_true) = Tr(sqrt(sqrt(ρ_pred) ρ_true sqrt(ρ_pred)))²
 * Simplified: F ≈ Tr(ρ_pred ρ_true) for pure states
 *
 * Benefits: Quantum-natural loss, captures coherence
 * Complexity: O(N × D²)
 */

#ifndef SAGUARO_NATIVE_OPS_QUANTUM_FIDELITY_LOSS_OP_H_
#define SAGUARO_NATIVE_OPS_QUANTUM_FIDELITY_LOSS_OP_H_

#include <cmath>
#include <algorithm>
#include <vector>

namespace saguaro {
namespace qfidelity {

// =============================================================================
// NUMERICAL STABILITY CONSTANTS (Enterprise-level guards)
// =============================================================================

// STABILITY FIX: Use 1e-6 instead of 1e-10 for float32 numerical stability.
// At 1e-10, division produces 1e10 → INF after accumulation.
constexpr float kQFidelityEpsilon = 1e-6f;

// Minimum norm product to prevent division explosion.
constexpr float kMinNormProduct = 1e-6f;

/**
 * @brief Compute quantum fidelity between density matrices.
 * Simplified: F = Tr(ρ_pred · ρ_true)
 */
inline float QuantumFidelity(const float* pred_density, const float* true_density, int dim) {
    // Trace of product for pure state approximation
    float trace = 0.0f;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            trace += pred_density[i * dim + j] * true_density[i * dim + j];
        }
    }
    return trace;
}

/**
 * @brief Quantum fidelity loss: L = 1 - F(ρ_pred, ρ_true)
 */
inline void QuantumFidelityLoss(
    const float* pred_states, const float* true_states,
    float* loss, float* grad,
    int batch, int dim) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        const float* pred = pred_states + b * dim;
        const float* true_s = true_states + b * dim;
        float* g = grad + b * dim;
        
        // Construct density matrices ρ = |ψ⟩⟨ψ|
        // But for efficiency, compute inner product directly
        float inner = 0.0f, norm_pred = 0.0f, norm_true = 0.0f;
        for (int d = 0; d < dim; ++d) {
            inner += pred[d] * true_s[d];
            norm_pred += pred[d] * pred[d];
            norm_true += true_s[d] * true_s[d];
        }
        
        // STABILITY FIX: Apply max() before sqrt to prevent underflow.
        // sqrt(1e-16) = 1e-8, then division = 1e8 → INF.
        // Instead, ensure norm product is at least kMinNormProduct.
        float norm_product = std::max(norm_pred * norm_true, kMinNormProduct);
        float sqrt_norm_product = std::sqrt(norm_product);

        // Fidelity for pure states: |⟨pred|true⟩|² / (||pred|| × ||true||)
        float fidelity = (inner * inner) / sqrt_norm_product;

        // Ensure fidelity is in valid range [0, 1]
        fidelity = std::min(fidelity, 1.0f);

        loss[b] = 1.0f - fidelity;

        // STABILITY FIX: Use max() for norm_pred to prevent division explosion.
        float safe_norm_pred = std::max(norm_pred, kQFidelityEpsilon);

        // Gradient: ∂L/∂pred = -2 * inner * true / ||pred||²
        float scale = -2.0f * inner / safe_norm_pred;
        for (int d = 0; d < dim; ++d) {
            float grad_val = scale * true_s[d];
            // Replace non-finite with zero
            g[d] = std::isfinite(grad_val) ? grad_val : 0.0f;
        }
    }
}

/**
 * @brief Von Neumann entropy of density matrix.
 * S(ρ) = -Tr(ρ log ρ)
 */
inline float VonNeumannEntropy(const float* eigenvalues, int dim) {
    float entropy = 0.0f;
    for (int i = 0; i < dim; ++i) {
        // STABILITY FIX: Use kQFidelityEpsilon for consistent epsilon handling.
        float p = std::max(eigenvalues[i], kQFidelityEpsilon);
        entropy -= p * std::log(p);
    }
    return entropy;
}

}}
#endif
