// saguaro.native/ops/entanglement_loss.h
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
 * @file entanglement_loss.h
 * @brief Phase 7: Entanglement Preservation Loss
 *
 * Implements regularization loss to maintain MPS bond entropy above a threshold.
 * This preserves "context-carrying capacity" during training by preventing
 * the MPS from collapsing to low-rank representations.
 *
 * Loss = max(0, min_entropy - bond_entropy)
 *
 * Reference: "Entanglement entropy in quantum machine learning"
 */

#ifndef SAGUARO_NATIVE_OPS_ENTANGLEMENT_LOSS_H_
#define SAGUARO_NATIVE_OPS_ENTANGLEMENT_LOSS_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace saguaro {
namespace entanglement {

// =============================================================================
// NUMERICAL STABILITY CONSTANTS (Enterprise-level guards)
// =============================================================================

// STABILITY FIX: Use 1e-6 instead of 1e-10 for float32 numerical stability.
// At 1e-10, division produces 1e10 → INF after accumulation.
constexpr float kEntanglementEpsilon = 1e-6f;

/**
 * @brief Compute von Neumann entropy from singular values.
 * 
 * S = -sum_i (s_i^2 * log(s_i^2)) where sum(s_i^2) = 1
 * 
 * @param singular_values Normalized singular values [dim]
 * @param dim Number of singular values
 * @return Von Neumann entropy (0 = pure state, log(dim) = maximally mixed)
 */
inline float ComputeVonNeumannEntropy(const float* singular_values, int dim) {
    float entropy = 0.0f;

    for (int i = 0; i < dim; ++i) {
        float p = singular_values[i] * singular_values[i];  // Probability
        // STABILITY FIX: Use kEntanglementEpsilon for consistent epsilon handling.
        if (p > kEntanglementEpsilon) {
            entropy -= p * std::log(p);
        }
    }

    return entropy;
}

/**
 * @brief Compute bond entropy from MPS core using reduced density matrix.
 * 
 * For MPS core A[i] of shape [chi_L, d, chi_R], the left bond entropy
 * is computed from the Schmidt values of the bipartition at bond i.
 * 
 * @param core MPS core tensor [chi_L * d, chi_R] (reshaped)
 * @param chi_L Left bond dimension
 * @param d Physical dimension
 * @param chi_R Right bond dimension
 * @return Bond entropy
 */
inline float ComputeBondEntropy(const float* core, int chi_L, int d, int chi_R) {
    const int rows = chi_L * d;
    const int cols = chi_R;
    
    // Compute A^T A for right bond
    std::vector<float> AtA(cols * cols, 0.0f);
    
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < rows; ++k) {
                sum += core[k * cols + i] * core[k * cols + j];
            }
            AtA[i * cols + j] = sum;
        }
    }
    
    // Approximate eigenvalues via power iteration (simplified)
    // For full accuracy, use proper SVD - this is a fast approximation
    std::vector<float> eigenvalues(cols);
    float trace = 0.0f;
    for (int i = 0; i < cols; ++i) {
        eigenvalues[i] = AtA[i * cols + i];  // Diagonal approximation
        trace += eigenvalues[i];
    }
    
    // STABILITY FIX: Use kEntanglementEpsilon for consistent epsilon handling.
    // Normalize to get probabilities
    if (trace > kEntanglementEpsilon) {
        for (int i = 0; i < cols; ++i) {
            eigenvalues[i] /= trace;
        }
    }
    
    return ComputeVonNeumannEntropy(eigenvalues.data(), cols);
}

/**
 * @brief Compute entanglement preservation loss.
 * 
 * Loss = weight * max(0, min_entropy - bond_entropy)
 * 
 * This encourages the model to maintain at least min_entropy of
 * entanglement across MPS bonds.
 * 
 * @param bond_entropies Entropies for each bond [num_bonds]
 * @param num_bonds Number of MPS bonds
 * @param min_entropy Minimum target entropy
 * @param weight Loss weight (regularization strength)
 * @return Entanglement regularization loss
 */
inline float ComputeEntanglementLoss(
    const float* bond_entropies, int num_bonds,
    float min_entropy, float weight) {
    
    float total_loss = 0.0f;
    
    for (int i = 0; i < num_bonds; ++i) {
        float deficit = min_entropy - bond_entropies[i];
        if (deficit > 0.0f) {
            total_loss += deficit;
        }
    }
    
    return weight * total_loss / num_bonds;
}

/**
 * @brief Compute gradient of entanglement loss w.r.t. singular values.
 * 
 * If s is below threshold, gradient pushes toward uniform distribution.
 * 
 * @param singular_values Current singular values [dim]
 * @param gradient Output gradient [dim]
 * @param dim Number of singular values
 * @param current_entropy Current entropy
 * @param min_entropy Target minimum entropy
 * @param weight Loss weight
 */
inline void ComputeEntanglementGradient(
    const float* singular_values, float* gradient,
    int dim, float current_entropy, float min_entropy, float weight) {
    
    if (current_entropy >= min_entropy) {
        // No loss, no gradient
        std::fill(gradient, gradient + dim, 0.0f);
        return;
    }
    
    // Gradient of -sum(p*log(p)) w.r.t. s is -2s*(log(s^2) + 1) / sum(s^2)
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum_sq += singular_values[i] * singular_values[i];
    }

    // STABILITY FIX: Use max() to prevent division explosion.
    float safe_sum_sq = std::max(sum_sq, kEntanglementEpsilon);

    for (int i = 0; i < dim; ++i) {
        float s = singular_values[i];
        float p = s * s / safe_sum_sq;

        if (p > kEntanglementEpsilon) {
            // Push toward uniform distribution
            float uniform_p = 1.0f / static_cast<float>(dim);
            float grad_val = weight * (p - uniform_p) * 2.0f * s / safe_sum_sq;
            // Replace non-finite with zero
            gradient[i] = std::isfinite(grad_val) ? grad_val : 0.0f;
        } else {
            gradient[i] = 0.0f;
        }
    }
}

/**
 * @brief Add truncation noise to gradients for robustness.
 * 
 * Injects small Gaussian noise at MPS truncation points to prevent
 * over-fitting to specific truncation patterns.
 * 
 * @param gradient Gradient to perturb [size]
 * @param size Gradient size
 * @param noise_scale Standard deviation of noise
 * @param seed Random seed
 */
inline void AddTruncationNoise(float* gradient, int size, float noise_scale, uint32_t seed) {
    // Simple LCG random number generator
    uint32_t state = seed;
    const uint32_t a = 1664525;
    const uint32_t c = 1013904223;
    
    for (int i = 0; i < size; ++i) {
        state = a * state + c;
        // Box-Muller approximation using uniform
        float u = static_cast<float>(state) / 4294967296.0f;
        float noise = noise_scale * (u - 0.5f) * 2.0f;  // Uniform approximation
        gradient[i] += noise;
    }
}

}  // namespace entanglement
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_ENTANGLEMENT_LOSS_H_
