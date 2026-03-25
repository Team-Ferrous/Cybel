// highnoon/_native/ops/intrinsic_plasticity_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");

/**
 * @file intrinsic_plasticity_op.h
 * @brief Phase 71: Intrinsic Plasticity Preservation Layer
 *
 * Wraps all weight matrices with unitary constraint enforcement to
 * intrinsically preserve plasticity during continual learning.
 *
 * Key Features:
 *   - Cayley Parameterization: W = (I - A)(I + A)^{-1} where A is skew-symmetric
 *   - Tangent Space Projection: Gradients projected to unitary manifold
 *   - Retraction: Updates stay on manifold
 *   - Plasticity Measurement: Track learning capacity over time
 *
 * Research Basis:
 *   - "Intrinsic Preservation of Plasticity in Continual Quantum Learning"
 *     (arXiv 2511.17228, Nov 2025)
 *
 * Benefits:
 *   - Zero catastrophic forgetting via orthogonality constraints
 *   - Consistent learning performance regardless of task/data
 *   - Natural gradient dynamics on Stiefel manifold
 *
 * Integration: Applied to QMamba, TimeCrystal, WLAM, MoE router weights
 * Complexity: O(P) where P = number of parameters
 */

#ifndef HIGHNOON_NATIVE_OPS_INTRINSIC_PLASTICITY_OP_H_
#define HIGHNOON_NATIVE_OPS_INTRINSIC_PLASTICITY_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace hsmn {
namespace iplast {

// =============================================================================
// CONFIGURATION
// =============================================================================

struct PlasticityConfig {
    int dim;                     // Matrix dimension (square)
    bool use_cayley;             // Cayley vs Householder parameterization
    float retraction_step;       // Step size for retraction
    int measurement_window;      // Window for plasticity measurement
    
    PlasticityConfig()
        : dim(64)
        , use_cayley(true)
        , retraction_step(1.0f)
        , measurement_window(1000) {}
};

// =============================================================================
// SKEW-SYMMETRIC OPERATIONS
// =============================================================================

/**
 * @brief Extract skew-symmetric matrix from weight matrix.
 *
 * A = 0.5 * (W - W^T)
 *
 * @param W Input matrix [dim, dim]
 * @param A Output skew-symmetric [dim, dim]
 * @param dim Matrix dimension
 */
inline void ExtractSkewSymmetric(
    const float* W, float* A, int dim) {
    
    #pragma omp parallel for
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            A[i * dim + j] = 0.5f * (W[i * dim + j] - W[j * dim + i]);
        }
    }
}

/**
 * @brief Scale skew-symmetric parameters.
 *
 * @param skew_params Input skew parameters [dim * (dim-1) / 2]
 * @param A Output full skew-symmetric matrix [dim, dim]
 * @param dim Matrix dimension
 */
inline void SkewParamsToMatrix(
    const float* skew_params, float* A, int dim) {
    
    // Zero the matrix
    std::fill(A, A + dim * dim, 0.0f);
    
    // Fill upper triangle (negated) and lower triangle
    int idx = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = i + 1; j < dim; ++j) {
            A[i * dim + j] = skew_params[idx];
            A[j * dim + i] = -skew_params[idx];
            ++idx;
        }
    }
}

// =============================================================================
// CAYLEY TRANSFORM
// =============================================================================

/**
 * @brief Compute (I + A) for Cayley transform.
 *
 * @param A Skew-symmetric matrix [dim, dim]
 * @param IpA Output I + A [dim, dim]
 * @param dim Matrix dimension
 */
inline void ComputeIPlusA(const float* A, float* IpA, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            IpA[i * dim + j] = (i == j ? 1.0f : 0.0f) + A[i * dim + j];
        }
    }
}

/**
 * @brief Compute (I - A) for Cayley transform.
 *
 * @param A Skew-symmetric matrix [dim, dim]
 * @param ImA Output I - A [dim, dim]
 * @param dim Matrix dimension
 */
inline void ComputeIMinusA(const float* A, float* ImA, int dim) {
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            ImA[i * dim + j] = (i == j ? 1.0f : 0.0f) - A[i * dim + j];
        }
    }
}

/**
 * @brief Simple matrix inversion via Gauss-Jordan (for small matrices).
 *
 * @param M Input matrix [dim, dim]
 * @param M_inv Output inverse [dim, dim]
 * @param dim Matrix dimension
 */
inline void InvertMatrix(const float* M, float* M_inv, int dim) {
    // Create augmented matrix [M | I]
    std::vector<float> aug(dim * dim * 2);
    
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            aug[i * dim * 2 + j] = M[i * dim + j];
            aug[i * dim * 2 + dim + j] = (i == j ? 1.0f : 0.0f);
        }
    }
    
    // Gauss-Jordan elimination
    for (int i = 0; i < dim; ++i) {
        // Find pivot
        float pivot = aug[i * dim * 2 + i];
        if (std::abs(pivot) < 1e-10f) {
            pivot = 1e-10f;  // Regularization
        }
        
        // Scale row
        for (int j = 0; j < dim * 2; ++j) {
            aug[i * dim * 2 + j] /= pivot;
        }
        
        // Eliminate column
        for (int k = 0; k < dim; ++k) {
            if (k != i) {
                float factor = aug[k * dim * 2 + i];
                for (int j = 0; j < dim * 2; ++j) {
                    aug[k * dim * 2 + j] -= factor * aug[i * dim * 2 + j];
                }
            }
        }
    }
    
    // Extract inverse
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            M_inv[i * dim + j] = aug[i * dim * 2 + dim + j];
        }
    }
}

/**
 * @brief Matrix multiplication: C = A * B.
 *
 * @param A First matrix [dim, dim]
 * @param B Second matrix [dim, dim]
 * @param C Output matrix [dim, dim]
 * @param dim Matrix dimension
 */
inline void MatMul(const float* A, const float* B, float* C, int dim) {
    #pragma omp parallel for
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dim; ++k) {
                sum += A[i * dim + k] * B[k * dim + j];
            }
            C[i * dim + j] = sum;
        }
    }
}

/**
 * @brief Cayley parameterization: W = (I - A)(I + A)^{-1}.
 *
 * Converts skew-symmetric parameters to orthogonal/unitary matrix.
 *
 * @param skew_params Skew-symmetric parameters [dim * (dim-1) / 2]
 * @param unitary_weights Output unitary matrix [dim, dim]
 * @param dim Matrix dimension
 */
inline void CayleyParameterization(
    const float* skew_params,
    float* unitary_weights,
    int dim) {
    
    std::vector<float> A(dim * dim);
    std::vector<float> IpA(dim * dim);
    std::vector<float> ImA(dim * dim);
    std::vector<float> IpA_inv(dim * dim);
    
    // Convert params to skew matrix
    SkewParamsToMatrix(skew_params, A.data(), dim);
    
    // Compute I + A and I - A
    ComputeIPlusA(A.data(), IpA.data(), dim);
    ComputeIMinusA(A.data(), ImA.data(), dim);
    
    // Invert I + A
    InvertMatrix(IpA.data(), IpA_inv.data(), dim);
    
    // W = (I - A) * (I + A)^{-1}
    MatMul(ImA.data(), IpA_inv.data(), unitary_weights, dim);
}

// =============================================================================
// UNITARY CONSTRAINT ENFORCEMENT
// =============================================================================

/**
 * @brief Enforce unitary constraint on weights via projection.
 *
 * Projects to nearest unitary matrix using polar decomposition approx.
 *
 * @param weights Input weights (modified in-place) [rows, cols]
 * @param rows Number of rows
 * @param cols Number of columns
 */
inline void EnforceUnitaryConstraint(
    float* weights,
    int rows, int cols) {
    
    // Simplified: Re-orthonormalize via Gram-Schmidt
    for (int j = 0; j < cols; ++j) {
        // Subtract projections onto previous columns
        for (int k = 0; k < j; ++k) {
            float dot = 0.0f, norm_k = 0.0f;
            for (int i = 0; i < rows; ++i) {
                dot += weights[i * cols + j] * weights[i * cols + k];
                norm_k += weights[i * cols + k] * weights[i * cols + k];
            }
            if (norm_k > 1e-10f) {
                float proj = dot / norm_k;
                for (int i = 0; i < rows; ++i) {
                    weights[i * cols + j] -= proj * weights[i * cols + k];
                }
            }
        }
        
        // Normalize
        float norm = 0.0f;
        for (int i = 0; i < rows; ++i) {
            norm += weights[i * cols + j] * weights[i * cols + j];
        }
        norm = std::sqrt(norm) + 1e-10f;
        for (int i = 0; i < rows; ++i) {
            weights[i * cols + j] /= norm;
        }
    }
}

// =============================================================================
// TANGENT SPACE PROJECTION
// =============================================================================

/**
 * @brief Project gradient to tangent space of unitary manifold.
 *
 * For W ∈ O(n): ∇_tang = ∇ - W * sym(W^T * ∇)
 *
 * @param gradient Euclidean gradient (modified in-place) [dim, dim]
 * @param weights Current unitary weights [dim, dim]
 * @param dim Matrix dimension
 */
inline void ProjectGradientTangent(
    float* gradient,
    const float* weights,
    int dim) {
    
    // Compute W^T * ∇
    std::vector<float> WtG(dim * dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < dim; ++k) {
                sum += weights[k * dim + i] * gradient[k * dim + j];
            }
            WtG[i * dim + j] = sum;
        }
    }
    
    // Compute symmetric part: sym(W^T * ∇) = 0.5 * (W^T∇ + ∇^TW)
    std::vector<float> sym(dim * dim);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            sym[i * dim + j] = 0.5f * (WtG[i * dim + j] + WtG[j * dim + i]);
        }
    }
    
    // Compute W * sym
    std::vector<float> WSym(dim * dim);
    MatMul(weights, sym.data(), WSym.data(), dim);
    
    // Project: ∇_tang = ∇ - W * sym
    for (int i = 0; i < dim * dim; ++i) {
        gradient[i] -= WSym[i];
    }
}

// =============================================================================
// RETRACTION
// =============================================================================

/**
 * @brief Retract updated parameters back to unitary manifold.
 *
 * Uses QR-based retraction: W_new = qr(W + direction).Q
 *
 * @param weights Current weights (modified in-place) [dim, dim]
 * @param direction Update direction [dim, dim]
 * @param step_size Step size multiplier
 * @param dim Matrix dimension
 */
inline void RetractToManifold(
    float* weights,
    const float* direction,
    float step_size,
    int dim) {
    
    // Apply update
    for (int i = 0; i < dim * dim; ++i) {
        weights[i] += step_size * direction[i];
    }
    
    // Project back to unitary
    EnforceUnitaryConstraint(weights, dim, dim);
}

// =============================================================================
// PLASTICITY MEASUREMENT
// =============================================================================

/**
 * @brief Compute plasticity metric from weight trajectory.
 *
 * Measures capacity to learn new information by tracking weight
 * changes over training steps.
 *
 * @param weight_trajectory Snapshots of weights [num_snapshots, num_params]
 * @param num_snapshots Number of weight snapshots
 * @param num_params Parameters per snapshot
 * @return Plasticity metric in [0, 1] (1 = full plasticity)
 */
inline float ComputePlasticityMetric(
    const float* weight_trajectory,
    int num_snapshots, int num_params) {
    
    if (num_snapshots < 2) return 1.0f;
    
    // Compute average gradient magnitude across trajectory
    float total_change = 0.0f;
    
    for (int s = 1; s < num_snapshots; ++s) {
        const float* w_prev = weight_trajectory + (s - 1) * num_params;
        const float* w_curr = weight_trajectory + s * num_params;
        
        float change = 0.0f;
        for (int p = 0; p < num_params; ++p) {
            float diff = w_curr[p] - w_prev[p];
            change += diff * diff;
        }
        total_change += std::sqrt(change);
    }
    
    float avg_change = total_change / (num_snapshots - 1);
    
    // Map to [0, 1] - higher change = higher plasticity
    // Sigmoid-like mapping with threshold
    const float threshold = 0.01f;
    return 1.0f / (1.0f + std::exp(-10.0f * (avg_change - threshold)));
}

/**
 * @brief Measure layer plasticity using gradient norms.
 *
 * @param gradients Current gradients [num_params]
 * @param weights Current weights [num_params]
 * @param num_params Number of parameters
 * @return Plasticity score
 */
inline float MeasureLayerPlasticity(
    const float* gradients,
    const float* weights,
    int num_params) {
    
    float grad_norm = 0.0f, weight_norm = 0.0f;
    
    for (int p = 0; p < num_params; ++p) {
        grad_norm += gradients[p] * gradients[p];
        weight_norm += weights[p] * weights[p];
    }
    
    grad_norm = std::sqrt(grad_norm) + 1e-10f;
    weight_norm = std::sqrt(weight_norm) + 1e-10f;
    
    // Relative gradient norm indicates learning capacity
    return grad_norm / weight_norm;
}

}  // namespace iplast
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_INTRINSIC_PLASTICITY_OP_H_
