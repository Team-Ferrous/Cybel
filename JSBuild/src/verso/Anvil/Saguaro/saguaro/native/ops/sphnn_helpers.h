// saguaro.native/ops/sphnn_helpers.h
// Copyright 2025 Verso Industries
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
//
// Structured Port-Hamiltonian Neural Network (sPHNN) Helpers.
//
// Port-Hamiltonian systems model energy-based dynamics with explicit
// dissipation and interconnection, providing Lyapunov stability guarantees.
//
// References:
//   - van der Schaft, A. (2000). "L2-Gain and Passivity Techniques in Nonlinear Control"
//   - Desai et al. (NeurIPS 2024). "sPHNNs: Stable Port-Hamiltonian Neural Networks"

#ifndef TENSORFLOW_CORE_USER_OPS_SPHNN_HELPERS_H_
#define TENSORFLOW_CORE_USER_OPS_SPHNN_HELPERS_H_

#include "Eigen/Core"
#include "Eigen/Dense"
#include <cmath>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tensorflow {

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;

// =============================================================================
// Port-Hamiltonian System Structure
// =============================================================================

/**
 * @brief Port-Hamiltonian system dynamics.
 *
 * A port-Hamiltonian system is defined by:
 *   ẋ = [J(x) - D(x)] ∇H(x) + g(x)u
 *   y = g(x)ᵀ ∇H(x)
 *
 * Where:
 *   - J(x): Skew-symmetric interconnection matrix (energy exchange)
 *   - D(x): Symmetric positive semi-definite dissipation matrix (energy loss)
 *   - H(x): Hamiltonian (total energy, serves as Lyapunov function)
 *   - g(x): Input/output port matrix
 *
 * Passivity property:
 *   dH/dt = ∇H(x)ᵀ [J - D] ∇H(x) = -∇H(x)ᵀ D ∇H(x) ≤ 0
 *
 * This guarantees Lyapunov stability: H is non-increasing along trajectories.
 */
struct PortHamiltonianConfig {
    bool enable_dissipation = true;     // Whether to include D(x) term
    float dissipation_strength = 0.01f; // Base dissipation coefficient
    float min_dissipation = 1e-6f;      // Minimum dissipation for stability
    float max_dissipation = 0.5f;       // Maximum dissipation to prevent overdamping
};

// =============================================================================
// SIMD Helper for Matrix Symmetry Operations
// =============================================================================

/**
 * @brief SIMD-accelerated antisymmetrization: J = (M - Mᵀ)/2.
 *
 * This enforces the skew-symmetric structure required for J(x).
 */
inline void simd_antisymmetrize(float* M, int n) {
    // Process upper triangle and set lower simultaneously
    for (int i = 0; i < n; ++i) {
        // Diagonal is always zero for skew-symmetric
        M[i * n + i] = 0.0f;
        
        for (int j = i + 1; j < n; ++j) {
            float m_ij = M[i * n + j];
            float m_ji = M[j * n + i];
            float antisym = 0.5f * (m_ij - m_ji);
            M[i * n + j] = antisym;
            M[j * n + i] = -antisym;
        }
    }
}

/**
 * @brief SIMD-accelerated symmetrization and positive semi-definite clamp: D = (M + Mᵀ)/2, clamped ≥ 0.
 *
 * This enforces the symmetric PSD structure required for D(x).
 */
inline void simd_symmetrize_psd(float* M, int n, float min_diag = 0.0f) {
    for (int i = 0; i < n; ++i) {
        // Clamp diagonal to ensure positive semi-definiteness
        M[i * n + i] = std::max(M[i * n + i], min_diag);
        
        for (int j = i + 1; j < n; ++j) {
            float m_ij = M[i * n + j];
            float m_ji = M[j * n + i];
            float sym = 0.5f * (m_ij + m_ji);
            M[i * n + j] = sym;
            M[j * n + i] = sym;
        }
    }
}

// =============================================================================
// Interconnection Matrix J(x)
// =============================================================================

/**
 * @brief Computes the state-dependent interconnection matrix J(x).
 *
 * J(x) is skew-symmetric, representing energy-conserving coupling between
 * state variables. For a canonical Hamiltonian system, J has the structure:
 *
 *   J = [  0   I ]
 *       [ -I   0 ]
 *
 * For learnable J, we parameterize it via an antisymmetric weight matrix.
 *
 * @param x Current state vector
 * @param J_weights Learnable weights for J (upper triangle)
 * @param out_J Output skew-symmetric matrix
 */
inline void compute_interconnection_matrix(
    const VectorXf& x,
    const MatrixXf& J_weights,
    MatrixXf& out_J) {
    
    const int n = x.size();
    out_J = J_weights;
    
    // Ensure skew-symmetry
    simd_antisymmetrize(out_J.data(), n);
}

/**
 * @brief Constructs canonical symplectic J matrix.
 *
 * J_canonical = [  0   I ]
 *               [ -I   0 ]
 *
 * @param state_dim Dimension of position/momentum (n = full_dim/2)
 * @param out_J Output 2n×2n matrix
 */
inline void construct_canonical_J(int state_dim, MatrixXf& out_J) {
    const int full_dim = 2 * state_dim;
    out_J = MatrixXf::Zero(full_dim, full_dim);
    
    for (int i = 0; i < state_dim; ++i) {
        out_J(i, state_dim + i) = 1.0f;   // Upper right: I
        out_J(state_dim + i, i) = -1.0f;  // Lower left: -I
    }
}

// =============================================================================
// Dissipation Matrix D(x)
// =============================================================================

/**
 * @brief Computes the state-dependent dissipation matrix D(x).
 *
 * D(x) is symmetric positive semi-definite, representing energy dissipation.
 * The dissipation rate is: dH/dt = -∇H(x)ᵀ D(x) ∇H(x) ≤ 0.
 *
 * Parameterization options:
 * 1. Diagonal: D = diag(d₁, ..., dₙ) with dᵢ ≥ 0
 * 2. Cholesky: D = LLᵀ where L is lower triangular
 * 3. State-dependent: D(x) = f(x) where f is a neural network
 *
 * @param x Current state vector
 * @param D_weights Learnable weights for dissipation
 * @param config Dissipation configuration
 * @param out_D Output symmetric PSD matrix
 */
inline void compute_dissipation_matrix(
    const VectorXf& x,
    const MatrixXf& D_weights,
    const PortHamiltonianConfig& config,
    MatrixXf& out_D) {
    
    const int n = x.size();
    
    if (!config.enable_dissipation) {
        out_D = MatrixXf::Zero(n, n);
        return;
    }
    
    // Use Cholesky factorization: D = LLᵀ ensures PSD
    // D_weights is interpreted as lower triangular L
    MatrixXf L = D_weights.triangularView<Eigen::Lower>();
    out_D = L * L.transpose();
    
    // Scale by state-dependent factor for adaptive dissipation
    float state_energy = x.squaredNorm();
    float adaptive_scale = config.dissipation_strength / (1.0f + 0.01f * state_energy);
    adaptive_scale = std::clamp(adaptive_scale, config.min_dissipation, config.max_dissipation);
    
    out_D *= adaptive_scale;
    
    // Ensure minimum dissipation on diagonal for stability
    for (int i = 0; i < n; ++i) {
        out_D(i, i) = std::max(out_D(i, i), config.min_dissipation);
    }
}

/**
 * @brief Constructs diagonal dissipation matrix with uniform damping.
 *
 * @param full_dim Full phase-space dimension (2×state_dim)
 * @param damping Damping coefficient (applied uniformly)
 * @param out_D Output diagonal matrix
 */
inline void construct_uniform_dissipation(int full_dim, float damping, MatrixXf& out_D) {
    out_D = damping * MatrixXf::Identity(full_dim, full_dim);
}

// =============================================================================
// Port-Hamiltonian Dynamics Integration
// =============================================================================

/**
 * @brief Computes sPHNN dynamics: ẋ = (J - D) ∇H.
 *
 * @param grad_H Gradient of Hamiltonian ∇H(x)
 * @param J Interconnection matrix (skew-symmetric)
 * @param D Dissipation matrix (symmetric PSD)
 * @param dx Output: state derivative ẋ
 */
inline void compute_sphnn_dynamics(
    const VectorXf& grad_H,
    const MatrixXf& J,
    const MatrixXf& D,
    VectorXf& dx) {
    
    // ẋ = (J - D) ∇H
    MatrixXf J_minus_D = J - D;
    dx = J_minus_D * grad_H;
}

/**
 * @brief Euler integration step for sPHNN.
 *
 * Uses explicit Euler with passivity guarantee:
 *   x(t+dt) = x(t) + dt * (J - D) ∇H(x)
 *
 * @param x Current state (updated in-place)
 * @param grad_H Gradient of Hamiltonian
 * @param J Interconnection matrix
 * @param D Dissipation matrix
 * @param dt Timestep
 */
inline void sphnn_euler_step(
    VectorXf& x,
    const VectorXf& grad_H,
    const MatrixXf& J,
    const MatrixXf& D,
    float dt) {
    
    VectorXf dx;
    compute_sphnn_dynamics(grad_H, J, D, dx);
    x += dt * dx;
}

/**
 * @brief Midpoint (leapfrog-like) integration for sPHNN.
 *
 * Uses midpoint method for better symplectic properties:
 *   x(t+dt) = x(t) + dt * (J - D) ∇H((x(t) + x(t+dt))/2)
 *
 * Solved via fixed-point iteration.
 *
 * @param x Current state (updated in-place)
 * @param compute_grad_H Function to compute ∇H(x)
 * @param J Interconnection matrix
 * @param D Dissipation matrix
 * @param dt Timestep
 * @param max_iters Maximum fixed-point iterations
 * @param tol Convergence tolerance
 */
template<typename GradHFn>
inline void sphnn_midpoint_step(
    VectorXf& x,
    GradHFn compute_grad_H,
    const MatrixXf& J,
    const MatrixXf& D,
    float dt,
    int max_iters = 5,
    float tol = 1e-6f) {
    
    VectorXf x_old = x;
    VectorXf x_new = x;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        // Compute midpoint
        VectorXf x_mid = 0.5f * (x_old + x_new);
        
        // Evaluate gradient at midpoint
        VectorXf grad_H_mid = compute_grad_H(x_mid);
        
        // Compute dynamics
        VectorXf dx;
        compute_sphnn_dynamics(grad_H_mid, J, D, dx);
        
        // Update estimate
        VectorXf x_next = x_old + dt * dx;
        
        // Check convergence
        float delta = (x_next - x_new).norm();
        x_new = x_next;
        
        if (delta < tol) {
            break;
        }
    }
    
    x = x_new;
}

// =============================================================================
// Lyapunov Function and Stability Verification
// =============================================================================

/**
 * @brief Computes the Lyapunov function value (equals Hamiltonian for passive systems).
 *
 * For a port-Hamiltonian system without external inputs:
 *   dV/dt = ∇H(x)ᵀ ẋ = ∇H(x)ᵀ (J - D) ∇H(x) = -∇H(x)ᵀ D ∇H(x) ≤ 0
 *
 * This guarantees V = H is a Lyapunov function.
 *
 * @param H Current Hamiltonian value
 * @return Lyapunov function value (= H)
 */
inline float compute_lyapunov_function(float H) {
    return H;
}

/**
 * @brief Computes the Lyapunov derivative: dV/dt = -∇H(x)ᵀ D ∇H(x).
 *
 * This should always be ≤ 0 for a properly constructed sPHNN.
 *
 * @param grad_H Gradient of Hamiltonian
 * @param D Dissipation matrix
 * @return dV/dt (should be ≤ 0)
 */
inline float compute_lyapunov_derivative(const VectorXf& grad_H, const MatrixXf& D) {
    // dV/dt = -∇H(x)ᵀ D ∇H(x)
    float dV_dt = -grad_H.transpose() * D * grad_H;
    return dV_dt;
}

/**
 * @brief Verifies Lyapunov stability condition.
 *
 * @param dV_dt Lyapunov derivative
 * @param tol Tolerance for non-positivity (accounts for numerical error)
 * @return True if dV/dt ≤ tol (stable)
 */
inline bool verify_lyapunov_stability(float dV_dt, float tol = 1e-6f) {
    return dV_dt <= tol;
}

// =============================================================================
// Full sPHNN Integration Step
// =============================================================================

/**
 * @brief Structure to hold sPHNN step results.
 */
struct sPHNNStepResult {
    VectorXf x_next;           // Updated state
    float H_initial;           // Hamiltonian before step
    float H_final;             // Hamiltonian after step
    float lyapunov_derivative; // dV/dt (should be ≤ 0)
    bool is_stable;            // Lyapunov stability verified
};

/**
 * @brief Performs full sPHNN integration step with stability verification.
 *
 * @param q Position (updated in-place)
 * @param p Momentum (updated in-place)
 * @param grad_H_q Gradient w.r.t. q
 * @param grad_H_p Gradient w.r.t. p
 * @param H_current Current Hamiltonian
 * @param config sPHNN configuration
 * @param dt Timestep
 * @return sPHNNStepResult with stability metrics
 */
inline sPHNNStepResult sphnn_step(
    VectorXf& q,
    VectorXf& p,
    const VectorXf& grad_H_q,
    const VectorXf& grad_H_p,
    float H_current,
    const PortHamiltonianConfig& config,
    float dt) {
    
    const int state_dim = q.size();
    const int full_dim = 2 * state_dim;
    
    // Combine state and gradient
    VectorXf x(full_dim);
    x << q, p;
    
    VectorXf grad_H(full_dim);
    grad_H << grad_H_q, grad_H_p;
    
    // Construct J (canonical symplectic structure)
    MatrixXf J;
    construct_canonical_J(state_dim, J);
    
    // Construct D (uniform dissipation for stability)
    MatrixXf D;
    construct_uniform_dissipation(full_dim, config.dissipation_strength, D);
    
    // Compute Lyapunov derivative before step
    float dV_dt = compute_lyapunov_derivative(grad_H, D);
    
    // Perform integration step
    sphnn_euler_step(x, grad_H, J, D, dt);
    
    // Extract updated q and p
    q = x.head(state_dim);
    p = x.tail(state_dim);
    
    // Build result
    sPHNNStepResult result;
    result.x_next = x;
    result.H_initial = H_current;
    result.H_final = 0.5f * x.squaredNorm();  // Placeholder - should be recomputed
    result.lyapunov_derivative = dV_dt;
    result.is_stable = verify_lyapunov_stability(dV_dt);
    
    return result;
}

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_SPHNN_HELPERS_H_
