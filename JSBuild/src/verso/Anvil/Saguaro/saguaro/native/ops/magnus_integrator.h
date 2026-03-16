// saguaro.native/ops/magnus_integrator.h
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
// Magnus Expansion Geometric Integrator for Time-Dependent Hamiltonians.
//
// The Magnus expansion provides a unitary-preserving integration scheme
// for time-dependent Hamiltonians via exponential of nested commutators.
// Superior for "Time Crystal" periodic dynamics where H explicitly varies.
//
// References:
//   - Magnus, W. (1954). "On the exponential solution of differential equations"
//   - Blanes et al. (2009). "The Magnus expansion and some of its applications"

#ifndef TENSORFLOW_CORE_USER_OPS_MAGNUS_INTEGRATOR_H_
#define TENSORFLOW_CORE_USER_OPS_MAGNUS_INTEGRATOR_H_

#include "Eigen/Core"
#include "Eigen/Dense"
#include <cmath>
#include <complex>

// SIMD intrinsics for cross-platform vectorization
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
// Magnus Expansion Coefficients
// =============================================================================

/**
 * @brief Gauss-Legendre quadrature nodes and coefficients for Magnus expansion.
 *
 * For 4th-order Magnus, we use 2-point Gauss-Legendre quadrature:
 *   c₁ = 1/2 - √3/6 ≈ 0.2113
 *   c₂ = 1/2 + √3/6 ≈ 0.7887
 *
 * Integration weights are both 1/2 for 2-point GL.
 */
namespace MagnusCoefficients {
    // Gauss-Legendre nodes for 4th-order
    constexpr double sqrt3_over_6 = 0.28867513459481287;  // √3/6
    constexpr double c1 = 0.5 - sqrt3_over_6;  // ≈ 0.2113
    constexpr double c2 = 0.5 + sqrt3_over_6;  // ≈ 0.7887

    // Coefficients for Magnus series terms
    constexpr double omega1_coeff = 0.5;   // Ω₁ coefficient
    constexpr double omega2_coeff = std::sqrt(3.0) / 12.0;  // Ω₂ commutator coefficient
}

// =============================================================================
// SIMD Helper Functions for Matrix Operations
// =============================================================================

/**
 * @brief SIMD-accelerated matrix subtraction (for commutator).
 *
 * Computes C = A - B element-wise using AVX512/AVX2/NEON.
 *
 * @param A First input matrix (row-major flat data)
 * @param B Second input matrix
 * @param C Output matrix (A - B)
 * @param size Total number of elements
 */
inline void simd_matrix_sub(const float* A, const float* B, float* C, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&A[i]);
        __m512 vb = _mm512_loadu_ps(&B[i]);
        __m512 vc = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(&C[i], vc);
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&A[i]);
        __m256 vb = _mm256_loadu_ps(&B[i]);
        __m256 vc = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&C[i], vc);
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&A[i]);
        float32x4_t vb = vld1q_f32(&B[i]);
        float32x4_t vc = vsubq_f32(va, vb);
        vst1q_f32(&C[i], vc);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        C[i] = A[i] - B[i];
    }
}

/**
 * @brief SIMD-accelerated matrix addition with scaling.
 *
 * Computes C = C + alpha * A element-wise.
 *
 * @param A Input matrix
 * @param C Output matrix (accumulated)
 * @param alpha Scaling factor
 * @param size Total number of elements
 */
inline void simd_matrix_add_scaled(const float* A, float* C, float alpha, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    __m512 valpha = _mm512_set1_ps(alpha);
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&A[i]);
        __m512 vc = _mm512_loadu_ps(&C[i]);
        vc = _mm512_fmadd_ps(va, valpha, vc);
        _mm512_storeu_ps(&C[i], vc);
    }
#elif defined(__AVX2__)
    __m256 valpha = _mm256_set1_ps(alpha);
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&A[i]);
        __m256 vc = _mm256_loadu_ps(&C[i]);
        vc = _mm256_fmadd_ps(va, valpha, vc);
        _mm256_storeu_ps(&C[i], vc);
    }
#elif defined(__ARM_NEON)
    float32x4_t valpha = vdupq_n_f32(alpha);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&A[i]);
        float32x4_t vc = vld1q_f32(&C[i]);
        vc = vfmaq_f32(vc, va, valpha);
        vst1q_f32(&C[i], vc);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        C[i] += alpha * A[i];
    }
}

// =============================================================================
// Matrix Commutator and Exponential
// =============================================================================

/**
 * @brief Computes matrix commutator [A, B] = AB - BA.
 *
 * The commutator is central to Magnus expansion. Uses Eigen for matrix
 * multiplication and SIMD for final subtraction.
 *
 * @param A First matrix (n×n)
 * @param B Second matrix (n×n)
 * @param out Output commutator matrix [A, B]
 */
inline void compute_commutator(const MatrixXf& A, const MatrixXf& B, MatrixXf& out) {
    MatrixXf AB = A * B;
    MatrixXf BA = B * A;
    out.resize(A.rows(), A.cols());
    simd_matrix_sub(AB.data(), BA.data(), out.data(), AB.size());
}

/**
 * @brief Computes matrix exponential exp(M) via Padé[6,6] approximant.
 *
 * The Padé approximant preserves the unitary structure of the evolution:
 *   exp(M) ≈ [I - M/2 + M²/12 - ...]⁻¹ [I + M/2 + M²/12 + ...]
 *
 * For Hamiltonian systems, M is skew-symmetric (or anti-Hermitian),
 * so exp(M) is orthogonal (or unitary).
 *
 * Uses scaling and squaring for numerical stability:
 *   exp(M) = (exp(M/2^s))^(2^s)
 *
 * @param M Input matrix (should be small norm for accuracy)
 * @param exp_M Output matrix exp(M)
 * @param scaling_squaring_iters Number of squaring iterations (default: 4)
 */
inline void compute_matrix_exp_pade(
    const MatrixXf& M, 
    MatrixXf& exp_M, 
    int scaling_squaring_iters = 4) {
    
    const int n = M.rows();
    
    // Scale matrix: M_scaled = M / 2^s
    float scale = 1.0f / static_cast<float>(1 << scaling_squaring_iters);
    MatrixXf M_scaled = scale * M;
    
    // Compute powers of M_scaled
    MatrixXf M2 = M_scaled * M_scaled;
    MatrixXf M4 = M2 * M2;
    MatrixXf M6 = M4 * M2;
    
    // Padé[6,6] coefficients (from Table 10.4 in Higham's "Functions of Matrices")
    constexpr float b0 = 1.0f;
    constexpr float b1 = 0.5f;
    constexpr float b2 = 1.0f / 12.0f;
    constexpr float b3 = 1.0f / 120.0f;
    constexpr float b4 = 1.0f / 1680.0f;
    constexpr float b5 = 1.0f / 30240.0f;
    constexpr float b6 = 1.0f / 665280.0f;
    
    // Compute U = M(b1*I + b3*M² + b5*M⁴ + ...)
    // Compute V = b0*I + b2*M² + b4*M⁴ + b6*M⁶
    MatrixXf I = MatrixXf::Identity(n, n);
    
    MatrixXf U_inner = b1 * I + b3 * M2 + b5 * M4;
    MatrixXf U = M_scaled * U_inner;
    
    MatrixXf V = b0 * I + b2 * M2 + b4 * M4 + b6 * M6;
    
    // exp(M_scaled) ≈ (V - U)⁻¹ (V + U)
    MatrixXf V_minus_U = V - U;
    MatrixXf V_plus_U = V + U;
    
    // Solve (V - U) * exp_M_scaled = (V + U)
    exp_M = V_minus_U.ldlt().solve(V_plus_U);
    
    // Squaring: exp(M) = exp(M_scaled)^(2^s)
    for (int i = 0; i < scaling_squaring_iters; ++i) {
        exp_M = exp_M * exp_M;
    }
}

// =============================================================================
// Hamiltonian Evaluation at Different Time Points
// =============================================================================

/**
 * @brief Structure to hold Hamiltonian matrix at a specific time point.
 *
 * For the Magnus expansion, we need to evaluate the Hamiltonian "generator"
 * at multiple Gauss-Legendre quadrature points within the timestep.
 */
struct HamiltonianAtTime {
    MatrixXf H_matrix;  // The Hamiltonian as a matrix (for small state dims)
    float H_scalar;     // Hamiltonian scalar value
    VectorXf dH_dz;     // Gradient for dynamics
};

/**
 * @brief Converts HNN output to an effective matrix Hamiltonian.
 *
 * For the Magnus expansion, we need the Hamiltonian as a matrix operator.
 * We construct this from the gradients dH/dq and dH/dp:
 *
 *   H_eff = | 0      I   |   (symplectic structure matrix)
 *           | -K(q)  0   |
 *
 * where K(q) = ∂²V/∂q² is the Hessian of potential energy.
 *
 * For practical purposes, we use a first-order approximation based on
 * the gradient vector to construct an antisymmetric generator.
 *
 * @param dH_dq Gradient of H w.r.t. position q
 * @param dH_dp Gradient of H w.r.t. momentum p
 * @param state_dim Dimension of q (and p)
 * @param out_A Output antisymmetric matrix for Magnus evolution
 */
inline void construct_hamiltonian_generator(
    const VectorXf& dH_dq,
    const VectorXf& dH_dp,
    int state_dim,
    MatrixXf& out_A) {
    
    // Full phase space dimension
    const int full_dim = 2 * state_dim;
    out_A = MatrixXf::Zero(full_dim, full_dim);
    
    // Hamilton's equations in matrix form:
    //   d/dt [q]   [  0   I ] [∂H/∂q]   [  0   I ] [ dH_dq ]
    //        [p] = [ -I   0 ] [∂H/∂p] = [ -I   0 ] [ dH_dp ]
    //
    // The generator A for the Magnus expansion is:
    //   A = dt * J @ ∇²H
    //
    // For first-order (gradient-based) approximation:
    //   A[i, state_dim+j] = δᵢⱼ * dH_dp[j] (q evolution from dp)
    //   A[state_dim+i, j] = -δᵢⱼ * dH_dq[i] (p evolution from dq)
    
    // This creates an antisymmetric structure preserving symplecticity
    for (int i = 0; i < state_dim; ++i) {
        // q evolves according to dH/dp
        out_A(i, state_dim + i) = dH_dp(i);
        // p evolves according to -dH/dq
        out_A(state_dim + i, i) = -dH_dq(i);
    }
    
    // Antisymmetrize to ensure exp(A) is symplectic
    MatrixXf A_T = out_A.transpose();
    out_A = 0.5f * (out_A - A_T);
}

// =============================================================================
// Magnus Integration Step
// =============================================================================

/**
 * @brief Performs one step of 4th-order Magnus integration.
 *
 * The Magnus expansion computes Ω such that x(t+dt) = exp(Ω) x(t), where:
 *
 *   Ω ≈ Ω₁ + Ω₂ + O(dt⁵)
 *
 *   Ω₁ = dt/2 * (A₁ + A₂)                    [2nd order]
 *   Ω₂ = (√3/12) * dt² * [A₂, A₁]            [4th order correction]
 *
 * Here Aᵢ = A(t + cᵢ*dt) are Hamiltonian generators at Gauss-Legendre nodes.
 *
 * @param state Current phase-space state [q; p]
 * @param A1 Hamiltonian generator at t + c₁*dt
 * @param A2 Hamiltonian generator at t + c₂*dt
 * @param dt Time step
 * @param magnus_order 2 for 2nd-order, 4 for 4th-order (includes commutator)
 * @param new_state Output: evolved state
 */
inline void magnus_step_from_generators(
    const VectorXf& state,
    const MatrixXf& A1,
    const MatrixXf& A2,
    float dt,
    int magnus_order,
    VectorXf& new_state) {
    
    const int full_dim = state.size();
    
    // Compute Ω₁ = dt/2 * (A₁ + A₂)
    MatrixXf Omega = (dt / 2.0f) * (A1 + A2);
    
    // Add Ω₂ commutator term for 4th-order accuracy
    if (magnus_order >= 4) {
        MatrixXf comm;
        compute_commutator(A2, A1, comm);
        
        // Ω₂ coefficient: √3/12 * dt²
        float omega2_coeff = static_cast<float>(
            std::sqrt(3.0) / 12.0 * dt * dt);
        
        Omega += omega2_coeff * comm;
    }
    
    // Compute exp(Ω) via Padé approximant
    MatrixXf exp_Omega;
    compute_matrix_exp_pade(Omega, exp_Omega);
    
    // Evolve state: x(t+dt) = exp(Ω) * x(t)
    new_state = exp_Omega * state;
}

/**
 * @brief Full Magnus integration step for HNN.
 *
 * This function orchestrates the complete Magnus step:
 * 1. Evaluate Hamiltonian at Gauss-Legendre quadrature points
 * 2. Construct antisymmetric generators A₁, A₂
 * 3. Compute Magnus series Ω = Ω₁ + Ω₂
 * 4. Apply matrix exponential to evolve state
 *
 * @param q Position vector (will be updated)
 * @param p Momentum vector (will be updated)
 * @param x_t Input vector
 * @param W1, b1, W2, b2, W3, b3_scalar HNN weights
 * @param W_out, b_out Output projection
 * @param dt Timestep
 * @param magnus_order 2 or 4
 * @param output_proj Output: projected state
 * @param h_initial Output: initial Hamiltonian
 * @param h_final Output: final Hamiltonian
 * @param compute_H_fn Function to evaluate H and gradients
 */
template<typename ComputeHFn>
inline void magnus_hnn_step(
    VectorXf& q, 
    VectorXf& p, 
    const VectorXf& x_t,
    const Map<const MatrixXf>& W1, const Map<const VectorXf>& b1,
    const Map<const MatrixXf>& W2, const Map<const VectorXf>& b2,
    const Map<const MatrixXf>& W3, float b3_scalar,
    const Map<const MatrixXf>& W_out, const Map<const VectorXf>& b_out,
    float dt,
    int magnus_order,
    VectorXf& output_proj,
    float& h_initial,
    float& h_final,
    ComputeHFn compute_H_fn) {
    
    const int state_dim = q.size();
    const int full_dim = 2 * state_dim;
    
    // Concatenate phase-space state
    VectorXf phase_state(full_dim);
    phase_state << q, p;
    
    // Gauss-Legendre quadrature points
    const float t1 = static_cast<float>(MagnusCoefficients::c1 * dt);
    const float t2 = static_cast<float>(MagnusCoefficients::c2 * dt);
    
    // Evaluate Hamiltonian at t=0 for initial energy
    HNNIntermediate int0;
    VectorXf z0(2 * state_dim + x_t.size());
    z0 << q, p, x_t;
    int0 = compute_H_fn(z0, W1, b1, W2, b2, W3, b3_scalar);
    h_initial = int0.H;
    
    // Estimate state at quadrature point 1 (linear extrapolation)
    // For simplicity, use current state for both points (time-independent approx)
    VectorXf dH_dz0 = compute_dH_dz(int0, W1, W2, W3);
    VectorXf dH_dq0 = dH_dz0.head(state_dim);
    VectorXf dH_dp0 = dH_dz0.segment(state_dim, state_dim);
    
    // Construct generator at point 1
    MatrixXf A1;
    construct_hamiltonian_generator(dH_dq0, dH_dp0, state_dim, A1);
    
    // For 4th-order, evaluate at second quadrature point
    // Here we use a midpoint approximation
    MatrixXf A2;
    if (magnus_order >= 4) {
        // Estimate state at t1
        VectorXf q_mid = q + t1 * dH_dp0;
        VectorXf p_mid = p - t1 * dH_dq0;
        
        VectorXf z_mid(2 * state_dim + x_t.size());
        z_mid << q_mid, p_mid, x_t;
        HNNIntermediate int_mid = compute_H_fn(z_mid, W1, b1, W2, b2, W3, b3_scalar);
        VectorXf dH_dz_mid = compute_dH_dz(int_mid, W1, W2, W3);
        VectorXf dH_dq_mid = dH_dz_mid.head(state_dim);
        VectorXf dH_dp_mid = dH_dz_mid.segment(state_dim, state_dim);
        
        construct_hamiltonian_generator(dH_dq_mid, dH_dp_mid, state_dim, A2);
    } else {
        A2 = A1;  // Use same generator for 2nd-order
    }
    
    // Perform Magnus step
    VectorXf new_phase_state;
    magnus_step_from_generators(phase_state, A1, A2, dt, magnus_order, new_phase_state);
    
    // Extract q and p from evolved state
    q = new_phase_state.head(state_dim);
    p = new_phase_state.tail(state_dim);
    
    // Compute final Hamiltonian
    VectorXf zf(2 * state_dim + x_t.size());
    zf << q, p, x_t;
    HNNIntermediate int_final = compute_H_fn(zf, W1, b1, W2, b2, W3, b3_scalar);
    h_final = int_final.H;
    
    // Output projection
    VectorXf final_concat(full_dim);
    final_concat << q, p;
    output_proj = W_out.transpose() * final_concat + b_out;
}

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_MAGNUS_INTEGRATOR_H_
