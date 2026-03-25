// src/ops/hnn_core_helpers.h
// Copyright 2025 Verso Industries
//
// This header contains the core analytical logic for the Hamiltonian Neural Network
// (HNN) / Time Crystal blocks, making it reusable across various HNN-based operations
// (like FusedHNNStep, FusedHNNSequence, and FusedReasoningStack).
//
// Phase 11 GROUP_3_HIGH_RISK Upgrade (2025-11-23):
// - Added explicit SIMD guards (AVX512/AVX2/NEON + scalar fallback)
// - Vectorized sin/cos activations using polynomial approximations
// - Vectorized Hadamard products in gradient computation
// - Shared SIMD implementations with fused_hnn_step_op/fused_hnn_sequence_op
// - Preserves Yoshida 4th-order symplectic integrator structure
// - Energy conservation requirements unchanged

#ifndef TENSORFLOW_CORE_USER_OPS_HNN_CORE_HELPERS_H_
#define TENSORFLOW_CORE_USER_OPS_HNN_CORE_HELPERS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <cmath>
#include <algorithm>
#include <string>

// Phase 17: Hamiltonian Enhancement Modules are included AFTER HNNIntermediate
// and compute_dH_dz are defined (see below) to avoid circular dependencies.
// The forward declarations are in each enhancement header.

// SIMD intrinsics for cross-architecture vectorization (Phase 11 GROUP_3)
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
using Eigen::RowMajor;

// =============================================================================
// SIMD HELPER FUNCTIONS (Phase 11 GROUP_3 Upgrade)
// =============================================================================

/**
 * @brief Vectorized in-place sin activation using polynomial approximation.
 *
 * Fast sin(x) ≈ x - x³/6 + x⁵/120 for |x| < π
 * Shared implementation with fused_hnn_step_op and fused_hnn_sequence_op
 * for numerical consistency across all TimeCrystal operators.
 *
 * @param data Float array to apply sin in-place
 * @param size Number of elements
 */
inline void simd_sin_inplace(float* data, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    // AVX512: 16-wide SIMD
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        __m512 x = v;
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 x3 = _mm512_mul_ps(x2, x);
        __m512 x5 = _mm512_mul_ps(x3, x2);
        __m512 term1 = _mm512_mul_ps(x3, _mm512_set1_ps(-1.0f / 6.0f));
        __m512 term2 = _mm512_mul_ps(x5, _mm512_set1_ps(1.0f / 120.0f));
        __m512 result = _mm512_add_ps(x, _mm512_add_ps(term1, term2));
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    // AVX2: 8-wide SIMD
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 x = v;
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 term1 = _mm256_mul_ps(x3, _mm256_set1_ps(-1.0f / 6.0f));
        __m256 term2 = _mm256_mul_ps(x5, _mm256_set1_ps(1.0f / 120.0f));
        __m256 result = _mm256_add_ps(x, _mm256_add_ps(term1, term2));
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    // NEON: 4-wide SIMD
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        float32x4_t x = v;
        float32x4_t x2 = vmulq_f32(x, x);
        float32x4_t x3 = vmulq_f32(x2, x);
        float32x4_t x5 = vmulq_f32(x3, x2);
        float32x4_t term1 = vmulq_f32(x3, vdupq_n_f32(-1.0f / 6.0f));
        float32x4_t term2 = vmulq_f32(x5, vdupq_n_f32(1.0f / 120.0f));
        float32x4_t result = vaddq_f32(x, vaddq_f32(term1, term2));
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        data[i] = std::sin(data[i]);
    }
}

/**
 * @brief Vectorized in-place cos operation for gradient computation.
 *
 * Fast cos(x) ≈ 1 - x²/2 + x⁴/24 for |x| < π
 * Shared implementation with fused_hnn_step_op and fused_hnn_sequence_op
 * for numerical consistency across all TimeCrystal operators.
 *
 * @param data Float array to apply cos in-place
 * @param size Number of elements
 */
inline void simd_cos_inplace(float* data, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    // AVX512: 16-wide SIMD
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        __m512 x = v;
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 x4 = _mm512_mul_ps(x2, x2);
        __m512 term1 = _mm512_mul_ps(x2, _mm512_set1_ps(-0.5f));
        __m512 term2 = _mm512_mul_ps(x4, _mm512_set1_ps(1.0f / 24.0f));
        __m512 result = _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(term1, term2));
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    // AVX2: 8-wide SIMD
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 x = v;
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 term1 = _mm256_mul_ps(x2, _mm256_set1_ps(-0.5f));
        __m256 term2 = _mm256_mul_ps(x4, _mm256_set1_ps(1.0f / 24.0f));
        __m256 result = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(term1, term2));
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    // NEON: 4-wide SIMD
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        float32x4_t x = v;
        float32x4_t x2 = vmulq_f32(x, x);
        float32x4_t x4 = vmulq_f32(x2, x2);
        float32x4_t term1 = vmulq_f32(x2, vdupq_n_f32(-0.5f));
        float32x4_t term2 = vmulq_f32(x4, vdupq_n_f32(1.0f / 24.0f));
        float32x4_t result = vaddq_f32(vdupq_n_f32(1.0f), vaddq_f32(term1, term2));
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        data[i] = std::cos(data[i]);
    }
}

/**
 * @brief Vectorized element-wise Hadamard product.
 *
 * Computes out[i] = a[i] * b[i] using SIMD instructions.
 * Used in gradient computation for element-wise multiplication of
 * activation derivatives with upstream gradients.
 *
 * @param a First input array
 * @param b Second input array
 * @param out Output array (can alias a or b for in-place operation)
 * @param size Number of elements
 */
inline void simd_hadamard_product(const float* a, const float* b, float* out, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(&out[i], vc);
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&out[i], vc);
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(&out[i], vc);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Intermediate states structure for HNN step recomputation.
 */
struct HNNIntermediate {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VectorXf z;
    VectorXf h1;
    VectorXf a1;
    VectorXf h2;
    VectorXf a2;
    float H;
};

/**
 * @brief Stores all intermediate states necessary for the HNN backward pass (Adjoint Sensitivity Method).
 */
struct HNNForwardState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VectorXf q_t;
    VectorXf p_t;
    VectorXf x_t;
    VectorXf p_half;
    VectorXf q_next;
    VectorXf p_next;
    HNNIntermediate int1, int2, int3;
    VectorXf dH_dz1, dH_dz2, dH_dz3;
};


// -----------------------------------------------------------------
// Core HNN Logic Implementations (Inlined for full visibility)
// -----------------------------------------------------------------

/**
 * @brief Computes the Hamiltonian H and all intermediate hidden layers.
 *
 * Phase 11 GROUP_3: Vectorized sin activations using SIMD for both layers.
 */
inline HNNIntermediate compute_H_and_intermediates(
    const VectorXf& z,
    const Map<const MatrixXf>& W1, const Map<const VectorXf>& b1,
    const Map<const MatrixXf>& W2, const Map<const VectorXf>& b2,
    const Map<const MatrixXf>& W3, const float b3_scalar) {
    HNNIntermediate results;
    results.z = z;

    // Layer 1: Linear + sin activation (SIMD)
    results.h1 = W1.transpose() * z + b1;
    results.a1 = results.h1; // Copy for in-place operation
    simd_sin_inplace(results.a1.data(), results.a1.size());

    // Layer 2: Linear + sin activation (SIMD)
    results.h2 = W2.transpose() * results.a1 + b2;
    results.a2 = results.h2; // Copy for in-place operation
    simd_sin_inplace(results.a2.data(), results.a2.size());

    // Output: Linear to scalar Hamiltonian
    results.H = (W3.transpose() * results.a2)(0, 0) + b3_scalar;
    return results;
}

/**
 * @brief Computes the gradient of the Hamiltonian w.r.t the state/input z (dH/dz).
 *
 * Phase 11 GROUP_3: Vectorized cos gradients and Hadamard products using SIMD.
 */
inline VectorXf compute_dH_dz(
    const HNNIntermediate& intermediates,
    const Map<const MatrixXf>& W1,
    const Map<const MatrixXf>& W2,
    const Map<const MatrixXf>& W3) {
    // Backprop through layer 2
    VectorXf dH_da2 = W3.col(0);

    // Compute cos(h2) for gradient (SIMD)
    VectorXf cos_h2 = intermediates.h2;
    simd_cos_inplace(cos_h2.data(), cos_h2.size());

    // Hadamard product: dH_dh2 = dH_da2 .* cos(h2) (SIMD)
    VectorXf dH_dh2(dH_da2.size());
    simd_hadamard_product(dH_da2.data(), cos_h2.data(), dH_dh2.data(), dH_da2.size());

    // Backprop through layer 1
    VectorXf dH_da1 = W2 * dH_dh2;

    // Compute cos(h1) for gradient (SIMD)
    VectorXf cos_h1 = intermediates.h1;
    simd_cos_inplace(cos_h1.data(), cos_h1.size());

    // Hadamard product: dH_dh1 = dH_da1 .* cos(h1) (SIMD)
    VectorXf dH_dh1(dH_da1.size());
    simd_hadamard_product(dH_da1.data(), cos_h1.data(), dH_dh1.data(), dH_da1.size());

    // Final backprop to input z
    VectorXf dH_dz = W1 * dH_dh1;
    return dH_dz;
}

}  // namespace tensorflow (temporary close for Phase 17 includes)

// =============================================================================
// Phase 17: Hamiltonian Enhancement Module Includes
// =============================================================================
// These MUST be included AFTER HNNIntermediate and compute_dH_dz are defined
// because the enhancement modules depend on these types/functions.
// They are included OUTSIDE the tensorflow namespace to avoid double-nesting.
#include "ops/magnus_integrator.h"
#include "ops/sphnn_helpers.h"
#include "ops/lie_poisson_helpers.h"
#include "ops/hamiltonian_superposition.h"

namespace tensorflow {  // Re-open namespace for remaining code

/**
 * @brief Backpropagates the gradient of the Hamiltonian (grad_H) w.r.t the HNN weights.
 *
 * Phase 11 GROUP_3: Vectorized cos gradients and Hadamard products using SIMD.
 */
inline void backprop_dH_dweights(
    const HNNIntermediate& intermediates,
    const Map<const MatrixXf>& W1, const Map<const MatrixXf>& W2, const Map<const MatrixXf>& W3,
    float grad_H,
    MatrixXf& grad_W1, VectorXf& grad_b1,
    MatrixXf& grad_W2, VectorXf& grad_b2,
    MatrixXf& grad_W3, float& grad_b3) {

    // Backprop through layer 2
    VectorXf dL_da2 = grad_H * W3.col(0);

    // Compute cos(h2) for gradient (SIMD)
    VectorXf cos_h2 = intermediates.h2;
    simd_cos_inplace(cos_h2.data(), cos_h2.size());

    // Hadamard product: dL_dh2 = dL_da2 .* cos(h2) (SIMD)
    VectorXf dL_dh2(dL_da2.size());
    simd_hadamard_product(dL_da2.data(), cos_h2.data(), dL_dh2.data(), dL_da2.size());

    // Backprop through layer 1
    VectorXf dL_da1 = W2 * dL_dh2;

    // Compute cos(h1) for gradient (SIMD)
    VectorXf cos_h1 = intermediates.h1;
    simd_cos_inplace(cos_h1.data(), cos_h1.size());

    // Hadamard product: dL_dh1 = dL_da1 .* cos(h1) (SIMD)
    VectorXf dL_dh1(dL_da1.size());
    simd_hadamard_product(dL_da1.data(), cos_h1.data(), dL_dh1.data(), dL_da1.size());

    // Accumulate weight gradients
    grad_W3.col(0) += intermediates.a2 * grad_H;
    grad_b3 += grad_H;
    grad_W2 += intermediates.a1 * dL_dh2.transpose();
    grad_b2 += dL_dh2;
    grad_W1 += intermediates.z * dL_dh1.transpose();
    grad_b1 += dL_dh1;
}


/**
 * @brief Yoshida 4th-order symplectic integrator coefficients.
 *
 * The Yoshida method achieves 4th-order accuracy by composing three 2nd-order
 * leapfrog steps with specific coefficients. These are exact analytical values
 * derived from the composition method:
 *   w1 = 1 / (2 - 2^(1/3))
 *   w0 = -2^(1/3) / (2 - 2^(1/3))
 *
 * This requires 3× the force evaluations of Leapfrog but provides significantly
 * better energy conservation and long-term stability for Hamiltonian dynamics.
 */
namespace YoshidaCoefficients {
    constexpr double cube_root_two = 1.2599210498948731647672106072782;
    constexpr double w1 = 1.0 / (2.0 - cube_root_two);  // ≈ 1.351207191959658
    constexpr double w0 = -cube_root_two / (2.0 - cube_root_two);  // ≈ -1.702414383919315
}

/**
 * @brief Soft-potential regularization parameters (Phase 3.2).
 *
 * Replaces hard saturation caps with smooth 6th-order potential:
 *   V_reg(q) = α Σᵢ (qᵢ / q_scale)⁶
 *   dV/dq = 6α (q / q_scale)⁵ / q_scale
 *
 * This provides gentle penalty for large phase-space excursions while
 * maintaining symplectic structure and differentiability.
 */
struct SoftPotentialRegularization {
    float reg_alpha;   // Regularization strength (default: 1e-4)
    float reg_scale;   // Characteristic scale for position variables (default: 10.0)

    SoftPotentialRegularization(float alpha = 1e-4f, float scale = 10.0f)
        : reg_alpha(alpha), reg_scale(scale) {}

    /**
     * @brief Computes the regularization gradient dV_reg/dq.
     *
     * @param q Position vector.
     * @return Gradient vector of same dimension as q.
     */
    VectorXf compute_gradient(const VectorXf& q) const {
        if (reg_alpha == 0.0f || reg_scale == 0.0f) {
            return VectorXf::Zero(q.size());
        }

        // Compute (q / q_scale)⁵
        VectorXf q_normalized = q / reg_scale;
        VectorXf q_pow5 = q_normalized.array().pow(5.0f);

        // Gradient: 6α (q / q_scale)⁵ / q_scale
        return (6.0f * reg_alpha / reg_scale) * q_pow5;
    }

    /**
     * @brief Computes the regularization potential energy V_reg(q).
     *
     * @param q Position vector.
     * @return Scalar potential energy.
     */
    float compute_potential(const VectorXf& q) const {
        if (reg_alpha == 0.0f || reg_scale == 0.0f) {
            return 0.0f;
        }

        // V_reg = α Σᵢ (qᵢ / q_scale)⁶
        VectorXf q_normalized = q / reg_scale;
        VectorXf q_pow6 = q_normalized.array().pow(6.0f);
        return reg_alpha * q_pow6.sum();
    }
};

/**
 * @brief Performs a single 2nd-order leapfrog sub-step for Yoshida composition.
 *
 * This is the atomic building block for the Yoshida integrator. It performs:
 *   1. p += -(dt/2) * (dH/dq + dV_reg/dq)
 *   2. q += dt * dH/dp
 *   3. p += -(dt/2) * (dH/dq + dV_reg/dq)
 *
 * @param q Position vector (updated in-place).
 * @param p Momentum vector (updated in-place).
 * @param x_t Input vector.
 * @param W1, b1, W2, b2, W3, b3_scalar Hamiltonian network parameters (mapped).
 * @param dt Sub-step time increment.
 * @param state_dim Dimension of position/momentum vectors.
 * @param h_input_dim Total input dimension to Hamiltonian network.
 * @param regularization Optional soft-potential regularization (Phase 3.2).
 */
inline void leapfrog_substep(
    VectorXf& q, VectorXf& p, const VectorXf& x_t,
    const Map<const MatrixXf>& W1, const Map<const VectorXf>& b1,
    const Map<const MatrixXf>& W2, const Map<const VectorXf>& b2,
    const Map<const MatrixXf>& W3, const float b3_scalar,
    const float dt, const int state_dim, const int h_input_dim,
    const SoftPotentialRegularization* regularization = nullptr) {

    VectorXf z(h_input_dim);
    z << q, p, x_t;

    // First half-step for momentum
    HNNIntermediate int1 = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    VectorXf dH_dz1 = compute_dH_dz(int1, W1, W2, W3);
    VectorXf dH_dq1 = dH_dz1.head(state_dim);

    // Augment gradient with soft-potential regularization (Phase 3.2)
    if (regularization != nullptr) {
        dH_dq1 += regularization->compute_gradient(q);
    }

    p -= (dt / 2.0f) * dH_dq1;

    // Full step for position
    z.segment(state_dim, state_dim) = p;
    HNNIntermediate int2 = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    VectorXf dH_dz2 = compute_dH_dz(int2, W1, W2, W3);
    VectorXf dH_dp = dH_dz2.segment(state_dim, state_dim);
    q += dt * dH_dp;

    // Second half-step for momentum
    z.head(state_dim) = q;
    HNNIntermediate int3 = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    VectorXf dH_dz3 = compute_dH_dz(int3, W1, W2, W3);
    VectorXf dH_dq3 = dH_dz3.head(state_dim);

    // Augment gradient with soft-potential regularization (Phase 3.2)
    if (regularization != nullptr) {
        dH_dq3 += regularization->compute_gradient(q);
    }

    p -= (dt / 2.0f) * dH_dq3;
}

/**
 * @brief Performs a single step of the HNN dynamics using the Yoshida integrator.
 *
 * This function takes the current state (q, p) and input (x_t), and computes
 * the next state using a 4th-order Yoshida symplectic integration step.
 * It also calculates the Hamiltonian energy before and after the step for
 * monitoring energy drift.
 *
 * @param q Current position vector (updated in-place).
 * @param p Current momentum vector (updated in-place).
 * @param x_t Input vector for the current time step.
 * @param W1_tensor First weight matrix of the Hamiltonian network.
 * @param b1_tensor First bias vector of the Hamiltonian network.
 * @param W2_tensor Second weight matrix of the Hamiltonian network.
 * @param b2_tensor Second bias vector of the Hamiltonian network.
 * @param W3_tensor Third weight matrix of the Hamiltonian network.
 * @param b3_tensor Third bias scalar of the Hamiltonian network.
 * @param W_out_tensor Output projection weight matrix.
 * @param b_out_tensor Output projection bias vector.
 * @param evolution_time The time step for the integration.
 * @param output_proj The resulting output vector after projection.
 * @param h_initial The calculated Hamiltonian energy at the beginning of the step.
 * @param h_final The calculated Hamiltonian energy at the end of the step.
 * @param reg_alpha Regularization strength (default: 0.0 = disabled).
 * @param reg_scale Regularization scale parameter (default: 10.0).
 */
inline void hnn_core_step(
    VectorXf& q, VectorXf& p, const VectorXf& x_t,
    const Tensor& W1_tensor, const Tensor& b1_tensor,
    const Tensor& W2_tensor, const Tensor& b2_tensor,
    const Tensor& W3_tensor, const Tensor& b3_tensor,
    const Tensor& W_out_tensor, const Tensor& b_out_tensor,
    const float evolution_time,
    VectorXf& output_proj,
    float& h_initial, float& h_final,
    const float reg_alpha = 0.0f, const float reg_scale = 10.0f) {

    const int state_dim = q.size();
    const int input_dim = x_t.size();
    const int h_input_dim = 2 * state_dim + input_dim;
    const int h1_dim = W1_tensor.shape().dim_size(1);
    const int h2_dim = W2_tensor.shape().dim_size(1);

    // Map Tensors to Eigen types for efficient computation
    Map<const MatrixXf> W1(W1_tensor.flat<float>().data(), h_input_dim, h1_dim);
    Map<const VectorXf> b1(b1_tensor.flat<float>().data(), h1_dim);
    Map<const MatrixXf> W2(W2_tensor.flat<float>().data(), h1_dim, h2_dim);
    Map<const VectorXf> b2(b2_tensor.flat<float>().data(), h2_dim);
    Map<const MatrixXf> W3(W3_tensor.flat<float>().data(), h2_dim, 1);
    const float b3_scalar = b3_tensor.scalar<float>()();
    Map<const MatrixXf> W_out(W_out_tensor.flat<float>().data(), 2 * state_dim, input_dim);
    Map<const VectorXf> b_out(b_out_tensor.flat<float>().data(), input_dim);

    // Concatenate q, p, and x_t to form the input to the Hamiltonian network
    VectorXf z(h_input_dim);
    z << q, p, x_t;

    // --- Calculate Initial Hamiltonian ---
    HNNIntermediate intermediates_initial = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    h_initial = intermediates_initial.H;

    // --- Soft-Potential Regularization (Phase 3.2) ---
    // Create regularization object if enabled (reg_alpha > 0)
    SoftPotentialRegularization regularization(reg_alpha, reg_scale);
    const SoftPotentialRegularization* reg_ptr = (reg_alpha > 0.0f) ? &regularization : nullptr;

    // --- Yoshida 4th-Order Symplectic Integration (Phase 3.1 + 3.2) ---
    // Compose three 2nd-order leapfrog steps with Yoshida coefficients for 4th-order accuracy.
    // Soft-potential regularization is applied within each sub-step to augment dH/dq.
    // Cost: 3× force evaluations per step (9 Hamiltonian evaluations total).

    const float dt1 = static_cast<float>(YoshidaCoefficients::w1 * evolution_time);
    const float dt0 = static_cast<float>(YoshidaCoefficients::w0 * evolution_time);

    // First Yoshida sub-step (w1 * dt)
    leapfrog_substep(q, p, x_t, W1, b1, W2, b2, W3, b3_scalar, dt1, state_dim, h_input_dim, reg_ptr);

    // Second Yoshida sub-step (w0 * dt)
    leapfrog_substep(q, p, x_t, W1, b1, W2, b2, W3, b3_scalar, dt0, state_dim, h_input_dim, reg_ptr);

    // Third Yoshida sub-step (w1 * dt)
    leapfrog_substep(q, p, x_t, W1, b1, W2, b2, W3, b3_scalar, dt1, state_dim, h_input_dim, reg_ptr);

    // --- Calculate Final Hamiltonian (including regularization potential) ---
    z.head(state_dim) = q;
    z.segment(state_dim, state_dim) = p;
    HNNIntermediate intermediates_final = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    h_final = intermediates_final.H;

    // Add regularization potential to final energy if enabled
    if (reg_ptr != nullptr) {
        h_final += regularization.compute_potential(q);
    }

    // --- Output Projection ---
    VectorXf final_state_concat(2 * state_dim);
    final_state_concat << q, p;
    output_proj = W_out.transpose() * final_state_concat + b_out;
}

// =============================================================================
// Phase 17: Enhanced HNN Configuration and Dispatch
// =============================================================================

/**
 * @brief Integrator type selection for HNN dynamics.
 */
enum class IntegratorType {
    YOSHIDA_4,    // Yoshida 4th-order symplectic (default)
    MAGNUS_4,     // Magnus 4th-order geometric (for time-dependent H)
    EULER,        // Simple Euler (for debugging)
    MIDPOINT      // Midpoint method (for sPHNN)
};

/**
 * @brief Dynamics type selection (Poisson structure).
 */
enum class DynamicsType {
    CANONICAL,        // Standard canonical Hamiltonian (T*Q)
    LIE_POISSON,      // Lie-Poisson on g* (preserves Casimirs)
    PORT_HAMILTONIAN  // sPHNN with dissipation (Lyapunov stable)
};

/**
 * @brief Configuration for enhanced HNN step.
 */
struct HNNEnhancedConfig {
    IntegratorType integrator = IntegratorType::YOSHIDA_4;
    DynamicsType dynamics = DynamicsType::CANONICAL;
    LieGroupType lie_group = LieGroupType::CANONICAL;  // For LIE_POISSON
    PortHamiltonianConfig sphnn_config;                 // For PORT_HAMILTONIAN
    int magnus_order = 4;                               // For MAGNUS integrator
    float reg_alpha = 0.0f;                             // Soft-potential regularization
    float reg_scale = 10.0f;
    bool enable_superposition = false;                  // Basis Hamiltonian mode
};

/**
 * @brief Result structure for enhanced HNN step.
 */
struct HNNEnhancedResult {
    VectorXf q_next;
    VectorXf p_next;
    VectorXf output_proj;
    float h_initial;
    float h_final;
    float energy_drift;       // |H_final - H_initial|
    float casimir_drift;      // For Lie-Poisson dynamics
    float lyapunov_derivative; // For sPHNN (should be ≤ 0)
    bool is_stable;           // Lyapunov stability verified
};

/**
 * @brief Unified enhanced HNN step with integrator/dynamics dispatch.
 *
 * This function selects the appropriate integrator and dynamics type
 * based on configuration, providing a single entry point for all
 * Hamiltonian layer enhancements.
 *
 * @param q Current position (state_dim)
 * @param p Current momentum (state_dim)
 * @param x_t Input features
 * @param W1_tensor, b1_tensor, ... HNN weights
 * @param W_out_tensor, b_out_tensor Output projection weights
 * @param evolution_time Timestep
 * @param config Enhanced configuration
 * @return HNNEnhancedResult with evolved state and metrics
 */
inline HNNEnhancedResult hnn_enhanced_step(
    VectorXf& q, 
    VectorXf& p, 
    const VectorXf& x_t,
    const Tensor& W1_tensor, const Tensor& b1_tensor,
    const Tensor& W2_tensor, const Tensor& b2_tensor,
    const Tensor& W3_tensor, const Tensor& b3_tensor,
    const Tensor& W_out_tensor, const Tensor& b_out_tensor,
    float evolution_time,
    const HNNEnhancedConfig& config) {
    
    HNNEnhancedResult result;
    result.casimir_drift = 0.0f;
    result.lyapunov_derivative = 0.0f;
    result.is_stable = true;
    
    const int state_dim = q.size();
    const int input_dim = x_t.size();
    const int h_input_dim = 2 * state_dim + input_dim;
    const int h1_dim = W1_tensor.shape().dim_size(1);
    const int h2_dim = W2_tensor.shape().dim_size(1);
    
    // Map tensors to Eigen
    Map<const MatrixXf> W1(W1_tensor.flat<float>().data(), h_input_dim, h1_dim);
    Map<const VectorXf> b1(b1_tensor.flat<float>().data(), h1_dim);
    Map<const MatrixXf> W2(W2_tensor.flat<float>().data(), h1_dim, h2_dim);
    Map<const VectorXf> b2(b2_tensor.flat<float>().data(), h2_dim);
    Map<const MatrixXf> W3(W3_tensor.flat<float>().data(), h2_dim, 1);
    const float b3_scalar = b3_tensor.scalar<float>()();
    Map<const MatrixXf> W_out(W_out_tensor.flat<float>().data(), 2*state_dim, input_dim);
    Map<const VectorXf> b_out(b_out_tensor.flat<float>().data(), input_dim);
    
    // Compute initial Hamiltonian
    VectorXf z(h_input_dim);
    z << q, p, x_t;
    HNNIntermediate int_initial = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    result.h_initial = int_initial.H;
    
    // Dispatch based on dynamics type
    switch (config.dynamics) {
        case DynamicsType::LIE_POISSON: {
            // Lie-Poisson dynamics on g*
            VectorXf dH_dz = compute_dH_dz(int_initial, W1, W2, W3);
            float casimir_initial = compute_casimir(z.head(2 * state_dim), config.lie_group);
            
            // Use Lie-Poisson step
            VectorXf full_state(2 * state_dim);
            full_state << q, p;
            LiePoissonStepResult lp_result = lie_poisson_step(
                full_state, result.h_initial, dH_dz.head(2 * state_dim),
                config.lie_group, evolution_time);
            
            q = lp_result.x_next.head(state_dim);
            p = lp_result.x_next.tail(state_dim);
            result.casimir_drift = lp_result.casimir_drift;
            break;
        }
        
        case DynamicsType::PORT_HAMILTONIAN: {
            // sPHNN with dissipation
            VectorXf dH_dz = compute_dH_dz(int_initial, W1, W2, W3);
            VectorXf dH_dq = dH_dz.head(state_dim);
            VectorXf dH_dp = dH_dz.segment(state_dim, state_dim);
            
            sPHNNStepResult sphnn_result = sphnn_step(
                q, p, dH_dq, dH_dp, result.h_initial,
                config.sphnn_config, evolution_time);
            
            result.lyapunov_derivative = sphnn_result.lyapunov_derivative;
            result.is_stable = sphnn_result.is_stable;
            break;
        }
        
        case DynamicsType::CANONICAL:
        default: {
            // Standard canonical dynamics - dispatch based on integrator
            switch (config.integrator) {
                case IntegratorType::MAGNUS_4: {
                    // Magnus geometric integrator
                    auto compute_H_lambda = [&](const VectorXf& z_in,
                                                const Map<const MatrixXf>& W1_in, 
                                                const Map<const VectorXf>& b1_in,
                                                const Map<const MatrixXf>& W2_in, 
                                                const Map<const VectorXf>& b2_in,
                                                const Map<const MatrixXf>& W3_in, 
                                                float b3_in) {
                        return compute_H_and_intermediates(z_in, W1_in, b1_in, 
                                                          W2_in, b2_in, W3_in, b3_in);
                    };
                    
                    magnus_hnn_step(
                        q, p, x_t,
                        W1, b1, W2, b2, W3, b3_scalar,
                        W_out, b_out,
                        evolution_time,
                        config.magnus_order,
                        result.output_proj,
                        result.h_initial,
                        result.h_final,
                        compute_H_lambda);
                    
                    result.q_next = q;
                    result.p_next = p;
                    result.energy_drift = std::abs(result.h_final - result.h_initial);
                    return result;
                }
                
                case IntegratorType::YOSHIDA_4:
                default: {
                    // Standard Yoshida integrator via hnn_core_step
                    hnn_core_step(
                        q, p, x_t,
                        W1_tensor, b1_tensor,
                        W2_tensor, b2_tensor,
                        W3_tensor, b3_tensor,
                        W_out_tensor, b_out_tensor,
                        evolution_time,
                        result.output_proj,
                        result.h_initial,
                        result.h_final,
                        config.reg_alpha,
                        config.reg_scale);
                    break;
                }
            }
            break;
        }
    }
    
    // Compute final Hamiltonian if not already set
    z.head(state_dim) = q;
    z.segment(state_dim, state_dim) = p;
    HNNIntermediate int_final = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    result.h_final = int_final.H;
    
    // Output projection
    VectorXf final_concat(2 * state_dim);
    final_concat << q, p;
    result.output_proj = W_out.transpose() * final_concat + b_out;
    
    result.q_next = q;
    result.p_next = p;
    result.energy_drift = std::abs(result.h_final - result.h_initial);
    
    return result;
}

/**
 * @brief Parse integrator type from string attribute.
 */
inline IntegratorType parse_integrator_type(const std::string& type_str) {
    if (type_str == "magnus" || type_str == "magnus_4") {
        return IntegratorType::MAGNUS_4;
    } else if (type_str == "euler") {
        return IntegratorType::EULER;
    } else if (type_str == "midpoint") {
        return IntegratorType::MIDPOINT;
    }
    return IntegratorType::YOSHIDA_4;
}

/**
 * @brief Parse dynamics type from string attribute.
 */
inline DynamicsType parse_dynamics_type(const std::string& type_str) {
    if (type_str == "lie_poisson") {
        return DynamicsType::LIE_POISSON;
    } else if (type_str == "port_hamiltonian" || type_str == "sphnn") {
        return DynamicsType::PORT_HAMILTONIAN;
    }
    return DynamicsType::CANONICAL;
}

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_HNN_CORE_HELPERS_H_
