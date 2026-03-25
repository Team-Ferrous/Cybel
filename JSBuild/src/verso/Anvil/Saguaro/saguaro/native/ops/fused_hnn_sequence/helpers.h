// src/ops/fused_hnn_sequence/helpers.h
// Copyright 2025 Verso Industries

#ifndef TENSORFLOW_CORE_USER_OPS_FUSED_HNN_SEQUENCE_HELPERS_H_
#define TENSORFLOW_CORE_USER_OPS_FUSED_HNN_SEQUENCE_HELPERS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <cmath>
#include <algorithm> // For std::min

// SIMD intrinsics for cross-architecture vectorization
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
 * Shared with fused_hnn_step_op for numerical consistency.
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
 * Shared with fused_hnn_step_op for numerical consistency.
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
 * This MUST be fully defined in the header for visibility across translation units.
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


// -----------------------------------------------------------------
// Core HNN Logic Implementations (Inlined for full visibility)
// -----------------------------------------------------------------

/**
 * @brief Computes the Hamiltonian H and all intermediate hidden layers.
 *
 * Phase 11 GROUP_3: Vectorized sin activations using SIMD.
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
 * @param D_state Dimension of position/momentum vectors.
 * @param D_in Total input dimension to Hamiltonian network.
 * @param regularization Optional soft-potential regularization (Phase 3.2).
 */
inline void leapfrog_substep(
    VectorXf& q, VectorXf& p, const VectorXf& x_t,
    const Map<const MatrixXf>& W1, const Map<const VectorXf>& b1,
    const Map<const MatrixXf>& W2, const Map<const VectorXf>& b2,
    const Map<const MatrixXf>& W3, const float b3_scalar,
    const float dt, const int D_state, const int D_in,
    const SoftPotentialRegularization* regularization = nullptr) {

    VectorXf z(D_in);
    z << q, p, x_t;

    // First half-step for momentum
    HNNIntermediate int1 = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    VectorXf dH_dz1 = compute_dH_dz(int1, W1, W2, W3);
    VectorXf dH_dq1 = dH_dz1.head(D_state);

    // Augment gradient with soft-potential regularization (Phase 3.2)
    if (regularization != nullptr) {
        dH_dq1 += regularization->compute_gradient(q);
    }

    p -= (dt / 2.0f) * dH_dq1;

    // Full step for position
    z.segment(D_state, D_state) = p;
    HNNIntermediate int2 = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    VectorXf dH_dz2 = compute_dH_dz(int2, W1, W2, W3);
    VectorXf dH_dp = dH_dz2.segment(D_state, D_state);
    q += dt * dH_dp;

    // Second half-step for momentum
    z.head(D_state) = q;
    HNNIntermediate int3 = compute_H_and_intermediates(z, W1, b1, W2, b2, W3, b3_scalar);
    VectorXf dH_dz3 = compute_dH_dz(int3, W1, W2, W3);
    VectorXf dH_dq3 = dH_dz3.head(D_state);

    // Augment gradient with soft-potential regularization (Phase 3.2)
    if (regularization != nullptr) {
        dH_dq3 += regularization->compute_gradient(q);
    }

    p -= (dt / 2.0f) * dH_dq3;
}

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_FUSED_HNN_SEQUENCE_HELPERS_H_
