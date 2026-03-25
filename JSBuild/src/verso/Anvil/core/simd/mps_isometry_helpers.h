// highnoon/_native/ops/mps_isometry_helpers.h
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
 * @file mps_isometry_helpers.h
 * @brief Phase 3: CA-TDVP Isometric MPS Helpers
 *
 * Implements isometry enforcement and TDVP gradient projection for
 * Matrix Product States (MPS) to enable O(depth) memory training.
 *
 * Key Functions:
 *   - enforce_left_canonical: QR-based isometry enforcement
 *   - tdvp_project_gradient: Tangent space projection
 *   - compute_isometry_error: Validation metric
 *
 * Reference: "Time-Dependent Variational Principle for Quantum Lattices" (TDVP)
 */

#ifndef HIGHNOON_NATIVE_OPS_MPS_ISOMETRY_HELPERS_H_
#define HIGHNOON_NATIVE_OPS_MPS_ISOMETRY_HELPERS_H_

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace hsmn {
namespace mps {

// =============================================================================
// ISOMETRY ENFORCEMENT VIA QR DECOMPOSITION
// =============================================================================

/**
 * @brief Enforce left-canonical (isometric) form on MPS core.
 * 
 * For a core A of shape [chi_L, d, chi_R], the left-canonical form requires:
 *   sum_i A[:, i, :]^H @ A[:, i, :] = I_{chi_R}
 * 
 * This is enforced by reshaping to [chi_L * d, chi_R] and applying QR.
 * 
 * @param core MPS core tensor [chi_L * d * chi_R] in row-major
 * @param chi_L Left bond dimension
 * @param d Physical dimension
 * @param chi_R Right bond dimension
 */
inline void EnforceLeftCanonical(float* core, int chi_L, int d, int chi_R) {
    const int rows = chi_L * d;
    const int cols = chi_R;
    
    if (rows < cols) {
        // Core is too wide for standard left-canonicalization
        // This case requires padding or alternative normalization
        return;
    }
    
    // Map to Eigen matrix
    Eigen::Map<Eigen::MatrixXf> A(core, rows, cols);
    
    // Householder QR decomposition
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(A);
    
    // Extract Q (thin) - the isometric part
    Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(rows, cols);
    
    // Copy back to core
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            core[i * cols + j] = Q(i, j);
        }
    }
}

/**
 * @brief Enforce right-canonical form on MPS core.
 * 
 * For right-canonical: sum_i A[:, i, :] @ A[:, i, :]^H = I_{chi_L}
 * 
 * @param core MPS core tensor [chi_L * d * chi_R]
 * @param chi_L Left bond dimension
 * @param d Physical dimension
 * @param chi_R Right bond dimension
 */
inline void EnforceRightCanonical(float* core, int chi_L, int d, int chi_R) {
    const int rows = chi_L;
    const int cols = d * chi_R;
    
    if (cols < rows) {
        return;
    }
    
    // Transpose, apply QR, transpose back
    Eigen::Map<Eigen::MatrixXf> A(core, chi_L, d * chi_R);
    Eigen::MatrixXf AT = A.transpose();
    
    Eigen::HouseholderQR<Eigen::MatrixXf> qr(AT);
    Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(cols, rows);
    
    // Transpose Q back
    Eigen::MatrixXf QT = Q.transpose();
    for (int i = 0; i < chi_L; ++i) {
        for (int j = 0; j < d * chi_R; ++j) {
            core[i * cols + j] = QT(i, j);
        }
    }
}

/**
 * @brief Compute isometry error: ||A^H A - I||_F
 * 
 * @param core MPS core [chi_L * d * chi_R]
 * @param chi_L Left bond dimension
 * @param d Physical dimension
 * @param chi_R Right bond dimension
 * @return Frobenius norm of isometry violation
 */
inline float ComputeIsometryError(const float* core, int chi_L, int d, int chi_R) {
    const int rows = chi_L * d;
    const int cols = chi_R;
    
    Eigen::Map<const Eigen::MatrixXf> A(core, rows, cols);
    Eigen::MatrixXf AtA = A.transpose() * A;
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(cols, cols);
    
    return (AtA - I).norm();
}

// =============================================================================
// TDVP GRADIENT PROJECTION
// =============================================================================

/**
 * @brief Project gradient onto MPS tangent space (TDVP).
 * 
 * The tangent space at an isometric MPS core is the space of allowed
 * variations that preserve the isometry constraint. For a left-canonical
 * core A, the tangent vector X must satisfy:
 *   A^H X + X^H A = 0  (skew-Hermitian condition for real matrices)
 * 
 * The projection is: X_proj = X - A @ (A^H @ X)
 * 
 * @param grad Gradient tensor [chi_L * d * chi_R]
 * @param core Current isometric core [chi_L * d * chi_R]
 * @param projected Output projected gradient [chi_L * d * chi_R]
 * @param chi_L Left bond dimension
 * @param d Physical dimension
 * @param chi_R Right bond dimension
 */
inline void TDVPProjectGradient(
    const float* grad, const float* core, float* projected,
    int chi_L, int d, int chi_R) {
    
    const int rows = chi_L * d;
    const int cols = chi_R;
    
    Eigen::Map<const Eigen::MatrixXf> G(grad, rows, cols);
    Eigen::Map<const Eigen::MatrixXf> A(core, rows, cols);
    Eigen::Map<Eigen::MatrixXf> P(projected, rows, cols);
    
    // Project: P = G - A @ (A^T @ G)
    // This ensures: A^T @ P = A^T @ G - (A^T @ A) @ (A^T @ G) = A^T @ G - A^T @ G = 0
    // (assuming A is isometric: A^T @ A = I)
    Eigen::MatrixXf AtG = A.transpose() * G;
    P = G - A * AtG;
}

/**
 * @brief Compute and store both left and right environments for TDVP.
 * 
 * The left environment L[t] contracts cores 0..t-1.
 * The right environment R[t] contracts cores t+1..L-1.
 * 
 * These are needed for efficient TDVP gradient computation.
 * 
 * @param cores MPS cores [L, chi_L, d, chi_R]
 * @param left_envs Output left environments [L+1, chi, chi]
 * @param right_envs Output right environments [L+1, chi, chi]
 * @param seq_len Number of sites
 * @param chi Bond dimension
 * @param d Physical dimension
 */
inline void ComputeTDVPEnvironments(
    const float* cores,
    float* left_envs,
    float* right_envs,
    int seq_len, int chi, int d) {
    
    const int core_size = chi * d * chi;
    const int env_size = chi * chi;
    
    // Initialize left boundary: L[0] = I
    Eigen::Map<Eigen::MatrixXf> L0(left_envs, chi, chi);
    L0.setIdentity();
    
    // Forward pass: compute L[t+1] = L[t] contracted with A[t]
    for (int t = 0; t < seq_len; ++t) {
        Eigen::Map<const Eigen::MatrixXf> L_t(left_envs + t * env_size, chi, chi);
        Eigen::Map<Eigen::MatrixXf> L_next(left_envs + (t + 1) * env_size, chi, chi);
        
        // Sum over physical dimension: L_next = sum_p L_t @ A_t[:, p, :]
        L_next.setZero();
        for (int p = 0; p < d; ++p) {
            // A_t[:, p, :] is at offset (p * chi) in the reshaped core
            // But cores are [chi_L, d, chi_R], so A[:, p, :] = core[chi_L * p : chi_L * (p+1), :]
            Eigen::Map<const Eigen::MatrixXf> A_p(
                cores + t * core_size + p * chi, chi, chi);
            L_next += L_t * A_p;
        }
        L_next /= d;  // Normalize
    }
    
    // Initialize right boundary: R[L] = I
    Eigen::Map<Eigen::MatrixXf> RL(right_envs + seq_len * env_size, chi, chi);
    RL.setIdentity();
    
    // Backward pass: compute R[t] = A[t] contracted with R[t+1]
    for (int t = seq_len - 1; t >= 0; --t) {
        Eigen::Map<const Eigen::MatrixXf> R_next(right_envs + (t + 1) * env_size, chi, chi);
        Eigen::Map<Eigen::MatrixXf> R_t(right_envs + t * env_size, chi, chi);
        
        R_t.setZero();
        for (int p = 0; p < d; ++p) {
            Eigen::Map<const Eigen::MatrixXf> A_p(
                cores + t * core_size + p * chi, chi, chi);
            R_t += A_p * R_next;
        }
        R_t /= d;
    }
}

/**
 * @brief Apply TDVP-projected gradient update to MPS core.
 * 
 * Combines gradient projection with isometry re-enforcement.
 * 
 * @param core MPS core to update [chi_L * d * chi_R]
 * @param grad Raw gradient [chi_L * d * chi_R]
 * @param learning_rate Step size
 * @param chi_L Left bond dimension
 * @param d Physical dimension
 * @param chi_R Right bond dimension
 */
inline void TDVPGradientStep(
    float* core, const float* grad, float learning_rate,
    int chi_L, int d, int chi_R) {
    
    const int size = chi_L * d * chi_R;
    std::vector<float> projected(size);
    
    // Project gradient onto tangent space
    TDVPProjectGradient(grad, core, projected.data(), chi_L, d, chi_R);
    
    // Apply gradient step
    for (int i = 0; i < size; ++i) {
        core[i] -= learning_rate * projected[i];
    }
    
    // Re-enforce isometry (retraction to manifold)
    EnforceLeftCanonical(core, chi_L, d, chi_R);
}

// =============================================================================
// SIMD-ACCELERATED HELPERS
// =============================================================================

/**
 * @brief Vectorized Frobenius norm squared.
 */
inline float VectorizedNormSquared(const float* data, int64_t size) {
    float norm_sq = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    norm_sq = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    norm_sq = _mm_cvtss_f32(sum);
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        acc = vmlaq_f32(acc, v, v);
    }
    float32x2_t sum = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum = vpadd_f32(sum, sum);
    norm_sq = vget_lane_f32(sum, 0);
#endif

    for (; i < size; ++i) {
        norm_sq += data[i] * data[i];
    }
    return norm_sq;
}

}  // namespace mps
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_MPS_ISOMETRY_HELPERS_H_
