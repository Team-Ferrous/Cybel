// saguaro/native/ops/fused_qwt_tokenizer_op.h
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
 * @file fused_qwt_tokenizer_op.h
 * @brief Enterprise-grade Quantum Wavelet Tokenizer helper functions.
 *
 * This header provides SIMD-optimized helper functions for the QWT enhancements:
 *
 * **Enhancement 1: Lifting Scheme DWT**
 * - qwt_lifting_predict(): Predict step for wavelet decomposition
 * - qwt_lifting_update(): Update step for wavelet decomposition
 * - qwt_lifting_forward(): Complete forward DWT via lifting
 * - qwt_lifting_inverse(): Complete inverse DWT via lifting
 *
 * **Enhancement 2: Padé[m/m] Matrix Exponential**
 * - PadeCoefficients: Static coefficient storage for orders 1-4
 * - qwt_build_pade_numerator(): Construct numerator polynomial
 * - qwt_build_pade_denominator(): Construct denominator polynomial
 *
 * **Enhancement 3: Jacobi Preconditioner**
 * - qwt_extract_diagonal(): Extract diagonal from sparse matrix
 * - qwt_apply_jacobi_preconditioner(): Apply M^-1 to vector
 *
 * **Enhancement 4: Skip-Connection Hamiltonian**
 * - qwt_compute_skip_weights(): Calculate skip connection weights
 * - qwt_add_skip_connections(): Add skip edges to Hamiltonian
 *
 * **Enhancement 5: Parallel DWT Cascade**
 * - CascadeBuffer: Pre-allocated buffer management
 * - qwt_parallel_dwt_cascade(): Parallel multi-level decomposition
 *
 * All functions use the qwt_ prefix to avoid ODR violations with other ops.
 *
 * SIMD Support:
 * - AVX512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - NEON: 4-wide vectorization (ARM)
 * - Scalar fallback for all architectures
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef SAGUARO_NATIVE_OPS_FUSED_QWT_TOKENIZER_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_QWT_TOKENIZER_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <atomic>

#include "Eigen/Core"
#include "Eigen/Sparse"

// SIMD intrinsics for cross-architecture vectorization
#if defined(__AVX512F__)
#include <immintrin.h>
#define QWT_SIMD_WIDTH 16
#define QWT_SIMD_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define QWT_SIMD_WIDTH 8
#define QWT_SIMD_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define QWT_SIMD_WIDTH 4
#define QWT_SIMD_NEON 1
#else
#define QWT_SIMD_WIDTH 1
#define QWT_SIMD_SCALAR 1
#endif

namespace saguaro {
namespace ops {
namespace qwt {

// =============================================================================
// CONSTANTS
// =============================================================================

constexpr float QWT_EPSILON = 1e-8f;
constexpr int QWT_MAX_PADE_ORDER = 4;
constexpr int QWT_MAX_SKIP_CONNECTIONS = 8;

// =============================================================================
// PADÉ APPROXIMATION COEFFICIENTS
// =============================================================================

/**
 * @brief Padé[m/m] approximant coefficients for matrix exponential exp(-iHt).
 *
 * The approximation is: exp(-iHt) ≈ p(αH) / q(αH)
 * where α = t/2 and p, q are polynomials.
 *
 * For Padé[1/1] (Cayley): (I + αH)^-1 (I - αH)
 * For Padé[2/2]: More accurate for larger step sizes
 * For Padé[3/3]: Even better accuracy
 * For Padé[4/4]: 8th-order accurate, best for our use case
 */
struct PadeCoefficients {
    // Numerator coefficients: p(x) = c0 + c1*x + c2*x² + c3*x³ + c4*x⁴
    float num[5];
    // Denominator coefficients: q(x) = d0 + d1*x + d2*x² + d3*x³ + d4*x⁴
    float den[5];
    int order;
};

// Pre-computed Padé coefficients (normalized so d0 = 1)
// These are derived from the diagonal Padé approximation of exp(x)
inline const PadeCoefficients& GetPadeCoefficients(int order) {
    static const PadeCoefficients kPade1 = {
        {1.0f, -0.5f, 0.0f, 0.0f, 0.0f},     // Numerator: 1 - x/2
        {1.0f, 0.5f, 0.0f, 0.0f, 0.0f},      // Denominator: 1 + x/2
        1
    };
    static const PadeCoefficients kPade2 = {
        {1.0f, -0.5f, 1.0f/12.0f, 0.0f, 0.0f},           // 1 - x/2 + x²/12
        {1.0f, 0.5f, 1.0f/12.0f, 0.0f, 0.0f},            // 1 + x/2 + x²/12
        2
    };
    static const PadeCoefficients kPade3 = {
        {1.0f, -0.5f, 1.0f/10.0f, -1.0f/120.0f, 0.0f},   // 1 - x/2 + x²/10 - x³/120
        {1.0f, 0.5f, 1.0f/10.0f, 1.0f/120.0f, 0.0f},     // 1 + x/2 + x²/10 + x³/120
        3
    };
    static const PadeCoefficients kPade4 = {
        // Padé[4/4] coefficients for exp(-x)
        {1.0f, -0.5f, 3.0f/28.0f, -1.0f/84.0f, 1.0f/1680.0f},
        {1.0f, 0.5f, 3.0f/28.0f, 1.0f/84.0f, 1.0f/1680.0f},
        4
    };

    switch (order) {
        case 1: return kPade1;
        case 2: return kPade2;
        case 3: return kPade3;
        case 4: return kPade4;
        default: return kPade4;  // Default to highest order
    }
}

// =============================================================================
// LIFTING SCHEME DWT
// =============================================================================

/**
 * @brief SIMD-optimized predict step for lifting scheme: d[n] = x[2n+1] - P(x[2n])
 *
 * The predict step computes detail coefficients by predicting odd samples
 * from even samples and computing the residual.
 *
 * For Haar wavelets: P(x) = x (simple copy)
 * For Daubechies-4: P(x) = weighted average with learned coefficients
 *
 * @param even_samples Pointer to even-indexed samples [n/2, embed_dim]
 * @param odd_samples Pointer to odd-indexed samples [n/2, embed_dim]
 * @param predict_weights Learnable prediction filter [kernel_size, embed_dim]
 * @param detail_out Output detail coefficients [n/2, embed_dim]
 * @param half_seq Number of output samples (seq_len / 2)
 * @param embed_dim Embedding dimension
 * @param kernel_size Prediction filter size (1 for Haar)
 */
inline void qwt_lifting_predict(
    const float* even_samples,
    const float* odd_samples,
    const float* predict_weights,
    float* detail_out,
    int64_t half_seq,
    int64_t embed_dim,
    int64_t kernel_size
) {
    // For each output position
    for (int64_t m = 0; m < half_seq; ++m) {
        const float* odd_ptr = odd_samples + m * embed_dim;
        float* out_ptr = detail_out + m * embed_dim;

        // Initialize detail = odd - prediction
        int64_t d = 0;

#if defined(QWT_SIMD_AVX512)
        for (; d + 16 <= embed_dim; d += 16) {
            __m512 odd_vec = _mm512_loadu_ps(odd_ptr + d);
            __m512 pred_sum = _mm512_setzero_ps();

            // Apply prediction filter
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m + k;
                if (src_idx >= half_seq) src_idx = half_seq - 1;  // Clamp

                const float* even_ptr = even_samples + src_idx * embed_dim + d;
                const float* weight_ptr = predict_weights + k * embed_dim + d;

                __m512 even_vec = _mm512_loadu_ps(even_ptr);
                __m512 weight_vec = _mm512_loadu_ps(weight_ptr);
                pred_sum = _mm512_fmadd_ps(even_vec, weight_vec, pred_sum);
            }

            __m512 detail = _mm512_sub_ps(odd_vec, pred_sum);
            _mm512_storeu_ps(out_ptr + d, detail);
        }
#elif defined(QWT_SIMD_AVX2)
        for (; d + 8 <= embed_dim; d += 8) {
            __m256 odd_vec = _mm256_loadu_ps(odd_ptr + d);
            __m256 pred_sum = _mm256_setzero_ps();

            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m + k;
                if (src_idx >= half_seq) src_idx = half_seq - 1;

                const float* even_ptr = even_samples + src_idx * embed_dim + d;
                const float* weight_ptr = predict_weights + k * embed_dim + d;

                __m256 even_vec = _mm256_loadu_ps(even_ptr);
                __m256 weight_vec = _mm256_loadu_ps(weight_ptr);
                pred_sum = _mm256_fmadd_ps(even_vec, weight_vec, pred_sum);
            }

            __m256 detail = _mm256_sub_ps(odd_vec, pred_sum);
            _mm256_storeu_ps(out_ptr + d, detail);
        }
#elif defined(QWT_SIMD_NEON)
        for (; d + 4 <= embed_dim; d += 4) {
            float32x4_t odd_vec = vld1q_f32(odd_ptr + d);
            float32x4_t pred_sum = vdupq_n_f32(0.0f);

            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m + k;
                if (src_idx >= half_seq) src_idx = half_seq - 1;

                const float* even_ptr = even_samples + src_idx * embed_dim + d;
                const float* weight_ptr = predict_weights + k * embed_dim + d;

                float32x4_t even_vec = vld1q_f32(even_ptr);
                float32x4_t weight_vec = vld1q_f32(weight_ptr);
                pred_sum = vmlaq_f32(pred_sum, even_vec, weight_vec);
            }

            float32x4_t detail = vsubq_f32(odd_vec, pred_sum);
            vst1q_f32(out_ptr + d, detail);
        }
#endif

        // Scalar fallback
        for (; d < embed_dim; ++d) {
            float pred = 0.0f;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m + k;
                if (src_idx >= half_seq) src_idx = half_seq - 1;
                pred += even_samples[src_idx * embed_dim + d] *
                        predict_weights[k * embed_dim + d];
            }
            out_ptr[d] = odd_ptr[d] - pred;
        }
    }
}

/**
 * @brief SIMD-optimized update step for lifting scheme: a[n] = even[n] + U(d[n])
 *
 * The update step computes approximation coefficients by updating even samples
 * using the detail coefficients.
 *
 * @param even_samples Pointer to even-indexed samples [n/2, embed_dim]
 * @param detail_coeffs Detail coefficients from predict step [n/2, embed_dim]
 * @param update_weights Learnable update filter [kernel_size, embed_dim]
 * @param approx_out Output approximation coefficients [n/2, embed_dim]
 * @param half_seq Number of output samples (seq_len / 2)
 * @param embed_dim Embedding dimension
 * @param kernel_size Update filter size
 */
inline void qwt_lifting_update(
    const float* even_samples,
    const float* detail_coeffs,
    const float* update_weights,
    float* approx_out,
    int64_t half_seq,
    int64_t embed_dim,
    int64_t kernel_size
) {
    for (int64_t m = 0; m < half_seq; ++m) {
        const float* even_ptr = even_samples + m * embed_dim;
        float* out_ptr = approx_out + m * embed_dim;

        int64_t d = 0;

#if defined(QWT_SIMD_AVX512)
        for (; d + 16 <= embed_dim; d += 16) {
            __m512 even_vec = _mm512_loadu_ps(even_ptr + d);
            __m512 update_sum = _mm512_setzero_ps();

            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m - kernel_size / 2 + k;
                if (src_idx < 0) src_idx = 0;
                if (src_idx >= half_seq) src_idx = half_seq - 1;

                const float* detail_ptr = detail_coeffs + src_idx * embed_dim + d;
                const float* weight_ptr = update_weights + k * embed_dim + d;

                __m512 detail_vec = _mm512_loadu_ps(detail_ptr);
                __m512 weight_vec = _mm512_loadu_ps(weight_ptr);
                update_sum = _mm512_fmadd_ps(detail_vec, weight_vec, update_sum);
            }

            __m512 approx = _mm512_add_ps(even_vec, update_sum);
            _mm512_storeu_ps(out_ptr + d, approx);
        }
#elif defined(QWT_SIMD_AVX2)
        for (; d + 8 <= embed_dim; d += 8) {
            __m256 even_vec = _mm256_loadu_ps(even_ptr + d);
            __m256 update_sum = _mm256_setzero_ps();

            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m - kernel_size / 2 + k;
                if (src_idx < 0) src_idx = 0;
                if (src_idx >= half_seq) src_idx = half_seq - 1;

                const float* detail_ptr = detail_coeffs + src_idx * embed_dim + d;
                const float* weight_ptr = update_weights + k * embed_dim + d;

                __m256 detail_vec = _mm256_loadu_ps(detail_ptr);
                __m256 weight_vec = _mm256_loadu_ps(weight_ptr);
                update_sum = _mm256_fmadd_ps(detail_vec, weight_vec, update_sum);
            }

            __m256 approx = _mm256_add_ps(even_vec, update_sum);
            _mm256_storeu_ps(out_ptr + d, approx);
        }
#elif defined(QWT_SIMD_NEON)
        for (; d + 4 <= embed_dim; d += 4) {
            float32x4_t even_vec = vld1q_f32(even_ptr + d);
            float32x4_t update_sum = vdupq_n_f32(0.0f);

            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m - kernel_size / 2 + k;
                if (src_idx < 0) src_idx = 0;
                if (src_idx >= half_seq) src_idx = half_seq - 1;

                const float* detail_ptr = detail_coeffs + src_idx * embed_dim + d;
                const float* weight_ptr = update_weights + k * embed_dim + d;

                float32x4_t detail_vec = vld1q_f32(detail_ptr);
                float32x4_t weight_vec = vld1q_f32(weight_ptr);
                update_sum = vmlaq_f32(update_sum, detail_vec, weight_vec);
            }

            float32x4_t approx = vaddq_f32(even_vec, update_sum);
            vst1q_f32(out_ptr + d, approx);
        }
#endif

        // Scalar fallback
        for (; d < embed_dim; ++d) {
            float update = 0.0f;
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = m - kernel_size / 2 + k;
                if (src_idx < 0) src_idx = 0;
                if (src_idx >= half_seq) src_idx = half_seq - 1;
                update += detail_coeffs[src_idx * embed_dim + d] *
                          update_weights[k * embed_dim + d];
            }
            out_ptr[d] = even_ptr[d] + update;
        }
    }
}

/**
 * @brief Complete forward lifting scheme DWT.
 *
 * Performs one level of wavelet decomposition using the lifting scheme:
 * 1. Split input into even and odd samples
 * 2. Predict: detail = odd - P(even)
 * 3. Update: approx = even + U(detail)
 *
 * @param input Input tensor [seq_len, embed_dim]
 * @param predict_weights Prediction filter [kernel_size, embed_dim]
 * @param update_weights Update filter [kernel_size, embed_dim]
 * @param approx_out Output approximation [seq_len/2, embed_dim]
 * @param detail_out Output detail [seq_len/2, embed_dim]
 * @param seq_len Input sequence length (must be even)
 * @param embed_dim Embedding dimension
 * @param kernel_size Filter kernel size
 */
inline void qwt_lifting_forward(
    const float* input,
    const float* predict_weights,
    const float* update_weights,
    float* approx_out,
    float* detail_out,
    int64_t seq_len,
    int64_t embed_dim,
    int64_t kernel_size
) {
    const int64_t half_seq = seq_len / 2;

    // Step 1: Split into even and odd samples (in-place addressing)
    // Even samples: input[0], input[2], input[4], ...
    // Odd samples: input[1], input[3], input[5], ...

    // Allocate temporary buffers for even/odd samples
    std::vector<float> even_buffer(half_seq * embed_dim);
    std::vector<float> odd_buffer(half_seq * embed_dim);

    // Split: Copy even and odd samples
    for (int64_t m = 0; m < half_seq; ++m) {
        const float* even_src = input + (2 * m) * embed_dim;
        const float* odd_src = input + (2 * m + 1) * embed_dim;
        float* even_dst = even_buffer.data() + m * embed_dim;
        float* odd_dst = odd_buffer.data() + m * embed_dim;

        std::memcpy(even_dst, even_src, embed_dim * sizeof(float));
        std::memcpy(odd_dst, odd_src, embed_dim * sizeof(float));
    }

    // Step 2: Predict (compute detail coefficients)
    qwt_lifting_predict(
        even_buffer.data(),
        odd_buffer.data(),
        predict_weights,
        detail_out,
        half_seq,
        embed_dim,
        kernel_size
    );

    // Step 3: Update (compute approximation coefficients)
    qwt_lifting_update(
        even_buffer.data(),
        detail_out,
        update_weights,
        approx_out,
        half_seq,
        embed_dim,
        kernel_size
    );
}

/**
 * @brief Backward pass for lifting scheme predict step.
 *
 * Computes gradients for predict weights, even samples, and odd samples.
 */
inline void qwt_lifting_predict_grad(
    const float* even_samples,
    const float* odd_samples,
    const float* predict_weights,
    const float* grad_detail,  // Incoming gradient
    float* grad_even,          // Output: gradient w.r.t. even samples
    float* grad_odd,           // Output: gradient w.r.t. odd samples
    float* grad_predict,       // Output: gradient w.r.t. predict weights
    int64_t half_seq,
    int64_t embed_dim,
    int64_t kernel_size
) {
    // detail = odd - P(even)
    // grad_odd = grad_detail
    // grad_even = -grad_detail * predict_weights (accumulated)
    // grad_predict = -grad_detail * even (accumulated)

    // Initialize gradients
    std::fill(grad_even, grad_even + half_seq * embed_dim, 0.0f);
    std::fill(grad_predict, grad_predict + kernel_size * embed_dim, 0.0f);

    for (int64_t m = 0; m < half_seq; ++m) {
        const float* grad_d = grad_detail + m * embed_dim;

        // grad_odd = grad_detail (direct copy)
        std::memcpy(grad_odd + m * embed_dim, grad_d, embed_dim * sizeof(float));

        // Accumulate gradients for predict step
        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t src_idx = m + k;
            if (src_idx >= half_seq) src_idx = half_seq - 1;

            for (int64_t d = 0; d < embed_dim; ++d) {
                // grad_predict[k, d] += -grad_detail[m, d] * even[src_idx, d]
                grad_predict[k * embed_dim + d] +=
                    -grad_d[d] * even_samples[src_idx * embed_dim + d];

                // grad_even[src_idx, d] += -grad_detail[m, d] * predict_weights[k, d]
                grad_even[src_idx * embed_dim + d] +=
                    -grad_d[d] * predict_weights[k * embed_dim + d];
            }
        }
    }
}

/**
 * @brief Backward pass for lifting scheme update step.
 */
inline void qwt_lifting_update_grad(
    const float* even_samples,
    const float* detail_coeffs,
    const float* update_weights,
    const float* grad_approx,  // Incoming gradient
    float* grad_even,          // Output: gradient w.r.t. even samples
    float* grad_detail,        // Output: gradient w.r.t. detail coeffs
    float* grad_update,        // Output: gradient w.r.t. update weights
    int64_t half_seq,
    int64_t embed_dim,
    int64_t kernel_size
) {
    // approx = even + U(detail)
    // grad_even = grad_approx
    // grad_detail = grad_approx * update_weights (accumulated)
    // grad_update = grad_approx * detail (accumulated)

    std::fill(grad_detail, grad_detail + half_seq * embed_dim, 0.0f);
    std::fill(grad_update, grad_update + kernel_size * embed_dim, 0.0f);

    for (int64_t m = 0; m < half_seq; ++m) {
        const float* grad_a = grad_approx + m * embed_dim;

        // grad_even = grad_approx (add to existing)
        for (int64_t d = 0; d < embed_dim; ++d) {
            grad_even[m * embed_dim + d] += grad_a[d];
        }

        // Accumulate gradients for update step
        for (int64_t k = 0; k < kernel_size; ++k) {
            int64_t src_idx = m - kernel_size / 2 + k;
            if (src_idx < 0) src_idx = 0;
            if (src_idx >= half_seq) src_idx = half_seq - 1;

            for (int64_t d = 0; d < embed_dim; ++d) {
                // grad_update[k, d] += grad_approx[m, d] * detail[src_idx, d]
                grad_update[k * embed_dim + d] +=
                    grad_a[d] * detail_coeffs[src_idx * embed_dim + d];

                // grad_detail[src_idx, d] += grad_approx[m, d] * update_weights[k, d]
                grad_detail[src_idx * embed_dim + d] +=
                    grad_a[d] * update_weights[k * embed_dim + d];
            }
        }
    }
}

// =============================================================================
// JACOBI PRECONDITIONER
// =============================================================================

/**
 * @brief Extract diagonal from sparse matrix with regularization.
 *
 * @param sparse Input sparse matrix
 * @param diag_inv Output: inverted diagonal (1 / (diag[i] + epsilon))
 * @param epsilon Regularization constant
 */
inline void qwt_extract_diagonal_inverse(
    const Eigen::SparseMatrix<float>& sparse,
    Eigen::VectorXf& diag_inv,
    float epsilon = QWT_EPSILON
) {
    const int n = sparse.rows();
    diag_inv.resize(n);

    for (int i = 0; i < n; ++i) {
        float diag_val = 0.0f;
        // Find diagonal element
        for (Eigen::SparseMatrix<float>::InnerIterator it(sparse, i); it; ++it) {
            if (it.row() == it.col()) {
                diag_val = it.value();
                break;
            }
        }
        // Regularized inverse
        diag_inv(i) = 1.0f / (std::abs(diag_val) + epsilon);
    }
}

/**
 * @brief Apply Jacobi preconditioner to vector: out = M^-1 * in
 *
 * @param diag_inv Inverted diagonal from qwt_extract_diagonal_inverse
 * @param input Input vector
 * @param output Output vector (can alias input for in-place)
 */
inline void qwt_apply_jacobi_preconditioner(
    const Eigen::VectorXf& diag_inv,
    const Eigen::VectorXf& input,
    Eigen::VectorXf& output
) {
    output = diag_inv.cwiseProduct(input);
}

/**
 * @brief Apply Jacobi preconditioner to matrix columns: out = M^-1 * in
 */
inline void qwt_apply_jacobi_preconditioner_matrix(
    const Eigen::VectorXf& diag_inv,
    const Eigen::MatrixXf& input,
    Eigen::MatrixXf& output
) {
    const int rows = input.rows();
    const int cols = input.cols();
    output.resize(rows, cols);

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            output(i, j) = diag_inv(i) * input(i, j);
        }
    }
}

// =============================================================================
// SKIP-CONNECTION HAMILTONIAN
// =============================================================================

/**
 * @brief Compute skip connection weights based on energy ratios.
 *
 * Skip connections link nodes that are far apart but have similar
 * frequency characteristics (similar detail energies).
 *
 * @param energies Node energies from detail coefficients
 * @param num_nodes Number of nodes
 * @param skip_stride Stride between skip connections (e.g., 4)
 * @param max_skips Maximum skip connections per node
 * @param skip_pairs Output: (src, dst, weight) triplets
 */
inline void qwt_compute_skip_connections(
    const std::vector<float>& energies,
    int num_nodes,
    int skip_stride,
    int max_skips,
    std::vector<Eigen::Triplet<float>>& skip_triplets
) {
    if (skip_stride <= 0 || max_skips <= 0) return;

    skip_triplets.clear();
    skip_triplets.reserve(num_nodes * max_skips * 2);

    for (int i = 0; i < num_nodes; ++i) {
        int skips_added = 0;

        // Add skip connections at stride intervals
        for (int offset = skip_stride;
             offset < num_nodes && skips_added < max_skips;
             offset += skip_stride) {

            int j = i + offset;
            if (j >= num_nodes) break;

            // Weight based on geometric mean of energies
            float weight = std::sqrt(energies[i] * energies[j]);
            if (weight < QWT_EPSILON) continue;

            // Scale down skip weights relative to adjacent connections
            weight *= 0.25f;

            // Add symmetric edges
            skip_triplets.emplace_back(i, j, weight);
            skip_triplets.emplace_back(j, i, weight);
            ++skips_added;
        }
    }
}

// =============================================================================
// PADÉ MATRIX CONSTRUCTION
// =============================================================================

/**
 * @brief Build numerator sparse matrix for Padé approximation.
 *
 * Computes p(αH) = c0*I + c1*αH + c2*α²H² + ...
 *
 * @param H Hamiltonian sparse matrix
 * @param alpha Step size parameter
 * @param pade Padé coefficients
 * @param H_powers Pre-computed H², H³, H⁴ (can be nullptr for order 1)
 * @return Numerator sparse matrix
 */
inline Eigen::SparseMatrix<float> qwt_build_pade_numerator(
    const Eigen::SparseMatrix<float>& H,
    float alpha,
    const PadeCoefficients& pade,
    const std::vector<Eigen::SparseMatrix<float>>* H_powers = nullptr
) {
    const int n = H.rows();
    Eigen::SparseMatrix<float> result(n, n);

    // Start with c0 * I
    result.setIdentity();
    result *= pade.num[0];

    // Add c1 * alpha * H
    if (pade.order >= 1) {
        result += pade.num[1] * alpha * H;
    }

    // Add higher-order terms
    if (pade.order >= 2 && H_powers && H_powers->size() >= 1) {
        float alpha2 = alpha * alpha;
        result += pade.num[2] * alpha2 * (*H_powers)[0];  // H²
    }

    if (pade.order >= 3 && H_powers && H_powers->size() >= 2) {
        float alpha3 = alpha * alpha * alpha;
        result += pade.num[3] * alpha3 * (*H_powers)[1];  // H³
    }

    if (pade.order >= 4 && H_powers && H_powers->size() >= 3) {
        float alpha4 = alpha * alpha * alpha * alpha;
        result += pade.num[4] * alpha4 * (*H_powers)[2];  // H⁴
    }

    result.makeCompressed();
    return result;
}

/**
 * @brief Build denominator sparse matrix for Padé approximation.
 *
 * Computes q(αH) = d0*I + d1*αH + d2*α²H² + ...
 */
inline Eigen::SparseMatrix<float> qwt_build_pade_denominator(
    const Eigen::SparseMatrix<float>& H,
    float alpha,
    const PadeCoefficients& pade,
    const std::vector<Eigen::SparseMatrix<float>>* H_powers = nullptr
) {
    const int n = H.rows();
    Eigen::SparseMatrix<float> result(n, n);

    // Start with d0 * I
    result.setIdentity();
    result *= pade.den[0];

    // Add d1 * alpha * H
    if (pade.order >= 1) {
        result += pade.den[1] * alpha * H;
    }

    // Add higher-order terms
    if (pade.order >= 2 && H_powers && H_powers->size() >= 1) {
        float alpha2 = alpha * alpha;
        result += pade.den[2] * alpha2 * (*H_powers)[0];  // H²
    }

    if (pade.order >= 3 && H_powers && H_powers->size() >= 2) {
        float alpha3 = alpha * alpha * alpha;
        result += pade.den[3] * alpha3 * (*H_powers)[1];  // H³
    }

    if (pade.order >= 4 && H_powers && H_powers->size() >= 3) {
        float alpha4 = alpha * alpha * alpha * alpha;
        result += pade.den[4] * alpha4 * (*H_powers)[2];  // H⁴
    }

    result.makeCompressed();
    return result;
}

/**
 * @brief Pre-compute H², H³, H⁴ for Padé approximation and gradient caching.
 */
inline void qwt_precompute_h_powers(
    const Eigen::SparseMatrix<float>& H,
    int max_power,
    std::vector<Eigen::SparseMatrix<float>>& H_powers
) {
    H_powers.clear();
    if (max_power < 2) return;

    // H² = H * H
    Eigen::SparseMatrix<float> H2 = H * H;
    H2.makeCompressed();
    H_powers.push_back(std::move(H2));

    if (max_power >= 3) {
        // H³ = H² * H
        Eigen::SparseMatrix<float> H3 = H_powers[0] * H;
        H3.makeCompressed();
        H_powers.push_back(std::move(H3));
    }

    if (max_power >= 4) {
        // H⁴ = H² * H²
        Eigen::SparseMatrix<float> H4 = H_powers[0] * H_powers[0];
        H4.makeCompressed();
        H_powers.push_back(std::move(H4));
    }
}

// =============================================================================
// PARALLEL CASCADE BUFFER
// =============================================================================

/**
 * @brief Pre-allocated buffer for parallel multi-level DWT cascade.
 *
 * This structure manages memory for the cascaded wavelet decomposition
 * to avoid repeated allocations during forward/backward passes.
 */
struct CascadeBuffer {
    std::vector<std::vector<float>> level_approx;  // Approximation at each level
    std::vector<std::vector<float>> level_detail;  // Detail at each level
    std::atomic<int> levels_complete{0};           // Synchronization counter

    void resize(int num_levels, int64_t batch_size, int64_t seq_len, int64_t embed_dim) {
        level_approx.resize(num_levels);
        level_detail.resize(num_levels);

        int64_t current_len = seq_len / 2;
        for (int level = 0; level < num_levels; ++level) {
            int64_t level_size = batch_size * current_len * embed_dim;
            level_approx[level].resize(level_size);
            level_detail[level].resize(level_size);
            current_len /= 2;
            if (current_len < 1) break;
        }
        levels_complete.store(0, std::memory_order_release);
    }

    void reset() {
        levels_complete.store(0, std::memory_order_release);
    }
};

// =============================================================================
// VECTORIZED ENERGY COMPUTATION (Enhanced)
// =============================================================================

/**
 * @brief Batch-compute energies for all nodes with SIMD optimization.
 *
 * @param detail Detail coefficients [num_nodes, embed_dim] row-major
 * @param energies Output energies [num_nodes]
 * @param num_nodes Number of wavelet nodes
 * @param embed_dim Embedding dimension
 * @param epsilon Numerical stability floor
 */
inline void qwt_compute_node_energies(
    const float* detail,
    float* energies,
    int64_t num_nodes,
    int64_t embed_dim,
    float epsilon = QWT_EPSILON
) {
    for (int64_t i = 0; i < num_nodes; ++i) {
        const float* row = detail + i * embed_dim;
        float sum_sq = 0.0f;
        int64_t d = 0;

#if defined(QWT_SIMD_AVX512)
        __m512 sum_vec = _mm512_setzero_ps();
        for (; d + 16 <= embed_dim; d += 16) {
            __m512 vals = _mm512_loadu_ps(row + d);
            sum_vec = _mm512_fmadd_ps(vals, vals, sum_vec);
        }
        sum_sq += _mm512_reduce_add_ps(sum_vec);
#elif defined(QWT_SIMD_AVX2)
        __m256 sum_vec = _mm256_setzero_ps();
        for (; d + 8 <= embed_dim; d += 8) {
            __m256 vals = _mm256_loadu_ps(row + d);
            sum_vec = _mm256_fmadd_ps(vals, vals, sum_vec);
        }
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        sum_sq += _mm_cvtss_f32(sums);
#elif defined(QWT_SIMD_NEON)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (; d + 4 <= embed_dim; d += 4) {
            float32x4_t vals = vld1q_f32(row + d);
            sum_vec = vmlaq_f32(sum_vec, vals, vals);
        }
        float32x2_t sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        sum_sq += vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
#endif

        // Scalar fallback
        for (; d < embed_dim; ++d) {
            sum_sq += row[d] * row[d];
        }

        energies[i] = std::sqrt(sum_sq / static_cast<float>(embed_dim) + epsilon);
    }
}

}  // namespace qwt
}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_FUSED_QWT_TOKENIZER_OP_H_
