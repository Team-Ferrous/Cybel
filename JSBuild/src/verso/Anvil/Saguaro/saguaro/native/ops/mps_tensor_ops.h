// saguaro.native/ops/mps_tensor_ops.h
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
 * @file mps_tensor_ops.h
 * @brief UQHA v3.1 Phase 4: Randomized SVD for MPS Tensor Networks.
 *
 * Implements randomized truncated SVD for Matrix Product State operations.
 * Key optimization: Reduces SVD complexity from O(χ³) to O(χ² log χ).
 *
 * Algorithm (Halko, Martinsson, Tropp 2011):
 * 1. Random projection: Y = A @ Ω where Ω is Gaussian [n, k+p]
 * 2. QR decomposition: Q, _ = QR(Y)
 * 3. Project to low-dim: B = Q^T @ A
 * 4. Full SVD on small B: Ũ, S, V^T = SVD(B)
 * 5. Recover U: U = Q @ Ũ
 *
 * Complexity: O(mn(k+p) + (k+p)³) vs O(mn min(m,n)) for full SVD
 * For MPS with χ bonds: O(χ² log χ) vs O(χ³)
 *
 * References:
 * - "Finding Structure with Randomness" (Halko et al., SIAM Review 2011)
 * - "Randomized algorithms for low-rank matrix approximations" (Liberty et al., 2007)
 */

#ifndef SAGUARO_NATIVE_OPS_MPS_TENSOR_OPS_H_
#define SAGUARO_NATIVE_OPS_MPS_TENSOR_OPS_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace saguaro {
namespace ops {
namespace mps {

/**
 * @brief Configuration for Randomized SVD.
 */
struct RandomizedSVDConfig {
    int target_rank = 64;        // Target rank k
    int oversampling = 10;       // Oversampling parameter p
    int power_iterations = 2;    // Power iterations for accuracy q
    float threshold = 1e-6f;     // Truncation threshold
    uint32_t seed = 42;          // Random seed for reproducibility
};

// =============================================================================
// HELPER: LCG Random Number Generator (fast, deterministic)
// =============================================================================

/**
 * @brief Fast Linear Congruential Generator for Gaussian random numbers.
 */
class FastLCG {
public:
    explicit FastLCG(uint32_t seed) : state_(seed) {}

    float uniform() {
        state_ = state_ * 1664525u + 1013904223u;
        return static_cast<float>(state_) / static_cast<float>(UINT32_MAX);
    }

    float gaussian() {
        // Box-Muller transform
        float u1 = uniform();
        float u2 = uniform();
        if (u1 < 1e-10f) u1 = 1e-10f;
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    }

private:
    uint32_t state_;
};

// =============================================================================
// HELPER: Matrix Operations (no external dependencies)
// =============================================================================

/**
 * @brief Matrix multiplication: C = A @ B
 */
inline void matmul(
    const float* A, const float* B, float* C,
    int m, int k, int n,
    bool transpose_a = false,
    bool transpose_b = false
) {
    // Zero output
    std::memset(C, 0, m * n * sizeof(float));

    if (!transpose_a && !transpose_b) {
        // C[i,j] = sum_l A[i,l] * B[l,j]
        for (int i = 0; i < m; ++i) {
            for (int l = 0; l < k; ++l) {
                float a_il = A[i * k + l];
                for (int j = 0; j < n; ++j) {
                    C[i * n + j] += a_il * B[l * n + j];
                }
            }
        }
    } else if (transpose_a && !transpose_b) {
        // C[i,j] = sum_l A[l,i] * B[l,j]  (A is k×m, transposed to m×k)
        for (int l = 0; l < k; ++l) {
            for (int i = 0; i < m; ++i) {
                float a_li = A[l * m + i];
                for (int j = 0; j < n; ++j) {
                    C[i * n + j] += a_li * B[l * n + j];
                }
            }
        }
    } else if (!transpose_a && transpose_b) {
        // C[i,j] = sum_l A[i,l] * B[j,l]  (B is n×k, transposed to k×n)
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += A[i * k + l] * B[j * k + l];
                }
                C[i * n + j] = sum;
            }
        }
    } else {
        // C[i,j] = sum_l A[l,i] * B[j,l]
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += A[l * m + i] * B[j * k + l];
                }
                C[i * n + j] = sum;
            }
        }
    }
}

/**
 * @brief Modified Gram-Schmidt QR decomposition.
 *
 * @param A Input matrix [m, n] (modified in-place to become Q)
 * @param R Output upper triangular [n, n]
 * @param m Number of rows
 * @param n Number of columns (must have n <= m)
 */
inline void qr_decomposition(float* A, float* R, int m, int n) {
    std::memset(R, 0, n * n * sizeof(float));

    for (int j = 0; j < n; ++j) {
        // Compute norm of column j
        float norm = 0.0f;
        for (int i = 0; i < m; ++i) {
            float val = A[i * n + j];
            norm += val * val;
        }
        norm = std::sqrt(norm);
        R[j * n + j] = norm;

        if (norm < 1e-10f) {
            // Column is zero; fill with random orthogonal vector
            for (int i = 0; i < m; ++i) {
                A[i * n + j] = (i == j % m) ? 1.0f : 0.0f;
            }
            continue;
        }

        // Normalize column j
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < m; ++i) {
            A[i * n + j] *= inv_norm;
        }

        // Orthogonalize subsequent columns against column j
        for (int k = j + 1; k < n; ++k) {
            float dot = 0.0f;
            for (int i = 0; i < m; ++i) {
                dot += A[i * n + j] * A[i * n + k];
            }
            R[j * n + k] = dot;

            for (int i = 0; i < m; ++i) {
                A[i * n + k] -= dot * A[i * n + j];
            }
        }
    }
}

/**
 * @brief Full SVD using power iteration (for small matrices).
 *
 * Computes: A = U @ diag(S) @ V^T
 * This is used on the reduced matrix B after randomized projection.
 *
 * @param A Input matrix [m, n] (destroyed)
 * @param U Output left singular vectors [m, min(m,n)]
 * @param S Output singular values [min(m,n)]
 * @param Vt Output right singular vectors transposed [min(m,n), n]
 * @param m Rows
 * @param n Cols
 */
inline void small_svd(
    float* A,
    float* U, float* S, float* Vt,
    int m, int n
) {
    int k = std::min(m, n);
    int max_iters = 100;
    float tol = 1e-8f;

    // Initialize U to identity-like
    std::memset(U, 0, m * k * sizeof(float));
    for (int i = 0; i < k && i < m; ++i) {
        U[i * k + i] = 1.0f;
    }

    std::vector<float> AtA(n * n);
    std::vector<float> V(n * k);
    std::vector<float> w(n);

    // Compute A^T @ A
    matmul(A, A, AtA.data(), n, m, n, true, false);

    // Power iteration for each singular vector
    for (int j = 0; j < k; ++j) {
        // Initialize v_j randomly
        for (int i = 0; i < n; ++i) {
            V[i * k + j] = (i == j) ? 1.0f : 0.0f;
        }

        // Power iteration
        for (int iter = 0; iter < max_iters; ++iter) {
            // w = AtA @ v_j
            for (int i = 0; i < n; ++i) {
                w[i] = 0.0f;
                for (int l = 0; l < n; ++l) {
                    w[i] += AtA[i * n + l] * V[l * k + j];
                }
            }

            // Orthogonalize against previous vectors
            for (int prev = 0; prev < j; ++prev) {
                float dot = 0.0f;
                for (int i = 0; i < n; ++i) {
                    dot += w[i] * V[i * k + prev];
                }
                for (int i = 0; i < n; ++i) {
                    w[i] -= dot * V[i * k + prev];
                }
            }

            // Normalize
            float norm = 0.0f;
            for (int i = 0; i < n; ++i) {
                norm += w[i] * w[i];
            }
            norm = std::sqrt(norm);

            if (norm < 1e-12f) break;

            float inv_norm = 1.0f / norm;
            for (int i = 0; i < n; ++i) {
                V[i * k + j] = w[i] * inv_norm;
            }
        }

        // Compute singular value: σ_j = ||A @ v_j||
        std::vector<float> Av(m);
        for (int i = 0; i < m; ++i) {
            Av[i] = 0.0f;
            for (int l = 0; l < n; ++l) {
                Av[i] += A[i * n + l] * V[l * k + j];
            }
        }

        float sigma = 0.0f;
        for (int i = 0; i < m; ++i) {
            sigma += Av[i] * Av[i];
        }
        sigma = std::sqrt(sigma);
        S[j] = sigma;

        // Compute u_j = A @ v_j / σ_j
        if (sigma > 1e-12f) {
            float inv_sigma = 1.0f / sigma;
            for (int i = 0; i < m; ++i) {
                U[i * k + j] = Av[i] * inv_sigma;
            }
        }
    }

    // Copy V to Vt (transposed)
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            Vt[i * n + j] = V[j * k + i];
        }
    }
}

// =============================================================================
// MAIN API: Randomized SVD
// =============================================================================

/**
 * @brief Randomized truncated SVD for large matrices.
 *
 * Computes approximate rank-k SVD: A ≈ U @ diag(S) @ V^T
 *
 * This is the main API for MPS tensor network gradient computation.
 * Reduces complexity from O(mn min(m,n)) to O(mn(k+p) + (k+p)³).
 *
 * @param A Input matrix [m, n] (not modified)
 * @param U Output left singular vectors [m, k]
 * @param S Output singular values [k]
 * @param Vt Output right singular vectors transposed [k, n]
 * @param m Number of rows
 * @param n Number of columns
 * @param config Randomized SVD configuration
 */
inline void RandomizedSVD(
    const float* A,
    float* U, float* S, float* Vt,
    int m, int n,
    const RandomizedSVDConfig& config
) {
    int k = config.target_rank;
    int p = config.oversampling;
    int l = std::min(k + p, std::min(m, n));

    FastLCG rng(config.seed);

    // Step 1: Create random Gaussian matrix Ω [n, l]
    std::vector<float> Omega(n * l);
    for (int i = 0; i < n * l; ++i) {
        Omega[i] = rng.gaussian();
    }

    // Step 2: Form Y = A @ Ω  [m, l]
    std::vector<float> Y(m * l);
    matmul(A, Omega.data(), Y.data(), m, n, l);

    // Step 3: Power iteration for improved accuracy (optional)
    std::vector<float> Z(n * l);
    for (int q = 0; q < config.power_iterations; ++q) {
        // Z = A^T @ Y
        matmul(A, Y.data(), Z.data(), n, m, l, true, false);
        // Y = A @ Z
        matmul(A, Z.data(), Y.data(), m, n, l);
    }

    // Step 4: QR decomposition: Q, R = QR(Y)
    // Q is stored in Y after this call
    std::vector<float> R(l * l);
    qr_decomposition(Y.data(), R.data(), m, l);
    float* Q = Y.data();  // Q is now [m, l]

    // Step 5: Form B = Q^T @ A  [l, n]
    std::vector<float> B(l * n);
    matmul(Q, A, B.data(), l, m, n, true, false);

    // Step 6: SVD of small matrix B: Ũ, S, Ṽ^T = SVD(B)
    std::vector<float> U_tilde(l * l);
    std::vector<float> S_full(l);
    std::vector<float> Vt_tilde(l * n);

    small_svd(B.data(), U_tilde.data(), S_full.data(), Vt_tilde.data(), l, n);

    // Step 7: Truncate to rank k
    int actual_k = std::min(k, l);
    for (int i = 0; i < actual_k; ++i) {
        S[i] = S_full[i];
    }

    // Step 8: U = Q @ Ũ[:, :k]  [m, k]
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < actual_k; ++j) {
            float sum = 0.0f;
            for (int ll = 0; ll < l; ++ll) {
                sum += Q[i * l + ll] * U_tilde[ll * l + j];
            }
            U[i * actual_k + j] = sum;
        }
    }

    // Step 9: Vt = Ṽt[:k, :]  [k, n]
    for (int i = 0; i < actual_k; ++i) {
        for (int j = 0; j < n; ++j) {
            Vt[i * n + j] = Vt_tilde[i * n + j];
        }
    }
}

/**
 * @brief Truncated SVD using randomized algorithm with automatic rank selection.
 *
 * Selects rank based on singular value threshold.
 *
 * @param A Input matrix [m, n]
 * @param U Output left singular vectors [m, k_out]
 * @param S Output singular values [k_out]
 * @param Vt Output right singular vectors [k_out, n]
 * @param m Rows
 * @param n Cols
 * @param max_rank Maximum rank to keep
 * @param threshold Relative threshold for singular value truncation
 * @param[out] k_out Actual rank selected
 */
inline void TruncatedRandomizedSVD(
    const float* A,
    float* U, float* S, float* Vt,
    int m, int n,
    int max_rank,
    float threshold,
    int& k_out
) {
    RandomizedSVDConfig config;
    config.target_rank = max_rank;
    config.oversampling = 10;
    config.threshold = threshold;

    // Compute randomized SVD
    RandomizedSVD(A, U, S, Vt, m, n, config);

    // Determine actual rank based on threshold
    float s_max = S[0];
    k_out = 0;
    for (int i = 0; i < max_rank; ++i) {
        if (S[i] / s_max >= threshold) {
            k_out = i + 1;
        }
    }
    if (k_out == 0) k_out = 1;  // Keep at least one singular value
}

/**
 * @brief MPS bond truncation using randomized SVD.
 *
 * Given a contracted two-site tensor Θ[χL, d, d, χR], reshape to 
 * [χL*d, d*χR] and apply truncated SVD to reduce bond dimension.
 *
 * @param Theta Contracted tensor [chi_L * d * d * chi_R]
 * @param chi_L Left bond dimension
 * @param d Physical dimension
 * @param chi_R Right bond dimension
 * @param max_chi Maximum output bond dimension
 * @param threshold Truncation threshold
 * @param A_out Left core output [chi_L * d * chi_new]
 * @param B_out Right core output [chi_new * d * chi_R]
 * @param[out] chi_new New bond dimension
 */
inline void MPSBondTruncation(
    const float* Theta,
    int chi_L, int d, int chi_R,
    int max_chi, float threshold,
    float* A_out, float* B_out,
    int& chi_new
) {
    int m = chi_L * d;
    int n = d * chi_R;

    // Allocate workspace
    std::vector<float> U(m * max_chi);
    std::vector<float> S(max_chi);
    std::vector<float> Vt(max_chi * n);

    // Compute truncated SVD
    TruncatedRandomizedSVD(Theta, U.data(), S.data(), Vt.data(),
                           m, n, max_chi, threshold, chi_new);

    // A_out = U @ sqrt(S) reshaped to [chi_L, d, chi_new]
    // B_out = sqrt(S) @ Vt reshaped to [chi_new, d, chi_R]
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < chi_new; ++j) {
            A_out[i * chi_new + j] = U[i * max_chi + j] * std::sqrt(S[j]);
        }
    }

    for (int i = 0; i < chi_new; ++i) {
        float sqrt_s = std::sqrt(S[i]);
        for (int j = 0; j < n; ++j) {
            B_out[i * n + j] = sqrt_s * Vt[i * n + j];
        }
    }
}

}  // namespace mps
}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_MPS_TENSOR_OPS_H_
