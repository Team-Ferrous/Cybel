// saguaro.native/controllers/tensor_network_kalman.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Tensor Network Kalman Filter with O(n * r²) complexity via TT decomposition.
// Enables 10-100x memory reduction for high-dimensional state spaces.
//
// Reference: "Tensor Network Kalman Filter" (IEEE 2023)
// Phase 1.1 of QUANTUM_CONTROL_ENHANCEMENT_ROADMAP.md

#ifndef SAGUARO_CONTROLLERS_TENSOR_NETWORK_KALMAN_H_
#define SAGUARO_CONTROLLERS_TENSOR_NETWORK_KALMAN_H_

#include <Eigen/Dense>
#include <Eigen/QR>
#include <vector>
#include <memory>
#include "spdlog/spdlog.h"

namespace saguaro {
namespace controllers {

/**
 * @struct TTCore
 * @brief Tensor-Train core for low-rank matrix representation.
 *
 * A TT-decomposition represents a matrix as a sequence of 3D tensors:
 *   M ≈ G_1 × G_2 × ... × G_d
 *
 * Each core has shape [r_{i-1}, n_i, r_i] where:
 *   - r_i are the TT ranks (determine compression level)
 *   - n_i are the mode dimensions
 */
struct TTCore {
    Eigen::MatrixXf data;   ///< Flattened core data [r_left * n_i, r_right]
    int rank_left;          ///< Left TT rank
    int rank_right;         ///< Right TT rank
    int mode_dim;           ///< Mode dimension (n_i)

    /**
     * @brief Get element at position (left_idx, mode_idx, right_idx).
     */
    float get(int left_idx, int mode_idx, int right_idx) const {
        return data(left_idx * mode_dim + mode_idx, right_idx);
    }

    /**
     * @brief Set element at position.
     */
    void set(int left_idx, int mode_idx, int right_idx, float value) {
        data(left_idx * mode_dim + mode_idx, right_idx) = value;
    }

    /**
     * @brief Get slice for fixed mode index.
     */
    Eigen::MatrixXf getSlice(int mode_idx) const {
        return data.block(mode_idx * rank_left, 0, rank_left, rank_right);
    }
};

/**
 * @class TensorNetworkKalmanFilter
 * @brief Kalman filter with TT-compressed covariance for O(n*r²) complexity.
 *
 * Standard Kalman filters have O(n²) memory for the covariance matrix and
 * O(n³) complexity for the matrix operations. For high-dimensional states
 * (n > 64), this becomes prohibitive.
 *
 * The Tensor Network Kalman Filter uses TT decomposition to:
 * - Reduce covariance storage from O(n²) to O(n*r²)
 * - Speed up matrix-vector products from O(n²) to O(n*r²)
 * - Maintain numerical stability via square-root formulation
 *
 * Key features:
 * - TT-matrix-vector product for efficient state propagation
 * - Square-root TNKF for numerical stability
 * - Adaptive TT rank based on approximation error
 * - Fallback to dense operations for small state dims
 *
 * Example:
 *   TensorNetworkKalmanFilter tnkf;
 *   tnkf.init(A, B, C, D, Q, R, 8);  // max TT rank = 8
 *   tnkf.predict(u);
 *   tnkf.update(y);
 *   auto state = tnkf.getState();
 */
class TensorNetworkKalmanFilter {
public:
    using MatrixXf = Eigen::MatrixXf;
    using VectorXf = Eigen::VectorXf;

    TensorNetworkKalmanFilter();
    ~TensorNetworkKalmanFilter() = default;

    /**
     * @brief Initialize with state-space model.
     *
     * Matrices A and P are TT-decomposed for efficient operations.
     * B, C, D, Q, R are kept in dense form (typically smaller).
     *
     * @param A State transition matrix (will be TT-decomposed)
     * @param B Control input matrix
     * @param C Observation matrix
     * @param D Feedthrough matrix
     * @param Q Process noise covariance
     * @param R Measurement noise covariance
     * @param max_rank Maximum TT rank for compression (typical: 4-16)
     */
    void init(const MatrixXf& A,
              const MatrixXf& B,
              const MatrixXf& C,
              const MatrixXf& D,
              const MatrixXf& Q,
              const MatrixXf& R,
              int max_rank = 8);

    /**
     * @brief Update A matrix (e.g., from RLS system identification).
     *
     * Re-decomposes A into TT format.
     *
     * @param A New state transition matrix
     */
    void updateA(const MatrixXf& A);

    /**
     * @brief Predict step with TT-accelerated operations.
     *
     * Uses TT-matrix-vector product for O(n*r²) complexity.
     *
     * @param u Control input vector
     */
    void predict(const std::vector<float>& u);

    /**
     * @brief Update step with measurement.
     *
     * Uses square-root covariance update for stability.
     *
     * @param y Measurement vector
     */
    void update(const std::vector<float>& y);

    /**
     * @brief Get current state estimate.
     */
    std::vector<float> getState() const;

    /**
     * @brief Get covariance diagonal (fast, O(n) extraction).
     *
     * Returns diagonal of P without full reconstruction.
     */
    std::vector<float> getCovarianceDiagonal() const;

    /**
     * @brief Get full covariance matrix (expensive, O(n²)).
     *
     * Reconstructs dense P from TT format. Use sparingly.
     */
    MatrixXf getFullCovariance() const;

    /**
     * @brief Get current TT ranks of covariance.
     */
    std::vector<int> getTTRanks() const;

    /**
     * @brief Get memory usage in bytes.
     */
    size_t getMemoryUsage() const;

    /**
     * @brief Reset filter state.
     */
    void reset();

    /**
     * @brief Set initial state.
     */
    void setInitialState(const VectorXf& x0);

    /**
     * @brief Check if TT mode is active (vs dense fallback).
     */
    bool isTTModeActive() const { return use_tt_mode_; }

private:
    // TT-decomposed matrices
    std::vector<TTCore> A_tt_;   ///< State transition in TT format
    std::vector<TTCore> P_tt_;   ///< Covariance in TT format

    // Dense matrices (typically small)
    MatrixXf B_, C_, D_, Q_, R_;

    // State estimate
    VectorXf x_hat_;

    // Configuration
    int state_dim_;
    int input_dim_;
    int output_dim_;
    int max_rank_;
    float rounding_tolerance_ = 1e-6f;

    // Mode flags
    bool is_initialized_ = false;
    bool use_tt_mode_ = true;
    static constexpr int kMinDimForTT = 16;  // Use dense below this

    // Dense fallback (for small state dims)
    MatrixXf A_dense_;
    MatrixXf P_dense_;

    /**
     * @brief TT-matrix-vector product: y = A_tt * x.
     *
     * O(n * r²) complexity vs O(n²) for dense.
     */
    VectorXf ttMatvec(const std::vector<TTCore>& tt_matrix,
                      const VectorXf& vec);

    /**
     * @brief TT rounding with controlled rank.
     *
     * Truncates TT ranks while maintaining approximation error.
     */
    void ttRound(std::vector<TTCore>& tt_matrix, int target_rank);

    /**
     * @brief Square-root TT rounding for stability.
     *
     * Uses QR-based orthogonalization for better numerics.
     */
    void ttRoundSqrt(std::vector<TTCore>& tt_matrix, int target_rank);

    /**
     * @brief Decompose dense matrix to TT format.
     *
     * Uses SVD-based decomposition with rank truncation.
     */
    std::vector<TTCore> denseToTT(const MatrixXf& matrix, int max_rank);

    /**
     * @brief Reconstruct dense matrix from TT format.
     */
    MatrixXf ttToDense(const std::vector<TTCore>& tt_matrix);

    /**
     * @brief Extract diagonal from TT-format matrix.
     *
     * O(n * r²) complexity vs O(n²) for dense extraction.
     */
    VectorXf extractDiagonal(const std::vector<TTCore>& tt_matrix);

    /**
     * @brief TT-format covariance update for predict step.
     *
     * Computes P = A * P * A^T + Q in TT format.
     */
    void updateCovarianceTT();

    /**
     * @brief Dense covariance update (fallback).
     */
    void updateCovarianceDense(const VectorXf& u);

    std::shared_ptr<spdlog::logger> logger_;
};

}  // namespace controllers
}  // namespace saguaro

#endif  // SAGUARO_CONTROLLERS_TENSOR_NETWORK_KALMAN_H_
