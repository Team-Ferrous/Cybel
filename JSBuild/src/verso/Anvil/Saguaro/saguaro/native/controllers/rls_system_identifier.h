// saguaro.native/controllers/rls_system_identifier.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Recursive Least Squares System Identifier for fast online system identification.
// Replaces N4SID (850-batch) with O(n²) per-batch updates for 170x faster adaptation.
//
// Reference: dadep88/RLSFilter (header-only Eigen RLS)
// Phase 2.1 of QUANTUM_CONTROL_ENHANCEMENT_ROADMAP.md

#ifndef SAGUARO_CONTROLLERS_RLS_SYSTEM_IDENTIFIER_H_
#define SAGUARO_CONTROLLERS_RLS_SYSTEM_IDENTIFIER_H_

#include <Eigen/Dense>
#include <deque>
#include <tuple>
#include <memory>
#include <cmath>
#include "spdlog/spdlog.h"

namespace saguaro {
namespace controllers {

/**
 * @class RLSSystemIdentifier
 * @brief Recursive Least Squares System Identifier for online state-space estimation.
 *
 * This class implements a recursive least squares algorithm for continuous
 * system identification, replacing the expensive N4SID batch processing.
 * Key advantages:
 *   - O(n²) per update vs O(n³) for N4SID
 *   - Updates every batch vs every 850 batches
 *   - Forgetting factor for tracking time-varying systems
 *
 * The algorithm estimates an ARX model which is then converted to state-space form.
 *
 * @tparam T Scalar type (float or double)
 *
 * Example:
 *   RLSSystemIdentifier<float> rls;
 *   rls.configure(4, 2, 3, 0.998f);  // 4th order, 2 inputs, 3 outputs
 *   rls.update(y_measurement, u_control);
 *   auto [A, B, C, D] = rls.getSystemMatrices();
 */
template<typename T = float>
class RLSSystemIdentifier {
public:
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    /**
     * @brief Default constructor.
     */
    RLSSystemIdentifier();

    /**
     * @brief Destructor.
     */
    ~RLSSystemIdentifier() = default;

    /**
     * @brief Configure identifier parameters.
     *
     * @param state_order Estimated system order (number of past samples to use)
     * @param num_inputs Number of control inputs
     * @param num_outputs Number of measured outputs
     * @param forgetting_factor Lambda (0.95-0.999, typical: 0.998)
     *        Lower values adapt faster but are noisier.
     *        Higher values are more stable but adapt slower.
     */
    void configure(int state_order, int num_inputs, int num_outputs,
                   T forgetting_factor = static_cast<T>(0.998));

    /**
     * @brief Recursive update with new input/output pair.
     *
     * Performs O(n²) RLS update, much faster than N4SID's O(n³) SVD.
     *
     * @param y_new Current output measurement vector
     * @param u_new Current control input vector
     */
    void update(const VectorT& y_new, const VectorT& u_new);

    /**
     * @brief Get current system matrices estimate.
     *
     * Converts the internal ARX representation to state-space form:
     *   x[k+1] = A*x[k] + B*u[k]
     *   y[k]   = C*x[k] + D*u[k]
     *
     * @return Tuple of (A, B, C, D) matrices
     */
    std::tuple<MatrixT, MatrixT, MatrixT, MatrixT> getSystemMatrices() const;

    /**
     * @brief Check if model has converged (sufficient data collected).
     *
     * @return true if enough samples have been collected for reliable estimation
     */
    bool isConverged() const;

    /**
     * @brief Reset state for full re-identification.
     *
     * Call this after a full N4SID reset to re-seed the RLS from accurate baseline.
     */
    void reset();

    /**
     * @brief Get current parameter estimation error covariance trace.
     *
     * Useful for monitoring convergence - lower trace means more confidence.
     *
     * @return Trace of the covariance matrix
     */
    T getCovarianceTrace() const;

    /**
     * @brief Set initial parameter estimates from N4SID result.
     *
     * Seeds the RLS with known-good parameters from a full N4SID run.
     *
     * @param A State transition matrix
     * @param B Control input matrix
     * @param C Observation matrix
     * @param D Feedthrough matrix
     */
    void seedFromStateSpace(const MatrixT& A, const MatrixT& B,
                            const MatrixT& C, const MatrixT& D);

private:
    // Model parameters (ARX representation: y[k] = sum(a_i * y[k-i]) + sum(b_j * u[k-j]))
    MatrixT theta_;           ///< Parameter matrix [num_outputs x regressor_dim]
    MatrixT P_cov_;           ///< Covariance matrix (inverse of Hessian)

    // Configuration
    int state_order_;         ///< System order (number of lags)
    int num_inputs_;          ///< Number of control inputs
    int num_outputs_;         ///< Number of measured outputs
    T forgetting_factor_;     ///< Lambda for exponential forgetting (0.95-0.999)
    int regressor_dim_;       ///< Dimension of regressor vector

    // Data history for regressor construction
    std::deque<VectorT> y_history_;  ///< Past output measurements
    std::deque<VectorT> u_history_;  ///< Past control inputs

    int update_count_ = 0;
    int min_samples_for_convergence_;

    /**
     * @brief Build regressor vector from history.
     *
     * Constructs phi = [y[k-1], y[k-2], ..., y[k-n], u[k-1], u[k-2], ..., u[k-n]]
     *
     * @return Regressor vector
     */
    VectorT buildRegressor() const;

    /**
     * @brief Convert ARX parameters to state-space (A, B, C, D).
     *
     * Uses observable canonical form for the conversion.
     *
     * @param A Output state transition matrix
     * @param B Output control input matrix
     * @param C Output observation matrix
     * @param D Output feedthrough matrix
     */
    void arxToStateSpace(MatrixT& A, MatrixT& B, MatrixT& C, MatrixT& D) const;

    std::shared_ptr<spdlog::logger> logger_;
};

// Explicit instantiation declarations
extern template class RLSSystemIdentifier<float>;
extern template class RLSSystemIdentifier<double>;

}  // namespace controllers
}  // namespace saguaro

#endif  // SAGUARO_CONTROLLERS_RLS_SYSTEM_IDENTIFIER_H_
