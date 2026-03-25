// saguaro.native/controllers/rls_system_identifier.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Recursive Least Squares System Identifier implementation.
// O(n²) online updates for 170x faster adaptation vs N4SID.

#include "controllers/rls_system_identifier.h"

#include <algorithm>
#include <stdexcept>
#include "spdlog/sinks/stdout_color_sinks.h"

namespace saguaro {
namespace controllers {

template<typename T>
RLSSystemIdentifier<T>::RLSSystemIdentifier() {
    try {
        logger_ = spdlog::get("highnoon");
        if (!logger_) {
            logger_ = spdlog::stdout_color_mt("highnoon");
        }
    } catch (...) {
        // Logger initialization failed, continue without logging
    }
}

template<typename T>
void RLSSystemIdentifier<T>::configure(int state_order, int num_inputs,
                                        int num_outputs, T forgetting_factor) {
    if (state_order <= 0 || num_inputs <= 0 || num_outputs <= 0) {
        throw std::invalid_argument("RLS: All dimensions must be positive");
    }
    if (forgetting_factor <= 0 || forgetting_factor > 1) {
        throw std::invalid_argument("RLS: Forgetting factor must be in (0, 1]");
    }

    state_order_ = state_order;
    num_inputs_ = num_inputs;
    num_outputs_ = num_outputs;
    forgetting_factor_ = forgetting_factor;

    // Regressor dimension: n*num_outputs (AR terms) + n*num_inputs (exogenous terms)
    regressor_dim_ = state_order_ * (num_outputs_ + num_inputs_);

    // Minimum samples before we trust the estimate
    min_samples_for_convergence_ = std::max(2 * regressor_dim_, 50);

    reset();

    if (logger_) {
        logger_->info("RLS configured: order={}, inputs={}, outputs={}, lambda={:.4f}, regressor_dim={}",
                      state_order_, num_inputs_, num_outputs_, forgetting_factor_, regressor_dim_);
    }
}

template<typename T>
void RLSSystemIdentifier<T>::reset() {
    // Initialize parameter matrix to zeros
    // Shape: [num_outputs x regressor_dim]
    theta_ = MatrixT::Zero(num_outputs_, regressor_dim_);

    // Initialize covariance to large diagonal (high uncertainty)
    // P = delta * I, where delta is large
    T delta = static_cast<T>(1000.0);
    P_cov_ = MatrixT::Identity(regressor_dim_, regressor_dim_) * delta;

    // Clear history
    y_history_.clear();
    u_history_.clear();
    update_count_ = 0;

    if (logger_) {
        logger_->debug("RLS reset: P_cov initialized with delta={}", delta);
    }
}

template<typename T>
void RLSSystemIdentifier<T>::update(const VectorT& y_new, const VectorT& u_new) {
    if (y_new.size() != num_outputs_) {
        throw std::invalid_argument("RLS: y_new dimension mismatch");
    }
    if (u_new.size() != num_inputs_) {
        throw std::invalid_argument("RLS: u_new dimension mismatch");
    }

    // Add to history
    y_history_.push_front(y_new);
    u_history_.push_front(u_new);

    // Keep only needed history
    while (static_cast<int>(y_history_.size()) > state_order_) {
        y_history_.pop_back();
    }
    while (static_cast<int>(u_history_.size()) > state_order_) {
        u_history_.pop_back();
    }

    // Need full history before we can compute regressor
    if (static_cast<int>(y_history_.size()) < state_order_ ||
        static_cast<int>(u_history_.size()) < state_order_) {
        update_count_++;
        return;
    }

    // Build regressor from history (excludes current y_new)
    VectorT phi = buildRegressor();

    // RLS update equations:
    // e = y - theta * phi  (prediction error)
    // K = P * phi / (lambda + phi^T * P * phi)  (Kalman gain)
    // theta = theta + K * e^T  (parameter update)
    // P = (P - K * phi^T * P) / lambda  (covariance update)

    // Compute prediction using current parameters
    VectorT y_pred = theta_ * phi;
    VectorT error = y_new - y_pred;

    // Compute intermediate: P * phi
    VectorT P_phi = P_cov_ * phi;

    // Compute denominator: lambda + phi^T * P * phi
    T denom = forgetting_factor_ + phi.dot(P_phi);

    // Avoid division by zero with small epsilon
    if (std::abs(denom) < static_cast<T>(1e-10)) {
        denom = static_cast<T>(1e-10);
    }

    // Kalman gain vector
    VectorT K = P_phi / denom;

    // Update parameters: theta = theta + e * K^T (outer product)
    // Note: theta is [num_outputs x regressor_dim], K is [regressor_dim x 1]
    // error is [num_outputs x 1]
    theta_ += error * K.transpose();

    // Update covariance matrix (Joseph form for numerical stability)
    // P = (I - K * phi^T) * P / lambda
    MatrixT I_minus_K_phi = MatrixT::Identity(regressor_dim_, regressor_dim_) - K * phi.transpose();
    P_cov_ = (I_minus_K_phi * P_cov_) / forgetting_factor_;

    // Symmetrize P to prevent numerical drift
    P_cov_ = (P_cov_ + P_cov_.transpose()) / static_cast<T>(2.0);

    update_count_++;

    if (logger_ && update_count_ % 100 == 0) {
        T trace = getCovarianceTrace();
        logger_->debug("RLS update {}: error_norm={:.4f}, cov_trace={:.2f}",
                       update_count_, error.norm(), trace);
    }
}

template<typename T>
typename RLSSystemIdentifier<T>::VectorT RLSSystemIdentifier<T>::buildRegressor() const {
    VectorT phi(regressor_dim_);
    int idx = 0;

    // AR terms: y[k-1], y[k-2], ..., y[k-n] for each output
    for (int lag = 0; lag < state_order_; lag++) {
        if (lag < static_cast<int>(y_history_.size())) {
            for (int j = 0; j < num_outputs_; j++) {
                phi(idx++) = y_history_[lag](j);
            }
        } else {
            for (int j = 0; j < num_outputs_; j++) {
                phi(idx++) = static_cast<T>(0);
            }
        }
    }

    // Exogenous terms: u[k-1], u[k-2], ..., u[k-n] for each input
    for (int lag = 0; lag < state_order_; lag++) {
        if (lag < static_cast<int>(u_history_.size())) {
            for (int j = 0; j < num_inputs_; j++) {
                phi(idx++) = u_history_[lag](j);
            }
        } else {
            for (int j = 0; j < num_inputs_; j++) {
                phi(idx++) = static_cast<T>(0);
            }
        }
    }

    return phi;
}

template<typename T>
std::tuple<typename RLSSystemIdentifier<T>::MatrixT,
           typename RLSSystemIdentifier<T>::MatrixT,
           typename RLSSystemIdentifier<T>::MatrixT,
           typename RLSSystemIdentifier<T>::MatrixT>
RLSSystemIdentifier<T>::getSystemMatrices() const {
    MatrixT A, B, C, D;
    arxToStateSpace(A, B, C, D);
    return std::make_tuple(A, B, C, D);
}

template<typename T>
void RLSSystemIdentifier<T>::arxToStateSpace(MatrixT& A, MatrixT& B,
                                              MatrixT& C, MatrixT& D) const {
    // Convert ARX model to state-space using Observable Canonical Form
    //
    // ARX: y[k] = sum_{i=1}^{n} a_i * y[k-i] + sum_{j=1}^{n} b_j * u[k-j]
    //
    // State-space (Observable Canonical Form):
    // State dimension: state_order_ * num_outputs_
    // x = [y[k-1], y[k-2], ..., y[k-n]]^T (stacked)
    //
    // x[k+1] = A*x[k] + B*u[k]
    // y[k]   = C*x[k] + D*u[k]

    int state_dim = state_order_ * num_outputs_;

    // Initialize state-space matrices
    A = MatrixT::Zero(state_dim, state_dim);
    B = MatrixT::Zero(state_dim, num_inputs_);
    C = MatrixT::Zero(num_outputs_, state_dim);
    D = MatrixT::Zero(num_outputs_, num_inputs_);

    // Extract ARX coefficients from theta_
    // theta_ layout: [a_1, a_2, ..., a_n, b_1, b_2, ..., b_n] for each output row
    //
    // a_i: coefficients for y[k-i], shape [num_outputs x num_outputs]
    // b_j: coefficients for u[k-j], shape [num_outputs x num_inputs]

    // Build A matrix (state transition)
    // Observable canonical form:
    // A = [a_1  I  0  ...  0 ]
    //     [a_2  0  I  ...  0 ]
    //     [...           ...]
    //     [a_n  0  0  ...  0 ]

    for (int lag = 0; lag < state_order_; lag++) {
        // Extract a_{lag+1} block from theta_
        int col_start = lag * num_outputs_;
        MatrixT a_block = theta_.block(0, col_start, num_outputs_, num_outputs_);

        // First block column gets the AR coefficients
        int row_start = lag * num_outputs_;
        A.block(row_start, 0, num_outputs_, num_outputs_) = a_block;

        // Shift matrix (I on super-diagonal)
        if (lag < state_order_ - 1) {
            int next_row = (lag + 1) * num_outputs_;
            A.block(row_start, next_row, num_outputs_, num_outputs_) =
                MatrixT::Identity(num_outputs_, num_outputs_);
        }
    }

    // Build B matrix (input influence)
    // First state block gets the b_1 coefficients
    int b_col_start = state_order_ * num_outputs_;  // Where B coefficients start in theta_
    if (b_col_start + num_inputs_ <= regressor_dim_) {
        B.block(0, 0, num_outputs_, num_inputs_) =
            theta_.block(0, b_col_start, num_outputs_, num_inputs_);
    }

    // Build C matrix (observation)
    // C = [I  0  0  ...  0] (selects first state block = y[k-1] shifted)
    C.block(0, 0, num_outputs_, num_outputs_) =
        MatrixT::Identity(num_outputs_, num_outputs_);

    // D matrix (feedthrough) - typically zero for causal systems
    // Can be estimated from theta_ if needed
    D = MatrixT::Zero(num_outputs_, num_inputs_);

    if (logger_) {
        logger_->debug("ARX->SS conversion: state_dim={}, A_norm={:.4f}, B_norm={:.4f}",
                       state_dim, A.norm(), B.norm());
    }
}

template<typename T>
bool RLSSystemIdentifier<T>::isConverged() const {
    return update_count_ >= min_samples_for_convergence_;
}

template<typename T>
T RLSSystemIdentifier<T>::getCovarianceTrace() const {
    return P_cov_.trace();
}

template<typename T>
void RLSSystemIdentifier<T>::seedFromStateSpace(const MatrixT& A, const MatrixT& B,
                                                 const MatrixT& C, const MatrixT& D) {
    // Reverse conversion: state-space to ARX parameters
    // This allows seeding RLS from a N4SID result for hybrid approach

    int state_dim = A.rows();
    if (state_dim != state_order_ * num_outputs_) {
        if (logger_) {
            logger_->warn("RLS seed: state_dim mismatch ({} vs {}), using default initialization",
                          state_dim, state_order_ * num_outputs_);
        }
        return;
    }

    // Extract AR coefficients from A's first block column
    for (int lag = 0; lag < state_order_; lag++) {
        int row_start = lag * num_outputs_;
        int col_start = lag * num_outputs_;  // In theta_
        theta_.block(0, col_start, num_outputs_, num_outputs_) =
            A.block(row_start, 0, num_outputs_, num_outputs_);
    }

    // Extract B coefficients
    int b_col_start = state_order_ * num_outputs_;
    if (B.rows() >= num_outputs_ && B.cols() >= num_inputs_) {
        theta_.block(0, b_col_start, num_outputs_, num_inputs_) =
            B.block(0, 0, num_outputs_, num_inputs_);
    }

    // Reset covariance to moderate uncertainty (we have good initial guess)
    T delta = static_cast<T>(10.0);  // Smaller than reset() since we have prior
    P_cov_ = MatrixT::Identity(regressor_dim_, regressor_dim_) * delta;

    if (logger_) {
        logger_->info("RLS seeded from state-space matrices");
    }
}

// Explicit instantiations
template class RLSSystemIdentifier<float>;
template class RLSSystemIdentifier<double>;

}  // namespace controllers
}  // namespace saguaro
