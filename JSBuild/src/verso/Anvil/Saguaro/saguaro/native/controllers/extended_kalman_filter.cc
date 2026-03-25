// saguaro.native/controllers/extended_kalman_filter.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Extended Kalman Filter implementation with adaptive noise estimation.

#include "controllers/extended_kalman_filter.h"

#include <algorithm>
#include <stdexcept>
#include "spdlog/sinks/stdout_color_sinks.h"

namespace saguaro {
namespace controllers {

ExtendedKalmanFilter::ExtendedKalmanFilter() {
    try {
        logger_ = spdlog::get("highnoon");
        if (!logger_) {
            logger_ = spdlog::stdout_color_mt("highnoon");
        }
    } catch (...) {
        // Continue without logging
    }
}

void ExtendedKalmanFilter::initLinear(const MatrixXf& A,
                                       const MatrixXf& B,
                                       const MatrixXf& C,
                                       const MatrixXf& D,
                                       const MatrixXf& Q,
                                       const MatrixXf& R) {
    // Validate dimensions
    state_dim_ = A.rows();
    input_dim_ = B.cols();
    output_dim_ = C.rows();

    if (A.cols() != state_dim_ || B.rows() != state_dim_ ||
        C.cols() != state_dim_ || D.rows() != output_dim_ ||
        D.cols() != input_dim_ || Q.rows() != state_dim_ ||
        Q.cols() != state_dim_ || R.rows() != output_dim_ ||
        R.cols() != output_dim_) {
        throw std::invalid_argument("EKF: Matrix dimension mismatch");
    }

    A_ = A;
    B_ = B;
    C_ = C;
    D_ = D;
    Q_ = Q;
    R_ = R;

    // Initialize state to zero
    x_hat_ = VectorXf::Zero(state_dim_);
    P_ = MatrixXf::Identity(state_dim_, state_dim_) * 1.0f;

    x_pred_ = VectorXf::Zero(state_dim_);
    P_pred_ = MatrixXf::Identity(state_dim_, state_dim_);

    is_linear_ = true;
    is_initialized_ = true;

    // Phase X: Initialize HD fingerprinting projection
    if (use_hd_fingerprinting_ && output_dim_ > 0) {
        saguaro::hd_state::HDStateConfig hd_config;
        hd_config.compression_ratio = std::max(1, output_dim_ / kHDFingerprintDim);
        hd_config.use_sparse_projection = true;
        hd_config.sparse_density = 3;
        hd_projection_.resize(output_dim_ * kHDFingerprintDim);
        saguaro::hd_state::generate_sparse_projection(
            hd_projection_.data(), output_dim_, kHDFingerprintDim, hd_config);
        innovation_fingerprint_.assign(kHDFingerprintDim, 0.0f);
        prev_fingerprint_.assign(kHDFingerprintDim, 0.0f);
    }

    if (logger_) {
        logger_->info("EKF initialized (linear): state_dim={}, input_dim={}, output_dim={}, hd_fingerprint={}",
                      state_dim_, input_dim_, output_dim_, use_hd_fingerprinting_);
    }
}

void ExtendedKalmanFilter::initNonlinear(DynamicsFunc f, JacobianFunc jacobian_f,
                                          DynamicsFunc h, JacobianFunc jacobian_h,
                                          const MatrixXf& Q,
                                          const MatrixXf& R) {
    if (!f || !jacobian_f || !h || !jacobian_h) {
        throw std::invalid_argument("EKF: All function pointers must be valid");
    }

    f_ = f;
    jacobian_f_ = jacobian_f;
    h_ = h;
    jacobian_h_ = jacobian_h;

    state_dim_ = Q.rows();
    output_dim_ = R.rows();
    input_dim_ = 1;  // Will be determined from first predict call

    Q_ = Q;
    R_ = R;

    // Initialize state
    x_hat_ = VectorXf::Zero(state_dim_);
    P_ = MatrixXf::Identity(state_dim_, state_dim_) * 1.0f;

    x_pred_ = VectorXf::Zero(state_dim_);
    P_pred_ = MatrixXf::Identity(state_dim_, state_dim_);

    is_linear_ = false;
    is_initialized_ = true;

    // Phase X: Initialize HD fingerprinting projection
    if (use_hd_fingerprinting_ && output_dim_ > 0) {
        saguaro::hd_state::HDStateConfig hd_config;
        hd_config.compression_ratio = std::max(1, output_dim_ / kHDFingerprintDim);
        hd_config.use_sparse_projection = true;
        hd_config.sparse_density = 3;
        hd_projection_.resize(output_dim_ * kHDFingerprintDim);
        saguaro::hd_state::generate_sparse_projection(
            hd_projection_.data(), output_dim_, kHDFingerprintDim, hd_config);
        innovation_fingerprint_.assign(kHDFingerprintDim, 0.0f);
        prev_fingerprint_.assign(kHDFingerprintDim, 0.0f);
    }

    if (logger_) {
        logger_->info("EKF initialized (nonlinear): state_dim={}, output_dim={}, hd_fingerprint={}",
                      state_dim_, output_dim_, use_hd_fingerprinting_);
    }
}

void ExtendedKalmanFilter::updateMatrices(const MatrixXf& A, const MatrixXf& B,
                                           const MatrixXf& C, const MatrixXf& D) {
    if (!is_linear_) {
        if (logger_) {
            logger_->warn("EKF: updateMatrices called on nonlinear filter, ignoring");
        }
        return;
    }

    // Validate dimensions
    if (A.rows() != state_dim_ || A.cols() != state_dim_ ||
        B.rows() != state_dim_ || C.rows() != output_dim_ ||
        C.cols() != state_dim_) {
        if (logger_) {
            logger_->error("EKF: updateMatrices dimension mismatch");
        }
        return;
    }

    A_ = A;
    B_ = B;
    C_ = C;
    D_ = D;
}

void ExtendedKalmanFilter::enableAdaptiveNoise(bool enable, float adaptation_rate) {
    adaptive_noise_ = enable;
    adaptation_rate_ = std::clamp(adaptation_rate, 0.001f, 0.5f);

    if (enable && logger_) {
        logger_->info("EKF adaptive noise enabled with rate={:.4f}", adaptation_rate_);
    }
}

void ExtendedKalmanFilter::predict(const std::vector<float>& u) {
    if (!is_initialized_) {
        throw std::runtime_error("EKF: Filter not initialized");
    }

    // Convert input to Eigen
    VectorXf u_vec(u.size());
    for (size_t i = 0; i < u.size(); i++) {
        u_vec(i) = u[i];
    }

    if (is_linear_) {
        // Standard Kalman predict
        // x_pred = A * x_hat + B * u
        x_pred_ = A_ * x_hat_;
        if (B_.cols() == u_vec.size()) {
            x_pred_ += B_ * u_vec;
        }

        // P_pred = A * P * A^T + Q
        P_pred_ = A_ * P_ * A_.transpose() + Q_;
    } else {
        // EKF predict with Jacobian
        // x_pred = f(x_hat, u)
        x_pred_ = f_(x_hat_, u_vec);

        // F = df/dx at x_hat
        MatrixXf F = jacobian_f_(x_hat_);

        // P_pred = F * P * F^T + Q
        P_pred_ = F * P_ * F.transpose() + Q_;
    }

    // Ensure numerical stability
    regularizeCovariance(P_pred_);
}

void ExtendedKalmanFilter::update(const std::vector<float>& y) {
    if (!is_initialized_) {
        throw std::runtime_error("EKF: Filter not initialized");
    }

    // Convert measurement to Eigen
    VectorXf y_vec(y.size());
    for (size_t i = 0; i < y.size(); i++) {
        y_vec(i) = y[i];
    }

    VectorXf y_pred;
    MatrixXf H;

    if (is_linear_) {
        // Linear observation
        y_pred = C_ * x_pred_;
        H = C_;
    } else {
        // Nonlinear observation with Jacobian
        VectorXf zero_input = VectorXf::Zero(input_dim_);
        y_pred = h_(x_pred_, zero_input);
        H = jacobian_h_(x_pred_);
    }

    // Innovation (prediction error)
    VectorXf innovation = y_vec - y_pred;
    last_innovation_ = innovation;

    // Innovation covariance: S = H * P_pred * H^T + R
    MatrixXf S = H * P_pred_ * H.transpose() + R_;

    // Kalman gain: K = P_pred * H^T * S^-1
    // Use LDLT decomposition for numerical stability
    Eigen::LDLT<MatrixXf> S_ldlt(S);
    if (S_ldlt.info() != Eigen::Success) {
        if (logger_) {
            logger_->warn("EKF: S matrix decomposition failed, skipping update");
        }
        x_hat_ = x_pred_;
        P_ = P_pred_;
        return;
    }

    MatrixXf K = P_pred_ * H.transpose() * S_ldlt.solve(MatrixXf::Identity(output_dim_, output_dim_));

    // State update: x_hat = x_pred + K * innovation
    x_hat_ = x_pred_ + K * innovation;

    // Covariance update (Joseph form for numerical stability):
    // P = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
    MatrixXf I_KH = MatrixXf::Identity(state_dim_, state_dim_) - K * H;
    P_ = I_KH * P_pred_ * I_KH.transpose() + K * R_ * K.transpose();

    // Ensure symmetry
    P_ = (P_ + P_.transpose()) / 2.0f;

    // Adaptive noise estimation
    if (adaptive_noise_) {
        adaptNoiseCovariances(innovation, S);
    }

    // Phase X: HD fingerprinting for stagnation detection
    if (use_hd_fingerprinting_ && !hd_projection_.empty()) {
        // Save previous fingerprint
        prev_fingerprint_ = innovation_fingerprint_;

        // Compute new fingerprint via HD projection
        std::vector<float> innovation_data(output_dim_);
        for (int i = 0; i < output_dim_; i++) {
            innovation_data[i] = innovation(i);
        }
        saguaro::hd_state::HDStateEncode(
            innovation_data.data(), hd_projection_.data(),
            innovation_fingerprint_.data(), output_dim_, kHDFingerprintDim);

        // Compute cosine similarity between current and previous
        float dot = 0.0f, norm_curr = 0.0f, norm_prev = 0.0f;
        for (int i = 0; i < kHDFingerprintDim; i++) {
            dot += innovation_fingerprint_[i] * prev_fingerprint_[i];
            norm_curr += innovation_fingerprint_[i] * innovation_fingerprint_[i];
            norm_prev += prev_fingerprint_[i] * prev_fingerprint_[i];
        }
        float denom = std::sqrt(norm_curr) * std::sqrt(norm_prev);
        fingerprint_similarity_ = (denom > 1e-12f) ? (dot / denom) : 0.0f;
        fingerprint_similarity_ = std::max(0.0f, std::min(1.0f, fingerprint_similarity_));
    }
}

void ExtendedKalmanFilter::adaptNoiseCovariances(const VectorXf& innovation,
                                                  const MatrixXf& S) {
    // Store innovation for statistics
    innovation_history_.push_front(innovation);
    if (innovation_history_.size() > kInnovationHistorySize) {
        innovation_history_.pop_back();
    }

    if (innovation_history_.size() < kInnovationHistorySize / 2) {
        return;  // Not enough data
    }

    // Compute normalized innovation squared (NIS)
    // NIS = innovation^T * S^-1 * innovation
    // Should be chi-squared distributed with dof = output_dim_

    Eigen::LDLT<MatrixXf> S_ldlt(S);
    if (S_ldlt.info() != Eigen::Success) {
        return;
    }

    float nis = innovation.transpose() * S_ldlt.solve(innovation);

    // Expected NIS is output_dim_ (mean of chi-squared)
    float expected_nis = static_cast<float>(output_dim_);
    float nis_ratio = nis / expected_nis;

    // If NIS is consistently too high, increase R (measurement noise)
    // If NIS is consistently too low, decrease R
    if (nis_ratio > 2.0f) {
        // Underestimating measurement noise
        R_ = R_ * (1.0f + adaptation_rate_);
        if (logger_) {
            logger_->debug("EKF adaptive: increasing R (NIS ratio={:.2f})", nis_ratio);
        }
    } else if (nis_ratio < 0.5f) {
        // Overestimating measurement noise
        R_ = R_ * (1.0f - adaptation_rate_);
        // Ensure R stays positive definite
        for (int i = 0; i < R_.rows(); i++) {
            R_(i, i) = std::max(R_(i, i), 1e-6f);
        }
        if (logger_) {
            logger_->debug("EKF adaptive: decreasing R (NIS ratio={:.2f})", nis_ratio);
        }
    }

    // Also adapt Q based on innovation variance vs prediction variance
    // This is a simplified heuristic
    if (innovation_history_.size() >= kInnovationHistorySize) {
        // Compute sample covariance of innovations
        VectorXf mean = VectorXf::Zero(output_dim_);
        for (const auto& innov : innovation_history_) {
            mean += innov;
        }
        mean /= innovation_history_.size();

        MatrixXf sample_cov = MatrixXf::Zero(output_dim_, output_dim_);
        for (const auto& innov : innovation_history_) {
            VectorXf centered = innov - mean;
            sample_cov += centered * centered.transpose();
        }
        sample_cov /= (innovation_history_.size() - 1);

        // Innovation covariance should approximate H*P*H^T + R
        // If sample_cov >> S, we're underestimating Q
        float sample_trace = sample_cov.trace();
        float expected_trace = S.trace();

        if (sample_trace > expected_trace * 1.5f) {
            Q_ = Q_ * (1.0f + adaptation_rate_ * 0.5f);
        } else if (sample_trace < expected_trace * 0.5f) {
            Q_ = Q_ * (1.0f - adaptation_rate_ * 0.5f);
            // Ensure Q stays positive
            for (int i = 0; i < Q_.rows(); i++) {
                Q_(i, i) = std::max(Q_(i, i), 1e-6f);
            }
        }
    }
}

void ExtendedKalmanFilter::regularizeCovariance(MatrixXf& P) {
    // Ensure symmetry
    P = (P + P.transpose()) / 2.0f;

    // Ensure positive definiteness by clamping eigenvalues
    Eigen::SelfAdjointEigenSolver<MatrixXf> solver(P);
    if (solver.info() == Eigen::Success) {
        VectorXf eigenvalues = solver.eigenvalues();
        bool needs_fix = false;
        for (int i = 0; i < eigenvalues.size(); i++) {
            if (eigenvalues(i) < 1e-8f) {
                eigenvalues(i) = 1e-8f;
                needs_fix = true;
            }
        }
        if (needs_fix) {
            P = solver.eigenvectors() * eigenvalues.asDiagonal() * 
                solver.eigenvectors().transpose();
        }
    }
}

std::vector<float> ExtendedKalmanFilter::getState() const {
    std::vector<float> state(state_dim_);
    for (int i = 0; i < state_dim_; i++) {
        state[i] = x_hat_(i);
    }
    return state;
}

std::tuple<ExtendedKalmanFilter::MatrixXf, ExtendedKalmanFilter::MatrixXf>
ExtendedKalmanFilter::getNoiseCovariances() const {
    return std::make_tuple(Q_, R_);
}

void ExtendedKalmanFilter::reset(int state_dim) {
    state_dim_ = state_dim;
    x_hat_ = VectorXf::Zero(state_dim_);
    P_ = MatrixXf::Identity(state_dim_, state_dim_) * 1.0f;
    x_pred_ = VectorXf::Zero(state_dim_);
    P_pred_ = MatrixXf::Identity(state_dim_, state_dim_);
    innovation_history_.clear();
}

void ExtendedKalmanFilter::setInitialState(const VectorXf& x0,
                                            const MatrixXf& P0) {
    if (x0.size() != state_dim_) {
        throw std::invalid_argument("EKF: Initial state dimension mismatch");
    }

    x_hat_ = x0;

    if (P0.rows() > 0 && P0.cols() > 0) {
        if (P0.rows() != state_dim_ || P0.cols() != state_dim_) {
            throw std::invalid_argument("EKF: Initial covariance dimension mismatch");
        }
        P_ = P0;
    } else {
        P_ = MatrixXf::Identity(state_dim_, state_dim_) * 1.0f;
    }
}

}  // namespace controllers
}  // namespace saguaro
