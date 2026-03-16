#include "kalman_filter.h"
#include <Eigen/Dense>
#include <stdexcept>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace {
auto kalman_logger = spdlog::stdout_color_mt("kalman_filter");
} // anonymous namespace

KalmanFilter::KalmanFilter() = default;

void KalmanFilter::init(const Eigen::MatrixXf& A,
                        const Eigen::MatrixXf& B,
                        const Eigen::MatrixXf& C,
                        const Eigen::MatrixXf& D,
                        const Eigen::MatrixXf& Q,
                        const Eigen::MatrixXf& R_kalman) {
    A_ = A; B_ = B; C_ = C; D_ = D; Q_ = Q; R_kalman_ = R_kalman;
    int state_dim = A.rows();
    if (state_dim == 0) {
        throw std::invalid_argument("State dimension cannot be zero.");
    }
    x_hat_ = Eigen::VectorXf::Zero(state_dim);
    P_ = Eigen::MatrixXf::Identity(state_dim, state_dim);
}

void KalmanFilter::predict(const std::vector<float>& u_vec) {
    if (static_cast<Eigen::Index>(u_vec.size()) != B_.cols()) {
        kalman_logger->error("Control input vector size {} does not match B matrix columns {}.", u_vec.size(), B_.cols());
        throw std::invalid_argument("Control input vector size does not match B matrix columns.");
    }
    Eigen::Map<const Eigen::VectorXf> u(u_vec.data(), u_vec.size());

    // x_hat(k|k-1) = A * x_hat(k-1|k-1) + B * u(k-1)
    x_hat_ = A_ * x_hat_ + B_ * u;

    // P(k|k-1) = A * P(k-1|k-1) * A^T + Q
    P_ = A_ * P_ * A_.transpose() + Q_;
}

void KalmanFilter::update(const std::vector<float>& y_vec) {
    if (static_cast<Eigen::Index>(y_vec.size()) != C_.rows()) {
        kalman_logger->error("Measurement vector size {} does not match C matrix rows {}.", y_vec.size(), C_.rows());
        throw std::invalid_argument("Measurement vector size does not match C matrix rows.");
    }
    if (x_hat_.size() != C_.cols()) {
        kalman_logger->error("State vector size {} does not match C matrix columns {}.", x_hat_.size(), C_.cols());
        throw std::invalid_argument("State vector size does not match C matrix columns.");
    }

    Eigen::Map<const Eigen::VectorXf> y(y_vec.data(), y_vec.size());

    // Innovation or measurement residual: y_tilde = y - C * x_hat
    Eigen::VectorXf y_tilde = y - C_ * x_hat_;

    // Innovation covariance: S = C * P * C^T + R
    Eigen::MatrixXf S = C_ * P_ * C_.transpose() + R_kalman_;

    // Kalman gain: K = P_ * C_.transpose() * S.inverse()
    Eigen::MatrixXf K = P_ * C_.transpose() * S.inverse();

    // Update state estimate: x_hat(k|k) = x_hat(k-1|k-1) + K * y_tilde
    x_hat_ = x_hat_ + K * y_tilde;

    // Update error covariance: P(k|k) = (I - K * C) * P(k|k-1)
    int state_dim = A_.rows();
    Eigen::MatrixXf I = Eigen::MatrixXf::Identity(state_dim, state_dim);
    P_ = (I - K * C_) * P_;
}

std::vector<float> KalmanFilter::getState() const {
    std::vector<float> state_vec(x_hat_.data(), x_hat_.data() + x_hat_.size());
    return state_vec;
}