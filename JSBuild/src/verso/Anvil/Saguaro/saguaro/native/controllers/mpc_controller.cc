#include "mpc_controller.h"

#include <iostream>
#include <vector>
#include <stdexcept>
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include "common/parallel/parallel_backend.h"

namespace {
auto mpc_logger = spdlog::stdout_color_mt("mpc_controller");
} // anonymous namespace

MPCController::MPCController(int state_dim, int control_dim, int prediction_horizon)
    : state_dim_(state_dim),
      control_dim_(control_dim),
      prediction_horizon_(prediction_horizon) {
    if (state_dim <= 0 || control_dim <= 0 || prediction_horizon <= 0) {
        mpc_logger->error("Invalid dimensions provided to MPCController constructor.");
        throw std::invalid_argument("Invalid dimensions for MPCController.");
    }
}

MPCController::~MPCController() = default;

void MPCController::init(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B,
                         const Eigen::MatrixXf& Q, const Eigen::MatrixXf& R) {
    if (A.rows() != state_dim_ || A.cols() != state_dim_) {
        mpc_logger->error("A matrix dimensions are incorrect. Expected {}x{}, got {}x{}", state_dim_, state_dim_, A.rows(), A.cols());
        throw std::invalid_argument("A matrix dimensions are incorrect.");
    }
    if (B.rows() != state_dim_ || B.cols() != control_dim_) {
        mpc_logger->error("B matrix dimensions are incorrect. Expected {}x{}, got {}x{}", state_dim_, control_dim_, B.rows(), B.cols());
        throw std::invalid_argument("B matrix dimensions are incorrect.");
    }
    if (Q.rows() != state_dim_ || Q.cols() != state_dim_) {
        mpc_logger->error("Q matrix dimensions are incorrect. Expected {}x{}, got {}x{}", state_dim_, state_dim_, Q.rows(), Q.cols());
        throw std::invalid_argument("Q matrix dimensions are incorrect.");
    }
    if (R.rows() != control_dim_ || R.cols() != control_dim_) {
        mpc_logger->error("R matrix dimensions are incorrect. Expected {}x{}, got {}x{}", control_dim_, control_dim_, R.rows(), R.cols());
        throw std::invalid_argument("R matrix dimensions are incorrect.");
    }
    A_ = A;
    B_ = B;
    Q_ = Q;
    R_ = R;
}

Eigen::VectorXf MPCController::compute_control_action(const Eigen::VectorXf& current_state) {
    if (current_state.size() != state_dim_) {
        mpc_logger->error("Current state vector size {} does not match state dimension {}.", current_state.size(), state_dim_);
        throw std::invalid_argument("Current state vector size does not match state dimension.");
    }
    // --- 1. Formulate the QP problem matrices (P and q) ---
    // The goal is to find a sequence of control inputs U = [u_0, u_1, ..., u_{N-1}]
    // that minimizes the cost function: J = sum_{i=0 to N-1} (x_i^T Q x_i + u_i^T R u_i)
    // The state evolves as: x_{i+1} = A*x_i + B*u_i
    // We can express the entire future state sequence X in terms of the initial state x_0
    // and the control sequence U:  X = A_bar * x_0 + B_bar * U

    // Augmented dynamics matrices
    int n_states_horizon = prediction_horizon_ * state_dim_;
    int m_controls_horizon = prediction_horizon_ * control_dim_;

    Eigen::MatrixXf A_bar = Eigen::MatrixXf::Zero(n_states_horizon, state_dim_);
    Eigen::MatrixXf B_bar = Eigen::MatrixXf::Zero(n_states_horizon, m_controls_horizon);

    // Precompute powers of A
    std::vector<Eigen::MatrixXf> A_powers(prediction_horizon_ + 1);
    A_powers[0] = Eigen::MatrixXf::Identity(state_dim_, state_dim_);
    for (int i = 0; i < prediction_horizon_; ++i) {
        A_powers[i+1] = A_ * A_powers[i];
    }

    // Fill A_bar
    saguaro::parallel::ForRange(
        0, static_cast<size_t>(prediction_horizon_), 8,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const int i = static_cast<int>(idx);
                A_bar.block(i * state_dim_, 0, state_dim_, state_dim_) = A_powers[i + 1];
            }
        });

    // Fill B_bar
    saguaro::parallel::ForRange2D(
        0, static_cast<size_t>(prediction_horizon_), 4,
        0, static_cast<size_t>(prediction_horizon_), 4,
        [&](size_t row_begin, size_t row_end, size_t col_begin, size_t col_end) {
            for (size_t row = row_begin; row < row_end; ++row) {
                for (size_t col = col_begin; col < col_end; ++col) {
                    const int i = static_cast<int>(row);
                    const int j = static_cast<int>(col);
                    if (j <= i) {
                        B_bar.block(i * state_dim_, j * control_dim_, state_dim_, control_dim_) =
                            A_powers[i - j] * B_;
                    }
                }
            }
        });


    // Augmented cost matrices
    Eigen::MatrixXf Q_bar = Eigen::MatrixXf::Zero(n_states_horizon, n_states_horizon);
    Eigen::MatrixXf R_bar = Eigen::MatrixXf::Zero(m_controls_horizon, m_controls_horizon);
    saguaro::parallel::ForRange(
        0, static_cast<size_t>(prediction_horizon_), 8,
        [&](size_t begin, size_t end) {
            for (size_t idx = begin; idx < end; ++idx) {
                const int i = static_cast<int>(idx);
                Q_bar.block(i * state_dim_, i * state_dim_, state_dim_, state_dim_) = Q_;
                R_bar.block(i * control_dim_, i * control_dim_, control_dim_, control_dim_) = R_;
            }
        });

    // Substituting X into the cost function J gives a QP in terms of U only:
    // J(U) = 0.5 * U^T * P * U + q^T * U
    // where P = B_bar^T * Q_bar * B_bar + R_bar
    // and   q = B_bar^T * Q_bar * A_bar * x_0

    Eigen::MatrixXf P = B_bar.transpose() * Q_bar * B_bar + R_bar;
    Eigen::VectorXf q = B_bar.transpose() * Q_bar * A_bar * current_state;

    // --- 2. Solve the QP Analytically ---
    // For an unconstrained problem, the minimum is found by setting the gradient to zero:
    // dJ/dU = P * U + q = 0  =>  P * U = -q
    // We can solve this linear system for the optimal control sequence U.

    // Use LDLT decomposition, which is robust for symmetric positive semidefinite matrices.
    Eigen::LDLT<Eigen::MatrixXf> ldlt(P);
    if (ldlt.info() != Eigen::Success) {
        mpc_logger->error("LDLT decomposition failed. The P matrix is likely not positive definite.");
        return Eigen::VectorXf::Zero(control_dim_);
    }

    Eigen::VectorXf optimal_control_sequence = ldlt.solve(-q);
    if (ldlt.info() != Eigen::Success) {
        mpc_logger->error("Failed to solve the linear system for the optimal control sequence.");
        return Eigen::VectorXf::Zero(control_dim_);
    }

    // --- 3. Return the first control action from the optimal sequence ---
    // This is the core principle of MPC: "plan for the horizon, apply the first step".
    return optimal_control_sequence.head(control_dim_);
}

void MPCController::reset() {
    // No stateful buffers yet, but keep the hook to support future warm starts.
}
