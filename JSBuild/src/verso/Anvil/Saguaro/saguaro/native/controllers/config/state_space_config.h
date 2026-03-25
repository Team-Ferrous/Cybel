#ifndef CONFIG_STATE_SPACE_CONFIG_H_
#define CONFIG_STATE_SPACE_CONFIG_H_

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense> // Include Eigen for MatrixXf

// This struct holds the configuration for the state-space controller,
// including the system matrices and noise covariance matrices for the Kalman filter,
// and an optional state-feedback gain matrix.
struct ControllerConfig {
    // State-space system matrices
    Eigen::MatrixXf A;
    Eigen::MatrixXf B;
    Eigen::MatrixXf C;
    Eigen::MatrixXf D;

    // --- START: DEFINITIVE FIX for LQR vs Kalman R Matrix ---
    // Q: Process noise covariance for Kalman Filter / State cost for LQR.
    Eigen::MatrixXf Q; // Process noise
    // R_kalman: Measurement noise covariance for Kalman Filter. Must be num_outputs x num_outputs.
    Eigen::MatrixXf R_kalman; // For Kalman Filter
    // R_lqr: Control input cost matrix for LQR solver (used in Python). Not used in C++.
    Eigen::MatrixXf R_lqr;    // For LQR/MPC Controller
    // --- END: DEFINITIVE FIX ---

    // State-feedback gain matrix for LQR controller (optional)
    Eigen::MatrixXf K;

    // --- START: DEFINITIVE FIX for Data Scaling ---
    // Scaler parameters for normalizing/un-normalizing data to match the system ID model.
    std::vector<float> u_scaler_mean;
    std::vector<float> u_scaler_scale;
    std::vector<float> y_scaler_mean;
    std::vector<float> y_scaler_scale;
    // --- END: DEFINITIVE FIX ---
};

#endif // CONFIG_STATE_SPACE_CONFIG_H_
