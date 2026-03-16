#ifndef SRC_CONTROLLERS_KALMAN_FILTER_H_
#define SRC_CONTROLLERS_KALMAN_FILTER_H_

#include <vector>
#include <string>
#include <Eigen/Dense>

/**
 * @class KalmanFilter
 * @brief Encapsulates the state estimation logic using a Kalman filter.
 *
 * This class implements a standard linear Kalman filter for state estimation
 * of a system described by state-space equations. It provides methods for
 * predicting the next state and updating the estimate based on new measurements.
 */
class KalmanFilter {
public:
    /**
     * @brief Default constructor.
     */
    KalmanFilter();

    /**
     * @brief Initializes or re-initializes the Kalman filter with system matrices.
     * @param A The state transition matrix.
     * @param B The control input matrix.
     * @param C The observation matrix.
     * @param D The feed-forward matrix.
     * @param Q The process noise covariance matrix.
     * @param R_kalman The measurement noise covariance matrix.
     */
    void init(const Eigen::MatrixXf& A,
                 const Eigen::MatrixXf& B,
                 const Eigen::MatrixXf& C,
                 const Eigen::MatrixXf& D,
                 const Eigen::MatrixXf& Q,
                 const Eigen::MatrixXf& R_kalman);

    /**
     * @brief Predicts the next state based on the control input.
     * Implements the prediction step of the Kalman filter:
     * x_hat(k|k-1) = A * x_hat(k-1|k-1) + B * u(k-1)
     * P(k|k-1) = A * P(k-1|k-1) * A^T + Q
     * @param u The control input vector u(k-1).
     */
    void predict(const std::vector<float>& u);

    /**
     * @brief Updates the state estimate based on a new measurement.
     * Implements the update/correction step of the Kalman filter.
     * @param y The measurement vector y(k).
     */
    void update(const std::vector<float>& y);

    /**
     * @brief Gets the current state estimate.
     * @return The current state estimate vector x_hat.
     */
    std::vector<float> getState() const;

private:
    // State-space matrices
    Eigen::MatrixXf A_;
    Eigen::MatrixXf B_;
    Eigen::MatrixXf C_;
    Eigen::MatrixXf D_;

    // Noise covariance matrices
    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_kalman_;

    // Filter state
    Eigen::VectorXf x_hat_; // State estimate
    Eigen::MatrixXf P_; // Error covariance
};

#endif // SRC_CONTROLLERS_KALMAN_FILTER_H_
