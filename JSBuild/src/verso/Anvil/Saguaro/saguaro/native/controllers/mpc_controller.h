#ifndef CONTROLLERS_MPC_CONTROLLER_H_
#define CONTROLLERS_MPC_CONTROLLER_H_

#include <vector>
#include "Eigen/Dense"

class MPCController {
public:
    MPCController(int state_dim, int control_dim, int prediction_horizon);
    ~MPCController();

    // Initializes the MPC controller with the system model and cost matrices
    void init(const Eigen::MatrixXf& A, const Eigen::MatrixXf& B,
              const Eigen::MatrixXf& Q, const Eigen::MatrixXf& R);

    // Computes the optimal control action for the given state
    Eigen::VectorXf compute_control_action(const Eigen::VectorXf& current_state);
    void reset();

private:
    // MPC parameters
    int state_dim_;
    int control_dim_;
    int prediction_horizon_;

    // System dynamics
    Eigen::MatrixXf A_;
    Eigen::MatrixXf B_;

    // Cost function weights
    Eigen::MatrixXf Q_;
    Eigen::MatrixXf R_;

    // Helper function to setup the QP problem for OSQP
};

#endif // CONTROLLERS_MPC_CONTROLLER_H_
