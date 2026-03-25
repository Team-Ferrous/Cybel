#ifndef CONFIG_PID_CONFIG_H_
#define CONFIG_PID_CONFIG_H_

#include <string>
#include <vector>
#include <map>
#include <Eigen/Dense> // Include Eigen for MatrixXf

// This struct holds the configuration for the PID controller logic,
// including gains, setpoints, and auto-tuning parameters.
struct PIDConfig {
    // List of metric names the controller will monitor.
    std::vector<std::string> metric_names;

    // List of control input names the controller will manage.
    std::vector<std::string> control_input_names;

    // Target values (R) for each metric.
    std::map<std::string, float> Setpoint;

    // Proportional and Integral gain matrices for MIMO control.
    // Dimensions: (num_control_inputs x num_system_outputs)
    Eigen::MatrixXf Kp_matrix;
    Eigen::MatrixXf Ki_matrix;
    Eigen::MatrixXf Kd_matrix;

    // Per-control clamping bounds aligned with control_input_names.
    std::vector<float> control_min_bounds;
    std::vector<float> control_max_bounds;

    // Mapping of control inputs to the metric names they primarily regulate.
    std::map<std::string, std::vector<std::string>> control_metric_map;

    // Parameters for the Relay with Hysteresis auto-tuner.
    float relay_output_amplitude = 1.5f;
    float relay_hysteresis = 0.03f;
    float error_threshold = 2.5f;
    int patience = 50;
};

#endif // CONFIG_PID_CONFIG_H_
