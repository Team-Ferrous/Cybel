// src/controllers/hardware/fallback_controller.h
// Fallback PID controller for HSMN HIL deployment
// Provides safety fallback when HSMN exceeds latency budget (<15ms watchdog)
// Ensures seamless transition with no control signal discontinuity

#ifndef SAGUARO_CONTROLLERS_HARDWARE_FALLBACK_CONTROLLER_H_
#define SAGUARO_CONTROLLERS_HARDWARE_FALLBACK_CONTROLLER_H_

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

namespace saguaro {
namespace controllers {
namespace hardware {

// PID controller configuration
struct PIDConfig {
    float kp = 1.0f;               // Proportional gain
    float ki = 0.1f;               // Integral gain
    float kd = 0.01f;              // Derivative gain
    float integral_limit = 10.0f;  // Anti-windup limit
    float output_min = 0.0f;       // Minimum output clamp
    float output_max = 1.0f;       // Maximum output clamp
    float derivative_filter_alpha = 0.1f; // Low-pass filter for derivative (0 = no filter, 1 = full filter)
    float dt_default = 1.0f / 120.0f;     // Default timestep (120 Hz)
};

// PID state for single control loop
struct PIDState {
    float integral = 0.0f;         // Accumulated integral term
    float prev_error = 0.0f;       // Previous error for derivative
    float filtered_derivative = 0.0f; // Filtered derivative for noise reduction
    bool initialized = false;      // First-run flag
};

// Fallback PID controller with multiple channels
class FallbackPIDController {
 public:
    explicit FallbackPIDController(const PIDConfig& config, uint32_t num_channels = 1);
    ~FallbackPIDController() = default;

    // Compute PID control output for all channels
    // setpoints: desired values for each channel
    // measurements: actual measured values for each channel
    // dt_seconds: time since last update (use default if <= 0)
    // Returns: control outputs for each channel
    std::vector<float> compute_control(
        const std::vector<float>& setpoints,
        const std::vector<float>& measurements,
        float dt_seconds = 0.0f);

    // Reset controller state (call when switching from HSMN to PID)
    void reset();

    // Reset specific channel only
    void reset_channel(uint32_t channel_idx);

    // Update PID gains for all channels
    void update_gains(float kp, float ki, float kd);

    // Update PID gains for specific channel
    void update_channel_gains(uint32_t channel_idx, float kp, float ki, float kd);

    // Get/set configuration
    const PIDConfig& get_config() const { return config_; }
    void set_config(const PIDConfig& config);

    // Get current state for monitoring/debugging
    const std::vector<PIDState>& get_states() const { return states_; }

 private:
    PIDConfig config_;
    std::vector<PIDState> states_;
    uint32_t num_channels_;

    // Compute PID for single channel
    float compute_single_channel(
        float setpoint,
        float measurement,
        float dt_seconds,
        PIDState& state);

    // Apply anti-windup and clamping
    float apply_limits(float output);
    float apply_integral_limit(float integral);
};

}  // namespace hardware
}  // namespace controllers
}  // namespace saguaro

#endif  // SAGUARO_CONTROLLERS_HARDWARE_FALLBACK_CONTROLLER_H_
