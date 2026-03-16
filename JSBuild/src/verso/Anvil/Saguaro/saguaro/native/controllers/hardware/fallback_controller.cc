// src/controllers/hardware/fallback_controller.cc
// Implementation of fallback PID controller for HSMN HIL deployment

#include "src/controllers/hardware/fallback_controller.h"

#include <algorithm>
#include <cmath>

namespace saguaro {
namespace controllers {
namespace hardware {

FallbackPIDController::FallbackPIDController(const PIDConfig& config, uint32_t num_channels)
    : config_(config), num_channels_(num_channels) {
    states_.resize(num_channels_);
    for (auto& state : states_) {
        state.initialized = false;
    }
}

std::vector<float> FallbackPIDController::compute_control(
    const std::vector<float>& setpoints,
    const std::vector<float>& measurements,
    float dt_seconds) {

    // Use default dt if not provided or invalid
    if (dt_seconds <= 0.0f) {
        dt_seconds = config_.dt_default;
    }

    // Validate input dimensions
    if (setpoints.size() != num_channels_ || measurements.size() != num_channels_) {
        // Return safe zero output on dimension mismatch
        return std::vector<float>(num_channels_, 0.0f);
    }

    std::vector<float> outputs;
    outputs.reserve(num_channels_);

    // Compute PID for each channel
    for (uint32_t i = 0; i < num_channels_; ++i) {
        float output = compute_single_channel(
            setpoints[i],
            measurements[i],
            dt_seconds,
            states_[i]);
        outputs.push_back(output);
    }

    return outputs;
}

float FallbackPIDController::compute_single_channel(
    float setpoint,
    float measurement,
    float dt_seconds,
    PIDState& state) {

    // Compute error
    float error = setpoint - measurement;

    // Proportional term
    float p_term = config_.kp * error;

    // Integral term with anti-windup
    state.integral += error * dt_seconds;
    state.integral = apply_integral_limit(state.integral);
    float i_term = config_.ki * state.integral;

    // Derivative term with filtering (to reduce noise amplification)
    float derivative = 0.0f;
    if (state.initialized) {
        // Compute raw derivative
        float raw_derivative = (error - state.prev_error) / dt_seconds;

        // Apply low-pass filter: filtered = alpha * raw + (1-alpha) * prev_filtered
        state.filtered_derivative =
            config_.derivative_filter_alpha * raw_derivative +
            (1.0f - config_.derivative_filter_alpha) * state.filtered_derivative;

        derivative = state.filtered_derivative;
    } else {
        // First run: no derivative, just initialize
        state.initialized = true;
    }
    float d_term = config_.kd * derivative;

    // Update prev_error for next iteration
    state.prev_error = error;

    // Compute total output
    float output = p_term + i_term + d_term;

    // Apply output limits
    output = apply_limits(output);

    // Anti-windup back-calculation: if output is saturated, stop integrator
    // This prevents integral windup when actuator is at limits
    if (output == config_.output_max || output == config_.output_min) {
        // Revert integral update (optional: could also reduce integral toward zero)
        state.integral -= error * dt_seconds;
    }

    return output;
}

float FallbackPIDController::apply_limits(float output) {
    return std::max(config_.output_min, std::min(config_.output_max, output));
}

float FallbackPIDController::apply_integral_limit(float integral) {
    return std::max(-config_.integral_limit, std::min(config_.integral_limit, integral));
}

void FallbackPIDController::reset() {
    for (auto& state : states_) {
        state.integral = 0.0f;
        state.prev_error = 0.0f;
        state.filtered_derivative = 0.0f;
        state.initialized = false;
    }
}

void FallbackPIDController::reset_channel(uint32_t channel_idx) {
    if (channel_idx < num_channels_) {
        auto& state = states_[channel_idx];
        state.integral = 0.0f;
        state.prev_error = 0.0f;
        state.filtered_derivative = 0.0f;
        state.initialized = false;
    }
}

void FallbackPIDController::update_gains(float kp, float ki, float kd) {
    config_.kp = kp;
    config_.ki = ki;
    config_.kd = kd;
}

void FallbackPIDController::update_channel_gains(uint32_t channel_idx, float kp, float ki, float kd) {
    // For now, apply gains globally (channel-specific gains would require per-channel configs)
    update_gains(kp, ki, kd);
}

void FallbackPIDController::set_config(const PIDConfig& config) {
    config_ = config;
}

}  // namespace hardware
}  // namespace controllers
}  // namespace saguaro
