// src/controllers/maglev_controller.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/controllers/maglev_controller.h"
#include <algorithm>
#include <cmath>
#include <limits>

MaglevController::MaglevController()
    : integral_(0.0f),
      prev_error_(0.0f),
      filtered_derivative_(0.0f),
      last_output_(0.0f),
      settling_start_time_(-1.0f),
      current_time_(0.0f),
      is_settling_(false) {
    // Default initialization, init() must be called before use
}

void MaglevController::init(const Config& config) {
    config_ = config;
    reset();
}

float MaglevController::compute(
    const std::vector<float>& x_ref,
    const std::vector<float>& x_measured,
    float u_ref,
    float hnn_feedforward
) {
    // Validate input dimensions
    if (x_ref.size() < 3 || x_measured.size() < 3) {
        // Insufficient state (need position, velocity, acceleration)
        return 0.0f;
    }

    // Extract position error (primary feedback variable)
    float position_ref = x_ref[0];
    float position_measured = x_measured[0];
    float error = position_ref - position_measured;

    // Update time (assume fixed sample rate)
    float dt = 1.0f / config_.sample_rate_hz;
    current_time_ += dt;

    // ===== Proportional term =====
    float u_p = config_.kp * error;

    // ===== Integral term (with anti-windup) =====
    integral_ += error * dt;
    integral_ = applyAntiWindup(integral_);
    float u_i = config_.ki * integral_;

    // ===== Derivative term (with low-pass filter) =====
    // Derivative of error: de/dt ≈ (e[n] - e[n-1]) / dt
    float derivative = (error - prev_error_) / dt;

    // Low-pass filter to reduce noise amplification
    // Filtered_derivative[n] = α * derivative + (1-α) * filtered_derivative[n-1]
    float alpha = config_.derivative_filter_coeff;
    filtered_derivative_ = alpha * derivative + (1.0f - alpha) * filtered_derivative_;

    float u_d = config_.kd * filtered_derivative_;

    // ===== Feedback control =====
    float u_feedback = u_p + u_i + u_d;

    // ===== Feedforward control (HNN-predicted) =====
    float u_feedforward = 0.0f;
    if (config_.enable_feedforward) {
        // Combine nominal reference and HNN prediction
        u_feedforward = u_ref + config_.feedforward_gain * hnn_feedforward;
    }

    // ===== Total control output =====
    float u_total = u_feedforward + u_feedback;

    // Clamp to physical actuator limits
    u_total = clampOutput(u_total);

    // Update state for next iteration
    prev_error_ = error;
    last_output_ = u_total;

    // ===== Settling time tracking =====
    // Consider "settled" when |error| < threshold for continuous period
    float settling_threshold = 5.0f;  // 5 nm
    if (std::abs(error * 1e9f) < settling_threshold) {  // Convert to nm
        if (!is_settling_) {
            // Just entered settling region
            is_settling_ = true;
            settling_start_time_ = current_time_;
        }
    } else {
        // Exited settling region, reset
        is_settling_ = false;
        settling_start_time_ = -1.0f;
    }

    return u_total;
}

void MaglevController::reset() {
    integral_ = 0.0f;
    prev_error_ = 0.0f;
    filtered_derivative_ = 0.0f;
    last_output_ = 0.0f;
    settling_start_time_ = -1.0f;
    current_time_ = 0.0f;
    is_settling_ = false;
}

float MaglevController::getIntegrator() const {
    return integral_;
}

float MaglevController::getLastOutput() const {
    return last_output_;
}

float MaglevController::getSettlingTime(float threshold) const {
    if (is_settling_ && settling_start_time_ >= 0.0f) {
        return current_time_ - settling_start_time_;
    }
    return -1.0f;  // Not settled
}

float MaglevController::applyAntiWindup(float value) const {
    return std::max(config_.integrator_min, std::min(config_.integrator_max, value));
}

float MaglevController::clampOutput(float value) const {
    return std::max(config_.output_min, std::min(config_.output_max, value));
}
