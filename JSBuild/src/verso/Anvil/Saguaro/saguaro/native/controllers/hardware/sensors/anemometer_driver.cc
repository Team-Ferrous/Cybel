// src/controllers/hardware/sensors/anemometer_driver.cc
// Copyright 2025 Verso Industries

#include "anemometer_driver.h"
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <cmath>

namespace hardware {
namespace sensors {

AnemometerDriver::AnemometerDriver(const AnemometerConfig& config)
    : config_(config), initialized_(false), adc_fd_(-1) {
    last_reading_.velocity_ms = 0.0f;
    last_reading_.direction_deg = 0.0f;
    last_reading_.timestamp_ns = 0;
    last_reading_.valid = false;
}

bool AnemometerDriver::initialize() {
    // Placeholder: Open ADC device (platform-specific)
    initialized_ = true;
    return true;
}

AnemometerReading AnemometerDriver::read() {
    if (!initialized_) {
        AnemometerReading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    float voltage = read_adc_voltage();
    float velocity = voltage_to_velocity(voltage);

    AnemometerReading reading;
    reading.velocity_ms = velocity;
    reading.direction_deg = 0.0f;  // Optional, set to 0 if not available
    reading.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    reading.valid = true;

    last_reading_ = reading;
    return reading;
}

float AnemometerDriver::read_adc_voltage() {
    // Placeholder: Read from ADC (platform-specific)
    return 1.5f;  // Simulated voltage
}

float AnemometerDriver::voltage_to_velocity(float voltage) {
    // Typical calibration: 0-5V maps to 0-40 m/s
    float velocity = (voltage / 5.0f) * config_.max_velocity_ms;
    velocity = velocity * config_.calibration_scale + config_.calibration_offset;
    return std::max(config_.min_velocity_ms, std::min(config_.max_velocity_ms, velocity));
}

}  // namespace sensors
}  // namespace hardware
