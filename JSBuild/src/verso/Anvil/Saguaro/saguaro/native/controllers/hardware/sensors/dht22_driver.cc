// src/controllers/hardware/sensors/dht22_driver.cc
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

#include "dht22_driver.h"

#include <fcntl.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

namespace hardware {
namespace sensors {

DHT22Driver::DHT22Driver(const DHT22Config& config)
    : config_(config), initialized_(false), gpio_fd_(-1) {
    last_reading_.temperature_celsius = 0.0f;
    last_reading_.humidity_percent = 0.0f;
    last_reading_.timestamp_ns = 0;
    last_reading_.valid = false;
}

DHT22Driver::~DHT22Driver() {
    if (gpio_fd_ >= 0) {
        close(gpio_fd_);
    }
}

bool DHT22Driver::initialize() {
    if (config_.gpio_pin < 0 && config_.device_path.empty()) {
        std::cerr << "DHT22: No GPIO pin or device path specified" << std::endl;
        return false;
    }

    // If device path is provided, use it (simpler interface on some systems)
    if (!config_.device_path.empty()) {
        gpio_fd_ = open(config_.device_path.c_str(), O_RDWR);
        if (gpio_fd_ < 0) {
            std::cerr << "DHT22: Failed to open device " << config_.device_path << std::endl;
            return false;
        }
    } else {
        // Direct GPIO access (requires root or GPIO permissions)
        // For production, use libgpiod or wiringPi
        // Here we provide a basic interface
        std::cerr << "DHT22: GPIO pin " << config_.gpio_pin << " configured" << std::endl;
    }

    initialized_ = true;
    return true;
}

DHT22Reading DHT22Driver::read() {
    if (!initialized_) {
        DHT22Reading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    // Enforce minimum 2-second interval between readings (DHT22 limitation)
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_read_time_).count();

    if (elapsed < 2000 && last_reading_.valid) {
        // Return cached reading if too soon
        return last_reading_;
    }

    // Retry loop for robustness
    for (int attempt = 0; attempt < config_.retry_count; ++attempt) {
        uint8_t data[5] = {0};

        // Send start signal
        send_start_signal();

        // Wait for response
        if (!wait_for_response()) {
            if (attempt < config_.retry_count - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config_.retry_delay_ms));
                continue;
            }
            DHT22Reading error_reading;
            error_reading.valid = false;
            return error_reading;
        }

        // Read 40-bit data packet
        if (!read_data_packet(data)) {
            if (attempt < config_.retry_count - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config_.retry_delay_ms));
                continue;
            }
            DHT22Reading error_reading;
            error_reading.valid = false;
            return error_reading;
        }

        // Verify checksum
        if (!verify_checksum(data)) {
            if (attempt < config_.retry_count - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config_.retry_delay_ms));
                continue;
            }
            DHT22Reading error_reading;
            error_reading.valid = false;
            return error_reading;
        }

        // Parse data
        float temp, humidity;
        if (!parse_data(data, &temp, &humidity)) {
            if (attempt < config_.retry_count - 1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(config_.retry_delay_ms));
                continue;
            }
            DHT22Reading error_reading;
            error_reading.valid = false;
            return error_reading;
        }

        // Success - create reading
        DHT22Reading reading;
        reading.temperature_celsius = temp;
        reading.humidity_percent = humidity;
        reading.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
        reading.valid = true;

        // Apply calibration
        apply_calibration(&reading);

        // Convert to Fahrenheit if requested
        if (config_.use_fahrenheit) {
            reading.temperature_celsius = reading.temperature_celsius * 9.0f / 5.0f + 32.0f;
        }

        // Cache successful reading
        last_reading_ = reading;
        last_read_time_ = now;

        return reading;
    }

    // All retries failed
    DHT22Reading error_reading;
    error_reading.valid = false;
    return error_reading;
}

void DHT22Driver::reset() {
    // Power cycle if supported (typically requires external relay)
    // For now, just reinitialize
    if (gpio_fd_ >= 0) {
        close(gpio_fd_);
        gpio_fd_ = -1;
    }
    initialized_ = false;
    initialize();
}

void DHT22Driver::send_start_signal() {
    // DHT22 protocol: Pull data line low for at least 18ms
    set_gpio_mode(config_.gpio_pin, true);  // Output mode
    write_gpio(config_.gpio_pin, false);    // Pull low
    delay_microseconds(18000);              // 18ms
    write_gpio(config_.gpio_pin, true);     // Release (pull-up will raise it)
    delay_microseconds(40);                 // Wait 40μs
    set_gpio_mode(config_.gpio_pin, false); // Switch to input mode
}

bool DHT22Driver::wait_for_response() {
    // DHT22 response: 80μs low, then 80μs high
    int timeout = 1000;  // 1ms timeout

    // Wait for low pulse
    while (read_gpio(config_.gpio_pin) && timeout-- > 0) {
        delay_microseconds(1);
    }
    if (timeout <= 0) return false;

    // Wait for high pulse
    timeout = 1000;
    while (!read_gpio(config_.gpio_pin) && timeout-- > 0) {
        delay_microseconds(1);
    }
    if (timeout <= 0) return false;

    // Wait for end of high pulse
    timeout = 1000;
    while (read_gpio(config_.gpio_pin) && timeout-- > 0) {
        delay_microseconds(1);
    }
    if (timeout <= 0) return false;

    return true;
}

bool DHT22Driver::read_data_packet(uint8_t* data) {
    // Read 40 bits (5 bytes)
    for (int i = 0; i < 40; ++i) {
        int bit = read_bit();
        if (bit < 0) return false;

        // Pack bits into bytes (MSB first)
        data[i / 8] |= (bit << (7 - (i % 8)));
    }

    return true;
}

int DHT22Driver::read_bit() {
    // DHT22 bit encoding:
    // - All bits start with 50μs low pulse
    // - 0: followed by 26-28μs high pulse
    // - 1: followed by 70μs high pulse

    int timeout = 1000;

    // Wait for low pulse start
    while (read_gpio(config_.gpio_pin) && timeout-- > 0) {
        delay_microseconds(1);
    }
    if (timeout <= 0) return -1;

    // Wait for high pulse start
    timeout = 1000;
    while (!read_gpio(config_.gpio_pin) && timeout-- > 0) {
        delay_microseconds(1);
    }
    if (timeout <= 0) return -1;

    // Measure high pulse duration
    int high_duration = 0;
    timeout = 200;  // 200μs max
    while (read_gpio(config_.gpio_pin) && timeout-- > 0) {
        delay_microseconds(1);
        high_duration++;
    }

    // Threshold: >40μs = 1, <40μs = 0
    return (high_duration > 40) ? 1 : 0;
}

bool DHT22Driver::verify_checksum(const uint8_t* data) {
    // Checksum = sum of first 4 bytes
    uint8_t checksum = data[0] + data[1] + data[2] + data[3];
    return (checksum == data[4]);
}

bool DHT22Driver::parse_data(const uint8_t* data, float* temp, float* humidity) {
    // Humidity: 16 bits, MSB first (data[0] = high byte, data[1] = low byte)
    // Scale: 0.1% RH per count
    uint16_t raw_humidity = (static_cast<uint16_t>(data[0]) << 8) | data[1];
    *humidity = static_cast<float>(raw_humidity) / 10.0f;

    // Temperature: 16 bits, MSB first (data[2] = high byte, data[3] = low byte)
    // Scale: 0.1°C per count
    // Bit 15 = sign (1 = negative)
    uint16_t raw_temp = (static_cast<uint16_t>(data[2] & 0x7F) << 8) | data[3];
    *temp = static_cast<float>(raw_temp) / 10.0f;

    // Apply sign
    if (data[2] & 0x80) {
        *temp = -*temp;
    }

    // Sanity checks
    if (*humidity < 0.0f || *humidity > 100.0f) return false;
    if (*temp < -40.0f || *temp > 80.0f) return false;

    return true;
}

void DHT22Driver::apply_calibration(DHT22Reading* reading) {
    reading->temperature_celsius += config_.temperature_offset;
    reading->humidity_percent += config_.humidity_offset;

    // Clamp to valid ranges
    reading->humidity_percent = std::max(0.0f, std::min(100.0f, reading->humidity_percent));
    reading->temperature_celsius = std::max(-40.0f, std::min(80.0f, reading->temperature_celsius));
}

void DHT22Driver::delay_microseconds(int microseconds) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
}

void DHT22Driver::set_gpio_mode(int pin, bool output) {
    // Platform-specific GPIO mode setting
    // For production, use libgpiod or wiringPi
    // This is a placeholder for the interface
    (void)pin;
    (void)output;
}

void DHT22Driver::write_gpio(int pin, bool value) {
    // Platform-specific GPIO write
    // For production, use libgpiod or wiringPi
    // This is a placeholder for the interface
    (void)pin;
    (void)value;
}

bool DHT22Driver::read_gpio(int pin) {
    // Platform-specific GPIO read
    // For production, use libgpiod or wiringPi
    // This is a placeholder that returns simulated data
    (void)pin;
    return (rand() % 2) == 0;  // Placeholder: random bit
}

}  // namespace sensors
}  // namespace hardware
