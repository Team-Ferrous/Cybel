// src/controllers/hardware/sensors/dht22_driver.h
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

#ifndef HARDWARE_SENSORS_DHT22_DRIVER_H_
#define HARDWARE_SENSORS_DHT22_DRIVER_H_

#include <chrono>
#include <cstdint>
#include <string>

namespace hardware {
namespace sensors {

/**
 * DHT22 temperature and humidity sensor reading.
 */
struct DHT22Reading {
    float temperature_celsius;    // Temperature in °C (-40 to 80°C)
    float humidity_percent;       // Relative humidity in % (0-100%)
    uint64_t timestamp_ns;        // Nanosecond timestamp
    bool valid;                   // True if reading is valid
};

/**
 * DHT22 sensor configuration.
 */
struct DHT22Config {
    int gpio_pin = -1;                    // GPIO pin number for single-wire interface
    std::string device_path = "";         // Device path (e.g., "/dev/dht22")
    int retry_count = 3;                  // Number of retries on read failure
    int retry_delay_ms = 100;             // Delay between retries (ms)
    float temperature_offset = 0.0f;      // Calibration offset for temperature (°C)
    float humidity_offset = 0.0f;         // Calibration offset for humidity (%)
    bool use_fahrenheit = false;          // Convert to Fahrenheit if true
};

/**
 * DHT22 Temperature and Humidity Sensor Driver.
 *
 * Interfaces with DHT22 digital sensor via single-wire protocol.
 *
 * Specifications:
 * - Temperature range: -40 to 80°C
 * - Humidity range: 0-100% RH
 * - Accuracy: ±0.5°C, ±2-5% RH
 * - Resolution: 0.1°C, 0.1% RH
 * - Response time: <2 seconds
 * - Protocol: Single-wire digital (proprietary timing)
 *
 * Hardware Interface:
 * - DHT22 uses a proprietary single-wire protocol with precise timing
 * - Data pin requires 10kΩ pull-up resistor
 * - Communication sequence: Start signal (18ms low) → Wait for response → Read 40 bits
 * - Bit encoding: 26-28μs low pulse, followed by 26-28μs (0) or 70μs (1) high pulse
 *
 * Usage:
 *   DHT22Config config;
 *   config.gpio_pin = 4;  // GPIO4 on Raspberry Pi
 *   DHT22Driver sensor(config);
 *
 *   if (sensor.initialize()) {
 *       DHT22Reading reading = sensor.read();
 *       if (reading.valid) {
 *           printf("Temp: %.1f°C, Humidity: %.1f%%\n",
 *                  reading.temperature_celsius, reading.humidity_percent);
 *       }
 *   }
 *
 * Roadmap: docs/roadmaps/SAGUARO_EHD_Architecture_Roadmap.md (Phase 4.5, Task 4.5.1.1)
 */
class DHT22Driver {
public:
    /**
     * Construct DHT22 driver with configuration.
     *
     * @param config DHT22 configuration
     */
    explicit DHT22Driver(const DHT22Config& config);
    ~DHT22Driver();

    /**
     * Initialize sensor hardware interface.
     *
     * @return True if initialization succeeds
     */
    bool initialize();

    /**
     * Read temperature and humidity from sensor.
     *
     * Blocks for ~2 seconds during sensor response time.
     * Automatically retries on checksum failure.
     *
     * @return DHT22Reading with temperature, humidity, and validity flag
     */
    DHT22Reading read();

    /**
     * Check if sensor is initialized and ready.
     *
     * @return True if sensor is ready
     */
    bool is_ready() const { return initialized_; }

    /**
     * Get last successful reading (cached).
     *
     * @return Last valid DHT22Reading
     */
    DHT22Reading get_last_reading() const { return last_reading_; }

    /**
     * Get sensor configuration.
     *
     * @return Current DHT22Config
     */
    DHT22Config get_config() const { return config_; }

    /**
     * Reset sensor (power cycle if supported).
     */
    void reset();

private:
    /**
     * Send start signal to DHT22 (18ms low pulse).
     */
    void send_start_signal();

    /**
     * Wait for DHT22 response signal.
     *
     * @return True if response detected
     */
    bool wait_for_response();

    /**
     * Read 40-bit data packet from DHT22.
     *
     * @param data Output buffer for 5 bytes (40 bits)
     * @return True if read succeeds
     */
    bool read_data_packet(uint8_t* data);

    /**
     * Read a single bit from DHT22 using timing protocol.
     *
     * @return 0 or 1, or -1 on error
     */
    int read_bit();

    /**
     * Verify checksum of 5-byte data packet.
     *
     * @param data 5-byte packet (bytes 0-3 = data, byte 4 = checksum)
     * @return True if checksum matches
     */
    bool verify_checksum(const uint8_t* data);

    /**
     * Parse raw data bytes into temperature and humidity.
     *
     * @param data 5-byte data packet
     * @param temp Output temperature (°C)
     * @param humidity Output relative humidity (%)
     * @return True if parsing succeeds
     */
    bool parse_data(const uint8_t* data, float* temp, float* humidity);

    /**
     * Apply calibration offsets to sensor reading.
     *
     * @param reading DHT22Reading to calibrate (modified in-place)
     */
    void apply_calibration(DHT22Reading* reading);

    /**
     * Wait for specified duration in microseconds.
     *
     * @param microseconds Duration to wait
     */
    void delay_microseconds(int microseconds);

    /**
     * Set GPIO pin mode (input/output).
     *
     * @param pin GPIO pin number
     * @param output True for output mode, false for input
     */
    void set_gpio_mode(int pin, bool output);

    /**
     * Write to GPIO pin.
     *
     * @param pin GPIO pin number
     * @param value True for high, false for low
     */
    void write_gpio(int pin, bool value);

    /**
     * Read from GPIO pin.
     *
     * @param pin GPIO pin number
     * @return True if high, false if low
     */
    bool read_gpio(int pin);

    DHT22Config config_;
    bool initialized_;
    DHT22Reading last_reading_;
    std::chrono::steady_clock::time_point last_read_time_;
    int gpio_fd_;  // File descriptor for GPIO access
};

}  // namespace sensors
}  // namespace hardware

#endif  // HARDWARE_SENSORS_DHT22_DRIVER_H_
