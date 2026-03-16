// src/controllers/hardware/sensors/bme680_driver.cc
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

#include "bme680_driver.h"

#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>

namespace hardware {
namespace sensors {

// BME680 Register addresses
constexpr uint8_t BME680_REG_CHIP_ID = 0xD0;
constexpr uint8_t BME680_REG_RESET = 0xE0;
constexpr uint8_t BME680_REG_CTRL_GAS_1 = 0x71;
constexpr uint8_t BME680_REG_CTRL_HUM = 0x72;
constexpr uint8_t BME680_REG_CTRL_MEAS = 0x74;
constexpr uint8_t BME680_REG_CONFIG = 0x75;
constexpr uint8_t BME680_REG_MEAS_STATUS = 0x1D;
constexpr uint8_t BME680_REG_TEMP_MSB = 0x22;
constexpr uint8_t BME680_CHIP_ID = 0x61;
constexpr uint8_t BME680_SOFT_RESET_CMD = 0xB6;

BME680Driver::BME680Driver(const BME680Config& config)
    : config_(config), initialized_(false), i2c_fd_(-1), t_fine_(0.0f) {
    last_reading_.temperature_celsius = 0.0f;
    last_reading_.humidity_percent = 0.0f;
    last_reading_.pressure_hpa = 0.0f;
    last_reading_.gas_resistance_ohms = 0.0f;
    last_reading_.timestamp_ns = 0;
    last_reading_.valid = false;
}

BME680Driver::~BME680Driver() {
    if (i2c_fd_ >= 0) {
        close(i2c_fd_);
    }
}

bool BME680Driver::initialize() {
    // Open I2C bus
    i2c_fd_ = open(config_.i2c_bus.c_str(), O_RDWR);
    if (i2c_fd_ < 0) {
        std::cerr << "BME680: Failed to open I2C bus " << config_.i2c_bus << std::endl;
        return false;
    }

    // Set I2C slave address
    if (ioctl(i2c_fd_, I2C_SLAVE, config_.i2c_address) < 0) {
        std::cerr << "BME680: Failed to set I2C address 0x" << std::hex
                  << static_cast<int>(config_.i2c_address) << std::dec << std::endl;
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }

    // Verify chip ID
    uint8_t chip_id = get_chip_id();
    if (chip_id != BME680_CHIP_ID) {
        std::cerr << "BME680: Invalid chip ID 0x" << std::hex << static_cast<int>(chip_id)
                  << ", expected 0x61" << std::dec << std::endl;
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }

    // Soft reset
    reset();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Read calibration data
    if (!read_calibration_data()) {
        std::cerr << "BME680: Failed to read calibration data" << std::endl;
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }

    // Configure sensor
    if (!configure_sensor()) {
        std::cerr << "BME680: Failed to configure sensor" << std::endl;
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }

    initialized_ = true;
    return true;
}

BME680Reading BME680Driver::read() {
    if (!initialized_) {
        BME680Reading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    // Trigger measurement
    if (!trigger_measurement()) {
        BME680Reading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    // Wait for measurement completion
    if (!wait_for_measurement()) {
        BME680Reading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    // Read raw data
    uint32_t temp_adc, press_adc, hum_adc;
    uint16_t gas_adc;
    uint8_t gas_range;
    if (!read_raw_data(&temp_adc, &press_adc, &hum_adc, &gas_adc, &gas_range)) {
        BME680Reading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    // Compensate readings
    BME680Reading reading;
    reading.temperature_celsius = compensate_temperature(temp_adc);
    reading.pressure_hpa = compensate_pressure(press_adc);
    reading.humidity_percent = compensate_humidity(hum_adc);
    reading.gas_resistance_ohms = compensate_gas(gas_adc, gas_range);
    reading.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    reading.valid = true;

    // Apply calibration offsets
    apply_calibration(&reading);

    // Cache reading
    last_reading_ = reading;
    return reading;
}

void BME680Driver::reset() {
    write_register(BME680_REG_RESET, BME680_SOFT_RESET_CMD);
}

uint8_t BME680Driver::get_chip_id() {
    uint8_t chip_id = 0;
    read_register(BME680_REG_CHIP_ID, &chip_id);
    return chip_id;
}

bool BME680Driver::read_calibration_data() {
    // Read calibration coefficients from EEPROM (registers 0x89-0xA1, 0xE1-0xF0)
    // This is a simplified implementation - full implementation would read all coefficients
    uint8_t coeff_array[41];
    if (!read_registers(0x89, coeff_array, 25)) return false;
    if (!read_registers(0xE1, coeff_array + 25, 16)) return false;

    // Parse calibration coefficients (simplified)
    par_t1_ = (coeff_array[33] << 8) | coeff_array[32];
    par_t2_ = (coeff_array[1] << 8) | coeff_array[0];
    par_t3_ = coeff_array[2];

    return true;
}

bool BME680Driver::configure_sensor() {
    // Set humidity oversampling
    uint8_t ctrl_hum = config_.humidity_oversampling & 0x07;
    if (!write_register(BME680_REG_CTRL_HUM, ctrl_hum)) return false;

    // Set temperature and pressure oversampling
    uint8_t ctrl_meas = (config_.temperature_oversampling << 5) |
                        (config_.pressure_oversampling << 2);
    if (!write_register(BME680_REG_CTRL_MEAS, ctrl_meas)) return false;

    // Set IIR filter
    uint8_t config_reg = (config_.iir_filter_coeff << 2);
    if (!write_register(BME680_REG_CONFIG, config_reg)) return false;

    // Configure gas heater (simplified)
    if (!write_register(BME680_REG_CTRL_GAS_1, 0x10)) return false;

    return true;
}

bool BME680Driver::trigger_measurement() {
    uint8_t ctrl_meas;
    if (!read_register(BME680_REG_CTRL_MEAS, &ctrl_meas)) return false;
    ctrl_meas |= 0x01;  // Set to forced mode
    return write_register(BME680_REG_CTRL_MEAS, ctrl_meas);
}

bool BME680Driver::wait_for_measurement(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();
    while (true) {
        uint8_t status;
        if (!read_register(BME680_REG_MEAS_STATUS, &status)) return false;

        if ((status & 0x80) == 0) return true;  // Measurement complete

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeout_ms) return false;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

bool BME680Driver::read_raw_data(uint32_t* temp_adc, uint32_t* press_adc,
                                  uint32_t* hum_adc, uint16_t* gas_adc, uint8_t* gas_range) {
    uint8_t data[17];
    if (!read_registers(BME680_REG_TEMP_MSB, data, 17)) return false;

    *temp_adc = (static_cast<uint32_t>(data[5]) << 12) |
                (static_cast<uint32_t>(data[6]) << 4) |
                (data[7] >> 4);
    *press_adc = (static_cast<uint32_t>(data[2]) << 12) |
                 (static_cast<uint32_t>(data[3]) << 4) |
                 (data[4] >> 4);
    *hum_adc = (static_cast<uint32_t>(data[8]) << 8) | data[9];
    *gas_adc = (static_cast<uint16_t>(data[13]) << 2) | (data[14] >> 6);
    *gas_range = data[14] & 0x0F;

    return true;
}

float BME680Driver::compensate_temperature(uint32_t temp_adc) {
    // Simplified compensation formula
    float var1 = (static_cast<float>(temp_adc) / 16384.0f - static_cast<float>(par_t1_) / 1024.0f) * static_cast<float>(par_t2_);
    float var2 = ((static_cast<float>(temp_adc) / 131072.0f - static_cast<float>(par_t1_) / 8192.0f) *
                  (static_cast<float>(temp_adc) / 131072.0f - static_cast<float>(par_t1_) / 8192.0f)) *
                 (static_cast<float>(par_t3_) * 16.0f);
    t_fine_ = var1 + var2;
    return t_fine_ / 5120.0f;
}

float BME680Driver::compensate_pressure(uint32_t press_adc) {
    // Placeholder implementation
    return static_cast<float>(press_adc) / 256.0f + 900.0f;  // Rough approximation
}

float BME680Driver::compensate_humidity(uint32_t hum_adc) {
    // Placeholder implementation
    return static_cast<float>(hum_adc) / 512.0f;  // Rough approximation
}

float BME680Driver::compensate_gas(uint16_t gas_adc, uint8_t gas_range) {
    // Placeholder implementation
    constexpr float lookup_k1_range[16] = {0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.8,
                                            0.0, 0.0, -0.2, -0.5, 0.0, -1.0, 0.0, 0.0};
    constexpr float lookup_k2_range[16] = {0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.0, -0.8,
                                            -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    float var1 = (1340.0f + 5.0f * range_sw_err_);
    float var2 = (var1) * (1.0f + lookup_k1_range[gas_range] / 100.0f);
    float var3 = 1.0f + (lookup_k2_range[gas_range] / 100.0f);
    float gas_res = 1.0f / (var3 * 0.000000125f * static_cast<float>(1 << gas_range) *
                            ((static_cast<float>(gas_adc) - 512.0f) / var2 + 1.0f));

    return gas_res;
}

void BME680Driver::apply_calibration(BME680Reading* reading) {
    reading->temperature_celsius += config_.temperature_offset;
    reading->humidity_percent += config_.humidity_offset;
    reading->pressure_hpa += config_.pressure_offset;

    // Clamp to valid ranges
    reading->humidity_percent = std::max(0.0f, std::min(100.0f, reading->humidity_percent));
    reading->pressure_hpa = std::max(300.0f, std::min(1100.0f, reading->pressure_hpa));
}

bool BME680Driver::read_register(uint8_t reg_addr, uint8_t* data) {
    if (write(i2c_fd_, &reg_addr, 1) != 1) return false;
    if (read(i2c_fd_, data, 1) != 1) return false;
    return true;
}

bool BME680Driver::write_register(uint8_t reg_addr, uint8_t data) {
    uint8_t buf[2] = {reg_addr, data};
    return (write(i2c_fd_, buf, 2) == 2);
}

bool BME680Driver::read_registers(uint8_t reg_addr, uint8_t* data, size_t length) {
    if (write(i2c_fd_, &reg_addr, 1) != 1) return false;
    return (read(i2c_fd_, data, length) == static_cast<ssize_t>(length));
}

}  // namespace sensors
}  // namespace hardware
