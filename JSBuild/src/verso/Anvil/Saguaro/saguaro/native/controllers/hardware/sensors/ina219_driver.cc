// src/controllers/hardware/sensors/ina219_driver.cc
// Copyright 2025 Verso Industries

#include "ina219_driver.h"
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <chrono>
#include <cmath>

namespace hardware {
namespace sensors {

// INA219 Register addresses
constexpr uint8_t INA219_REG_CONFIG = 0x00;
constexpr uint8_t INA219_REG_SHUNT_VOLTAGE = 0x01;
constexpr uint8_t INA219_REG_BUS_VOLTAGE = 0x02;
constexpr uint8_t INA219_REG_POWER = 0x03;
constexpr uint8_t INA219_REG_CURRENT = 0x04;
constexpr uint8_t INA219_REG_CALIBRATION = 0x05;

INA219Driver::INA219Driver(const INA219Config& config)
    : config_(config), initialized_(false), i2c_fd_(-1), calibration_value_(0) {
    last_reading_.bus_voltage_v = 0.0f;
    last_reading_.shunt_voltage_mv = 0.0f;
    last_reading_.current_ma = 0.0f;
    last_reading_.power_mw = 0.0f;
    last_reading_.timestamp_ns = 0;
    last_reading_.valid = false;
}

INA219Driver::~INA219Driver() {
    if (i2c_fd_ >= 0) {
        close(i2c_fd_);
    }
}

bool INA219Driver::initialize() {
    i2c_fd_ = open(config_.i2c_bus.c_str(), O_RDWR);
    if (i2c_fd_ < 0) return false;

    if (ioctl(i2c_fd_, I2C_SLAVE, config_.i2c_address) < 0) {
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }

    // Configure: 32V range, ±320mV shunt range, 12-bit resolution, continuous mode
    uint16_t config = 0x399F;  // Standard configuration
    if (!write_register(INA219_REG_CONFIG, config)) {
        close(i2c_fd_);
        i2c_fd_ = -1;
        return false;
    }

    if (config_.calibrate_on_init) {
        calibrate();
    }

    initialized_ = true;
    return true;
}

void INA219Driver::calibrate() {
    // Calibration formula: Cal = trunc(0.04096 / (Current_LSB * Rshunt))
    // Current_LSB = Max_Expected_Current / 32768
    float current_lsb = config_.max_current_a / 32768.0f;
    calibration_value_ = static_cast<uint16_t>(
        0.04096f / (current_lsb * config_.shunt_resistance_ohms));
    write_register(INA219_REG_CALIBRATION, calibration_value_);
}

INA219Reading INA219Driver::read() {
    if (!initialized_) {
        INA219Reading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    uint16_t shunt_raw, bus_raw, power_raw, current_raw;

    if (!read_register(INA219_REG_SHUNT_VOLTAGE, &shunt_raw) ||
        !read_register(INA219_REG_BUS_VOLTAGE, &bus_raw) ||
        !read_register(INA219_REG_POWER, &power_raw) ||
        !read_register(INA219_REG_CURRENT, &current_raw)) {
        INA219Reading error_reading;
        error_reading.valid = false;
        return error_reading;
    }

    INA219Reading reading;
    // Shunt voltage: 10μV per LSB
    reading.shunt_voltage_mv = static_cast<int16_t>(shunt_raw) * 0.01f;

    // Bus voltage: 4mV per LSB (shift right 3 bits, mask status bits)
    reading.bus_voltage_v = ((bus_raw >> 3) * 4) / 1000.0f;

    // Current: depends on calibration
    float current_lsb = config_.max_current_a / 32768.0f;
    reading.current_ma = static_cast<int16_t>(current_raw) * current_lsb * 1000.0f;

    // Power: 20x current LSB
    reading.power_mw = power_raw * current_lsb * 20.0f * 1000.0f;

    reading.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    reading.valid = true;

    last_reading_ = reading;
    return reading;
}

bool INA219Driver::read_register(uint8_t reg_addr, uint16_t* data) {
    if (write(i2c_fd_, &reg_addr, 1) != 1) return false;
    uint8_t buf[2];
    if (read(i2c_fd_, buf, 2) != 2) return false;
    *data = (buf[0] << 8) | buf[1];  // Big-endian
    return true;
}

bool INA219Driver::write_register(uint8_t reg_addr, uint16_t data) {
    uint8_t buf[3] = {reg_addr, static_cast<uint8_t>(data >> 8), static_cast<uint8_t>(data & 0xFF)};
    return (write(i2c_fd_, buf, 3) == 3);
}

}  // namespace sensors
}  // namespace hardware
