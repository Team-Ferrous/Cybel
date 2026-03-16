// src/controllers/hardware/sensors/ina219_driver.h
// Phase 4.5 Task 4.5.1.4: INA219 voltage/current sensor driver
// Copyright 2025 Verso Industries

#ifndef HARDWARE_SENSORS_INA219_DRIVER_H_
#define HARDWARE_SENSORS_INA219_DRIVER_H_

#include <cstdint>
#include <string>

namespace hardware {
namespace sensors {

struct INA219Reading {
    float bus_voltage_v;      // Bus voltage in V
    float shunt_voltage_mv;   // Shunt voltage in mV
    float current_ma;         // Current in mA
    float power_mw;           // Power in mW
    uint64_t timestamp_ns;
    bool valid;
};

struct INA219Config {
    std::string i2c_bus = "/dev/i2c-1";
    uint8_t i2c_address = 0x40;  // Default INA219 address
    float shunt_resistance_ohms = 0.1f;  // Shunt resistor value
    float max_current_a = 3.2f;          // Maximum expected current
    bool calibrate_on_init = true;
};

/**
 * INA219 high-side DC current/voltage sensor driver.
 * Roadmap: Phase 4.5, Task 4.5.1.4
 */
class INA219Driver {
public:
    explicit INA219Driver(const INA219Config& config);
    ~INA219Driver();
    bool initialize();
    INA219Reading read();
    bool is_ready() const { return initialized_; }
    void calibrate();

private:
    bool read_register(uint8_t reg_addr, uint16_t* data);
    bool write_register(uint8_t reg_addr, uint16_t data);

    INA219Config config_;
    bool initialized_;
    INA219Reading last_reading_;
    int i2c_fd_;
    uint16_t calibration_value_;
};

}  // namespace sensors
}  // namespace hardware

#endif  // HARDWARE_SENSORS_INA219_DRIVER_H_
