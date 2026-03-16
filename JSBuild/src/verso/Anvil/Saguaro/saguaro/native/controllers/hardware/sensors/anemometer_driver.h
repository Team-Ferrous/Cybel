// src/controllers/hardware/sensors/anemometer_driver.h
// Phase 4.5 Task 4.5.1.3: Anemometer (air velocity) sensor driver
// Copyright 2025 Verso Industries

#ifndef HARDWARE_SENSORS_ANEMOMETER_DRIVER_H_
#define HARDWARE_SENSORS_ANEMOMETER_DRIVER_H_

#include <cstdint>

namespace hardware {
namespace sensors {

struct AnemometerReading {
    float velocity_ms;      // Air velocity in m/s
    float direction_deg;    // Wind direction in degrees (0-360, optional)
    uint64_t timestamp_ns;
    bool valid;
};

struct AnemometerConfig {
    int adc_channel = 0;           // ADC channel for analog output
    float calibration_offset = 0.0f;
    float calibration_scale = 1.0f;
    float min_velocity_ms = 0.0f;
    float max_velocity_ms = 40.0f;
};

/**
 * Hot-wire or ultrasonic anemometer driver.
 * Roadmap: Phase 4.5, Task 4.5.1.3
 */
class AnemometerDriver {
public:
    explicit AnemometerDriver(const AnemometerConfig& config);
    bool initialize();
    AnemometerReading read();
    bool is_ready() const { return initialized_; }

private:
    float read_adc_voltage();
    float voltage_to_velocity(float voltage);

    AnemometerConfig config_;
    bool initialized_;
    AnemometerReading last_reading_;
    int adc_fd_;
};

}  // namespace sensors
}  // namespace hardware

#endif  // HARDWARE_SENSORS_ANEMOMETER_DRIVER_H_
