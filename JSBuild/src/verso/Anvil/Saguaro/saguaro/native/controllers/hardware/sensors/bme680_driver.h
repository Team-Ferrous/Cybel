// src/controllers/hardware/sensors/bme680_driver.h
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

#ifndef HARDWARE_SENSORS_BME680_DRIVER_H_
#define HARDWARE_SENSORS_BME680_DRIVER_H_

#include <chrono>
#include <cstdint>
#include <string>

namespace hardware {
namespace sensors {

/**
 * BME680 multi-sensor reading.
 */
struct BME680Reading {
    float temperature_celsius;    // Temperature in °C
    float humidity_percent;       // Relative humidity in %
    float pressure_hpa;           // Atmospheric pressure in hPa
    float gas_resistance_ohms;    // Gas sensor resistance in Ω (VOC indicator)
    uint64_t timestamp_ns;        // Nanosecond timestamp
    bool valid;                   // True if reading is valid
};

/**
 * BME680 sensor configuration.
 */
struct BME680Config {
    std::string i2c_bus = "/dev/i2c-1";  // I2C bus device path
    uint8_t i2c_address = 0x77;          // I2C address (0x76 or 0x77)

    // Temperature oversampling (0-5: skip, 1×, 2×, 4×, 8×, 16×)
    uint8_t temperature_oversampling = 2;

    // Humidity oversampling (0-5: skip, 1×, 2×, 4×, 8×, 16×)
    uint8_t humidity_oversampling = 2;

    // Pressure oversampling (0-5: skip, 1×, 2×, 4×, 8×, 16×)
    uint8_t pressure_oversampling = 2;

    // IIR filter coefficient (0-7: off, 1, 3, 7, 15, 31, 63, 127)
    uint8_t iir_filter_coeff = 3;

    // Gas heater temperature (°C, 200-400°C typical)
    uint16_t gas_heater_temp_celsius = 320;

    // Gas heater duration (ms, 1-4032ms)
    uint16_t gas_heater_duration_ms = 150;

    // Calibration offsets
    float temperature_offset = 0.0f;
    float humidity_offset = 0.0f;
    float pressure_offset = 0.0f;
};

/**
 * BME680 Environmental Sensor Driver.
 *
 * Interfaces with BME680 4-in-1 sensor via I2C.
 *
 * Specifications:
 * - Temperature: -40 to 85°C, ±1°C accuracy
 * - Humidity: 0-100% RH, ±3% RH accuracy
 * - Pressure: 300-1100 hPa, ±1 hPa accuracy
 * - Gas: VOC detection (resistance 0.1-10 MΩ)
 * - Response time: <1 second (gas: up to 30 seconds for stable reading)
 *
 * I2C Protocol:
 * - Standard/Fast mode (100kHz / 400kHz)
 * - 7-bit address: 0x76 (SDO low) or 0x77 (SDO high)
 * - Register-based interface
 * - Supports burst reads for efficiency
 *
 * Usage:
 *   BME680Config config;
 *   config.i2c_bus = "/dev/i2c-1";
 *   config.i2c_address = 0x77;
 *   config.gas_heater_temp_celsius = 320;
 *   BME680Driver sensor(config);
 *
 *   if (sensor.initialize()) {
 *       BME680Reading reading = sensor.read();
 *       if (reading.valid) {
 *           printf("Temp: %.1f°C, Humidity: %.1f%%, Pressure: %.1f hPa, Gas: %.0f Ω\n",
 *                  reading.temperature_celsius, reading.humidity_percent,
 *                  reading.pressure_hpa, reading.gas_resistance_ohms);
 *       }
 *   }
 *
 * Roadmap: docs/roadmaps/SAGUARO_EHD_Architecture_Roadmap.md (Phase 4.5, Task 4.5.1.2)
 */
class BME680Driver {
public:
    /**
     * Construct BME680 driver with configuration.
     *
     * @param config BME680 configuration
     */
    explicit BME680Driver(const BME680Config& config);
    ~BME680Driver();

    /**
     * Initialize sensor hardware interface and calibration.
     *
     * Reads calibration coefficients from sensor EEPROM.
     *
     * @return True if initialization succeeds
     */
    bool initialize();

    /**
     * Read all sensor values (temperature, humidity, pressure, gas).
     *
     * Triggers measurement, waits for completion, reads results.
     *
     * @return BME680Reading with all sensor values and validity flag
     */
    BME680Reading read();

    /**
     * Check if sensor is initialized and ready.
     *
     * @return True if sensor is ready
     */
    bool is_ready() const { return initialized_; }

    /**
     * Get last successful reading (cached).
     *
     * @return Last valid BME680Reading
     */
    BME680Reading get_last_reading() const { return last_reading_; }

    /**
     * Get sensor configuration.
     *
     * @return Current BME680Config
     */
    BME680Config get_config() const { return config_; }

    /**
     * Reset sensor (soft reset via register).
     */
    void reset();

    /**
     * Get sensor chip ID (should be 0x61 for BME680).
     *
     * @return Chip ID from register 0xD0
     */
    uint8_t get_chip_id();

private:
    /**
     * Read calibration coefficients from sensor EEPROM.
     *
     * @return True if calibration read succeeds
     */
    bool read_calibration_data();

    /**
     * Configure sensor with oversampling, IIR filter, and gas heater settings.
     *
     * @return True if configuration succeeds
     */
    bool configure_sensor();

    /**
     * Trigger a forced mode measurement.
     *
     * @return True if trigger succeeds
     */
    bool trigger_measurement();

    /**
     * Wait for measurement to complete.
     *
     * @param timeout_ms Maximum wait time in milliseconds
     * @return True if measurement completes within timeout
     */
    bool wait_for_measurement(int timeout_ms = 1000);

    /**
     * Read raw ADC data from sensor registers.
     *
     * @param temp_adc Output raw temperature ADC
     * @param press_adc Output raw pressure ADC
     * @param hum_adc Output raw humidity ADC
     * @param gas_adc Output raw gas ADC
     * @param gas_range Output gas range
     * @return True if read succeeds
     */
    bool read_raw_data(uint32_t* temp_adc, uint32_t* press_adc,
                       uint32_t* hum_adc, uint16_t* gas_adc, uint8_t* gas_range);

    /**
     * Compensate temperature from raw ADC using calibration coefficients.
     *
     * @param temp_adc Raw temperature ADC value
     * @return Compensated temperature in °C
     */
    float compensate_temperature(uint32_t temp_adc);

    /**
     * Compensate pressure from raw ADC using calibration coefficients.
     *
     * @param press_adc Raw pressure ADC value
     * @return Compensated pressure in hPa
     */
    float compensate_pressure(uint32_t press_adc);

    /**
     * Compensate humidity from raw ADC using calibration coefficients.
     *
     * @param hum_adc Raw humidity ADC value
     * @return Compensated humidity in %
     */
    float compensate_humidity(uint32_t hum_adc);

    /**
     * Compensate gas resistance from raw ADC using calibration coefficients.
     *
     * @param gas_adc Raw gas ADC value
     * @param gas_range Gas range setting
     * @return Compensated gas resistance in Ω
     */
    float compensate_gas(uint16_t gas_adc, uint8_t gas_range);

    /**
     * Apply calibration offsets to sensor reading.
     *
     * @param reading BME680Reading to calibrate (modified in-place)
     */
    void apply_calibration(BME680Reading* reading);

    /**
     * Read byte from I2C register.
     *
     * @param reg_addr Register address
     * @param data Output byte
     * @return True if read succeeds
     */
    bool read_register(uint8_t reg_addr, uint8_t* data);

    /**
     * Write byte to I2C register.
     *
     * @param reg_addr Register address
     * @param data Byte to write
     * @return True if write succeeds
     */
    bool write_register(uint8_t reg_addr, uint8_t data);

    /**
     * Read multiple bytes from I2C (burst read).
     *
     * @param reg_addr Starting register address
     * @param data Output buffer
     * @param length Number of bytes to read
     * @return True if read succeeds
     */
    bool read_registers(uint8_t reg_addr, uint8_t* data, size_t length);

    BME680Config config_;
    bool initialized_;
    BME680Reading last_reading_;
    int i2c_fd_;  // I2C file descriptor

    // Calibration coefficients (read from sensor EEPROM)
    uint16_t par_t1_, par_p1_;
    int16_t par_t2_, par_t3_, par_p2_, par_p3_, par_p5_, par_p6_, par_p8_, par_p9_;
    uint8_t par_p4_, par_p7_, par_p10_;
    uint16_t par_h1_, par_h2_;
    int8_t par_h3_, par_h4_, par_h5_, par_h6_, par_h7_;
    int8_t par_g1_;
    int16_t par_g2_;
    uint8_t par_g3_;
    uint8_t res_heat_range_;
    int8_t res_heat_val_;
    int8_t range_sw_err_;

    // Temperature for compensation calculations
    float t_fine_;
};

}  // namespace sensors
}  // namespace hardware

#endif  // HARDWARE_SENSORS_BME680_DRIVER_H_
