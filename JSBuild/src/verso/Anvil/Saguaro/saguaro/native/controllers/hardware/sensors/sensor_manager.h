// src/controllers/hardware/sensors/sensor_manager.h
// Phase 4.5 Task 4.5.1.5: Unified sensor manager
// Copyright 2025 Verso Industries

#ifndef HARDWARE_SENSORS_SENSOR_MANAGER_H_
#define HARDWARE_SENSORS_SENSOR_MANAGER_H_

#include "dht22_driver.h"
#include "bme680_driver.h"
#include "anemometer_driver.h"
#include "ina219_driver.h"
#include <memory>
#include <vector>
#include <functional>

namespace hardware {
namespace sensors {

/**
 * Unified sensor reading aggregating all sensor types.
 */
struct UnifiedSensorReading {
    // Environmental sensors
    float temperature_celsius;
    float humidity_percent;
    float pressure_hpa;
    float gas_resistance_ohms;

    // Airflow sensor
    float air_velocity_ms;

    // Power sensors
    float bus_voltage_v;
    float current_ma;
    float power_mw;

    uint64_t timestamp_ns;
    bool valid;
};

/**
 * Sensor manager configuration.
 */
struct SensorManagerConfig {
    bool enable_dht22 = true;
    bool enable_bme680 = true;
    bool enable_anemometer = true;
    bool enable_ina219 = true;

    float sampling_rate_hz = 120.0f;  // Target sampling rate
    bool synchronize_sampling = true;  // Synchronize all sensors to same timestamp
    bool use_default_on_failure = true;  // Use default values if sensor fails

    // Default values for failed sensors
    float default_temperature = 25.0f;
    float default_humidity = 50.0f;
    float default_pressure = 1013.25f;
    float default_air_velocity = 0.0f;
};

/**
 * Unified Sensor Manager.
 *
 * Aggregates readings from multiple sensors (DHT22, BME680, anemometer, INA219)
 * into a unified data structure for operational AI.
 *
 * Features:
 * - Synchronized sampling at configurable rate (default 120 Hz)
 * - Graceful degradation: Use default values if sensor fails
 * - Sensor fusion: Combine DHT22 + BME680 temperature/humidity (average or prefer BME680)
 * - Thread-safe asynchronous sampling
 *
 * Usage:
 *   SensorManagerConfig config;
 *   config.sampling_rate_hz = 120.0f;
 *   SensorManager manager(config);
 *
 *   manager.add_sensor(std::make_unique<DHT22Driver>(dht22_config));
 *   manager.add_sensor(std::make_unique<BME680Driver>(bme680_config));
 *   manager.add_sensor(std::make_unique<AnemometerDriver>(anem_config));
 *   manager.add_sensor(std::make_unique<INA219Driver>(ina219_config));
 *
 *   if (manager.initialize_all()) {
 *       UnifiedSensorReading reading = manager.read_all();
 *   }
 *
 * Roadmap: Phase 4.5, Task 4.5.1.5
 */
class SensorManager {
public:
    explicit SensorManager(const SensorManagerConfig& config);
    ~SensorManager();

    /**
     * Add sensor to manager.
     *
     * @param sensor Unique pointer to sensor driver (takes ownership)
     */
    template<typename T>
    void add_sensor(std::unique_ptr<T> sensor);

    /**
     * Initialize all sensors.
     *
     * @return True if at least one sensor initializes successfully
     */
    bool initialize_all();

    /**
     * Read from all sensors and return unified reading.
     *
     * @return UnifiedSensorReading with all sensor values
     */
    UnifiedSensorReading read_all();

    /**
     * Start asynchronous sampling loop in background thread.
     *
     * @param callback Function to call with each unified reading
     */
    void start_async_sampling(std::function<void(const UnifiedSensorReading&)> callback);

    /**
     * Stop asynchronous sampling loop.
     */
    void stop_async_sampling();

    /**
     * Get last unified reading (cached).
     *
     * @return Last UnifiedSensorReading
     */
    UnifiedSensorReading get_last_reading() const { return last_reading_; }

    /**
     * Check if any sensors are initialized.
     *
     * @return True if at least one sensor is ready
     */
    bool is_ready() const;

    /**
     * Get sensor health status.
     *
     * @return Vector of (sensor_name, is_healthy) pairs
     */
    std::vector<std::pair<std::string, bool>> get_sensor_health() const;

private:
    /**
     * Read from DHT22 sensor.
     *
     * @param reading Output unified reading (modifies temperature/humidity fields)
     */
    void read_dht22(UnifiedSensorReading* reading);

    /**
     * Read from BME680 sensor.
     *
     * @param reading Output unified reading (modifies temp/humidity/pressure/gas fields)
     */
    void read_bme680(UnifiedSensorReading* reading);

    /**
     * Read from anemometer sensor.
     *
     * @param reading Output unified reading (modifies air_velocity field)
     */
    void read_anemometer(UnifiedSensorReading* reading);

    /**
     * Read from INA219 sensor.
     *
     * @param reading Output unified reading (modifies voltage/current/power fields)
     */
    void read_ina219(UnifiedSensorReading* reading);

    /**
     * Fuse sensor readings (e.g., average DHT22 + BME680 temperature).
     *
     * @param reading Unified reading to fuse (modified in-place)
     */
    void fuse_sensor_readings(UnifiedSensorReading* reading);

    /**
     * Apply default values for failed sensors.
     *
     * @param reading Unified reading to fill with defaults (modified in-place)
     */
    void apply_defaults(UnifiedSensorReading* reading);

    /**
     * Asynchronous sampling loop (runs in background thread).
     */
    void async_sampling_loop();

    SensorManagerConfig config_;
    UnifiedSensorReading last_reading_;

    // Sensor driver pointers
    std::unique_ptr<DHT22Driver> dht22_;
    std::unique_ptr<BME680Driver> bme680_;
    std::unique_ptr<AnemometerDriver> anemometer_;
    std::unique_ptr<INA219Driver> ina219_;

    // Async sampling state
    bool async_sampling_active_;
    std::function<void(const UnifiedSensorReading&)> async_callback_;
    // Note: In production, add std::thread member for async loop
};

}  // namespace sensors
}  // namespace hardware

#endif  // HARDWARE_SENSORS_SENSOR_MANAGER_H_
