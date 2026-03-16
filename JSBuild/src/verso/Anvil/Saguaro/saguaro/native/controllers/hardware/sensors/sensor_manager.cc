// src/controllers/hardware/sensors/sensor_manager.cc
// Copyright 2025 Verso Industries

#include "sensor_manager.h"
#include <chrono>
#include <cmath>
#include <iostream>

namespace hardware {
namespace sensors {

SensorManager::SensorManager(const SensorManagerConfig& config)
    : config_(config), async_sampling_active_(false) {
    last_reading_.temperature_celsius = config_.default_temperature;
    last_reading_.humidity_percent = config_.default_humidity;
    last_reading_.pressure_hpa = config_.default_pressure;
    last_reading_.gas_resistance_ohms = 0.0f;
    last_reading_.air_velocity_ms = config_.default_air_velocity;
    last_reading_.bus_voltage_v = 0.0f;
    last_reading_.current_ma = 0.0f;
    last_reading_.power_mw = 0.0f;
    last_reading_.timestamp_ns = 0;
    last_reading_.valid = false;
}

SensorManager::~SensorManager() {
    stop_async_sampling();
}

template<typename T>
void SensorManager::add_sensor(std::unique_ptr<T> sensor) {
    // Type-based dispatch to assign sensor to appropriate member
    if constexpr (std::is_same_v<T, DHT22Driver>) {
        dht22_ = std::move(sensor);
    } else if constexpr (std::is_same_v<T, BME680Driver>) {
        bme680_ = std::move(sensor);
    } else if constexpr (std::is_same_v<T, AnemometerDriver>) {
        anemometer_ = std::move(sensor);
    } else if constexpr (std::is_same_v<T, INA219Driver>) {
        ina219_ = std::move(sensor);
    }
}

// Explicit template instantiations
template void SensorManager::add_sensor<DHT22Driver>(std::unique_ptr<DHT22Driver>);
template void SensorManager::add_sensor<BME680Driver>(std::unique_ptr<BME680Driver>);
template void SensorManager::add_sensor<AnemometerDriver>(std::unique_ptr<AnemometerDriver>);
template void SensorManager::add_sensor<INA219Driver>(std::unique_ptr<INA219Driver>);

bool SensorManager::initialize_all() {
    bool any_success = false;

    if (config_.enable_dht22 && dht22_) {
        if (dht22_->initialize()) {
            std::cout << "SensorManager: DHT22 initialized" << std::endl;
            any_success = true;
        } else {
            std::cerr << "SensorManager: DHT22 initialization failed" << std::endl;
        }
    }

    if (config_.enable_bme680 && bme680_) {
        if (bme680_->initialize()) {
            std::cout << "SensorManager: BME680 initialized" << std::endl;
            any_success = true;
        } else {
            std::cerr << "SensorManager: BME680 initialization failed" << std::endl;
        }
    }

    if (config_.enable_anemometer && anemometer_) {
        if (anemometer_->initialize()) {
            std::cout << "SensorManager: Anemometer initialized" << std::endl;
            any_success = true;
        } else {
            std::cerr << "SensorManager: Anemometer initialization failed" << std::endl;
        }
    }

    if (config_.enable_ina219 && ina219_) {
        if (ina219_->initialize()) {
            std::cout << "SensorManager: INA219 initialized" << std::endl;
            any_success = true;
        } else {
            std::cerr << "SensorManager: INA219 initialization failed" << std::endl;
        }
    }

    return any_success;
}

UnifiedSensorReading SensorManager::read_all() {
    UnifiedSensorReading reading;
    reading.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Initialize with defaults
    reading.temperature_celsius = config_.default_temperature;
    reading.humidity_percent = config_.default_humidity;
    reading.pressure_hpa = config_.default_pressure;
    reading.gas_resistance_ohms = 0.0f;
    reading.air_velocity_ms = config_.default_air_velocity;
    reading.bus_voltage_v = 0.0f;
    reading.current_ma = 0.0f;
    reading.power_mw = 0.0f;
    reading.valid = false;

    // Read from each sensor
    if (config_.enable_dht22 && dht22_ && dht22_->is_ready()) {
        read_dht22(&reading);
    }

    if (config_.enable_bme680 && bme680_ && bme680_->is_ready()) {
        read_bme680(&reading);
    }

    if (config_.enable_anemometer && anemometer_ && anemometer_->is_ready()) {
        read_anemometer(&reading);
    }

    if (config_.enable_ina219 && ina219_ && ina219_->is_ready()) {
        read_ina219(&reading);
    }

    // Fuse sensor readings (e.g., average temp from DHT22 and BME680)
    fuse_sensor_readings(&reading);

    // Apply defaults if configured
    if (config_.use_default_on_failure) {
        apply_defaults(&reading);
    }

    // Mark as valid if at least one sensor succeeded
    reading.valid = is_ready();

    last_reading_ = reading;
    return reading;
}

void SensorManager::read_dht22(UnifiedSensorReading* reading) {
    DHT22Reading dht_reading = dht22_->read();
    if (dht_reading.valid) {
        reading->temperature_celsius = dht_reading.temperature_celsius;
        reading->humidity_percent = dht_reading.humidity_percent;
    }
}

void SensorManager::read_bme680(UnifiedSensorReading* reading) {
    BME680Reading bme_reading = bme680_->read();
    if (bme_reading.valid) {
        // Prefer BME680 over DHT22 (more accurate)
        reading->temperature_celsius = bme_reading.temperature_celsius;
        reading->humidity_percent = bme_reading.humidity_percent;
        reading->pressure_hpa = bme_reading.pressure_hpa;
        reading->gas_resistance_ohms = bme_reading.gas_resistance_ohms;
    }
}

void SensorManager::read_anemometer(UnifiedSensorReading* reading) {
    AnemometerReading anem_reading = anemometer_->read();
    if (anem_reading.valid) {
        reading->air_velocity_ms = anem_reading.velocity_ms;
    }
}

void SensorManager::read_ina219(UnifiedSensorReading* reading) {
    INA219Reading ina_reading = ina219_->read();
    if (ina_reading.valid) {
        reading->bus_voltage_v = ina_reading.bus_voltage_v;
        reading->current_ma = ina_reading.current_ma;
        reading->power_mw = ina_reading.power_mw;
    }
}

void SensorManager::fuse_sensor_readings(UnifiedSensorReading* reading) {
    // Optional: Average DHT22 and BME680 readings if both present
    // For now, BME680 takes precedence (handled in read_bme680)
    (void)reading;  // Suppress unused parameter warning
}

void SensorManager::apply_defaults(UnifiedSensorReading* reading) {
    // Fill in default values for any failed sensors
    if (reading->temperature_celsius == 0.0f) {
        reading->temperature_celsius = config_.default_temperature;
    }
    if (reading->humidity_percent == 0.0f) {
        reading->humidity_percent = config_.default_humidity;
    }
    if (reading->pressure_hpa == 0.0f) {
        reading->pressure_hpa = config_.default_pressure;
    }
}

bool SensorManager::is_ready() const {
    return (dht22_ && dht22_->is_ready()) ||
           (bme680_ && bme680_->is_ready()) ||
           (anemometer_ && anemometer_->is_ready()) ||
           (ina219_ && ina219_->is_ready());
}

std::vector<std::pair<std::string, bool>> SensorManager::get_sensor_health() const {
    std::vector<std::pair<std::string, bool>> health;

    if (dht22_) {
        health.emplace_back("DHT22", dht22_->is_ready());
    }
    if (bme680_) {
        health.emplace_back("BME680", bme680_->is_ready());
    }
    if (anemometer_) {
        health.emplace_back("Anemometer", anemometer_->is_ready());
    }
    if (ina219_) {
        health.emplace_back("INA219", ina219_->is_ready());
    }

    return health;
}

void SensorManager::start_async_sampling(std::function<void(const UnifiedSensorReading&)> callback) {
    if (async_sampling_active_) return;

    async_sampling_active_ = true;
    async_callback_ = callback;

    // In production, launch std::thread here for async_sampling_loop()
    // For now, provide placeholder
    std::cout << "SensorManager: Async sampling started at " << config_.sampling_rate_hz << " Hz" << std::endl;
}

void SensorManager::stop_async_sampling() {
    if (!async_sampling_active_) return;

    async_sampling_active_ = false;

    // In production, join thread here
    std::cout << "SensorManager: Async sampling stopped" << std::endl;
}

void SensorManager::async_sampling_loop() {
    // Placeholder for background sampling thread
    // In production: while (async_sampling_active_) { read_all(); callback(reading); sleep(); }
}

}  // namespace sensors
}  // namespace hardware
