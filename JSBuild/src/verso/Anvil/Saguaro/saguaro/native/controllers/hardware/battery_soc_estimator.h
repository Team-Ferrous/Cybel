// src/controllers/hardware/battery_soc_estimator.h
// State-of-charge estimator for Samsung 21700 batteries in EHD thruster systems
// Uses Coulomb counting + voltage-based correction (Phase 4.5.4)

#ifndef SAGUARO_CONTROLLERS_HARDWARE_BATTERY_SOC_ESTIMATOR_H_
#define SAGUARO_CONTROLLERS_HARDWARE_BATTERY_SOC_ESTIMATOR_H_

#include <cstdint>
#include <vector>

namespace saguaro {
namespace controllers {
namespace hardware {

// Battery model parameters (Samsung 21700-50E as reference)
struct BatteryParams {
    float nominal_capacity_mAh = 5000.0f;
    float nominal_voltage_V = 3.6f;
    float max_voltage_V = 4.2f;
    float min_voltage_V = 2.75f;
    float internal_resistance_mohm = 18.0f;

    // Discharge curve: SOC% → Voltage (V)
    // Interpolated linearly at runtime
    std::vector<float> soc_curve_percent{100.0f, 90.0f, 80.0f, 70.0f, 60.0f, 50.0f, 40.0f, 30.0f, 20.0f, 10.0f, 5.0f, 0.0f};
    std::vector<float> voltage_curve_V{4.15f, 4.05f, 3.95f, 3.85f, 3.75f, 3.65f, 3.55f, 3.45f, 3.35f, 3.20f, 3.10f, 2.75f};
};

// State-of-charge estimator
class BatterySOCEstimator {
 public:
    explicit BatterySOCEstimator(const BatteryParams& params, float initial_soc = 1.0f);
    ~BatterySOCEstimator() = default;

    // Update SOC from INA219 reading
    // current_mA: Measured current (mA, positive = discharge)
    // voltage_V: Measured terminal voltage (V)
    // dt_seconds: Time since last update (seconds)
    // Returns: Updated SOC [0.0, 1.0]
    float update(float current_mA, float voltage_V, float dt_seconds);

    // Get current SOC estimate
    float get_soc() const { return soc_; }

    // Get remaining capacity (mAh)
    float get_remaining_capacity_mAh() const;

    // Estimate remaining runtime at current discharge rate
    // current_draw_mA: Average current draw (mA)
    // Returns: Runtime in hours
    float estimate_runtime_hours(float current_draw_mA) const;

    // Reset SOC (e.g., after battery swap or full charge)
    void reset(float soc = 1.0f);

    // Get voltage from SOC (forward lookup)
    float voltage_from_soc(float soc, float current_A = 0.0f) const;

    // Get SOC from voltage (inverse lookup)
    float soc_from_voltage(float voltage_V, float current_A = 0.0f) const;

 private:
    BatteryParams params_;
    float soc_;                       // Current state of charge [0.0, 1.0]
    float accumulated_charge_mAh_;    // Accumulated charge (mAh)
    float voltage_correction_weight_; // Weight for voltage-based correction (0.1 = 10%)

    // Linear interpolation helper
    float interpolate(const std::vector<float>& x, const std::vector<float>& y, float xi) const;
};

}  // namespace hardware
}  // namespace controllers
}  // namespace saguaro

#endif  // SAGUARO_CONTROLLERS_HARDWARE_BATTERY_SOC_ESTIMATOR_H_
