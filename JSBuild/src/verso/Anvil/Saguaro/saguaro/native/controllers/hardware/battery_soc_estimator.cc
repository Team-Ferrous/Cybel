// src/controllers/hardware/battery_soc_estimator.cc
// Implementation of battery SOC estimator for EHD thruster systems

#include "src/controllers/hardware/battery_soc_estimator.h"

#include <algorithm>
#include <cmath>

namespace saguaro {
namespace controllers {
namespace hardware {

BatterySOCEstimator::BatterySOCEstimator(const BatteryParams& params, float initial_soc)
    : params_(params),
      soc_(initial_soc),
      voltage_correction_weight_(0.1f) {
    // Clamp initial SOC to valid range
    soc_ = std::max(0.0f, std::min(1.0f, soc_));
    accumulated_charge_mAh_ = soc_ * params_.nominal_capacity_mAh;
}

float BatterySOCEstimator::update(float current_mA, float voltage_V, float dt_seconds) {
    // Coulomb counting: integrate current over time
    // dQ = I * dt (mAh = mA * hours)
    float charge_delta_mAh = current_mA * (dt_seconds / 3600.0f);  // Convert seconds to hours
    accumulated_charge_mAh_ -= charge_delta_mAh;  // Discharge decreases accumulated charge

    // Compute SOC from accumulated charge
    float soc_coulomb = accumulated_charge_mAh_ / params_.nominal_capacity_mAh;

    // Voltage-based correction: estimate SOC from measured voltage
    float soc_voltage = soc_from_voltage(voltage_V, current_mA / 1000.0f);

    // Blend Coulomb and voltage estimates (weighted average)
    // This compensates for Coulomb counting drift
    soc_ = (1.0f - voltage_correction_weight_) * soc_coulomb +
           voltage_correction_weight_ * soc_voltage;

    // Clamp to valid range
    soc_ = std::max(0.0f, std::min(1.0f, soc_));

    // Update accumulated charge to match corrected SOC
    accumulated_charge_mAh_ = soc_ * params_.nominal_capacity_mAh;

    return soc_;
}

float BatterySOCEstimator::get_remaining_capacity_mAh() const {
    return soc_ * params_.nominal_capacity_mAh;
}

float BatterySOCEstimator::estimate_runtime_hours(float current_draw_mA) const {
    if (current_draw_mA <= 0.0f) {
        return INFINITY;
    }

    float remaining_capacity_mAh = get_remaining_capacity_mAh();
    return remaining_capacity_mAh / current_draw_mA;
}

void BatterySOCEstimator::reset(float soc) {
    soc_ = std::max(0.0f, std::min(1.0f, soc));
    accumulated_charge_mAh_ = soc_ * params_.nominal_capacity_mAh;
}

float BatterySOCEstimator::voltage_from_soc(float soc, float current_A) const {
    // Clamp SOC to valid range
    soc = std::max(0.0f, std::min(1.0f, soc));

    // Convert SOC [0.0, 1.0] to percent [0.0, 100.0]
    float soc_percent = soc * 100.0f;

    // Interpolate discharge curve: SOC% → Voltage
    float v_oc = interpolate(params_.soc_curve_percent, params_.voltage_curve_V, soc_percent);

    // IR drop compensation: V_terminal = V_oc - I * R
    float ir_drop_V = current_A * (params_.internal_resistance_mohm / 1000.0f);
    float v_terminal = v_oc - ir_drop_V;

    return v_terminal;
}

float BatterySOCEstimator::soc_from_voltage(float voltage_V, float current_A) const {
    // Compensate for IR drop to get open-circuit voltage
    float ir_drop_V = current_A * (params_.internal_resistance_mohm / 1000.0f);
    float v_oc = voltage_V + ir_drop_V;

    // Inverse lookup: Voltage → SOC%
    // Interpolate backwards: given voltage, find SOC
    float soc_percent = interpolate(params_.voltage_curve_V, params_.soc_curve_percent, v_oc);

    // Convert SOC% to fraction [0.0, 1.0]
    float soc = soc_percent / 100.0f;

    return std::max(0.0f, std::min(1.0f, soc));
}

float BatterySOCEstimator::interpolate(
    const std::vector<float>& x,
    const std::vector<float>& y,
    float xi) const {
    // Linear interpolation with clamping outside bounds
    if (x.size() != y.size() || x.empty()) {
        return 0.0f;
    }

    // Find surrounding points
    size_t i = 0;
    for (size_t j = 0; j < x.size() - 1; ++j) {
        if (xi >= x[j] && xi <= x[j + 1]) {
            i = j;
            break;
        }
    }

    // Clamp if outside range
    if (xi < x.front()) {
        return y.front();
    }
    if (xi > x.back()) {
        return y.back();
    }

    // Linear interpolation: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    float x0 = x[i];
    float x1 = x[i + 1];
    float y0 = y[i];
    float y1 = y[i + 1];

    if (std::abs(x1 - x0) < 1e-6f) {
        return y0;  // Avoid division by zero
    }

    float slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (xi - x0);
}

}  // namespace hardware
}  // namespace controllers
}  // namespace saguaro
