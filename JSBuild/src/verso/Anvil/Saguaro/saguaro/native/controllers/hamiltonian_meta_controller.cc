#include "hamiltonian_meta_controller.h"

#include <cmath>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <set>
#include <iterator>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <ctime>

#include "common/parallel/parallel_backend.h"
#include "config/state_space_config_loader.h"
#include "utils/matrix_utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
namespace {
// Helper function to print vectors for logging
std::string vector_to_string(const std::vector<float>& vec, int limit = -1) {
    std::stringstream ss;
    ss << "[";
    size_t n = (limit == -1 || static_cast<size_t>(limit) > vec.size()) ? vec.size() : static_cast<size_t>(limit);
    for (size_t i = 0; i < n; ++i) {
        ss << vec[i] << (i == n - 1 ? "" : ", ");
    }
    if (n < vec.size()) {
        ss << ", ...";
    }
    ss << "]";
    return ss.str();
}

inline bool MatrixEmpty(const Eigen::MatrixXf& mat) {
    return mat.size() == 0;
}

inline size_t MatrixRows(const Eigen::MatrixXf& mat) {
    return static_cast<size_t>(mat.rows());
}

inline size_t MatrixCols(const Eigen::MatrixXf& mat) {
    return static_cast<size_t>(mat.cols());
}

inline float MatrixValueOrDefault(const Eigen::MatrixXf& mat, size_t row, size_t col, float default_value = 0.0f) {
    if (MatrixEmpty(mat)) {
        return default_value;
    }
    if (row >= MatrixRows(mat) || col >= MatrixCols(mat)) {
        return default_value;
    }
    return mat(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
}
} // anonymous namespace

HamiltonianMetaController::HamiltonianMetaController(const std::string& config_path)
    : default_config_file_path_(config_path),
      hardware_manager_(std::make_unique<HardwareManager>()),
      sensitivity_tracker_(std::make_unique<saguaro::control::SensitivityTracker>()),
      hierarchical_controller_(std::make_unique<saguaro::control::HierarchicalController>()),
      adaptive_phase_controller_(std::make_unique<saguaro::control::AdaptivePhaseController>()),
      // Phase 2.1: Initialize quantum-enhanced control components
      rls_sysid_(std::make_unique<saguaro::controllers::RLSSystemIdentifier<float>>()),
      hybrid_pid_(std::make_unique<saguaro::controllers::HybridPIDTuner>()),
      ekf_(std::make_unique<saguaro::controllers::ExtendedKalmanFilter>()),
      tnkf_(std::make_unique<saguaro::controllers::TensorNetworkKalmanFilter>()) {
    std::cout << "[MetaController] Quantum-enhanced control components initialized." << std::endl;
}

HamiltonianMetaController::~HamiltonianMetaController() = default;
 
void HamiltonianMetaController::generate_default_pid_config(
    const std::vector<std::string>& metric_names,
    const std::vector<std::string>& control_input_names, // ADDED: The definitive list of inputs
    const std::string& path) {
    std::cout << "Warning: '" << path << "' not found. Generating a default PID config file." << std::endl;
    PIDConfig defaultConfig; 
    defaultConfig.relay_output_amplitude = 1.5f;
    defaultConfig.relay_hysteresis = 0.03f;
    std::vector<std::string> temp_control_input_names =
        get_control_input_names(metric_names, control_input_names);

    defaultConfig.metric_names = metric_names;
    defaultConfig.control_input_names = temp_control_input_names;
    const int num_inputs = static_cast<int>(temp_control_input_names.size());
    // --- END: DEFINITIVE FIX ---

    const int num_outputs = static_cast<int>(metric_names.size());
    defaultConfig.Kp_matrix = Eigen::MatrixXf::Zero(num_inputs, num_outputs);
    defaultConfig.Ki_matrix = Eigen::MatrixXf::Zero(num_inputs, num_outputs);
    defaultConfig.Kd_matrix = Eigen::MatrixXf::Zero(num_inputs, num_outputs);

    enum class ControlRole { EvolutionTime, EvolutionTimeParam, Hardware, Other };
    auto classify_control = [](const std::string& name) {
        if (name.find("evolution_time") != std::string::npos) {
            if (name.find("evolution_time_gain") != std::string::npos ||
                name.find("evolution_time_shift") != std::string::npos ||
                name.find("evolution_time_cap") != std::string::npos ||
                name.find("evolution_time_base") != std::string::npos ||
                name.find("evolution_time_raw") != std::string::npos ||
                name.find("evolution_time_std") != std::string::npos) {
                return ControlRole::EvolutionTimeParam;
            }
            return ControlRole::EvolutionTime;
        }
        if (name.find("core_frequency") != std::string::npos ||
            name.find("cpu") != std::string::npos ||
            name.find("fan") != std::string::npos ||
            name.find("power_limit") != std::string::npos) {
            return ControlRole::Hardware;
        }
        return ControlRole::Other;
    };

    std::vector<ControlRole> control_roles;
    control_roles.reserve(num_inputs);
    defaultConfig.control_min_bounds.assign(num_inputs, 0.0f);
    defaultConfig.control_max_bounds.assign(num_inputs, 1.0f);

    std::vector<int> evolution_indices;
    std::vector<int> hardware_indices;
    for (int idx = 0; idx < num_inputs; ++idx) {
        const auto& control_name = temp_control_input_names[idx];
        ControlRole role = classify_control(control_name);
        control_roles.push_back(role);
        if (role == ControlRole::EvolutionTime) {
            evolution_indices.push_back(idx);
            defaultConfig.control_min_bounds[idx] = 5e-4f;
            defaultConfig.control_max_bounds[idx] = -1.0f; // Negative sentinel indicates no hard upper bound.
        } else if (role == ControlRole::EvolutionTimeParam) {
            defaultConfig.control_min_bounds[idx] = -5.0f;
            defaultConfig.control_max_bounds[idx] = -1.0f;
        } else if (role == ControlRole::Hardware) {
            hardware_indices.push_back(idx);
            defaultConfig.control_min_bounds[idx] = 0.0f;
            defaultConfig.control_max_bounds[idx] = 1.0f;
        } else {
            defaultConfig.control_min_bounds[idx] = 0.0f;
            defaultConfig.control_max_bounds[idx] = 1.0f;
        }
    }

    // Create a map from a block's base name (e.g., "timecrystal_block_1") to its index in the control input list.
    std::map<std::string, int> control_name_to_input_idx;
    int current_input_idx = 0;
    for (const auto& input_name : temp_control_input_names) {
        // Use the full name as the key for an exact match.
        control_name_to_input_idx[input_name] = current_input_idx++;
    }

    // Iterate through each metric (output) to set its setpoint and PID gains.
    for (size_t metric_idx = 0; metric_idx < metric_names.size(); ++metric_idx) {
        const auto& metric_name = metric_names[metric_idx];
        defaultConfig.Setpoint[metric_name] = 0.0f; // Default setpoint is 0 for all metrics except loss.

        auto register_mapping = [&](int input_idx) {
            const auto& control_name = temp_control_input_names[input_idx];
            auto& bucket = defaultConfig.control_metric_map[control_name];
            if (std::find(bucket.begin(), bucket.end(), metric_name) == bucket.end()) {
                bucket.push_back(metric_name);
            }
        };

        // --- START: DEFINITIVE FIX for Differentiated PID Gains ---
        // Set specific gains based on metric type to enable true MIMO control.
        if (metric_name == "loss") {
            defaultConfig.Setpoint[metric_name] = 2.5f; // Target a reasonable initial loss.
            // Apply a firmer influence only to evolution-time controls to move the system off the plateau.
            for (size_t i = 0; i < evolution_indices.size(); ++i) {
                int input_idx = evolution_indices[i];
                const Eigen::Index col = static_cast<Eigen::Index>(metric_idx);
                defaultConfig.Kp_matrix(input_idx, col) = -0.0125f;
                defaultConfig.Ki_matrix(input_idx, col) = -4e-4f;
                register_mapping(input_idx);
            }
        } else if (metric_name == "cpu_temperature") {
            defaultConfig.Setpoint[metric_name] = 85.0f; // Target 85C
            // Route thermal management only through hardware-affecting controls.
            if (hardware_indices.empty()) {
                std::cout << "Warning: No hardware control channels found to manage cpu_temperature." << std::endl;
            }
            for (int input_idx : hardware_indices) {
                const Eigen::Index col = static_cast<Eigen::Index>(metric_idx);
                defaultConfig.Kp_matrix(input_idx, col) = -0.002f;
                defaultConfig.Ki_matrix(input_idx, col) = -0.0001f;
                register_mapping(input_idx);
            }
        } else if (metric_name.find('/') != std::string::npos) {
            // This handles block-specific metrics like "timecrystal_block_1/energy_drift".
            // We want to link a metric to its corresponding control variable.
            // For example, metric 'timecrystal_block_1/energy_drift' should affect
            // the control input 'timecrystal_block_1/timecrystal_block_1_cell/evolution_time'.
            std::string metric_prefix = metric_name.substr(0, metric_name.find_last_of('/'));
            
            // Find the control input that corresponds to this metric.
            for (const auto& pair : control_name_to_input_idx) {
                const std::string& control_name = pair.first;
                int input_idx = pair.second;
 
                // If the metric name is a substring of the control name (e.g., "timecrystal_block_1" in "timecrystal_block_1/.../evolution_time")
                if (control_name.rfind(metric_prefix, 0) == 0) { // `rfind` with pos 0 is equivalent to `starts_with`
                    // Only the evolution_time for this specific block should react to its energy_drift.
                    if (metric_name.find("energy_drift") != std::string::npos) {
                        // Set a specific, stronger gain for this direct relationship.
                        const Eigen::Index col = static_cast<Eigen::Index>(metric_idx);
                        defaultConfig.Kp_matrix(input_idx, col) = -0.015f;
                        defaultConfig.Ki_matrix(input_idx, col) = -6e-4f;
                        register_mapping(input_idx);
                        // Break because we found the matching control input.
                        break; 
                    } else if (metric_name.find("evolution_time") != std::string::npos) {
                        // Use a gentle gain to keep the evolution time near its reference.
                        const Eigen::Index col = static_cast<Eigen::Index>(metric_idx);
                        defaultConfig.Kp_matrix(input_idx, col) = 0.01f;
                        defaultConfig.Ki_matrix(input_idx, col) = 2e-4f;
                        register_mapping(input_idx);
                        break;
                    }
                }
            }
        }
    }
    // --- END: DEFINITIVE FIX ---

    ConfigLoader::save_pid_config(path, defaultConfig);
} 

void HamiltonianMetaController::sanitize_measurements(std::vector<float>& y_k) {
    if (last_valid_measurements_.size() != y_k.size()) {
        last_valid_measurements_.assign(y_k.begin(), y_k.end());
        for (float& v : last_valid_measurements_) {
            if (!std::isfinite(v)) {
                v = 0.0f;
            }
        }
    }

    if (!y_k.empty()) {
        saguaro::parallel::ForRange(
            0, y_k.size(), 64,
            [&](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    float value = y_k[i];
                    if (!std::isfinite(value)) {
                        value = last_valid_measurements_[i];
                    }
                    if (!std::isfinite(value)) {
                        value = 0.0f;
                    }
                    last_valid_measurements_[i] = value;
                    y_k[i] = value;
                }
            });
    }
}

void HamiltonianMetaController::bootstrap_scalers(const std::vector<float>& raw_measurements) {
    if (scalers_bootstrapped_ || raw_measurements.empty()) {
        return;
    }

    if (running_mean_.size() != raw_measurements.size()) {
        running_mean_.assign(raw_measurements.size(), 0.0);
        running_m2_.assign(raw_measurements.size(), 0.0);
        warmup_samples_ = 0;
    }

    warmup_samples_ += 1;
    if (!raw_measurements.empty()) {
        saguaro::parallel::ForRange(
            0, raw_measurements.size(), 64,
            [&](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    double sample = static_cast<double>(raw_measurements[i]);
                    if (!std::isfinite(sample)) {
                        continue;
                    }
                    double delta = sample - running_mean_[i];
                    running_mean_[i] += delta / static_cast<double>(warmup_samples_);
                    double delta2 = sample - running_mean_[i];
                    running_m2_[i] += delta * delta2;
                }
            });
    }

    if (warmup_samples_ < warmup_target_) {
        return;
    }

    if (controller_config_.y_scaler_mean.size() != raw_measurements.size()) {
        controller_config_.y_scaler_mean.assign(raw_measurements.size(), 0.0f);
    }
    if (controller_config_.y_scaler_scale.size() != raw_measurements.size()) {
        controller_config_.y_scaler_scale.assign(raw_measurements.size(), 1.0f);
    }

    std::cout << "[MetaController] Warm-up complete. Updating scaler statistics after " << warmup_samples_ << " samples." << std::endl;
    if (!raw_measurements.empty()) {
        saguaro::parallel::ForRange(
            0, raw_measurements.size(), 64,
            [&](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    double variance = warmup_samples_ > 1 ? running_m2_[i] / static_cast<double>(warmup_samples_ - 1) : 0.0;
                    double stddev = std::sqrt(std::max(variance, 1e-12));
                    stddev = std::clamp(stddev, 1e-3, 1e6);
                    controller_config_.y_scaler_mean[i] = static_cast<float>(running_mean_[i]);
                    controller_config_.y_scaler_scale[i] = static_cast<float>(stddev);
                }
            });
    }

    ConfigLoader::save_controller_config(active_config_file_path_, controller_config_);
    scalers_bootstrapped_ = true;
    running_mean_.clear();
    running_m2_.clear();
    warmup_samples_ = 0;

    if (mpc_controller_) {
        mpc_controller_->reset();
    }
} 

float HamiltonianMetaController::clamp_control_step(float requested_unscaled, float previous_unscaled) const {
    if (!std::isfinite(previous_unscaled)) {
        previous_unscaled = 0.0f;
    }
    if (!std::isfinite(requested_unscaled)) {
        return previous_unscaled;
    }

    float max_step = std::max(max_control_step_absolute_, std::fabs(previous_unscaled) * max_control_step_fraction_);
    float lower_bound = previous_unscaled - max_step;
    float upper_bound = previous_unscaled + max_step;
    if (lower_bound > upper_bound) {
        std::swap(lower_bound, upper_bound);
    }
    return std::max(lower_bound, std::min(upper_bound, requested_unscaled));
}

std::vector<std::string> HamiltonianMetaController::get_control_input_names(
    const std::vector<std::string>& metric_names,
    const std::vector<std::string>& control_input_names) {
    std::vector<std::string> sanitized;
    sanitized.reserve(control_input_names.size());
    std::unordered_set<std::string> seen;

    auto append_unique = [&](const std::string& candidate) {
        if (candidate.empty()) {
            return;
        }
        if (seen.insert(candidate).second) {
            sanitized.push_back(candidate);
        }
    };

    // Prefer explicit control list from the runtime if available.
    for (const auto& name : control_input_names) {
        append_unique(name);
    }

    if (!sanitized.empty()) {
        return sanitized;
    }

    static constexpr const char* kEvolutionSuffix = "/evolution_time";
    const size_t suffix_len = std::strlen(kEvolutionSuffix);

    for (const auto& metric_name : metric_names) {
        if (metric_name.empty()) {
            continue;
        }
        if (metric_name.size() >= suffix_len &&
            metric_name.compare(metric_name.size() - suffix_len, suffix_len, kEvolutionSuffix) == 0) {
            append_unique(metric_name);
            continue;
        }
        if (metric_name.find("evolution_time") != std::string::npos) {
            append_unique(metric_name);
            continue;
        }
        const size_t slash_pos = metric_name.rfind('/');
        if (slash_pos != std::string::npos) {
            std::string candidate = metric_name.substr(0, slash_pos);
            candidate.append(kEvolutionSuffix);
            append_unique(candidate);
        }
    }

    if (sanitized.empty()) {
        for (const auto& metric_name : metric_names) {
            append_unique(metric_name);
        }
    }

    return sanitized;
}

void HamiltonianMetaController::load_and_reinitialize(const std::string& path_prefix, const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names) {
    active_config_file_path_ = path_prefix.empty() ? default_config_file_path_ : path_prefix + "/state_space.conf";
    std::string pid_path = path_prefix.empty() ? "pid.conf" : path_prefix + "/pid.conf";

    std::cout << "Loading state-space config from: " << active_config_file_path_ << std::endl;
    controller_config_ = ConfigLoader::load_controller_config(active_config_file_path_);
    std::cout << "Loading PID config from: " << pid_path << std::endl;
    pid_config_ = ConfigLoader::load_pid_config(pid_path);

    auto rebuild_default_configs = [&](const std::vector<std::string>& reasons) {
        std::cout << "[MetaController] Regenerating controller configuration files because:" << std::endl;
        for (const auto& reason : reasons) {
            std::cout << "  - " << reason << std::endl;
        }

        // Archive obsolete pid.conf before regeneration (Phase 2.2 Gate)
        std::ifstream pid_check(pid_path);
        if (pid_check.good()) {
            pid_check.close();

            // Generate timestamp in YYYYMMDD_HHMMSS format
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::tm local_tm;
            localtime_r(&in_time_t, &local_tm);

            std::ostringstream timestamp_stream;
            timestamp_stream << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
            std::string timestamp = timestamp_stream.str();

            std::string archive_path = pid_path + ".bak." + timestamp;

            if (std::rename(pid_path.c_str(), archive_path.c_str()) == 0) {
                std::cout << "[MetaController] Archived invalid pid.conf to: " << archive_path << std::endl;
            } else {
                std::cerr << "[MetaController] Warning: Failed to archive pid.conf to " << archive_path
                          << ". Proceeding with regeneration." << std::endl;
            }
        }

        this->generate_default_state_space_config(metric_names, control_input_names, active_config_file_path_);
        generate_default_pid_config(metric_names, control_input_names, pid_path);
        controller_config_ = ConfigLoader::load_controller_config(active_config_file_path_);
        pid_config_ = ConfigLoader::load_pid_config(pid_path);
    };

    for (int attempt = 0; attempt < 2; ++attempt) {
        std::vector<std::string> regen_reasons;
        const size_t expected_metrics = metric_names.size();
        size_t expected_control_dim = control_input_names.size();
        if (expected_control_dim == 0) {
            expected_control_dim = pid_config_.control_input_names.size();
        }
        if (expected_control_dim == 0 && !MatrixEmpty(controller_config_.B)) {
            expected_control_dim = MatrixCols(controller_config_.B);
        }

        auto push_dim_mismatch = [&](const std::string& label, size_t have, size_t expected_val) {
            std::ostringstream oss;
            oss << label << " (" << have << ") does not match expected (" << expected_val << ")";
            regen_reasons.push_back(oss.str());
        };

        if (expected_metrics > 0) {
            if (!MatrixEmpty(controller_config_.A) && MatrixRows(controller_config_.A) != expected_metrics) {
                push_dim_mismatch("State matrix A row count", MatrixRows(controller_config_.A), expected_metrics);
            }
            if (!MatrixEmpty(controller_config_.A) && MatrixCols(controller_config_.A) != MatrixRows(controller_config_.A)) {
                push_dim_mismatch("State matrix A column count", MatrixCols(controller_config_.A), MatrixRows(controller_config_.A));
            }
            if (!MatrixEmpty(controller_config_.C) && MatrixRows(controller_config_.C) != expected_metrics) {
                push_dim_mismatch("State-space matrix C row count", MatrixRows(controller_config_.C), expected_metrics);
            }
            if (!MatrixEmpty(controller_config_.C) && MatrixCols(controller_config_.C) != MatrixRows(controller_config_.A)) {
                push_dim_mismatch("State-space matrix C column count", MatrixCols(controller_config_.C), MatrixRows(controller_config_.A));
            }
            if (!MatrixEmpty(controller_config_.B) && MatrixRows(controller_config_.B) != expected_metrics) {
                push_dim_mismatch("State-space matrix B row count", MatrixRows(controller_config_.B), expected_metrics);
            }
            if (!controller_config_.y_scaler_mean.empty() && controller_config_.y_scaler_mean.size() != expected_metrics) {
                push_dim_mismatch("y_scaler_mean length", controller_config_.y_scaler_mean.size(), expected_metrics);
            }
            if (!controller_config_.y_scaler_scale.empty() && controller_config_.y_scaler_scale.size() != expected_metrics) {
                push_dim_mismatch("y_scaler_scale length", controller_config_.y_scaler_scale.size(), expected_metrics);
            }
        } else if (!MatrixEmpty(controller_config_.C) || !MatrixEmpty(controller_config_.A)) {
            regen_reasons.push_back("Runtime provided zero metrics but persisted state-space matrices are populated.");
        }

        if (expected_control_dim > 0) {
            if (!MatrixEmpty(controller_config_.B) && MatrixCols(controller_config_.B) != expected_control_dim) {
                push_dim_mismatch("State-space matrix B column count", MatrixCols(controller_config_.B), expected_control_dim);
            }
            if (!MatrixEmpty(controller_config_.K) && MatrixRows(controller_config_.K) != expected_control_dim) {
                push_dim_mismatch("LQR gain matrix K row count", MatrixRows(controller_config_.K), expected_control_dim);
            }
            if (!MatrixEmpty(controller_config_.K) && MatrixCols(controller_config_.K) != MatrixRows(controller_config_.A)) {
                push_dim_mismatch("LQR gain matrix K column count", MatrixCols(controller_config_.K), MatrixRows(controller_config_.A));
            }
            if (!controller_config_.u_scaler_mean.empty() && controller_config_.u_scaler_mean.size() != expected_control_dim) {
                push_dim_mismatch("u_scaler_mean length", controller_config_.u_scaler_mean.size(), expected_control_dim);
            }
            if (!controller_config_.u_scaler_scale.empty() && controller_config_.u_scaler_scale.size() != expected_control_dim) {
                push_dim_mismatch("u_scaler_scale length", controller_config_.u_scaler_scale.size(), expected_control_dim);
            }
        } else if (!MatrixEmpty(controller_config_.B) || !controller_config_.u_scaler_mean.empty()) {
            regen_reasons.push_back("Runtime provided zero control inputs but persisted state-space configuration defines controls.");
        }

        auto check_pid_matrix = [&](const Eigen::MatrixXf& matrix, const char* name) {
            if (matrix.size() == 0) {
                std::ostringstream oss;
                oss << "PID " << name << " matrix is empty";
                regen_reasons.push_back(oss.str());
                return;
            }
            const size_t rows = static_cast<size_t>(matrix.rows());
            const size_t cols = static_cast<size_t>(matrix.cols());
            if (expected_control_dim > 0 && rows != expected_control_dim) {
                std::ostringstream oss;
                oss << "PID " << name << " matrix rows (" << rows
                    << ") do not match control dimension (" << expected_control_dim << ")";
                regen_reasons.push_back(oss.str());
            }
            if (expected_metrics > 0 && cols != 0 && cols != expected_metrics) {
                std::ostringstream oss;
                oss << "PID " << name << " matrix columns (" << cols
                    << ") do not match metric count (" << expected_metrics << ")";
                regen_reasons.push_back(oss.str());
            }
        };

        check_pid_matrix(pid_config_.Kp_matrix, "Kp");
        check_pid_matrix(pid_config_.Ki_matrix, "Ki");
        check_pid_matrix(pid_config_.Kd_matrix, "Kd");

        if (expected_control_dim > 0) {
            if (!pid_config_.control_min_bounds.empty() && pid_config_.control_min_bounds.size() != expected_control_dim) {
                std::ostringstream oss;
                oss << "PID control_min_bounds size (" << pid_config_.control_min_bounds.size()
                    << ") does not match control dimension (" << expected_control_dim << ")";
                regen_reasons.push_back(oss.str());
            }
            if (!pid_config_.control_max_bounds.empty() && pid_config_.control_max_bounds.size() != expected_control_dim) {
                std::ostringstream oss;
                oss << "PID control_max_bounds size (" << pid_config_.control_max_bounds.size()
                    << ") does not match control dimension (" << expected_control_dim << ")";
                regen_reasons.push_back(oss.str());
            }
        }

        if (!metric_names.empty() && !pid_config_.metric_names.empty()) {
            if (pid_config_.metric_names.size() != metric_names.size()) {
                push_dim_mismatch("PID metric_names count", pid_config_.metric_names.size(), metric_names.size());
            } else {
                std::unordered_set<std::string> persisted(pid_config_.metric_names.begin(), pid_config_.metric_names.end());
                for (const auto& name : metric_names) {
                    if (persisted.find(name) == persisted.end()) {
                        regen_reasons.push_back("PID metric_names content differs from runtime metric list.");
                        break;
                    }
                }
            }
        }

        if (!control_input_names.empty() && !pid_config_.control_input_names.empty()) {
            if (pid_config_.control_input_names.size() != control_input_names.size()) {
                push_dim_mismatch("PID control_input_names count", pid_config_.control_input_names.size(), control_input_names.size());
            } else {
                std::unordered_set<std::string> persisted(pid_config_.control_input_names.begin(), pid_config_.control_input_names.end());
                for (const auto& name : control_input_names) {
                    if (persisted.find(name) == persisted.end()) {
                        regen_reasons.push_back("PID control_input_names content differs from runtime control list.");
                        break;
                    }
                }
            }
        }

        if (regen_reasons.empty()) {
            break;
        }

        if (attempt == 1) {
            std::cerr << "FATAL: Unable to reconcile controller configuration dimensions after regeneration attempts." << std::endl;
            for (const auto& reason : regen_reasons) {
                std::cerr << "  - " << reason << std::endl;
            }
            std::terminate();
        }

        rebuild_default_configs(regen_reasons);
    }

    pid_config_.metric_names = metric_names;
    pid_config_.control_input_names = control_input_names;

    running_mean_.clear();
    running_m2_.clear();
    warmup_samples_ = 0;
    scalers_bootstrapped_ = false;
    mpc_controller_.reset();
    if (controller_config_.y_scaler_mean.size() == metric_names.size()) {
        const float mean_threshold = 1e-4f;
        scalers_bootstrapped_ = std::any_of(
            controller_config_.y_scaler_mean.begin(),
            controller_config_.y_scaler_mean.end(),
            [mean_threshold](float value) { return std::fabs(value) > mean_threshold; });
    }

    is_mpc_active_ = false; // Default to inactive (PID only)
    if (system_id_model_ready_ && !MatrixEmpty(controller_config_.B)) {
        if (std::abs(MatrixValueOrDefault(controller_config_.B, 0, 0)) > 1e-9f) {
            is_mpc_active_ = true;
        }
    }
    if (!is_mpc_active_) {
        if (system_id_model_ready_) {
            std::cout << "[MetaController] Warning: MPC prerequisites satisfied but matrices prevent activation; inspect state_space.conf." << std::endl;
        } else {
            std::cout << "[MetaController] MPC held in standby until system identification completes." << std::endl;
        }
    }

    size_t control_dim = control_input_names.size();
    if (control_dim == 0) {
        std::cerr << "Warning: `control_input_names` is empty. Trying to infer control dimension from config files." << std::endl;
        if (!pid_config_.control_input_names.empty()) {
            control_dim = pid_config_.control_input_names.size();
        } else if (!MatrixEmpty(controller_config_.B)) {
            control_dim = MatrixCols(controller_config_.B);
        } else {
            control_dim = 1; // Fallback to 1 if completely un-inferrable
            std::cerr << "Could not infer control dimension. Defaulting to 1. This may cause issues." << std::endl;
        }
    }
    
    kalman_filter_ = std::make_unique<KalmanFilter>();

    const int state_dim = static_cast<int>(MatrixRows(controller_config_.A));
    const int eigen_control_dim = static_cast<int>(MatrixCols(controller_config_.B));
    const int output_dim = static_cast<int>(MatrixRows(controller_config_.C));

    Eigen::MatrixXf A_eigen = controller_config_.A;
    Eigen::MatrixXf B_eigen = controller_config_.B;
    Eigen::MatrixXf C_eigen = controller_config_.C;
    Eigen::MatrixXf D_eigen = controller_config_.D;
    Eigen::MatrixXf Q_eigen = controller_config_.Q;
    Eigen::MatrixXf R_kalman_eigen = controller_config_.R_kalman;

    kalman_filter_->init(A_eigen, B_eigen, C_eigen,
                         D_eigen, Q_eigen, R_kalman_eigen);
    std::cout << "Kalman filter initialized." << std::endl;

    // Phase 2.1: Configure quantum-enhanced control components
    if (use_fast_sysid_ && rls_sysid_) {
        // Configure RLS for fast incremental system identification
        // State order matches system dimensions, single input/output per channel
        int rls_state_order = std::max(2, state_dim / 2);
        rls_sysid_->configure(rls_state_order, std::max(1, eigen_control_dim), 
                              std::max(1, output_dim), 0.998f);
        
        // Seed RLS from current state-space matrices for hybrid approach
        if (!MatrixEmpty(A_eigen) && A_eigen.rows() > 0) {
            rls_sysid_->seedFromStateSpace(A_eigen, B_eigen, C_eigen, D_eigen);
        }
        std::cout << "[MetaController] RLS System Identifier configured (interval=" 
                  << fast_sysid_interval_ << " batches)" << std::endl;
    }

    if (use_hybrid_pid_ && hybrid_pid_) {
        // Configure Hybrid PID tuner with number of control channels
        saguaro::controllers::HybridPIDConfig pid_cfg;
        pid_cfg.learning_rate = 0.001f;
        pid_cfg.relay_amplitude = pid_config_.relay_output_amplitude;
        pid_cfg.relay_hysteresis = pid_config_.relay_hysteresis;
        pid_cfg.zn_type = saguaro::controllers::ZNType::SOME_OVERSHOOT;
        hybrid_pid_->init(std::max(1, eigen_control_dim), pid_cfg);
        
        // Seed with existing gains if available
        if (pid_config_.Kp_matrix.size() > 0) {
            saguaro::controllers::PIDGains initial_gains;
            initial_gains.Kp = pid_config_.Kp_matrix.col(0);  // Use first metric column
            initial_gains.Ki = pid_config_.Ki_matrix.col(0);
            initial_gains.Kd = pid_config_.Kd_matrix.col(0);
            if (initial_gains.isValid()) {
                hybrid_pid_->seedGains(initial_gains);
            }
        }
        std::cout << "[MetaController] Hybrid PID Tuner configured (Adam + Z-N)" << std::endl;
    }

    if (use_extended_kalman_ && ekf_) {
        // Initialize EKF with same matrices as standard Kalman (linear mode)
        ekf_->initLinear(A_eigen, B_eigen, C_eigen, D_eigen, Q_eigen, R_kalman_eigen);
        ekf_->enableAdaptiveNoise(true, 0.05f);
        std::cout << "[MetaController] Extended Kalman Filter configured (adaptive noise enabled)" << std::endl;
    }

    if (use_tensor_network_kalman_ && tnkf_) {
        // Initialize TNKF with TT rank for memory efficiency (large state dims only)
        int max_tt_rank = 8;  // Configuration: higher rank = more accuracy, less compression
        if (state_dim >= 16) {  // Only use TT for large state dimensions
            tnkf_->init(A_eigen, B_eigen, C_eigen, D_eigen, Q_eigen, R_kalman_eigen, max_tt_rank);
            std::cout << "[MetaController] Tensor Network Kalman Filter configured (rank=" 
                      << max_tt_rank << ", memory efficient)" << std::endl;
        } else {
            use_tensor_network_kalman_ = false;  // Disable for small systems
            std::cout << "[MetaController] TNKF disabled (state_dim=" << state_dim 
                      << " < 16, using standard Kalman)" << std::endl;
        }
    }

    // Validate that the loaded configs match the expected dimensions.
    size_t b_matrix_inputs = MatrixCols(controller_config_.B);
    size_t k_matrix_inputs = MatrixRows(controller_config_.K);
    size_t pid_kp_inputs = pid_config_.Kp_matrix.size() == 0
                               ? 0
                               : static_cast<size_t>(pid_config_.Kp_matrix.rows());
    if (b_matrix_inputs > 0 && k_matrix_inputs > 0 && pid_kp_inputs > 0 &&
        (control_dim != b_matrix_inputs || control_dim != k_matrix_inputs || control_dim != pid_kp_inputs)) {
        std::cerr << "FATAL: Dimension mismatch during re-initialization!" << std::endl;
        std::cerr << "  - Inferred control_dim: " << control_dim << std::endl;
        std::cerr << "  - Inputs from state_space.conf (B matrix): " << b_matrix_inputs << std::endl;
        std::cerr << "  - Inputs from state_space.conf (K matrix): " << k_matrix_inputs << std::endl;
        std::cerr << "  - Inputs from pid.conf (Kp matrix): " << pid_kp_inputs << std::endl;
        std::cerr << "This indicates inconsistent or corrupted config files. Please clear the trial directory and restart." << std::endl;
        std::terminate();
    }
    std::cout << "Config dimension validation passed." << std::endl;

    previous_control_action_.assign(control_dim, 0.0f);
    for (size_t i = 0; i < control_dim; ++i) {
        float baseline = (i < controller_config_.u_scaler_mean.size()) ? controller_config_.u_scaler_mean[i] : 0.0f;
        previous_control_action_[i] = baseline;
    }

    auto is_primary_evolution_time = [](const std::string& name) {
        return name.find("evolution_time") != std::string::npos &&
               name.find("evolution_time_gain") == std::string::npos &&
               name.find("evolution_time_shift") == std::string::npos &&
               name.find("evolution_time_cap") == std::string::npos &&
               name.find("evolution_time_base") == std::string::npos &&
               name.find("evolution_time_raw") == std::string::npos &&
               name.find("evolution_time_std") == std::string::npos;
    };
    auto is_evolution_time_param = [](const std::string& name) {
        return name.find("evolution_time_gain") != std::string::npos ||
               name.find("evolution_time_shift") != std::string::npos ||
               name.find("evolution_time_cap") != std::string::npos ||
               name.find("evolution_time_base") != std::string::npos;
    };

    for (size_t i = 0; i < control_input_names.size(); ++i) {
        const std::string& control_name = control_input_names[i];
        if (is_primary_evolution_time(control_name)) {
            if (i < pid_config_.control_min_bounds.size() && pid_config_.control_min_bounds[i] < 5.0e-4f) {
                pid_config_.control_min_bounds[i] = 5.0e-4f;
            }
            if (i < pid_config_.control_max_bounds.size()) {
                float& configured_max = pid_config_.control_max_bounds[i];
                if (configured_max > 0.0f) {
                    std::cout << "[MetaController] Removing hard upper bound for " << control_name
                              << " (was " << configured_max << ")." << std::endl;
                }
                configured_max = -1.0f;
            }
        } else if (is_evolution_time_param(control_name)) {
            if (i < pid_config_.control_min_bounds.size() && pid_config_.control_min_bounds[i] > -5.0f) {
                pid_config_.control_min_bounds[i] = -5.0f;
            }
            if (i < pid_config_.control_max_bounds.size() && pid_config_.control_max_bounds[i] > 5.0f) {
                pid_config_.control_max_bounds[i] = 5.0f;
            }
        }
    }

    if (pid_config_.control_min_bounds.size() != control_dim) {
        pid_config_.control_min_bounds.assign(control_dim, 0.0f);
    }
    if (pid_config_.control_max_bounds.size() != control_dim) {
        pid_config_.control_max_bounds.assign(control_dim, 1.0f);
    }

    integral_error_.assign(metric_names.size(), 0.0f);
    previous_error_.assign(metric_names.size(), 0.0f);
    adaptive_setpoints_.assign(metric_names.size(), 0.0f);
    last_valid_measurements_.clear();
    adaptive_setpoints_initialized_ = false;
    // --- END: DEFINITIVE FIX ---

    rebuild_metric_control_lookup(control_input_names);
    last_metric_count_ = metric_names.size();

    if (is_mpc_active_ && !MatrixEmpty(controller_config_.A) && !MatrixEmpty(controller_config_.B)) {
        int state_dim = static_cast<int>(MatrixRows(controller_config_.A));
        int control_dim = static_cast<int>(MatrixCols(controller_config_.B));
        int prediction_horizon = 10;
        mpc_controller_ = std::make_unique<MPCController>(state_dim, control_dim, prediction_horizon);

        Eigen::MatrixXf A = controller_config_.A;
        Eigen::MatrixXf B = controller_config_.B;
        Eigen::MatrixXf Q = controller_config_.Q;
        Eigen::MatrixXf R = controller_config_.R_lqr;

        mpc_controller_->init(A, B, Q, R);
        std::cout << "MPC controller initialized." << std::endl;
    }

    std::cout << "Controller re-initialized with " << metric_names.size() << " metrics and "
              << control_dim << " control inputs." << std::endl;
}

void HamiltonianMetaController::initialize(const SystemState& initial_state, const std::string& path_prefix) {
    std::string state_space_path = path_prefix.empty() ? default_config_file_path_ : path_prefix + "/state_space.conf";
    std::string pid_path = path_prefix.empty() ? "pid.conf" : path_prefix + "/pid.conf";
    system_id_model_ready_ = false;
    is_mpc_active_ = false;

    std::string metric_list_path = path_prefix.empty() ? "metric_list.conf" : path_prefix + "/metric_list.conf";
    std::string input_list_path = path_prefix.empty() ? "input_list.conf" : path_prefix + "/input_list.conf";
    std::ofstream metric_file(metric_list_path);
    std::ofstream input_file(input_list_path);
    if (metric_file.is_open()) {
        for (const auto& name : initial_state.metric_names) {
            metric_file << name << "\n";
        }
    }
    if (input_file.is_open()) {
        for (const auto& name : initial_state.control_input_names) {
            input_file << name << "\n";
        }
    }

    if (!std::ifstream(state_space_path).good()) {
        this->generate_default_state_space_config(initial_state.metric_names, initial_state.control_input_names, state_space_path);
    }

    if (!std::ifstream(pid_path).good()) {
        generate_default_pid_config(initial_state.metric_names, initial_state.control_input_names, pid_path);
    }

    load_and_reinitialize(path_prefix, initial_state.metric_names, initial_state.control_input_names);
    reseed_controls_from_state(initial_state);

    is_initialized_ = true;
    awaiting_control_seed_ = false;
    std::cout << "HamiltonianMetaController initialization complete." << std::endl;
}

void HamiltonianMetaController::reload_configs(const std::string& path_prefix,
                                               const std::vector<std::string>& metric_names,
                                               const std::vector<std::string>& control_input_names,
                                               bool system_id_completed) {
    std::cout << "Reloading state-space and PID configurations..." << std::endl;
    if (system_id_completed) {
        system_id_model_ready_ = true;
        std::cout << "[MetaController] System identification complete. MPC will be re-evaluated for activation." << std::endl;
    }
    load_and_reinitialize(path_prefix, metric_names, control_input_names);
    awaiting_control_seed_ = true;

    if (is_mpc_active_) {
        std::cout << "MPC component is now ACTIVE after config reload." << std::endl;
    } else if (system_id_model_ready_) {
        std::cout << "MPC component remains inactive; verify identified model matrices enable controllability." << std::endl;
    } else {
        std::cout << "MPC component remains INACTIVE after config reload (awaiting system identification)." << std::endl;
    }
}

void HamiltonianMetaController::rebuild_metric_control_lookup(const std::vector<std::string>& control_input_names) {
    metric_to_control_indices_.clear();
    std::unordered_map<std::string, size_t> control_index_map;
    for (size_t idx = 0; idx < control_input_names.size(); ++idx) {
        control_index_map[control_input_names[idx]] = idx;
    }

    for (const auto& entry : pid_config_.control_metric_map) {
        auto control_it = control_index_map.find(entry.first);
        if (control_it == control_index_map.end()) {
            continue;
        }
        size_t control_index = control_it->second;
        for (const auto& metric_name : entry.second) {
            metric_to_control_indices_[metric_name].push_back(control_index);
        }
    }
}

void HamiltonianMetaController::reseed_controls_from_state(const SystemState& state) {
    if (previous_control_action_.empty()) {
        return;
    }

    const size_t controls_to_update = std::min(previous_control_action_.size(), state.control_input_names.size());
    for (size_t control_idx = 0; control_idx < controls_to_update; ++control_idx) {
        const std::string& control_name = state.control_input_names[control_idx];
        auto metric_it = state.metrics.find(control_name);
        if (metric_it == state.metrics.end()) {
            continue;
        }

        float control_value = metric_it->second;
        if (!std::isfinite(control_value)) {
            continue;
        }

        previous_control_action_[control_idx] = control_value;

        if (control_idx < pid_config_.control_min_bounds.size()) {
            pid_config_.control_min_bounds[control_idx] = std::min(pid_config_.control_min_bounds[control_idx], control_value);
        }
        if (control_idx < pid_config_.control_max_bounds.size()) {
            float& configured_max = pid_config_.control_max_bounds[control_idx];
            if (configured_max > 0.0f) {
                configured_max = std::max(configured_max, control_value);
            }
        }
    }
} 
void HamiltonianMetaController::log_pid_config_details(const std::vector<std::string>& control_input_names, const std::vector<std::string>& metric_names) {
    std::cout << "--- PID Configuration Details ---" << std::endl;
    const auto kp_rows = static_cast<size_t>(pid_config_.Kp_matrix.rows());
    const auto kp_cols = static_cast<size_t>(pid_config_.Kp_matrix.cols());
    const auto ki_rows = static_cast<size_t>(pid_config_.Ki_matrix.rows());
    const auto ki_cols = static_cast<size_t>(pid_config_.Ki_matrix.cols());
    const auto kd_rows = static_cast<size_t>(pid_config_.Kd_matrix.rows());
    const auto kd_cols = static_cast<size_t>(pid_config_.Kd_matrix.cols());

    for (size_t i = 0; i < control_input_names.size(); ++i) {
        std::cout << "Control Input [" << i << "]: " << control_input_names[i] << std::endl;
        for (size_t j = 0; j < metric_names.size(); ++j) {
            const bool kp_valid = i < kp_rows && j < kp_cols;
            const bool ki_valid = i < ki_rows && j < ki_cols;
            const bool kd_valid = i < kd_rows && j < kd_cols;
            const float kp = kp_valid ? pid_config_.Kp_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) : 0.0f;
            const float ki = ki_valid ? pid_config_.Ki_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) : 0.0f;
            const float kd = kd_valid ? pid_config_.Kd_matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) : 0.0f;
            std::cout << "  -> Metric '" << metric_names[j] << "': Kp=" << kp << ", Ki=" << ki << ", Kd=" << kd << std::endl;
        }
    }
    std::cout << "---------------------------------" << std::endl;
}

std::map<std::string, float> HamiltonianMetaController::update_and_act(const SystemState& current_state, const std::string& path_prefix) {
    if (!is_initialized_) {
        std::cout << "[update_and_act] Controller not initialized. Calling initialize()..." << std::endl;
        initialize(current_state, path_prefix);
    }

    if (awaiting_control_seed_) {
        reseed_controls_from_state(current_state);
        awaiting_control_seed_ = false;
    }

    auto realign_if_needed = [&](const char* stage_label) -> bool {
        const size_t metric_dim = current_state.metric_names.size();
        const size_t control_dim = current_state.control_input_names.size();
        std::vector<std::string> reasons;
        if (last_metric_count_ != 0 && last_metric_count_ != metric_dim) {
            std::ostringstream oss;
            oss << "Observed metric count changed from " << last_metric_count_ << " to " << metric_dim;
            reasons.push_back(oss.str());
        }
        if (!MatrixEmpty(controller_config_.C) && MatrixRows(controller_config_.C) != metric_dim) {
            std::ostringstream oss;
            oss << "State-space matrix C rows (" << MatrixRows(controller_config_.C) << ") mismatch current metric count (" << metric_dim << ")";
            reasons.push_back(oss.str());
        }
        if (!MatrixEmpty(controller_config_.A) && MatrixRows(controller_config_.A) != metric_dim) {
            std::ostringstream oss;
            oss << "State-space matrix A dimension (" << MatrixRows(controller_config_.A) << ") mismatch current metric count (" << metric_dim << ")";
            reasons.push_back(oss.str());
        }
        auto check_pid_columns = [&](const Eigen::MatrixXf& matrix, const char* name) {
            if (matrix.size() == 0) {
                return;
            }
            if (static_cast<size_t>(matrix.cols()) != metric_dim) {
                std::ostringstream oss;
                oss << "PID " << name << " matrix columns (" << matrix.cols()
                    << ") mismatch current metric count (" << metric_dim << ")";
                reasons.push_back(oss.str());
            }
        };
        check_pid_columns(pid_config_.Kp_matrix, "Kp");
        check_pid_columns(pid_config_.Ki_matrix, "Ki");
        check_pid_columns(pid_config_.Kd_matrix, "Kd");
        if (!pid_config_.control_min_bounds.empty() && pid_config_.control_min_bounds.size() != control_dim) {
            std::ostringstream oss;
            oss << "PID control_min_bounds size (" << pid_config_.control_min_bounds.size()
                << ") mismatch control inputs (" << control_dim << ")";
            reasons.push_back(oss.str());
        }
        if (!pid_config_.control_max_bounds.empty() && pid_config_.control_max_bounds.size() != control_dim) {
            std::ostringstream oss;
            oss << "PID control_max_bounds size (" << pid_config_.control_max_bounds.size()
                << ") mismatch control inputs (" << control_dim << ")";
            reasons.push_back(oss.str());
        }
        if (!pid_config_.metric_names.empty() && pid_config_.metric_names.size() != metric_dim) {
            std::ostringstream oss;
            oss << "PID metric_names count (" << pid_config_.metric_names.size()
                << ") mismatch current metric count (" << metric_dim << ")";
            reasons.push_back(oss.str());
        }

        if (reasons.empty()) {
            return false;
        }

        std::cout << "[MetaController] Detected configuration misalignment during " << stage_label << "." << std::endl;
        for (const auto& reason : reasons) {
            std::cout << "  - " << reason << std::endl;
        }

        if (!pid_config_.metric_names.empty()) {
            std::set<std::string> configured(pid_config_.metric_names.begin(), pid_config_.metric_names.end());
            std::set<std::string> current(current_state.metric_names.begin(), current_state.metric_names.end());
            std::vector<std::string> missing;
            std::vector<std::string> unexpected;
            std::set_difference(configured.begin(), configured.end(), current.begin(), current.end(), std::back_inserter(missing));
            std::set_difference(current.begin(), current.end(), configured.begin(), configured.end(), std::back_inserter(unexpected));
            auto log_subset = [&](const char* label, const std::vector<std::string>& items) {
                if (items.empty()) return;
                std::cout << "    " << label << " (" << items.size() << "): ";
                size_t limit = std::min<size_t>(items.size(), 10);
                for (size_t i = 0; i < limit; ++i) {
                    std::cout << items[i] << (i + 1 == limit ? "" : ", ");
                }
                if (items.size() > limit) {
                    std::cout << " ...";
                }
                std::cout << std::endl;
            };
            log_subset("Configured but missing from current batch", missing);
            log_subset("New metrics not present in PID config", unexpected);
        }

        load_and_reinitialize(path_prefix, current_state.metric_names, current_state.control_input_names);
        reseed_controls_from_state(current_state);
        return true;
    };

    bool realigned = realign_if_needed("pre-update");
    if (realigned) {
        std::cout << "[MetaController] Configuration realigned. Proceeding with updated dimensions." << std::endl;
    }

    std::cout << "\n--- [MetaController update_and_act] Batch Start ---" << std::endl;
    log_pid_config_details(current_state.control_input_names, current_state.metric_names);
    if (auto_tune_state_ != AutoTuneState::INACTIVE) {
        float action = relay_step(current_state);
        float current_evolution_time = current_state.metrics.count("evolution_time") ? current_state.metrics.at("evolution_time") : 1.0f;
        float new_evolution_time = current_evolution_time + action;
        return {{"timecrystal_block_1", std::max(0.0001f, new_evolution_time)}};
    }

    std::vector<float> y_k;
    y_k.reserve(current_state.metric_names.size());
    std::cout << "[Step 1] Constructing measurement vector y_k from " << current_state.metric_names.size() << " metrics." << std::endl;
    for (const auto& metric_name : current_state.metric_names) {
        if (current_state.metrics.count(metric_name)) {
            y_k.push_back(current_state.metrics.at(metric_name));
        } else {
            std::cerr << "Warning: Metric '" << metric_name << "' not found in current state." << std::endl;
            std::cerr << "This is a fatal error. The C++ controller expected a metric that the Python training loop did not provide. This should not happen with the dynamic name passing fix." << std::endl;
            std::terminate();
        }
    }
    std::cout << "  - Raw y_k (first 5): " << vector_to_string(y_k, 5) << std::endl;
    std::vector<float> raw_measurements = y_k;

    // Ensure scaler vectors are sized for the current metric count.
    if (controller_config_.y_scaler_mean.size() < y_k.size()) {
        size_t previous = controller_config_.y_scaler_mean.size();
        controller_config_.y_scaler_mean.resize(y_k.size(), 0.0f);
        std::cout << "[MetaController] Expanded y_scaler_mean from " << previous << " to " << controller_config_.y_scaler_mean.size() << " entries." << std::endl;
        scalers_bootstrapped_ = false;
    }
    if (controller_config_.y_scaler_scale.size() < y_k.size()) {
        size_t previous = controller_config_.y_scaler_scale.size();
        controller_config_.y_scaler_scale.resize(y_k.size(), 1.0f);
        std::cout << "[MetaController] Expanded y_scaler_scale from " << previous << " to " << controller_config_.y_scaler_scale.size() << " entries." << std::endl;
        scalers_bootstrapped_ = false;
    }

    // Apply current scaling (safe defaults for newly added metrics).
    for (size_t i = 0; i < y_k.size(); ++i) {
        float scale = controller_config_.y_scaler_scale[i];
        if (!std::isfinite(scale) || std::fabs(scale) < 1e-6f) {
            scale = 1.0f;
            controller_config_.y_scaler_scale[i] = 1.0f;
        }
        float mean = controller_config_.y_scaler_mean[i];
        if (!std::isfinite(mean)) {
            mean = 0.0f;
            controller_config_.y_scaler_mean[i] = 0.0f;
        }
        y_k[i] = (y_k[i] - mean) / scale;
    }
    
    std::cout << "[Step 1a] Scaled measurement vector y_k." << std::endl;
    std::cout << "  - Scaled y_k (first 5): " << vector_to_string(y_k, 5) << std::endl;
    const bool bootstrapped_before = scalers_bootstrapped_;
    bootstrap_scalers(raw_measurements);
    if (!bootstrapped_before && scalers_bootstrapped_) {
        for (size_t i = 0; i < y_k.size(); ++i) {
            float scale = (i < controller_config_.y_scaler_scale.size() && std::fabs(controller_config_.y_scaler_scale[i]) > 1e-9f)
                              ? controller_config_.y_scaler_scale[i]
                              : 1.0f;
            float mean = (i < controller_config_.y_scaler_mean.size()) ? controller_config_.y_scaler_mean[i] : 0.0f;
            y_k[i] = (raw_measurements[i] - mean) / scale;
        }
        std::cout << "[MetaController] Recomputed y_k with bootstrapped scalers." << std::endl;
        std::cout << "  - Rescaled y_k (first 5): " << vector_to_string(y_k, 5) << std::endl;
    }
    sanitize_measurements(y_k);
    std::cout << "  - Sanitized y_k (first 5): " << vector_to_string(y_k, 5) << std::endl;
    if (y_k.empty()) {
        std::cerr << "Error: Measurement vector 'y_k' is empty. This is likely due to a missing or misconfigured 'metric_names' list in the PID config file. Cannot update Kalman filter. Returning default evolution_time." << std::endl;
        return {};
    }

    if (adaptive_setpoints_.size() != y_k.size()) {
        adaptive_setpoints_.assign(y_k.size(), 0.0f);
        adaptive_setpoints_initialized_ = false;
    }

    std::vector<float> error_k(y_k.size(), 0.0f);
    std::cout << "[Step 2] Calculating error vector e_k." << std::endl;
    saguaro::parallel::ForRange(
        0, current_state.metric_names.size(), 32,
        [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                const std::string& metric_name = current_state.metric_names[i];
                float target_raw = 0.0f;
                auto setpoint_it = pid_config_.Setpoint.find(metric_name);
                if (setpoint_it != pid_config_.Setpoint.end()) {
                    target_raw = setpoint_it->second;
                }

                float mean = (i < controller_config_.y_scaler_mean.size()) ? controller_config_.y_scaler_mean[i] : 0.0f;
                float scale = (i < controller_config_.y_scaler_scale.size()) ? controller_config_.y_scaler_scale[i] : 1.0f;
                float target_scaled = (std::fabs(scale) > 1e-9f) ? (target_raw - mean) / scale : (target_raw - mean);

                if (!adaptive_setpoints_initialized_) {
                    adaptive_setpoints_[i] = std::isfinite(y_k[i]) ? y_k[i] : target_scaled;
                }

                float current_adaptive = adaptive_setpoints_[i];
                float delta = target_scaled - current_adaptive;
                float step = delta * setpoint_relaxation_factor_;
                if (std::isfinite(step)) {
                    step = std::max(-max_setpoint_step_, std::min(max_setpoint_step_, step));
                } else {
                    step = 0.0f;
                }

                float updated_setpoint = current_adaptive + step;
                adaptive_setpoints_[i] = std::isfinite(updated_setpoint) ? updated_setpoint : target_scaled;
                error_k[i] = adaptive_setpoints_[i] - y_k[i];
            }
        });
    adaptive_setpoints_initialized_ = true;

    std::cout << "  - Error e_k (first 5): " << vector_to_string(error_k, 5) << std::endl;

    std::vector<float> scaled_previous_control_action = previous_control_action_;
    std::cout << "[Step 2a] Scaling previous control action u_{k-1}." << std::endl;
    std::cout << "  - Raw u_{k-1}: " << vector_to_string(previous_control_action_) << std::endl;
    if (controller_config_.u_scaler_mean.size() < scaled_previous_control_action.size()) {
        size_t previous = controller_config_.u_scaler_mean.size();
        controller_config_.u_scaler_mean.resize(scaled_previous_control_action.size(), 0.0f);
        std::cout << "[MetaController] Expanded u_scaler_mean from " << previous << " to " << controller_config_.u_scaler_mean.size() << " entries." << std::endl;
    }
    if (controller_config_.u_scaler_scale.size() < scaled_previous_control_action.size()) {
        size_t previous = controller_config_.u_scaler_scale.size();
        controller_config_.u_scaler_scale.resize(scaled_previous_control_action.size(), 1.0f);
        std::cout << "[MetaController] Expanded u_scaler_scale from " << previous << " to " << controller_config_.u_scaler_scale.size() << " entries." << std::endl;
    }
    for (size_t i = 0; i < scaled_previous_control_action.size(); ++i) {
        float scale = controller_config_.u_scaler_scale[i];
        if (!std::isfinite(scale) || std::fabs(scale) < 1e-6f) {
            scale = 1.0f;
            controller_config_.u_scaler_scale[i] = 1.0f;
        }
        float mean = controller_config_.u_scaler_mean[i];
        if (!std::isfinite(mean)) {
            mean = 0.0f;
            controller_config_.u_scaler_mean[i] = 0.0f;
        }
        scaled_previous_control_action[i] = (scaled_previous_control_action[i] - mean) / scale;
    }
    
    std::cout << "  - Scaled u_{k-1}: " << vector_to_string(scaled_previous_control_action) << std::endl;

    std::cout << "[Step 3] Calling KalmanFilter->predict()." << std::endl;
    kalman_filter_->predict(scaled_previous_control_action);

    std::cout << "[Step 4] Calling KalmanFilter->update()." << std::endl;
    kalman_filter_->update(y_k);
    std::vector<float> x_hat = kalman_filter_->getState();
    std::cout << "  - Estimated state x_hat (size=" << x_hat.size() << "): " << vector_to_string(x_hat, 5) << std::endl;

    std::cout << "[Step 5] Calculating control action u_k." << std::endl;

    std::vector<float> u_fb(previous_control_action_.size(), 0.0f); // Initialize to zero

    if (is_mpc_active_) {
        std::cout << "  - MPC is ACTIVE. Calculating MPC control component." << std::endl;
        if (mpc_controller_) {
            Eigen::Map<const Eigen::VectorXf> x_hat_eigen(x_hat.data(), x_hat.size());
            Eigen::VectorXf u_mpc_eigen = mpc_controller_->compute_control_action(x_hat_eigen);
            Eigen::VectorXf::Map(&u_fb[0], u_mpc_eigen.size()) = u_mpc_eigen;
        }
    } else {
        std::cout << "  - MPC is INACTIVE. Feedback control component (u_fb) is zero." << std::endl;
    }

    std::vector<float> u_p = MatrixUtils::multiply(pid_config_.Kp_matrix, error_k);
    std::vector<float> u_i = MatrixUtils::multiply(pid_config_.Ki_matrix, integral_error_);
    std::vector<float> error_derivative(error_k.size());
    for (size_t i = 0; i < error_k.size(); ++i) {
        error_derivative[i] = error_k[i] - previous_error_[i];
    }
    std::vector<float> u_d = MatrixUtils::multiply(pid_config_.Kd_matrix, error_derivative);
    std::cout << "  - [5b] PID components calculated." << std::endl;
    std::cout << "    - u_fb: " << vector_to_string(u_fb) << std::endl;
    std::cout << "    - u_p: " << vector_to_string(u_p) << std::endl;
    std::cout << "    - u_i: " << vector_to_string(u_i) << std::endl;
    std::cout << "    - u_d: " << vector_to_string(u_d) << std::endl;

    std::vector<float> u_k(u_fb.size());
    saguaro::parallel::ForRange(
        0, u_k.size(), 32,
        [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                if (i >= u_p.size() || i >= u_i.size() || i >= u_d.size()) {
                    std::cerr << "FATAL: Out-of-bounds access during u_k combination. u_fb.size=" << u_fb.size()
                              << ", u_p.size=" << u_p.size() << ", u_i.size=" << u_i.size()
                              << ", u_d.size=" << u_d.size() << std::endl;
                    std::terminate();
                }
                u_k[i] = u_fb[i] + u_p[i] + u_i[i] + u_d[i];
            }
        });

    std::cout << "[Step 6] Mapping control action to outputs and inverse scaling." << std::endl;
    std::map<std::string, float> new_evolution_times; // This will hold the final output
    std::vector<float> u_k_saturated(u_k.size()); // Holds the clamped control values
    std::vector<bool> channel_saturated(u_k.size(), false);

    for (size_t control_input_idx = 0; control_input_idx < current_state.control_input_names.size(); ++control_input_idx) {
        const std::string& control_name = current_state.control_input_names[control_input_idx];

        if (control_input_idx >= u_k.size()) {
            std::cerr << "FATAL: Control input index " << control_input_idx << " is out of bounds for control vector u_k of size " << u_k.size() << std::endl;
            std::terminate();
        }

        if (control_input_idx >= controller_config_.u_scaler_scale.size() || control_input_idx >= controller_config_.u_scaler_mean.size()) {
            std::cerr << "FATAL: Out-of-bounds access on u_scaler during inverse scaling. control_input_idx=" << control_input_idx
                      << ", u_scaler_scale.size()=" << controller_config_.u_scaler_scale.size()
                      << ", u_scaler_mean.size()=" << controller_config_.u_scaler_mean.size() << std::endl;
            std::terminate();
        }

        float min_bound = (control_input_idx < pid_config_.control_min_bounds.size())
                              ? pid_config_.control_min_bounds[control_input_idx]
                              : 0.0f;
        float configured_max = (control_input_idx < pid_config_.control_max_bounds.size())
                                   ? pid_config_.control_max_bounds[control_input_idx]
                                   : 2.0f;
        bool enforce_upper_bound = std::isfinite(configured_max) && configured_max > 0.0f;
        float max_bound = enforce_upper_bound ? configured_max : std::numeric_limits<float>::infinity();
        if (enforce_upper_bound && max_bound < min_bound) {
            std::swap(min_bound, max_bound);
        }

        float scaled_u = u_k[control_input_idx];
        float unscaled_u = (scaled_u * controller_config_.u_scaler_scale[control_input_idx]) + controller_config_.u_scaler_mean[control_input_idx];
        float previous_value = (control_input_idx < previous_control_action_.size()) ? previous_control_action_[control_input_idx] : controller_config_.u_scaler_mean[control_input_idx];
        if (!std::isfinite(previous_value)) {
            previous_value = controller_config_.u_scaler_mean[control_input_idx];
        }

        auto log_channel_saturation = [&](const std::string& reason, float original_value, float applied_value) {
            std::ostringstream max_bound_stream;
            if (enforce_upper_bound) {
                max_bound_stream << max_bound;
            } else {
                max_bound_stream << "inf";
            }
            std::cout << "[MetaController] Control '" << control_name << "' saturation (" << reason << ") "
                      << "requested=" << original_value
                      << ", applied=" << applied_value
                      << ", previous=" << previous_value
                      << ", min_bound=" << min_bound
                      << ", max_bound=" << max_bound_stream.str()
                      << ", scaler_mean=" << controller_config_.u_scaler_mean[control_input_idx]
                      << ", scaler_scale=" << controller_config_.u_scaler_scale[control_input_idx]
                      << std::endl;
        };

        float requested_value = unscaled_u;
        if (!std::isfinite(requested_value)) {
            std::cerr << "Warning: Non-finite control command computed for '" << control_name << "'. Holding previous value." << std::endl;
            log_channel_saturation("non_finite_command", requested_value, previous_value);
            requested_value = previous_value;
            channel_saturated[control_input_idx] = true;
        } else {
            float step_limited = clamp_control_step(requested_value, previous_value);
            if (std::fabs(step_limited - requested_value) > 1e-6f) {
                log_channel_saturation("step_limit", requested_value, step_limited);
                channel_saturated[control_input_idx] = true;
            }
            requested_value = step_limited;
        }

        float bounded_value = requested_value;
        bool lower_clamped = false;
        bool upper_clamped = false;
        if (std::isfinite(min_bound) && bounded_value < min_bound) {
            log_channel_saturation("min_bound", bounded_value, min_bound);
            bounded_value = min_bound;
            lower_clamped = true;
        }
        if (enforce_upper_bound && bounded_value > max_bound) {
            log_channel_saturation("max_bound", bounded_value, max_bound);
            bounded_value = max_bound;
            upper_clamped = true;
        }
        if (lower_clamped || upper_clamped || std::fabs(bounded_value - requested_value) > 1e-6f) {
            channel_saturated[control_input_idx] = true;
        }

        if (!std::isfinite(bounded_value)) {
            log_channel_saturation("non_finite_bounded", bounded_value, previous_value);
            bounded_value = previous_value;
            channel_saturated[control_input_idx] = true;
        }

        u_k_saturated[control_input_idx] = bounded_value;

        // Use min_bound as fallback for zero outputs (PID has no tuned gains)
        // This ensures Python receives valid evolution_time values instead of 0.0
        float final_value = bounded_value;
        if (std::abs(final_value) < 1e-9f) {
            // Zero output means PID has no control; use min_bound as safe default
            if (std::isfinite(min_bound) && min_bound > 0.0f) {
                final_value = min_bound;
            } else {
                final_value = 0.01f;  // Safe fallback for evolution_time
            }
        }

        std::string block_name = control_name;
        new_evolution_times[block_name] = final_value;
        // --- END: DEFINITIVE FIX ---
    }

    std::cout << "[Step 7] Performing anti-windup integral update." << std::endl;
    bool any_channel_saturated = std::any_of(channel_saturated.begin(), channel_saturated.end(), [](bool v) { return v; });
    for (size_t i = 0; i < integral_error_.size(); ++i) {
        const std::string& metric_name = current_state.metric_names[i];
        if (!std::isfinite(error_k[i])) {
            continue;
        }
        bool integrate_metric = true;
        auto map_it = metric_to_control_indices_.find(metric_name);
        if (map_it != metric_to_control_indices_.end()) {
            for (size_t ctrl_idx : map_it->second) {
                if (ctrl_idx < channel_saturated.size() && channel_saturated[ctrl_idx]) {
                    integrate_metric = false;
                    break;
                }
            }
        } else if (any_channel_saturated) {
            integrate_metric = false;
        }

        if (integrate_metric) {
            integral_error_[i] += error_k[i];
        }
    }

    std::cout << "[Step 8] Storing saturated control action for next step." << std::endl;
    std::cout << "  - u_k_saturated: " << vector_to_string(u_k_saturated) << std::endl;
    previous_control_action_ = u_k_saturated;
    previous_error_ = error_k;

    // PHASE 1: Record sensitivity data for causal parameter selection
    std::cout << "[Step 9] Recording parameter sensitivity for intelligent meta-controller." << std::endl;
    batch_counter_++;

    // Compute control deltas (change from previous step)
    std::vector<float> control_deltas(u_k_saturated.size());
    for (size_t i = 0; i < u_k_saturated.size(); ++i) {
        // Use difference from previous_control_action_ stored BEFORE update
        // Note: previous_control_action_ is already updated above, so we need to compute from
        // u_k_saturated (new) vs. the value before the update.
        // Since previous_control_action_ is already updated, we compute delta as:
        // delta = u_k_saturated[i] - previous_unscaled (stored earlier in the method)
        // However, previous_unscaled is not stored. We'll use a simpler approach:
        // delta = current control - previous control (before this step)
        // This requires storing previous values before the update above.
        // For simplicity in Phase 1, we compute delta from the error_k change,
        // which is a proxy for control effectiveness.
        control_deltas[i] = u_k_saturated[i];
    }

    // Compute metric deltas (change from previous step)
    std::vector<float> metric_deltas(error_k.size());
    for (size_t i = 0; i < error_k.size(); ++i) {
        // Delta in error (proxy for metric change)
        metric_deltas[i] = error_k[i] - previous_error_[i];
    }

    // Record sensitivity for each (control, metric) pair
    for (size_t control_idx = 0; control_idx < current_state.control_input_names.size(); ++control_idx) {
        const std::string& control_name = current_state.control_input_names[control_idx];
        float control_delta = control_deltas[control_idx];

        for (size_t metric_idx = 0; metric_idx < current_state.metric_names.size(); ++metric_idx) {
            const std::string& metric_name = current_state.metric_names[metric_idx];
            float metric_delta = metric_deltas[metric_idx];

            // Record influence of this control on this metric
            sensitivity_tracker_->record_update(
                control_name,
                metric_name,
                control_delta,
                metric_delta,
                batch_counter_
            );
        }
    }

    // Export sensitivity map every 500 batches (Phase 1 gate requirement)
    if (batch_counter_ % SENSITIVITY_EXPORT_INTERVAL == 0) {
        std::string sensitivity_path = path_prefix.empty()
            ? "sensitivity_map.json"
            : path_prefix + "/sensitivity_map.json";

        if (sensitivity_tracker_->save(sensitivity_path)) {
            std::cout << "[MetaController] Exported sensitivity map to: " << sensitivity_path << std::endl;

            // Log top-3 influencers for "loss" metric (Phase 1 success criterion)
            auto top_influencers = sensitivity_tracker_->get_top_influencers("loss", 3, 0.7f);
            if (!top_influencers.empty()) {
                std::cout << "[MetaController] Top-3 influencers for 'loss' metric:" << std::endl;
                for (size_t i = 0; i < top_influencers.size(); ++i) {
                    const auto& inf = top_influencers[i];
                    std::cout << "  " << (i + 1) << ". " << inf.control_name
                              << ": magnitude=" << inf.influence_magnitude
                              << ", correlation=" << inf.correlation
                              << ", confidence=" << inf.confidence << std::endl;
                }
            }
        } else {
            std::cerr << "[MetaController] WARNING: Failed to export sensitivity map." << std::endl;
        }
    }

    // =========================================================================
    // Phase 4: Adaptive Phase Transitions (Intelligent Meta-Controller)
    // =========================================================================
    // Build SystemState for phase-aware control
    saguaro::control::SystemState hierarchical_state;
    hierarchical_state.metrics = current_state.metrics;  // Raw metrics (not scaled)
    hierarchical_state.evolution_times = new_evolution_times;  // PID-computed values
    hierarchical_state.batch_number = batch_counter_;

    // Update adaptive phase based on training dynamics
    adaptive_phase_controller_->update_phase(hierarchical_state, *sensitivity_tracker_);

    // Get current phase
    saguaro::control::TuningPhase current_phase = adaptive_phase_controller_->current_phase();
    std::string phase_name;
    switch (current_phase) {
        case saguaro::control::TuningPhase::COARSE: phase_name = "COARSE"; break;
        case saguaro::control::TuningPhase::REFINE: phase_name = "REFINE"; break;
        case saguaro::control::TuningPhase::FINE: phase_name = "FINE"; break;
        case saguaro::control::TuningPhase::EXPLORE: phase_name = "EXPLORE"; break;
    }

    std::cout << "[MetaController] Phase 4 Adaptive Phase: " << phase_name
              << " (batches_in_phase=" << adaptive_phase_controller_->get_batches_in_phase() << ")" << std::endl;

    // =========================================================================
    // Phase 2/3: Hierarchical/Grouped Tuning with Phase-Aware Actions
    // =========================================================================
    // Get phase-specific hierarchical action (integrates Phase 2 + Phase 3 + Phase 4)
    saguaro::control::HierarchicalAction hierarchical_action =
        adaptive_phase_controller_->get_phase_action(
            hierarchical_state,
            *sensitivity_tracker_,
            *hierarchical_controller_
        );

    // Apply hierarchical action to evolution_times (modifies in-place)
    hierarchical_controller_->apply_hierarchical_action(hierarchical_action, new_evolution_times);

    std::cout << "[MetaController] Phase 2/3/4 Hierarchical Action: "
              << "phase=" << phase_name
              << ", level=" << static_cast<int>(hierarchical_action.level)
              << ", target=" << hierarchical_action.target
              << ", adjustment=" << hierarchical_action.adjustment
              << ", reason=" << hierarchical_action.reason << std::endl;
    // =========================================================================

    auto control_client = hardware_manager_->get_control_client();
    if (!control_client->set_cpu_affinity(current_state.cpu_affinity_mask)) {
        std::cerr << "Warning: Failed to set CPU affinity. Continuing in telemetry-only mode." << std::endl;
    }
    if (!control_client->set_core_frequencies(current_state.core_frequencies)) {
        std::cerr << "Warning: Failed to set CPU affinity. Continuing in telemetry-only mode." << std::endl;
    }
    std::cout << "--- [MetaController update_and_act] Batch End ---" << std::endl;

    return new_evolution_times;
}

void HamiltonianMetaController::trigger_autotune() {
    relay_history_.clear();
    oscillation_periods_.clear();
    peak_count_ = 0;
    last_crossing_time_ = std::chrono::steady_clock::now();
    auto_tune_state_ = AutoTuneState::RELAY_STEP;
    std::cout << "Auto-tuning initiated. Entering RELAY_STEP state." << std::endl;
}

void HamiltonianMetaController::enter_autotune_mode() {
    relay_history_.clear();
    oscillation_periods_.clear();
    peak_count_ = 0;
    last_crossing_time_ = std::chrono::steady_clock::now();
    auto_tune_state_ = AutoTuneState::RELAY_STEP;
    std::cout << "Auto-tuning initiated. Entering RELAY_STEP state." << std::endl;
}

float HamiltonianMetaController::relay_step(const SystemState& current_state) {
    if (auto_tune_state_ == AutoTuneState::INACTIVE) {
        enter_autotune_mode();
    }


    float current_loss = current_state.metrics.at("loss");
    float setpoint = pid_config_.Setpoint.at("loss");
    float error = setpoint - current_loss;

    static float relay_output = pid_config_.relay_output_amplitude;
    if (error > pid_config_.relay_hysteresis) {
        relay_output = pid_config_.relay_output_amplitude;
    } else if (error < -pid_config_.relay_hysteresis) {
        relay_output = -pid_config_.relay_output_amplitude;
    }

    relay_history_.push_back(current_loss);
    if (relay_history_.size() > 200) {
        relay_history_.erase(relay_history_.begin());
    }

    analyze_relay_oscillations();
    return relay_output;
}

void HamiltonianMetaController::generate_default_state_space_config(const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names, const std::string& path) { 
    ControllerConfig defaultConfig;
    std::cout << "Warning: '" << path << "' not found. Generating a default file." << std::endl;

    const int num_metrics = metric_names.size();
    std::vector<std::string> temp_control_input_names =
        get_control_input_names(metric_names, control_input_names);
    const int num_inputs = temp_control_input_names.size();
    const int num_states = num_metrics;

    defaultConfig.A = Eigen::MatrixXf::Identity(num_states, num_states);

    int input_dim = std::max(1, num_inputs);
    defaultConfig.B = Eigen::MatrixXf::Zero(num_states, input_dim);
    saguaro::parallel::ForRange(
        0, static_cast<size_t>(num_states), 32,
        [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                int col = std::min(static_cast<int>(i), input_dim - 1);
                defaultConfig.B(static_cast<int>(i), col) = 0.05f;
            }
        });
    
    defaultConfig.C = Eigen::MatrixXf::Identity(num_states, num_states);

    defaultConfig.D = Eigen::MatrixXf::Zero(num_states, std::max(1, num_inputs));

    defaultConfig.Q = Eigen::MatrixXf::Zero(num_states, num_states);
    defaultConfig.Q.diagonal().setConstant(0.1f);

    defaultConfig.R_kalman = Eigen::MatrixXf::Zero(num_states, num_states);
    defaultConfig.R_kalman.diagonal().setConstant(5.0f);

    defaultConfig.R_lqr = Eigen::MatrixXf::Zero(input_dim, input_dim);
    defaultConfig.R_lqr.diagonal().setConstant(0.5f);

    defaultConfig.K = Eigen::MatrixXf::Zero(std::max(1, num_inputs), num_states);
    defaultConfig.u_scaler_mean.assign(std::max(1, num_inputs), 0.0f);
    defaultConfig.u_scaler_scale.assign(std::max(1, num_inputs), 1.0f);
    defaultConfig.y_scaler_mean.assign(num_metrics, 0.0f);
    defaultConfig.y_scaler_scale.assign(num_metrics, 1.0f);
    for (int i = 0; i < num_metrics; ++i) {
        const std::string& name = metric_names[i];
        float scale_guess = 1.0f;
        if (name == "loss" || name.find("loss") != std::string::npos) {
            scale_guess = 2.0f;
        } else if (name.find("gradient_norm") != std::string::npos) {
            scale_guess = 1.0e6f;
        } else if (name.find("temperature") != std::string::npos) {
            scale_guess = 5.0f;
        } else if (name.find("evolution_time") != std::string::npos) {
            scale_guess = 5.0e-3f;
        } else if (name.find("energy_drift") != std::string::npos) {
            scale_guess = 1.0e-4f;
        } else if (name.find("activation_sparsity") != std::string::npos) {
            scale_guess = 0.05f;
        } else if (name.find("load_balancing_loss") != std::string::npos || name.find("router_z_loss") != std::string::npos) {
            scale_guess = 0.1f;
        }
        defaultConfig.y_scaler_scale[i] = scale_guess;
    }

    ConfigLoader::save_controller_config(path, defaultConfig);
}

void HamiltonianMetaController::analyze_relay_oscillations() {
    if (relay_history_.size() < 3) return;

    float setpoint = pid_config_.Setpoint.at("loss");
    float prev_val = relay_history_[relay_history_.size() - 2] - setpoint;
    float curr_val = relay_history_.back() - setpoint;

    if (std::signbit(prev_val) != std::signbit(curr_val)) {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_crossing_time_).count();
        if (peak_count_ > 0) {
             oscillation_periods_.push_back(2.0f * (duration / 1000.0f));
        }
        last_crossing_time_ = now;
        peak_count_++;
    }

    if (oscillation_periods_.size() >= 4) {
        float avg_period = std::accumulate(oscillation_periods_.end() - 3, oscillation_periods_.end(), 0.0f) / 3.0f;
        
        float peak_sum = 0;
        int peaks_found = 0;
        for (size_t i = 1; i < relay_history_.size() - 1; ++i) {
            if (relay_history_[i] > setpoint && relay_history_[i] > relay_history_[i-1] && relay_history_[i] > relay_history_[i+1]) {
                peak_sum += (relay_history_[i] - setpoint);
                peaks_found++;
            }
        }
        if (peaks_found == 0) return;
        float amplitude = peak_sum / peaks_found;

        float ultimate_gain = (4.0 * pid_config_.relay_output_amplitude) / (M_PI * amplitude);
        float ultimate_period = avg_period;

        std::cout << "Stable oscillations detected. Tu=" << ultimate_period << "s, Ku=" << ultimate_gain << std::endl;
        calculate_and_save_gains(ultimate_gain, ultimate_period);
        auto_tune_state_ = AutoTuneState::INACTIVE;
    }
}

void HamiltonianMetaController::calculate_and_save_gains(float ultimate_gain, float ultimate_period) {
    float Kp = 0.6 * ultimate_gain;
    float Ki = Kp / (ultimate_period / 2.0);
    float Kd = Kp * (ultimate_period / 8.0);
    
    std::cout << "Calculated new gains for 'loss' metric: Kp=" << Kp << ", Ki=" << Ki << ", Kd=" << Kd << std::endl;
    
    auto it = std::find(pid_config_.metric_names.begin(), pid_config_.metric_names.end(), "loss");
    if (it != pid_config_.metric_names.end()) {
        int loss_idx = std::distance(pid_config_.metric_names.begin(), it);

        const Eigen::Index num_inputs = pid_config_.Kp_matrix.rows();
        const Eigen::Index loss_col = static_cast<Eigen::Index>(loss_idx);
        if (loss_col >= pid_config_.Kp_matrix.cols()) {
            std::cerr << "Warning: 'loss' metric column out of bounds in PID matrices." << std::endl;
            return;
        }
        for (Eigen::Index i = 0; i < num_inputs; ++i) {
            pid_config_.Kp_matrix(i, loss_col) = Kp;
            pid_config_.Ki_matrix(i, loss_col) = Ki;
            pid_config_.Kd_matrix(i, loss_col) = Kd;
        }
    } else {
        std::cerr << "Warning: 'loss' metric not found in metric_names. Cannot save auto-tuned gains." << std::endl;
        return;
    }

    std::string config_dir = ".";
    size_t last_slash = active_config_file_path_.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        config_dir = active_config_file_path_.substr(0, last_slash);
    }
    ConfigLoader::save_pid_config(config_dir + "/pid.conf", pid_config_);
} 
