#ifndef HAMILTONIAN_META_CONTROLLER_H_
#define HAMILTONIAN_META_CONTROLLER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <chrono>
#include <memory>
#include <limits>

#include "config/state_space_config.h"
#include "config/pid_config.h"

#include "controllers/kalman_filter.h" // Include the Kalman Filter
#include "controllers/mpc_controller.h" // Include the MPC Controller
#include "controllers/hardware/hardware_manager.h"
#include "controllers/sensitivity_tracker.h" // Phase 1: Intelligent meta-controller
#include "controllers/hierarchical_controller.h" // Phase 2: Hierarchical/Grouped tuning
#include "controllers/adaptive_phase_controller.h" // Phase 4: Adaptive phase transitions

// Phase 2.1: Quantum-Enhanced Control System (QUANTUM_CONTROL_ENHANCEMENT_ROADMAP.md)
#include "controllers/rls_system_identifier.h"    // Fast O(n²) recursive system identification
#include "controllers/hybrid_pid_tuner.h"         // Relay/Z-N + Adam gradient descent hybrid
#include "controllers/extended_kalman_filter.h"   // EKF for nonlinear dynamics
#include "controllers/tensor_network_kalman.h"    // TT-decomposed Kalman (O(n×r²) memory)

#include <stdexcept>

// A struct to hold the inputs from the main simulation loop.
struct SystemState {
    std::map<std::string, float> metrics;
    std::vector<std::string> metric_names; // ADDED: To pass metric names dynamically
    std::vector<std::string> control_input_names; // ADDED: To pass control input names for MIMO
    uint64_t cpu_affinity_mask;
    std::vector<int> core_frequencies;
    // ADDED: Hardware metrics
    float cpu_temperature;
};

class ControllerException : public std::runtime_error {
public:
    explicit ControllerException(const std::string& message)
        : std::runtime_error(message) {}
};

// The main controller class, now refactored to use a Kalman Filter and LQR control law.
class HamiltonianMetaController { 
 public:
  explicit HamiltonianMetaController(const std::string& config_path);
  ~HamiltonianMetaController();

  // Public interface
  std::map<std::string, float> update_and_act(const SystemState& current_state, const std::string& path_prefix);
  void reload_configs(const std::string& path_prefix,
                      const std::vector<std::string>& metric_names,
                      const std::vector<std::string>& control_input_names,
                      bool system_id_completed);
  void trigger_autotune();

  /**
   * @brief Get HD-based stagnation metric from EKF innovation fingerprinting.
   *
   * Returns cosine similarity between consecutive innovation fingerprints (0-1).
   * High values (>0.95) indicate training stagnation (barren plateau).
   * Used by QAHPO for tunneling probability adjustment.
   *
   * @return Stagnation similarity (0=diverse innovations, 1=identical)
   */
  float getStagnationMetric() const {
      if (ekf_) {
          return ekf_->getStagnationMetric();
      }
      return 0.0f;  // No stagnation if EKF not available
  }

  // Disable copy and move constructors for singleton-like behavior
  HamiltonianMetaController(const HamiltonianMetaController&) = delete;
  HamiltonianMetaController& operator=(const HamiltonianMetaController&) = delete;
  HamiltonianMetaController(HamiltonianMetaController&&) = delete;
  HamiltonianMetaController& operator=(HamiltonianMetaController&&) = delete;

 private:
  // Initialization and configuration management
  void initialize(const SystemState& initial_state, const std::string& path_prefix);
  void load_and_reinitialize(const std::string& path_prefix, const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names);
  void check_config_dimensions(const ControllerConfig& controller_config, const PIDConfig& pid_config, const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names, std::vector<std::string>& regen_reasons);
  void reinitialize_controller_state(const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names);
  void generate_default_state_space_config(const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names, const std::string& path);
  void generate_default_pid_config(const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names, const std::string& path);
  std::vector<std::string> get_control_input_names(const std::vector<std::string>& metric_names, const std::vector<std::string>& control_input_names);
  void reseed_controls_from_state(const SystemState& state);
  void rebuild_metric_control_lookup(const std::vector<std::string>& control_input_names);

  // Core control loop helpers
  void sanitize_measurements(std::vector<float>& y_k);
  void bootstrap_scalers(const std::vector<float>& raw_measurements);
  float clamp_control_step(float requested_unscaled, float previous_unscaled) const;

  // Auto-tuning methods
  void enter_autotune_mode();
  float relay_step(const SystemState& current_state);
  void analyze_relay_oscillations();
  void calculate_and_save_gains(float ultimate_gain, float ultimate_period);

  // Logging
  void log_pid_config_details(const std::vector<std::string>& control_input_names, const std::vector<std::string>& metric_names);

  // Member Variables
  std::unique_ptr<KalmanFilter> kalman_filter_;
  std::unique_ptr<MPCController> mpc_controller_;
  std::unique_ptr<HardwareManager> hardware_manager_;
  std::unique_ptr<saguaro::control::SensitivityTracker> sensitivity_tracker_; // Phase 1
  std::unique_ptr<saguaro::control::HierarchicalController> hierarchical_controller_; // Phase 2
  std::unique_ptr<saguaro::control::AdaptivePhaseController> adaptive_phase_controller_; // Phase 4

  // Phase 2.1: Quantum-Enhanced Control Components
  std::unique_ptr<saguaro::controllers::RLSSystemIdentifier<float>> rls_sysid_;  // Fast recursive sysid
  std::unique_ptr<saguaro::controllers::HybridPIDTuner> hybrid_pid_;             // Relay + Adam PID tuner
  std::unique_ptr<saguaro::controllers::ExtendedKalmanFilter> ekf_;              // Nonlinear EKF
  std::unique_ptr<saguaro::controllers::TensorNetworkKalmanFilter> tnkf_;        // TT-compressed Kalman

  // Quantum control configuration flags
  bool use_fast_sysid_ = true;             // Use RLS for fast system ID (vs N4SID only)
  bool use_hybrid_pid_ = true;             // Use Hybrid PID tuner (vs relay only)
  bool use_extended_kalman_ = true;        // Use EKF for nonlinear quantum dynamics
  bool use_tensor_network_kalman_ = true;  // Use TNKF for O(n*r²) memory efficiency

  // Update intervals for hybrid approach
  int fast_sysid_interval_ = 5;            // RLS update every 5 batches
  int full_sysid_interval_ = 500;          // Full N4SID every 500 batches

  ControllerConfig controller_config_;
  PIDConfig pid_config_;

  std::string default_config_file_path_;
  std::string active_config_file_path_;

  bool is_initialized_ = false;
  bool awaiting_control_seed_ = true;
  bool system_id_model_ready_ = false;
  bool is_mpc_active_ = false;

  // PID state
  std::vector<float> integral_error_;
  std::vector<float> previous_error_;
  std::vector<float> previous_control_action_;
  std::vector<float> last_valid_measurements_;

  // Adaptive setpoints
  std::vector<float> adaptive_setpoints_;
  bool adaptive_setpoints_initialized_ = false;
  const float setpoint_relaxation_factor_ = 0.05f;
  const float max_setpoint_step_ = 0.1f;

  // Control action constraints
  const float max_control_step_absolute_ = 0.1f;
  const float max_control_step_fraction_ = 0.05f;

  // Scaler bootstrapping state
  bool scalers_bootstrapped_ = false;
  int warmup_samples_ = 0;
  const int warmup_target_ = 50;
  std::vector<double> running_mean_;
  std::vector<double> running_m2_;

  // Auto-tuning state
  enum class AutoTuneState {
      INACTIVE,
      RELAY_STEP,
      ANALYZING
  };
  AutoTuneState auto_tune_state_ = AutoTuneState::INACTIVE;
  std::vector<float> relay_history_;
  std::vector<float> oscillation_periods_;
  int peak_count_ = 0;
  std::chrono::steady_clock::time_point last_crossing_time_;

  // State for detecting config changes
  size_t last_metric_count_ = 0;
  std::unordered_map<std::string, std::vector<size_t>> metric_to_control_indices_;

  // Phase 1: Sensitivity tracking state
  int batch_counter_ = 0;                          // Global batch counter for JSON export
  static constexpr int SENSITIVITY_EXPORT_INTERVAL = 500;  // Export every 500 batches
};
#endif // HAMILTONIAN_META_CONTROLLER_H_