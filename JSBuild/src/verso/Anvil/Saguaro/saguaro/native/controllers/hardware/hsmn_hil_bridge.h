// src/controllers/hardware/hsmn_hil_bridge.h
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

#ifndef HARDWARE_SAGUARO_HIL_BRIDGE_H_
#define HARDWARE_SAGUARO_HIL_BRIDGE_H_

#include "hil_interface.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Forward declarations for TensorFlow
namespace tensorflow {
class Session;
class Tensor;
}  // namespace tensorflow

namespace hardware {

/**
 * HSMN inference statistics.
 */
struct HSMNInferenceStats {
    uint64_t inference_count;
    double mean_latency_us;
    double max_latency_us;
    uint64_t latency_budget_violations;
};

/**
 * HSMN HIL configuration.
 */
struct HSMNHILConfig {
    // Model configuration
    int max_seq_len = 128;           // Maximum sequence length for HSMN
    int num_input_features = 8;      // Number of sensor inputs
    int num_output_channels = 4;     // Number of actuator outputs
    std::vector<uint32_t> output_channels = {0, 1, 2, 3};

    // TensorFlow configuration
    int num_threads = 4;             // CPU threads for inference
    std::string input_tensor_name = "serving_default_input_ids:0";
    std::string output_tensor_name = "StatefulPartitionedCall:0";

    // Latency configuration
    double max_latency_us = 10000.0; // 10ms latency budget

    // Adaptive sampling
    bool adaptive_sampling = false;   // Enable adaptive sampling rate
    double min_sampling_hz = 10.0;    // Minimum sampling frequency
    double max_sampling_hz = 120.0;   // Maximum sampling frequency
};

/**
 * HSMN Hardware-in-the-Loop Bridge.
 *
 * Integrates HSMN model inference with the HIL control interface.
 *
 * Features:
 * - Wraps HSMN inference in HILInterface::ControlCallback
 * - Sensor input → HSMN forward pass → Actuator command
 * - Variable-frequency updates (adaptive sampling)
 * - Latency tracking and budget enforcement
 * - Preprocessing/postprocessing hooks
 *
 * Example usage:
 *   HSMNHILConfig config;
 *   config.max_seq_len = 128;
 *   config.num_input_features = 8;
 *   config.num_output_channels = 4;
 *
 *   HSMNHILBridge bridge("models/hsmn_quantized.tflite", config);
 *
 *   HILInterface hil(std::make_unique<SimulatedDAQ>(8, 4));
 *   hil.set_control_callback(bridge.get_control_callback());
 *   hil.start_control_loop(120);  // 120 Hz control loop
 *
 * Roadmap: docs/roadmaps/SAGUARO_EHD_Architecture_Roadmap.md (Phase 4, Task 4.2)
 */
class HSMNHILBridge {
public:
    using PreprocessingCallback = std::function<
        std::vector<SensorReading>(const std::vector<SensorReading>&)
    >;
    using PostprocessingCallback = std::function<
        std::vector<ActuatorCommand>(const std::vector<ActuatorCommand>&)
    >;

    /**
     * Construct HSMN HIL bridge with model path.
     *
     * @param model_path Path to TensorFlow SavedModel or TFLite model
     * @param config HSMN HIL configuration
     */
    HSMNHILBridge(const std::string& model_path, const HSMNHILConfig& config);
    ~HSMNHILBridge();

    /**
     * Run HSMN inference on sensor readings.
     *
     * @param sensors Vector of sensor readings from DAQ
     * @return Vector of actuator commands
     */
    std::vector<ActuatorCommand> infer(const std::vector<SensorReading>& sensors);

    /**
     * Get control callback for HILInterface.
     *
     * @return ControlCallback function wrapping infer()
     */
    HILInterface::ControlCallback get_control_callback();

    /**
     * Get inference statistics.
     *
     * @return HSMNInferenceStats with latency metrics
     */
    HSMNInferenceStats get_stats() const;

    /**
     * Reset inference statistics.
     */
    void reset_stats();

    /**
     * Enable/disable adaptive sampling.
     *
     * @param enable True to enable adaptive sampling
     */
    void set_adaptive_sampling(bool enable);

    /**
     * Set preprocessing callback for sensor data.
     *
     * @param callback Function to preprocess sensor readings
     */
    void set_preprocessing_callback(PreprocessingCallback callback);

    /**
     * Set postprocessing callback for actuator commands.
     *
     * @param callback Function to postprocess actuator commands
     */
    void set_postprocessing_callback(PostprocessingCallback callback);

private:
    void load_tensorflow_model(const std::string& model_path);
    void load_tflite_model(const std::string& model_path);

    HSMNHILConfig config_;
    std::unique_ptr<tensorflow::Session> tf_session_;
    bool use_tflite_ = false;

    // Callbacks
    PreprocessingCallback preprocessing_callback_;
    PostprocessingCallback postprocessing_callback_;

    // Statistics
    mutable std::atomic<uint64_t> inference_count_;
    mutable std::atomic<double> total_inference_time_us_;
    mutable std::atomic<double> max_inference_time_us_;
};

}  // namespace hardware

#endif  // HARDWARE_SAGUARO_HIL_BRIDGE_H_
