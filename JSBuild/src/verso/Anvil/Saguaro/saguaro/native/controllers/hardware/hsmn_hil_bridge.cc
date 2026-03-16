// src/controllers/hardware/hsmn_hil_bridge.cc
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

/**
 * HSMN Hardware-in-the-Loop Bridge
 *
 * Task 4.2: Integrate HSMN with HIL interface
 * - Wraps HSMN inference in HILInterface::ControlCallback
 * - Sensor input → HSMN forward pass → Actuator command
 * - Handles variable-frequency updates (adaptive sampling)
 * - Target: <10ms inference latency for 120 Hz control loops
 *
 * Roadmap: docs/roadmaps/SAGUARO_EHD_Architecture_Roadmap.md (Phase 4, Task 4.2)
 */

#include "hsmn_hil_bridge.h"

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/framework/tensor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace hardware {

namespace {

// Convert sensor readings to TensorFlow input tensor
tensorflow::Tensor sensors_to_tensor(
    const std::vector<SensorReading>& sensors,
    int max_seq_len) {
    // Create input tensor: [batch_size=1, seq_len, features]
    // For EHD thruster: features = [voltage, current, temp, humidity, ...]
    int num_features = static_cast<int>(sensors.size());

    tensorflow::Tensor input_tensor(
        tensorflow::DT_FLOAT,
        tensorflow::TensorShape({1, max_seq_len, num_features})
    );

    auto tensor_data = input_tensor.tensor<float, 3>();

    // Fill tensor with sensor values (replicate to fill sequence length)
    for (int t = 0; t < max_seq_len; ++t) {
        for (int f = 0; f < num_features; ++f) {
            tensor_data(0, t, f) = sensors[f].valid ? sensors[f].value : 0.0f;
        }
    }

    return input_tensor;
}

// Convert HSMN output tensor to actuator commands
std::vector<ActuatorCommand> tensor_to_commands(
    const tensorflow::Tensor& output_tensor,
    const std::vector<uint32_t>& output_channels) {
    std::vector<ActuatorCommand> commands;
    commands.reserve(output_channels.size());

    // Extract predictions from output tensor
    // HSMN outputs: [batch_size=1, seq_len, output_dim]
    // We take the last timestep's predictions
    auto output_data = output_tensor.tensor<float, 3>();
    int seq_len = output_tensor.dim_size(1);
    int output_dim = output_tensor.dim_size(2);

    // Map output dimensions to actuator channels
    for (size_t i = 0; i < output_channels.size() && i < static_cast<size_t>(output_dim); ++i) {
        ActuatorCommand cmd;
        cmd.channel = output_channels[i];
        // Take last timestep prediction
        cmd.value = output_data(0, seq_len - 1, static_cast<int>(i));
        // Clamp to [0, 1] range for safety
        cmd.value = std::max(0.0f, std::min(1.0f, cmd.value));
        cmd.deadline_ns = 0;  // Immediate execution
        commands.push_back(cmd);
    }

    return commands;
}

}  // namespace

// ============================================================================
// HSMNHILBridge Implementation
// ============================================================================

HSMNHILBridge::HSMNHILBridge(
    const std::string& model_path,
    const HSMNHILConfig& config)
    : config_(config),
      inference_count_(0),
      total_inference_time_us_(0.0),
      max_inference_time_us_(0.0) {

    // Load TensorFlow SavedModel or TFLite model
    if (model_path.find(".tflite") != std::string::npos) {
        use_tflite_ = true;
        load_tflite_model(model_path);
    } else {
        use_tflite_ = false;
        load_tensorflow_model(model_path);
    }

    std::cout << "[HSMNHILBridge] Loaded model from: " << model_path << std::endl;
    std::cout << "[HSMNHILBridge] Max sequence length: " << config_.max_seq_len << std::endl;
    std::cout << "[HSMNHILBridge] Input features: " << config_.num_input_features << std::endl;
    std::cout << "[HSMNHILBridge] Output channels: " << config_.num_output_channels << std::endl;
}

HSMNHILBridge::~HSMNHILBridge() {
    if (tf_session_) {
        tf_session_->Close();
    }
}

void HSMNHILBridge::load_tensorflow_model(const std::string& model_path) {
    tensorflow::SessionOptions session_options;

    // Configure CPU optimization
    session_options.config.set_intra_op_parallelism_threads(config_.num_threads);
    session_options.config.set_inter_op_parallelism_threads(1);  // Sequential execution
    session_options.config.set_use_per_session_threads(true);

    // Disable GPU (CPU-only for deterministic timing)
    (*session_options.config.mutable_device_count())["GPU"] = 0;

    tf_session_.reset(tensorflow::NewSession(session_options));

    // Load SavedModel
    tensorflow::MetaGraphDef meta_graph_def;
    tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path + "/saved_model.pb", &meta_graph_def);

    tensorflow::Status status = tf_session_->Create(meta_graph_def.graph_def());
    if (!status.ok()) {
        throw std::runtime_error("Failed to load TensorFlow model: " + status.ToString());
    }
}

void HSMNHILBridge::load_tflite_model(const std::string& model_path) {
    // TFLite model loading (placeholder - requires tflite C++ API)
    // For production, use TFLite Interpreter
    throw std::runtime_error("TFLite model loading not yet implemented in C++ bridge");
}

std::vector<ActuatorCommand> HSMNHILBridge::infer(
    const std::vector<SensorReading>& sensors) {
    auto start_time = std::chrono::high_resolution_clock::now();

    // Convert sensors to tensor
    tensorflow::Tensor input_tensor = sensors_to_tensor(sensors, config_.max_seq_len);

    // Run inference
    std::vector<tensorflow::Tensor> outputs;
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {config_.input_tensor_name, input_tensor}
    };

    tensorflow::Status status = tf_session_->Run(
        inputs,
        {config_.output_tensor_name},
        {},  // No target nodes
        &outputs
    );

    if (!status.ok()) {
        std::cerr << "[HSMNHILBridge] Inference failed: " << status.ToString() << std::endl;
        return {};  // Return empty commands on failure
    }

    // Convert output to actuator commands
    std::vector<ActuatorCommand> commands = tensor_to_commands(
        outputs[0],
        config_.output_channels
    );

    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    double inference_time_us = std::chrono::duration<double, std::micro>(
        end_time - start_time
    ).count();

    inference_count_++;
    total_inference_time_us_ += inference_time_us;
    max_inference_time_us_ = std::max(max_inference_time_us_, inference_time_us);

    // Check latency budget
    if (inference_time_us > config_.max_latency_us) {
        std::cerr << "[HSMNHILBridge] Warning: Inference exceeded latency budget: "
                  << inference_time_us << " us (max: " << config_.max_latency_us << " us)" << std::endl;
    }

    return commands;
}

HILInterface::ControlCallback HSMNHILBridge::get_control_callback() {
    return [this](const std::vector<SensorReading>& sensors) -> std::vector<ActuatorCommand> {
        return this->infer(sensors);
    };
}

HSMNInferenceStats HSMNHILBridge::get_stats() const {
    HSMNInferenceStats stats;
    stats.inference_count = inference_count_.load();
    stats.mean_latency_us = inference_count_ > 0
        ? total_inference_time_us_.load() / inference_count_
        : 0.0;
    stats.max_latency_us = max_inference_time_us_.load();
    stats.latency_budget_violations = 0;  // TODO: Track violations

    return stats;
}

void HSMNHILBridge::reset_stats() {
    inference_count_.store(0);
    total_inference_time_us_.store(0.0);
    max_inference_time_us_.store(0.0);
}

void HSMNHILBridge::set_adaptive_sampling(bool enable) {
    config_.adaptive_sampling = enable;
}

void HSMNHILBridge::set_preprocessing_callback(PreprocessingCallback callback) {
    preprocessing_callback_ = callback;
}

void HSMNHILBridge::set_postprocessing_callback(PostprocessingCallback callback) {
    postprocessing_callback_ = callback;
}

}  // namespace hardware
