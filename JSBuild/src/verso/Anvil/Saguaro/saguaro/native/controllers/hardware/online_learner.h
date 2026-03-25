// src/controllers/hardware/online_learner.h
// Online learning controller for HSMN hardware-in-the-loop deployment
// Implements replay buffer and periodic weight updates for real-time adaptation

#ifndef SAGUARO_CONTROLLERS_HARDWARE_ONLINE_LEARNER_H_
#define SAGUARO_CONTROLLERS_HARDWARE_ONLINE_LEARNER_H_

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace saguaro {
namespace controllers {
namespace hardware {

// Single experience tuple: (sensor_reading, actuator_command, reward, next_sensor_reading)
struct Experience {
    std::vector<float> sensor_state;      // Current sensor readings (e.g., voltage, current, temp, humidity)
    std::vector<float> actuator_command;  // Control output (e.g., voltage setpoint, timing)
    float reward;                         // Scalar reward (e.g., -|thrust_error| - power_penalty)
    std::vector<float> next_sensor_state; // Sensor readings after action
    uint64_t timestamp_ns;                // Timestamp for debugging/analysis
    bool terminal;                        // Episode termination flag
};

// Configuration for online learner
struct OnlineLearnerConfig {
    uint32_t buffer_capacity = 10000;      // Maximum replay buffer size
    uint32_t batch_size = 32;              // Mini-batch size for weight updates
    uint32_t update_frequency = 120;       // Update HSMN every N experiences (1 Hz at 120 Hz sampling)
    float learning_rate_initial = 1e-4f;   // Initial learning rate
    float learning_rate_decay_tau = 1000.0f; // Decay time constant (updates)
    float discount_factor = 0.99f;         // Gamma for temporal difference learning
    float gradient_clip_norm = 1.0f;       // Gradient clipping threshold
    bool enable_prioritized_replay = false; // Use TD-error-weighted sampling
    uint32_t target_network_update_freq = 500; // Update target network every N updates (for stability)
};

// Online learner for HSMN weight updates during HIL operation
class OnlineLearner {
 public:
    explicit OnlineLearner(const OnlineLearnerConfig& config);
    ~OnlineLearner();

    // Add new experience to replay buffer (thread-safe)
    void add_experience(const Experience& exp);

    // Check if update is due and perform it if so
    // Returns true if update was performed, false otherwise
    bool maybe_update_weights();

    // Force immediate weight update (for manual triggering)
    void force_update();

    // Get current learning rate (decays over time)
    float get_current_learning_rate() const;

    // Get replay buffer statistics
    uint32_t get_buffer_size() const;
    uint32_t get_total_updates() const;
    float get_last_loss() const;

    // Load/save model weights (for checkpointing)
    bool load_weights(const std::string& filepath);
    bool save_weights(const std::string& filepath) const;

    // Clear replay buffer (for reset/debugging)
    void clear_buffer();

 private:
    OnlineLearnerConfig config_;
    std::deque<Experience> replay_buffer_;
    mutable std::mutex buffer_mutex_;

    std::atomic<uint32_t> experience_count_{0};
    std::atomic<uint32_t> update_count_{0};
    std::atomic<float> last_loss_{0.0f};

    // Sample mini-batch from replay buffer
    std::vector<Experience> sample_batch();

    // Compute TD-error loss for prioritized replay
    float compute_td_error(const Experience& exp) const;

    // Perform gradient descent step on sampled batch
    void update_step(const std::vector<Experience>& batch);

    // Apply gradient clipping for stability
    void clip_gradients(std::vector<float>& gradients, float max_norm) const;
};

}  // namespace hardware
}  // namespace controllers
}  // namespace saguaro

#endif  // SAGUARO_CONTROLLERS_HARDWARE_ONLINE_LEARNER_H_
