#ifndef HARDWARE_HIL_INTERFACE_H_
#define HARDWARE_HIL_INTERFACE_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hardware {

/**
 * Hardware-in-the-Loop (HIL) Interface for real-time control.
 *
 * This interface provides:
 * - Real-time scheduling (PREEMPT_RT support)
 * - DAQ integration (ADC sensors, DAC/PWM actuators)
 * - Deterministic timing (<1ms sensor→inference→actuator latency)
 * - High-frequency control loops (1 kHz target)
 *
 * Compliance: GEMINI.md, AGENTS.md, technical_roadmap.md Phase 8.3
 */

// Sensor reading from ADC
struct SensorReading {
    uint64_t timestamp_ns;  // Nanosecond timestamp (CLOCK_MONOTONIC)
    uint32_t channel;       // ADC channel number
    float value;            // Normalized value [0.0, 1.0] or physical units
    bool valid;             // Reading validity flag
};

// Actuator command for DAC/PWM
struct ActuatorCommand {
    uint32_t channel;       // DAC/PWM channel number
    float value;            // Command value [0.0, 1.0] or physical units
    uint64_t deadline_ns;   // Deadline for command execution (0 = immediate)
};

// Control loop statistics
struct ControlLoopStats {
    uint64_t loop_count;
    uint64_t missed_deadlines;
    double mean_latency_us;
    double max_latency_us;
    double p99_latency_us;
    double mean_jitter_us;
};

// Real-time scheduling priority
enum class SchedulingPolicy {
    NORMAL,         // Standard Linux scheduling (SCHED_OTHER)
    REALTIME_RR,    // Real-time round-robin (SCHED_RR)
    REALTIME_FIFO,  // Real-time FIFO (SCHED_FIFO)
};

/**
 * DAQ (Data Acquisition) device abstraction.
 *
 * Concrete implementations provide hardware-specific interfaces:
 * - GPIO for simple digital I/O
 * - SPI/I2C ADCs (MCP3008, ADS1115, etc.)
 * - USB DAQ devices (NI USB-6001, LabJack, etc.)
 * - Industrial fieldbus (EtherCAT, Modbus RTU)
 */
class DAQDevice {
public:
    virtual ~DAQDevice() = default;

    // Read sensor from ADC channel
    virtual SensorReading read_sensor(uint32_t channel) = 0;

    // Write actuator command to DAC/PWM channel
    virtual bool write_actuator(const ActuatorCommand& cmd) = 0;

    // Get number of available input channels
    virtual uint32_t get_input_channels() const = 0;

    // Get number of available output channels
    virtual uint32_t get_output_channels() const = 0;

    // Device identification
    virtual std::string get_device_info() const = 0;
};

/**
 * Simulated DAQ device for testing without hardware.
 */
class SimulatedDAQ : public DAQDevice {
public:
    SimulatedDAQ(uint32_t num_inputs, uint32_t num_outputs);
    ~SimulatedDAQ() override = default;

    SensorReading read_sensor(uint32_t channel) override;
    bool write_actuator(const ActuatorCommand& cmd) override;
    uint32_t get_input_channels() const override { return num_inputs_; }
    uint32_t get_output_channels() const override { return num_outputs_; }
    std::string get_device_info() const override;

    // Simulation control
    void set_sensor_value(uint32_t channel, float value);
    float get_actuator_value(uint32_t channel) const;

private:
    uint32_t num_inputs_;
    uint32_t num_outputs_;
    std::vector<float> sensor_values_;
    std::vector<float> actuator_values_;
};

/**
 * Hardware-in-the-Loop Interface.
 *
 * Provides real-time control loop execution with deterministic timing.
 *
 * Example usage:
 *   HILInterface hil(std::make_unique<SimulatedDAQ>(8, 4));
 *   hil.set_control_callback([](const std::vector<SensorReading>& sensors) {
 *       // Run HSMN inference
 *       std::vector<ActuatorCommand> cmds = infer(sensors);
 *       return cmds;
 *   });
 *   hil.start_control_loop(1000);  // 1 kHz loop
 */
class HILInterface {
public:
    using ControlCallback = std::function<
        std::vector<ActuatorCommand>(const std::vector<SensorReading>&)
    >;

    explicit HILInterface(std::unique_ptr<DAQDevice> daq_device);
    ~HILInterface();

    // Configure control loop
    void set_control_callback(ControlCallback callback);
    void set_input_channels(const std::vector<uint32_t>& channels);
    void set_output_channels(const std::vector<uint32_t>& channels);

    // Real-time scheduling
    bool set_scheduling_policy(SchedulingPolicy policy, int priority = 50);
    bool set_cpu_affinity(int cpu_core);

    // Control loop management
    bool start_control_loop(uint32_t frequency_hz, uint64_t max_latency_us = 1000);
    void stop_control_loop();
    bool is_running() const { return running_.load(); }

    // Statistics
    ControlLoopStats get_stats() const;
    void reset_stats();

    // Safety
    void set_watchdog_timeout_ms(uint32_t timeout_ms);
    bool check_watchdog() const;

private:
    void control_loop_thread();
    uint64_t get_monotonic_time_ns() const;

    std::unique_ptr<DAQDevice> daq_device_;
    ControlCallback control_callback_;
    std::vector<uint32_t> input_channels_;
    std::vector<uint32_t> output_channels_;

    std::atomic<bool> running_;
    std::atomic<uint64_t> last_heartbeat_ns_;
    uint64_t watchdog_timeout_ns_;

    // Thread handle (implementation-specific)
    void* thread_handle_;

    // Statistics
    mutable std::atomic<uint64_t> loop_count_;
    mutable std::atomic<uint64_t> missed_deadlines_;
    mutable std::atomic<double> mean_latency_us_;
    mutable std::atomic<double> max_latency_us_;
    mutable std::atomic<double> p99_latency_us_;
    mutable std::atomic<double> mean_jitter_us_;

    uint32_t loop_period_us_;
    uint64_t max_latency_us_threshold_;
};

}  // namespace hardware

#endif  // HARDWARE_HIL_INTERFACE_H_
