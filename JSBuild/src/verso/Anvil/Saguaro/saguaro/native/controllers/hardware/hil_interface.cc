#include "hil_interface.h"

#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

namespace hardware {

namespace {

// Convert timespec to nanoseconds
uint64_t timespec_to_ns(const struct timespec& ts) {
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
}

// Get current monotonic time in nanoseconds
uint64_t get_monotonic_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return timespec_to_ns(ts);
}

// Sleep until absolute time (nanoseconds)
void sleep_until_ns(uint64_t target_ns) {
    struct timespec ts;
    ts.tv_sec = target_ns / 1000000000ULL;
    ts.tv_nsec = target_ns % 1000000000ULL;
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
}

}  // namespace

// ============================================================================
// SimulatedDAQ Implementation
// ============================================================================

SimulatedDAQ::SimulatedDAQ(uint32_t num_inputs, uint32_t num_outputs)
    : num_inputs_(num_inputs),
      num_outputs_(num_outputs),
      sensor_values_(num_inputs, 0.0f),
      actuator_values_(num_outputs, 0.0f) {}

SensorReading SimulatedDAQ::read_sensor(uint32_t channel) {
    SensorReading reading;
    reading.timestamp_ns = get_monotonic_ns();
    reading.channel = channel;

    if (channel < num_inputs_) {
        reading.value = sensor_values_[channel];
        reading.valid = true;
    } else {
        reading.value = 0.0f;
        reading.valid = false;
    }

    return reading;
}

bool SimulatedDAQ::write_actuator(const ActuatorCommand& cmd) {
    if (cmd.channel >= num_outputs_) {
        return false;
    }

    actuator_values_[cmd.channel] = cmd.value;
    return true;
}

std::string SimulatedDAQ::get_device_info() const {
    return "SimulatedDAQ (inputs=" + std::to_string(num_inputs_) +
           ", outputs=" + std::to_string(num_outputs_) + ")";
}

void SimulatedDAQ::set_sensor_value(uint32_t channel, float value) {
    if (channel < num_inputs_) {
        sensor_values_[channel] = value;
    }
}

float SimulatedDAQ::get_actuator_value(uint32_t channel) const {
    if (channel < num_outputs_) {
        return actuator_values_[channel];
    }
    return 0.0f;
}

// ============================================================================
// HILInterface Implementation
// ============================================================================

HILInterface::HILInterface(std::unique_ptr<DAQDevice> daq_device)
    : daq_device_(std::move(daq_device)),
      running_(false),
      last_heartbeat_ns_(0),
      watchdog_timeout_ns_(5000000000ULL),  // 5 seconds default
      thread_handle_(nullptr),
      loop_count_(0),
      missed_deadlines_(0),
      mean_latency_us_(0.0),
      max_latency_us_(0.0),
      p99_latency_us_(0.0),
      mean_jitter_us_(0.0),
      loop_period_us_(1000),
      max_latency_us_threshold_(1000) {
    std::cout << "[HIL] Initialized with " << daq_device_->get_device_info() << std::endl;
}

HILInterface::~HILInterface() {
    if (running_.load()) {
        stop_control_loop();
    }
}

void HILInterface::set_control_callback(ControlCallback callback) {
    control_callback_ = std::move(callback);
}

void HILInterface::set_input_channels(const std::vector<uint32_t>& channels) {
    input_channels_ = channels;
}

void HILInterface::set_output_channels(const std::vector<uint32_t>& channels) {
    output_channels_ = channels;
}

bool HILInterface::set_scheduling_policy(SchedulingPolicy policy, int priority) {
    // This sets the policy for the calling thread
    // The control loop thread will inherit or set its own policy

    struct sched_param param;
    param.sched_priority = priority;

    int sched_policy;
    switch (policy) {
        case SchedulingPolicy::REALTIME_RR:
            sched_policy = SCHED_RR;
            break;
        case SchedulingPolicy::REALTIME_FIFO:
            sched_policy = SCHED_FIFO;
            break;
        case SchedulingPolicy::NORMAL:
        default:
            sched_policy = SCHED_OTHER;
            param.sched_priority = 0;  // SCHED_OTHER must have priority 0
            break;
    }

    if (sched_setscheduler(0, sched_policy, &param) != 0) {
        std::cerr << "[HIL] Failed to set scheduling policy: " << strerror(errno) << std::endl;
        std::cerr << "[HIL] Note: Real-time scheduling requires CAP_SYS_NICE capability" << std::endl;
        return false;
    }

    std::cout << "[HIL] Scheduling policy set: "
              << (policy == SchedulingPolicy::REALTIME_RR ? "SCHED_RR" :
                  policy == SchedulingPolicy::REALTIME_FIFO ? "SCHED_FIFO" : "SCHED_OTHER")
              << " (priority=" << priority << ")" << std::endl;

    return true;
}

bool HILInterface::set_cpu_affinity(int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "[HIL] Failed to set CPU affinity: " << strerror(errno) << std::endl;
        return false;
    }

    std::cout << "[HIL] CPU affinity set to core " << cpu_core << std::endl;
    return true;
}

bool HILInterface::start_control_loop(uint32_t frequency_hz, uint64_t max_latency_us) {
    if (running_.load()) {
        std::cerr << "[HIL] Control loop already running" << std::endl;
        return false;
    }

    if (!control_callback_) {
        std::cerr << "[HIL] No control callback set" << std::endl;
        return false;
    }

    if (input_channels_.empty()) {
        std::cerr << "[HIL] No input channels configured" << std::endl;
        return false;
    }

    loop_period_us_ = 1000000 / frequency_hz;
    max_latency_us_threshold_ = max_latency_us;

    std::cout << "[HIL] Starting control loop at " << frequency_hz << " Hz "
              << "(period=" << loop_period_us_ << " us, max_latency=" << max_latency_us << " us)"
              << std::endl;

    // Lock memory to prevent page faults in real-time loop
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::cerr << "[HIL] Warning: Failed to lock memory (mlockall): " << strerror(errno) << std::endl;
        std::cerr << "[HIL] Real-time performance may be degraded by page faults" << std::endl;
    }

    reset_stats();
    running_.store(true);
    last_heartbeat_ns_.store(get_monotonic_ns());

    // Launch control loop thread
    auto thread_func = [this]() { control_loop_thread(); };
    thread_handle_ = new std::thread(thread_func);

    return true;
}

void HILInterface::stop_control_loop() {
    if (!running_.load()) {
        return;
    }

    std::cout << "[HIL] Stopping control loop..." << std::endl;
    running_.store(false);

    // Wait for thread to finish
    if (thread_handle_) {
        static_cast<std::thread*>(thread_handle_)->join();
        delete static_cast<std::thread*>(thread_handle_);
        thread_handle_ = nullptr;
    }

    // Unlock memory
    munlockall();

    std::cout << "[HIL] Control loop stopped" << std::endl;
}

void HILInterface::control_loop_thread() {
    // Set real-time scheduling for this thread
    struct sched_param param;
    param.sched_priority = 80;  // High priority for real-time control

    if (sched_setscheduler(0, SCHED_FIFO, &param) == 0) {
        std::cout << "[HIL] Control loop thread: SCHED_FIFO priority 80" << std::endl;
    } else {
        std::cerr << "[HIL] Warning: Failed to set real-time scheduling" << std::endl;
    }

    uint64_t next_cycle_ns = get_monotonic_ns() + (loop_period_us_ * 1000);
    std::vector<uint64_t> latency_samples;
    latency_samples.reserve(1000);

    uint64_t count = 0;
    uint64_t missed = 0;

    while (running_.load()) {
        uint64_t cycle_start_ns = get_monotonic_ns();

        // Update heartbeat
        last_heartbeat_ns_.store(cycle_start_ns);

        // Step 1: Read sensors
        std::vector<SensorReading> sensor_readings;
        sensor_readings.reserve(input_channels_.size());

        for (uint32_t channel : input_channels_) {
            sensor_readings.push_back(daq_device_->read_sensor(channel));
        }

        uint64_t sensor_end_ns = get_monotonic_ns();

        // Step 2: Run control callback (HSMN inference)
        std::vector<ActuatorCommand> actuator_commands;

        try {
            actuator_commands = control_callback_(sensor_readings);
        } catch (const std::exception& e) {
            std::cerr << "[HIL] Control callback exception: " << e.what() << std::endl;
            // Continue with empty commands (safe state)
        }

        uint64_t inference_end_ns = get_monotonic_ns();

        // Step 3: Write actuators
        for (const auto& cmd : actuator_commands) {
            daq_device_->write_actuator(cmd);
        }

        uint64_t actuator_end_ns = get_monotonic_ns();

        // Step 4: Compute latency
        uint64_t latency_ns = actuator_end_ns - cycle_start_ns;
        double latency_us = latency_ns / 1000.0;

        latency_samples.push_back(latency_ns);
        count++;

        // Check for missed deadline
        if (latency_us > max_latency_us_threshold_) {
            missed++;
        }

        // Step 5: Sleep until next cycle
        next_cycle_ns += (loop_period_us_ * 1000);
        uint64_t now_ns = get_monotonic_ns();

        if (now_ns < next_cycle_ns) {
            sleep_until_ns(next_cycle_ns);
        } else {
            // Missed deadline, reset next cycle
            next_cycle_ns = now_ns + (loop_period_us_ * 1000);
            missed++;
        }

        // Update statistics every 100 cycles
        if (count % 100 == 0 && !latency_samples.empty()) {
            std::vector<uint64_t> sorted_samples = latency_samples;
            std::sort(sorted_samples.begin(), sorted_samples.end());

            double mean = std::accumulate(sorted_samples.begin(), sorted_samples.end(), 0.0) /
                          sorted_samples.size() / 1000.0;  // Convert to us

            double max_us = sorted_samples.back() / 1000.0;

            size_t p99_idx = static_cast<size_t>(sorted_samples.size() * 0.99);
            double p99_us = sorted_samples[p99_idx] / 1000.0;

            loop_count_.store(count);
            missed_deadlines_.store(missed);
            mean_latency_us_.store(mean);
            max_latency_us_.store(max_us);
            p99_latency_us_.store(p99_us);

            latency_samples.clear();
        }
    }

    std::cout << "[HIL] Control loop thread exiting" << std::endl;
}

uint64_t HILInterface::get_monotonic_time_ns() const {
    return get_monotonic_ns();
}

ControlLoopStats HILInterface::get_stats() const {
    ControlLoopStats stats;
    stats.loop_count = loop_count_.load();
    stats.missed_deadlines = missed_deadlines_.load();
    stats.mean_latency_us = mean_latency_us_.load();
    stats.max_latency_us = max_latency_us_.load();
    stats.p99_latency_us = p99_latency_us_.load();
    stats.mean_jitter_us = mean_jitter_us_.load();
    return stats;
}

void HILInterface::reset_stats() {
    loop_count_.store(0);
    missed_deadlines_.store(0);
    mean_latency_us_.store(0.0);
    max_latency_us_.store(0.0);
    p99_latency_us_.store(0.0);
    mean_jitter_us_.store(0.0);
}

void HILInterface::set_watchdog_timeout_ms(uint32_t timeout_ms) {
    watchdog_timeout_ns_ = static_cast<uint64_t>(timeout_ms) * 1000000ULL;
}

bool HILInterface::check_watchdog() const {
    if (!running_.load()) {
        return true;  // Not running, watchdog OK
    }

    uint64_t now_ns = get_monotonic_ns();
    uint64_t last_heartbeat = last_heartbeat_ns_.load();

    return (now_ns - last_heartbeat) < watchdog_timeout_ns_;
}

}  // namespace hardware
