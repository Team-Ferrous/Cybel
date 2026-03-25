#ifndef HARDWARE_HARDWARE_TELEMETRY_H_
#define HARDWARE_HARDWARE_TELEMETRY_H_

#include <thread>
#include <atomic>

// This class manages all direct hardware interactions for monitoring
// system metrics in a separate thread.
class HardwareTelemetry {
public:
    HardwareTelemetry();
    ~HardwareTelemetry();

    // --- Getters for Monitored State ---
    float get_cpu_temperature() const;
    float get_cpu_power_draw() const;
    float get_memory_usage_mb() const;

private:
    void monitoring_thread_func();

    // Monitored state variables
    std::atomic<float> cpu_temperature_{0.0f};
    std::atomic<float> cpu_power_draw_{0.0f};
    std::atomic<float> memory_usage_mb_{0.0f};

    // Thread management
    std::thread monitoring_thread_;
    std::atomic<bool> stop_monitoring_{false};
};

#endif // HARDWARE_HARDWARE_TELEMETRY_H_
