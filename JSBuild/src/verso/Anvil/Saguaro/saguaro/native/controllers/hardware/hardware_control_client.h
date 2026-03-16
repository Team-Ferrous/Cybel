#ifndef HARDWARE_HARDWARE_CONTROL_CLIENT_H_
#define HARDWARE_HARDWARE_CONTROL_CLIENT_H_

#include <string>
#include <vector>
#include <cstdint>

// This class provides a client interface for sending hardware control
// commands to a privileged daemon.
class HardwareControlClient {
public:
    enum class PerformanceMode {
        High,
        Normal,
        PowerSave
    };

    virtual ~HardwareControlClient() = default;

    virtual bool set_performance_mode(PerformanceMode mode) = 0;
    virtual bool set_cpu_affinity(uint64_t mask) = 0;
    virtual bool set_core_frequencies(const std::vector<int>& freqs_khz) = 0;
    virtual bool apply_memory_throttle() = 0;
    virtual bool set_memory_limit(long long limit_bytes) = 0;
};

#endif // HARDWARE_HARDWARE_CONTROL_CLIENT_H_
