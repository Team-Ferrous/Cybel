#ifndef HARDWARE_DUMMY_HARDWARE_CONTROL_CLIENT_H_
#define HARDWARE_DUMMY_HARDWARE_CONTROL_CLIENT_H_

#include "hardware_control_client.h"
#include <cstdlib>
#include <iostream>
#include <vector>

/**
 * Lightweight no-op hardware client used whenever real hardware control is not
 * available (e.g. developer desktops or CI). All methods simply log the request
 * and return success so higher level training code can continue to run.
 */
class DummyHardwareControlClient : public HardwareControlClient {
public:
    bool set_performance_mode(PerformanceMode mode) override {
        log_request("set_performance_mode", static_cast<int>(mode));
        return true;
    }

    bool set_cpu_affinity(uint64_t mask) override {
        log_request("set_cpu_affinity", mask);
        return true;
    }

    bool set_core_frequencies(const std::vector<int>& freqs_khz) override {
        if (!freqs_khz.empty()) {
            log_request("set_core_frequencies", freqs_khz.front());
        } else {
            log_request("set_core_frequencies", 0);
        }
        return true;
    }

    bool apply_memory_throttle() override {
        log_request("apply_memory_throttle", 0);
        return true;
    }

    bool set_memory_limit(long long limit_bytes) override {
        log_request("set_memory_limit", limit_bytes);
        return true;
    }

private:
    template <typename T>
    void log_request(const char* action, T value) {
        if (std::getenv("SAGUARO_SUPPRESS_DUMMY_HARDWARE_LOGS")) {
            return;
        }
        std::cout << "[DummyHardwareControlClient] " << action
                  << " -> " << value << " (no-op)" << std::endl;
    }
};

#endif // HARDWARE_DUMMY_HARDWARE_CONTROL_CLIENT_H_
