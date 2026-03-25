#ifndef HARDWARE_HARDWARE_MANAGER_H_
#define HARDWARE_HARDWARE_MANAGER_H_

#include <memory>
#include <iostream>
#include "hardware_control_client.h"
#include "hardware/hardware_control_socket_client.h"
#include "hardware/dummy_hardware_control_client.h"

class ResilientHardwareControlClient : public HardwareControlClient {
public:
    ResilientHardwareControlClient()
        : socket_client_(std::make_unique<HardwareControlSocketClient>()),
          dummy_client_(std::make_unique<DummyHardwareControlClient>()),
          using_dummy_(false) {}

    bool set_performance_mode(PerformanceMode mode) override {
        return Dispatch([&](HardwareControlClient* client) {
            return client->set_performance_mode(mode);
        });
    }

    bool set_cpu_affinity(uint64_t mask) override {
        return Dispatch([&](HardwareControlClient* client) {
            return client->set_cpu_affinity(mask);
        });
    }

    bool set_core_frequencies(const std::vector<int>& freqs_khz) override {
        return Dispatch([&](HardwareControlClient* client) {
            return client->set_core_frequencies(freqs_khz);
        });
    }

    bool apply_memory_throttle() override {
        return Dispatch([&](HardwareControlClient* client) {
            return client->apply_memory_throttle();
        });
    }

    bool set_memory_limit(long long limit_bytes) override {
        return Dispatch([&](HardwareControlClient* client) {
            return client->set_memory_limit(limit_bytes);
        });
    }

private:
    template <typename Fn>
    bool Dispatch(Fn&& fn) {
        if (!using_dummy_) {
            if (fn(socket_client_.get())) {
                return true;
            }
            using_dummy_ = true;
            std::cerr << "[HardwareManager] Falling back to dummy hardware control client "
                      << "(daemon unavailable or permission denied)." << std::endl;
        }
        return fn(dummy_client_.get());
    }

    std::unique_ptr<HardwareControlSocketClient> socket_client_;
    std::unique_ptr<DummyHardwareControlClient> dummy_client_;
    bool using_dummy_;
};

// This class manages hardware interactions by holding telemetry and control clients.
class HardwareManager {
public:
    HardwareManager()
        : control_client_(std::make_unique<ResilientHardwareControlClient>()) {}

    HardwareControlClient* get_control_client() const {
        return control_client_.get();
    }

private:
    std::unique_ptr<HardwareControlClient> control_client_;
};
#endif // HARDWARE_HARDWARE_MANAGER_H_
