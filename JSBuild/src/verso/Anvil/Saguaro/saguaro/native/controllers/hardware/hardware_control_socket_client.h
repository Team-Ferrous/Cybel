#ifndef HARDWARE_HARDWARE_CONTROL_SOCKET_CLIENT_H_
#define HARDWARE_HARDWARE_CONTROL_SOCKET_CLIENT_H_

#include "hardware_control_client.h"
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <thread>
#include <iostream>
#include <cerrno>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "hardware/hardware_control_paths.h"

class HardwareControlSocketClient : public HardwareControlClient {
public:
    HardwareControlSocketClient() = default;

    bool set_performance_mode(PerformanceMode mode) override {
        std::string mode_str;
        switch (mode) {
            case PerformanceMode::High:
                mode_str = "High";
                break;
            case PerformanceMode::Normal:
                mode_str = "Normal";
                break;
            case PerformanceMode::PowerSave:
                mode_str = "PowerSave";
                break;
        }
        return send_command("set_performance_mode:" + mode_str);
    }

    bool set_cpu_affinity(uint64_t mask) override {
        return send_command("set_cpu_affinity:" + std::to_string(mask));
    }

    bool set_core_frequencies(const std::vector<int>& freqs_khz) override {
        // This is a simplified implementation. A real implementation would need a more robust way to pass the vector.
        if (freqs_khz.empty()) {
            return true;
        }
        return send_command("set_core_frequencies:" + std::to_string(freqs_khz[0]));
    }

    bool apply_memory_throttle() override {
        return send_command("apply_memory_throttle:");
    }

    bool set_memory_limit(long long limit_bytes) override {
        return send_command("set_memory_limit:" + std::to_string(limit_bytes));
    }

private:
    const std::string socket_path_ = hardware::paths::ResolveSocketPath();

    bool send_command(const std::string& command) {
        int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("socket");
            return false;
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

        // Retry connecting with exponential backoff
        int retries = 5;
        int backoff_ms = 100;
        for (int i = 0; i < retries; ++i) {
            if (connect(sockfd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
                // Connection successful
                if (write(sockfd, command.c_str(), command.length()) < 0) {
                    perror("write");
                    close(sockfd);
                    return false;
                }
                close(sockfd);
                return true;
            }
            std::cerr << "Failed to connect to hardware control daemon. Retrying in " << backoff_ms << "ms..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
            backoff_ms *= 2;
        }

        int failure_errno = errno;
        perror("connect");
        if (failure_errno == EACCES) {
            std::cerr << "[HardwareControlSocketClient] Permission denied when connecting to "
                      << socket_path_
                      << ". Ensure hardware-control-daemon.service is running and that the training user "
                      << "is included in the allowlist (sudo ./install_daemon_service.sh)." << std::endl;
        } else if (failure_errno == ENOENT) {
            std::cerr << "[HardwareControlSocketClient] Hardware daemon socket "
                      << socket_path_ << " was not found. Start the daemon via "
                      << "sudo ./install_daemon_service.sh or set SAGUARO_DAEMON_SOCKET_PATH "
                      << "to a reachable location." << std::endl;
        }
        close(sockfd);
        return false;
    }
};

#endif // HARDWARE_HARDWARE_CONTROL_SOCKET_CLIENT_H_
