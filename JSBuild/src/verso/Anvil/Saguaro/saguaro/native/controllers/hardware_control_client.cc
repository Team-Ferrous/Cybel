#include "hardware/hardware_control_client.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <string.h> // For memset

#include "hardware/hardware_control_client.h"
#include "hardware/hardware_control_paths.h"

// Define the concrete implementation of HardwareControlClient
class HardwareControlSocketClient : public HardwareControlClient {
public:
    explicit HardwareControlSocketClient(std::string socket_path = hardware::paths::ResolveSocketPath());
    ~HardwareControlSocketClient() override;

    bool set_performance_mode(PerformanceMode mode) override;
    bool set_cpu_affinity(uint64_t mask) override;
    bool set_core_frequencies(const std::vector<int>& freqs_khz) override;
    bool apply_memory_throttle() override;
    bool set_memory_limit(long long limit_bytes) override;

private:
    bool connect();
    void disconnect();
    bool send_command(const std::string& command);

    std::string socket_path_;
    int sock_fd_;
    bool connected_ = false;
    int retry_delay_ms_ = 100;
    const int max_retry_delay_ms_ = 5000; // Max retry delay of 5 seconds
};

HardwareControlSocketClient::HardwareControlSocketClient(std::string socket_path)
    : socket_path_(std::move(socket_path)), sock_fd_(-1) {}

HardwareControlSocketClient::~HardwareControlSocketClient() {
    if (connected_) {
        disconnect();
    }
}

bool HardwareControlSocketClient::connect() {
    if (connected_) return true;

    sock_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd_ < 0) {
        std::cerr << "Error creating socket for hardware control daemon." << std::endl;
        return false;
    }

    struct sockaddr_un serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sun_family = AF_UNIX;
    strncpy(serv_addr.sun_path, socket_path_.c_str(), sizeof(serv_addr.sun_path) - 1);

    if (::connect(sock_fd_, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Error connecting to hardware control daemon socket. Is the daemon running?" << std::endl;
        close(sock_fd_);
        sock_fd_ = -1;
        return false;
    }

    std::cout << "Connected to hardware control daemon." << std::endl;
    connected_ = true;
    retry_delay_ms_ = 100; // Reset retry delay on successful connection
    return true;
}

void HardwareControlSocketClient::disconnect() {
    if (sock_fd_ != -1) {
        close(sock_fd_);
        sock_fd_ = -1;
    }
    connected_ = false;
    std::cout << "Disconnected from hardware control daemon." << std::endl;
}

bool HardwareControlSocketClient::send_command(const std::string& command) {
    if (!connected_) {
        std::cerr << "Not connected to hardware control daemon. Attempting to reconnect..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms_));
        retry_delay_ms_ = std::min(retry_delay_ms_ * 2, max_retry_delay_ms_); // Exponential backoff

        if (!connect()) {
            std::cerr << "Failed to reconnect to hardware control daemon. Command not sent: " << command << std::endl;
            return false;
        }
    }

    std::string full_command = command + "\n";
    if (send(sock_fd_, full_command.c_str(), full_command.length(), 0) < 0) {
        std::cerr << "Failed to send command to hardware control daemon. Connection lost." << std::endl;
        disconnect();
        return false;
    }

    char buffer[256] = {0};
    if (recv(sock_fd_, buffer, sizeof(buffer) - 1, 0) <= 0) {
        std::cerr << "Failed to receive response from daemon or connection closed." << std::endl;
        disconnect();
        return false;
    }

    if (std::string(buffer) != "OK\n") {
        std::cerr << "Hardware control daemon returned an error: " << buffer << std::endl;
        return false;
    }

    return true;
}

// Implementations of the virtual methods from HardwareControlClient
bool HardwareControlSocketClient::set_performance_mode(PerformanceMode mode) {
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

bool HardwareControlSocketClient::set_cpu_affinity(uint64_t mask) {
    return send_command("set_cpu_affinity:" + std::to_string(mask));
}

bool HardwareControlSocketClient::set_core_frequencies(const std::vector<int>& freqs_khz) {
    // This is a simplified implementation. A real implementation would need a more robust way to pass the vector.
    if (freqs_khz.empty()) {
        return true;
    }
    // For now, we'll just send the first frequency.
    return send_command("set_core_frequencies:" + std::to_string(freqs_khz[0]));
}

bool HardwareControlSocketClient::apply_memory_throttle() {
    return send_command("apply_memory_throttle:");
}

bool HardwareControlSocketClient::set_memory_limit(long long limit_bytes) {
    return send_command("set_memory_limit:" + std::to_string(limit_bytes));
}
