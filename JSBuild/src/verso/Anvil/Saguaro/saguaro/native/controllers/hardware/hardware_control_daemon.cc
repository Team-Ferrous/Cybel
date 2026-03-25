#include "hardware_control_daemon.h"

#include <dirent.h>
#include <pwd.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "hardware_control_paths.h"

namespace {
void write_to_sysfs(const std::string& path, const std::string& value) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        return; // Fail silently to avoid crashing the daemon.
    }
    ofs << value;
}

std::string trim(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\r\n");
    size_t end = input.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    return input.substr(start, end - start + 1);
}

constexpr mode_t kSocketPermissions =
    S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH; // 0666
} // namespace

HardwareControlDaemon::HardwareControlDaemon()
    : sockfd_(-1),
      socket_path_(hardware::paths::ResolveSocketPath()),
      allowlist_path_(hardware::paths::ResolveAllowlistPath()),
      allow_all_clients_(false) {
    load_allowlist();
    configure_socket_path();

    sockfd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sockfd_ < 0) {
        perror("socket");
        exit(EXIT_FAILURE);
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
    unlink(socket_path_.c_str());

    if (bind(sockfd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    if (listen(sockfd_, 5) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    configure_socket_permissions();
}

HardwareControlDaemon::~HardwareControlDaemon() {
    if (sockfd_ >= 0) {
        close(sockfd_);
    }
    unlink(socket_path_.c_str());
}

void HardwareControlDaemon::run() {
    while (true) {
        int client_sockfd = accept(sockfd_, nullptr, nullptr);
        if (client_sockfd < 0) {
            perror("accept");
            continue;
        }

        char buffer[256];
        int n = read(client_sockfd, buffer, sizeof(buffer) - 1);
        if (n <= 0) {
            perror("read");
            close(client_sockfd);
            continue;
        }
        buffer[n] = '\0';

        if (!authorize_client(client_sockfd)) {
            std::cerr << "Unauthorized client attempted hardware control access." << std::endl;
            const char* denied = "UNAUTHORIZED\n";
            if (write(client_sockfd, denied, strlen(denied)) < 0) {
                perror("write");
            }
            close(client_sockfd);
            continue;
        }

        bool success = handle_request(buffer);
        const char* response = success ? "OK\n" : "ERROR\n";
        if (write(client_sockfd, response, strlen(response)) < 0) {
            perror("write");
        }

        close(client_sockfd);
    }
}

bool HardwareControlDaemon::handle_request(const std::string& request) {
    size_t colon_pos = request.find(':');
    if (colon_pos == std::string::npos) {
        return false;
    }

    std::string command = trim(request.substr(0, colon_pos));
    std::string value = trim(request.substr(colon_pos + 1));

    if (command == "set_performance_mode") {
        std::string governor = "schedutil";
        int priority = 0;
        if (value == "High") {
            governor = "performance";
            priority = -10;
        } else if (value == "PowerSave") {
            governor = "powersave";
            priority = 10;
        }

        setpriority(PRIO_PROCESS, 0, priority);
        DIR* dp = opendir("/sys/devices/system/cpu");
        if (dp) {
            dirent* de;
            while ((de = readdir(dp)) != nullptr) {
                if (std::string(de->d_name).rfind("cpu", 0) == 0) {
                    write_to_sysfs(std::string("/sys/devices/system/cpu/") + de->d_name +
                                       "/cpufreq/scaling_governor",
                                   governor);
                }
            }
            closedir(dp);
        }
        return true;
    } else if (command == "set_cpu_affinity") {
        uint64_t mask = 0;
        try {
            mask = std::stoull(value);
        } catch (const std::exception&) {
            return false;
        }
        if (mask == 0) {
            return false;
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        for (int i = 0; i < 64; ++i) {
            if ((mask >> i) & 1) {
                CPU_SET(i, &cpuset);
            }
        }
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            perror("sched_setaffinity");
            return false;
        }
        return true;
    } else if (command == "set_core_frequencies") {
        int freq_khz = 0;
        try {
            freq_khz = std::stoi(value);
        } catch (const std::exception&) {
            return false;
        }
        if (freq_khz > 0) {
            std::string cpu_path = "/sys/devices/system/cpu/cpu0";
            write_to_sysfs(cpu_path + "/cpufreq/scaling_governor", "userspace");
            write_to_sysfs(cpu_path + "/cpufreq/scaling_setspeed", std::to_string(freq_khz));
        }
        return true;
    } else if (command == "apply_memory_throttle") {
        write_to_sysfs("/sys/fs/cgroup/memory.high", "90%");
        return true;
    } else if (command == "set_memory_limit") {
        long long limit_bytes = 0;
        try {
            limit_bytes = std::stoll(value);
        } catch (const std::exception&) {
            return false;
        }
        write_to_sysfs("/sys/fs/cgroup/memory.max", std::to_string(limit_bytes));
        return true;
    }

    return false;
}

bool HardwareControlDaemon::authorize_client(int client_sockfd) const {
    if (allow_all_clients_) {
        return true;
    }
    struct ucred credentials {};
    socklen_t len = sizeof(credentials);
    if (getsockopt(client_sockfd, SOL_SOCKET, SO_PEERCRED, &credentials, &len) < 0) {
        perror("getsockopt(SO_PEERCRED)");
        return false;
    }

    if (!is_uid_allowed(credentials.uid)) {
        std::cerr << "Rejected hardware control request from UID " << credentials.uid << std::endl;
        return false;
    }
    return true;
}

bool HardwareControlDaemon::is_uid_allowed(uid_t uid) const {
    return allowed_uids_.find(uid) != allowed_uids_.end();
}

void HardwareControlDaemon::load_allowlist() {
    allowed_uids_.clear();
    allowed_uids_.insert(0); // Always allow root for maintenance.

    auto try_create_allowlist = [this]() -> bool {
        namespace fs = std::filesystem;
        try {
            fs::create_directories(fs::path(allowlist_path_).parent_path());
            std::ofstream ofs(allowlist_path_, std::ios::out | std::ios::app);
            if (!ofs.is_open()) {
                return false;
            }
            if (ofs.tellp() == 0) {
                ofs << "# HSMN Hardware Control Daemon Allowlist\n";
                ofs << "# Auto-generated because the configured file was missing.\n";
            }
            const char* default_user = std::getenv("SAGUARO_DAEMON_DEFAULT_USER");
            if (default_user != nullptr && default_user[0] != '\0') {
                ofs << default_user << "\n";
            }
            ofs.close();
            return true;
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Warning: Failed to create hardware control allowlist directory: "
                      << e.what() << std::endl;
            return false;
        }
    };

    std::ifstream ifs(allowlist_path_);
    if (!ifs.is_open()) {
        if (!try_create_allowlist()) {
            std::cerr << "Warning: Unable to open hardware control allowlist at "
                      << allowlist_path_
                      << ". Falling back to socket-permission based access (all local users allowed)." << std::endl;
            allow_all_clients_ = true;
            return;
        }
        ifs.open(allowlist_path_);
        if (!ifs.is_open()) {
            std::cerr << "Warning: Unable to open hardware control allowlist at "
                      << allowlist_path_
                      << " even after creation attempt. Granting access to all local users with socket permissions."
                      << std::endl;
            allow_all_clients_ = true;
            return;
        }
    }
    allow_all_clients_ = false;

    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        if (std::all_of(line.begin(), line.end(), ::isdigit)) {
            uid_t uid = static_cast<uid_t>(std::stoul(line));
            allowed_uids_.insert(uid);
        } else {
            struct passwd* pwd = getpwnam(line.c_str());
            if (pwd == nullptr) {
                std::cerr << "Warning: Skipping unknown user '" << line
                          << "' in hardware control allowlist." << std::endl;
                continue;
            }
            allowed_uids_.insert(pwd->pw_uid);
        }
    }

    if (allowed_uids_.empty()) {
        allowed_uids_.insert(0);
    }
}

void HardwareControlDaemon::configure_socket_path() {
    namespace fs = std::filesystem;
    try {
        fs::path sock_path(socket_path_);
        fs::create_directories(sock_path.parent_path());
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Warning: Failed to ensure directory for hardware control socket: "
                  << e.what() << std::endl;
    }
}

void HardwareControlDaemon::configure_socket_permissions() const {
    if (chmod(socket_path_.c_str(), kSocketPermissions) < 0) {
        perror("chmod");
    }
}
