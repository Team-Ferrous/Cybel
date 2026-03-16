#ifndef HARDWARE_HARDWARE_CONTROL_PATHS_H_
#define HARDWARE_HARDWARE_CONTROL_PATHS_H_

#include <cstdlib>
#include <string>

namespace hardware::paths {
inline constexpr const char kSocketPath[] = "/run/hsmn/hardware_control_internal.sock";
inline constexpr const char kAllowlistPath[] = "/etc/hsmn/hardware_control_allowlist.conf";

inline std::string ResolveSocketPath() {
    const char* override_path = std::getenv("SAGUARO_DAEMON_SOCKET_PATH");
    if (override_path != nullptr && override_path[0] != '\0') {
        return override_path;
    }
    return kSocketPath;
}

inline std::string ResolveAllowlistPath() {
    const char* override_path = std::getenv("SAGUARO_DAEMON_ALLOWLIST_PATH");
    if (override_path != nullptr && override_path[0] != '\0') {
        return override_path;
    }
    return kAllowlistPath;
}
} // namespace hardware::paths

#endif // HARDWARE_HARDWARE_CONTROL_PATHS_H_
