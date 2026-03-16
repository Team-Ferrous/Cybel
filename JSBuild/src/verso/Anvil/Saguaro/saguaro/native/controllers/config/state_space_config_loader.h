#ifndef CONFIG_STATE_SPACE_CONFIG_LOADER_H_
#define CONFIG_STATE_SPACE_CONFIG_LOADER_H_
#include <vector>
#include <string>
#include "state_space_config.h"
#include "pid_config.h" // Include the new PID config header

// This class encapsulates the logic for loading controller configurations
// from files. It is designed to parse state-space matrices from a text file.
class ConfigLoader {
public:
    // Loads state-space controller configuration from the specified file path.
    // The file should contain matrices in a simple text format, like the one
    // generated from a .npz file.
    static ControllerConfig load_controller_config(const std::string& path);

    // Saves the provided controller configuration to the specified file path.
    // Note: This is currently a no-op as matrices are expected to be generated
    // by external Python scripts.
    static void save_controller_config(const std::string& path, const ControllerConfig& config);

    // Loads PID controller configuration from the specified file path.
    static PIDConfig load_pid_config(const std::string& path);

    // Saves the provided PID configuration to the specified file path.
    static void save_pid_config(const std::string& path, const PIDConfig& config);
};

#endif // CONFIG_STATE_SPACE_CONFIG_LOADER_H_
