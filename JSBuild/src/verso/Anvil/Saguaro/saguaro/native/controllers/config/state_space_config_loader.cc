#include "state_space_config_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <Eigen/Dense> // Include Eigen for MatrixXf

namespace {
// Helper to parse a matrix from a file stream.
// It reads lines until an empty line, a comment, or a new section is encountered.
std::string trim(const std::string& input) {
    const auto first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

Eigen::MatrixXf parse_matrix(std::ifstream& file, const std::string& section_name) {
    std::vector<std::vector<float>> temp_matrix;
    std::string line;
    while (std::getline(file, line) && !line.empty() && line.find('[') == std::string::npos) {
        std::stringstream ss(line);
        std::vector<float> row;
        float value;
        while (ss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            temp_matrix.push_back(row);
        }
    }
    // If we read a new section header, we need to put it back for the next parser
    if (!line.empty() && line.find('[') != std::string::npos) {
        file.seekg(-static_cast<std::streamoff>(line.length()) - 1, std::ios_base::cur);
    }

    if (temp_matrix.empty()) {
        return Eigen::MatrixXf(); // Return empty Eigen matrix
    }

    // Convert to Eigen::MatrixXf
    int rows = temp_matrix.size();
    int cols = temp_matrix[0].size();
    Eigen::MatrixXf eigen_matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        if (temp_matrix[i].size() != static_cast<size_t>(cols)) {
            throw std::runtime_error("Inconsistent row length in matrix section: " + section_name);
        }
        for (int j = 0; j < cols; ++j) {
            eigen_matrix(i, j) = temp_matrix[i][j];
        }
    }
    return eigen_matrix;
}

// Helper to parse a map of floats (like gains or setpoints)
std::map<std::string, float> parse_key_value_map(std::ifstream& file) {
    std::map<std::string, float> map_data;
    std::string line;
    while (std::getline(file, line) && !line.empty() && line.find('[') == std::string::npos) {
        if (line.empty() || line[0] == '#') continue;

        // Find the '=' and parse key/value
        std::stringstream ss(line);
        std::string key;
        float value;
        char equals;
        if (ss >> key >> equals >> value && equals == '=') {
            map_data[key] = value;
        }
    }
    if (!line.empty() && line.find('[') != std::string::npos) {
        file.seekg(-static_cast<std::streamoff>(line.length()) - 1, std::ios_base::cur);
    }
    return map_data;
}

// Helper to parse a vector of strings
std::vector<std::string> parse_string_vector(std::ifstream& file) {
    std::vector<std::string> strings;
    std::string line, item;
    while (std::getline(file, line) && !line.empty() && line.find('[') == std::string::npos) {
        std::stringstream ss(line);
        while (ss >> item) {
            strings.push_back(item);
        }
    }
    if (!line.empty() && line.find('[') != std::string::npos) {
        file.seekg(-static_cast<std::streamoff>(line.length()) - 1, std::ios_base::cur);
    }
    return strings;
}

// Helper to parse a single-row matrix (a vector) from a file stream.
std::vector<float> parse_vector(std::ifstream& file, const std::string& section_name) {
    std::vector<std::vector<float>> matrix_rows;
    std::string line;
    while (std::getline(file, line) && !line.empty() && line.find('[') == std::string::npos) {
        std::stringstream ss(line);
        std::vector<float> row;
        float value;
        while (ss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix_rows.push_back(row);
        }
    }
    if (!line.empty() && line.find('[') != std::string::npos) {
        file.seekg(-static_cast<std::streamoff>(line.length()) - 1, std::ios_base::cur);
    }
    if (matrix_rows.empty()) {
        return {};
    }
    return matrix_rows[0]; // Assuming it's always a single row for a vector
}

// Helper to write an Eigen::MatrixXf to a file stream in the .conf format.
void write_matrix(std::ofstream& file, const std::string& name, const Eigen::MatrixXf& matrix) {
    if (matrix.rows() == 0 || matrix.cols() == 0) return;
    file << "[" << name << "]\n";
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << std::fixed << std::setprecision(8) << matrix(i, j) << (j == matrix.cols() - 1 ? "" : " ");
        }
        file << "\n";
    }
    file << "\n";
}

// Helper to write a vector as a single-row matrix
void write_vector(std::ofstream& file, const std::string& name, const std::vector<float>& vec) {
    if (vec.empty()) return;
    file << "[" << name << "]\n";
    for (size_t i = 0; i < vec.size(); ++i) {
        file << std::fixed << std::setprecision(8) << vec[i] << (i == vec.size() - 1 ? "" : " ");
    }
    file << "\n\n";
}

// Helper to parse a map of string -> list<string>
std::map<std::string, std::vector<std::string>> parse_string_list_map(std::ifstream& file) {
    std::map<std::string, std::vector<std::string>> mapping;
    std::string line;
    while (std::getline(file, line) && !line.empty() && line.find('[') == std::string::npos) {
        if (line.empty() || line[0] == '#') continue;
        auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trim(line.substr(0, pos));
        std::string raw_values = line.substr(pos + 1);
        for (char& ch : raw_values) {
            if (ch == ',') ch = ' ';
        }
        std::stringstream ss(raw_values);
        std::string token;
        std::vector<std::string> values;
        while (ss >> token) {
            values.push_back(trim(token));
        }
        mapping[key] = values;
    }
    if (!line.empty() && line.find('[') != std::string::npos) {
        file.seekg(-static_cast<std::streamoff>(line.length()) - 1, std::ios_base::cur);
    }
    return mapping;
}

void write_string_list_map(std::ofstream& file, const std::string& name, const std::map<std::string, std::vector<std::string>>& map_data) {
    if (map_data.empty()) return;
    file << "[" << name << "]\n";
    for (const auto& pair : map_data) {
        file << pair.first << " =";
        for (const auto& value : pair.second) {
            file << " " << value;
        }
        file << "\n";
    }
    file << "\n";
}

} // anonymous namespace

ControllerConfig ConfigLoader::load_controller_config(const std::string& path) {
    ControllerConfig config;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open state-space config file: " << path << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line == "[A]") config.A = parse_matrix(file, "A");
        else if (line == "[B]") config.B = parse_matrix(file, "B");
        else if (line == "[C]") config.C = parse_matrix(file, "C");
        else if (line == "[D]") config.D = parse_matrix(file, "D");
        else if (line == "[Q]") config.Q = parse_matrix(file, "Q");
        else if (line == "[R_kalman]") config.R_kalman = parse_matrix(file, "R_kalman");
        else if (line == "[R_lqr]") config.R_lqr = parse_matrix(file, "R_lqr");
        else if (line == "[K]") config.K = parse_matrix(file, "K");
        else if (line == "[u_scaler_mean]") config.u_scaler_mean = parse_vector(file, "u_scaler_mean");
        else if (line == "[u_scaler_scale]") config.u_scaler_scale = parse_vector(file, "u_scaler_scale");
        else if (line == "[y_scaler_mean]") config.y_scaler_mean = parse_vector(file, "y_scaler_mean");
        else if (line == "[y_scaler_scale]") config.y_scaler_scale = parse_vector(file, "y_scaler_scale");
    }
    return config;
}

void ConfigLoader::save_controller_config(const std::string& path, const ControllerConfig& config) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to save controller config: " + path);
    }

    write_matrix(file, "A", config.A);
    write_matrix(file, "B", config.B);
    write_matrix(file, "C", config.C);
    write_matrix(file, "D", config.D);
    write_matrix(file, "Q", config.Q);
    write_matrix(file, "R_kalman", config.R_kalman);
    write_matrix(file, "R_lqr", config.R_lqr);
    write_matrix(file, "K", config.K);
    write_vector(file, "u_scaler_mean", config.u_scaler_mean);
    write_vector(file, "u_scaler_scale", config.u_scaler_scale);
    write_vector(file, "y_scaler_mean", config.y_scaler_mean);
    write_vector(file, "y_scaler_scale", config.y_scaler_scale);
}

PIDConfig ConfigLoader::load_pid_config(const std::string& path) {
    PIDConfig config;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: PID config file '" << path << "' not found. Returning empty config." << std::endl;
        return config;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line == "[metric_names]") { config.metric_names = parse_string_vector(file); }
        else if (line == "[control_input_names]") { config.control_input_names = parse_string_vector(file); }
        else if (line == "[Setpoint]") { config.Setpoint = parse_key_value_map(file); }
        else if (line == "[Kp_matrix]") { config.Kp_matrix = parse_matrix(file, "Kp_matrix"); }
        else if (line == "[Ki_matrix]") { config.Ki_matrix = parse_matrix(file, "Ki_matrix"); }
        else if (line == "[Kd_matrix]") { config.Kd_matrix = parse_matrix(file, "Kd_matrix"); }
        else if (line == "[control_min_bounds]") { config.control_min_bounds = parse_vector(file, "control_min_bounds"); }
        else if (line == "[control_max_bounds]") { config.control_max_bounds = parse_vector(file, "control_max_bounds"); }
        else if (line == "[control_metric_map]") { config.control_metric_map = parse_string_list_map(file); }
        else if (line == "[AutoTune]") {
             auto autotune_map = parse_key_value_map(file);
             if (autotune_map.count("output_amplitude")) config.relay_output_amplitude = autotune_map["output_amplitude"];
             if (autotune_map.count("hysteresis")) config.relay_hysteresis = autotune_map["hysteresis"];
             if (autotune_map.count("error_threshold")) config.error_threshold = autotune_map["error_threshold"];
             if (autotune_map.count("patience")) config.patience = static_cast<int>(autotune_map["patience"]);
        }
    }
    return config;
}

void ConfigLoader::save_pid_config(const std::string& path, const PIDConfig& config) {
    std::ofstream config_file(path);
    if (!config_file.is_open()) {
        throw std::runtime_error("Could not open PID config file for writing: " + path);
    }

    auto write_map = [&](const std::string& section_name, const std::map<std::string, float>& map_data) {
        if (map_data.empty()) return;
        config_file << "[" << section_name << "]\n";
        for (const auto& pair : map_data) {
            config_file << pair.first << " = " << std::fixed << std::setprecision(6) << pair.second << "\n";
        }
        config_file << "\n";
    };
    
    auto write_string_vector = [&](const std::string& name, const std::vector<std::string>& vec) {
        if (vec.empty()) return;
        config_file << "[" << name << "]\n";
        for (size_t i = 0; i < vec.size(); ++i) {
            config_file << vec[i] << (i == vec.size() - 1 ? "" : " ");
        }
        config_file << "\n\n";
    };

    write_string_vector("metric_names", config.metric_names);
    write_string_vector("control_input_names", config.control_input_names);

    write_map("Setpoint", config.Setpoint);
    write_matrix(config_file, "Kp_matrix", config.Kp_matrix);
    write_matrix(config_file, "Ki_matrix", config.Ki_matrix);
    write_matrix(config_file, "Kd_matrix", config.Kd_matrix);
    write_vector(config_file, "control_min_bounds", config.control_min_bounds);
    write_vector(config_file, "control_max_bounds", config.control_max_bounds);
    write_string_list_map(config_file, "control_metric_map", config.control_metric_map);
    
    std::map<std::string, float> autotune_params;
    autotune_params["output_amplitude"] = config.relay_output_amplitude;
    autotune_params["hysteresis"] = config.relay_hysteresis;
    autotune_params["error_threshold"] = config.error_threshold;
    autotune_params["patience"] = static_cast<float>(config.patience);
    write_map("AutoTune", autotune_params);

    std::cout << "Saved PID config to '" << path << "'" << std::endl;
}
