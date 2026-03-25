// saguaro.native/ops/hpo_main.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// HPO Orchestrator - Pure C++ implementation for hyperparameter optimization.
// This binary coordinates HPO trials via file-based IPC with Python training.
// 
// Usage:
//   ./hpo_main --config=/path/to/hpo_config.json
//   ./hpo_main --target-arch=x86_64 --config=/path/to/config.json
//
// NO TensorFlow C++ dependencies - works with pip-installed TensorFlow.

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <limits>
#include <atomic>
#include <csignal>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <map>
#include <random>
#include <sys/ptrace.h>

#include "common/parallel/parallel_backend.h"
#include "common/runtime_security.h"

namespace {

// Global flag for graceful shutdown
std::atomic<bool> g_shutdown_requested{false};

void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "[HPO] Shutdown signal received, gracefully stopping trials..." << std::endl;
        g_shutdown_requested.store(true);
    }
}

// =============================================================================
// Simple JSON Parser (minimal, for config files only)
// =============================================================================

class JsonValue {
public:
    enum Type { NONE, NUMBER, STRING, BOOL, ARRAY, OBJECT };
    
    Type type = NONE;
    double number_val = 0.0;
    std::string string_val;
    bool bool_val = false;
    std::vector<JsonValue> array_val;
    std::map<std::string, JsonValue> object_val;
    
    bool is_object() const { return type == OBJECT; }
    bool is_array() const { return type == ARRAY; }
    bool is_string() const { return type == STRING; }
    bool is_number() const { return type == NUMBER; }
    
    const JsonValue& operator[](const std::string& key) const {
        static JsonValue empty;
        if (type != OBJECT) return empty;
        auto it = object_val.find(key);
        return (it != object_val.end()) ? it->second : empty;
    }
    
    const JsonValue& operator[](size_t idx) const {
        static JsonValue empty;
        if (type != ARRAY || idx >= array_val.size()) return empty;
        return array_val[idx];
    }
    
    bool has(const std::string& key) const {
        return type == OBJECT && object_val.find(key) != object_val.end();
    }
    
    size_t size() const {
        if (type == ARRAY) return array_val.size();
        if (type == OBJECT) return object_val.size();
        return 0;
    }
};

class JsonParser {
public:
    static JsonValue parse(const std::string& json) {
        size_t pos = 0;
        return parseValue(json, pos);
    }
    
private:
    static void skipWhitespace(const std::string& s, size_t& pos) {
        while (pos < s.size() && std::isspace(s[pos])) pos++;
    }
    
    static JsonValue parseValue(const std::string& s, size_t& pos) {
        skipWhitespace(s, pos);
        if (pos >= s.size()) return JsonValue();
        
        char c = s[pos];
        if (c == '{') return parseObject(s, pos);
        if (c == '[') return parseArray(s, pos);
        if (c == '"') return parseString(s, pos);
        if (c == 't' || c == 'f') return parseBool(s, pos);
        if (c == 'n') { pos += 4; return JsonValue(); } // null
        if (c == '-' || std::isdigit(c)) return parseNumber(s, pos);
        return JsonValue();
    }
    
    static JsonValue parseObject(const std::string& s, size_t& pos) {
        JsonValue result;
        result.type = JsonValue::OBJECT;
        pos++; // skip '{'
        skipWhitespace(s, pos);
        
        while (pos < s.size() && s[pos] != '}') {
            skipWhitespace(s, pos);
            if (s[pos] != '"') break;
            
            std::string key = parseString(s, pos).string_val;
            skipWhitespace(s, pos);
            if (pos < s.size() && s[pos] == ':') pos++;
            skipWhitespace(s, pos);
            
            result.object_val[key] = parseValue(s, pos);
            
            skipWhitespace(s, pos);
            if (pos < s.size() && s[pos] == ',') pos++;
        }
        
        if (pos < s.size() && s[pos] == '}') pos++;
        return result;
    }
    
    static JsonValue parseArray(const std::string& s, size_t& pos) {
        JsonValue result;
        result.type = JsonValue::ARRAY;
        pos++; // skip '['
        skipWhitespace(s, pos);
        
        while (pos < s.size() && s[pos] != ']') {
            result.array_val.push_back(parseValue(s, pos));
            skipWhitespace(s, pos);
            if (pos < s.size() && s[pos] == ',') pos++;
            skipWhitespace(s, pos);
        }
        
        if (pos < s.size() && s[pos] == ']') pos++;
        return result;
    }
    
    static JsonValue parseString(const std::string& s, size_t& pos) {
        JsonValue result;
        result.type = JsonValue::STRING;
        pos++; // skip opening quote
        
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] == '\\' && pos + 1 < s.size()) {
                pos++;
                char escaped = s[pos];
                switch (escaped) {
                    case 'n': result.string_val += '\n'; break;
                    case 't': result.string_val += '\t'; break;
                    case 'r': result.string_val += '\r'; break;
                    case '"': result.string_val += '"'; break;
                    case '\\': result.string_val += '\\'; break;
                    default: result.string_val += escaped;
                }
            } else {
                result.string_val += s[pos];
            }
            pos++;
        }
        
        if (pos < s.size() && s[pos] == '"') pos++;
        return result;
    }
    
    static JsonValue parseNumber(const std::string& s, size_t& pos) {
        JsonValue result;
        result.type = JsonValue::NUMBER;
        size_t start = pos;
        
        if (s[pos] == '-') pos++;
        while (pos < s.size() && std::isdigit(s[pos])) pos++;
        if (pos < s.size() && s[pos] == '.') {
            pos++;
            while (pos < s.size() && std::isdigit(s[pos])) pos++;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            pos++;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) pos++;
            while (pos < s.size() && std::isdigit(s[pos])) pos++;
        }
        
        result.number_val = std::stod(s.substr(start, pos - start));
        return result;
    }
    
    static JsonValue parseBool(const std::string& s, size_t& pos) {
        JsonValue result;
        result.type = JsonValue::BOOL;
        if (s.substr(pos, 4) == "true") {
            result.bool_val = true;
            pos += 4;
        } else if (s.substr(pos, 5) == "false") {
            result.bool_val = false;
            pos += 5;
        }
        return result;
    }
};

// =============================================================================
// HPO Configuration
// =============================================================================

struct HPOConfig {
    // Trial settings
    int max_steps_per_trial = 100;
    int checkpoint_frequency = 10;
    int metrics_report_frequency = 1;
    int early_stopping_patience = 10;
    float early_stopping_min_delta = 0.001f;
    
    // Training parameters
    int batch_size = 1;
    int sequence_length = 128;
    int input_dim = 128;
    int label_dim = 1000;
    
    // Model configuration (user-provided, Lite edition limits enforced)
    int vocab_size = 32000;
    int context_window = 4096;
    int embedding_dim = 512;
    int num_reasoning_blocks = 8;
    int num_moe_experts = 8;
    std::string position_embedding = "rope";
    
    // HSMN-specific architecture params (for tuning)
    int mamba_state_dim = 64;
    int moe_top_k = 2;
    int superposition_dim = 2;
    int tt_rank_middle = 16;
    int hamiltonian_hidden_dim = 256;
    
    // Architecture tuning flags
    bool tune_embedding_dim = false;
    bool tune_reasoning_blocks = false;
    bool tune_moe_experts = false;
    
    // Paths
    std::string trial_dir_base = "artifacts/hpo_trials";
    std::string curriculum_path = "";
    
    // Resource limits
    int max_memory_mb = 8192;
    int max_wall_time_seconds = 3600;
    
    // Lite Edition Limits (enforced via obfuscated runtime calc)
    static int64_t GetLiteMaxParams() { 
        volatile int64_t a = 0x9502F900; 
        volatile int64_t b = 0x5;
        return (a * b) + 0x3800; // 20B
    }
    static int64_t GetLiteMaxContext() { return 0x4C4B40 ^ 0x0; } // 5M
    static int32_t GetLiteMaxVocab() { return 0x10000; } // 65536
    static int32_t GetLiteMaxEmbed() { return 0x1000; } // 4096
    static int32_t GetLiteMaxBlocks() { return 0x18; } // 24
    static int32_t GetLiteMaxExperts() { return 0xC; } // 12
    static int32_t GetLiteMaxSuperposition() { return 0x4; } // 4

    // User param budget (default to max)
    int64_t param_budget = GetLiteMaxParams();
    
    // Estimate parameter count for this configuration
    int64_t EstimateParams() const {
        // Simplified param estimation matching Python logic
        // embedding = vocab_size * embedding_dim
        // blocks = num_reasoning_blocks * embedding_dim^2 * 4 (FFN)
        // moe = num_moe_experts * embedding_dim^2
        int64_t embedding_params = static_cast<int64_t>(vocab_size) * embedding_dim;
        int64_t block_params = static_cast<int64_t>(num_reasoning_blocks) * 
                               embedding_dim * embedding_dim * 4;
        int64_t moe_params = static_cast<int64_t>(num_moe_experts) * 
                             embedding_dim * embedding_dim;
        return embedding_params + block_params + moe_params;
    }
    
    bool ValidateLiteLimits() const {
        return vocab_size <= GetLiteMaxVocab() &&
               context_window <= GetLiteMaxContext() &&
               embedding_dim <= GetLiteMaxEmbed() &&
               num_reasoning_blocks <= GetLiteMaxBlocks() &&
               num_moe_experts <= GetLiteMaxExperts() &&
               superposition_dim <= GetLiteMaxSuperposition();
    }
    
    bool ValidateParamBudget() const {
        int64_t estimated = EstimateParams();
        // Must not exceed both Lite limit AND user-specified budget
        int64_t effective_budget = std::min(param_budget, GetLiteMaxParams());
        return estimated <= effective_budget;
    }
};

struct RunnerOptions {
    std::string target_arch;
    std::string config_path;
};

// =============================================================================
// Utility Functions
// =============================================================================

std::string NormalizeToken(const std::string& raw) {
    std::string normalized;
    normalized.reserve(raw.size());
    for (char ch : raw) {
        unsigned char lowered = static_cast<unsigned char>(std::tolower(static_cast<unsigned char>(ch)));
        if (lowered == '-') lowered = '_';
        normalized.push_back(static_cast<char>(lowered));
    }
    return normalized;
}

std::string CanonicalizeTargetArch(const std::string& raw) {
    auto normalized = NormalizeToken(raw);
    if (normalized.empty()) return {};
    if (normalized == "x86_64" || normalized == "amd64" || normalized == "x64") return "x86_64";
    if (normalized == "arm64" || normalized == "aarch64" || normalized == "armv8") return "arm64";
    return {};
}

RunnerOptions ParseRunnerOptions(int argc, char* argv[]) {
    RunnerOptions options;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--target-arch" && i + 1 < argc) {
            options.target_arch = argv[++i];
            continue;
        }
        const std::string arch_prefix = "--target-arch=";
        if (arg.rfind(arch_prefix, 0) == 0) {
            options.target_arch = arg.substr(arch_prefix.size());
            continue;
        }
        if (arg == "--config" && i + 1 < argc) {
            options.config_path = argv[++i];
            continue;
        }
        const std::string config_prefix = "--config=";
        if (arg.rfind(config_prefix, 0) == 0) {
            options.config_path = arg.substr(config_prefix.size());
            continue;
        }
    }
    return options;
}

bool ApplyTargetArchEnv(const std::string& value) {
    if (value.empty()) return false;
    const std::string canonical = CanonicalizeTargetArch(value);
    if (canonical.empty()) {
        std::cerr << "Unsupported target architecture '" << value
                  << "'. Expected one of: x86_64, arm64." << std::endl;
        return false;
    }
    ::setenv("VERSO_TARGET_ARCH", canonical.c_str(), 1);
    std::cout << "[HPO] Using target architecture: " << canonical << std::endl;
    return true;
}

std::string ReadFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool WriteJsonFile(const std::string& path, const std::string& json_content) {
    std::string temp_path = path + ".tmp";
    std::ofstream temp_file(temp_path);
    if (!temp_file.is_open()) {
        std::cerr << "[HPO] Failed to open temporary file: " << temp_path << std::endl;
        return false;
    }
    temp_file << json_content;
    temp_file.close();
    
    if (std::rename(temp_path.c_str(), path.c_str()) != 0) {
        std::cerr << "[HPO] Failed to rename " << temp_path << " to " << path << std::endl;
        return false;
    }
    return true;
}

bool WriteTrialStatus(const std::string& trial_dir, const std::string& status,
                      double best_loss = -1.0, int total_steps = 0) {
    std::string status_file = trial_dir + "/status.json";
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    
    std::ostringstream json;
    json << "{\n";
    json << "  \"trial_id\": \"" << trial_dir.substr(trial_dir.find_last_of("/") + 1) << "\",\n";
    json << "  \"status\": \"" << status << "\",\n";
    json << "  \"timestamp\": " << timestamp << ",\n";
    json << "  \"total_steps\": " << total_steps;
    if (best_loss >= 0.0) {
        json << ",\n  \"best_loss\": " << best_loss;
    }
    json << "\n}\n";
    
    return WriteJsonFile(status_file, json.str());
}

bool AppendTrialMetrics(const std::string& trial_dir, int step,
                        double loss, double wall_time_seconds) {
    std::string metrics_file = trial_dir + "/metrics.jsonl";
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() / 1000.0;
    
    std::ostringstream json;
    json << "{";
    json << "\"trial_id\":\"" << trial_dir.substr(trial_dir.find_last_of("/") + 1) << "\",";
    json << "\"step\":" << step << ",";
    json << "\"timestamp\":" << std::fixed << std::setprecision(3) << timestamp << ",";
    json << "\"loss\":" << std::fixed << std::setprecision(6) << loss << ",";
    json << "\"wall_time_seconds\":" << std::fixed << std::setprecision(2) << wall_time_seconds;
    json << "}\n";
    
    std::ofstream metrics_out(metrics_file, std::ios::app);
    if (!metrics_out.is_open()) {
        std::cerr << "[HPO] Failed to open metrics file: " << metrics_file << std::endl;
        return false;
    }
    metrics_out << json.str();
    metrics_out.flush();
    return true;
}

HPOConfig LoadHPOConfig(const std::string& config_path) {
    HPOConfig config;
    
    std::string json_str = ReadFile(config_path);
    if (json_str.empty()) {
        std::cerr << "[HPO] Warning: Could not open config file '" << config_path
                  << "', using defaults." << std::endl;
        return config;
    }
    
    JsonValue root = JsonParser::parse(json_str);
    if (!root.is_object()) {
        std::cerr << "[HPO] Warning: Failed to parse config JSON, using defaults." << std::endl;
        return config;
    }
    
    // Parse hpo_runner section
    if (root.has("hpo_runner")) {
        const JsonValue& runner = root["hpo_runner"];
        
        // Trial settings
        if (runner.has("trial_settings")) {
            const JsonValue& trial = runner["trial_settings"];
            if (trial.has("max_steps_per_trial") && trial["max_steps_per_trial"].is_number()) {
                config.max_steps_per_trial = static_cast<int>(trial["max_steps_per_trial"].number_val);
            }
            if (trial.has("checkpoint_frequency") && trial["checkpoint_frequency"].is_number()) {
                config.checkpoint_frequency = static_cast<int>(trial["checkpoint_frequency"].number_val);
            }
            if (trial.has("early_stopping") && trial["early_stopping"].is_object()) {
                const JsonValue& es = trial["early_stopping"];
                if (es.has("patience")) config.early_stopping_patience = static_cast<int>(es["patience"].number_val);
                if (es.has("min_delta")) config.early_stopping_min_delta = static_cast<float>(es["min_delta"].number_val);
            }
        }
        
        // Training parameters
        if (runner.has("training_parameters")) {
            const JsonValue& training = runner["training_parameters"];
            if (training.has("batch_size")) config.batch_size = static_cast<int>(training["batch_size"].number_val);
            if (training.has("sequence_length")) config.sequence_length = static_cast<int>(training["sequence_length"].number_val);
            if (training.has("input_dim")) config.input_dim = static_cast<int>(training["input_dim"].number_val);
            if (training.has("label_dim")) config.label_dim = static_cast<int>(training["label_dim"].number_val);
        }
        
        // Curriculum
        if (runner.has("curriculum") && runner["curriculum"].is_object()) {
            const JsonValue& curriculum = runner["curriculum"];
            if (curriculum.has("path") && curriculum["path"].is_string()) {
                config.curriculum_path = curriculum["path"].string_val;
            }
        }
        
        // Output
        if (runner.has("output") && runner["output"].is_object()) {
            const JsonValue& output = runner["output"];
            if (output.has("trial_dir_base") && output["trial_dir_base"].is_string()) {
                config.trial_dir_base = output["trial_dir_base"].string_val;
            }
        }
    }
    
    // Parse resource_limits section
    if (root.has("resource_limits")) {
        const JsonValue& limits = root["resource_limits"];
        if (limits.has("max_memory_mb")) config.max_memory_mb = static_cast<int>(limits["max_memory_mb"].number_val);
        if (limits.has("max_wall_time_seconds")) config.max_wall_time_seconds = static_cast<int>(limits["max_wall_time_seconds"].number_val);
    }
    
    // Parse model_config section (from WebUI frontend)
    if (root.has("model_config")) {
        const JsonValue& model = root["model_config"];
        if (model.has("vocab_size") && model["vocab_size"].is_number()) {
            config.vocab_size = static_cast<int>(model["vocab_size"].number_val);
        }
        if (model.has("context_window") && model["context_window"].is_number()) {
            config.context_window = static_cast<int>(model["context_window"].number_val);
        }
        if (model.has("embedding_dim") && model["embedding_dim"].is_number()) {
            config.embedding_dim = static_cast<int>(model["embedding_dim"].number_val);
        }
        if (model.has("num_reasoning_blocks") && model["num_reasoning_blocks"].is_number()) {
            config.num_reasoning_blocks = static_cast<int>(model["num_reasoning_blocks"].number_val);
        }
        if (model.has("num_moe_experts") && model["num_moe_experts"].is_number()) {
            config.num_moe_experts = static_cast<int>(model["num_moe_experts"].number_val);
        }
        if (model.has("param_budget") && model["param_budget"].is_number()) {
            config.param_budget = static_cast<int64_t>(model["param_budget"].number_val);
        }
        if (model.has("position_embedding") && model["position_embedding"].is_string()) {
            config.position_embedding = model["position_embedding"].string_val;
        }
    }
    
    // Parse architecture_tuning flags
    if (root.has("architecture_tuning")) {
        const JsonValue& tuning = root["architecture_tuning"];
        if (tuning.has("tune_embedding_dim")) {
            config.tune_embedding_dim = tuning["tune_embedding_dim"].bool_val;
        }
        if (tuning.has("tune_reasoning_blocks")) {
            config.tune_reasoning_blocks = tuning["tune_reasoning_blocks"].bool_val;
        }
        if (tuning.has("tune_moe_experts")) {
            config.tune_moe_experts = tuning["tune_moe_experts"].bool_val;
        }
    }
    
    // Validate Lite edition limits
    if (!config.ValidateLiteLimits()) {
        std::cerr << "[HPO] Error 0x1002: Configuration integrity check failed." << std::endl;
        std::cerr << "  Reference Code: 0xV" << std::hex << HPOConfig::GetLiteMaxVocab() << std::dec << std::endl;
        std::cerr << "  Reference Code: 0xC" << std::hex << HPOConfig::GetLiteMaxContext() << std::dec << std::endl;
        std::cerr << "  Reference Code: 0xE" << std::hex << HPOConfig::GetLiteMaxEmbed() << std::dec << std::endl;
        std::cerr << "  Reference Code: 0xB" << std::hex << HPOConfig::GetLiteMaxBlocks() << std::dec << std::endl;
        std::cerr << "  Reference Code: 0xX" << std::hex << HPOConfig::GetLiteMaxExperts() << std::dec << std::endl;
        std::cerr << "  Reference Code: 0xS" << std::hex << HPOConfig::GetLiteMaxSuperposition() << std::dec << std::endl;
        // Clamp to limits instead of failing
        config.vocab_size = std::min(config.vocab_size, (int)HPOConfig::GetLiteMaxVocab());
        config.context_window = std::min(config.context_window, (int)HPOConfig::GetLiteMaxContext());
        config.embedding_dim = std::min(config.embedding_dim, (int)HPOConfig::GetLiteMaxEmbed());
        config.num_reasoning_blocks = std::min(config.num_reasoning_blocks, (int)HPOConfig::GetLiteMaxBlocks());
        config.num_moe_experts = std::min(config.num_moe_experts, (int)HPOConfig::GetLiteMaxExperts());
        config.superposition_dim = std::min(config.superposition_dim, (int)HPOConfig::GetLiteMaxSuperposition());
    }
    
    // Validate parameter budget (both user-specified AND Lite 20B max)
    if (!config.ValidateParamBudget()) {
        int64_t estimated = config.EstimateParams();
        int64_t effective_budget = std::min(config.param_budget, HPOConfig::GetLiteMaxParams());
        std::cerr << "[HPO] Error 0x1003: Resource allocation failed (Code " 
                  << std::hex << estimated << " > " << effective_budget << std::dec << ")" << std::endl;
        // Note: Cannot clamp params - must reduce architecture dimensions
    }
    
    std::cout << "[HPO] Loaded configuration from: " << config_path << std::endl;
    std::cout << "  Max steps per trial: " << config.max_steps_per_trial << std::endl;
    std::cout << "  Batch size: " << config.batch_size << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Context window: " << config.context_window << std::endl;
    std::cout << "  Embedding dim: " << config.embedding_dim << std::endl;
    std::cout << "  Reasoning blocks: " << config.num_reasoning_blocks << std::endl;
    std::cout << "  MoE experts: " << config.num_moe_experts << std::endl;
    
    return config;
}

// =============================================================================
// Hyperparameter Sampling
// =============================================================================

double SampleLogUniform(double min_val, double max_val, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(std::log(min_val), std::log(max_val));
    return std::exp(dist(rng));
}

double SampleLinearUniform(double min_val, double max_val, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(min_val, max_val);
    return dist(rng);
}

int SampleIntUniform(int min_val, int max_val, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    return dist(rng);
}

std::string SampleHyperparameters(const JsonValue& config_value, int trial_id, const HPOConfig& hpo_config) {
    std::mt19937 rng(trial_id);  // Deterministic per trial
    
    std::ostringstream json;
    json << "{\n";
    
    // Base configuration from hpo_config
    json << "  \"batch_size\": " << hpo_config.batch_size << ",\n";
    json << "  \"sequence_length\": " << hpo_config.sequence_length << ",\n";
    json << "  \"input_dim\": " << hpo_config.input_dim << ",\n";
    json << "  \"label_dim\": " << hpo_config.label_dim << ",\n";
    json << "  \"total_training_steps\": " << hpo_config.max_steps_per_trial << ",\n";
    
    // Sample hyperparameters from search space if available
    if (config_value.has("search_space") && config_value["search_space"].is_object()) {
        const JsonValue& search_space = config_value["search_space"];
        
        // Learning rate (log scale)
        if (search_space.has("learning_rate") && search_space["learning_rate"].is_object()) {
            const JsonValue& lr = search_space["learning_rate"];
            double lr_min = lr.has("min") ? lr["min"].number_val : 1e-5;
            double lr_max = lr.has("max") ? lr["max"].number_val : 1e-2;
            double learning_rate = SampleLogUniform(lr_min, lr_max, rng);
            json << "  \"learning_rate\": " << learning_rate << ",\n";
        } else {
            json << "  \"learning_rate\": 0.001,\n";
        }
        
        // Warmup steps
        if (search_space.has("warmup_steps") && search_space["warmup_steps"].is_object()) {
            const JsonValue& ws = search_space["warmup_steps"];
            int ws_min = ws.has("min") ? static_cast<int>(ws["min"].number_val) : 0;
            int ws_max = ws.has("max") ? static_cast<int>(ws["max"].number_val) : 1000;
            int warmup_steps = SampleIntUniform(ws_min, ws_max, rng);
            json << "  \"warmup_steps\": " << warmup_steps << ",\n";
        } else {
            json << "  \"warmup_steps\": 100,\n";
        }
        
        // Weight decay
        if (search_space.has("weight_decay") && search_space["weight_decay"].is_object()) {
            const JsonValue& wd = search_space["weight_decay"];
            double wd_min = wd.has("min") ? wd["min"].number_val : 0.0;
            double wd_max = wd.has("max") ? wd["max"].number_val : 0.1;
            double weight_decay = SampleLinearUniform(wd_min, wd_max, rng);
            json << "  \"weight_decay\": " << weight_decay << ",\n";
        } else {
            json << "  \"weight_decay\": 0.01,\n";
        }
        
        // Optimizer choice
        if (search_space.has("optimizer") && search_space["optimizer"].is_object()) {
            const JsonValue& opt = search_space["optimizer"];
            if (opt.has("choices") && opt["choices"].is_array() && opt["choices"].size() > 0) {
                size_t idx = static_cast<size_t>(rng()) % opt["choices"].size();
                json << "  \"optimizer\": \"" << opt["choices"][idx].string_val << "\",\n";
            } else {
                json << "  \"optimizer\": \"adam\",\n";
            }
        } else {
            json << "  \"optimizer\": \"adam\",\n";
        }
    } else {
        // Default hyperparameters
        json << "  \"learning_rate\": 0.001,\n";
        json << "  \"warmup_steps\": 100,\n";
        json << "  \"weight_decay\": 0.01,\n";
        json << "  \"optimizer\": \"adam\",\n";
    }
    
    // Add model configuration from user settings
    json << "  \"vocab_size\": " << hpo_config.vocab_size << ",\n";
    json << "  \"context_window\": " << hpo_config.context_window << ",\n";
    json << "  \"embedding_dim\": " << hpo_config.embedding_dim << ",\n";
    json << "  \"num_reasoning_blocks\": " << hpo_config.num_reasoning_blocks << ",\n";
    json << "  \"num_moe_experts\": " << hpo_config.num_moe_experts << ",\n";
    json << "  \"position_embedding\": \"" << hpo_config.position_embedding << "\",\n";
    
    // Add HSMN-specific architecture params
    json << "  \"mamba_state_dim\": " << hpo_config.mamba_state_dim << ",\n";
    json << "  \"moe_top_k\": " << hpo_config.moe_top_k << ",\n";
    json << "  \"superposition_dim\": " << hpo_config.superposition_dim << ",\n";
    json << "  \"tt_rank_middle\": " << hpo_config.tt_rank_middle << ",\n";
    json << "  \"hamiltonian_hidden_dim\": " << hpo_config.hamiltonian_hidden_dim << ",\n";
    
    // Add early stopping parameters
    json << "  \"early_stopping_patience\": " << hpo_config.early_stopping_patience << ",\n";
    json << "  \"early_stopping_min_delta\": " << hpo_config.early_stopping_min_delta << "\n";
    json << "}\n";
    
    return json.str();
}

bool WriteTrialConfig(const std::string& trial_dir, const std::string& config_json) {
    std::string config_file = trial_dir + "/config.json";
    std::ofstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "[HPO] Failed to open trial config file: " << config_file << std::endl;
        return false;
    }
    file << config_json;
    file.close();
    return true;
}

// =============================================================================
// Trial Execution
// =============================================================================

bool InvokePythonTrialRunner(const std::string& trial_id, const std::string& trial_dir, 
                              int max_steps, const std::string& project_root) {
    std::string config_path = trial_dir + "/config.json";
    
    // Build Python command
    std::ostringstream cmd;
    cmd << "cd " << project_root << " && "
        << "source venv/bin/activate && "
        << "python3 -m highnoon.training.hpo_trial_runner "
        << "--trial_id=" << trial_id << " "
        << "--config=" << config_path << " "
        << "--max_steps=" << max_steps << " "
        << "2>&1 | tee " << trial_dir << "/training.log";
    
    std::cout << "[HPO] Executing: " << cmd.str() << std::endl;
    
    int exit_code = std::system(cmd.str().c_str());
    
    if (exit_code != 0) {
        std::cerr << "[HPO] Python trial runner failed with exit code: " << exit_code << std::endl;
        return false;
    }
    
    return true;
}

double ReadTrialResults(const std::string& trial_dir) {
    std::string status_file = trial_dir + "/status.json";
    std::string json_str = ReadFile(status_file);
    if (json_str.empty()) {
        std::cerr << "[HPO] Failed to read status file: " << status_file << std::endl;
        return std::numeric_limits<double>::infinity();
    }
    
    JsonValue status = JsonParser::parse(json_str);
    if (status.has("best_loss") && status["best_loss"].is_number()) {
        return status["best_loss"].number_val;
    }
    
    return std::numeric_limits<double>::infinity();
}

}  // namespace

// =============================================================================
// Main Entry Point
// =============================================================================

int main(int argc, char* argv[]) {
    // Register signal handlers for graceful shutdown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    // Basic Anti-Debugging Check (Phase 1)
    if (ptrace(PTRACE_TRACEME, 0, 1, 0) < 0) {
        // Debugger detected - exit silently
        return 0;
    }
    
    RunnerOptions options = ParseRunnerOptions(argc, argv);
    if (!options.target_arch.empty() && !ApplyTargetArchEnv(options.target_arch)) {
        return 2;
    }
    
    // Detect project root
    std::string project_root = ".";
    char* env_root = std::getenv("SAGUARO_PROJECT_ROOT");
    if (env_root) {
        project_root = env_root;
    }
    
    // Load HPO configuration
    std::string config_path = options.config_path.empty()
        ? "artifacts/hpo_trials/hpo_config.json"
        : options.config_path;
    HPOConfig hpo_config = LoadHPOConfig(config_path);
    
    // Load config JSON for hyperparameter sampling
    std::string config_json_str = ReadFile(config_path);
    JsonValue config_value = JsonParser::parse(config_json_str);
    
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    HighNoon HPO Orchestrator v2.0                        ║" << std::endl;
    std::cout << "║                         Verso Industries 2025                            ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;
    
    // Get number of trials to run
    int num_trials = 10;  // Default
    if (config_value.has("num_trials") && config_value["num_trials"].is_number()) {
        num_trials = static_cast<int>(config_value["num_trials"].number_val);
    }
    
    std::cout << "[HPO] Starting " << num_trials << " trials" << std::endl;
    
    // Create base trial directory
    std::string mkdir_cmd = "mkdir -p " + hpo_config.trial_dir_base;
    std::system(mkdir_cmd.c_str());
    
    // Track best result
    double best_loss = std::numeric_limits<double>::infinity();
    std::string best_trial;
    
    // Run trials
    for (int trial_idx = 0; trial_idx < num_trials; ++trial_idx) {
        SAGUARO_SECURITY_HEARTBEAT();
        if (g_shutdown_requested.load()) {
            break;
        }
        std::string trial_id = "trial_" + std::to_string(trial_idx);
        std::string trial_dir = hpo_config.trial_dir_base + "/" + trial_id;
        
        // Create trial directory
        std::string trial_mkdir_cmd = "mkdir -p " + trial_dir;
        if (std::system(trial_mkdir_cmd.c_str()) != 0) {
            std::cerr << "[HPO] Failed to create trial directory: " << trial_dir << std::endl;
            continue;
        }
        
        // Write initial trial status
        WriteTrialStatus(trial_dir, "running");
        
        auto trial_start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "[HPO] Running trial " << trial_id << " (" << (trial_idx + 1) << "/" << num_trials << ")" << std::endl;
        
        // Sample hyperparameters
        std::string trial_config_json = SampleHyperparameters(config_value, trial_idx, hpo_config);
        std::cout << "  Sampled hyperparameters:\n" << trial_config_json << std::endl;
        
        // Write trial configuration
        if (!WriteTrialConfig(trial_dir, trial_config_json)) {
            std::cerr << "[HPO] Failed to write trial config for " << trial_id << std::endl;
            WriteTrialStatus(trial_dir, "failed");
            continue;
        }
        
        // Invoke Python trial runner
        bool success = InvokePythonTrialRunner(trial_id, trial_dir, hpo_config.max_steps_per_trial, project_root);
        
        auto trial_end_time = std::chrono::high_resolution_clock::now();
        double wall_time = std::chrono::duration<double>(trial_end_time - trial_start_time).count();
        
        if (!success) {
            std::cerr << "[HPO] Trial " << trial_id << " failed during Python execution" << std::endl;
            WriteTrialStatus(trial_dir, "failed", -1.0, 0);
        } else {
            // Read results
            double loss = ReadTrialResults(trial_dir);
            
            if (std::isinf(loss)) {
                std::cerr << "  [HPO] Trial " << trial_id << " completed but failed to read loss" << std::endl;
                WriteTrialStatus(trial_dir, "failed");
            } else {
                std::cout << "  [HPO] Trial " << trial_id << " completed. Best Loss: " << loss
                          << " (wall time: " << wall_time << "s)" << std::endl;
                WriteTrialStatus(trial_dir, "completed", loss, hpo_config.max_steps_per_trial);
                
                // Track best
                if (loss < best_loss) {
                    best_loss = loss;
                    best_trial = trial_id;
                    std::cout << "  [HPO] New best trial: " << best_trial << " with loss " << best_loss << std::endl;
                }
            }
        }
    }
    
    // Summary
    std::cout << std::endl;
    std::cout << "════════════════════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "[HPO] Search completed!" << std::endl;
    if (!best_trial.empty()) {
        std::cout << "[HPO] Best trial: " << best_trial << " with loss " << best_loss << std::endl;
    }
    std::cout << "════════════════════════════════════════════════════════════════════════════" << std::endl;
    
    return 0;
}
