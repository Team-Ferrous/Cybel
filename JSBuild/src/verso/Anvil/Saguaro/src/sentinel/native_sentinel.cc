#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <algorithm>
#include <cstring>

namespace saguaro {
namespace sentinel {

struct Violation {
    std::string file;
    int line;
    std::string rule_id;
    std::string message;
    std::string severity;
};

class NativeSentinelEngine {
public:
    NativeSentinelEngine() {
        // Initialize common secret patterns
        // Standard high-entropy strings, API keys, etc.
        patterns_.push_back({ "SECRET_KEY", std::regex("secret[_-]?key\\s*[:=]\\s*['\"]([a-zA-Z0-9]{20,})['\"]", std::regex_constants::icase) });
        patterns_.push_back({ "AWS_KEY", std::regex("AKIA[0-9A-Z]{16}") });
        patterns_.push_back({ "GENERIC_PWD", std::regex("password\\s*[:=]\\s*['\"]([^'\"]{8,})['\"]", std::regex_constants::icase) });
    }

    std::vector<Violation> scan_file(const std::string& file_path) {
        std::vector<Violation> violations;
        std::ifstream file(file_path);
        if (!file.is_open()) return violations;

        std::string line;
        int line_num = 1;
        while (std::getline(file, line)) {
            // Check for secrets
            for (const auto& p : patterns_) {
                if (std::regex_search(line, p.pattern)) {
                    violations.push_back({
                        file_path,
                        line_num,
                        p.id,
                        "Potential secret exposure detected: " + p.id,
                        "error"
                    });
                }
            }

            // Check for TODOs (Governance)
            if (line.find("TODO") != std::string::npos || line.find("FIXME") != std::string::npos) {
                violations.push_back({
                    file_path,
                    line_num,
                    "NATIVE_GOV_CLEANUP",
                    "Unresolved TODO/FIXME found in production-bound code",
                    "warning"
                });
            }

            line_num++;
        }
        return violations;
    }

private:
    struct Pattern {
        std::string id;
        std::regex pattern;
    };
    std::vector<Pattern> patterns_;
};

} // namespace sentinel
} // namespace saguaro

// C API Wrapper
extern "C" {
    typedef void* saguaro_sentinel_handle_t;

    saguaro_sentinel_handle_t saguaro_native_sentinel_create() {
        return static_cast<saguaro_sentinel_handle_t>(new saguaro::sentinel::NativeSentinelEngine());
    }

    void saguaro_native_sentinel_destroy(saguaro_sentinel_handle_t handle) {
        if (handle) {
            delete static_cast<saguaro::sentinel::NativeSentinelEngine*>(handle);
        }
    }

    int saguaro_native_sentinel_scan(
        saguaro_sentinel_handle_t handle,
        const char* file_path,
        char* output_json,
        int max_len
    ) {
        if (!handle || !file_path || !output_json) return -1;
        
        auto engine = static_cast<saguaro::sentinel::NativeSentinelEngine*>(handle);
        auto violations = engine->scan_file(file_path);

        // Simple JSON serialization (manual for now to avoid heavyweight dependencies)
        std::string json = "[";
        for (size_t i = 0; i < violations.size(); ++i) {
            const auto& v = violations[i];
            json += "{";
            json += "\"file\":\"" + v.file + "\",";
            json += "\"line\":" + std::to_string(v.line) + ",";
            json += "\"rule_id\":\"" + v.rule_id + "\",";
            json += "\"message\":\"" + v.message + "\",";
            json += "\"severity\":\"" + v.severity + "\"";
            json += "}";
            if (i < violations.size() - 1) json += ",";
        }
        json += "]";

        if (json.length() >= static_cast<size_t>(max_len)) {
            return -2; // Buffer too small
        }

        std::strncpy(output_json, json.c_str(), max_len);
        return static_cast<int>(json.length());
    }
}
