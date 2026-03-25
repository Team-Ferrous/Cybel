#include "sensitivity_tracker.h"

#include <ctime>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace saguaro {
namespace control {

SensitivityTracker::SensitivityTracker()
    : current_batch_(0) {}

void SensitivityTracker::record_update(
    const std::string& control_name,
    const std::string& metric_name,
    float control_delta,
    float metric_delta,
    int batch_number
) {
    // Update global batch tracker
    current_batch_ = batch_number;

    // Skip if both deltas are effectively zero (no change)
    if (std::fabs(control_delta) < 1e-9f && std::fabs(metric_delta) < 1e-9f) {
        return;
    }

    // Get or create sensitivity entry
    auto key = std::make_pair(control_name, metric_name);
    SensitivityEntry& entry = sensitivity_map_[key];

    // Update sample count
    entry.sample_count++;
    entry.last_update_batch = batch_number;

    // Update influence magnitude (moving average of |Δy| / |Δu|)
    // Avoid division by zero with small epsilon
    const float epsilon = 1e-9f;
    float new_influence = std::fabs(metric_delta) / (std::fabs(control_delta) + epsilon);

    // Exponential moving average (EMA) with alpha = 0.9 (recent bias)
    const float alpha = 0.9f;
    if (entry.sample_count == 1) {
        entry.influence_magnitude = new_influence;
    } else {
        entry.influence_magnitude = alpha * new_influence + (1.0f - alpha) * entry.influence_magnitude;
    }

    // Update incremental correlation statistics (Welford's online algorithm)
    // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    entry.sum_control_delta += static_cast<double>(control_delta);
    entry.sum_metric_delta += static_cast<double>(metric_delta);
    entry.sum_control_sq += static_cast<double>(control_delta) * static_cast<double>(control_delta);
    entry.sum_metric_sq += static_cast<double>(metric_delta) * static_cast<double>(metric_delta);
    entry.sum_control_metric += static_cast<double>(control_delta) * static_cast<double>(metric_delta);

    // Recompute correlation from accumulated statistics
    entry.correlation = compute_correlation(entry);

    // Update confidence (1 - 1/sqrt(n))
    // Approaches 1.0 as sample count increases
    // n=1: conf=0.0, n=4: conf=0.5, n=9: conf=0.67, n=100: conf=0.90
    entry.confidence = 1.0f - (1.0f / std::sqrt(static_cast<float>(entry.sample_count)));
}

std::vector<InfluencerRanking> SensitivityTracker::get_top_influencers(
    const std::string& metric_name,
    int top_k,
    float min_confidence
) const {
    std::vector<InfluencerRanking> candidates;
    candidates.reserve(sensitivity_map_.size());

    // Collect all controls that influence this metric with sufficient confidence
    for (const auto& kv : sensitivity_map_) {
        const auto& [control_metric_pair, entry] = kv;
        const auto& [control_name, m_name] = control_metric_pair;

        if (m_name == metric_name && entry.confidence >= min_confidence) {
            candidates.emplace_back(
                control_name,
                entry.influence_magnitude,
                entry.correlation,
                entry.confidence
            );
        }
    }

    // Sort by influence magnitude (descending)
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const InfluencerRanking& a, const InfluencerRanking& b) {
            return a.influence_magnitude > b.influence_magnitude;
        }
    );

    // Return top-K
    if (static_cast<int>(candidates.size()) > top_k) {
        candidates.resize(static_cast<size_t>(top_k));
    }

    return candidates;
}

bool SensitivityTracker::influences(
    const std::string& control_name,
    const std::string& metric_name,
    float min_magnitude,
    float min_confidence
) const {
    auto key = std::make_pair(control_name, metric_name);
    auto it = sensitivity_map_.find(key);

    if (it == sensitivity_map_.end()) {
        return false;
    }

    const SensitivityEntry& entry = it->second;

    return entry.influence_magnitude >= min_magnitude &&
           entry.confidence >= min_confidence;
}

bool SensitivityTracker::save(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[SensitivityTracker] Failed to open file for writing: " << path << std::endl;
        return false;
    }

    // Begin JSON object
    ofs << "{\n";
    ofs << "  \"sensitivity_map\": {\n";

    // Group entries by control name for nested structure
    std::unordered_map<std::string, std::vector<std::pair<std::string, const SensitivityEntry*>>> grouped;
    for (const auto& kv : sensitivity_map_) {
        const auto& [control_metric_pair, entry] = kv;
        const auto& [control_name, metric_name] = control_metric_pair;
        grouped[control_name].emplace_back(metric_name, &entry);
    }

    // Write grouped sensitivity data
    bool first_control = true;
    for (const auto& control_kv : grouped) {
        const auto& control_name = control_kv.first;
        const auto& metrics = control_kv.second;

        if (!first_control) {
            ofs << ",\n";
        }
        first_control = false;

        ofs << "    \"" << escape_json_string(control_name) << "\": {\n";

        bool first_metric = true;
        for (const auto& metric_entry : metrics) {
            const auto& metric_name = metric_entry.first;
            const SensitivityEntry* entry = metric_entry.second;

            if (!first_metric) {
                ofs << ",\n";
            }
            first_metric = false;

            ofs << "      \"" << escape_json_string(metric_name) << "\": {\n";
            ofs << "        \"influence_magnitude\": " << entry->influence_magnitude << ",\n";
            ofs << "        \"correlation\": " << entry->correlation << ",\n";
            ofs << "        \"sample_count\": " << entry->sample_count << ",\n";
            ofs << "        \"confidence\": " << entry->confidence << ",\n";
            ofs << "        \"last_update_batch\": " << entry->last_update_batch << "\n";
            ofs << "      }";
        }

        ofs << "\n    }";
    }

    ofs << "\n  },\n";

    // Write inferred parameter groups (Phase 2 prep)
    ofs << "  \"parameter_groups\": {\n";
    auto inferred_groups = infer_parameter_groups();
    bool first_group = true;
    for (const auto& group_kv : inferred_groups) {
        const auto& group_name = group_kv.first;
        const auto& control_names = group_kv.second;

        if (!first_group) {
            ofs << ",\n";
        }
        first_group = false;

        ofs << "    \"" << escape_json_string(group_name) << "\": [\n";
        for (size_t i = 0; i < control_names.size(); ++i) {
            ofs << "      \"" << escape_json_string(control_names[i]) << "\"";
            if (i < control_names.size() - 1) {
                ofs << ",";
            }
            ofs << "\n";
        }
        ofs << "    ]";
    }
    ofs << "\n  },\n";

    // Write metadata
    ofs << "  \"metadata\": {\n";
    ofs << "    \"last_batch\": " << current_batch_ << ",\n";
    ofs << "    \"total_entries\": " << sensitivity_map_.size() << ",\n";
    ofs << "    \"timestamp\": \"" << get_timestamp() << "\"\n";
    ofs << "  }\n";

    ofs << "}\n";

    ofs.close();

    std::cout << "[SensitivityTracker] Saved sensitivity map to: " << path
              << " (" << sensitivity_map_.size() << " entries, batch " << current_batch_ << ")"
              << std::endl;

    return true;
}

bool SensitivityTracker::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[SensitivityTracker] Failed to open file for reading: " << path << std::endl;
        return false;
    }

    // NOTE: Full JSON parsing is complex. For Phase 1, we implement a minimal
    // loader that supports the exact format we produce. For production, consider
    // using a JSON library (nlohmann/json, RapidJSON, etc.).
    //
    // Current implementation: Simple line-by-line parsing with state machine.
    // Limitations: Requires exact formatting (no minified JSON, strict whitespace).
    // Future: Replace with robust JSON library when adding Phase 2+ features.

    clear();  // Reset state

    std::string line;
    std::string current_control;
    std::string current_metric;
    SensitivityEntry current_entry;

    enum class ParseState {
        ROOT,
        SENSITIVITY_MAP,
        CONTROL_OBJECT,
        METRIC_OBJECT,
        METRIC_FIELDS,
        METADATA
    };

    ParseState state = ParseState::ROOT;

    while (std::getline(ifs, line)) {
        // Trim whitespace
        size_t start = line.find_first_not_of(" \t\r\n");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;  // Empty line
        line = line.substr(start, end - start + 1);

        // State machine transitions
        if (line.find("\"sensitivity_map\":") != std::string::npos) {
            state = ParseState::SENSITIVITY_MAP;
        } else if (state == ParseState::SENSITIVITY_MAP && line.find("\"") == 0 && line.find(": {") != std::string::npos) {
            // Extract control name: "control_name": {
            size_t quote_start = 1;
            size_t quote_end = line.find("\"", quote_start);
            if (quote_end != std::string::npos) {
                current_control = line.substr(quote_start, quote_end - quote_start);
                state = ParseState::CONTROL_OBJECT;
            }
        } else if (state == ParseState::CONTROL_OBJECT && line.find("\"") == 0 && line.find(": {") != std::string::npos) {
            // Extract metric name: "metric_name": {
            size_t quote_start = 1;
            size_t quote_end = line.find("\"", quote_start);
            if (quote_end != std::string::npos) {
                current_metric = line.substr(quote_start, quote_end - quote_start);
                current_entry = SensitivityEntry();  // Reset entry
                state = ParseState::METRIC_OBJECT;
            }
        } else if (state == ParseState::METRIC_OBJECT) {
            // Parse metric fields
            if (line.find("\"influence_magnitude\":") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    std::string value_str = line.substr(colon_pos + 1);
                    size_t comma_pos = value_str.find(",");
                    if (comma_pos != std::string::npos) {
                        value_str = value_str.substr(0, comma_pos);
                    }
                    current_entry.influence_magnitude = std::stof(value_str);
                }
            } else if (line.find("\"correlation\":") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    std::string value_str = line.substr(colon_pos + 1);
                    size_t comma_pos = value_str.find(",");
                    if (comma_pos != std::string::npos) {
                        value_str = value_str.substr(0, comma_pos);
                    }
                    current_entry.correlation = std::stof(value_str);
                }
            } else if (line.find("\"sample_count\":") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    std::string value_str = line.substr(colon_pos + 1);
                    size_t comma_pos = value_str.find(",");
                    if (comma_pos != std::string::npos) {
                        value_str = value_str.substr(0, comma_pos);
                    }
                    current_entry.sample_count = std::stoi(value_str);
                }
            } else if (line.find("\"confidence\":") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    std::string value_str = line.substr(colon_pos + 1);
                    size_t comma_pos = value_str.find(",");
                    if (comma_pos != std::string::npos) {
                        value_str = value_str.substr(0, comma_pos);
                    }
                    current_entry.confidence = std::stof(value_str);
                }
            } else if (line.find("\"last_update_batch\":") != std::string::npos) {
                size_t colon_pos = line.find(":");
                if (colon_pos != std::string::npos) {
                    std::string value_str = line.substr(colon_pos + 1);
                    size_t comma_pos = value_str.find(",");
                    if (comma_pos != std::string::npos) {
                        value_str = value_str.substr(0, comma_pos);
                    }
                    current_entry.last_update_batch = std::stoi(value_str);
                }
            } else if (line.find("}") == 0) {
                // End of metric object, save entry
                if (!current_control.empty() && !current_metric.empty()) {
                    auto key = std::make_pair(current_control, current_metric);
                    sensitivity_map_[key] = current_entry;
                }
                state = ParseState::CONTROL_OBJECT;
            }
        } else if (line.find("\"metadata\":") != std::string::npos) {
            state = ParseState::METADATA;
        } else if (state == ParseState::METADATA && line.find("\"last_batch\":") != std::string::npos) {
            size_t colon_pos = line.find(":");
            if (colon_pos != std::string::npos) {
                std::string value_str = line.substr(colon_pos + 1);
                size_t comma_pos = value_str.find(",");
                if (comma_pos != std::string::npos) {
                    value_str = value_str.substr(0, comma_pos);
                }
                current_batch_ = std::stoi(value_str);
            }
        }
    }

    ifs.close();

    std::cout << "[SensitivityTracker] Loaded sensitivity map from: " << path
              << " (" << sensitivity_map_.size() << " entries, batch " << current_batch_ << ")"
              << std::endl;

    return true;
}

void SensitivityTracker::apply_temporal_decay(float alpha) {
    if (alpha < 0.0f || alpha > 1.0f) {
        std::cerr << "[SensitivityTracker] Invalid decay factor: " << alpha
                  << ". Must be in [0, 1]. Skipping decay." << std::endl;
        return;
    }

    for (auto& kv : sensitivity_map_) {
        SensitivityEntry& entry = kv.second;

        // Decay sample count (effective history length)
        entry.sample_count = static_cast<int>(alpha * static_cast<float>(entry.sample_count));

        // Decay incremental statistics
        entry.sum_control_delta *= static_cast<double>(alpha);
        entry.sum_metric_delta *= static_cast<double>(alpha);
        entry.sum_control_sq *= static_cast<double>(alpha);
        entry.sum_metric_sq *= static_cast<double>(alpha);
        entry.sum_control_metric *= static_cast<double>(alpha);

        // Recompute confidence after decay
        if (entry.sample_count > 0) {
            entry.confidence = 1.0f - (1.0f / std::sqrt(static_cast<float>(entry.sample_count)));
        } else {
            entry.confidence = 0.0f;
        }

        // Recompute correlation from decayed statistics
        entry.correlation = compute_correlation(entry);
    }

    std::cout << "[SensitivityTracker] Applied temporal decay (alpha=" << alpha
              << ") to " << sensitivity_map_.size() << " entries." << std::endl;
}

void SensitivityTracker::clear() {
    sensitivity_map_.clear();
    parameter_groups_.clear();
    current_batch_ = 0;
}

void SensitivityTracker::register_parameter_group(const ParameterGroup& group) {
    parameter_groups_.push_back(group);
}

std::unordered_map<std::string, std::vector<std::string>> SensitivityTracker::infer_parameter_groups() const {
    std::unordered_map<std::string, std::vector<std::string>> groups;

    // Extract unique control names
    std::unordered_set<std::string> control_names;
    for (const auto& kv : sensitivity_map_) {
        control_names.insert(kv.first.first);
    }

    // Heuristic grouping based on name patterns
    for (const auto& control_name : control_names) {
        if (control_name.find("timecrystal_block_") != std::string::npos) {
            groups["timecrystal_blocks"].push_back(control_name);
        } else if (control_name.find("mamba2_block_") != std::string::npos) {
            groups["mamba2_blocks"].push_back(control_name);
        } else if (control_name.find("moe_block_") != std::string::npos) {
            groups["moe_blocks"].push_back(control_name);
        } else if (control_name.find("wlam_block_") != std::string::npos) {
            groups["wlam_blocks"].push_back(control_name);
        } else {
            // Fallback: group by first component (before first '/')
            size_t slash_pos = control_name.find('/');
            if (slash_pos != std::string::npos) {
                std::string prefix = control_name.substr(0, slash_pos);
                groups[prefix].push_back(control_name);
            } else {
                groups["other"].push_back(control_name);
            }
        }
    }

    // Sort control names within each group for deterministic output
    for (auto& group_kv : groups) {
        std::sort(group_kv.second.begin(), group_kv.second.end());
    }

    return groups;
}

float SensitivityTracker::compute_correlation(const SensitivityEntry& entry) {
    if (entry.sample_count < 2) {
        return 0.0f;  // Not enough samples
    }

    const double n = static_cast<double>(entry.sample_count);

    // Compute covariance: cov(X,Y) = E[XY] - E[X]E[Y]
    const double mean_control = entry.sum_control_delta / n;
    const double mean_metric = entry.sum_metric_delta / n;
    const double cov = (entry.sum_control_metric / n) - (mean_control * mean_metric);

    // Compute standard deviations: σ_X = sqrt(E[X²] - E[X]²)
    const double var_control = (entry.sum_control_sq / n) - (mean_control * mean_control);
    const double var_metric = (entry.sum_metric_sq / n) - (mean_metric * mean_metric);

    const double sigma_control = std::sqrt(std::max(0.0, var_control));
    const double sigma_metric = std::sqrt(std::max(0.0, var_metric));

    // Compute Pearson correlation: corr = cov / (σ_X × σ_Y)
    const double epsilon = 1e-12;
    if (sigma_control < epsilon || sigma_metric < epsilon) {
        return 0.0f;  // Degenerate case: no variation
    }

    double corr = cov / (sigma_control * sigma_metric);

    // Clamp to [-1, 1] to handle numerical errors
    corr = std::max(-1.0, std::min(1.0, corr));

    return static_cast<float>(corr);
}

std::string SensitivityTracker::escape_json_string(const std::string& str) {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b";  break;
            case '\f': oss << "\\f";  break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (c >= 0 && c < 32) {
                    // Control characters: use \uXXXX encoding
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

std::string SensitivityTracker::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm;
    localtime_r(&now_c, &now_tm);

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

}  // namespace control
}  // namespace saguaro
