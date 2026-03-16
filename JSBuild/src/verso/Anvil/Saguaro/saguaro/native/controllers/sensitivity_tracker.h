#ifndef SENSITIVITY_TRACKER_H_
#define SENSITIVITY_TRACKER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <utility>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace saguaro {
namespace control {

/**
 * @brief Tracks influence of control inputs on metrics over time.
 *
 * Maintains rolling statistics for each (control_input, metric) pair:
 * - Influence magnitude: |Δmetric| / |Δcontrol|
 * - Correlation: Pearson correlation coefficient
 * - Sample count: Number of observations
 * - Confidence: 1.0 - (1.0 / sqrt(sample_count))
 *
 * Designed for integration with HamiltonianMetaController to enable
 * causal parameter selection and hierarchical HPO transfer learning.
 *
 * Thread safety: Not thread-safe. Caller must ensure single-threaded access.
 */
struct SensitivityEntry {
    float influence_magnitude;   // |Δmetric| / |Δcontrol| (moving average)
    float correlation;            // Pearson correlation (moving average)
    int sample_count;             // Number of observations
    float confidence;             // 1.0 - (1.0 / sqrt(sample_count))
    int last_update_batch;        // Batch number of last update

    // Incremental statistics for correlation computation
    double sum_control_delta;     // Σ(Δcontrol)
    double sum_metric_delta;      // Σ(Δmetric)
    double sum_control_sq;        // Σ(Δcontrol²)
    double sum_metric_sq;         // Σ(Δmetric²)
    double sum_control_metric;    // Σ(Δcontrol × Δmetric)

    SensitivityEntry()
        : influence_magnitude(0.0f)
        , correlation(0.0f)
        , sample_count(0)
        , confidence(0.0f)
        , last_update_batch(0)
        , sum_control_delta(0.0)
        , sum_metric_delta(0.0)
        , sum_control_sq(0.0)
        , sum_metric_sq(0.0)
        , sum_control_metric(0.0) {}
};

/**
 * @brief Influencer ranking result for causal parameter selection.
 */
struct InfluencerRanking {
    std::string control_name;
    float influence_magnitude;
    float correlation;
    float confidence;

    InfluencerRanking() = default;

    InfluencerRanking(
        const std::string& name,
        float mag,
        float corr,
        float conf
    ) : control_name(name)
      , influence_magnitude(mag)
      , correlation(corr)
      , confidence(conf) {}
};

/**
 * @brief Hash function for std::pair<std::string, std::string>.
 */
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const {
        auto h1 = std::hash<std::string>{}(p.first);
        auto h2 = std::hash<std::string>{}(p.second);
        // Boost-style hash combine
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

/**
 * @brief Parameter group definition for hierarchical tuning.
 *
 * Groups related control inputs (e.g., all timecrystal_block_N/evolution_time).
 * Used by Phase 2 hierarchical controller.
 */
struct ParameterGroup {
    std::string group_name;
    std::vector<std::string> control_names;

    ParameterGroup() = default;
    ParameterGroup(const std::string& name, const std::vector<std::string>& controls)
        : group_name(name), control_names(controls) {}
};

/**
 * @brief Sensitivity tracking system for causal parameter optimization.
 *
 * Core component for Phase 1 of Intelligent Meta-Controller roadmap.
 * Provides:
 * 1. Incremental sensitivity tracking (Welford's algorithm for correlation)
 * 2. Top-K influencer queries (for causal parameter selection)
 * 3. JSON serialization (for HPO transfer learning)
 * 4. Temporal decay (exponential forgetting of old data)
 *
 * Integration with HamiltonianMetaController:
 * - Call record_update() in update_and_act() after computing control actions
 * - Call save() every 500 batches to export sensitivity maps
 * - HPO system reads sensitivity_map.json for grouped parameter search
 *
 * Performance:
 * - record_update(): O(1) per call (incremental statistics)
 * - get_top_influencers(): O(n log k) where n = control_inputs, k = top_k
 * - save()/load(): O(m × n) where m = metrics, n = controls (JSON I/O)
 *
 * Memory:
 * - O(metrics × controls) entries in sensitivity_map_
 * - Typically ~100 metrics × ~50 controls = 5000 entries (240 KB)
 */
class SensitivityTracker {
public:
    SensitivityTracker();
    ~SensitivityTracker() = default;

    /**
     * @brief Record influence of control input on metric.
     *
     * Updates incremental statistics for (control_name, metric_name) pair:
     * - Influence magnitude (moving average of |Δmetric| / |Δcontrol|)
     * - Correlation (Pearson coefficient via Welford's algorithm)
     * - Sample count (incremented)
     * - Confidence (1 - 1/sqrt(n))
     *
     * @param control_name Name of control input (e.g., "timecrystal_block_1/evolution_time")
     * @param metric_name Name of metric (e.g., "loss", "gradient_norm")
     * @param control_delta Change in control input (Δu)
     * @param metric_delta Change in metric (Δy)
     * @param batch_number Current batch number (for last_update tracking)
     *
     * Thread safety: Not thread-safe. Caller must serialize calls.
     *
     * Example:
     * ```cpp
     * tracker->record_update(
     *     "timecrystal_block_1/evolution_time",
     *     "loss",
     *     0.05f,   // control increased by 0.05
     *     -0.12f,  // loss decreased by 0.12 (good correlation)
     *     batch_counter_
     * );
     * ```
     */
    void record_update(
        const std::string& control_name,
        const std::string& metric_name,
        float control_delta,
        float metric_delta,
        int batch_number
    );

    /**
     * @brief Get top-K most influential controls for a metric.
     *
     * Ranks control inputs by influence magnitude, filtered by minimum confidence.
     * Used for causal parameter selection in hierarchical controller.
     *
     * @param metric_name Target metric (e.g., "loss")
     * @param top_k Number of top influencers to return (default: 3)
     * @param min_confidence Minimum confidence threshold (default: 0.7, requires ~9 samples)
     * @return Vector of InfluencerRanking sorted by influence_magnitude (descending)
     *
     * Example:
     * ```cpp
     * auto influencers = tracker->get_top_influencers("loss", 3, 0.7f);
     * for (const auto& inf : influencers) {
     *     std::cout << inf.control_name << ": mag=" << inf.influence_magnitude
     *               << ", corr=" << inf.correlation << ", conf=" << inf.confidence << "\n";
     * }
     * ```
     */
    std::vector<InfluencerRanking> get_top_influencers(
        const std::string& metric_name,
        int top_k = 3,
        float min_confidence = 0.7f
    ) const;

    /**
     * @brief Check if control input significantly influences metric.
     *
     * @param control_name Control input name
     * @param metric_name Metric name
     * @param min_magnitude Minimum influence magnitude threshold (default: 0.1)
     * @param min_confidence Minimum confidence threshold (default: 0.5)
     * @return true if influence exists and exceeds thresholds, false otherwise
     */
    bool influences(
        const std::string& control_name,
        const std::string& metric_name,
        float min_magnitude = 0.1f,
        float min_confidence = 0.5f
    ) const;

    /**
     * @brief Serialize sensitivity map to JSON file.
     *
     * Exports:
     * 1. sensitivity_map: nested dict {control_name: {metric_name: {stats}}}
     * 2. parameter_groups: inferred from control name patterns (Phase 2 prep)
     * 3. metadata: batch_number, timestamp
     *
     * Format:
     * ```json
     * {
     *   "sensitivity_map": {
     *     "timecrystal_block_1/evolution_time": {
     *       "loss": {
     *         "influence_magnitude": 0.023,
     *         "correlation": -0.65,
     *         "sample_count": 1200,
     *         "confidence": 0.97,
     *         "last_update_batch": 12450
     *       }
     *     }
     *   },
     *   "parameter_groups": {
     *     "timecrystal_blocks": ["timecrystal_block_1/evolution_time", ...]
     *   },
     *   "metadata": {
     *     "last_batch": 12450,
     *     "total_entries": 5000,
     *     "timestamp": "2025-11-24T12:34:56"
     *   }
     * }
     * ```
     *
     * @param path Output file path (e.g., "artifacts/trial_0001/sensitivity_map.json")
     * @return true on success, false on I/O error
     */
    bool save(const std::string& path) const;

    /**
     * @brief Load sensitivity map from JSON file.
     *
     * Replaces current sensitivity_map_ with deserialized data.
     * Used for transfer learning across trials.
     *
     * @param path Input file path
     * @return true on success, false on I/O or parse error
     */
    bool load(const std::string& path);

    /**
     * @brief Apply temporal decay to sensitivity entries.
     *
     * Exponentially decays sample counts and statistics to forget old data.
     * Useful for non-stationary systems where parameter influences change over time.
     *
     * @param alpha Decay factor (default: 0.99, gentle decay)
     *              alpha=1.0: no decay (infinite memory)
     *              alpha=0.95: ~20 batch half-life
     *              alpha=0.99: ~100 batch half-life
     *
     * Call periodically (e.g., every 1000 batches) to prevent stale data accumulation.
     */
    void apply_temporal_decay(float alpha = 0.99f);

    /**
     * @brief Clear all sensitivity data.
     *
     * Resets sensitivity_map_ to empty state. Used for fresh starts or debugging.
     */
    void clear();

    /**
     * @brief Get total number of tracked (control, metric) pairs.
     *
     * @return Number of entries in sensitivity_map_
     */
    size_t size() const { return sensitivity_map_.size(); }

    /**
     * @brief Register parameter group for hierarchical tuning (Phase 2).
     *
     * Groups related control inputs (e.g., all timecrystal blocks).
     * Exported in JSON for HPO grouped parameter search.
     *
     * @param group Parameter group definition
     */
    void register_parameter_group(const ParameterGroup& group);

    /**
     * @brief Get all registered parameter groups.
     *
     * @return Vector of parameter groups
     */
    const std::vector<ParameterGroup>& get_parameter_groups() const {
        return parameter_groups_;
    }

private:
    // Map: (control_name, metric_name) → SensitivityEntry
    std::unordered_map<
        std::pair<std::string, std::string>,
        SensitivityEntry,
        PairHash
    > sensitivity_map_;

    // Parameter groups for hierarchical tuning (Phase 2)
    std::vector<ParameterGroup> parameter_groups_;

    // Batch number tracking
    int current_batch_;

    /**
     * @brief Infer parameter groups from control names.
     *
     * Heuristics:
     * - "timecrystal_block_*" → "timecrystal_blocks" group
     * - "mamba2_block_*" → "mamba2_blocks" group
     * - "moe_block_*" → "moe_blocks" group
     * - "wlam_block_*" → "wlam_blocks" group
     *
     * Called during save() to auto-generate groups for HPO.
     */
    std::unordered_map<std::string, std::vector<std::string>> infer_parameter_groups() const;

    /**
     * @brief Compute Pearson correlation coefficient from incremental statistics.
     *
     * Formula: corr = cov(X,Y) / (σ_X × σ_Y)
     * where cov(X,Y) = E[XY] - E[X]E[Y]
     *
     * @param entry Sensitivity entry with incremental stats
     * @return Correlation coefficient in [-1, 1] or 0.0 if invalid
     */
    static float compute_correlation(const SensitivityEntry& entry);

    /**
     * @brief Escape string for JSON (quote and backslash escaping).
     *
     * @param str Input string
     * @return JSON-safe escaped string
     */
    static std::string escape_json_string(const std::string& str);

    /**
     * @brief Get current timestamp in ISO 8601 format.
     *
     * @return Timestamp string (e.g., "2025-11-24T12:34:56")
     */
    static std::string get_timestamp();
};

}  // namespace control
}  // namespace saguaro

#endif  // SENSITIVITY_TRACKER_H_
