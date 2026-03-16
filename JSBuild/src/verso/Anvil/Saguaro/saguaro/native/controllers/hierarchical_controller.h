#ifndef HIERARCHICAL_CONTROLLER_H_
#define HIERARCHICAL_CONTROLLER_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

#include "sensitivity_tracker.h"

namespace saguaro {
namespace control {

/**
 * @brief Tuning hierarchy levels for coarse-to-fine parameter optimization.
 *
 * Hierarchical controller operates in phases, starting from coarse global adjustments
 * and progressively refining to individual parameter tuning.
 *
 * Phase transitions:
 * GLOBAL (0-500 batches) → GROUP (500-1500 batches) → INDIVIDUAL (1500+ batches)
 *
 * Automatic phase detection based on:
 * - Loss convergence rate
 * - Control action stability
 * - Time spent at current level
 */
enum class TuningLevel {
    GLOBAL = 0,        ///< Uniform scale all evolution_times by factor α (coarse tuning)
    GROUP = 1,         ///< Adjust block type groups (TimeCrystal, Mamba2, MoE, WaveLetAttention)
    INDIVIDUAL = 2,    ///< Fine-tune single blocks (e.g., timecrystal_block_1/evolution_time)
    SUB_PARAMETER = 3  ///< Future: evolution_time_gain, evolution_time_shift, evolution_time_cap
};

/**
 * @brief Action types for hierarchical adjustments.
 */
enum class ActionType {
    SCALE,  ///< Multiplicative adjustment (evolution_time *= α)
    SHIFT,  ///< Additive adjustment (evolution_time += Δ)
    CLAMP   ///< Hard limit enforcement (evolution_time = clamp(value, min, max))
};

/**
 * @brief Hierarchical control action with justification.
 *
 * Represents a single control decision from the hierarchical controller,
 * including the target (global/group/individual), adjustment magnitude,
 * and human-readable reasoning based on sensitivity analysis.
 */
struct HierarchicalAction {
    TuningLevel level;          ///< Hierarchy level (GLOBAL, GROUP, INDIVIDUAL)
    std::string target;         ///< Target identifier ("global", "timecrystal_group", "timecrystal_block_1")
    float adjustment;           ///< Adjustment magnitude (multiplicative for SCALE, additive for SHIFT)
    ActionType type;            ///< Action type (SCALE, SHIFT, CLAMP)
    std::string reason;         ///< Human-readable justification (e.g., "Bottleneck metric: loss, top influencer: timecrystal_group")

    HierarchicalAction()
        : level(TuningLevel::GLOBAL)
        , adjustment(1.0f)
        , type(ActionType::SCALE) {}

    HierarchicalAction(
        TuningLevel lvl,
        const std::string& tgt,
        float adj,
        ActionType act_type,
        const std::string& rsn
    ) : level(lvl)
      , target(tgt)
      , adjustment(adj)
      , type(act_type)
      , reason(rsn) {}
};

/**
 * @brief System state snapshot for hierarchical decision-making.
 *
 * Encapsulates current training metrics and control input values
 * required for bottleneck detection and hierarchical action computation.
 */
struct SystemState {
    std::map<std::string, float> metrics;           ///< Current metric values (e.g., {"loss": 2.34, "gradient_norm": 0.56})
    std::map<std::string, float> evolution_times;   ///< Current evolution_time values (e.g., {"timecrystal_block_1": 0.001})
    int batch_number;                               ///< Current training batch number

    SystemState() : batch_number(0) {}
};

/**
 * @brief Hierarchical controller for intelligent, causally-aware parameter tuning.
 *
 * Part of Phase 2 of Intelligent Meta-Controller roadmap.
 * Implements coarse-to-fine optimization strategy:
 *
 * 1. **Phase 1 (GLOBAL):** Uniform scaling when loss far from target (first 500 batches)
 * 2. **Phase 2 (GROUP):** Group-level adjustments targeting bottleneck block types (batches 500-1500)
 * 3. **Phase 3 (INDIVIDUAL):** Fine-tuning individual blocks (batches 1500+)
 *
 * Key features:
 * - Bottleneck detection via sensitivity analysis (top-K influencers)
 * - SIMD-optimized group adjustment computation (AVX512/AVX2/NEON)
 * - Automatic phase transitions based on convergence heuristics
 * - Explainable control decisions (human-readable justifications)
 *
 * Integration with SensitivityTracker:
 * - Queries `get_top_influencers(metric)` to identify causal parameters
 * - Respects influence magnitude and correlation for adjustment computation
 * - Supports grouped parameter queries (e.g., "Which block type group affects loss most?")
 *
 * Thread safety: Not thread-safe. Caller must serialize calls to decide_action().
 *
 * Performance:
 * - decide_action(): O(n log k) where n = controls, k = top_k (dominated by sensitivity ranking)
 * - apply_hierarchical_action(): O(m) where m = affected controls (SIMD-accelerated for GROUP level)
 * - SIMD speedup: 3-4x (AVX512) for group-level adjustment computation
 */
class HierarchicalController {
public:
    /**
     * @brief Construct hierarchical controller with default thresholds.
     */
    HierarchicalController();

    ~HierarchicalController() = default;

    /**
     * @brief Decide next hierarchical action based on system state and sensitivity.
     *
     * Executes hierarchical decision logic:
     * 1. Detect bottleneck metric (largest deviation from setpoint)
     * 2. Query sensitivity tracker for top influencers of bottleneck metric
     * 3. Select tuning level (GLOBAL, GROUP, INDIVIDUAL) based on current phase
     * 4. Compute adjustment magnitude based on error magnitude and sensitivity
     * 5. Return HierarchicalAction with justification
     *
     * Phase transition rules:
     * - GLOBAL → GROUP: After 500 batches AND loss deviation < 1.0
     * - GROUP → INDIVIDUAL: After 1000 batches at GROUP AND adjustment < 0.05
     * - INDIVIDUAL → GLOBAL: If loss spikes > 2.0 (reset to coarse tuning)
     *
     * @param state Current system state (metrics, evolution_times, batch_number)
     * @param sensitivity Sensitivity tracker with historical influence data
     * @return HierarchicalAction describing control decision and reasoning
     *
     * Example:
     * ```cpp
     * SystemState state;
     * state.metrics = {{"loss", 2.34}, {"gradient_norm", 0.56}};
     * state.evolution_times = {{"timecrystal_block_1", 0.001}, {"mamba2_block_1", 0.002}};
     * state.batch_number = 1200;
     *
     * HierarchicalAction action = controller->decide_action(state, *sensitivity_tracker);
     * // action.level = TuningLevel::GROUP
     * // action.target = "timecrystal_group"
     * // action.adjustment = 0.95f (scale down by 5%)
     * // action.reason = "Bottleneck: loss (dev=0.34), top group: timecrystal (inf_mag=0.023)"
     * ```
     */
    HierarchicalAction decide_action(
        const SystemState& state,
        const SensitivityTracker& sensitivity
    );

    /**
     * @brief Apply hierarchical action to evolution_times map.
     *
     * Modifies evolution_times in-place according to action specification:
     * - GLOBAL SCALE: Multiply all evolution_times by adjustment factor
     * - GROUP SCALE/SHIFT: Apply adjustment to all controls in target group
     * - INDIVIDUAL SCALE/SHIFT: Apply adjustment to single target control
     *
     * SIMD optimization:
     * - GROUP-level adjustments use AVX512/AVX2/NEON for batch processing
     * - 3-4x speedup for groups with 10+ parameters (typical: timecrystal_group has 8-12 blocks)
     *
     * @param action Hierarchical action from decide_action()
     * @param evolution_times Map of control_name → value (modified in-place)
     *
     * Example:
     * ```cpp
     * std::map<std::string, float> evo_times = {
     *     {"timecrystal_block_1", 0.001f},
     *     {"timecrystal_block_2", 0.0015f},
     *     {"mamba2_block_1", 0.002f}
     * };
     *
     * HierarchicalAction action(TuningLevel::GROUP, "timecrystal_group", 0.95f, ActionType::SCALE, "...");
     * controller->apply_hierarchical_action(action, evo_times);
     * // Result: timecrystal_block_1 = 0.00095, timecrystal_block_2 = 0.001425, mamba2_block_1 = 0.002 (unchanged)
     * ```
     */
    void apply_hierarchical_action(
        const HierarchicalAction& action,
        std::map<std::string, float>& evolution_times
    );

    /**
     * @brief Get current tuning level.
     *
     * @return Current TuningLevel (GLOBAL, GROUP, INDIVIDUAL, SUB_PARAMETER)
     */
    TuningLevel get_current_level() const { return current_level_; }

    /**
     * @brief Get number of batches spent at current level.
     *
     * @return Batches at current level (resets to 0 on phase transition)
     */
    int get_batches_at_current_level() const { return batches_at_current_level_; }

    /**
     * @brief Force phase transition to specified level.
     *
     * Useful for manual intervention or testing.
     *
     * @param level Target tuning level
     */
    void set_tuning_level(TuningLevel level);

    /**
     * @brief Reset hierarchical controller to initial state.
     *
     * Resets to GLOBAL level with batch counter = 0.
     * Clears adjustment history.
     */
    void reset();

private:
    // Current state
    TuningLevel current_level_;           ///< Current tuning level
    int batches_at_current_level_;        ///< Batches spent at current level
    int total_batches_;                   ///< Total batches since initialization

    // Phase transition thresholds
    float global_to_group_loss_threshold_;     ///< Max loss deviation to transition GLOBAL → GROUP
    int global_to_group_min_batches_;          ///< Min batches before transitioning GLOBAL → GROUP
    int group_to_individual_min_batches_;      ///< Min batches before transitioning GROUP → INDIVIDUAL
    float group_to_individual_adj_threshold_;  ///< Max adjustment magnitude to transition GROUP → INDIVIDUAL
    float individual_to_global_loss_spike_;    ///< Loss spike threshold to reset INDIVIDUAL → GLOBAL

    // Adjustment history (for phase detection)
    std::vector<float> recent_adjustments_;   ///< Last 10 adjustment magnitudes
    float adjustment_stability_window_;       ///< Window size for adjustment stability check

    // Setpoints (for bottleneck detection)
    std::map<std::string, float> metric_setpoints_;  ///< Target values per metric (e.g., {"loss": 2.5})

    /**
     * @brief Detect bottleneck metric (largest deviation from setpoint).
     *
     * Computes normalized deviation for each metric:
     * deviation = |current_value - setpoint| / max(|setpoint|, 1.0)
     *
     * Returns metric name with largest deviation.
     *
     * @param state System state with current metrics
     * @return Bottleneck metric name (e.g., "loss", "gradient_norm")
     */
    std::string find_bottleneck_metric(const SystemState& state) const;

    /**
     * @brief Compute global-level adjustment (uniform scaling).
     *
     * Simple proportional control: adjustment = 1.0 + K_p * error
     * where error = (target_loss - current_loss) / target_loss
     *
     * @param state System state
     * @return Scale factor (e.g., 0.9 to reduce all evolution_times by 10%)
     */
    float compute_global_adjustment(const SystemState& state) const;

    /**
     * @brief Compute group-level adjustment (target specific block type group).
     *
     * Uses sensitivity-weighted proportional control:
     * adjustment = 1.0 + K_p * error * (influence_magnitude / max_influence)
     *
     * @param state System state
     * @param group_name Target group (e.g., "timecrystal_group")
     * @param sensitivity Sensitivity tracker
     * @return Scale factor for group (e.g., 0.95 to reduce group evolution_times by 5%)
     */
    float compute_group_adjustment(
        const SystemState& state,
        const std::string& group_name,
        const SensitivityTracker& sensitivity
    ) const;

    /**
     * @brief Compute individual-level adjustment (target single block).
     *
     * Uses sensitivity-weighted PD control:
     * adjustment = K_p * error * influence + K_d * error_rate
     *
     * @param state System state
     * @param control_name Target control (e.g., "timecrystal_block_1/evolution_time")
     * @param sensitivity Sensitivity tracker
     * @return Adjustment delta (additive, e.g., +0.0001 to increase evolution_time)
     */
    float compute_individual_adjustment(
        const SystemState& state,
        const std::string& control_name,
        const SensitivityTracker& sensitivity
    ) const;

    /**
     * @brief Select causal controls for bottleneck metric (Phase 3).
     *
     * Queries SensitivityTracker for top-K most influential controls that causally
     * affect the identified bottleneck metric. Filters by confidence and influence magnitude.
     *
     * Phase 3 causal selection criteria:
     * - INDIVIDUAL level: top_k=3, min_confidence=0.7, min_magnitude=0.1
     * - GROUP level: top_k=10, aggregate by inferred group, select top group
     * - GLOBAL level: return ["global"] (no filtering)
     *
     * @param bottleneck_metric Name of bottleneck metric (from find_bottleneck_metric)
     * @param sensitivity Sensitivity tracker with historical influence data
     * @param level Current tuning level (GLOBAL, GROUP, INDIVIDUAL)
     * @return Vector of causal control names (or group names for GROUP level)
     *
     * Example:
     * ```cpp
     * std::string bottleneck = "loss";
     * auto causal_controls = select_causal_controls(bottleneck, sensitivity, TuningLevel::INDIVIDUAL);
     * // Returns: ["timecrystal_block_1/evolution_time", "moe_block_2/evolution_time", "mamba2_block_1/evolution_time"]
     * // (top-3 controls that causally affect loss)
     * ```
     */
    std::vector<std::string> select_causal_controls(
        const std::string& bottleneck_metric,
        const SensitivityTracker& sensitivity,
        TuningLevel level
    ) const;

    /**
     * @brief Check if controller should refine to next level.
     *
     * Phase transition conditions:
     * - GLOBAL → GROUP: batches >= 500 AND loss deviation < 1.0
     * - GROUP → INDIVIDUAL: batches >= 1000 AND recent adjustments stable (stdev < 0.05)
     *
     * @return true if phase transition should occur, false otherwise
     */
    bool should_refine_level() const;

    /**
     * @brief Check if controller should reset to coarse tuning (GLOBAL).
     *
     * Triggered by large loss spikes indicating instability.
     *
     * @param state System state
     * @return true if loss spike detected (reset to GLOBAL), false otherwise
     */
    bool should_reset_to_coarse(const SystemState& state) const;

    /**
     * @brief Update adjustment history for phase detection.
     *
     * Maintains sliding window of last 10 adjustments.
     *
     * @param adjustment Recent adjustment magnitude
     */
    void update_adjustment_history(float adjustment);

    /**
     * @brief Compute standard deviation of recent adjustments.
     *
     * Used for phase transition detection (GROUP → INDIVIDUAL).
     *
     * @return Standard deviation of recent_adjustments_
     */
    float compute_adjustment_stability() const;

    /**
     * @brief Infer parameter group from control name.
     *
     * Heuristics:
     * - "timecrystal_block_*" → "timecrystal_group"
     * - "mamba2_block_*" → "mamba2_group"
     * - "moe_block_*" → "moe_group"
     * - "wlam_block_*" → "wlam_group"
     * - Other → "other_group"
     *
     * @param control_name Control input name
     * @return Group name
     */
    static std::string infer_group_name(const std::string& control_name);

    /**
     * @brief Get all control names belonging to a group.
     *
     * Scans evolution_times map for controls matching group pattern.
     *
     * @param group_name Target group (e.g., "timecrystal_group")
     * @param evolution_times Current evolution_times map
     * @return Vector of control names in group
     */
    static std::vector<std::string> get_group_members(
        const std::string& group_name,
        const std::map<std::string, float>& evolution_times
    );
};

}  // namespace control
}  // namespace saguaro

#endif  // HIERARCHICAL_CONTROLLER_H_
