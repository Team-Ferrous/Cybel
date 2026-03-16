#include "hierarchical_controller.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>

// Phase 11 SIMD compliance: Hierarchical SIMD guard pattern
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__AVX__)
  #include <immintrin.h>  // x86 SIMD intrinsics (AVX512/AVX2/AVX)
#elif defined(__ARM_NEON)
  #include <arm_neon.h>   // ARM NEON intrinsics
#endif

#include "common/parallel/parallel_backend.h"

namespace saguaro {
namespace control {

// =============================================================================
// Constructors and Initialization
// =============================================================================

HierarchicalController::HierarchicalController()
    : current_level_(TuningLevel::GLOBAL)
    , batches_at_current_level_(0)
    , total_batches_(0)
    , global_to_group_loss_threshold_(1.0f)
    , global_to_group_min_batches_(500)
    , group_to_individual_min_batches_(1000)
    , group_to_individual_adj_threshold_(0.05f)
    , individual_to_global_loss_spike_(2.0f)
    , adjustment_stability_window_(10.0f) {

    // Default setpoints
    metric_setpoints_["loss"] = 2.5f;
    metric_setpoints_["gradient_norm"] = 1.0f;
    metric_setpoints_["energy_drift"] = 0.0f;

    recent_adjustments_.reserve(static_cast<size_t>(adjustment_stability_window_));
}

void HierarchicalController::reset() {
    current_level_ = TuningLevel::GLOBAL;
    batches_at_current_level_ = 0;
    total_batches_ = 0;
    recent_adjustments_.clear();
}

void HierarchicalController::set_tuning_level(TuningLevel level) {
    if (level != current_level_) {
        std::cout << "[HierarchicalController] Manual phase transition: "
                  << static_cast<int>(current_level_) << " → "
                  << static_cast<int>(level) << std::endl;
        current_level_ = level;
        batches_at_current_level_ = 0;
    }
}

// =============================================================================
// Main Decision Logic
// =============================================================================

HierarchicalAction HierarchicalController::decide_action(
    const SystemState& state,
    const SensitivityTracker& sensitivity
) {
    // Update batch counters
    batches_at_current_level_++;
    total_batches_++;

    // Check for reset to coarse tuning (loss spike)
    if (current_level_ != TuningLevel::GLOBAL && should_reset_to_coarse(state)) {
        std::cout << "[HierarchicalController] Loss spike detected. Resetting to GLOBAL tuning."
                  << std::endl;
        current_level_ = TuningLevel::GLOBAL;
        batches_at_current_level_ = 0;
        recent_adjustments_.clear();
    }

    // Phase 3 Step 1: Detect bottleneck metric (SIMD-optimized)
    std::string bottleneck_metric = find_bottleneck_metric(state);

    // Phase 3 Step 2: Select causal controls for bottleneck metric
    std::vector<std::string> causal_controls = select_causal_controls(
        bottleneck_metric,
        sensitivity,
        current_level_
    );

    // If no causal controls identified, skip action (Phase 3 enhancement)
    if (causal_controls.empty()) {
        std::cout << "[HierarchicalController] No causal controls identified for bottleneck: "
                  << bottleneck_metric << ", skipping action." << std::endl;
        // Return no-op action that respects current tuning level
        return HierarchicalAction(
            current_level_,
            "",  // No target
            (current_level_ == TuningLevel::INDIVIDUAL) ? 0.0f : 1.0f,  // 0 for SHIFT, 1.0 for SCALE
            (current_level_ == TuningLevel::INDIVIDUAL) ? ActionType::SHIFT : ActionType::SCALE,
            "No causal controls identified for bottleneck: " + bottleneck_metric
        );
    }

    // =========================================================================
    // Phase 1: GLOBAL (Uniform Scaling)
    // =========================================================================
    if (current_level_ == TuningLevel::GLOBAL) {
        // causal_controls[0] == "global"
        float global_adj = compute_global_adjustment(state);
        update_adjustment_history(std::abs(global_adj - 1.0f));

        // Check for phase transition: GLOBAL → GROUP
        if (should_refine_level()) {
            std::cout << "[HierarchicalController] Phase transition: GLOBAL → GROUP (batches: "
                      << batches_at_current_level_ << ")" << std::endl;
            current_level_ = TuningLevel::GROUP;
            batches_at_current_level_ = 0;
        }

        std::ostringstream reason;
        reason << "GLOBAL tuning: bottleneck=" << bottleneck_metric
               << ", loss_dev=" << (state.metrics.count("loss") ? std::abs(state.metrics.at("loss") - metric_setpoints_["loss"]) : 0.0f)
               << ", batch=" << total_batches_;

        return HierarchicalAction(
            TuningLevel::GLOBAL,
            "global",
            global_adj,
            ActionType::SCALE,
            reason.str()
        );
    }

    // =========================================================================
    // Phase 2: GROUP (Block Type Groups) - Phase 3 Enhanced
    // =========================================================================
    if (current_level_ == TuningLevel::GROUP) {
        // Phase 3: Use causal_controls[0] as target group
        std::string target_group = causal_controls[0];

        // Fallback to GLOBAL if target is "global" (no sensitivity data)
        if (target_group == "global") {
            float global_adj = compute_global_adjustment(state);
            update_adjustment_history(std::abs(global_adj - 1.0f));

            return HierarchicalAction(
                TuningLevel::GLOBAL,
                "global",
                global_adj,
                ActionType::SCALE,
                "GROUP fallback (no sensitivity data): using GLOBAL"
            );
        }

        // Compute group adjustment
        float group_adj = compute_group_adjustment(state, target_group, sensitivity);
        update_adjustment_history(std::abs(group_adj - 1.0f));

        // Check for phase transition: GROUP → INDIVIDUAL
        if (should_refine_level()) {
            std::cout << "[HierarchicalController] Phase transition: GROUP → INDIVIDUAL (batches: "
                      << batches_at_current_level_ << ", adj_stability: "
                      << compute_adjustment_stability() << ")" << std::endl;
            current_level_ = TuningLevel::INDIVIDUAL;
            batches_at_current_level_ = 0;
        }

        std::ostringstream reason;
        reason << "GROUP tuning: bottleneck=" << bottleneck_metric
               << ", causal_group=" << target_group
               << ", batch=" << total_batches_;

        return HierarchicalAction(
            TuningLevel::GROUP,
            target_group,
            group_adj,
            ActionType::SCALE,
            reason.str()
        );
    }

    // =========================================================================
    // Phase 3: INDIVIDUAL (Fine-Tuning) - Phase 3 Enhanced
    // =========================================================================
    if (current_level_ == TuningLevel::INDIVIDUAL) {
        // Phase 3: Use causal_controls[0] as target control (top influencer)
        if (causal_controls.empty()) {
            // No high-confidence causal influencers, remain stable (no action)
            return HierarchicalAction(
                TuningLevel::INDIVIDUAL,
                "",
                0.0f,
                ActionType::SHIFT,
                "INDIVIDUAL: no causal influencers, holding steady"
            );
        }

        std::string target_control = causal_controls[0];

        // Compute individual adjustment
        float individual_adj = compute_individual_adjustment(state, target_control, sensitivity);
        update_adjustment_history(std::abs(individual_adj));

        // Check for insignificant adjustment (no action needed)
        if (std::abs(individual_adj) < 0.01f) {
            return HierarchicalAction(
                TuningLevel::INDIVIDUAL,
                target_control,
                0.0f,
                ActionType::SHIFT,
                "INDIVIDUAL: adjustment insignificant (< 0.01), holding steady"
            );
        }

        std::ostringstream reason;
        reason << "INDIVIDUAL tuning: bottleneck=" << bottleneck_metric
               << ", causal_control=" << target_control
               << ", batch=" << total_batches_;

        return HierarchicalAction(
            TuningLevel::INDIVIDUAL,
            target_control,
            individual_adj,
            ActionType::SHIFT,
            reason.str()
        );
    }

    // Fallback (should not reach here)
    return HierarchicalAction(
        TuningLevel::GLOBAL,
        "global",
        1.0f,
        ActionType::SCALE,
        "Fallback: no action"
    );
}

// =============================================================================
// Hierarchical Action Application (SIMD-Optimized for GROUP level)
// =============================================================================

void HierarchicalController::apply_hierarchical_action(
    const HierarchicalAction& action,
    std::map<std::string, float>& evolution_times
) {
    if (action.target.empty() || std::abs(action.adjustment) < 1e-9f) {
        return;  // No-op
    }

    // =========================================================================
    // GLOBAL: Uniform scaling (SIMD-optimized batch processing)
    // =========================================================================
    if (action.level == TuningLevel::GLOBAL && action.type == ActionType::SCALE) {
        // Extract values to contiguous vector for SIMD processing
        std::vector<float> times;
        times.reserve(evolution_times.size());
        for (const auto& [name, time] : evolution_times) {
            times.push_back(time);
        }

        const size_t n = times.size();
        const float scale = action.adjustment;
        size_t i = 0;

        // SIMD processing with hierarchical tier detection
#if defined(__AVX512F__)
        // AVX512: 16-wide SIMD (512-bit registers)
        __m512 vscale = _mm512_set1_ps(scale);
        for (; i + 16 <= n; i += 16) {
            __m512 vtimes = _mm512_loadu_ps(&times[i]);
            __m512 vscaled = _mm512_mul_ps(vtimes, vscale);
            _mm512_storeu_ps(&times[i], vscaled);
        }
#elif defined(__AVX2__)
        // AVX2: 8-wide SIMD (256-bit registers)
        __m256 vscale = _mm256_set1_ps(scale);
        for (; i + 8 <= n; i += 8) {
            __m256 vtimes = _mm256_loadu_ps(&times[i]);
            __m256 vscaled = _mm256_mul_ps(vtimes, vscale);
            _mm256_storeu_ps(&times[i], vscaled);
        }
#elif defined(__ARM_NEON)
        // NEON: 4-wide SIMD (128-bit registers)
        float32x4_t vscale = vdupq_n_f32(scale);
        for (; i + 4 <= n; i += 4) {
            float32x4_t vtimes = vld1q_f32(&times[i]);
            float32x4_t vscaled = vmulq_f32(vtimes, vscale);
            vst1q_f32(&times[i], vscaled);
        }
#endif

        // Scalar fallback for remaining elements
        for (; i < n; ++i) {
            times[i] *= scale;
        }

        // Write back to map
        auto it = evolution_times.begin();
        for (size_t j = 0; j < times.size(); ++j, ++it) {
            it->second = times[j];
        }

        std::cout << "[HierarchicalController] Applied GLOBAL SCALE: "
                  << scale << " to " << evolution_times.size() << " controls" << std::endl;
        return;
    }

    // =========================================================================
    // GROUP: Group-level adjustment (SIMD-optimized)
    // =========================================================================
    if (action.level == TuningLevel::GROUP && action.type == ActionType::SCALE) {
        // Get group members
        std::vector<std::string> group_members = get_group_members(action.target, evolution_times);

        if (group_members.empty()) {
            std::cerr << "[HierarchicalController] Warning: group '" << action.target
                      << "' has no members. No action applied." << std::endl;
            return;
        }

        // Extract group values to contiguous vector for SIMD processing
        std::vector<float> group_times;
        group_times.reserve(group_members.size());
        for (const auto& member : group_members) {
            group_times.push_back(evolution_times[member]);
        }

        const size_t n = group_times.size();
        const float scale = action.adjustment;
        size_t i = 0;

        // SIMD processing with hierarchical tier detection
#if defined(__AVX512F__)
        // AVX512: 16-wide SIMD
        __m512 vscale = _mm512_set1_ps(scale);
        for (; i + 16 <= n; i += 16) {
            __m512 vtimes = _mm512_loadu_ps(&group_times[i]);
            __m512 vscaled = _mm512_mul_ps(vtimes, vscale);
            _mm512_storeu_ps(&group_times[i], vscaled);
        }
#elif defined(__AVX2__)
        // AVX2: 8-wide SIMD
        __m256 vscale = _mm256_set1_ps(scale);
        for (; i + 8 <= n; i += 8) {
            __m256 vtimes = _mm256_loadu_ps(&group_times[i]);
            __m256 vscaled = _mm256_mul_ps(vtimes, vscale);
            _mm256_storeu_ps(&group_times[i], vscaled);
        }
#elif defined(__ARM_NEON)
        // NEON: 4-wide SIMD
        float32x4_t vscale = vdupq_n_f32(scale);
        for (; i + 4 <= n; i += 4) {
            float32x4_t vtimes = vld1q_f32(&group_times[i]);
            float32x4_t vscaled = vmulq_f32(vtimes, vscale);
            vst1q_f32(&group_times[i], vscaled);
        }
#endif

        // Scalar fallback
        for (; i < n; ++i) {
            group_times[i] *= scale;
        }

        // Write back to map
        for (size_t j = 0; j < group_members.size(); ++j) {
            evolution_times[group_members[j]] = group_times[j];
        }

        std::cout << "[HierarchicalController] Applied GROUP SCALE: "
                  << action.target << " (" << group_members.size() << " members) by "
                  << scale << std::endl;
        return;
    }

    // =========================================================================
    // INDIVIDUAL: Single control adjustment
    // =========================================================================
    if (action.level == TuningLevel::INDIVIDUAL) {
        auto it = evolution_times.find(action.target);
        if (it == evolution_times.end()) {
            std::cerr << "[HierarchicalController] Warning: control '" << action.target
                      << "' not found. No action applied." << std::endl;
            return;
        }

        if (action.type == ActionType::SCALE) {
            it->second *= action.adjustment;
        } else if (action.type == ActionType::SHIFT) {
            it->second += action.adjustment;
        } else if (action.type == ActionType::CLAMP) {
            // Clamp to bounds (future: integrate with control_limits)
            it->second = std::max(5e-4f, std::min(1.0f, it->second));
        }

        std::cout << "[HierarchicalController] Applied INDIVIDUAL " << (action.type == ActionType::SCALE ? "SCALE" : "SHIFT")
                  << ": " << action.target << " → " << it->second << std::endl;
        return;
    }

    std::cerr << "[HierarchicalController] Warning: unknown action type. No action applied." << std::endl;
}

// =============================================================================
// Bottleneck Detection (Phase 3: SIMD-Optimized)
// =============================================================================

std::string HierarchicalController::find_bottleneck_metric(const SystemState& state) const {
    if (state.metrics.empty()) {
        return "loss";  // Default fallback
    }

    // Phase 3 Enhancement: SIMD-optimized deviation computation
    // Extract metrics to parallel arrays for vectorization
    std::vector<std::string> metric_names;
    std::vector<float> current_values;
    std::vector<float> setpoints;
    std::vector<float> scales;

    metric_names.reserve(state.metrics.size());
    current_values.reserve(state.metrics.size());
    setpoints.reserve(state.metrics.size());
    scales.reserve(state.metrics.size());

    for (const auto& [metric_name, current_value] : state.metrics) {
        metric_names.push_back(metric_name);
        current_values.push_back(current_value);

        // Get setpoint (default to current value if not configured)
        float setpoint = metric_setpoints_.count(metric_name)
                        ? metric_setpoints_.at(metric_name)
                        : current_value;
        setpoints.push_back(setpoint);

        // Scale for normalization (max of |setpoint| or 1.0)
        scales.push_back(std::max(std::abs(setpoint), 1.0f));
    }

    const size_t n = current_values.size();
    std::vector<float> normalized_deviations(n);
    size_t i = 0;

    // SIMD-optimized deviation computation with Phase 11 hierarchical guards
#if defined(__AVX512F__)
    // AVX512: Process 16 metrics at once (512-bit registers)
    for (; i + 16 <= n; i += 16) {
        __m512 vcurrent = _mm512_loadu_ps(&current_values[i]);
        __m512 vsetpoints = _mm512_loadu_ps(&setpoints[i]);
        __m512 vscales = _mm512_loadu_ps(&scales[i]);

        // Compute diff = current - setpoint
        __m512 vdiff = _mm512_sub_ps(vcurrent, vsetpoints);

        // Compute abs(diff) = max(diff, -diff)
        __m512 vneg_diff = _mm512_sub_ps(_mm512_setzero_ps(), vdiff);
        __m512 vabs_diff = _mm512_max_ps(vdiff, vneg_diff);

        // Compute normalized deviation = abs_diff / scale
        __m512 vnorm = _mm512_div_ps(vabs_diff, vscales);

        _mm512_storeu_ps(&normalized_deviations[i], vnorm);
    }
#elif defined(__AVX2__)
    // AVX2: Process 8 metrics at once (256-bit registers)
    for (; i + 8 <= n; i += 8) {
        __m256 vcurrent = _mm256_loadu_ps(&current_values[i]);
        __m256 vsetpoints = _mm256_loadu_ps(&setpoints[i]);
        __m256 vscales = _mm256_loadu_ps(&scales[i]);

        // Compute diff = current - setpoint
        __m256 vdiff = _mm256_sub_ps(vcurrent, vsetpoints);

        // Compute abs(diff) = max(diff, -diff)
        __m256 vneg_diff = _mm256_sub_ps(_mm256_setzero_ps(), vdiff);
        __m256 vabs_diff = _mm256_max_ps(vdiff, vneg_diff);

        // Compute normalized deviation = abs_diff / scale
        __m256 vnorm = _mm256_div_ps(vabs_diff, vscales);

        _mm256_storeu_ps(&normalized_deviations[i], vnorm);
    }
#elif defined(__ARM_NEON)
    // NEON: Process 4 metrics at once (128-bit registers)
    for (; i + 4 <= n; i += 4) {
        float32x4_t vcurrent = vld1q_f32(&current_values[i]);
        float32x4_t vsetpoints = vld1q_f32(&setpoints[i]);
        float32x4_t vscales = vld1q_f32(&scales[i]);

        // Compute diff = current - setpoint
        float32x4_t vdiff = vsubq_f32(vcurrent, vsetpoints);

        // Compute abs(diff)
        float32x4_t vabs_diff = vabsq_f32(vdiff);

        // Compute normalized deviation = abs_diff / scale (use reciprocal approximation)
        float32x4_t vrecip = vrecpeq_f32(vscales);  // Reciprocal estimate
        float32x4_t vnorm = vmulq_f32(vabs_diff, vrecip);

        vst1q_f32(&normalized_deviations[i], vnorm);
    }
#endif

    // Scalar fallback for remaining elements
    for (; i < n; ++i) {
        float deviation = std::abs(current_values[i] - setpoints[i]) / scales[i];
        normalized_deviations[i] = deviation;
    }

    // Find maximum deviation (scalar search, already optimized by compiler)
    float max_deviation = 0.0f;
    size_t max_index = 0;
    for (size_t j = 0; j < n; ++j) {
        if (normalized_deviations[j] > max_deviation) {
            max_deviation = normalized_deviations[j];
            max_index = j;
        }
    }

    return metric_names[max_index];
}

// =============================================================================
// Adjustment Computation
// =============================================================================

float HierarchicalController::compute_global_adjustment(const SystemState& state) const {
    // Simple proportional control on loss
    float current_loss = state.metrics.count("loss") ? state.metrics.at("loss") : metric_setpoints_.at("loss");
    float target_loss = metric_setpoints_.at("loss");

    float error = (target_loss - current_loss) / std::max(target_loss, 1.0f);

    // Proportional gain: K_p = 0.2 (20% adjustment per unit error)
    float K_p = 0.2f;
    float adjustment = 1.0f + K_p * error;

    // Clamp adjustment to reasonable range [0.8, 1.2]
    adjustment = std::max(0.8f, std::min(1.2f, adjustment));

    return adjustment;
}

float HierarchicalController::compute_group_adjustment(
    const SystemState& state,
    const std::string& group_name,
    const SensitivityTracker& sensitivity
) const {
    // Use sensitivity-weighted proportional control
    float current_loss = state.metrics.count("loss") ? state.metrics.at("loss") : metric_setpoints_.at("loss");
    float target_loss = metric_setpoints_.at("loss");
    float error = (target_loss - current_loss) / std::max(target_loss, 1.0f);

    // Query group influence (average of top influencers in group)
    auto top_influencers = sensitivity.get_top_influencers("loss", 10, 0.5f);
    float group_influence = 0.0f;
    int group_count = 0;

    for (const auto& inf : top_influencers) {
        if (infer_group_name(inf.control_name) == group_name) {
            group_influence += inf.influence_magnitude;
            group_count++;
        }
    }

    if (group_count > 0) {
        group_influence /= static_cast<float>(group_count);
    } else {
        group_influence = 0.1f;  // Default influence if no data
    }

    // Sensitivity-weighted gain: K_p = 0.15 * (influence / 0.1) = 1.5 * influence
    float K_p = 0.15f * (group_influence / std::max(group_influence, 0.01f));
    float adjustment = 1.0f + K_p * error;

    // Clamp adjustment to reasonable range [0.85, 1.15]
    adjustment = std::max(0.85f, std::min(1.15f, adjustment));

    return adjustment;
}

float HierarchicalController::compute_individual_adjustment(
    const SystemState& state,
    const std::string& control_name,
    const SensitivityTracker& sensitivity
) const {
    // Use sensitivity-weighted PD control (additive adjustment)
    float current_loss = state.metrics.count("loss") ? state.metrics.at("loss") : metric_setpoints_.at("loss");
    float target_loss = metric_setpoints_.at("loss");
    float error = target_loss - current_loss;

    // Query individual influence
    auto top_influencers = sensitivity.get_top_influencers("loss", 10, 0.7f);
    float influence_magnitude = 0.1f;  // Default
    float correlation = 0.0f;

    for (const auto& inf : top_influencers) {
        if (inf.control_name == control_name) {
            influence_magnitude = inf.influence_magnitude;
            correlation = inf.correlation;
            break;
        }
    }

    // Sensitivity-weighted proportional gain: K_p = 0.1 * influence
    float K_p = 0.1f * influence_magnitude;

    // Derivative term (simple: use correlation as proxy for error rate)
    float K_d = 0.05f;
    float adjustment = K_p * error + K_d * correlation * error;

    // Clamp adjustment to reasonable range [-0.0005, +0.0005]
    adjustment = std::max(-0.0005f, std::min(0.0005f, adjustment));

    return adjustment;
}

// =============================================================================
// Causal Parameter Selection (Phase 3)
// =============================================================================

std::vector<std::string> HierarchicalController::select_causal_controls(
    const std::string& bottleneck_metric,
    const SensitivityTracker& sensitivity,
    TuningLevel level
) const {
    std::vector<std::string> selected;

    // =========================================================================
    // GLOBAL Level: No filtering (return global placeholder)
    // =========================================================================
    if (level == TuningLevel::GLOBAL) {
        selected.push_back("global");
        std::cout << "[CausalSelection] Level=GLOBAL, bottleneck=" << bottleneck_metric
                  << ", selected=[global] (no filtering)" << std::endl;
        return selected;
    }

    // =========================================================================
    // INDIVIDUAL Level: Top-3 causal controls
    // =========================================================================
    if (level == TuningLevel::INDIVIDUAL) {
        // Query top-K influencers with Phase 3 thresholds
        constexpr int TOP_K = 3;
        constexpr float MIN_CONFIDENCE = 0.7f;
        constexpr float MIN_MAGNITUDE = 0.1f;

        auto influencers = sensitivity.get_top_influencers(
            bottleneck_metric,
            TOP_K,
            MIN_CONFIDENCE
        );

        // Filter by influence magnitude
        for (const auto& inf : influencers) {
            if (inf.influence_magnitude >= MIN_MAGNITUDE) {
                selected.push_back(inf.control_name);
            }
        }

        // Log causal selection rationale
        std::cout << "[CausalSelection] Level=INDIVIDUAL, bottleneck=" << bottleneck_metric
                  << ", selected " << selected.size() << " causal controls:" << std::endl;
        for (size_t i = 0; i < selected.size() && i < influencers.size(); ++i) {
            std::cout << "  " << (i+1) << ". " << influencers[i].control_name
                      << " (influence=" << influencers[i].influence_magnitude
                      << ", correlation=" << influencers[i].correlation
                      << ", confidence=" << influencers[i].confidence << ")" << std::endl;
        }

        return selected;
    }

    // =========================================================================
    // GROUP Level: Top group inferred from top influencers
    // =========================================================================
    if (level == TuningLevel::GROUP) {
        // Query top influencers (larger pool for group aggregation)
        constexpr int TOP_K = 10;
        constexpr float MIN_CONFIDENCE = 0.6f;

        auto influencers = sensitivity.get_top_influencers(
            bottleneck_metric,
            TOP_K,
            MIN_CONFIDENCE
        );

        if (influencers.empty()) {
            std::cout << "[CausalSelection] Level=GROUP, bottleneck=" << bottleneck_metric
                      << ", no influencers found (fallback to GLOBAL)" << std::endl;
            selected.push_back("global");
            return selected;
        }

        // Aggregate influence by group
        std::map<std::string, float> group_influences;
        std::map<std::string, int> group_counts;

        for (const auto& inf : influencers) {
            std::string group = infer_group_name(inf.control_name);
            group_influences[group] += inf.influence_magnitude;
            group_counts[group]++;
        }

        // Find top group by total influence
        std::string top_group;
        float max_influence = 0.0f;
        int top_group_count = 0;

        for (const auto& [group, total_inf] : group_influences) {
            if (total_inf > max_influence) {
                max_influence = total_inf;
                top_group = group;
                top_group_count = group_counts[group];
            }
        }

        selected.push_back(top_group);

        // Log causal selection rationale
        std::cout << "[CausalSelection] Level=GROUP, bottleneck=" << bottleneck_metric
                  << ", selected group: " << top_group
                  << " (total_influence=" << max_influence
                  << ", member_count=" << top_group_count << ")" << std::endl;

        return selected;
    }

    // Fallback (should not reach here)
    std::cerr << "[CausalSelection] Warning: unknown tuning level, returning empty selection"
              << std::endl;
    return selected;
}

// =============================================================================
// Phase Detection
// =============================================================================

bool HierarchicalController::should_refine_level() const {
    if (current_level_ == TuningLevel::GLOBAL) {
        // Transition: GLOBAL → GROUP
        if (batches_at_current_level_ >= global_to_group_min_batches_) {
            // Check if adjustments are stable (low variance)
            float stability = compute_adjustment_stability();
            if (stability < 0.1f) {
                return true;
            }
        }
        return false;
    }

    if (current_level_ == TuningLevel::GROUP) {
        // Transition: GROUP → INDIVIDUAL
        if (batches_at_current_level_ >= group_to_individual_min_batches_) {
            // Check if adjustments are very stable
            float stability = compute_adjustment_stability();
            if (stability < group_to_individual_adj_threshold_) {
                return true;
            }
        }
        return false;
    }

    // INDIVIDUAL → no further refinement (stay at INDIVIDUAL)
    return false;
}

bool HierarchicalController::should_reset_to_coarse(const SystemState& state) const {
    // Check for large loss spike
    float current_loss = state.metrics.count("loss") ? state.metrics.at("loss") : 0.0f;
    float target_loss = metric_setpoints_.at("loss");

    float loss_deviation = std::abs(current_loss - target_loss);

    return loss_deviation > individual_to_global_loss_spike_;
}

void HierarchicalController::update_adjustment_history(float adjustment) {
    recent_adjustments_.push_back(adjustment);

    // Keep only last N adjustments
    if (recent_adjustments_.size() > static_cast<size_t>(adjustment_stability_window_)) {
        recent_adjustments_.erase(recent_adjustments_.begin());
    }
}

float HierarchicalController::compute_adjustment_stability() const {
    if (recent_adjustments_.size() < 2) {
        return 1.0f;  // Insufficient data, assume unstable
    }

    // Compute standard deviation
    float mean = std::accumulate(recent_adjustments_.begin(), recent_adjustments_.end(), 0.0f) / static_cast<float>(recent_adjustments_.size());
    float variance = 0.0f;
    for (float adj : recent_adjustments_) {
        float diff = adj - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(recent_adjustments_.size());

    return std::sqrt(variance);
}

// =============================================================================
// Helper Functions
// =============================================================================

std::string HierarchicalController::infer_group_name(const std::string& control_name) {
    if (control_name.find("timecrystal_block_") != std::string::npos) {
        return "timecrystal_group";
    } else if (control_name.find("mamba2_block_") != std::string::npos) {
        return "mamba2_group";
    } else if (control_name.find("moe_block_") != std::string::npos) {
        return "moe_group";
    } else if (control_name.find("wlam_block_") != std::string::npos) {
        return "wlam_group";
    } else {
        return "other_group";
    }
}

std::vector<std::string> HierarchicalController::get_group_members(
    const std::string& group_name,
    const std::map<std::string, float>& evolution_times
) {
    std::vector<std::string> members;
    members.reserve(evolution_times.size());

    for (const auto& [control_name, value] : evolution_times) {
        if (infer_group_name(control_name) == group_name) {
            members.push_back(control_name);
        }
    }

    return members;
}

}  // namespace control
}  // namespace saguaro
