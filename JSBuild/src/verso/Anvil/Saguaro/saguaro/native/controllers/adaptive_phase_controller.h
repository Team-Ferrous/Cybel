#ifndef ADAPTIVE_PHASE_CONTROLLER_H_
#define ADAPTIVE_PHASE_CONTROLLER_H_

#include <string>
#include <vector>
#include <map>
#include <random>
#include <chrono>

#include "hierarchical_controller.h"
#include "sensitivity_tracker.h"

namespace saguaro {
namespace control {

/**
 * @brief Adaptive tuning phases for automatic control strategy adjustment.
 *
 * State machine for coarse-to-fine optimization with exploration:
 * COARSE (initial) → REFINE (stable) → FINE (converged) → EXPLORE (plateau) → FINE/COARSE
 *
 * Phase characteristics:
 * - COARSE: Global scaling, large adjustments, handles initial instability
 * - REFINE: Group-level tuning, medium adjustments, targets bottleneck groups
 * - FINE: Individual block tuning, small adjustments, precise optimization
 * - EXPLORE: Random perturbations to escape local optima, plateau breaking
 *
 * Automatic transitions based on:
 * - Loss convergence (deviation from setpoint)
 * - Gradient stability (plateau detection)
 * - Time spent in current phase
 * - Loss spike detection (instability reset)
 */
enum class TuningPhase {
    COARSE = 0,   ///< Global scaling, large adjustments (initial phase)
    REFINE = 1,   ///< Group-level tuning, medium adjustments
    FINE = 2,     ///< Individual block tuning, small adjustments
    EXPLORE = 3   ///< Random perturbations to escape local optima
};

/**
 * @brief Adaptive phase controller for intelligent training phase management.
 *
 * Part of Phase 4 of Intelligent Meta-Controller roadmap.
 * Implements automatic phase transitions based on training dynamics:
 *
 * **Phase Transition Flow:**
 * ```
 * Start → COARSE (500+ batches) → REFINE (1000+ batches) → FINE (2000+ batches) → EXPLORE (500 batches) → FINE
 *                    ↑                                                                           ↓
 *                    └────────────────────────── (loss spike > 2.0) ─────────────────────────────┘
 * ```
 *
 * **Phase Transition Criteria:**
 * - COARSE → REFINE: Loss deviation < 0.5 AND batches_in_phase > 500
 * - REFINE → FINE: Loss deviation < 0.1 AND batches_in_phase > 1000
 * - FINE → EXPLORE: Gradient norm < 0.01 (plateau) AND batches_in_phase > 2000
 * - EXPLORE → COARSE: Loss spike > 2.0 (instability detected)
 * - EXPLORE → FINE: batches_in_phase > 500 (exploration complete)
 *
 * **Integration with HierarchicalController:**
 * - COARSE phase → GLOBAL tuning level (uniform scaling)
 * - REFINE phase → GROUP tuning level (block type groups)
 * - FINE phase → INDIVIDUAL tuning level (single blocks)
 * - EXPLORE phase → INDIVIDUAL level with random perturbations
 *
 * Key features:
 * - Automatic phase detection and transition
 * - Plateau detection via gradient norm monitoring
 * - Instability reset (EXPLORE → COARSE on loss spikes)
 * - Exploration strategy with Gaussian perturbations
 * - Telemetry and explainable phase transitions
 *
 * Thread safety: Not thread-safe. Caller must serialize calls to update_phase().
 *
 * Performance:
 * - update_phase(): O(1) (simple threshold checks)
 * - get_phase_action(): O(n log k) where n = controls, k = top_k (dominated by sensitivity ranking)
 */
class AdaptivePhaseController {
public:
    /**
     * @brief Construct adaptive phase controller with default thresholds.
     *
     * Initializes to COARSE phase with standard transition thresholds.
     */
    AdaptivePhaseController();

    ~AdaptivePhaseController() = default;

    /**
     * @brief Update current phase based on system state and sensitivity.
     *
     * Evaluates phase transition criteria and updates phase if conditions met:
     * 1. Compute loss deviation from setpoint
     * 2. Check gradient norm for plateau detection
     * 3. Evaluate phase-specific transition conditions
     * 4. Transition to next phase if criteria satisfied
     * 5. Reset batch counter on transition
     *
     * Phase transition logic:
     * - COARSE: Check if loss stable (deviation < 0.5) after 500 batches → REFINE
     * - REFINE: Check if loss converged (deviation < 0.1) after 1000 batches → FINE
     * - FINE: Check if plateau (gradient norm < 0.01) after 2000 batches → EXPLORE
     * - EXPLORE: Check if instability (loss spike > 2.0) → COARSE, else after 500 batches → FINE
     *
     * @param state Current system state (metrics, evolution_times, batch_number)
     * @param sensitivity Sensitivity tracker with historical influence data
     *
     * Example:
     * ```cpp
     * SystemState state;
     * state.metrics = {{"loss", 2.8}, {"gradient_norm", 0.56}};
     * state.batch_number = 600;
     *
     * phase_controller->update_phase(state, *sensitivity_tracker);
     * // After 600 batches with loss deviation = 0.3, transitions COARSE → REFINE
     * ```
     */
    void update_phase(
        const SystemState& state,
        const SensitivityTracker& sensitivity
    );

    /**
     * @brief Get hierarchical action appropriate for current phase.
     *
     * Maps current phase to corresponding hierarchical action type:
     * - COARSE phase: Global scale action (GLOBAL level, uniform scaling)
     * - REFINE phase: Group adjustment action (GROUP level, bottleneck groups)
     * - FINE phase: Individual adjustment action (INDIVIDUAL level, causal controls)
     * - EXPLORE phase: Exploration action (INDIVIDUAL level, random perturbation)
     *
     * Delegates to HierarchicalController for COARSE/REFINE/FINE phases.
     * Generates random exploration action for EXPLORE phase.
     *
     * @param state Current system state
     * @param sensitivity Sensitivity tracker
     * @param hierarchical_controller Reference to hierarchical controller for action computation
     * @return HierarchicalAction appropriate for current phase
     *
     * Example:
     * ```cpp
     * HierarchicalAction action = phase_controller->get_phase_action(
     *     state, *sensitivity_tracker, *hierarchical_controller
     * );
     * // In EXPLORE phase: returns random perturbation action
     * // action.level = TuningLevel::INDIVIDUAL
     * // action.target = "mamba2_block_3/evolution_time" (random)
     * // action.adjustment = 0.023 (Gaussian noise)
     * ```
     */
    HierarchicalAction get_phase_action(
        const SystemState& state,
        const SensitivityTracker& sensitivity,
        HierarchicalController& hierarchical_controller
    );

    /**
     * @brief Get current tuning phase.
     *
     * @return Current TuningPhase (COARSE, REFINE, FINE, EXPLORE)
     */
    TuningPhase current_phase() const { return phase_; }

    /**
     * @brief Get number of batches spent in current phase.
     *
     * @return Batches in current phase (resets to 0 on transition)
     */
    int get_batches_in_phase() const { return batches_in_phase_; }

    /**
     * @brief Force phase transition to specified phase.
     *
     * Useful for manual intervention, testing, or HPO-driven phase control.
     *
     * @param phase Target tuning phase
     */
    void set_phase(TuningPhase phase);

    /**
     * @brief Reset adaptive phase controller to initial state.
     *
     * Resets to COARSE phase with batch counter = 0.
     */
    void reset();

    /**
     * @brief Get target loss setpoint for phase transition calculations.
     *
     * @return Target loss value (default: 2.5)
     */
    float get_target_loss() const { return target_loss_; }

    /**
     * @brief Set target loss setpoint.
     *
     * Allows dynamic adjustment of loss target based on dataset or user preference.
     *
     * @param target Target loss value
     */
    void set_target_loss(float target);

private:
    // Current state
    TuningPhase phase_;           ///< Current tuning phase
    int batches_in_phase_;        ///< Batches spent in current phase

    // Loss setpoint for deviation calculation
    float target_loss_;           ///< Target loss value for phase transitions (default: 2.5)

    // Phase transition thresholds
    float coarse_to_refine_threshold_;     ///< Max loss deviation to transition COARSE → REFINE (default: 0.5)
    int coarse_min_batches_;               ///< Min batches before transitioning COARSE → REFINE (default: 500)

    float refine_to_fine_threshold_;       ///< Max loss deviation to transition REFINE → FINE (default: 0.1)
    int refine_min_batches_;               ///< Min batches before transitioning REFINE → FINE (default: 1000)

    float fine_to_explore_threshold_;      ///< Max gradient norm to transition FINE → EXPLORE (default: 0.01)
    int fine_min_batches_;                 ///< Min batches before transitioning FINE → EXPLORE (default: 2000)

    float explore_to_coarse_instability_;  ///< Loss spike threshold to reset EXPLORE → COARSE (default: 2.0)
    int explore_duration_;                 ///< Batches to spend in EXPLORE before returning to FINE (default: 500)

    // Random number generation for exploration
    std::mt19937 rng_;                     ///< Random number generator
    std::normal_distribution<float> gaussian_dist_;  ///< Gaussian distribution for perturbations

    /**
     * @brief Compute exploration action (random perturbation).
     *
     * Generates random perturbation action to escape local optima:
     * 1. Sample random control from available evolution_times
     * 2. Generate Gaussian perturbation (mean=0, std=0.1)
     * 3. Return INDIVIDUAL-level SHIFT action
     *
     * @param state System state with evolution_times
     * @return HierarchicalAction with random target and Gaussian adjustment
     */
    HierarchicalAction compute_exploration_action(const SystemState& state);

    /**
     * @brief Sample random control name from evolution_times.
     *
     * Uniformly samples a control input from available evolution_times map.
     *
     * @param evolution_times Map of control_name → value
     * @return Random control name (e.g., "timecrystal_block_2/evolution_time")
     */
    std::string sample_random_control(
        const std::map<std::string, float>& evolution_times
    );
};

}  // namespace control
}  // namespace saguaro

#endif  // ADAPTIVE_PHASE_CONTROLLER_H_
