#include "adaptive_phase_controller.h"

#include <iostream>
#include <cmath>
#include <algorithm>

namespace saguaro {
namespace control {

AdaptivePhaseController::AdaptivePhaseController()
    : phase_(TuningPhase::COARSE)
    , batches_in_phase_(0)
    , target_loss_(2.5f)
    , coarse_to_refine_threshold_(0.5f)
    , coarse_min_batches_(500)
    , refine_to_fine_threshold_(0.1f)
    , refine_min_batches_(1000)
    , fine_to_explore_threshold_(0.01f)
    , fine_min_batches_(2000)
    , explore_to_coarse_instability_(2.0f)
    , explore_duration_(500)
    , rng_(std::chrono::steady_clock::now().time_since_epoch().count())
    , gaussian_dist_(0.0f, 0.1f)  // Mean=0, StdDev=0.1
{
    std::cout << "[AdaptivePhaseController] Initialized to COARSE phase" << std::endl;
    std::cout << "  Transition thresholds:" << std::endl;
    std::cout << "    COARSE → REFINE: loss_dev < " << coarse_to_refine_threshold_
              << ", batches > " << coarse_min_batches_ << std::endl;
    std::cout << "    REFINE → FINE: loss_dev < " << refine_to_fine_threshold_
              << ", batches > " << refine_min_batches_ << std::endl;
    std::cout << "    FINE → EXPLORE: grad_norm < " << fine_to_explore_threshold_
              << ", batches > " << fine_min_batches_ << std::endl;
    std::cout << "    EXPLORE → COARSE: loss_spike > " << explore_to_coarse_instability_ << std::endl;
    std::cout << "    EXPLORE → FINE: batches > " << explore_duration_ << std::endl;
}

void AdaptivePhaseController::update_phase(
    const SystemState& state,
    const SensitivityTracker& sensitivity
) {
    batches_in_phase_++;

    // Compute loss deviation from target
    float loss = 0.0f;
    float gradient_norm = 0.0f;
    bool has_valid_loss = false;
    bool has_valid_grad = false;

    auto loss_it = state.metrics.find("loss");
    auto grad_it = state.metrics.find("gradient_norm");

    if (loss_it != state.metrics.end() && std::isfinite(loss_it->second)) {
        loss = loss_it->second;
        has_valid_loss = true;
    } else {
        // Only log warning once per 100 batches to avoid spam
        if (batches_in_phase_ % 100 == 1) {
            std::cerr << "[AdaptivePhaseController] Warning: 'loss' metric not found or non-finite. "
                      << "Phase transitions disabled until metric stabilizes." << std::endl;
        }
    }

    if (grad_it != state.metrics.end() && std::isfinite(grad_it->second)) {
        gradient_norm = grad_it->second;
        has_valid_grad = true;
    } else {
        // Only log warning once per 100 batches to avoid spam
        if (batches_in_phase_ % 100 == 1) {
            std::cerr << "[AdaptivePhaseController] Warning: 'gradient_norm' metric not found or non-finite. "
                      << "Using default value 0.0" << std::endl;
        }
    }

    // Skip phase transitions if critical metrics are invalid
    if (!has_valid_loss) {
        // Keep current phase but don't advance transition logic
        return;
    }

    float deviation = std::abs(loss - target_loss_);

    // Phase transition logic
    switch (phase_) {
        case TuningPhase::COARSE:
            // COARSE → REFINE: Loss stabilizes (deviation < 0.5) after 500 batches
            if (deviation < coarse_to_refine_threshold_ && batches_in_phase_ > coarse_min_batches_) {
                std::cout << "[AdaptivePhaseController] Transition: COARSE → REFINE" << std::endl;
                std::cout << "  Reason: Loss stable (deviation=" << deviation
                          << " < " << coarse_to_refine_threshold_ << "), "
                          << "batches=" << batches_in_phase_ << std::endl;
                phase_ = TuningPhase::REFINE;
                batches_in_phase_ = 0;
            }
            break;

        case TuningPhase::REFINE:
            // REFINE → FINE: Loss converges (deviation < 0.1) after 1000 batches
            if (deviation < refine_to_fine_threshold_ && batches_in_phase_ > refine_min_batches_) {
                std::cout << "[AdaptivePhaseController] Transition: REFINE → FINE" << std::endl;
                std::cout << "  Reason: Loss converged (deviation=" << deviation
                          << " < " << refine_to_fine_threshold_ << "), "
                          << "batches=" << batches_in_phase_ << std::endl;
                phase_ = TuningPhase::FINE;
                batches_in_phase_ = 0;
            }
            break;

        case TuningPhase::FINE:
            // FINE → EXPLORE: Plateau detected (gradient norm < 0.01) after 2000 batches
            // Only transition if gradient_norm is valid
            if (has_valid_grad && gradient_norm < fine_to_explore_threshold_ && batches_in_phase_ > fine_min_batches_) {
                std::cout << "[AdaptivePhaseController] Transition: FINE → EXPLORE" << std::endl;
                std::cout << "  Reason: Plateau detected (grad_norm=" << gradient_norm
                          << " < " << fine_to_explore_threshold_ << "), "
                          << "batches=" << batches_in_phase_ << std::endl;
                phase_ = TuningPhase::EXPLORE;
                batches_in_phase_ = 0;
            }
            break;

        case TuningPhase::EXPLORE:
            // EXPLORE → COARSE: Instability detected (loss spike > 2.0)
            if (deviation > explore_to_coarse_instability_) {
                std::cout << "[AdaptivePhaseController] Transition: EXPLORE → COARSE" << std::endl;
                std::cout << "  Reason: Instability detected (deviation=" << deviation
                          << " > " << explore_to_coarse_instability_ << ")" << std::endl;
                phase_ = TuningPhase::COARSE;
                batches_in_phase_ = 0;
            }
            // EXPLORE → FINE: Exploration complete (after 500 batches)
            else if (batches_in_phase_ > explore_duration_) {
                std::cout << "[AdaptivePhaseController] Transition: EXPLORE → FINE" << std::endl;
                std::cout << "  Reason: Exploration complete (batches=" << batches_in_phase_
                          << " > " << explore_duration_ << ")" << std::endl;
                phase_ = TuningPhase::FINE;
                batches_in_phase_ = 0;
            }
            break;
    }
}

HierarchicalAction AdaptivePhaseController::get_phase_action(
    const SystemState& state,
    const SensitivityTracker& sensitivity,
    HierarchicalController& hierarchical_controller
) {
    switch (phase_) {
        case TuningPhase::COARSE:
            // COARSE phase: Use GLOBAL level from hierarchical controller
            hierarchical_controller.set_tuning_level(TuningLevel::GLOBAL);
            return hierarchical_controller.decide_action(state, sensitivity);

        case TuningPhase::REFINE:
            // REFINE phase: Use GROUP level from hierarchical controller
            hierarchical_controller.set_tuning_level(TuningLevel::GROUP);
            return hierarchical_controller.decide_action(state, sensitivity);

        case TuningPhase::FINE:
            // FINE phase: Use INDIVIDUAL level from hierarchical controller
            hierarchical_controller.set_tuning_level(TuningLevel::INDIVIDUAL);
            return hierarchical_controller.decide_action(state, sensitivity);

        case TuningPhase::EXPLORE:
            // EXPLORE phase: Random perturbation for plateau escape
            return compute_exploration_action(state);
    }

    // Fallback (should never reach here)
    return HierarchicalAction();
}

HierarchicalAction AdaptivePhaseController::compute_exploration_action(
    const SystemState& state
) {
    // Check if we have any evolution_times to perturb
    if (state.evolution_times.empty()) {
        std::cerr << "[AdaptivePhaseController] Warning: No evolution_times available for exploration. "
                  << "Returning no-op action." << std::endl;
        return HierarchicalAction(
            TuningLevel::INDIVIDUAL,
            "",
            0.0f,
            ActionType::SHIFT,
            "No evolution_times available for exploration"
        );
    }

    // Sample random control
    std::string target_control = sample_random_control(state.evolution_times);

    // Generate Gaussian perturbation (mean=0, std=0.1)
    float perturbation = gaussian_dist_(rng_);

    std::cout << "[AdaptivePhaseController] EXPLORE action: target=" << target_control
              << ", perturbation=" << perturbation << std::endl;

    return HierarchicalAction(
        TuningLevel::INDIVIDUAL,
        target_control,
        perturbation,
        ActionType::SHIFT,
        "Exploration: random perturbation to escape plateau"
    );
}

std::string AdaptivePhaseController::sample_random_control(
    const std::map<std::string, float>& evolution_times
) {
    // Build vector of control names
    std::vector<std::string> control_names;
    control_names.reserve(evolution_times.size());
    for (const auto& kv : evolution_times) {
        control_names.push_back(kv.first);
    }

    // Uniform random selection
    std::uniform_int_distribution<size_t> uniform_dist(0, control_names.size() - 1);
    size_t random_idx = uniform_dist(rng_);

    return control_names[random_idx];
}

void AdaptivePhaseController::set_phase(TuningPhase phase) {
    if (phase_ != phase) {
        std::cout << "[AdaptivePhaseController] Manual phase transition: ";

        // Print phase names
        auto print_phase = [](TuningPhase p) {
            switch (p) {
                case TuningPhase::COARSE: return "COARSE";
                case TuningPhase::REFINE: return "REFINE";
                case TuningPhase::FINE: return "FINE";
                case TuningPhase::EXPLORE: return "EXPLORE";
                default: return "UNKNOWN";
            }
        };

        std::cout << print_phase(phase_) << " → " << print_phase(phase) << std::endl;

        phase_ = phase;
        batches_in_phase_ = 0;
    }
}

void AdaptivePhaseController::reset() {
    std::cout << "[AdaptivePhaseController] Reset to COARSE phase" << std::endl;
    phase_ = TuningPhase::COARSE;
    batches_in_phase_ = 0;
}

void AdaptivePhaseController::set_target_loss(float target) {
    if (target < 0.0f) {
        std::cerr << "[AdaptivePhaseController] Warning: Invalid target_loss " << target
                  << ". Must be non-negative. Using default 2.5." << std::endl;
        target_loss_ = 2.5f;
    } else {
        std::cout << "[AdaptivePhaseController] Updated target_loss: " << target_loss_
                  << " → " << target << std::endl;
        target_loss_ = target;
    }
}

}  // namespace control
}  // namespace saguaro
