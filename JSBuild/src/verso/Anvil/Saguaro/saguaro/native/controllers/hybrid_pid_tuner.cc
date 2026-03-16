// saguaro.native/controllers/hybrid_pid_tuner.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Hybrid PID Tuner implementation combining Relay/Ziegler-Nichols with Adam.

#include "controllers/hybrid_pid_tuner.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace saguaro {
namespace controllers {

HybridPIDTuner::HybridPIDTuner() {
    try {
        logger_ = spdlog::get("highnoon");
        if (!logger_) {
            logger_ = spdlog::stdout_color_mt("highnoon");
        }
    } catch (...) {
        // Continue without logging
    }
}

void HybridPIDTuner::init(int num_channels) {
    init(num_channels, Config{});
}

void HybridPIDTuner::init(int num_channels, const Config& config) {
    if (num_channels <= 0) {
        throw std::invalid_argument("HybridPIDTuner: num_channels must be positive");
    }

    config_ = config;
    num_channels_ = num_channels;

    // Initialize gains to safe defaults
    kp_ = Eigen::VectorXf::Constant(num_channels_, 0.1f);
    ki_ = Eigen::VectorXf::Constant(num_channels_, 0.01f);
    kd_ = Eigen::VectorXf::Constant(num_channels_, 0.001f);

    // Initialize Adam state
    adam_kp_.m = Eigen::VectorXf::Zero(num_channels_);
    adam_kp_.v = Eigen::VectorXf::Zero(num_channels_);
    adam_ki_.m = Eigen::VectorXf::Zero(num_channels_);
    adam_ki_.v = Eigen::VectorXf::Zero(num_channels_);
    adam_kd_.m = Eigen::VectorXf::Zero(num_channels_);
    adam_kd_.v = Eigen::VectorXf::Zero(num_channels_);
    adam_step_ = 0;

    // Initialize error tracking
    integral_error_ = Eigen::VectorXf::Zero(num_channels_);
    prev_error_ = Eigen::VectorXf::Zero(num_channels_);

    // Start in relay experiment mode
    state_ = TunerState::RELAY_EXPERIMENT;
    relay_start_ = std::chrono::steady_clock::now();
    relay_high_ = true;
    relay_output_ = config_.relay_amplitude;
    oscillation_count_ = 0;
    relay_crossings_.clear();
    relay_amplitudes_.clear();

    if (logger_) {
        logger_->info("HybridPIDTuner initialized: {} channels, lr={:.4f}, relay_amp={:.3f}",
                      num_channels_, config_.learning_rate, config_.relay_amplitude);
    }
}

float HybridPIDTuner::relayStep(float error) {
    if (state_ != TunerState::RELAY_EXPERIMENT) {
        return 0.0f;
    }

    auto now = std::chrono::steady_clock::now();
    float elapsed_ms = std::chrono::duration<float, std::milli>(now - relay_start_).count();

    // Check for zero crossing with hysteresis
    bool should_switch = false;
    if (relay_high_ && error < -config_.relay_hysteresis) {
        should_switch = true;
    } else if (!relay_high_ && error > config_.relay_hysteresis) {
        should_switch = true;
    }

    if (should_switch) {
        // Record crossing time
        float crossing_time = elapsed_ms;
        relay_crossings_.push_back(crossing_time);

        // Record amplitude (absolute error at crossing)
        relay_amplitudes_.push_back(std::abs(error));

        // Toggle relay
        relay_high_ = !relay_high_;
        relay_output_ = relay_high_ ? config_.relay_amplitude : -config_.relay_amplitude;

        // Count full cycles (2 crossings = 1 cycle)
        if (relay_crossings_.size() >= 2 && relay_crossings_.size() % 2 == 0) {
            oscillation_count_++;
            if (logger_) {
                logger_->debug("Relay oscillation cycle {}/{}", 
                              oscillation_count_, config_.min_oscillation_cycles);
            }
        }

        last_crossing_ = now;
    }

    return relay_output_;
}

bool HybridPIDTuner::isRelayComplete() const {
    return state_ == TunerState::RELAY_EXPERIMENT &&
           oscillation_count_ >= config_.min_oscillation_cycles;
}

void HybridPIDTuner::finalizeRelay() {
    if (relay_crossings_.size() < 4) {
        if (logger_) {
            logger_->warn("Relay experiment incomplete, using default gains");
        }
        // Use conservative defaults
        relay_gains_.Kp = Eigen::VectorXf::Constant(num_channels_, 0.5f);
        relay_gains_.Ki = Eigen::VectorXf::Constant(num_channels_, 0.1f);
        relay_gains_.Kd = Eigen::VectorXf::Constant(num_channels_, 0.05f);
        
        kp_ = relay_gains_.Kp;
        ki_ = relay_gains_.Ki;
        kd_ = relay_gains_.Kd;
        
        state_ = TunerState::ADAM_OPTIMIZING;
        return;
    }

    // Calculate ultimate period (average of half-periods * 2)
    std::vector<float> periods;
    for (size_t i = 1; i < relay_crossings_.size(); i++) {
        periods.push_back((relay_crossings_[i] - relay_crossings_[i-1]) * 2.0f);
    }
    float Tu = std::accumulate(periods.begin(), periods.end(), 0.0f) / periods.size();
    Tu /= 1000.0f;  // Convert to seconds

    // Calculate ultimate gain: Ku = 4*d / (pi*a)
    // where d = relay amplitude, a = average oscillation amplitude
    float avg_amplitude = std::accumulate(relay_amplitudes_.begin(), 
                                          relay_amplitudes_.end(), 0.0f) / 
                          relay_amplitudes_.size();
    
    // Avoid division by zero
    if (avg_amplitude < 1e-6f) avg_amplitude = 1e-6f;
    
    float Ku = (4.0f * config_.relay_amplitude) / (3.14159f * avg_amplitude);

    ultimate_gain_ = Ku;
    ultimate_period_ = Tu;

    if (logger_) {
        logger_->info("Relay analysis: Ku={:.4f}, Tu={:.4f}s", Ku, Tu);
    }

    // Calculate Z-N gains
    calculateZNGains(Ku, Tu);

    // Seed Adam with relay gains
    kp_ = relay_gains_.Kp;
    ki_ = relay_gains_.Ki;
    kd_ = relay_gains_.Kd;

    // Reset Adam state for fresh start
    adam_kp_.m = Eigen::VectorXf::Zero(num_channels_);
    adam_kp_.v = Eigen::VectorXf::Zero(num_channels_);
    adam_ki_.m = Eigen::VectorXf::Zero(num_channels_);
    adam_ki_.v = Eigen::VectorXf::Zero(num_channels_);
    adam_kd_.m = Eigen::VectorXf::Zero(num_channels_);
    adam_kd_.v = Eigen::VectorXf::Zero(num_channels_);
    adam_step_ = 0;

    // Record baseline performance for divergence detection
    baseline_error_norm_ = avg_amplitude;

    state_ = TunerState::ADAM_OPTIMIZING;

    if (logger_) {
        logger_->info("Transitioning to Adam optimization with Z-N initialization");
    }
}

void HybridPIDTuner::calculateZNGains(float Ku, float Tu) {
    float Kp = 0.0f;
    float Ki = 0.0f;
    float Kd = 0.0f;

    // Ziegler-Nichols formulas based on tuning type
    switch (config_.zn_type) {
        case ZNType::CLASSIC:
            // Classic Z-N: aggressive, may have ~25% overshoot
            Kp = 0.6f * Ku;
            Ki = 2.0f * Kp / Tu;
            Kd = Kp * Tu / 8.0f;
            break;

        case ZNType::SOME_OVERSHOOT:
            // Some overshoot: balanced response
            Kp = 0.33f * Ku;
            Ki = 2.0f * Kp / Tu;
            Kd = Kp * Tu / 3.0f;
            break;

        case ZNType::NO_OVERSHOOT:
            // No overshoot: conservative, slower but stable
            Kp = 0.2f * Ku;
            Ki = 2.0f * Kp / Tu;
            Kd = Kp * Tu / 3.0f;
            break;

        case ZNType::PESSEN:
            // Pessen Integral Rule: good load rejection
            Kp = 0.7f * Ku;
            Ki = 2.5f * Kp / Tu;
            Kd = 0.15f * Kp * Tu;
            break;
    }

    // Apply per-channel (same gains for all channels initially)
    relay_gains_.Kp = Eigen::VectorXf::Constant(num_channels_, Kp);
    relay_gains_.Ki = Eigen::VectorXf::Constant(num_channels_, Ki);
    relay_gains_.Kd = Eigen::VectorXf::Constant(num_channels_, Kd);

    // Clamp to bounds
    relay_gains_.Kp = relay_gains_.Kp.cwiseMax(config_.kp_min).cwiseMin(config_.kp_max);
    relay_gains_.Ki = relay_gains_.Ki.cwiseMax(config_.ki_min).cwiseMin(config_.ki_max);
    relay_gains_.Kd = relay_gains_.Kd.cwiseMax(config_.kd_min).cwiseMin(config_.kd_max);

    if (logger_) {
        logger_->info("Z-N gains computed: Kp={:.4f}, Ki={:.4f}, Kd={:.4f}",
                      relay_gains_.Kp(0), relay_gains_.Ki(0), relay_gains_.Kd(0));
    }
}

void HybridPIDTuner::update(const Eigen::VectorXf& error,
                            const Eigen::VectorXf& control_effort,
                            float loss) {
    if (state_ != TunerState::ADAM_OPTIMIZING && state_ != TunerState::FALLBACK_MODE) {
        return;
    }

    if (error.size() != num_channels_ || control_effort.size() != num_channels_) {
        if (logger_) {
            logger_->warn("HybridPIDTuner: dimension mismatch in update");
        }
        return;
    }

    // Check for divergence
    if (checkDivergence(loss)) {
        if (state_ != TunerState::FALLBACK_MODE) {
            if (logger_) {
                logger_->warn("Divergence detected, falling back to relay gains");
            }
            kp_ = relay_gains_.Kp;
            ki_ = relay_gains_.Ki;
            kd_ = relay_gains_.Kd;
            state_ = TunerState::FALLBACK_MODE;
            divergence_detected_ = true;
        }
        return;
    }

    // Update error history for derivative
    error_history_.push_front(error);
    if (error_history_.size() > 10) {
        error_history_.pop_back();
    }

    // Update integral (anti-windup: clamp integral term)
    integral_error_ += error;
    float integral_limit = 10.0f;
    integral_error_ = integral_error_.cwiseMax(-integral_limit).cwiseMin(integral_limit);

    // Store previous gains for rate limiting
    Eigen::VectorXf prev_kp = kp_;
    Eigen::VectorXf prev_ki = ki_;
    Eigen::VectorXf prev_kd = kd_;

    // Compute gradients
    Eigen::VectorXf grad_kp, grad_ki, grad_kd;
    computeGradients(error, control_effort, loss, grad_kp, grad_ki, grad_kd);

    // Apply Adam updates
    adam_step_++;
    adamUpdate(kp_, adam_kp_, grad_kp);
    adamUpdate(ki_, adam_ki_, grad_ki);
    adamUpdate(kd_, adam_kd_, grad_kd);

    // Rate limit changes
    rateLimitGains(prev_kp, prev_ki, prev_kd);

    // Clamp to bounds
    clampGains();

    // Update previous error
    prev_error_ = error;

    // Update performance metrics
    updateMetrics(error, control_effort);

    if (logger_ && adam_step_ % 100 == 0) {
        logger_->debug("Adam step {}: Kp_avg={:.4f}, Ki_avg={:.4f}, Kd_avg={:.4f}",
                       adam_step_, kp_.mean(), ki_.mean(), kd_.mean());
    }
}

void HybridPIDTuner::computeGradients(const Eigen::VectorXf& error,
                                       const Eigen::VectorXf& control_effort,
                                       float loss,
                                       Eigen::VectorXf& grad_kp,
                                       Eigen::VectorXf& grad_ki,
                                       Eigen::VectorXf& grad_kd) {
    // Proxy loss: L = error^2 + lambda * control_effort^2
    // We want to minimize error while not using excessive control
    float control_penalty = 0.01f;

    // Derivative of error w.r.t. time (approximation)
    Eigen::VectorXf error_derivative = Eigen::VectorXf::Zero(num_channels_);
    if (error_history_.size() >= 2) {
        error_derivative = error_history_[0] - error_history_[1];
    }

    // Gradient approximations using chain rule:
    // dL/dKp ≈ d(error²)/dKp ≈ 2*error * d(error)/dKp
    // Since PID output affects error reduction, we use negative correlation
    //
    // For PID: u = Kp*e + Ki*∫e + Kd*de/dt
    // Higher gains generally reduce error faster but can cause overshoot
    
    // Use sign of error * control alignment as gradient direction
    // If error and control are aligned (both reducing error), decrease gradient
    Eigen::VectorXf error_sq = error.array().square();
    Eigen::VectorXf control_sq = control_effort.array().square();

    // P-term gradient: proportional to error magnitude
    grad_kp = -2.0f * error.cwiseProduct(error) + 
              control_penalty * 2.0f * control_effort.cwiseProduct(error);

    // I-term gradient: proportional to integral error
    grad_ki = -2.0f * error.cwiseProduct(integral_error_) +
              control_penalty * 2.0f * control_effort.cwiseProduct(integral_error_);

    // D-term gradient: proportional to error derivative
    grad_kd = -2.0f * error.cwiseProduct(error_derivative) +
              control_penalty * 2.0f * control_effort.cwiseProduct(error_derivative);

    // Scale by loss to incorporate external training signal
    float loss_scale = std::min(loss, 10.0f);  // Prevent explosion
    grad_kp *= (1.0f + 0.1f * loss_scale);
    grad_ki *= (1.0f + 0.1f * loss_scale);
    grad_kd *= (1.0f + 0.1f * loss_scale);
}

void HybridPIDTuner::adamUpdate(Eigen::VectorXf& param,
                                 AdamState& state,
                                 const Eigen::VectorXf& grad) {
    // Adam update with bias correction
    state.m = config_.beta1 * state.m + (1.0f - config_.beta1) * grad;
    state.v = config_.beta2 * state.v + (1.0f - config_.beta2) * grad.array().square().matrix();

    // Bias correction
    float beta1_t = std::pow(config_.beta1, adam_step_);
    float beta2_t = std::pow(config_.beta2, adam_step_);
    Eigen::VectorXf m_hat = state.m / (1.0f - beta1_t);
    Eigen::VectorXf v_hat = state.v / (1.0f - beta2_t);

    // Update parameters
    param -= config_.learning_rate * m_hat.cwiseQuotient(
        (v_hat.array().sqrt() + config_.epsilon).matrix());
}

void HybridPIDTuner::clampGains() {
    kp_ = kp_.cwiseMax(config_.kp_min).cwiseMin(config_.kp_max);
    ki_ = ki_.cwiseMax(config_.ki_min).cwiseMin(config_.ki_max);
    kd_ = kd_.cwiseMax(config_.kd_min).cwiseMin(config_.kd_max);
}

void HybridPIDTuner::rateLimitGains(const Eigen::VectorXf& prev_kp,
                                     const Eigen::VectorXf& prev_ki,
                                     const Eigen::VectorXf& prev_kd) {
    float max_change = config_.max_gain_change_rate;

    for (int i = 0; i < num_channels_; i++) {
        // Kp rate limiting
        float delta_kp = kp_(i) - prev_kp(i);
        if (std::abs(delta_kp) > max_change) {
            kp_(i) = prev_kp(i) + std::copysign(max_change, delta_kp);
        }

        // Ki rate limiting
        float delta_ki = ki_(i) - prev_ki(i);
        if (std::abs(delta_ki) > max_change) {
            ki_(i) = prev_ki(i) + std::copysign(max_change, delta_ki);
        }

        // Kd rate limiting
        float delta_kd = kd_(i) - prev_kd(i);
        if (std::abs(delta_kd) > max_change) {
            kd_(i) = prev_kd(i) + std::copysign(max_change, delta_kd);
        }
    }
}

bool HybridPIDTuner::checkDivergence(float loss) {
    loss_history_.push_front(loss);
    if (loss_history_.size() > static_cast<size_t>(config_.divergence_window)) {
        loss_history_.pop_back();
    }

    if (loss_history_.size() < static_cast<size_t>(config_.divergence_window / 2)) {
        return false;  // Not enough data
    }

    // Compare recent average to historical average
    size_t half = loss_history_.size() / 2;
    float recent_avg = 0.0f;
    float older_avg = 0.0f;

    for (size_t i = 0; i < half; i++) {
        recent_avg += loss_history_[i];
    }
    for (size_t i = half; i < loss_history_.size(); i++) {
        older_avg += loss_history_[i];
    }
    recent_avg /= half;
    older_avg /= (loss_history_.size() - half);

    // Divergence if recent is much worse than older
    if (older_avg > 1e-6f) {
        float ratio = recent_avg / older_avg;
        if (ratio > config_.divergence_threshold) {
            return true;
        }
    }

    return false;
}

void HybridPIDTuner::updateMetrics(const Eigen::VectorXf& error,
                                    const Eigen::VectorXf& control) {
    error_norms_.push_front(error.norm());
    control_norms_.push_front(control.norm());

    if (error_norms_.size() > static_cast<size_t>(config_.performance_window)) {
        error_norms_.pop_back();
    }
    if (control_norms_.size() > static_cast<size_t>(config_.performance_window)) {
        control_norms_.pop_back();
    }

    // Update current average
    if (!error_norms_.empty()) {
        current_error_avg_ = std::accumulate(error_norms_.begin(), 
                                              error_norms_.end(), 0.0f) / 
                             error_norms_.size();
    }
}

PIDGains HybridPIDTuner::getGains() const {
    PIDGains gains;
    gains.Kp = kp_;
    gains.Ki = ki_;
    gains.Kd = kd_;
    return gains;
}

void HybridPIDTuner::setGains(const PIDGains& gains) {
    if (gains.Kp.size() == num_channels_ &&
        gains.Ki.size() == num_channels_ &&
        gains.Kd.size() == num_channels_) {
        kp_ = gains.Kp;
        ki_ = gains.Ki;
        kd_ = gains.Kd;
        clampGains();
    }
}

void HybridPIDTuner::seedGains(const PIDGains& gains) {
    if (!gains.isValid() || gains.Kp.size() != num_channels_) {
        if (logger_) {
            logger_->warn("Invalid gains for seeding, starting relay experiment");
        }
        return;
    }

    kp_ = gains.Kp;
    ki_ = gains.Ki;
    kd_ = gains.Kd;
    clampGains();

    // Store as relay gains for fallback
    relay_gains_ = gains;

    // Skip relay experiment
    state_ = TunerState::ADAM_OPTIMIZING;

    if (logger_) {
        logger_->info("Gains seeded, skipping relay experiment");
    }
}

void HybridPIDTuner::reset(bool keep_relay_gains) {
    PIDGains saved_relay = relay_gains_;

    // Reinitialize
    init(num_channels_, config_);

    if (keep_relay_gains && saved_relay.isValid()) {
        relay_gains_ = saved_relay;
        kp_ = relay_gains_.Kp;
        ki_ = relay_gains_.Ki;
        kd_ = relay_gains_.Kd;
        state_ = TunerState::ADAM_OPTIMIZING;
    }

    divergence_detected_ = false;
}

void HybridPIDTuner::triggerRetune() {
    if (logger_) {
        logger_->info("Retune triggered, starting relay experiment");
    }

    // Keep current gains as baseline
    PIDGains current;
    current.Kp = kp_;
    current.Ki = ki_;
    current.Kd = kd_;

    // Reset to relay mode
    state_ = TunerState::RELAY_EXPERIMENT;
    relay_start_ = std::chrono::steady_clock::now();
    relay_high_ = true;
    relay_output_ = config_.relay_amplitude;
    oscillation_count_ = 0;
    relay_crossings_.clear();
    relay_amplitudes_.clear();
}

std::tuple<float, float, float> HybridPIDTuner::getMetrics() const {
    float avg_error = current_error_avg_;
    float avg_control = control_norms_.empty() ? 0.0f :
        std::accumulate(control_norms_.begin(), control_norms_.end(), 0.0f) /
        control_norms_.size();

    // Improvement ratio: baseline / current (>1 means improvement)
    float improvement = (baseline_error_norm_ > 1e-6f && current_error_avg_ > 1e-6f) ?
        baseline_error_norm_ / current_error_avg_ : 1.0f;

    return std::make_tuple(avg_error, avg_control, improvement);
}

}  // namespace controllers
}  // namespace saguaro
