#ifndef SRC_CONTROLLERS_MAGLEV_CONTROLLER_H_
#define SRC_CONTROLLERS_MAGLEV_CONTROLLER_H_

#include <vector>
#include <string>
#include <Eigen/Dense>

/**
 * @file maglev_controller.h
 * @brief Ultra-precision maglev controller with feedforward + feedback control.
 *
 * Phase 8.4: Nanometer Precision Control
 *
 * This controller implements hybrid control for magnetic levitation systems:
 * - Feedforward: HNN-predicted magnetic field for trajectory tracking
 * - Feedback: PID loop for residual error correction
 * - Anti-windup: Integrator saturation to prevent overshoot
 *
 * Performance targets:
 * - Position accuracy: ± 5 nm RMS (1σ)
 * - Settling time: < 50 ms after 1 mm step command
 * - Disturbance rejection: > 60 dB at mechanical resonance
 *
 * Integration:
 * - State estimation: KalmanHNN (sensor fusion at 10 kHz)
 * - Vibration rejection: VibrationFilter (adaptive notch + HNN prediction)
 * - Hardware interface: DAC output to magnetic coil driver
 *
 * @author Michael B. Zimmerman
 * @date 2025-11-20
 * @copyright Verso Industries, Apache License 2.0
 */

class MaglevController {
public:
    /**
     * @brief Controller configuration structure.
     */
    struct Config {
        // PID gains
        float kp;  ///< Proportional gain
        float ki;  ///< Integral gain (with anti-windup)
        float kd;  ///< Derivative gain

        // Feedforward configuration
        bool enable_feedforward;  ///< Enable HNN-based feedforward
        float feedforward_gain;   ///< Feedforward output scaling

        // Anti-windup
        float integrator_min;  ///< Minimum integrator value
        float integrator_max;  ///< Maximum integrator value

        // Output limits
        float output_min;  ///< Minimum control output (magnetic field)
        float output_max;  ///< Maximum control output

        // Derivative filter (to reduce noise amplification)
        float derivative_filter_coeff;  ///< Low-pass filter: 0 = no filter, 1 = full filter

        // Sampling rate
        float sample_rate_hz;  ///< Control loop frequency (10 kHz default)

        /**
         * @brief Default constructor with safe defaults.
         */
        Config()
            : kp(1000.0f),
              ki(500.0f),
              kd(100.0f),
              enable_feedforward(true),
              feedforward_gain(1.0f),
              integrator_min(-10.0f),
              integrator_max(10.0f),
              output_min(-100.0f),
              output_max(100.0f),
              derivative_filter_coeff(0.1f),
              sample_rate_hz(10000.0f) {}
    };

    /**
     * @brief Default constructor.
     */
    MaglevController();

    /**
     * @brief Initialize controller with configuration.
     * @param config Controller configuration parameters.
     */
    void init(const Config& config);

    /**
     * @brief Compute control output for one timestep.
     *
     * Hybrid control law:
     *   u_ff = HNN_feedforward(x, x_ref, u_ref)  [optional]
     *   e = x_ref - x
     *   u_fb = kp*e + ki*integral(e) + kd*de/dt
     *   u = u_ff + u_fb
     *
     * @param x_ref Reference state [position, velocity, acceleration].
     * @param x_measured Measured state from KalmanHNN.
     * @param u_ref Reference control (feedforward nominal).
     * @param hnn_feedforward HNN-predicted control (optional, can be zero).
     * @return Control output (magnetic field command).
     */
    float compute(
        const std::vector<float>& x_ref,
        const std::vector<float>& x_measured,
        float u_ref,
        float hnn_feedforward
    );

    /**
     * @brief Reset controller state (integrator, derivative filter).
     */
    void reset();

    /**
     * @brief Get current integrator value (for monitoring).
     * @return Integral error accumulator.
     */
    float getIntegrator() const;

    /**
     * @brief Get last control output (for logging/telemetry).
     * @return Last computed control value.
     */
    float getLastOutput() const;

    /**
     * @brief Get settling time estimate (time since error < threshold).
     * @param threshold Position error threshold (nm).
     * @return Settling time in seconds, or -1 if not settled.
     */
    float getSettlingTime(float threshold = 5.0f) const;

private:
    // Configuration
    Config config_;

    // Controller state
    float integral_;           ///< Integral error accumulator
    float prev_error_;         ///< Previous position error (for derivative)
    float filtered_derivative_;///< Low-pass filtered derivative
    float last_output_;        ///< Last control output

    // Settling time tracking
    float settling_start_time_;///< Time when error first crossed threshold
    float current_time_;       ///< Current simulation time
    bool is_settling_;         ///< Flag: currently within settling threshold

    /**
     * @brief Apply anti-windup to integrator.
     * @param value Integrator value to clamp.
     * @return Clamped value within [integrator_min, integrator_max].
     */
    float applyAntiWindup(float value) const;

    /**
     * @brief Clamp control output to physical limits.
     * @param value Control output to clamp.
     * @return Clamped value within [output_min, output_max].
     */
    float clampOutput(float value) const;
};

#endif // SRC_CONTROLLERS_MAGLEV_CONTROLLER_H_
