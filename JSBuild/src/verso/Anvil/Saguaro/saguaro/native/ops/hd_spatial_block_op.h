// saguaro.native/ops/hd_spatial_block_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file hd_spatial_block_op.h
 * @brief Phase 200+: HD Spatial Block - Mamba SSM in HD space via FFT.
 *
 * SAGUARO_UPGRADE_ROADMAP.md Phase 2.2 - Block-level HD integration.
 *
 * This op replaces QMamba/SpatialBlock (block_type 0) when HD streaming is
 * enabled. It processes HD bundles directly using FFT-domain SSM operations
 * for O(D log D) complexity instead of O(D²).
 *
 * The core innovation is performing the Mamba selective scan in Fourier
 * domain, leveraging the convolution theorem: circular convolution in
 * spatial domain = element-wise multiplication in frequency domain.
 *
 * Shape: [B, L, hd_dim] -> [B, L, hd_dim]
 */

#ifndef SAGUARO_NATIVE_OPS_HD_SPATIAL_BLOCK_OP_H_
#define SAGUARO_NATIVE_OPS_HD_SPATIAL_BLOCK_OP_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

namespace saguaro {
namespace hd_spatial {

/**
 * HD Spatial Block Configuration.
 */
struct HDSpatialConfig {
    int hd_dim = 4096;           // Hyperdimensional embedding dimension
    int hidden_dim = 512;        // Internal hidden dimension
    int state_dim = 16;          // SSM state dimension (Mamba N)
    int conv_dim = 4;            // Conv1D kernel size
    float dt_min = 0.001f;       // Minimum discretization step
    float dt_max = 0.1f;         // Maximum discretization step
};

/**
 * In-place FFT butterfly for power-of-2 dimensions.
 * Uses Cooley-Tukey decimation-in-time.
 *
 * @param data Complex data as interleaved [re0, im0, re1, im1, ...]
 * @param n Length of data (must be power of 2)
 * @param inverse If true, compute inverse FFT
 */
inline void fft_butterfly(float* data, int n, bool inverse = false) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            // Swap pairs (data is interleaved real/imag)
            std::swap(data[2*i], data[2*j]);
            std::swap(data[2*i + 1], data[2*j + 1]);
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Butterfly computation
    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        float angle = sign * 2.0f * M_PI / static_cast<float>(len);
        float wlen_re = std::cos(angle);
        float wlen_im = std::sin(angle);

        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f;
            float w_im = 0.0f;

            for (int jj = 0; jj < len / 2; ++jj) {
                int u_idx = 2 * (i + jj);
                int v_idx = 2 * (i + jj + len / 2);

                float u_re = data[u_idx];
                float u_im = data[u_idx + 1];
                float v_re = data[v_idx];
                float v_im = data[v_idx + 1];

                // Twiddle factor multiplication: v' = v * w
                float tv_re = v_re * w_re - v_im * w_im;
                float tv_im = v_re * w_im + v_im * w_re;

                // Butterfly: u' = u + v', v' = u - v'
                data[u_idx] = u_re + tv_re;
                data[u_idx + 1] = u_im + tv_im;
                data[v_idx] = u_re - tv_re;
                data[v_idx + 1] = u_im - tv_im;

                // Update twiddle factor
                float new_w_re = w_re * wlen_re - w_im * wlen_im;
                float new_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
        }
    }

    // Scale for inverse FFT
    if (inverse) {
        float scale = 1.0f / static_cast<float>(n);
        for (int i = 0; i < 2 * n; ++i) {
            data[i] *= scale;
        }
    }
}

/**
 * Initialize Floquet phases with geometric frequency progression.
 * 
 * Similar to sinusoidal position encoding, creates a frequency scale
 * that enables infinite position extrapolation.
 *
 * @param phases Output phase array [hd_dim]
 * @param hd_dim Hyperdimensional size
 * @param base_frequency Base frequency (default 10000.0)
 */
template <typename T>
inline void InitFloquetPhases(T* phases, int hd_dim, T base_frequency = 10000.0f) {
    for (int d = 0; d < hd_dim; ++d) {
        // Geometric frequency progression (like sinusoidal PE)
        phases[d] = std::pow(base_frequency, static_cast<T>(-2 * d) / hd_dim);
    }
}

/**
 * Apply Floquet position-dependent phase shift in frequency domain.
 * 
 * For position t, each frequency bin d receives phase rotation:
 *   X'(d) = X(d) · exp(-i · t · θ(d))
 *
 * This enables infinite context extrapolation with only hd_dim parameters.
 * Complexity: O(D) per position, absorbed into FFT complexity.
 *
 * @param x_freq Complex input [2 * hd_dim] (interleaved re/im)
 * @param floquet_phases Learned phase per bin [hd_dim]
 * @param position Current position index
 * @param hd_dim Dimension
 */
inline void apply_floquet_position(
    float* x_freq,
    const float* floquet_phases,
    int position,
    int hd_dim
) {
    float pos = static_cast<float>(position);
    
    for (int d = 0; d < hd_dim; ++d) {
        float phase = pos * floquet_phases[d];
        float cos_p = std::cos(phase);
        float sin_p = std::sin(phase);
        
        float re = x_freq[2 * d];
        float im = x_freq[2 * d + 1];
        
        // Complex multiply by exp(-i * phase) = cos(phase) - i*sin(phase)
        x_freq[2 * d]     = re * cos_p + im * sin_p;
        x_freq[2 * d + 1] = im * cos_p - re * sin_p;
    }
}

/**
 * Backward pass for Floquet position encoding.
 * 
 * Gradient for Floquet phases:
 *   ∂L/∂θ(d) = Σ_t t · (grad_im·x_re - grad_re·x_im)
 *
 * @param grad_output Gradient of loss w.r.t. output [2 * hd_dim]
 * @param x_freq_original Original input before phase rotation [2 * hd_dim]
 * @param floquet_phases Learned phase per bin [hd_dim]
 * @param grad_floquet_phases Gradient accumulator [hd_dim] (add to)
 * @param position Current position index
 * @param hd_dim Dimension
 */
inline void apply_floquet_position_grad(
    const float* grad_output,
    const float* x_freq_original,
    const float* floquet_phases,
    float* grad_floquet_phases,
    int position,
    int hd_dim
) {
    float pos = static_cast<float>(position);
    
    for (int d = 0; d < hd_dim; ++d) {
        float phase = pos * floquet_phases[d];
        float cos_p = std::cos(phase);
        float sin_p = std::sin(phase);
        
        float x_re = x_freq_original[2 * d];
        float x_im = x_freq_original[2 * d + 1];
        
        float grad_re = grad_output[2 * d];
        float grad_im = grad_output[2 * d + 1];
        
        // After rotation: x_re_rot = x_re·cos_p + x_im·sin_p
        float x_re_rot = x_re * cos_p + x_im * sin_p;
        float x_im_rot = x_im * cos_p - x_re * sin_p;
        
        // Gradient contribution: t · (grad_im·x_re_rot - grad_re·x_im_rot)
        grad_floquet_phases[d] += pos * (grad_im * x_re_rot - grad_re * x_im_rot);
    }
}

/**
 * SSM state update in Fourier domain.
 *
 * For Mamba SSM: h_t = A_bar * h_{t-1} + B_bar * x_t
 * In frequency domain: H(w) = A_bar(w) * H_prev(w) + B_bar(w) * X(w)
 *
 * This leverages the convolution theorem for O(D) per-position update
 * in frequency domain vs O(D * state_dim) in spatial domain.
 *
 * @param h_freq_re State in frequency domain (real) [state_dim, hd_dim]
 * @param h_freq_im State in frequency domain (imag) [state_dim, hd_dim]
 * @param x_freq_re Input in frequency domain (real) [hd_dim]
 * @param x_freq_im Input in frequency domain (imag) [hd_dim]
 * @param a_log Log of decay rates [state_dim]
 * @param b_proj B projection weights [hd_dim, state_dim]
 * @param dt Discretization step [hd_dim]
 * @param config Configuration
 */
inline void ssm_freq_update(
    float* h_freq_re,
    float* h_freq_im,
    const float* x_freq_re,
    const float* x_freq_im,
    const float* a_log,
    const float* b_proj,
    const float* dt,
    const HDSpatialConfig& config
) {
    const int hd_dim = config.hd_dim;
    const int state_dim = config.state_dim;

    for (int s = 0; s < state_dim; ++s) {
        // A_bar = exp(dt * A) where A = -exp(a_log)
        float a = -std::exp(a_log[s]);

        for (int d = 0; d < hd_dim; ++d) {
            int idx = s * hd_dim + d;

            // Discretized A: A_bar = exp(dt * A)
            float dt_d = dt[d];
            float a_bar = std::exp(dt_d * a);

            // Get B value for this (d, s) pair
            // b_proj is [hd_dim, state_dim] so index is d * state_dim + s
            float b_val = b_proj[d * state_dim + s];

            // B_bar ≈ (1/A) * (A_bar - I) * B ≈ dt * B for small dt
            float b_bar = dt_d * b_val;

            // State update: h = a_bar * h + b_bar * x
            float h_re = h_freq_re[idx];
            float h_im = h_freq_im[idx];

            // Complex multiply a_bar (real scalar) with h
            h_re = a_bar * h_re + b_bar * x_freq_re[d];
            h_im = a_bar * h_im + b_bar * x_freq_im[d];

            h_freq_re[idx] = h_re;
            h_freq_im[idx] = h_im;
        }
    }
}

/**
 * Output computation in frequency domain.
 *
 * y = C * h in frequency domain becomes element-wise for diagonal C.
 *
 * @param h_freq_re State in frequency domain (real) [state_dim, hd_dim]
 * @param h_freq_im State in frequency domain (imag) [state_dim, hd_dim]
 * @param c_proj C projection weights [hd_dim, state_dim]
 * @param y_freq_re Output in frequency domain (real) [hd_dim]
 * @param y_freq_im Output in frequency domain (imag) [hd_dim]
 * @param config Configuration
 */
inline void ssm_freq_output(
    const float* h_freq_re,
    const float* h_freq_im,
    const float* c_proj,
    float* y_freq_re,
    float* y_freq_im,
    const HDSpatialConfig& config
) {
    const int hd_dim = config.hd_dim;
    const int state_dim = config.state_dim;

    // Initialize output to zero
    std::memset(y_freq_re, 0, hd_dim * sizeof(float));
    std::memset(y_freq_im, 0, hd_dim * sizeof(float));

    // Sum over state dimension: y = sum_s(C_s * h_s)
    for (int s = 0; s < state_dim; ++s) {
        for (int d = 0; d < hd_dim; ++d) {
            int idx = s * hd_dim + d;
            float c_val = c_proj[d * state_dim + s];

            y_freq_re[d] += c_val * h_freq_re[idx];
            y_freq_im[d] += c_val * h_freq_im[idx];
        }
    }
}

/**
 * HD Spatial Block Forward Pass.
 *
 * Processes HD bundles through Mamba SSM in Fourier domain.
 * Complexity: O(L * D log D) vs O(L * D * N) for spatial Mamba.
 *
 * @param hd_input Input HD bundles [batch, seq_len, hd_dim]
 * @param a_log Log decay rates [state_dim]
 * @param b_proj B projection [hd_dim, state_dim]
 * @param c_proj C projection [hd_dim, state_dim]
 * @param dt Discretization steps [hd_dim] (Phase 900.2: broadcasted internally per token)
 * @param skip_proj Skip connection projection [hd_dim, hd_dim]
 * @param floquet_phases Floquet position encoding phases [hd_dim] (NULL to disable)
 * @param hd_output Output HD bundles [batch, seq_len, hd_dim]
 * @param h_final Final states [batch, state_dim, hd_dim]
 * @param config Configuration
 * @param batch_size Batch size
 * @param seq_len Sequence length
 */
inline void HDSpatialForward(
    const float* hd_input,
    const float* a_log,
    const float* b_proj,
    const float* c_proj,
    const float* dt,
    const float* skip_proj,
    const float* floquet_phases,
    float* hd_output,
    float* h_final,
    const HDSpatialConfig& config,
    int batch_size,
    int seq_len
) {
    const int hd_dim = config.hd_dim;
    const int state_dim = config.state_dim;

    // Per-sample scratch space for FFT (complex interleaved)
    std::vector<float> x_freq(2 * hd_dim);
    std::vector<float> y_freq(2 * hd_dim);
    std::vector<float> h_freq_re(state_dim * hd_dim, 0.0f);
    std::vector<float> h_freq_im(state_dim * hd_dim, 0.0f);

    for (int b = 0; b < batch_size; ++b) {
        // Reset state for each batch
        std::fill(h_freq_re.begin(), h_freq_re.end(), 0.0f);
        std::fill(h_freq_im.begin(), h_freq_im.end(), 0.0f);

        for (int t = 0; t < seq_len; ++t) {
            const float* x_t = hd_input + (b * seq_len + t) * hd_dim;
            float* y_t = hd_output + (b * seq_len + t) * hd_dim;

            // Pack input as complex (real part only, imag = 0)
            for (int d = 0; d < hd_dim; ++d) {
                x_freq[2 * d] = x_t[d];
                x_freq[2 * d + 1] = 0.0f;
            }

            // FFT: x -> X(w)
            fft_butterfly(x_freq.data(), hd_dim, false);

            // Apply Floquet position encoding in frequency domain (Phase 700)
            // X_pos(ω) = X(ω) · exp(-i·t·θ(ω)) for infinite context extrapolation
            if (floquet_phases != nullptr) {
                apply_floquet_position(x_freq.data(), floquet_phases, t, hd_dim);
            }

            // Extract real/imag components
            std::vector<float> x_re(hd_dim), x_im(hd_dim);
            for (int d = 0; d < hd_dim; ++d) {
                x_re[d] = x_freq[2 * d];
                x_im[d] = x_freq[2 * d + 1];
            }

            // SSM update in frequency domain
            // Phase 900.2: dt is now 1D [hd_dim], used directly (same for all timesteps)
            ssm_freq_update(
                h_freq_re.data(), h_freq_im.data(),
                x_re.data(), x_im.data(),
                a_log, b_proj, dt,  // 1D dt
                config
            );

            // Compute output in frequency domain
            std::vector<float> y_re(hd_dim), y_im(hd_dim);
            ssm_freq_output(
                h_freq_re.data(), h_freq_im.data(),
                c_proj, y_re.data(), y_im.data(),
                config
            );

            // Pack for IFFT
            for (int d = 0; d < hd_dim; ++d) {
                y_freq[2 * d] = y_re[d];
                y_freq[2 * d + 1] = y_im[d];
            }

            // IFFT: Y(w) -> y
            fft_butterfly(y_freq.data(), hd_dim, true);

            // Extract real part (output) and add skip connection
            for (int d = 0; d < hd_dim; ++d) {
                float ssm_out = y_freq[2 * d];

                // Skip connection: y = SSM(x) + skip_proj @ x
                float skip_sum = 0.0f;
                if (skip_proj != nullptr) {
                    for (int dd = 0; dd < hd_dim; ++dd) {
                        skip_sum += skip_proj[d * hd_dim + dd] * x_t[dd];
                    }
                } else {
                    skip_sum = x_t[d];  // Identity skip
                }

                y_t[d] = ssm_out + skip_sum;
            }
        }

        // Copy final state
        if (h_final != nullptr) {
            std::memcpy(
                h_final + b * state_dim * hd_dim,
                h_freq_re.data(),
                state_dim * hd_dim * sizeof(float)
            );
        }
    }
}

/**
 * HD Spatial Block Backward Pass (Gradient Computation).
 *
 * Computes gradients for all learnable parameters via BPTT through
 * the frequency-domain SSM.
 *
 * @param grad_output Gradient from downstream [batch, seq_len, hd_dim]
 * @param hd_input Forward pass input [batch, seq_len, hd_dim]
 * @param a_log Log decay rates [state_dim]
 * @param b_proj B projection [hd_dim, state_dim]
 * @param c_proj C projection [hd_dim, state_dim]
 * @param dt Discretization steps [hd_dim] (Phase 900.2: 1D)
 * @param skip_proj Skip projection [hd_dim, hd_dim]
 * @param floquet_phases Floquet position encoding phases [hd_dim] (NULL if disabled)
 * @param grad_input Gradient w.r.t. input [batch, seq_len, hd_dim]
 * @param grad_a_log Gradient w.r.t. a_log [state_dim]
 * @param grad_b_proj Gradient w.r.t. B [hd_dim, state_dim]
 * @param grad_c_proj Gradient w.r.t. C [hd_dim, state_dim]
 * @param grad_dt Gradient w.r.t. dt [hd_dim] (Phase 900.2: accumulated across timesteps)
 * @param grad_skip Gradient w.r.t. skip [hd_dim, hd_dim]
 * @param grad_floquet Gradient w.r.t. floquet_phases [hd_dim] (NULL if disabled)
 * @param config Configuration
 * @param batch_size Batch size
 * @param seq_len Sequence length
 */
inline void HDSpatialBackward(
    const float* grad_output,
    const float* hd_input,
    const float* a_log,
    const float* b_proj,
    const float* c_proj,
    const float* dt,
    const float* skip_proj,
    const float* floquet_phases,
    float* grad_input,
    float* grad_a_log,
    float* grad_b_proj,
    float* grad_c_proj,
    float* grad_dt,
    float* grad_skip,
    float* grad_floquet,
    const HDSpatialConfig& config,
    int batch_size,
    int seq_len
) {
    const int hd_dim = config.hd_dim;
    const int state_dim = config.state_dim;

    // Zero-initialize gradients
    std::memset(grad_a_log, 0, state_dim * sizeof(float));
    std::memset(grad_b_proj, 0, hd_dim * state_dim * sizeof(float));
    std::memset(grad_c_proj, 0, hd_dim * state_dim * sizeof(float));
    // Phase 900.2: grad_dt is now [hd_dim] (accumulated across timesteps)
    std::memset(grad_dt, 0, hd_dim * sizeof(float));
    if (grad_skip != nullptr) {
        std::memset(grad_skip, 0, hd_dim * hd_dim * sizeof(float));
    }
    if (grad_floquet != nullptr) {
        std::memset(grad_floquet, 0, hd_dim * sizeof(float));
    }

    // BPTT: iterate backwards through time
    for (int b = 0; b < batch_size; ++b) {
        // Adjoint state (gradient w.r.t. hidden state)
        std::vector<float> grad_h_re(state_dim * hd_dim, 0.0f);
        std::vector<float> grad_h_im(state_dim * hd_dim, 0.0f);

        for (int t = seq_len - 1; t >= 0; --t) {
            const float* x_t = hd_input + (b * seq_len + t) * hd_dim;
            const float* g_y = grad_output + (b * seq_len + t) * hd_dim;
            float* g_x = grad_input + (b * seq_len + t) * hd_dim;

            // Skip connection gradient: g_x += skip.T @ g_y
            for (int d = 0; d < hd_dim; ++d) {
                float sum = 0.0f;
                if (skip_proj != nullptr) {
                    for (int dd = 0; dd < hd_dim; ++dd) {
                        sum += skip_proj[dd * hd_dim + d] * g_y[dd];
                    }
                } else {
                    sum = g_y[d];  // Identity skip
                }
                g_x[d] = sum;
            }

            // Skip weight gradient: grad_skip += g_y @ x.T
            if (grad_skip != nullptr) {
                for (int i = 0; i < hd_dim; ++i) {
                    for (int j = 0; j < hd_dim; ++j) {
                        grad_skip[i * hd_dim + j] += g_y[i] * x_t[j];
                    }
                }
            }

            // Gradient through C: grad_c += g_y @ h.T, grad_h += C.T @ g_y
            for (int s = 0; s < state_dim; ++s) {
                for (int d = 0; d < hd_dim; ++d) {
                    int idx = s * hd_dim + d;
                    float c_val = c_proj[d * state_dim + s];

                    // grad_c_proj[d, s] += g_y[d] * h[s, d]
                    grad_c_proj[d * state_dim + s] += g_y[d] * grad_h_re[idx];

                    // grad_h[s, d] += c_proj[d, s] * g_y[d]
                    grad_h_re[idx] += c_val * g_y[d];
                }
            }

            // Gradient through SSM state update (simplified)
            // h = a_bar * h_prev + b_bar * x
            // grad_h_prev = a_bar * grad_h
            // grad_a_log += grad_h * h_prev * a_bar * dt * (-exp(a_log))
            // grad_b += grad_h * x
            // grad_x += b_bar * grad_h

            for (int s = 0; s < state_dim; ++s) {
                float a = -std::exp(a_log[s]);

                for (int d = 0; d < hd_dim; ++d) {
                    int idx = s * hd_dim + d;
                    // Phase 900.2: dt is now 1D [hd_dim]
                    float dt_d = dt[d];
                    float a_bar = std::exp(dt_d * a);
                    float b_val = b_proj[d * state_dim + s];
                    float b_bar = dt_d * b_val;

                    // grad_x[d] += b_bar * grad_h[s, d]
                    g_x[d] += b_bar * grad_h_re[idx];

                    // grad_b[d, s] += dt * grad_h[s, d] * x[d]
                    grad_b_proj[d * state_dim + s] += dt_d * grad_h_re[idx] * x_t[d];

                    // grad_dt[d] += grad_h[s, d] * (a * a_bar * h_prev + b * x)
                    // Phase 900.2: dt is 1D, accumulate gradient
                    grad_dt[d] += grad_h_re[idx] * b_val * x_t[d];

                    // Propagate gradient to previous timestep
                    grad_h_re[idx] *= a_bar;
                }
            }
        }
    }
}

}  // namespace hd_spatial
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_HD_SPATIAL_BLOCK_OP_H_
