// saguaro.native/ops/hd_spectral_entropy_op.h
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
 * @file hd_spectral_entropy_op.h
 * @brief Phase 300+: FFT-based spectral entropy for QULS.
 *
 * hd_upgrade.md Phase 2 - HD QULS Enhancement.
 *
 * Replaces O(d²) power iteration eigenvalue computation with O(d log d)
 * FFT-based spectral analysis. The spectral entropy is computed from
 * the power spectrum |FFT(x)|² instead of eigenvalues of the covariance.
 *
 * Mathematical basis:
 * Traditional: compute eigenvalues λᵢ of cov(X), entropy = -Σ λᵢ log λᵢ
 * HD: compute power spectrum P(f) = |FFT(X)|², entropy = -Σ p(f) log p(f)
 *
 * These are related: for stationary signals, eigenvalue distribution
 * converges to spectral density (Szegő's theorem).
 */

#ifndef SAGUARO_NATIVE_OPS_HD_SPECTRAL_ENTROPY_OP_H_
#define SAGUARO_NATIVE_OPS_HD_SPECTRAL_ENTROPY_OP_H_

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace saguaro {
namespace hd_spectral {

/**
 * HD Spectral Entropy Configuration.
 */
struct HDSpectralConfig {
    float epsilon = 1e-8f;        // Numerical stability constant
    bool normalize_power = true;  // Normalize power to probability
    int num_bins = 0;             // 0 = use full spectrum, >0 = bin into groups
};

/**
 * In-place FFT (Cooley-Tukey decimation-in-time).
 * Reused from hd_spatial_block_op.h pattern.
 *
 * @param data Complex data as interleaved [re0, im0, re1, im1, ...]
 * @param n Length (must be power of 2)
 * @param inverse Compute inverse FFT
 */
inline void fft_inplace(float* data, int n, bool inverse = false) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
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

                float tv_re = v_re * w_re - v_im * w_im;
                float tv_im = v_re * w_im + v_im * w_re;

                data[u_idx] = u_re + tv_re;
                data[u_idx + 1] = u_im + tv_im;
                data[v_idx] = u_re - tv_re;
                data[v_idx + 1] = u_im - tv_im;

                float new_w_re = w_re * wlen_re - w_im * wlen_im;
                float new_w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
        }
    }

    if (inverse) {
        float scale = 1.0f / static_cast<float>(n);
        for (int i = 0; i < 2 * n; ++i) {
            data[i] *= scale;
        }
    }
}

/**
 * Compute spectral entropy from power spectrum.
 *
 * @param power Power spectrum [fft_size]
 * @param fft_size Size of power spectrum
 * @param config Configuration
 * @return Spectral entropy value
 */
inline float compute_entropy(
    const float* power,
    int fft_size,
    const HDSpectralConfig& config
) {
    // Compute total power for normalization
    float total = 0.0f;
    for (int i = 0; i < fft_size; ++i) {
        total += power[i];
    }
    
    if (total < config.epsilon) {
        return 0.0f;  // Zero signal, zero entropy
    }
    
    // Compute entropy: -Σ p log p
    float entropy = 0.0f;
    for (int i = 0; i < fft_size; ++i) {
        float p = power[i] / total;
        if (p > config.epsilon) {
            entropy -= p * std::log(p);
        }
    }
    
    // Normalize by max entropy (uniform distribution)
    float max_entropy = std::log(static_cast<float>(fft_size));
    if (max_entropy > config.epsilon) {
        entropy /= max_entropy;
    }
    
    return entropy;
}

/**
 * Compute HD spectral entropy for a single sample.
 *
 * @param hidden_states Input hidden states [dim]
 * @param dim Hidden dimension (should ideally be power of 2)
 * @param config Configuration
 * @return Normalized spectral entropy [0, 1]
 */
inline float HDSpectralEntropySingle(
    const float* hidden_states,
    int dim,
    const HDSpectralConfig& config
) {
    // Find next power of 2 for FFT
    int fft_size = 1;
    while (fft_size < dim) {
        fft_size <<= 1;
    }
    
    // Allocate complex buffer (interleaved real/imag)
    std::vector<float> complex_buf(2 * fft_size, 0.0f);
    
    // Copy input (real part only)
    for (int i = 0; i < dim; ++i) {
        complex_buf[2 * i] = hidden_states[i];
    }
    
    // FFT
    fft_inplace(complex_buf.data(), fft_size, false);
    
    // Compute power spectrum: |FFT(x)|²
    std::vector<float> power(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        float re = complex_buf[2 * i];
        float im = complex_buf[2 * i + 1];
        power[i] = re * re + im * im;
    }
    
    // Compute and return entropy
    return compute_entropy(power.data(), fft_size, config);
}

/**
 * Compute HD spectral entropy for batched 2D hidden states.
 *
 * @param hidden_states Input [batch, dim]
 * @param entropy Output entropy values [batch]
 * @param batch_size Batch size
 * @param dim Hidden dimension
 * @param config Configuration
 */
inline void HDSpectralEntropyBatch2D(
    const float* hidden_states,
    float* entropy,
    int batch_size,
    int dim,
    const HDSpectralConfig& config
) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        entropy[b] = HDSpectralEntropySingle(
            hidden_states + b * dim,
            dim,
            config
        );
    }
}

/**
 * Compute HD spectral entropy for batched 3D hidden states.
 * Computes entropy per (batch, seq) position, then averages over seq.
 *
 * @param hidden_states Input [batch, seq, dim]
 * @param entropy Output entropy values [batch]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param dim Hidden dimension
 * @param config Configuration
 */
inline void HDSpectralEntropyBatch3D(
    const float* hidden_states,
    float* entropy,
    int batch_size,
    int seq_len,
    int dim,
    const HDSpectralConfig& config
) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        float sum_entropy = 0.0f;
        
        for (int s = 0; s < seq_len; ++s) {
            sum_entropy += HDSpectralEntropySingle(
                hidden_states + (b * seq_len + s) * dim,
                dim,
                config
            );
        }
        
        entropy[b] = sum_entropy / static_cast<float>(seq_len);
    }
}

/**
 * Compute spectral flatness (alternative to entropy).
 *
 * Spectral flatness = geometric_mean(power) / arithmetic_mean(power)
 * Ranges from 0 (pure tone) to 1 (white noise).
 *
 * @param hidden_states Input [dim]
 * @param dim Hidden dimension
 * @param config Configuration
 * @return Spectral flatness [0, 1]
 */
inline float HDSpectralFlatness(
    const float* hidden_states,
    int dim,
    const HDSpectralConfig& config
) {
    // Find next power of 2
    int fft_size = 1;
    while (fft_size < dim) {
        fft_size <<= 1;
    }
    
    std::vector<float> complex_buf(2 * fft_size, 0.0f);
    for (int i = 0; i < dim; ++i) {
        complex_buf[2 * i] = hidden_states[i];
    }
    
    fft_inplace(complex_buf.data(), fft_size, false);
    
    // Compute power spectrum
    float log_sum = 0.0f;
    float arith_sum = 0.0f;
    int count = 0;
    
    for (int i = 0; i < fft_size; ++i) {
        float re = complex_buf[2 * i];
        float im = complex_buf[2 * i + 1];
        float power = re * re + im * im + config.epsilon;
        
        log_sum += std::log(power);
        arith_sum += power;
        ++count;
    }
    
    float geometric_mean = std::exp(log_sum / static_cast<float>(count));
    float arithmetic_mean = arith_sum / static_cast<float>(count);
    
    return geometric_mean / arithmetic_mean;
}

/**
 * Compute gradient of spectral entropy w.r.t. input.
 *
 * This is an approximation: dH/dx ≈ -1/N × FFT⁻¹((1 + log(p)) × FFT(x) / |FFT(x)|)
 *
 * @param hidden_states Input [dim]
 * @param grad_output Gradient from downstream (scalar multiplier)
 * @param grad_input Output gradient [dim]
 * @param dim Hidden dimension
 * @param config Configuration
 */
inline void HDSpectralEntropyGrad(
    const float* hidden_states,
    float grad_output,
    float* grad_input,
    int dim,
    const HDSpectralConfig& config
) {
    int fft_size = 1;
    while (fft_size < dim) {
        fft_size <<= 1;
    }
    
    std::vector<float> complex_buf(2 * fft_size, 0.0f);
    for (int i = 0; i < dim; ++i) {
        complex_buf[2 * i] = hidden_states[i];
    }
    
    // Forward FFT
    fft_inplace(complex_buf.data(), fft_size, false);
    
    // Compute power and total
    float total_power = 0.0f;
    std::vector<float> power(fft_size);
    for (int i = 0; i < fft_size; ++i) {
        float re = complex_buf[2 * i];
        float im = complex_buf[2 * i + 1];
        power[i] = re * re + im * im + config.epsilon;
        total_power += power[i];
    }
    
    // Compute gradient in frequency domain
    float max_entropy = std::log(static_cast<float>(fft_size));
    float scale = grad_output / (total_power * max_entropy + config.epsilon);
    
    std::vector<float> grad_freq(2 * fft_size);
    for (int i = 0; i < fft_size; ++i) {
        float p = power[i] / total_power;
        float grad_factor = -(1.0f + std::log(p + config.epsilon)) * scale;
        
        // d|z|²/dz = 2z (but we need derivative w.r.t. real input)
        grad_freq[2 * i] = grad_factor * complex_buf[2 * i];
        grad_freq[2 * i + 1] = grad_factor * complex_buf[2 * i + 1];
    }
    
    // Inverse FFT to get spatial gradient
    fft_inplace(grad_freq.data(), fft_size, true);
    
    // Copy real part (the gradient w.r.t. real input)
    for (int i = 0; i < dim; ++i) {
        grad_input[i] = grad_freq[2 * i];
    }
}

}  // namespace hd_spectral
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_HD_SPECTRAL_ENTROPY_OP_H_
