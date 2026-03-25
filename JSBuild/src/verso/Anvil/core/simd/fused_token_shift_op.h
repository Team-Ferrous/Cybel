// highnoon/_native/ops/fused_token_shift_op.h
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
 * @file fused_token_shift_op.h
 * @brief RWKV-6 style data-dependent token shifting helpers.
 *
 * This header provides helper functions for token shift operations.
 * Uses UNIQUE function names to avoid ODR violation with fused_min_gru_op.h
 *
 * Functions use the token_shift_ prefix to ensure uniqueness.
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_TOKEN_SHIFT_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_TOKEN_SHIFT_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

// SIMD intrinsics for cross-architecture vectorization
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace highnoon {
namespace ops {

// =============================================================================
// TOKEN SHIFT SIMD HELPERS
// All functions have token_shift_ prefix to avoid ODR violation with min_gru
// =============================================================================

/**
 * @brief In-place sigmoid using exact computation (scalar fallback).
 * Uses token_shift_ prefix to avoid ODR collision.
 */
inline void token_shift_sigmoid_inplace(float* data, int64_t size) {
    int64_t i = 0;
    
#if defined(__AVX512F__)
    // AVX512: 16-wide SIMD - use exact computation via exp
    for (; i + 16 <= size; i += 16) {
        for (int j = 0; j < 16; ++j) {
            data[i + j] = 1.0f / (1.0f + std::exp(-data[i + j]));
        }
    }
#elif defined(__AVX2__)
    // AVX2: 8-wide SIMD - use exact computation via exp
    for (; i + 8 <= size; i += 8) {
        for (int j = 0; j < 8; ++j) {
            data[i + j] = 1.0f / (1.0f + std::exp(-data[i + j]));
        }
    }
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

/**
 * @brief Token mixing: out = gate * x + (1 - gate) * prev
 */
inline void token_shift_mix(
    const float* gate, const float* x, const float* prev,
    float* out, int64_t size) {
    int64_t i = 0;
    
#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= size; i += 16) {
        __m512 g = _mm512_loadu_ps(&gate[i]);
        __m512 xv = _mm512_loadu_ps(&x[i]);
        __m512 pv = _mm512_loadu_ps(&prev[i]);
        __m512 one_minus_g = _mm512_sub_ps(one, g);
        __m512 term1 = _mm512_mul_ps(g, xv);
        __m512 term2 = _mm512_mul_ps(one_minus_g, pv);
        __m512 result = _mm512_add_ps(term1, term2);
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= size; i += 8) {
        __m256 g = _mm256_loadu_ps(&gate[i]);
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 pv = _mm256_loadu_ps(&prev[i]);
        __m256 one_minus_g = _mm256_sub_ps(one, g);
        __m256 term1 = _mm256_mul_ps(g, xv);
        __m256 term2 = _mm256_mul_ps(one_minus_g, pv);
        __m256 result = _mm256_add_ps(term1, term2);
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t g = vld1q_f32(&gate[i]);
        float32x4_t xv = vld1q_f32(&x[i]);
        float32x4_t pv = vld1q_f32(&prev[i]);
        float32x4_t one_minus_g = vsubq_f32(one, g);
        float32x4_t term1 = vmulq_f32(g, xv);
        float32x4_t term2 = vmulq_f32(one_minus_g, pv);
        float32x4_t result = vaddq_f32(term1, term2);
        vst1q_f32(&out[i], result);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        out[i] = gate[i] * x[i] + (1.0f - gate[i]) * prev[i];
    }
}

/**
 * @brief Element-wise multiply: out = a * b
 */
inline void token_shift_mul(
    const float* a, const float* b,
    float* out, int64_t size) {
    int64_t i = 0;
    
#if defined(__AVX512F__)
    for (; i + 16 <= size; i += 16) {
        __m512 av = _mm512_loadu_ps(&a[i]);
        __m512 bv = _mm512_loadu_ps(&b[i]);
        __m512 result = _mm512_mul_ps(av, bv);
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 av = _mm256_loadu_ps(&a[i]);
        __m256 bv = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_mul_ps(av, bv);
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t av = vld1q_f32(&a[i]);
        float32x4_t bv = vld1q_f32(&b[i]);
        float32x4_t result = vmulq_f32(av, bv);
        vst1q_f32(&out[i], result);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

// =============================================================================
// BACKWARD COMPATIBILITY ALIASES
// Map old names to new unique names so callers don't need to change
// =============================================================================

#define simd_sigmoid_inplace token_shift_sigmoid_inplace
#define simd_token_mix token_shift_mix
#define simd_mul token_shift_mul

// =============================================================================
// ENHANCEMENT 1: SIMPLIFIED TOKEN SHIFT (RWKV-7 STYLE)
// No input-dependent gate - uses only learned decay weights for 3x speedup
// =============================================================================

/**
 * @brief Simplified token mixing without input-dependent gating (RWKV-7 style).
 *
 * Uses only learned decay weights: gate = sigmoid(decay_weights)
 * This is ~3x faster than data-dependent gating.
 *
 * @param decay Pre-computed sigmoid(decay_weights) [embedding_dim]
 * @param x Current input [batch * seq * embedding_dim]
 * @param prev Previous input [batch * seq * embedding_dim]
 * @param out Output [batch * seq * embedding_dim]
 * @param batch_seq Combined batch * seq dimension
 * @param embedding_dim Dimension per token
 */
inline void token_shift_simplified_mix(
    const float* decay, const float* x, const float* prev,
    float* out, int64_t batch_seq, int64_t embedding_dim) {
    
    #pragma omp parallel for
    for (int64_t t = 0; t < batch_seq; ++t) {
        const int64_t offset = t * embedding_dim;
        int64_t d = 0;
        
#if defined(__AVX512F__)
        const __m512 one = _mm512_set1_ps(1.0f);
        for (; d + 16 <= embedding_dim; d += 16) {
            __m512 g = _mm512_loadu_ps(&decay[d]);
            __m512 xv = _mm512_loadu_ps(&x[offset + d]);
            __m512 pv = _mm512_loadu_ps(&prev[offset + d]);
            __m512 one_minus_g = _mm512_sub_ps(one, g);
            __m512 result = _mm512_fmadd_ps(g, xv, _mm512_mul_ps(one_minus_g, pv));
            _mm512_storeu_ps(&out[offset + d], result);
        }
#elif defined(__AVX2__)
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; d + 8 <= embedding_dim; d += 8) {
            __m256 g = _mm256_loadu_ps(&decay[d]);
            __m256 xv = _mm256_loadu_ps(&x[offset + d]);
            __m256 pv = _mm256_loadu_ps(&prev[offset + d]);
            __m256 one_minus_g = _mm256_sub_ps(one, g);
            __m256 result = _mm256_fmadd_ps(g, xv, _mm256_mul_ps(one_minus_g, pv));
            _mm256_storeu_ps(&out[offset + d], result);
        }
#elif defined(__ARM_NEON)
        const float32x4_t one = vdupq_n_f32(1.0f);
        for (; d + 4 <= embedding_dim; d += 4) {
            float32x4_t g = vld1q_f32(&decay[d]);
            float32x4_t xv = vld1q_f32(&x[offset + d]);
            float32x4_t pv = vld1q_f32(&prev[offset + d]);
            float32x4_t one_minus_g = vsubq_f32(one, g);
            float32x4_t term1 = vmulq_f32(g, xv);
            float32x4_t result = vmlaq_f32(term1, one_minus_g, pv);
            vst1q_f32(&out[offset + d], result);
        }
#endif
        for (; d < embedding_dim; ++d) {
            float g = decay[d];
            out[offset + d] = g * x[offset + d] + (1.0f - g) * prev[offset + d];
        }
    }
}

// =============================================================================
// ENHANCEMENT 2: FOURIER-ENHANCED TOKEN MIXING
// Real FFT for O(n log n) global context mixing
// Uses Cooley-Tukey in-place DFT for CPU efficiency
// =============================================================================

/**
 * @brief Bit reversal permutation for FFT.
 */
inline int64_t token_shift_bit_reverse(int64_t n, int64_t num_bits) {
    int64_t result = 0;
    for (int64_t i = 0; i < num_bits; ++i) {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    return result;
}

/**
 * @brief In-place Cooley-Tukey FFT (radix-2 DIT).
 *
 * Computes complex FFT in-place. Input is interleaved [re, im, re, im, ...].
 *
 * @param data Interleaved complex data [2 * n]
 * @param n Number of complex samples (must be power of 2)
 * @param inverse If true, compute inverse FFT
 */
inline void token_shift_fft_inplace(float* data, int64_t n, bool inverse = false) {
    // Bit reversal permutation
    int64_t num_bits = 0;
    int64_t temp = n;
    while (temp > 1) { temp >>= 1; ++num_bits; }
    
    for (int64_t i = 0; i < n; ++i) {
        int64_t j = token_shift_bit_reverse(i, num_bits);
        if (j > i) {
            std::swap(data[2 * i], data[2 * j]);
            std::swap(data[2 * i + 1], data[2 * j + 1]);
        }
    }
    
    // Cooley-Tukey DIT
    float sign = inverse ? 1.0f : -1.0f;
    for (int64_t len = 2; len <= n; len *= 2) {
        float angle = sign * 2.0f * 3.14159265358979323846f / static_cast<float>(len);
        float wlen_r = std::cos(angle);
        float wlen_i = std::sin(angle);
        
        for (int64_t i = 0; i < n; i += len) {
            float w_r = 1.0f, w_i = 0.0f;
            for (int64_t j = 0; j < len / 2; ++j) {
                int64_t u_idx = 2 * (i + j);
                int64_t v_idx = 2 * (i + j + len / 2);
                
                float u_r = data[u_idx];
                float u_i = data[u_idx + 1];
                float v_r = data[v_idx] * w_r - data[v_idx + 1] * w_i;
                float v_i = data[v_idx] * w_i + data[v_idx + 1] * w_r;
                
                data[u_idx] = u_r + v_r;
                data[u_idx + 1] = u_i + v_i;
                data[v_idx] = u_r - v_r;
                data[v_idx + 1] = u_i - v_i;
                
                // w *= wlen
                float new_w_r = w_r * wlen_r - w_i * wlen_i;
                w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = new_w_r;
            }
        }
    }
    
    // Scale for inverse FFT
    if (inverse) {
        float scale = 1.0f / static_cast<float>(n);
        for (int64_t i = 0; i < 2 * n; ++i) {
            data[i] *= scale;
        }
    }
}

/**
 * @brief Real-to-complex FFT for a single dimension.
 *
 * Converts real signal to complex spectrum using N/2 complex FFT trick.
 *
 * @param real_input Real input [seq_len]
 * @param complex_output Complex output [seq_len + 2] (interleaved re/im, includes DC and Nyquist)
 * @param seq_len Sequence length (must be power of 2)
 */
inline void token_shift_rfft(const float* real_input, float* complex_output, int64_t seq_len) {
    // Pack real input into complex pairs for N/2 FFT
    int64_t half_n = seq_len / 2;
    std::vector<float> temp(seq_len);  // Interleaved complex for N/2 FFT
    
    for (int64_t k = 0; k < half_n; ++k) {
        temp[2 * k] = real_input[2 * k];         // Even samples as real
        temp[2 * k + 1] = real_input[2 * k + 1]; // Odd samples as imag
    }
    
    // Compute N/2 complex FFT
    token_shift_fft_inplace(temp.data(), half_n, false);
    
    // Unpack to full spectrum
    // DC component
    complex_output[0] = temp[0] + temp[1];  // X[0].re
    complex_output[1] = 0.0f;               // X[0].im = 0 for real signal
    
    // Nyquist component
    complex_output[seq_len] = temp[0] - temp[1];  // X[N/2].re
    complex_output[seq_len + 1] = 0.0f;           // X[N/2].im = 0
    
    // Other frequency bins
    for (int64_t k = 1; k < half_n; ++k) {
        float z_r = temp[2 * k];
        float z_i = temp[2 * k + 1];
        float z_conj_r = temp[2 * (half_n - k)];
        float z_conj_i = -temp[2 * (half_n - k) + 1];
        
        float angle = -3.14159265358979323846f * static_cast<float>(k) / static_cast<float>(half_n);
        float w_r = std::cos(angle);
        float w_i = std::sin(angle);
        
        // X[k] = 0.5 * (Z[k] + conj(Z[N/2-k])) - 0.5j * W * (Z[k] - conj(Z[N/2-k]))
        float a_r = 0.5f * (z_r + z_conj_r);
        float a_i = 0.5f * (z_i + z_conj_i);
        float b_r = 0.5f * (z_r - z_conj_r);
        float b_i = 0.5f * (z_i - z_conj_i);
        
        // -j * W * (b_r + j*b_i) = -j * (w_r + j*w_i) * (b_r + j*b_i)
        //                       = -j * (w_r*b_r - w_i*b_i + j*(w_r*b_i + w_i*b_r))
        //                       = (w_r*b_i + w_i*b_r) - j*(w_r*b_r - w_i*b_i)
        float temp_r = w_r * b_i + w_i * b_r;
        float temp_i = -(w_r * b_r - w_i * b_i);
        
        complex_output[2 * k] = a_r + temp_r;
        complex_output[2 * k + 1] = a_i + temp_i;
    }
}

/**
 * @brief Complex-to-real inverse FFT.
 *
 * @param complex_input Complex spectrum [seq_len + 2] (interleaved)
 * @param real_output Real output [seq_len]
 * @param seq_len Sequence length (must be power of 2)
 */
inline void token_shift_irfft(const float* complex_input, float* real_output, int64_t seq_len) {
    int64_t half_n = seq_len / 2;
    std::vector<float> temp(seq_len);
    
    // Pack spectrum into N/2 complex for inverse
    for (int64_t k = 0; k < half_n; ++k) {
        float x_r = complex_input[2 * k];
        float x_i = complex_input[2 * k + 1];
        float x_conj_r = (k == 0) ? complex_input[seq_len] : complex_input[2 * (half_n - k)];
        float x_conj_i = (k == 0) ? 0.0f : -complex_input[2 * (half_n - k) + 1];
        
        float angle = 3.14159265358979323846f * static_cast<float>(k) / static_cast<float>(half_n);
        float w_r = std::cos(angle);
        float w_i = std::sin(angle);
        
        // Reverse the packing operation
        float a_r = x_r + x_conj_r;
        float a_i = x_i + x_conj_i;
        float b_r = x_r - x_conj_r;
        float b_i = x_i - x_conj_i;
        
        // j * W^(-k) * (b_r + j*b_i)
        float temp_r = -(w_r * b_i - w_i * b_r);
        float temp_i = w_r * b_r + w_i * b_i;
        
        temp[2 * k] = 0.5f * (a_r + temp_r);
        temp[2 * k + 1] = 0.5f * (a_i + temp_i);
    }
    
    // Compute N/2 inverse FFT
    token_shift_fft_inplace(temp.data(), half_n, true);
    
    // Unpack interleaved to real
    for (int64_t k = 0; k < half_n; ++k) {
        real_output[2 * k] = temp[2 * k];
        real_output[2 * k + 1] = temp[2 * k + 1];
    }
}

/**
 * @brief Apply learnable frequency filter in Fourier domain.
 *
 * Multiplies spectrum by complex filter weights.
 *
 * @param spectrum Complex spectrum [num_freqs * 2] (interleaved)
 * @param filter Complex filter weights [num_freqs * 2]
 * @param output Filtered spectrum [num_freqs * 2]
 * @param num_freqs Number of frequency bins (seq_len/2 + 1)
 */
inline void token_shift_freq_filter(
    const float* spectrum, const float* filter,
    float* output, int64_t num_freqs) {
    
    int64_t i = 0;
#if defined(__AVX2__)
    // Process 4 complex numbers (8 floats) at a time
    for (; i + 4 <= num_freqs; i += 4) {
        // Load spectrum: [re0, im0, re1, im1, re2, im2, re3, im3]
        __m256 spec = _mm256_loadu_ps(&spectrum[2 * i]);
        __m256 filt = _mm256_loadu_ps(&filter[2 * i]);
        
        // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // Rearrange for SIMD: spec=[a0,b0,a1,b1,...], filt=[c0,d0,c1,d1,...]
        // We need: [a0*c0-b0*d0, a0*d0+b0*c0, ...]
        
        // Shuffle to get [a0,a0,a1,a1,...] and [b0,b0,b1,b1,...]
        __m256 spec_rr = _mm256_shuffle_ps(spec, spec, 0xA0);  // [a,a,a,a,...]
        __m256 spec_ii = _mm256_shuffle_ps(spec, spec, 0xF5);  // [b,b,b,b,...]
        __m256 filt_ri = filt;                                  // [c,d,c,d,...]
        __m256 filt_ir = _mm256_shuffle_ps(filt, filt, 0xB1);  // [d,c,d,c,...]
        
        // ac, ad, ac, ad, ...
        __m256 ac_ad = _mm256_mul_ps(spec_rr, filt_ri);
        // bd, bc, bd, bc, ...
        __m256 bd_bc = _mm256_mul_ps(spec_ii, filt_ir);
        
        // Result: [ac-bd, ad+bc, ...]
        __m256 result = _mm256_addsub_ps(ac_ad, bd_bc);
        _mm256_storeu_ps(&output[2 * i], result);
    }
#endif
    // Scalar fallback
    for (; i < num_freqs; ++i) {
        float a = spectrum[2 * i];
        float b = spectrum[2 * i + 1];
        float c = filter[2 * i];
        float d = filter[2 * i + 1];
        output[2 * i] = a * c - b * d;
        output[2 * i + 1] = a * d + b * c;
    }
}

/**
 * @brief Blend time-domain and frequency-domain outputs.
 *
 * @param time_domain Time-domain token shift output
 * @param freq_domain Frequency-enhanced output
 * @param blend_gate Blending weights [embedding_dim]
 * @param output Blended output
 * @param size Total size (batch * seq * embed_dim)
 * @param embedding_dim Dimension per position
 */
inline void token_shift_freq_blend(
    const float* time_domain, const float* freq_domain,
    const float* blend_gate, float* output,
    int64_t batch_seq, int64_t embedding_dim) {
    
    #pragma omp parallel for
    for (int64_t t = 0; t < batch_seq; ++t) {
        const int64_t offset = t * embedding_dim;
        int64_t d = 0;
        
#if defined(__AVX2__)
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; d + 8 <= embedding_dim; d += 8) {
            __m256 g = _mm256_loadu_ps(&blend_gate[d]);
            __m256 tv = _mm256_loadu_ps(&time_domain[offset + d]);
            __m256 fv = _mm256_loadu_ps(&freq_domain[offset + d]);
            __m256 one_minus_g = _mm256_sub_ps(one, g);
            __m256 result = _mm256_fmadd_ps(g, tv, _mm256_mul_ps(one_minus_g, fv));
            _mm256_storeu_ps(&output[offset + d], result);
        }
#endif
        for (; d < embedding_dim; ++d) {
            float g = blend_gate[d];
            output[offset + d] = g * time_domain[offset + d] + (1.0f - g) * freq_domain[offset + d];
        }
    }
}

// =============================================================================
// ENHANCEMENT 3: HIERARCHICAL MULTI-SCALE DECAY
// Layer-position aware decay rates: earlier layers = faster decay (local)
//                                   later layers = slower decay (global)
// =============================================================================

/**
 * @brief Compute hierarchical decay based on layer position.
 *
 * decay_effective = base_decay ** (1.0 / (layer_pos + 1))
 *
 * @param base_decay Base decay weights [embedding_dim]
 * @param hierarchical_decay Output adjusted decay [embedding_dim]
 * @param embedding_dim Dimension
 * @param layer_position Layer index (0-indexed)
 * @param decay_factor Scaling factor for hierarchy
 */
inline void token_shift_compute_hierarchical_decay(
    const float* base_decay, float* hierarchical_decay,
    int64_t embedding_dim, int layer_position, float decay_factor) {
    
    float exponent = 1.0f / (static_cast<float>(layer_position) + decay_factor);
    
    int64_t d = 0;
#if defined(__AVX2__)
    for (; d + 8 <= embedding_dim; d += 8) {
        // Load base decay (already sigmoid-transformed)
        for (int j = 0; j < 8; ++j) {
            hierarchical_decay[d + j] = std::pow(base_decay[d + j], exponent);
        }
    }
#endif
    for (; d < embedding_dim; ++d) {
        hierarchical_decay[d] = std::pow(base_decay[d], exponent);
    }
}

// =============================================================================
// ENHANCEMENT 4: DELTA RULE INTEGRATION
// Gated Delta Networks: erase/write gates for precise memory control
// =============================================================================

/**
 * @brief Delta rule memory update: state = state * (1 - erase) + write * value
 *
 * @param state Current memory state [batch, state_dim]
 * @param erase Erase gate (sigmoid output) [batch, state_dim]
 * @param write Write gate (sigmoid output) [batch, state_dim]
 * @param value Value to write [batch, state_dim]
 * @param new_state Updated state [batch, state_dim]
 * @param size Total elements (batch * state_dim)
 */
inline void token_shift_delta_update(
    const float* state, const float* erase, const float* write,
    const float* value, float* new_state, int64_t size) {
    
    int64_t i = 0;
#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= size; i += 16) {
        __m512 s = _mm512_loadu_ps(&state[i]);
        __m512 e = _mm512_loadu_ps(&erase[i]);
        __m512 w = _mm512_loadu_ps(&write[i]);
        __m512 v = _mm512_loadu_ps(&value[i]);
        
        __m512 one_minus_e = _mm512_sub_ps(one, e);
        __m512 retained = _mm512_mul_ps(s, one_minus_e);
        __m512 written = _mm512_mul_ps(w, v);
        __m512 result = _mm512_add_ps(retained, written);
        _mm512_storeu_ps(&new_state[i], result);
    }
#elif defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= size; i += 8) {
        __m256 s = _mm256_loadu_ps(&state[i]);
        __m256 e = _mm256_loadu_ps(&erase[i]);
        __m256 w = _mm256_loadu_ps(&write[i]);
        __m256 v = _mm256_loadu_ps(&value[i]);
        
        __m256 one_minus_e = _mm256_sub_ps(one, e);
        __m256 retained = _mm256_mul_ps(s, one_minus_e);
        __m256 written = _mm256_mul_ps(w, v);
        __m256 result = _mm256_add_ps(retained, written);
        _mm256_storeu_ps(&new_state[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t s = vld1q_f32(&state[i]);
        float32x4_t e = vld1q_f32(&erase[i]);
        float32x4_t w = vld1q_f32(&write[i]);
        float32x4_t v = vld1q_f32(&value[i]);
        
        float32x4_t one_minus_e = vsubq_f32(one, e);
        float32x4_t retained = vmulq_f32(s, one_minus_e);
        float32x4_t written = vmulq_f32(w, v);
        float32x4_t result = vaddq_f32(retained, written);
        vst1q_f32(&new_state[i], result);
    }
#endif
    for (; i < size; ++i) {
        new_state[i] = state[i] * (1.0f - erase[i]) + write[i] * value[i];
    }
}

/**
 * @brief Gradient for delta rule update.
 *
 * Computes gradients for state, erase, write, and value.
 */
inline void token_shift_delta_update_grad(
    const float* grad_new_state,
    const float* state, const float* erase, const float* write, const float* value,
    float* grad_state, float* grad_erase, float* grad_write, float* grad_value,
    int64_t size) {
    
    int64_t i = 0;
#if defined(__AVX2__)
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    for (; i + 8 <= size; i += 8) {
        __m256 dL_dns = _mm256_loadu_ps(&grad_new_state[i]);
        __m256 s = _mm256_loadu_ps(&state[i]);
        __m256 e = _mm256_loadu_ps(&erase[i]);
        __m256 w = _mm256_loadu_ps(&write[i]);
        __m256 v = _mm256_loadu_ps(&value[i]);
        
        // grad_state = dL/dns * (1 - erase)
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 one_minus_e = _mm256_sub_ps(one, e);
        __m256 gs = _mm256_mul_ps(dL_dns, one_minus_e);
        _mm256_storeu_ps(&grad_state[i], gs);
        
        // grad_erase = dL/dns * (-state)
        __m256 ge = _mm256_mul_ps(dL_dns, _mm256_mul_ps(neg_one, s));
        _mm256_storeu_ps(&grad_erase[i], ge);
        
        // grad_write = dL/dns * value
        __m256 gw = _mm256_mul_ps(dL_dns, v);
        _mm256_storeu_ps(&grad_write[i], gw);
        
        // grad_value = dL/dns * write
        __m256 gv = _mm256_mul_ps(dL_dns, w);
        _mm256_storeu_ps(&grad_value[i], gv);
    }
#endif
    for (; i < size; ++i) {
        float dL_dns = grad_new_state[i];
        grad_state[i] = dL_dns * (1.0f - erase[i]);
        grad_erase[i] = dL_dns * (-state[i]);
        grad_write[i] = dL_dns * value[i];
        grad_value[i] = dL_dns * write[i];
    }
}

// =============================================================================
// ENHANCEMENT 5: MULTI-POSITION LOOK-AHEAD
// Access tokens at multiple shift distances [1, 2, 4, ...] directly
// =============================================================================

/**
 * @brief Create shifted versions at multiple distances.
 *
 * @param input Input tensor [batch, seq_len, embedding_dim]
 * @param shifts Output shifted tensors [num_distances, batch, seq_len, embedding_dim]
 * @param distances Shift distances [num_distances]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param embedding_dim Embedding dimension
 * @param num_distances Number of shift distances
 */
inline void token_shift_multi_distance(
    const float* input, float* shifts,
    const int* distances, int batch_size, int seq_len,
    int embedding_dim, int num_distances) {
    
    const int64_t seq_stride = embedding_dim;
    const int64_t batch_stride = static_cast<int64_t>(seq_len) * embedding_dim;
    const int64_t shift_stride = static_cast<int64_t>(batch_size) * batch_stride;
    
    #pragma omp parallel for collapse(2)
    for (int dist_idx = 0; dist_idx < num_distances; ++dist_idx) {
        for (int b = 0; b < batch_size; ++b) {
            int distance = distances[dist_idx];
            float* out_base = shifts + dist_idx * shift_stride + b * batch_stride;
            const float* in_base = input + b * batch_stride;
            
            for (int s = 0; s < seq_len; ++s) {
                int src_pos = s - distance;
                float* out_row = out_base + s * seq_stride;
                
                if (src_pos < 0) {
                    // Zero padding for positions before start
                    std::memset(out_row, 0, embedding_dim * sizeof(float));
                } else {
                    const float* in_row = in_base + src_pos * seq_stride;
                    std::memcpy(out_row, in_row, embedding_dim * sizeof(float));
                }
            }
        }
    }
}

/**
 * @brief Blend multiple shifted versions with position-dependent weights.
 *
 * @param shifts Shifted tensors [num_distances, batch, seq, embed_dim]
 * @param weights Blending weights [num_distances] (should sum to 1)
 * @param output Blended output [batch, seq, embed_dim]
 * @param batch_seq Combined batch * seq
 * @param embedding_dim Embedding dimension
 * @param num_distances Number of shift distances
 */
inline void token_shift_blend_distances(
    const float* shifts, const float* weights,
    float* output, int64_t batch_seq, int embedding_dim, int num_distances) {
    
    const int64_t token_size = embedding_dim;
    const int64_t shift_stride = batch_seq * embedding_dim;
    
    // Zero initialize output
    std::memset(output, 0, batch_seq * embedding_dim * sizeof(float));
    
    #pragma omp parallel for
    for (int64_t t = 0; t < batch_seq; ++t) {
        float* out_row = output + t * token_size;
        
        for (int dist_idx = 0; dist_idx < num_distances; ++dist_idx) {
            float w = weights[dist_idx];
            const float* shift_row = shifts + dist_idx * shift_stride + t * token_size;
            
            int64_t d = 0;
#if defined(__AVX2__)
            __m256 wv = _mm256_set1_ps(w);
            for (; d + 8 <= embedding_dim; d += 8) {
                __m256 sv = _mm256_loadu_ps(&shift_row[d]);
                __m256 ov = _mm256_loadu_ps(&out_row[d]);
                __m256 result = _mm256_fmadd_ps(wv, sv, ov);
                _mm256_storeu_ps(&out_row[d], result);
            }
#endif
            for (; d < embedding_dim; ++d) {
                out_row[d] += w * shift_row[d];
            }
        }
    }
}

/**
 * @brief Compute hierarchical decay gradient multiplier.
 */
inline void token_shift_compute_hierarchical_decay_grad(
    const float* base_decay, const float* hierarchical_decay,
    float* grad_mult, int64_t embedding_dim, int layer_position, float decay_factor) {
    
    float exponent = 1.0f / (static_cast<float>(layer_position) + decay_factor);
    
    for (int64_t d = 0; d < embedding_dim; ++d) {
        // df/dw = e * s^e * (1-s) = exponent * hierarchical_decay * (1 - base_decay)
        grad_mult[d] = exponent * hierarchical_decay[d] * (1.0f - base_decay[d]);
    }
}

/**
 * @brief Gradient for Multi-Position look-ahead.
 */
inline void token_shift_multi_distance_grad(
    const float* grad_output, const float* input,
    float* grad_input, float* grad_weights,
    const int* distances, const float* weights,
    int batch_size, int seq_len, int embedding_dim, int num_distances) {
    
    const int64_t seq_stride = embedding_dim;
    const int64_t batch_stride = static_cast<int64_t>(seq_len) * embedding_dim;
    
    // grad_input = sum_i(weight_i * shift_back(grad_output, distances[i]))
    std::memset(grad_input, 0, static_cast<int64_t>(batch_size) * batch_stride * sizeof(float));
    std::memset(grad_weights, 0, num_distances * sizeof(float));
    
    #pragma omp parallel
    {
        std::vector<float> local_grad_weights(num_distances, 0.0f);
        
        #pragma omp for collapse(2)
        for (int dist_idx = 0; dist_idx < num_distances; ++dist_idx) {
            for (int b = 0; b < batch_size; ++b) {
                int distance = distances[dist_idx];
                float w = weights[dist_idx];
                
                const float* g_out_base = grad_output + b * batch_stride;
                const float* in_base = input + b * batch_stride;
                float* g_in_base = grad_input + b * batch_stride;
                
                for (int s = 0; s < seq_len; ++s) {
                    int src_pos = s - distance;
                    if (src_pos < 0) continue;
                    
                    const float* g_out_row = g_out_base + s * seq_stride;
                    const float* in_row = in_base + src_pos * seq_stride;
                    float* g_in_row = g_in_base + src_pos * seq_stride;
                    
                    for (int d = 0; d < embedding_dim; ++d) {
                        float g = g_out_row[d];
                        // Weight gradient
                        local_grad_weights[dist_idx] += g * in_row[d];
                        // Input gradient
                        #pragma omp atomic
                        g_in_row[d] += w * g;
                    }
                }
            }
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < num_distances; ++i) {
                grad_weights[i] += local_grad_weights[i];
            }
        }
    }
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_TOKEN_SHIFT_OP_H_
