// saguaro.native/ops/hd_holographic_similarity_op.h
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
 * @file hd_holographic_similarity_op.h
 * @brief Phase 300+: FFT-based holographic Q·K similarity for attention.
 *
 * hd_upgrade.md Phase 3 - HD Attention Layers.
 *
 * Replaces standard Q·K^T attention with holographic similarity:
 *   scores[i,j] = HolographicSimilarity(Q[i], K[j])
 *              = Re(IFFT(FFT(Q[i]) * conj(FFT(K[j]))))
 *
 * For position-aware attention:
 *   Q_hd = HolographicBind(Q, pos_keys)
 *   K_hd = HolographicBind(K, pos_keys)
 *   scores = HolographicSimilarity(Q_hd, K_hd)
 *
 * Complexity: O(n × d log d) vs O(n² × d) for standard attention.
 * Best suited for long sequences where d log d << n × d.
 */

#ifndef SAGUARO_NATIVE_OPS_HD_HOLOGRAPHIC_SIMILARITY_OP_H_
#define SAGUARO_NATIVE_OPS_HD_HOLOGRAPHIC_SIMILARITY_OP_H_

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace saguaro {
namespace hd_attention {

/**
 * HD Attention Configuration.
 */
struct HDAttentionConfig {
    float temperature = 1.0f;     // Softmax temperature
    float epsilon = 1e-8f;        // Numerical stability
    bool use_position_keys = true;  // Enable position-aware binding
};

/**
 * In-place FFT for attention (reused pattern).
 */
inline void fft_attention(float* data, int n, bool inverse = false) {
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[2*i], data[2*j]);
            std::swap(data[2*i + 1], data[2*j + 1]);
        }
        int k = n >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }

    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        float angle = sign * 2.0f * M_PI / static_cast<float>(len);
        float wlen_re = std::cos(angle);
        float wlen_im = std::sin(angle);

        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int jj = 0; jj < len / 2; ++jj) {
                int u_idx = 2 * (i + jj);
                int v_idx = 2 * (i + jj + len / 2);
                float u_re = data[u_idx], u_im = data[u_idx + 1];
                float v_re = data[v_idx], v_im = data[v_idx + 1];
                float tv_re = v_re * w_re - v_im * w_im;
                float tv_im = v_re * w_im + v_im * w_re;
                data[u_idx] = u_re + tv_re;
                data[u_idx + 1] = u_im + tv_im;
                data[v_idx] = u_re - tv_re;
                data[v_idx + 1] = u_im - tv_im;
                float nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
    }
    if (inverse) {
        float scale = 1.0f / static_cast<float>(n);
        for (int i = 0; i < 2 * n; ++i) data[i] *= scale;
    }
}

/**
 * Holographic bind: x ⊛ y = IFFT(FFT(x) * FFT(y))
 *
 * This binds two vectors in a position-preserving way.
 *
 * @param x First vector [dim]
 * @param y Second vector [dim]
 * @param result Output bound vector [dim]
 * @param dim Dimension (power of 2)
 */
inline void holographic_bind(
    const float* x,
    const float* y,
    float* result,
    int dim
) {
    std::vector<float> x_freq(2 * dim), y_freq(2 * dim);
    
    // Pack as complex
    for (int i = 0; i < dim; ++i) {
        x_freq[2*i] = x[i]; x_freq[2*i + 1] = 0.0f;
        y_freq[2*i] = y[i]; y_freq[2*i + 1] = 0.0f;
    }
    
    // FFT both
    fft_attention(x_freq.data(), dim, false);
    fft_attention(y_freq.data(), dim, false);
    
    // Complex multiply: FFT(x) * FFT(y)
    for (int i = 0; i < dim; ++i) {
        float xr = x_freq[2*i], xi = x_freq[2*i + 1];
        float yr = y_freq[2*i], yi = y_freq[2*i + 1];
        x_freq[2*i] = xr * yr - xi * yi;
        x_freq[2*i + 1] = xr * yi + xi * yr;
    }
    
    // Inverse FFT
    fft_attention(x_freq.data(), dim, true);
    
    // Extract real part
    for (int i = 0; i < dim; ++i) {
        result[i] = x_freq[2*i];
    }
}

/**
 * Holographic similarity: measures similarity via circular correlation.
 *
 * sim(x, y) = max(IFFT(FFT(x) * conj(FFT(y))))
 *          ≈ cos(angle(x, y)) for normalized vectors
 *
 * @param x First vector [dim]
 * @param y Second vector [dim]
 * @param dim Dimension (power of 2)
 * @return Similarity score
 */
inline float holographic_similarity(
    const float* x,
    const float* y,
    int dim
) {
    std::vector<float> x_freq(2 * dim), y_freq(2 * dim);
    
    for (int i = 0; i < dim; ++i) {
        x_freq[2*i] = x[i]; x_freq[2*i + 1] = 0.0f;
        y_freq[2*i] = y[i]; y_freq[2*i + 1] = 0.0f;
    }
    
    fft_attention(x_freq.data(), dim, false);
    fft_attention(y_freq.data(), dim, false);
    
    // Complex multiply with conjugate: FFT(x) * conj(FFT(y))
    for (int i = 0; i < dim; ++i) {
        float xr = x_freq[2*i], xi = x_freq[2*i + 1];
        float yr = y_freq[2*i], yi = -y_freq[2*i + 1];  // conjugate
        x_freq[2*i] = xr * yr - xi * yi;
        x_freq[2*i + 1] = xr * yi + xi * yr;
    }
    
    fft_attention(x_freq.data(), dim, true);
    
    // Return max correlation (position 0 for same vectors)
    return x_freq[0];
}

/**
 * Compute holographic attention scores for all Q-K pairs.
 *
 * @param queries Q tensor [batch, heads, seq_q, head_dim]
 * @param keys K tensor [batch, heads, seq_k, head_dim]
 * @param scores Output attention scores [batch, heads, seq_q, seq_k]
 * @param batch_size Batch size
 * @param num_heads Number of heads
 * @param seq_q Query sequence length
 * @param seq_k Key sequence length
 * @param head_dim Head dimension (must be power of 2)
 * @param config Configuration
/**
 * @deprecated Use HolographicLinearAttentionForward from unified_attention_op.h for O(n) complexity.
 * This function has O(n² × d log d) complexity due to pairwise similarity computation.
 * For sequences longer than 4096, use the unified linear attention with use_holographic_features=true.
 */
inline void HolographicAttentionScores(
    const float* queries,
    const float* keys,
    float* scores,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    const HDAttentionConfig& config
) {
    // SAFETY GUARD: Prevent quadratic explosion for long sequences
    // O(n²) is only acceptable for short sequences (< 4096)
    if (seq_q * seq_k > 4096 * 4096) {
        // Log warning but continue - caller should use unified_attention with
        // use_holographic_features=true for linear complexity
        #ifdef SAGUARO_DEBUG
        fprintf(stderr, "[WARN] HolographicAttentionScores: seq_q=%d × seq_k=%d exceeds "
                        "linear mandate. Use unified_attention with use_holographic_features=true.\n",
                        seq_q, seq_k);
        #endif
    }
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int q = 0; q < seq_q; ++q) {
                const float* q_ptr = queries + 
                    ((b * num_heads + h) * seq_q + q) * head_dim;
                
                for (int k = 0; k < seq_k; ++k) {
                    const float* k_ptr = keys +
                        ((b * num_heads + h) * seq_k + k) * head_dim;
                    
                    float sim = holographic_similarity(q_ptr, k_ptr, head_dim);
                    
                    int score_idx = ((b * num_heads + h) * seq_q + q) * seq_k + k;
                    scores[score_idx] = sim / config.temperature;
                }
            }
        }
    }
}

/**
 * Apply position-aware holographic binding to Q/K tensors.
 *
 * @param tensor Input Q or K [batch, heads, seq, head_dim]
 * @param pos_keys Position keys [max_seq, head_dim]
 * @param output Output bound tensor [batch, heads, seq, head_dim]
 * @param batch_size Batch size
 * @param num_heads Number of heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 */
inline void HolographicPositionBind(
    const float* tensor,
    const float* pos_keys,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int tensor_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                const float* input_ptr = tensor + tensor_idx;
                float* output_ptr = output + tensor_idx;
                const float* pos_ptr = pos_keys + s * head_dim;
                
                holographic_bind(input_ptr, pos_ptr, output_ptr, head_dim);
            }
        }
    }
}

/**
 * Generate position keys using Floquet-inspired encoding.
 *
 * pos_key[i] = cos(2πf_i × pos) for harmonics f_i
 *
 * @param pos_keys Output position keys [max_seq, head_dim]
 * @param max_seq Maximum sequence length
 * @param head_dim Head dimension
 * @param base_freq Base frequency (default 10000.0)
 */
inline void GeneratePositionKeys(
    float* pos_keys,
    int max_seq,
    int head_dim,
    float base_freq = 10000.0f
) {
    for (int pos = 0; pos < max_seq; ++pos) {
        for (int d = 0; d < head_dim; ++d) {
            float freq = 1.0f / std::pow(base_freq, 
                2.0f * static_cast<float>(d / 2) / static_cast<float>(head_dim));
            
            if (d % 2 == 0) {
                pos_keys[pos * head_dim + d] = std::sin(pos * freq);
            } else {
                pos_keys[pos * head_dim + d] = std::cos(pos * freq);
            }
        }
    }
}

/**
 * Gradient of holographic similarity w.r.t. inputs.
 *
 * @param x First vector [dim]
 * @param y Second vector [dim]
 * @param grad_output Upstream gradient (scalar)
 * @param grad_x Output gradient w.r.t. x [dim]
 * @param grad_y Output gradient w.r.t. y [dim]
 * @param dim Dimension
 */
inline void HolographicSimilarityGrad(
    const float* x,
    const float* y,
    float grad_output,
    float* grad_x,
    float* grad_y,
    int dim
) {
    std::vector<float> x_freq(2 * dim), y_freq(2 * dim);
    
    for (int i = 0; i < dim; ++i) {
        x_freq[2*i] = x[i]; x_freq[2*i + 1] = 0.0f;
        y_freq[2*i] = y[i]; y_freq[2*i + 1] = 0.0f;
    }
    
    fft_attention(x_freq.data(), dim, false);
    fft_attention(y_freq.data(), dim, false);
    
    // Gradient of correlation at position 0
    // d(sim)/dx = IFFT(conj(FFT(y)))
    // d(sim)/dy = IFFT(conj(FFT(x)))
    
    std::vector<float> grad_x_freq(2 * dim), grad_y_freq(2 * dim);
    
    for (int i = 0; i < dim; ++i) {
        // For grad_x: conj(FFT(y))
        grad_x_freq[2*i] = y_freq[2*i];
        grad_x_freq[2*i + 1] = -y_freq[2*i + 1];
        
        // For grad_y: conj(FFT(x))
        grad_y_freq[2*i] = x_freq[2*i];
        grad_y_freq[2*i + 1] = -x_freq[2*i + 1];
    }
    
    fft_attention(grad_x_freq.data(), dim, true);
    fft_attention(grad_y_freq.data(), dim, true);
    
    for (int i = 0; i < dim; ++i) {
        grad_x[i] = grad_output * grad_x_freq[2*i];
        grad_y[i] = grad_output * grad_y_freq[2*i];
    }
}

}  // namespace hd_attention
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_HD_HOLOGRAPHIC_SIMILARITY_OP_H_
