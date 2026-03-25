// saguaro.native/ops/hd_kv_cache_op.h
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
 * @file hd_kv_cache_op.h
 * @brief Phase 300+: HD compressed KV cache for inference.
 *
 * hd_upgrade.md Phase 3 - HD Attention Layers.
 *
 * Compresses K/V vectors via holographic bundling for 8-16x cache reduction.
 * Key insight: consecutive K/V vectors can be bundled into a single HD
 * superposition, then retrieved via unbinding with position keys.
 *
 * This is particularly effective for long sequences where the cache
 * becomes a memory bottleneck.
 */

#ifndef SAGUARO_NATIVE_OPS_HD_KV_CACHE_OP_H_
#define SAGUARO_NATIVE_OPS_HD_KV_CACHE_OP_H_

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace saguaro {
namespace hd_kv {

/**
 * HD KV Cache Configuration.
 */
struct HDKVCacheConfig {
    int compression_ratio = 8;    // Tokens per HD bundle
    int hd_dim = 4096;            // HD vector dimension
    float epsilon = 1e-8f;        // Numerical stability
};

// Reuse FFT from other HD ops
inline void fft_kv(float* data, int n, bool inverse = false) {
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
 * Holographic bind for KV caching.
 */
inline void hd_bind_kv(
    const float* x,
    const float* key,
    float* result,
    int dim
) {
    std::vector<float> x_freq(2 * dim), k_freq(2 * dim);
    
    for (int i = 0; i < dim; ++i) {
        x_freq[2*i] = x[i]; x_freq[2*i + 1] = 0.0f;
        k_freq[2*i] = key[i]; k_freq[2*i + 1] = 0.0f;
    }
    
    fft_kv(x_freq.data(), dim, false);
    fft_kv(k_freq.data(), dim, false);
    
    for (int i = 0; i < dim; ++i) {
        float xr = x_freq[2*i], xi = x_freq[2*i + 1];
        float kr = k_freq[2*i], ki = k_freq[2*i + 1];
        x_freq[2*i] = xr * kr - xi * ki;
        x_freq[2*i + 1] = xr * ki + xi * kr;
    }
    
    fft_kv(x_freq.data(), dim, true);
    
    for (int i = 0; i < dim; ++i) {
        result[i] = x_freq[2*i];
    }
}

/**
 * Holographic unbind for retrieval.
 *
 * unbind(bundle, key) = IFFT(FFT(bundle) / FFT(key))
 *                     = IFFT(FFT(bundle) * conj(FFT(key)) / |FFT(key)|²)
 */
inline void hd_unbind_kv(
    const float* bundle,
    const float* key,
    float* result,
    int dim,
    float epsilon = 1e-8f
) {
    std::vector<float> b_freq(2 * dim), k_freq(2 * dim);
    
    for (int i = 0; i < dim; ++i) {
        b_freq[2*i] = bundle[i]; b_freq[2*i + 1] = 0.0f;
        k_freq[2*i] = key[i]; k_freq[2*i + 1] = 0.0f;
    }
    
    fft_kv(b_freq.data(), dim, false);
    fft_kv(k_freq.data(), dim, false);
    
    // Complex division: b / k = b * conj(k) / |k|²
    for (int i = 0; i < dim; ++i) {
        float br = b_freq[2*i], bi = b_freq[2*i + 1];
        float kr = k_freq[2*i], ki = k_freq[2*i + 1];
        float denom = kr * kr + ki * ki + epsilon;
        
        b_freq[2*i] = (br * kr + bi * ki) / denom;
        b_freq[2*i + 1] = (bi * kr - br * ki) / denom;
    }
    
    fft_kv(b_freq.data(), dim, true);
    
    for (int i = 0; i < dim; ++i) {
        result[i] = b_freq[2*i];
    }
}

/**
 * Compress KV cache using holographic bundling.
 *
 * Groups consecutive K/V vectors and bundles them with position keys.
 *
 * @param kv_cache Input KV cache [batch, heads, seq_len, head_dim]
 * @param pos_keys Position keys [max_seq, head_dim]
 * @param compressed Output compressed cache [batch, heads, num_bundles, head_dim]
 * @param batch_size Batch size
 * @param num_heads Number of heads
 * @param seq_len Sequence length
 * @param head_dim Head dimension
 * @param config Configuration
 */
inline void HDKVCacheCompress(
    const float* kv_cache,
    const float* pos_keys,
    float* compressed,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    const HDKVCacheConfig& config
) {
    const int ratio = config.compression_ratio;
    const int num_bundles = (seq_len + ratio - 1) / ratio;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int bundle_idx = 0; bundle_idx < num_bundles; ++bundle_idx) {
                const int start_pos = bundle_idx * ratio;
                float* bundle_ptr = compressed + 
                    ((b * num_heads + h) * num_bundles + bundle_idx) * head_dim;
                
                // Initialize bundle to zero
                std::memset(bundle_ptr, 0, head_dim * sizeof(float));
                
                std::vector<float> temp(head_dim);
                
                for (int offset = 0; offset < ratio; ++offset) {
                    const int pos = start_pos + offset;
                    if (pos >= seq_len) break;
                    
                    const float* kv_ptr = kv_cache +
                        ((b * num_heads + h) * seq_len + pos) * head_dim;
                    const float* key_ptr = pos_keys + pos * head_dim;
                    
                    // Bind KV with position key
                    hd_bind_kv(kv_ptr, key_ptr, temp.data(), head_dim);
                    
                    // Add to bundle (superposition)
                    for (int d = 0; d < head_dim; ++d) {
                        bundle_ptr[d] += temp[d];
                    }
                }
            }
        }
    }
}

/**
 * Decompress KV cache by unbinding specific positions.
 *
 * @param compressed Compressed cache [batch, heads, num_bundles, head_dim]
 * @param pos_keys Position keys [max_seq, head_dim]
 * @param decompressed Output KV cache [batch, heads, seq_len, head_dim]
 * @param batch_size Batch size
 * @param num_heads Number of heads
 * @param seq_len Target sequence length
 * @param head_dim Head dimension
 * @param config Configuration
 */
inline void HDKVCacheDecompress(
    const float* compressed,
    const float* pos_keys,
    float* decompressed,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    const HDKVCacheConfig& config
) {
    const int ratio = config.compression_ratio;
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int pos = 0; pos < seq_len; ++pos) {
                const int bundle_idx = pos / ratio;
                const float* bundle_ptr = compressed +
                    ((b * num_heads + h) * ((seq_len + ratio - 1) / ratio) + bundle_idx) * head_dim;
                const float* key_ptr = pos_keys + pos * head_dim;
                float* out_ptr = decompressed +
                    ((b * num_heads + h) * seq_len + pos) * head_dim;
                
                hd_unbind_kv(bundle_ptr, key_ptr, out_ptr, head_dim, config.epsilon);
            }
        }
    }
}

/**
 * Append new K/V to compressed cache (incremental update).
 *
 * @param compressed Current compressed cache [batch, heads, num_bundles, head_dim]
 * @param new_kv New K/V vector [batch, heads, head_dim]
 * @param pos_key Position key for new token [head_dim]
 * @param position Token position
 * @param batch_size Batch size
 * @param num_heads Number of heads
 * @param head_dim Head dimension
 * @param config Configuration
 */
inline void HDKVCacheAppend(
    float* compressed,
    const float* new_kv,
    const float* pos_key,
    int position,
    int batch_size,
    int num_heads,
    int head_dim,
    const HDKVCacheConfig& config
) {
    const int ratio = config.compression_ratio;
    const int bundle_idx = position / ratio;
    
    std::vector<float> bound(head_dim);
    
    #pragma omp parallel for collapse(2) firstprivate(bound)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            const float* kv_ptr = new_kv + (b * num_heads + h) * head_dim;
            
            // Bind with position key
            hd_bind_kv(kv_ptr, pos_key, bound.data(), head_dim);
            
            // Add to appropriate bundle
            // Note: Need to compute num_bundles from max expected seq_len
            // For now, assume it's pre-allocated
            float* bundle_ptr = compressed + 
                ((b * num_heads + h) * (position / ratio + 1) + bundle_idx) * head_dim;
            
            for (int d = 0; d < head_dim; ++d) {
                bundle_ptr[d] += bound[d];
            }
        }
    }
}

}  // namespace hd_kv
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_HD_KV_CACHE_OP_H_
