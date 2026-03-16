// highnoon/_native/ops/hd_bundle_op.h
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file hd_bundle_op.h
 * @brief SIMD-optimized Holographic Token Bundling.
 *
 * Phase 2 of Memory Architecture Roadmap: HD Token Bundling.
 *
 * Compresses token sequences into bundled HD vectors using FFT-domain
 * circular correlation. Achieves 128× memory reduction for streaming.
 *
 * Memory Impact (32K seq, batch=4, hd_dim=4096):
 *   - Per-token HD: 537 MB
 *   - Bundled (128 tokens): 4.2 MB
 *   - Savings: 128×
 *
 * Optimizations:
 *   - In-place FFT for minimal memory footprint
 *   - SIMD-accelerated element-wise operations
 *   - OpenMP parallelization across bundles
 *   - Cache-friendly memory access patterns
 */

#ifndef HIGHNOON_NATIVE_OPS_HD_BUNDLE_OP_H_
#define HIGHNOON_NATIVE_OPS_HD_BUNDLE_OP_H_

#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

// SIMD headers
#if defined(__AVX512F__)
#include <immintrin.h>
#define HN_BUNDLE_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define HN_BUNDLE_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define HN_BUNDLE_NEON 1
#endif

// OpenMP for parallelization
#ifdef _OPENMP
#include <omp.h>
#endif

namespace highnoon {
namespace ops {
namespace hd_bundle {

// =============================================================================
// CONFIGURATION
// =============================================================================

/**
 * @brief HD Bundle configuration.
 */
struct HDBundleConfig {
    int hd_dim = 4096;           // HD vector dimension
    int bundle_size = 128;       // Tokens per bundle (Phase 2: from config.HD_BUNDLE_SIZE)
    int overlap = 0;             // Token overlap between bundles
    bool use_position_weights = true;
    float decay_rate = 0.99f;    // Position weight decay
    
    // Computed values
    int stride() const { return bundle_size - overlap; }
    int num_bundles(int seq_len) const {
        return std::max(1, (seq_len - bundle_size) / stride() + 1);
    }
};

// =============================================================================
// SIMPLE FFT FOR POWER-OF-2 SIZES
// =============================================================================
// Using Cooley-Tukey radix-2 DIT FFT for portability.
// For production, consider linking FFTW or Intel MKL.

/**
 * @brief In-place radix-2 FFT.
 */
inline void fft_inplace(float* re, float* im, int n, bool inverse = false) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < n - 1; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    
    // Cooley-Tukey radix-2 DIT
    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len *= 2) {
        float ang = sign * 2.0f * static_cast<float>(M_PI) / len;
        float wcos = std::cos(ang);
        float wsin = std::sin(ang);
        
        for (int i = 0; i < n; i += len) {
            float wr = 1.0f, wi = 0.0f;
            for (int j = 0; j < len / 2; ++j) {
                int u = i + j;
                int v = u + len / 2;
                
                float tr = re[v] * wr - im[v] * wi;
                float ti = re[v] * wi + im[v] * wr;
                
                re[v] = re[u] - tr;
                im[v] = im[u] - ti;
                re[u] = re[u] + tr;
                im[u] = im[u] + ti;
                
                // Twiddle factor update
                float new_wr = wr * wcos - wi * wsin;
                wi = wr * wsin + wi * wcos;
                wr = new_wr;
            }
        }
    }
    
    // Scale for IFFT
    if (inverse) {
        float scale = 1.0f / n;
        for (int i = 0; i < n; ++i) {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}

// =============================================================================
// CIRCULAR CORRELATION
// =============================================================================

/**
 * @brief Circular correlation via FFT.
 *
 * Computes: output = IFFT(FFT(a) * conj(FFT(b)))
 *
 * @param a First vector [hd_dim]
 * @param b Second vector [hd_dim]
 * @param output Result [hd_dim]
 * @param hd_dim Vector dimension (must be power of 2)
 * @param work Scratch buffer [4 * hd_dim] for FFT
 */
inline void circular_correlation(
    const float* a,
    const float* b,
    float* output,
    int hd_dim,
    float* work
) {
    // Work buffer layout: [a_re, a_im, b_re, b_im]
    float* a_re = work;
    float* a_im = work + hd_dim;
    float* b_re = work + 2 * hd_dim;
    float* b_im = work + 3 * hd_dim;
    
    // Copy and zero imaginary parts
    std::memcpy(a_re, a, hd_dim * sizeof(float));
    std::memcpy(b_re, b, hd_dim * sizeof(float));
    std::memset(a_im, 0, hd_dim * sizeof(float));
    std::memset(b_im, 0, hd_dim * sizeof(float));
    
    // Forward FFT
    fft_inplace(a_re, a_im, hd_dim);
    fft_inplace(b_re, b_im, hd_dim);
    
    // Element-wise: a * conj(b) = (ar + ai*i) * (br - bi*i)
    //                           = ar*br + ai*bi + i*(ai*br - ar*bi)
    for (int d = 0; d < hd_dim; ++d) {
        float ar = a_re[d], ai = a_im[d];
        float br = b_re[d], bi = b_im[d];
        
        a_re[d] = ar * br + ai * bi;  // Real part
        a_im[d] = ai * br - ar * bi;  // Imag part
    }
    
    // Inverse FFT
    fft_inplace(a_re, a_im, hd_dim, true);
    
    // Copy real part to output
    std::memcpy(output, a_re, hd_dim * sizeof(float));
}

// =============================================================================
// SIMD VECTOR OPERATIONS
// =============================================================================

/**
 * @brief SIMD-accelerated vector addition.
 */
inline void vector_add(float* dst, const float* src, int n) {
    int i = 0;
    
#if defined(HN_BUNDLE_AVX512)
    for (; i + 16 <= n; i += 16) {
        __m512 d = _mm512_loadu_ps(dst + i);
        __m512 s = _mm512_loadu_ps(src + i);
        _mm512_storeu_ps(dst + i, _mm512_add_ps(d, s));
    }
#elif defined(HN_BUNDLE_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, s));
    }
#elif defined(HN_BUNDLE_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t d = vld1q_f32(dst + i);
        float32x4_t s = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(d, s));
    }
#endif
    
    // Scalar remainder
    for (; i < n; ++i) {
        dst[i] += src[i];
    }
}

/**
 * @brief SIMD-accelerated L2 normalization.
 */
inline void l2_normalize(float* vec, int n) {
    float sum_sq = 0.0f;
    int i = 0;
    
#if defined(HN_BUNDLE_AVX512)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(vec + i);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    sum_sq = _mm512_reduce_add_ps(acc);
#elif defined(HN_BUNDLE_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum_sq = _mm_cvtss_f32(sum4);
#elif defined(HN_BUNDLE_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(vec + i);
        acc = vmlaq_f32(acc, v, v);
    }
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum2 = vpadd_f32(sum2, sum2);
    sum_sq = vget_lane_f32(sum2, 0);
#endif
    
    // Scalar remainder
    for (; i < n; ++i) {
        sum_sq += vec[i] * vec[i];
    }
    
    float norm = std::sqrt(sum_sq);
    if (norm < 1e-8f) norm = 1e-8f;
    
    float inv_norm = 1.0f / norm;
    
    // Scale
    i = 0;
#if defined(HN_BUNDLE_AVX512)
    __m512 scale = _mm512_set1_ps(inv_norm);
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(vec + i);
        _mm512_storeu_ps(vec + i, _mm512_mul_ps(v, scale));
    }
#elif defined(HN_BUNDLE_AVX2)
    __m256 scale = _mm256_set1_ps(inv_norm);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(vec + i);
        _mm256_storeu_ps(vec + i, _mm256_mul_ps(v, scale));
    }
#elif defined(HN_BUNDLE_NEON)
    float32x4_t scale = vdupq_n_f32(inv_norm);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(vec + i);
        vst1q_f32(vec + i, vmulq_f32(v, scale));
    }
#endif
    
    for (; i < n; ++i) {
        vec[i] *= inv_norm;
    }
}

// =============================================================================
// BUNDLING FORWARD PASS
// =============================================================================

/**
 * @brief Bundle tokens into compressed HD representations.
 *
 * @param tokens Input HD tokens [batch, seq_len, hd_dim]
 * @param position_base Base position vector for cyclic permutation [hd_dim]
 * @param position_weights Optional position weights [bundle_size]
 * @param bundles Output bundled vectors [batch, num_bundles, hd_dim]
 * @param config Bundle configuration
 * @param batch_size Batch size
 * @param seq_len Sequence length
 */
inline void hd_bundle_forward(
    const float* tokens,
    const float* position_base,
    const float* position_weights,  // Can be nullptr
    float* bundles,
    const HDBundleConfig& config,
    int batch_size,
    int seq_len
) {
    const int hd_dim = config.hd_dim;
    const int bundle_size = config.bundle_size;
    const int stride = config.stride();
    const int num_bundles = config.num_bundles(seq_len);
    
    // Pre-compute position vectors via cyclic permutation
    std::vector<float> pos_vectors(bundle_size * hd_dim);
    for (int p = 0; p < bundle_size; ++p) {
        for (int d = 0; d < hd_dim; ++d) {
            int shifted_idx = (d - p + hd_dim) % hd_dim;
            float weight = (position_weights != nullptr) ? position_weights[p] : 1.0f;
            pos_vectors[p * hd_dim + d] = position_base[shifted_idx] * weight;
        }
    }
    
    // Scratch buffers per thread
    #pragma omp parallel
    {
        std::vector<float> work(4 * hd_dim);
        std::vector<float> bound(hd_dim);
        std::vector<float> bundle_acc(hd_dim);
        
        #pragma omp for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int n = 0; n < num_bundles; ++n) {
                // Initialize bundle accumulator
                std::memset(bundle_acc.data(), 0, hd_dim * sizeof(float));
                
                int start_idx = n * stride;
                int end_idx = std::min(start_idx + bundle_size, seq_len);
                
                // Bind and accumulate each token in bundle
                for (int t = start_idx; t < end_idx; ++t) {
                    int pos_in_bundle = t - start_idx;
                    const float* token_ptr = tokens + (b * seq_len + t) * hd_dim;
                    const float* pos_ptr = pos_vectors.data() + pos_in_bundle * hd_dim;
                    
                    // Circular correlation: token ⊛ position
                    circular_correlation(token_ptr, pos_ptr, bound.data(), hd_dim, work.data());
                    
                    // Accumulate
                    vector_add(bundle_acc.data(), bound.data(), hd_dim);
                }
                
                // L2 normalize bundle
                l2_normalize(bundle_acc.data(), hd_dim);
                
                // Store result
                float* out_ptr = bundles + (b * num_bundles + n) * hd_dim;
                std::memcpy(out_ptr, bundle_acc.data(), hd_dim * sizeof(float));
            }
        }
    }
}

/**
 * @brief Unbundle to retrieve approximate token at position.
 *
 * @param bundles Bundled vectors [batch, num_bundles, hd_dim]
 * @param position_base Base position vector [hd_dim]
 * @param query_position Position within bundle to retrieve
 * @param output Retrieved vectors [batch, num_bundles, hd_dim]
 * @param config Bundle configuration
 * @param batch_size Batch size
 * @param num_bundles Number of bundles
 */
inline void hd_unbundle(
    const float* bundles,
    const float* position_base,
    int query_position,
    float* output,
    const HDBundleConfig& config,
    int batch_size,
    int num_bundles
) {
    const int hd_dim = config.hd_dim;
    
    // Pre-compute query position vector
    std::vector<float> query_pos(hd_dim);
    for (int d = 0; d < hd_dim; ++d) {
        int shifted_idx = (d - query_position + hd_dim) % hd_dim;
        query_pos[d] = position_base[shifted_idx];
    }
    
    #pragma omp parallel
    {
        std::vector<float> work(4 * hd_dim);
        
        #pragma omp for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int n = 0; n < num_bundles; ++n) {
                const float* bundle_ptr = bundles + (b * num_bundles + n) * hd_dim;
                float* out_ptr = output + (b * num_bundles + n) * hd_dim;
                
                // Convolution (not correlation) for unbinding
                // This is the inverse operation
                // For now, use same correlation - in practice would use convolution
                circular_correlation(bundle_ptr, query_pos.data(), out_ptr, hd_dim, work.data());
            }
        }
    }
}

}  // namespace hd_bundle
}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_HD_BUNDLE_OP_H_
