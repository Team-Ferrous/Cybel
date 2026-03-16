// highnoon/_native/ops/hd_gradient_compression_op.h
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
 * @file hd_gradient_compression_op.h
 * @brief Phase 300+: HD-native frequency-domain gradient compression.
 *
 * Integrates GaLore concepts into hyperdimensional architecture using
 * FFT-based frequency filtering instead of SVD. Since HD embeddings
 * already operate in frequency domain (via FFT circular convolution),
 * gradient compression should also use frequency-domain filtering.
 *
 * Key insight: Low-rank gradient approximation can be expressed as
 * keeping top-K frequency components after FFT, which is mathematically
 * equivalent to projecting gradients onto a frequency subspace.
 *
 * Complexity: O(d log d) for FFT + O(bandwidth) for filtering
 * Memory: Reduces gradient storage by factor of (d / bandwidth)
 *
 * Benefits over SRHT:
 * - Native to HD architecture (same FFT infrastructure)
 * - Preserves phase information important for holographic binding
 * - More interpretable compression (frequency domain)
 */

#ifndef HIGHNOON_NATIVE_OPS_HD_GRADIENT_COMPRESSION_OP_H_
#define HIGHNOON_NATIVE_OPS_HD_GRADIENT_COMPRESSION_OP_H_

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <complex>

namespace hsmn {
namespace hd_grad_compress {

/**
 * Configuration for HD gradient compression.
 */
struct HDGradCompressConfig {
    int bandwidth = 256;         // Number of frequency components to keep
    bool preserve_dc = true;     // Always keep DC component (index 0)
    bool preserve_nyquist = true; // Keep Nyquist frequency
    float scale = 1.0f;          // Output scaling factor
};

/**
 * Compute complex magnitude for sorting.
 */
inline float complex_magnitude(float real, float imag) {
    return std::sqrt(real * real + imag * imag);
}

/**
 * FFT Cooley-Tukey radix-2 implementation.
 * O(n log n) complexity.
 *
 * @param real Real part [dim]
 * @param imag Imaginary part [dim]
 * @param dim Dimension (must be power of 2)
 * @param inverse If true, compute inverse FFT
 */
inline void fft_radix2(
    float* real,
    float* imag,
    int dim,
    bool inverse = false
) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < dim - 1; ++i) {
        int bit = dim >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // Cooley-Tukey iterative FFT
    const float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= dim; len <<= 1) {
        float angle = sign * 2.0f * M_PI / len;
        float w_real = std::cos(angle);
        float w_imag = std::sin(angle);

        for (int i = 0; i < dim; i += len) {
            float cur_real = 1.0f;
            float cur_imag = 0.0f;
            int half = len >> 1;

            for (int k = 0; k < half; ++k) {
                // Butterfly operation
                float u_real = real[i + k];
                float u_imag = imag[i + k];

                float t_real = cur_real * real[i + k + half] - cur_imag * imag[i + k + half];
                float t_imag = cur_real * imag[i + k + half] + cur_imag * real[i + k + half];

                real[i + k] = u_real + t_real;
                imag[i + k] = u_imag + t_imag;
                real[i + k + half] = u_real - t_real;
                imag[i + k + half] = u_imag - t_imag;

                // Update twiddle factor
                float next_real = cur_real * w_real - cur_imag * w_imag;
                float next_imag = cur_real * w_imag + cur_imag * w_real;
                cur_real = next_real;
                cur_imag = next_imag;
            }
        }
    }

    // Scale for inverse FFT
    if (inverse) {
        float scale = 1.0f / dim;
        for (int i = 0; i < dim; ++i) {
            real[i] *= scale;
            imag[i] *= scale;
        }
    }
}

/**
 * Compute top-K frequency indices by magnitude.
 *
 * @param magnitudes Magnitude array [dim]
 * @param indices Output top-K indices [bandwidth]
 * @param dim Input dimension
 * @param bandwidth Number of indices to select
 * @param config Compression configuration
 */
inline void compute_topk_indices(
    const float* magnitudes,
    int* indices,
    int dim,
    int bandwidth,
    const HDGradCompressConfig& config
) {
    // Create index-magnitude pairs
    std::vector<std::pair<float, int>> mag_idx(dim);
    for (int i = 0; i < dim; ++i) {
        mag_idx[i] = {magnitudes[i], i};
    }

    // Handle preserved components
    int reserved = 0;
    if (config.preserve_dc && reserved < bandwidth) {
        indices[reserved++] = 0;
        mag_idx[0].first = -1.0f;  // Exclude from sorting
    }
    if (config.preserve_nyquist && dim > 1 && reserved < bandwidth) {
        int nyquist = dim / 2;
        indices[reserved++] = nyquist;
        mag_idx[nyquist].first = -1.0f;
    }

    // Partial sort for remaining top-K
    int k = bandwidth - reserved;
    if (k > 0) {
        std::partial_sort(
            mag_idx.begin(), mag_idx.begin() + k, mag_idx.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );

        for (int i = 0; i < k && reserved + i < bandwidth; ++i) {
            indices[reserved + i] = mag_idx[i].second;
        }
    }
}

/**
 * Compress gradient via FFT frequency filtering.
 *
 * @param gradient Input gradient [dim]
 * @param compressed_real Output real part of kept frequencies [bandwidth]
 * @param compressed_imag Output imag part of kept frequencies [bandwidth]
 * @param indices Output indices of kept frequencies [bandwidth]
 * @param dim Input dimension
 * @param config Compression configuration
 * @return Actual number of frequencies kept
 */
inline int HDGradientCompress(
    const float* gradient,
    float* compressed_real,
    float* compressed_imag,
    int* indices,
    int dim,
    const HDGradCompressConfig& config
) {
    // Pad to power of 2
    int padded_dim = 1;
    while (padded_dim < dim) {
        padded_dim <<= 1;
    }

    // Allocate FFT buffers
    std::vector<float> real(padded_dim, 0.0f);
    std::vector<float> imag(padded_dim, 0.0f);

    // Copy input
    std::memcpy(real.data(), gradient, dim * sizeof(float));

    // Forward FFT
    fft_radix2(real.data(), imag.data(), padded_dim, false);

    // Compute magnitudes
    std::vector<float> magnitudes(padded_dim);
    for (int i = 0; i < padded_dim; ++i) {
        magnitudes[i] = complex_magnitude(real[i], imag[i]);
    }

    // Get top-K indices
    int bandwidth = std::min(config.bandwidth, padded_dim);
    compute_topk_indices(magnitudes.data(), indices, padded_dim, bandwidth, config);

    // Extract compressed coefficients
    for (int i = 0; i < bandwidth; ++i) {
        int idx = indices[i];
        compressed_real[i] = real[idx];
        compressed_imag[i] = imag[idx];
    }

    return bandwidth;
}

/**
 * Decompress gradient from frequency representation.
 *
 * @param compressed_real Real part of kept frequencies [bandwidth]
 * @param compressed_imag Imag part of kept frequencies [bandwidth]
 * @param indices Indices of kept frequencies [bandwidth]
 * @param gradient Output gradient [dim]
 * @param bandwidth Number of kept frequencies
 * @param dim Output dimension
 * @param config Compression configuration
 */
inline void HDGradientDecompress(
    const float* compressed_real,
    const float* compressed_imag,
    const int* indices,
    float* gradient,
    int bandwidth,
    int dim,
    const HDGradCompressConfig& config
) {
    // Pad to power of 2
    int padded_dim = 1;
    while (padded_dim < dim) {
        padded_dim <<= 1;
    }

    // Initialize FFT buffers to zero
    std::vector<float> real(padded_dim, 0.0f);
    std::vector<float> imag(padded_dim, 0.0f);

    // Scatter compressed coefficients
    for (int i = 0; i < bandwidth; ++i) {
        int idx = indices[i];
        if (idx < padded_dim) {
            real[idx] = compressed_real[i];
            imag[idx] = compressed_imag[i];
        }
    }

    // Inverse FFT
    fft_radix2(real.data(), imag.data(), padded_dim, true);

    // Copy to output with scaling
    for (int i = 0; i < dim; ++i) {
        gradient[i] = real[i] * config.scale;
    }
}

/**
 * Batch compress multiple gradients.
 * OpenMP parallelized for multi-core performance.
 *
 * @param gradients Input gradients [batch, dim]
 * @param compressed_real Output real [batch, bandwidth]
 * @param compressed_imag Output imag [batch, bandwidth]
 * @param indices Output indices [batch, bandwidth]
 * @param batch_size Number of gradients
 * @param dim Gradient dimension
 * @param config Compression configuration
 */
inline void HDGradientBatchCompress(
    const float* gradients,
    float* compressed_real,
    float* compressed_imag,
    int* indices,
    int batch_size,
    int dim,
    const HDGradCompressConfig& config
) {
    int bandwidth = std::min(config.bandwidth, dim);

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        HDGradientCompress(
            gradients + b * dim,
            compressed_real + b * bandwidth,
            compressed_imag + b * bandwidth,
            indices + b * bandwidth,
            dim,
            config
        );
    }
}

/**
 * Batch decompress multiple gradients.
 * OpenMP parallelized for multi-core performance.
 *
 * @param compressed_real Input real [batch, bandwidth]
 * @param compressed_imag Input imag [batch, bandwidth]
 * @param indices Input indices [batch, bandwidth]
 * @param gradients Output gradients [batch, dim]
 * @param batch_size Number of gradients
 * @param bandwidth Kept frequencies per gradient
 * @param dim Gradient dimension
 * @param config Compression configuration
 */
inline void HDGradientBatchDecompress(
    const float* compressed_real,
    const float* compressed_imag,
    const int* indices,
    float* gradients,
    int batch_size,
    int bandwidth,
    int dim,
    const HDGradCompressConfig& config
) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        HDGradientDecompress(
            compressed_real + b * bandwidth,
            compressed_imag + b * bandwidth,
            indices + b * bandwidth,
            gradients + b * dim,
            bandwidth,
            dim,
            config
        );
    }
}

/**
 * Get compression ratio for given parameters.
 *
 * @param dim Original dimension
 * @param bandwidth Kept frequencies
 * @return Compression ratio (original / compressed)
 */
inline float GetCompressionRatio(int dim, int bandwidth) {
    // Real + imag + index per frequency = 3 floats worth per frequency
    // vs 1 float per original dimension
    // Effective compression = dim / (3 * bandwidth)
    // But indices can be stored as int16 if dim < 65536
    float compressed_size = static_cast<float>(bandwidth) * 2.5f;  // 2 floats + 0.5 for index
    return static_cast<float>(dim) / compressed_size;
}

}  // namespace hd_grad_compress
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_HD_GRADIENT_COMPRESSION_OP_H_
