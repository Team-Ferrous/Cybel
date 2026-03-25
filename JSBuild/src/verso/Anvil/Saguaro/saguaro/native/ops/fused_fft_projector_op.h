// saguaro.native/ops/fused_fft_projector_op.h
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
 * @file fused_fft_projector_op.h
 * @brief UQHA v3.1 Phase 2: FFT-based Thought Projector.
 *
 * Replaces dense O(D²) MLP with O(D log D) FFT-convolutions.
 * This is used by both ContinuousThought and CoCoNut BFS expansion.
 */

#ifndef SAGUARO_NATIVE_OPS_FUSED_FFT_PROJECTOR_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_FFT_PROJECTOR_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

#include "hnn_simd_common.h"
#include "fft_utils.h"

namespace saguaro {
namespace ops {

/**
 * @brief Apply FFT-based projection to a batch of vectors.
 * 
 * Logic:
 * 1. LayerNorm(spatial)
 * 2. FFT(spatial) -> Frequency
 * 3. Element-wise complex multiply with freq_weights_1
 * 4. IFFT(Frequency) -> Hidden Spatial
 * 5. Add bias_1 + GELU
 * 6. FFT(Hidden Spatial) -> Hidden Frequency
 * 7. Element-wise complex multiply with freq_weights_2
 * 8. IFFT(Hidden Frequency) -> Output Spatial
 * 9. Add bias_2 + Residual
 *
 * @param input Input states (row stride = path_stride)
 * @param freq_weights_1 Complex weights for layer 1 [dim] (stored as real/imag pairs)
 * @param bias_1 Bias for layer 1 [dim]
 * @param freq_weights_2 Complex weights for layer 2 [dim]
 * @param bias_2 Bias for layer 2 [dim]
 * @param norm_gamma LayerNorm gamma [dim]
 * @param norm_beta LayerNorm beta [dim]
 * @param output Output states (row stride = path_stride)
 * @param total_paths Batch * num_paths
 * @param dim Hidden dimension (must be power of 2)
 * @param input_is_freq If true, input is in frequency domain (2*dim per row)
 * @param output_is_freq If true, output stays in frequency domain (2*dim per row)
 * @param path_stride Row stride in floats (dim for spatial, 2*dim for freq)
 */
inline void fft_projector_forward(
    float* state,           // [total_paths, path_stride]
    const float* freq_weights_1,  // [2 * dim]
    const float* bias_1,          // [dim]
    const float* freq_weights_2,  // [2 * dim]
    const float* bias_2,          // [dim]
    const float* norm_gamma,      // [dim]
    const float* norm_beta,       // [dim]
    int64_t total_paths,
    int64_t dim,
    bool input_is_freq = false,
    bool output_is_freq = false,
    int64_t path_stride = 0
) {
    // Default stride based on whether we're using freq domain
    if (path_stride == 0) {
        path_stride = output_is_freq ? (2 * dim) : dim;
    }
    
    std::vector<float> work_spatial(dim);
    std::vector<float> work_freq(2 * dim);

    for (int64_t i = 0; i < total_paths; ++i) {
        float* row = state + i * path_stride;

        // Step 1: FFT (if not already in freq domain)
        if (!input_is_freq) {
            // Input is spatial domain - copy real part, zero imaginary
            for (int64_t d = 0; d < dim; ++d) {
                work_freq[2 * d] = row[d];
                work_freq[2 * d + 1] = 0.0f;
            }
            fft_butterfly(work_freq.data(), dim, false);
        } else {
            // Input is already frequency domain - copy directly
            // Data is stored interleaved: [re0, im0, re1, im1, ...]
            std::memcpy(work_freq.data(), row, 2 * dim * sizeof(float));
        }

        // Step 2: Layer 1 Freq Multiply
        for (int64_t d = 0; d < dim; ++d) {
            float re = work_freq[2 * d];
            float im = work_freq[2 * d + 1];
            float wre = freq_weights_1[2 * d];
            float wim = freq_weights_1[2 * d + 1];
            
            work_freq[2 * d] = re * wre - im * wim;
            work_freq[2 * d + 1] = re * wim + im * wre;
        }

        // Step 3: IFFT (to apply non-linearity)
        fft_butterfly(work_freq.data(), dim, true);

        // Step 4: Bias + GELU
        for (int64_t d = 0; d < dim; ++d) {
            float x = work_freq[2 * d] + bias_1[d];
            work_spatial[d] = 0.5f * x * (1.0f + std::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
        }

        // Step 5: FFT back to freq
        for (int64_t d = 0; d < dim; ++d) {
            work_freq[2 * d] = work_spatial[d];
            work_freq[2 * d + 1] = 0.0f;
        }
        fft_butterfly(work_freq.data(), dim, false);

        // Step 6: Layer 2 Freq Multiply
        for (int64_t d = 0; d < dim; ++d) {
            float re = work_freq[2 * d];
            float im = work_freq[2 * d + 1];
            float wre = freq_weights_2[2 * d];
            float wim = freq_weights_2[2 * d + 1];
            
            work_freq[2 * d] = re * wre - im * wim;
            work_freq[2 * d + 1] = re * wim + im * wre;
        }

        if (output_is_freq) {
            // Store back in frequency domain (interleaved format)
            std::memcpy(row, work_freq.data(), 2 * dim * sizeof(float));
        } else {
            // Step 7: IFFT
            fft_butterfly(work_freq.data(), dim, true);

            // Step 8: Bias + Residual
            for (int64_t d = 0; d < dim; ++d) {
                float out = work_freq[2 * d] + bias_2[d];
                if (!input_is_freq) {
                    row[d] += out; // Residual add
                } else {
                    row[d] = out;  // No residual when coming from freq
                }
            }
        }
    }
}

} // namespace ops
} // namespace saguaro

#endif // SAGUARO_NATIVE_OPS_FUSED_FFT_PROJECTOR_OP_H_
