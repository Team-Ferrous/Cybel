// highnoon/_native/ops/hd_streaming_adapter_op.h
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

#ifndef HIGHNOON_NATIVE_OPS_HD_STREAMING_ADAPTER_OP_H_
#define HIGHNOON_NATIVE_OPS_HD_STREAMING_ADAPTER_OP_H_

#include <cstdint>
#include <cstring>
#include <cmath>

namespace hsmn {
namespace hd_streaming {

/**
 * HD Streaming Adapter Configuration.
 * 
 * This op projects HD bundles from HolographicCorpus to model hidden dimension,
 * adding a sequence dimension for compatibility with ReasoningModule.
 * 
 * Shape transformation: (batch, hd_dim) -> (batch, 1, hidden_dim)
 */
struct HDStreamingConfig {
    int hd_dim = 1024;           // Input HD bundle dimension
    int hidden_dim = 256;        // Output model hidden dimension
    bool add_sequence_dim = true; // If true: output (batch, 1, hidden_dim)
                                  // If false: output (batch, hidden_dim)
};

/**
 * Forward projection: (batch, hd_dim) -> (batch, [1,] hidden_dim)
 * 
 * Computes: output = matmul(hd_bundles, projection_weights) + projection_bias
 * Optionally adds sequence dimension for ReasoningModule compatibility.
 * 
 * @param hd_bundles Input HD bundles [batch, hd_dim], float32
 * @param projection_weights Projection matrix [hd_dim, hidden_dim], float32
 * @param projection_bias Bias vector [hidden_dim], float32 (may be nullptr)
 * @param output Output tensor [batch, hidden_dim] or [batch, 1, hidden_dim], float32
 * @param config HD streaming configuration
 * @param batch_size Batch size
 */
inline void HDStreamProject(
    const float* hd_bundles,
    const float* projection_weights,
    const float* projection_bias,
    float* output,
    const HDStreamingConfig& config,
    int batch_size
) {
    const int hd_dim = config.hd_dim;
    const int hidden_dim = config.hidden_dim;
    
    // For each batch element: output[b] = hd_bundles[b] @ weights + bias
    for (int b = 0; b < batch_size; ++b) {
        const float* bundle = hd_bundles + b * hd_dim;
        float* out = output + b * hidden_dim;
        
        // Matrix-vector multiply: out = bundle @ weights
        for (int h = 0; h < hidden_dim; ++h) {
            float sum = 0.0f;
            for (int d = 0; d < hd_dim; ++d) {
                // weights is [hd_dim, hidden_dim] stored row-major
                sum += bundle[d] * projection_weights[d * hidden_dim + h];
            }
            // Add bias if provided
            if (projection_bias != nullptr) {
                sum += projection_bias[h];
            }
            out[h] = sum;
        }
    }
    
    // Note: sequence dimension addition is handled by TensorFlow output shape
}

/**
 * Gradient computation for backward pass.
 * 
 * Computes gradients for:
 * - grad_bundles: dL/d(hd_bundles) = grad_output @ weights^T
 * - grad_weights: dL/d(weights) = hd_bundles^T @ grad_output  
 * - grad_bias: dL/d(bias) = sum(grad_output, axis=0)
 * 
 * @param grad_output Gradient from upstream [batch, hidden_dim], float32
 * @param hd_bundles Forward pass input [batch, hd_dim], float32
 * @param projection_weights Forward pass weights [hd_dim, hidden_dim], float32
 * @param grad_bundles Output gradient for bundles [batch, hd_dim], float32
 * @param grad_weights Output gradient for weights [hd_dim, hidden_dim], float32
 * @param grad_bias Output gradient for bias [hidden_dim], float32
 * @param config HD streaming configuration
 * @param batch_size Batch size
 */
inline void HDStreamProjectGrad(
    const float* grad_output,
    const float* hd_bundles,
    const float* projection_weights,
    float* grad_bundles,
    float* grad_weights,
    float* grad_bias,
    const HDStreamingConfig& config,
    int batch_size
) {
    const int hd_dim = config.hd_dim;
    const int hidden_dim = config.hidden_dim;
    
    // Initialize gradients to zero
    std::memset(grad_weights, 0, hd_dim * hidden_dim * sizeof(float));
    std::memset(grad_bias, 0, hidden_dim * sizeof(float));
    
    for (int b = 0; b < batch_size; ++b) {
        const float* bundle = hd_bundles + b * hd_dim;
        const float* g_out = grad_output + b * hidden_dim;
        float* g_bundle = grad_bundles + b * hd_dim;
        
        // grad_bundles[b] = grad_output[b] @ weights^T
        for (int d = 0; d < hd_dim; ++d) {
            float sum = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                sum += g_out[h] * projection_weights[d * hidden_dim + h];
            }
            g_bundle[d] = sum;
        }
        
        // grad_weights += bundle^T @ grad_output (outer product, accumulated)
        for (int d = 0; d < hd_dim; ++d) {
            for (int h = 0; h < hidden_dim; ++h) {
                grad_weights[d * hidden_dim + h] += bundle[d] * g_out[h];
            }
        }
        
        // grad_bias += grad_output (sum over batch)
        for (int h = 0; h < hidden_dim; ++h) {
            grad_bias[h] += g_out[h];
        }
    }
}

}  // namespace hd_streaming
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_HD_STREAMING_ADAPTER_OP_H_
