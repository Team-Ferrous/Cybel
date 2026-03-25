// saguaro.native/ops/td_moe_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

/**
 * @file td_moe_op.h
 * @brief Phase 57: TD-MoE Tucker Decomposition MoE
 *
 * Tensorized experts using Tucker decomposition for parameter efficiency.
 * Each expert W = G ×₁ A ×₂ B ×₃ C (core + mode matrices)
 *
 * Benefits: 10x parameter reduction, O(R³) per expert
 * Complexity: O(N × E × R³) where R = Tucker rank
 */

#ifndef SAGUARO_NATIVE_OPS_TD_MOE_OP_H_
#define SAGUARO_NATIVE_OPS_TD_MOE_OP_H_

#include <cmath>
#include <vector>

namespace saguaro {
namespace tdmoe {

struct TDMoEConfig {
    int num_experts;
    int input_dim;
    int output_dim;
    int tucker_rank;
    int top_k;
    
    TDMoEConfig() : num_experts(8), input_dim(256), output_dim(256),
                    tucker_rank(16), top_k(2) {}
};

/**
 * @brief Tucker decomposed linear: y = G ×₁ A ×₂ B(x)
 */
inline void TuckerLinear(
    const float* input, const float* core,
    const float* mode_a, const float* mode_b,
    float* output, int batch, int input_dim, int output_dim, int rank) {
    
    // x_proj = B^T x  [rank]
    std::vector<float> x_proj(batch * rank);
    
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rank; ++r) {
            float sum = 0.0f;
            for (int i = 0; i < input_dim; ++i) {
                sum += mode_b[i * rank + r] * input[b * input_dim + i];
            }
            x_proj[b * rank + r] = sum;
        }
    }
    
    // core_proj = G × x_proj  [rank]
    std::vector<float> core_proj(batch * rank);
    for (int b = 0; b < batch; ++b) {
        for (int r1 = 0; r1 < rank; ++r1) {
            float sum = 0.0f;
            for (int r2 = 0; r2 < rank; ++r2) {
                sum += core[r1 * rank + r2] * x_proj[b * rank + r2];
            }
            core_proj[b * rank + r1] = sum;
        }
    }
    
    // output = A × core_proj  [output_dim]
    for (int b = 0; b < batch; ++b) {
        for (int o = 0; o < output_dim; ++o) {
            float sum = 0.0f;
            for (int r = 0; r < rank; ++r) {
                sum += mode_a[o * rank + r] * core_proj[b * rank + r];
            }
            output[b * output_dim + o] = sum;
        }
    }
}

/**
 * @brief TD-MoE forward pass with top-k routing.
 */
inline void TDMoEForward(
    const float* input, const float* router_weights,
    const float* expert_cores, const float* expert_modes_a,
    const float* expert_modes_b, float* output,
    const TDMoEConfig& config, int batch) {
    
    int R = config.tucker_rank;
    int E = config.num_experts;
    int D_in = config.input_dim, D_out = config.output_dim;
    
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        // Compute router logits
        std::vector<float> logits(E);
        for (int e = 0; e < E; ++e) {
            float sum = 0.0f;
            for (int d = 0; d < D_in; ++d) {
                sum += input[b * D_in + d] * router_weights[d * E + e];
            }
            logits[e] = sum;
        }
        
        // Top-k selection
        std::vector<int> top_k_idx(config.top_k);
        std::vector<float> top_k_weights(config.top_k);
        for (int k = 0; k < config.top_k; ++k) {
            int max_idx = 0;
            float max_val = -1e9f;
            for (int e = 0; e < E; ++e) {
                bool already_selected = false;
                for (int kk = 0; kk < k; ++kk) {
                    if (top_k_idx[kk] == e) already_selected = true;
                }
                if (!already_selected && logits[e] > max_val) {
                    max_val = logits[e];
                    max_idx = e;
                }
            }
            top_k_idx[k] = max_idx;
            top_k_weights[k] = max_val;
        }
        
        // Softmax weights
        float sum_exp = 0.0f;
        for (int k = 0; k < config.top_k; ++k) {
            top_k_weights[k] = std::exp(top_k_weights[k]);
            sum_exp += top_k_weights[k];
        }
        for (int k = 0; k < config.top_k; ++k) {
            top_k_weights[k] /= sum_exp;
        }
        
        // Combine expert outputs
        float* out = output + b * D_out;
        std::fill(out, out + D_out, 0.0f);
        
        std::vector<float> expert_out(D_out);
        for (int k = 0; k < config.top_k; ++k) {
            int e = top_k_idx[k];
            
            const float* core = expert_cores + e * R * R;
            const float* mode_a = expert_modes_a + e * D_out * R;
            const float* mode_b = expert_modes_b + e * D_in * R;
            
            TuckerLinear(input + b * D_in, core, mode_a, mode_b,
                        expert_out.data(), 1, D_in, D_out, R);
            
            for (int d = 0; d < D_out; ++d) {
                out[d] += top_k_weights[k] * expert_out[d];
            }
        }
    }
}

}}
#endif
