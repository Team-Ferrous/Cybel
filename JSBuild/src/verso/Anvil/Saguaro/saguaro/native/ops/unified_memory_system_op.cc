// saguaro.native/ops/unified_memory_system_op.cc
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
 * @file unified_memory_system_op.cc
 * @brief Implementation of Unified Memory System Operations
 *
 * Phase 4 of V2 Performance Optimization.
 * Consolidates 5 memory mechanisms into unified kernels.
 */

#include "unified_memory_system_op.h"

#include <cstring>
#include <memory>

// TensorFlow op registration
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace saguaro {
namespace memory {

// =============================================================================
// CONTENT-ADDRESSED MEMORY KERNEL
// =============================================================================

void ContentAddressedMemoryRead(
    const float* query,
    const float* keys,
    const float* values,
    float* output,
    float* attention_weights,
    const MemoryConfig& config) {
    
    int num_slots = config.num_slots;
    int slot_dim = config.slot_dim;
    int query_dim = config.query_dim;
    float temperature = config.temperature;
    
    // Compute similarities
    primitives::batch_cosine_similarity(query, keys, attention_weights,
                                        num_slots, query_dim, config.epsilon);
    
    // Softmax
    primitives::softmax_inplace(attention_weights, num_slots, temperature);
    
    // Weighted read
    primitives::weighted_read(attention_weights, values, output, num_slots, slot_dim);
}

void ContentAddressedMemoryWrite(
    float* keys,
    float* values,
    const float* key,
    const float* value,
    int slot_idx,
    float gate,
    const MemoryConfig& config) {
    
    int slot_dim = config.slot_dim;
    int query_dim = config.query_dim;
    
    // Gated update for key
    primitives::gated_write(keys, slot_idx, key, gate, query_dim);
    
    // Gated update for value
    primitives::gated_write(values, slot_idx, value, gate, slot_dim);
}

// =============================================================================
// PRODUCT-KEY MEMORY KERNEL (O(√M) LOOKUP)
// =============================================================================

void ProductKeyMemoryRead(
    const float* query,
    const float* codebook_a,
    const float* codebook_b,
    const float* memory,
    float* output,
    float* attention_weights,
    const MemoryConfig& config) {
    
    int codebook_size = config.codebook_size;
    int subkey_dim = config.subkey_dim;
    int product_k = config.product_k;
    int num_slots = config.num_slots;
    int slot_dim = config.slot_dim;
    
    // Split query into two sub-keys
    const float* query_a = query;
    const float* query_b = query + subkey_dim;
    
    // Compute similarity to codebook A (O(√M))
    std::vector<float> sim_a(codebook_size);
    for (int i = 0; i < codebook_size; ++i) {
        sim_a[i] = primitives::simd_dot(query_a, codebook_a + i * subkey_dim, subkey_dim);
    }
    
    // Compute similarity to codebook B (O(√M))
    std::vector<float> sim_b(codebook_size);
    for (int i = 0; i < codebook_size; ++i) {
        sim_b[i] = primitives::simd_dot(query_b, codebook_b + i * subkey_dim, subkey_dim);
    }
    
    // Get top-k indices from each codebook
    std::vector<int32_t> top_indices_a(product_k);
    std::vector<float> top_values_a(product_k);
    primitives::top_k_select(sim_a.data(), top_indices_a.data(), top_values_a.data(),
                            codebook_size, product_k);
    
    std::vector<int32_t> top_indices_b(product_k);
    std::vector<float> top_values_b(product_k);
    primitives::top_k_select(sim_b.data(), top_indices_b.data(), top_values_b.data(),
                            codebook_size, product_k);
    
    // Combine indices and scores (k² candidates)
    int num_candidates = product_k * product_k;
    std::vector<float> combined_scores(num_candidates);
    std::vector<int32_t> combined_indices(num_candidates);
    
    for (int i = 0; i < product_k; ++i) {
        for (int j = 0; j < product_k; ++j) {
            int idx = i * product_k + j;
            combined_scores[idx] = top_values_a[i] + top_values_b[j];
            combined_indices[idx] = top_indices_a[i] * codebook_size + top_indices_b[j];
            
            // Clamp to valid memory range
            if (combined_indices[idx] >= num_slots) {
                combined_indices[idx] = combined_indices[idx] % num_slots;
            }
        }
    }
    
    // Softmax over combined scores
    primitives::softmax_inplace(combined_scores.data(), num_candidates, config.temperature);
    
    // Sparse weighted read using only the k² candidates
    std::fill(output, output + slot_dim, 0.0f);
    std::fill(attention_weights, attention_weights + num_slots, 0.0f);
    
    for (int c = 0; c < num_candidates; ++c) {
        int slot_idx = combined_indices[c];
        float weight = combined_scores[c];
        
        attention_weights[slot_idx] += weight;
        
        const float* slot = memory + slot_idx * slot_dim;
        for (int d = 0; d < slot_dim; ++d) {
            output[d] += weight * slot[d];
        }
    }
}

// =============================================================================
// HOPFIELD MEMORY KERNEL (ENERGY-BASED)
// =============================================================================

void HopfieldMemoryRead(
    const float* query,
    const float* patterns,
    float* output,
    const MemoryConfig& config) {
    
    int num_slots = config.num_slots;
    int slot_dim = config.slot_dim;
    float beta = config.beta;
    int num_iterations = config.num_iterations;
    
    // Initialize state from query
    std::vector<float> state(slot_dim);
    std::copy(query, query + slot_dim, state.data());
    
    // Hopfield update iterations
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Compute attention scores: β * state^T * patterns
        std::vector<float> scores(num_slots);
        
        #pragma omp parallel for
        for (int s = 0; s < num_slots; ++s) {
            const float* pattern = patterns + s * slot_dim;
            scores[s] = beta * primitives::simd_dot(state.data(), pattern, slot_dim);
        }
        
        // Softmax
        primitives::softmax_inplace(scores.data(), num_slots, 1.0f);
        
        // Update state: weighted sum of patterns
        std::fill(state.begin(), state.end(), 0.0f);
        for (int s = 0; s < num_slots; ++s) {
            const float* pattern = patterns + s * slot_dim;
            float w = scores[s];
            for (int d = 0; d < slot_dim; ++d) {
                state[d] += w * pattern[d];
            }
        }
    }
    
    std::copy(state.begin(), state.end(), output);
}

void HopfieldMemoryEnergy(
    const float* state,
    const float* patterns,
    float* energy,
    const MemoryConfig& config) {
    
    int num_slots = config.num_slots;
    int slot_dim = config.slot_dim;
    float beta = config.beta;
    
    // E(s) = -β⁻¹ log(Σᵢ exp(β s^T ξᵢ)) + ½||s||² + β⁻¹ log(M)
    
    // Compute log-sum-exp of similarities
    float max_score = -1e30f;
    std::vector<float> scores(num_slots);
    
    for (int s = 0; s < num_slots; ++s) {
        const float* pattern = patterns + s * slot_dim;
        scores[s] = beta * primitives::simd_dot(state, pattern, slot_dim);
        max_score = std::max(max_score, scores[s]);
    }
    
    float sum_exp = 0.0f;
    for (int s = 0; s < num_slots; ++s) {
        sum_exp += std::exp(scores[s] - max_score);
    }
    float lse = max_score + std::log(sum_exp);
    
    // State norm
    float state_norm_sq = primitives::simd_dot(state, state, slot_dim);
    
    // Energy
    float inv_beta = 1.0f / beta;
    *energy = -inv_beta * lse + 0.5f * state_norm_sq + inv_beta * std::log(static_cast<float>(num_slots));
}

// =============================================================================
// ADAPTIVE MEMORY KERNEL (SURPRISE-GATED)
// =============================================================================

void AdaptiveMemoryReadWrite(
    const float* input,
    float* memory,
    float* output,
    float* surprise_out,
    bool write_enabled,
    const MemoryConfig& config) {
    
    int num_slots = config.num_slots;
    int slot_dim = config.slot_dim;
    float surprise_threshold = config.surprise_threshold;
    float decay_rate = config.decay_rate;
    float write_strength = config.write_strength;
    
    // Read: content-addressed lookup
    std::vector<float> attention_weights(num_slots);
    primitives::batch_cosine_similarity(input, memory, attention_weights.data(),
                                        num_slots, slot_dim, config.epsilon);
    primitives::softmax_inplace(attention_weights.data(), num_slots, config.temperature);
    
    // Compute predicted value
    std::vector<float> predicted(slot_dim, 0.0f);
    primitives::weighted_read(attention_weights.data(), memory, predicted.data(),
                             num_slots, slot_dim);
    
    // Copy to output
    std::copy(predicted.begin(), predicted.end(), output);
    
    // Compute surprise for write gating
    float surprise = primitives::surprise_gate(input, predicted.data(), slot_dim, surprise_threshold);
    if (surprise_out != nullptr) {
        *surprise_out = surprise;
    }
    
    if (write_enabled && surprise > 0.5f) {
        // Find slot to write (least recently used = lowest attention)
        int lru_slot = 0;
        float min_attn = attention_weights[0];
        for (int s = 1; s < num_slots; ++s) {
            if (attention_weights[s] < min_attn) {
                min_attn = attention_weights[s];
                lru_slot = s;
            }
        }
        
        // Write with surprise-modulated gate
        float write_gate = decay_rate * (1.0f - surprise * write_strength);
        primitives::gated_write(memory, lru_slot, input, write_gate, slot_dim);
    }
    
    // Apply decay to all slots
    int total_elements = num_slots * slot_dim;
    for (int i = 0; i < total_elements; ++i) {
        memory[i] *= decay_rate;
    }
}

// =============================================================================
// HIERARCHICAL MEMORY KERNEL (MULTI-LEVEL WITH CTQW)
// =============================================================================

void HierarchicalMemoryRead(
    const float* query,
    float* const* level_memory,
    float* output,
    const MemoryConfig& config) {
    
    int num_levels = config.num_levels;
    int slot_dim = config.slot_dim;
    float ctqw_gamma = config.ctqw_gamma;
    
    // Accumulate across levels
    std::vector<float> accumulated(slot_dim, 0.0f);
    float total_weight = 0.0f;
    
    for (int level = 0; level < num_levels; ++level) {
        int slots_at_level = config.slots_per_level[level];
        const float* level_mem = level_memory[level];
        
        // Compute attention at this level
        std::vector<float> attention(slots_at_level);
        primitives::batch_cosine_similarity(query, level_mem, attention.data(),
                                           slots_at_level, slot_dim, config.epsilon);
        
        // CTQW-inspired hopping probability decay with level
        float level_weight = std::exp(-ctqw_gamma * level);
        
        // Softmax at this level
        primitives::softmax_inplace(attention.data(), slots_at_level, config.temperature);
        
        // Weighted read
        std::vector<float> level_output(slot_dim, 0.0f);
        primitives::weighted_read(attention.data(), level_mem, level_output.data(),
                                 slots_at_level, slot_dim);
        
        // Accumulate with CTQW weights
        for (int d = 0; d < slot_dim; ++d) {
            accumulated[d] += level_weight * level_output[d];
        }
        total_weight += level_weight;
    }
    
    // Normalize by total weight
    float inv_weight = 1.0f / (total_weight + 1e-10f);
    for (int d = 0; d < slot_dim; ++d) {
        output[d] = accumulated[d] * inv_weight;
    }
}

}  // namespace memory
}  // namespace saguaro

// =============================================================================
// TENSORFLOW OP REGISTRATION
// =============================================================================

using namespace tensorflow;

REGISTER_OP("UnifiedMemorySystemOp")
    .Input("query: float")
    .Input("memory: float")
    .Input("aux_data: float")
    .Output("output: float")
    .Output("attention_weights: float")
    .Attr("mem_type: int = 0")
    .Attr("batch_size: int = 1")
    .Attr("num_slots: int = 256")
    .Attr("slot_dim: int = 512")
    .Attr("query_dim: int = 512")
    .Attr("codebook_size: int = 64")
    .Attr("subkey_dim: int = 256")
    .Attr("product_k: int = 8")
    .Attr("temperature: float = 1.0")
    .Attr("beta: float = 1.0")
    .Attr("num_iterations: int = 1")
    .Attr("epsilon: float = 1e-6")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output shape depends on slot_dim
        int64_t slot_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("slot_dim", &slot_dim));
        int64_t num_slots;
        TF_RETURN_IF_ERROR(c->GetAttr("num_slots", &num_slots));
        
        c->set_output(0, c->MakeShape({slot_dim}));
        c->set_output(1, c->MakeShape({num_slots}));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Unified Memory System Operation.

Consolidates 5 memory mechanisms into a single dispatched op:
- CONTENT_ADDRESSED (0): Standard attention-based memory
- PRODUCT_KEY (1): Sub-linear O(√M) lookup
- HOPFIELD (2): Energy-based associative memory
- ADAPTIVE (3): Surprise-gated memory
- HIERARCHICAL (4): Multi-level with CTQW
)doc");

class UnifiedMemorySystemOp : public OpKernel {
public:
    explicit UnifiedMemorySystemOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        int mem_type_int;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("mem_type", &mem_type_int));
        config_.mem_type = static_cast<saguaro::memory::MemoryType>(mem_type_int);
        
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &config_.batch_size));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_slots", &config_.num_slots));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("slot_dim", &config_.slot_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("query_dim", &config_.query_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("codebook_size", &config_.codebook_size));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("subkey_dim", &config_.subkey_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("product_k", &config_.product_k));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &config_.temperature));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &config_.beta));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_iterations", &config_.num_iterations));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &config_.epsilon));
    }
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& query = ctx->input(0);
        const Tensor& memory = ctx->input(1);
        const Tensor& aux_data = ctx->input(2);
        
        // Allocate outputs
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({config_.slot_dim}), &output));
        
        Tensor* attention_weights = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({config_.num_slots}), &attention_weights));
        
        const float* query_data = query.flat<float>().data();
        const float* memory_data = memory.flat<float>().data();
        const float* aux_ptr = aux_data.NumElements() > 1 ? aux_data.flat<float>().data() : nullptr;
        float* output_data = output->flat<float>().data();
        float* attn_data = attention_weights->flat<float>().data();
        
        saguaro::memory::UnifiedMemoryRead(query_data, memory_data, output_data, attn_data,
                                        config_, aux_ptr);
    }

private:
    saguaro::memory::MemoryConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("UnifiedMemorySystemOp").Device(DEVICE_CPU), UnifiedMemorySystemOp);
