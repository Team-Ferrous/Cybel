// src/ops/fused_reasoning_stack/tt_helpers.h
// Copyright 2025 Verso Industries
//
// Helper functions for Tensor-Train (TT) decomposition in the fused reasoning stack.
// Implements forward and backward passes for TT layers in C++ for performance.

#ifndef VERSO_OPS_FUSED_REASONING_STACK_TT_HELPERS_H_
#define VERSO_OPS_FUSED_REASONING_STACK_TT_HELPERS_H_

#include <vector>
#include <string>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "Eigen/Dense"

namespace tensorflow {
namespace tt_helpers {

/**
 * @brief Metadata for a single TT layer extracted from block descriptor.
 */
struct TTLayerInfo {
    std::string name;
    std::vector<int> input_dims;   // Factorization of input dimension
    std::vector<int> output_dims;  // Factorization of output dimension
    std::vector<int> tt_ranks;     // TT-ranks [r0, r1, ..., rd]
    int num_cores;                 // Number of cores (d)
    std::vector<int> core_indices; // Which weight tensor indices are the cores

    // Computed values
    int input_dim_total;  // Product of input_dims
    int output_dim_total; // Product of output_dims

    TTLayerInfo() : num_cores(0), input_dim_total(0), output_dim_total(0) {}

    bool is_valid() const {
        return num_cores > 0 &&
               input_dims.size() == num_cores &&
               output_dims.size() == num_cores &&
               tt_ranks.size() == num_cores + 1 &&
               tt_ranks[0] == 1 &&
               tt_ranks[num_cores] == 1 &&
               core_indices.size() == num_cores;
    }

    void compute_totals() {
        input_dim_total = 1;
        for (int d : input_dims) input_dim_total *= d;
        output_dim_total = 1;
        for (int d : output_dims) output_dim_total *= d;
    }
};

/**
 * @brief Block info containing multiple TT layers.
 */
struct TTBlockInfo {
    std::vector<TTLayerInfo> tt_layers;
    bool has_tt_layers;

    TTBlockInfo() : has_tt_layers(false) {}
};

/**
 * @brief Parse TT layer metadata from block descriptor JSON.
 *
 * Expected JSON format:
 * {
 *   "metadata": {
 *     "tt_layers": [
 *       {
 *         "name": "in_proj",
 *         "input_dims": [8, 8, 8],
 *         "output_dims": [16, 16, 8],
 *         "tt_ranks": [1, 16, 16, 1],
 *         "num_cores": 3,
 *         "core_indices": [0, 1, 2]
 *       }
 *     ]
 *   }
 * }
 */
Status ParseTTBlockInfo(const std::string& descriptor_json, TTBlockInfo* info);

/**
 * @brief Validate TT layer weights have correct shapes.
 *
 * Checks that each core tensor has shape [r_{i-1}, m_i, n_i, r_i].
 */
bool ValidateTTLayerWeights(
    const OpInputList& weights,
    const TTLayerInfo& tt_info,
    std::string* failure_reason,
    int weight_offset = 0);

/**
 * @brief Execute forward pass for a single TT layer.
 *
 * Algorithm:
 * 1. Reshape input from [B, D_in] to [B, m1, m2, ..., md]
 * 2. For each core i:
 *    - Contract current result with core[i]
 *    - For i=0: squeeze rank dimension (rank=1)
 *    - For i=d-1: final core produces output
 * 3. Squeeze final rank dimension (rank=1)
 * 4. Reshape to [B, D_out]
 *
 * @param input Input matrix [batch, input_dim]
 * @param weights OpInputList containing all weights
 * @param tt_info TT layer metadata
 * @return Output matrix [batch, output_dim]
 */
Eigen::MatrixXf RunTTLayerForward(
    const Eigen::MatrixXf& input,
    const OpInputList& weights,
    const TTLayerInfo& tt_info,
    int weight_offset);

/**
 * @brief Execute forward pass for a stateless block with TT layers.
 *
 * Handles blocks like SpatialBlock that contain multiple TT layers
 * interspersed with other operations (activations, etc.).
 *
 * For simplicity in the initial implementation, this performs a
 * simplified forward pass. Full block logic would be implemented
 * in specialized handlers.
 */
Eigen::MatrixXf RunTTBlockForward(
    OpKernelContext* context,
    const Eigen::MatrixXf& input,
    const OpInputList& weights,
    const TTBlockInfo& tt_info,
    int weight_start_idx);

/**
 * @brief Compute gradients for TT layer cores.
 *
 * Uses chain rule to compute gradients w.r.t. each core tensor.
 *
 * @param grad_output Gradient from upstream [batch, output_dim]
 * @param input Original input [batch, input_dim]
 * @param weights Forward pass weights
 * @param tt_info TT layer metadata
 * @param grad_weights_tensors Output gradient tensors (preallocated)
 * @param grad_start_idx Starting index in grad_weights_tensors for this layer's cores
 */
void ComputeTTLayerGradients(
    OpKernelContext* context,
    const Eigen::MatrixXf& grad_output,
    const Eigen::MatrixXf& input,
    const OpInputList& weights,
    const TTLayerInfo& tt_info,
    std::vector<Tensor*>& grad_weights_tensors,
    int grad_start_idx);

/**
 * @brief Helper to perform tensor contraction for TT core.
 *
 * Performs einsum-style contraction: result, core -> output
 * where result has shape [B, ...previous outputs..., ...remaining inputs..., rank_in]
 * and core has shape [rank_in, m_i, n_i, rank_out]
 */
Eigen::MatrixXf ContractTTCore(
    const Eigen::MatrixXf& result,
    const Tensor& core_tensor,
    int core_idx,
    const TTLayerInfo& tt_info,
    bool is_first_core);

}  // namespace tt_helpers
}  // namespace tensorflow

#endif  // VERSO_OPS_FUSED_REASONING_STACK_TT_HELPERS_H_
