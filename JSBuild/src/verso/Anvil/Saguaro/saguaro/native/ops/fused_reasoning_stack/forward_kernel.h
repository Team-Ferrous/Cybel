// src/ops/fused_reasoning_stack/forward_kernel.h
// Copyright 2025 Verso Industries
//
// This file contains the definition and implementation of the forward pass
// for the FusedReasoningStack operator. It's designed to be a complete,
// production-ready kernel.

#ifndef TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_FORWARD_KERNEL_H_
#define TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_FORWARD_KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "ops/hnn_core_helpers.h" // For HNN/TimeCrystal logic
#include "ops/fused_reasoning_stack/helpers.h" // For BlockContext and helper declarations
#include "ops/fused_reasoning_stack/tt_helpers.h" // For TT decomposition support
#include "common/parallel/parallel_backend.h"  // Unified threading backend
#include "Eigen/Core"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h" // Added for TensorFlow's work sharder
#include "common/tensor_stream_pool.h"  // Phase 2: Zero-copy inter-kernel streaming

namespace tensorflow {

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;
using Eigen::RowMajor;


/**
 * @brief Helper for forward pass through Stateless Blocks (e.g., MoE, Mamba placeholder).
 *
 * Blocks with non-densifiable tensors (e.g., TT-layer cores, MoE routing weights)
 * are gracefully skipped: their declared tensors are consumed to keep the global
 * index in sync, but the residual branch effectively becomes an identity.
 */
inline Eigen::MatrixXf StatelessBlockForward(
    OpKernelContext* context,
    const BlockDescriptorInfo& descriptor,
    const Eigen::MatrixXf& input_seq,
    const tensorflow::OpInputList& weights,
    int& weight_idx,
    int num_weights_to_consume) {

    if (num_weights_to_consume <= 0) {
        weight_idx += std::max(0, num_weights_to_consume);
        return MatrixXf::Zero(input_seq.rows(), input_seq.cols());
    }

    // Check if this block has TT layers
    LOG_FIRST_N(WARNING, 1) << "@@@ BINARY VERSION 2025-11-21-10:05 - TT Support Enabled @@@";
    tt_helpers::TTBlockInfo tt_info;
    Status tt_parse_status = tt_helpers::ParseTTBlockInfo(descriptor.raw_json, &tt_info);

    if (tt_parse_status.ok() && tt_info.has_tt_layers) {
        // This block uses TT decomposition - use TT kernel
        LOG_FIRST_N(INFO, 5) << "[FusedReasoningStack] Using TT kernel for block type="
                             << descriptor.type << " with " << tt_info.tt_layers.size()
                             << " TT layer(s)";

        // Validate TT weights
        bool all_valid = true;
        for (const auto& layer : tt_info.tt_layers) {
            std::string failure_reason;
            if (!tt_helpers::ValidateTTLayerWeights(weights, layer, &failure_reason, weight_idx)) {
                LOG(WARNING) << "[FusedReasoningStack] TT validation failed for layer "
                             << layer.name << ": " << failure_reason;
                all_valid = false;
                break;
            }
        }

        if (all_valid) {
            MatrixXf result = tt_helpers::RunTTBlockForward(
                context, input_seq, weights, tt_info, weight_idx);

            if (!context->status().ok()) {
                return MatrixXf();
            }
            weight_idx += num_weights_to_consume;
            return result;
        } else {
            // Validation failed, fall through to dense kernel attempt
            LOG_FIRST_N(WARNING, 10) << "[FusedReasoningStack] TT validation failed, trying dense kernel";
        }
    }

    // Try dense kernel
    std::string failure_reason;
    if (!stateless_internal::CanRunDenseStatelessBlock(input_seq, weights, weight_idx,
                                                       num_weights_to_consume, &failure_reason)) {
        LOG_FIRST_N(WARNING, 10)
            << "[FusedReasoningStack] Skipping unsupported stateless block: "
            << failure_reason << " metadata=" << descriptor.raw_json;
        weight_idx += std::max(0, num_weights_to_consume);
        return MatrixXf::Zero(input_seq.rows(), input_seq.cols());
    }

    MatrixXf transformed = stateless_internal::RunStatelessDenseForward(
        context, input_seq, weights, weight_idx, num_weights_to_consume);

    if (!context->status().ok()) {
        return MatrixXf();
    }

    weight_idx += num_weights_to_consume;
    return transformed;
}

inline void TimeCrystalSequenceForward(
    const BlockContext& ctx,
    MatrixXf& current_sequence,
    const OpInputList& initial_float_states,
    const OpInputList& weights,
    int& float_state_idx,
    int& weight_idx,
    std::vector<Tensor>& final_float_state_tensors,
    std::vector<std::vector<HNNForwardState>>& hnn_forward_states) {

    OpKernelContext* context = ctx.op_context;
    const int64_t batch_size = ctx.batch_size;
    const int64_t seq_len_combined = ctx.seq_len_combined;
    const int64_t d_embed = ctx.d_embed;
    const int64_t total_rows = batch_size * seq_len_combined;

    const Tensor& h_padded_tensor = initial_float_states[float_state_idx];
    const Tensor& W1 = weights[weight_idx];
    const int64_t D_state = (W1.dim_size(0) - d_embed) / 2;
    const int64_t D_padded_seq = h_padded_tensor.dim_size(1);
    const int64_t D_mamba_state = h_padded_tensor.dim_size(2);
    const int64 num_slices = (2 * D_state) / D_mamba_state;

    Tensor q_init_tensor(DT_FLOAT, TensorShape({batch_size, D_state}));
    Tensor p_init_tensor(DT_FLOAT, TensorShape({batch_size, D_state}));

    auto h_padded_eigen = h_padded_tensor.shaped<float, 3>({batch_size, D_padded_seq, D_mamba_state});

    for(int b=0; b < batch_size; ++b) {
        Map<VectorXf> q_init_eigen(q_init_tensor.flat<float>().data() + b * D_state, D_state);
        Map<VectorXf> p_init_eigen(p_init_tensor.flat<float>().data() + b * D_state, D_state);
        Eigen::MatrixXf h_flat = Map<const Eigen::MatrixXf>(h_padded_eigen.data() + b * D_padded_seq * D_mamba_state, D_padded_seq, D_mamba_state).topRows(num_slices);
        VectorXf state_unpacked = Map<VectorXf>(h_flat.data(), D_state * 2);
        q_init_eigen = state_unpacked.head(D_state);
        p_init_eigen = state_unpacked.tail(D_state);
    }

    // Consume 9 weights for HNN
    const Tensor& W1_w = weights[weight_idx + 0];
    const Tensor& b1 = weights[weight_idx + 1];
    const Tensor& W2 = weights[weight_idx + 2];
    const Tensor& b2 = weights[weight_idx + 3];
    const Tensor& W3 = weights[weight_idx + 4];
    const Tensor& b3 = weights[weight_idx + 5]; // SCALAR
    const Tensor& epsilon_param = weights[weight_idx + 6]; // SCALAR
    const Tensor& W_out = weights[weight_idx + 7];
    const Tensor& b_out = weights[weight_idx + 8];

    // Use the same check as the forward pass for consistency
    OP_REQUIRES(context, b3.NumElements() == 1, errors::InvalidArgument("HNN b3 bias must be a 1-element tensor."));
    OP_REQUIRES(context, epsilon_param.NumElements() == 1, errors::InvalidArgument("HNN epsilon_param must be a 1-element tensor."));

    const float epsilon_param_val = epsilon_param.scalar<float>()();
    const float epsilon = std::min(1.0f, 0.01f + 0.99f * std::tanh(epsilon_param_val));
    const float b3_scalar = b3.scalar<float>()();

    const int64_t D_in_hnn = 2 * D_state + d_embed;
    const int64_t D_h = W1_w.dim_size(1);
    const int64_t D_output = b_out.dim_size(0);

    Map<const MatrixXf> W1_map(W1_w.flat<float>().data(), D_in_hnn, D_h);
    Map<const VectorXf> b1_map(b1.flat<float>().data(), D_h);
    Map<const MatrixXf> W2_map(W2.flat<float>().data(), D_h, D_h);
    Map<const VectorXf> b2_map(b2.flat<float>().data(), D_h);
    Map<const MatrixXf> W3_map(W3.flat<float>().data(), D_h, 1);

    Map<const MatrixXf> W_out_map(W_out.flat<float>().data(), 2 * D_state, D_output);
    Map<const VectorXf> b_out_map(b_out.flat<float>().data(), D_output);

    MatrixXf next_sequence(total_rows, d_embed);

    saguaro::parallel::ForRange(
        0, static_cast<std::size_t>(batch_size), 1,
        [&](std::size_t range_begin, std::size_t range_end) {
            for (std::size_t idx = range_begin; idx < range_end; ++idx) {
                const int b = static_cast<int>(idx);
                VectorXf q_t = Map<const VectorXf>(q_init_tensor.flat<float>().data() + b * D_state, D_state);
                VectorXf p_t = Map<const VectorXf>(p_init_tensor.flat<float>().data() + b * D_state, D_state);

                for (int l = 0; l < seq_len_combined; ++l) {
                    const int64_t row_idx = b * seq_len_combined + l;
                    const auto x_l_row = current_sequence.row(row_idx);
                    VectorXf x_l = x_l_row.transpose();

                    HNNForwardState& state = hnn_forward_states[b][l];
                    state.q_t = q_t;
                    state.p_t = p_t;
                    state.x_t = x_l;

                    VectorXf z(D_in_hnn);
                    z << q_t, p_t, x_l;
                    state.int1 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    state.dH_dz1 = compute_dH_dz(state.int1, W1_map, W2_map, W3_map);
                    state.p_half = p_t - (epsilon / 2.0f) * state.dH_dz1.head(D_state);

                    z.segment(D_state, D_state) = state.p_half;
                    state.int2 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    state.dH_dz2 = compute_dH_dz(state.int2, W1_map, W2_map, W3_map);
                    VectorXf q_next = q_t + epsilon * state.dH_dz2.segment(D_state, D_state);

                    z.head(D_state) = q_next;
                    state.int3 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    state.dH_dz3 = compute_dH_dz(state.int3, W1_map, W2_map, W3_map);
                    VectorXf p_next = state.p_half - (epsilon / 2.0f) * state.dH_dz3.head(D_state);

                    VectorXf final_state_vec(2 * D_state);
                    final_state_vec << q_next, p_next;
                    VectorXf output_l = (final_state_vec.transpose() * W_out_map).transpose() + b_out_map;

                    state.q_next = q_next;
                    state.p_next = p_next;
                    q_t = q_next;
                    p_t = p_next;
                    next_sequence.row(row_idx) = output_l.transpose();
                }
            }
        });

    current_sequence = next_sequence;
    float_state_idx += 2;
    weight_idx += 9;
}

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_FORWARD_KERNEL_H_
