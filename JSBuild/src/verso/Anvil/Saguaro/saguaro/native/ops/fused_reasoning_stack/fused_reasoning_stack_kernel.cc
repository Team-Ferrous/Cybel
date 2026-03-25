// src/ops/fused_reasoning_stack/fused_reasoning_stack_kernel.cc
// Copyright 2025 Verso Industries

#include "ops/fused_reasoning_stack/fused_reasoning_stack_kernel.h"
#include "ops/hnn_core_helpers.h"
#include "ops/fused_reasoning_stack/forward_kernel.h" // For TimeCrystalSequenceForward helpers
#include "ops/fused_qhd_timecrystal_op.h"  // F1 Phase 5.1: Fused QHD+TimeCrystal
#include "ops/fused_wlam_moe_op.h"         // F1 Phase 5.2: Fused WLAM+MoE
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "absl/synchronization/mutex.h" // For thread-safe gradient accumulation
#include "common/parallel/parallel_backend.h"
#include <algorithm>
#include <cctype>
#include <string>
#include <chrono>  // F1 Phase 7.1: Kernel timing

namespace tensorflow {

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;
using Eigen::RowMajor;

// =============================================================================
// Stateless Block Helpers (for recomputation and backward pass)
// =============================================================================

namespace {

struct TimeCrystalDriftMetadata {
    bool emit_drift = false;
    int64_t drift_state_slot = -1;
};

inline BlockDescriptorInfo BuildFallbackDescriptor(const tstring& json_descriptor,
                                                   const std::string& block_type,
                                                   int weight_count) {
    BlockDescriptorInfo descriptor;
    // Preserve original text only for logging; avoid re-parsing downstream.
    descriptor.raw_json = json_descriptor;
    descriptor.type = block_type.empty() ? "UnknownBlock" : block_type;
    descriptor.weight_count = weight_count;
    descriptor.stateful = false;
    return descriptor;
}

inline BlockDescriptorInfo BuildShallowDescriptor(const tstring& json_descriptor,
                                                  const std::string& block_type,
                                                  int weight_count) {
    BlockDescriptorInfo descriptor;
    descriptor.raw_json = tstring();  // leave empty to avoid downstream parsing
    descriptor.type = block_type.empty() ? "UnknownBlock" : block_type;
    descriptor.weight_count = weight_count;
    descriptor.stateful = false;
    return descriptor;
}

inline bool ParseMetadataBoolValue(const std::string& raw_value, bool* out) {
    if (out == nullptr) {
        return false;
    }
    std::string normalized;
    normalized.reserve(raw_value.size());
    for (char c : raw_value) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isspace(uc) || c == '"') {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(uc)));
    }
    if (normalized == "true" || normalized == "1") {
        *out = true;
        return true;
    }
    if (normalized == "false" || normalized == "0") {
        *out = false;
        return true;
    }
    return false;
}

inline TimeCrystalDriftMetadata ParseTimeCrystalDriftMetadata(
    const BlockDescriptorInfo& descriptor) {
    TimeCrystalDriftMetadata metadata;
    if (descriptor.raw_json.empty()) {
        return metadata;
    }

    BlockDescriptorInfo parsed_descriptor;
    const Status parse_status =
        ParseBlockDescriptor(descriptor.raw_json, &parsed_descriptor);
    if (!parse_status.ok()) {
        LOG_FIRST_N(WARNING, 5)
            << "[FusedReasoningStack] Failed to parse block descriptor metadata "
            << "for TimeCrystal drift emission: " << parse_status.message();
        return metadata;
    }

    std::string emit_drift_raw;
    if (parsed_descriptor.TryGetMetadataString("emit_drift", &emit_drift_raw)) {
        bool emit_drift = false;
        if (ParseMetadataBoolValue(emit_drift_raw, &emit_drift)) {
            metadata.emit_drift = emit_drift;
        } else {
            LOG_FIRST_N(WARNING, 5)
                << "[FusedReasoningStack] Invalid emit_drift metadata value '"
                << emit_drift_raw << "'; expected true/false or 1/0.";
        }
    }

    int64_t slot = -1;
    if (parsed_descriptor.TryGetMetadataInt("drift_state_slot", &slot)) {
        metadata.drift_state_slot = slot;
    }

    return metadata;
}

inline Tensor BuildTimeCrystalDriftTensor(
    const BlockContext& ctx,
    const std::vector<std::vector<HNNForwardState>>& hnn_forward_states) {
    Tensor drift_seq_tensor(
        DT_FLOAT, TensorShape({ctx.batch_size, ctx.seq_len_combined}));
    auto drift_seq = drift_seq_tensor.matrix<float>();
    constexpr float kDriftDenomEpsilon = 1e-6f;
    for (int b = 0; b < ctx.batch_size; ++b) {
        for (int l = 0; l < ctx.seq_len_combined; ++l) {
            const HNNForwardState& state = hnn_forward_states[b][l];
            const float h_initial = state.int1.H;
            const float h_final = state.int3.H;
            drift_seq(b, l) =
                std::abs(h_final - h_initial) /
                (std::abs(h_initial) + kDriftDenomEpsilon);
        }
    }
    return drift_seq_tensor;
}

inline void MaybeWriteDriftState(
    const Tensor& drift_seq_tensor,
    int64_t drift_state_slot,
    int num_float_states,
    std::vector<Tensor>* final_float_state_tensors) {
    if (final_float_state_tensors == nullptr) {
        return;
    }
    if (drift_state_slot < 0 || drift_state_slot >= num_float_states) {
        LOG_FIRST_N(WARNING, 5)
            << "[FusedReasoningStack] drift_state_slot=" << drift_state_slot
            << " is out of range for " << num_float_states
            << " float states; skipping drift emission.";
        return;
    }
    if (drift_state_slot >=
        static_cast<int64_t>(final_float_state_tensors->size())) {
        LOG_FIRST_N(WARNING, 5)
            << "[FusedReasoningStack] drift_state_slot=" << drift_state_slot
            << " exceeds final float state tensor list size="
            << static_cast<int64_t>(final_float_state_tensors->size())
            << "; skipping drift emission.";
        return;
    }
    Tensor& target = (*final_float_state_tensors)[drift_state_slot];
    if (target.dtype() != DT_FLOAT) {
        LOG_FIRST_N(WARNING, 5)
            << "[FusedReasoningStack] drift_state_slot=" << drift_state_slot
            << " is not float32; skipping drift emission.";
        return;
    }
    if (target.NumElements() != drift_seq_tensor.NumElements()) {
        LOG_FIRST_N(WARNING, 5)
            << "[FusedReasoningStack] drift_state_slot=" << drift_state_slot
            << " has " << target.NumElements()
            << " elements, expected " << drift_seq_tensor.NumElements()
            << "; skipping drift emission.";
        return;
    }

    const std::size_t num_values =
        static_cast<std::size_t>(drift_seq_tensor.NumElements());
    std::copy_n(drift_seq_tensor.flat<float>().data(),
                num_values,
                target.flat<float>().data());
}

}  // namespace

inline void StatelessBlockBackward(
    OpKernelContext* context,
    const BlockDescriptorInfo& descriptor,
    MatrixXf& grad_adj_state,
    const MatrixXf& input_seq,
    const OpInputList& weights,
    int weight_idx_start,
    int num_weights_to_consume,
    std::vector<Tensor*>& grad_weights_tensors,
    absl::Mutex& grad_mutex) {

    if (num_weights_to_consume <= 0) {
        // Identity/no-op blocks legitimately consume zero tensors; nothing to backprop.
        return;
    }

    const int pair_count = num_weights_to_consume / 2;
    std::string failure_reason;
    if (!stateless_internal::CanRunDenseStatelessBlock(
            input_seq,
            weights,
            weight_idx_start,
            num_weights_to_consume,
            &failure_reason)) {
        LOG_FIRST_N(INFO, 3)
            << "[FusedReasoningStack] Stateless block backward skipped for block="
            << (descriptor.type.empty() ? "UnknownBlock" : descriptor.type)
            << " reason=" << failure_reason
            << " weight_idx=" << weight_idx_start
            << " tensors=" << num_weights_to_consume;
        return;
    }

    std::vector<MatrixXf> layer_inputs;
    layer_inputs.reserve(pair_count + 1);
    std::vector<MatrixXf> pre_activations;
    pre_activations.reserve(pair_count);
    [[maybe_unused]] MatrixXf block_output = stateless_internal::RunStatelessDenseForward(
        context, input_seq, weights, weight_idx_start, num_weights_to_consume,
        &layer_inputs, &pre_activations);

    if (!context->status().ok()) {
        return;
    }

    if (layer_inputs.size() != pair_count + 1 ||
        pre_activations.size() != pair_count) {
        context->CtxFailure(errors::Internal(
            "Stateless block cache invariant violated during gradient computation."));
        return;
    }

    MatrixXf residual_grad = grad_adj_state;
    MatrixXf grad_current = residual_grad;

    for (int layer = pair_count - 1; layer >= 0; --layer) {
        const int weight_offset = weight_idx_start + 2 * layer;
        if (weight_offset + 1 >= grad_weights_tensors.size()) {
            context->CtxFailure(errors::InvalidArgument(
                "Gradient tensor list too small for stateless block layer ", layer));
            return;
        }

        const Tensor& weight_tensor = weights[weight_offset];
        const Tensor& bias_tensor = weights[weight_offset + 1];

        stateless_internal::FlattenedWeightShape weight_shape;
        if (!stateless_internal::ResolveWeightShape(context, weight_tensor, layer,
                                                    &weight_shape)) {
            return;
        }

        Map<const MatrixXf> weight_map(weight_tensor.flat<float>().data(),
                                       weight_shape.input_dim,
                                       weight_shape.output_dim);
        Map<MatrixXf> grad_weight_map(
            grad_weights_tensors[weight_offset]->flat<float>().data(),
            weight_shape.input_dim,
            weight_shape.output_dim);
        Map<VectorXf> grad_bias_map(
            grad_weights_tensors[weight_offset + 1]->flat<float>().data(),
            weight_shape.output_dim);

        const MatrixXf& layer_input = layer_inputs[layer];
        MatrixXf grad_weight_contrib = layer_input.transpose() * grad_current;
        VectorXf grad_bias_contrib = grad_current.colwise().sum();

        {
            absl::MutexLock lock(&grad_mutex);
            grad_weight_map.noalias() += grad_weight_contrib;
            grad_bias_map += grad_bias_contrib;
        }

        MatrixXf grad_prev = grad_current * weight_map.transpose();

        if (layer > 0) {
            const MatrixXf& pre_act_prev = pre_activations[layer - 1];
            MatrixXf activation_grad = pre_act_prev.unaryExpr([](float v) {
                return stateless_internal::gelu_grad(v);
            });
            grad_current = grad_prev.cwiseProduct(activation_grad);
        } else {
            grad_adj_state = grad_prev + residual_grad;
        }
    }
}


void ComputeFusedReasoningStackForward(
    OpKernelContext* context,
    const Tensor& sequence_input_tensor,
    const Tensor& block_types_tensor,
    const Tensor& block_weight_counts_tensor,
    const Tensor& block_descriptors_tensor,
    const OpInputList& initial_float_states,
    const OpInputList& initial_int_states,
    const OpInputList& weights,
    Tensor* output_sequence_tensor,
    OpOutputList* final_float_states_out,
    OpOutputList* final_int_states_out) {
    // Implementation moved from forward_kernel.h
    const int64_t batch_size = sequence_input_tensor.dim_size(0);
    const int64_t seq_len_combined = sequence_input_tensor.dim_size(1);
    const int64_t d_embed = sequence_input_tensor.dim_size(2);
    const int64_t total_rows = batch_size * seq_len_combined;

    BlockContext ctx = {context, batch_size, seq_len_combined, d_embed};

    Map<const MatrixXf, RowMajor> input_map(sequence_input_tensor.flat<float>().data(), total_rows, d_embed);
    MatrixXf current_sequence = input_map;

    auto block_weight_counts = block_weight_counts_tensor.flat<int32>();
    auto block_types = block_types_tensor.flat<tstring>();
    auto block_descriptors = block_descriptors_tensor.flat<tstring>();
    const int num_blocks = block_types.size();
    OP_REQUIRES(context, block_descriptors.size() == num_blocks,
                errors::InvalidArgument(
                    "block_descriptors length (", block_descriptors.size(),
                    ") must match number of block types (", num_blocks, ")"));
    std::vector<BlockDescriptorInfo> parsed_descriptors(num_blocks);
    int64_t total_reasoning_weights = 0;
    for (int i = 0; i < num_blocks; ++i) {
        OP_REQUIRES(context, block_weight_counts(i) >= 0,
                    errors::InvalidArgument("Block ", i,
                                            " weight count must be non-negative."));
        total_reasoning_weights += block_weight_counts(i);
    }
    OP_REQUIRES(context, weights.size() >= total_reasoning_weights,
                errors::InvalidArgument(
                    "Reasoning stack expects ", total_reasoning_weights,
                    " weight tensors but only ", weights.size(), " were provided."));
    const int reasoning_weight_base =
        static_cast<int>(weights.size() - total_reasoning_weights);
    for (int i = 0; i < num_blocks; ++i) {
        const std::string block_type_str(block_types(i).data(), block_types(i).size());
        const bool has_descriptor = block_descriptors(i).size() > 0;
        // Preserve raw JSON when available (needed for TT metadata); fall back to shallow if missing.
        parsed_descriptors[i] = has_descriptor
            ? BuildFallbackDescriptor(block_descriptors(i), block_type_str, block_weight_counts(i))
            : BuildShallowDescriptor(block_descriptors(i), block_type_str, block_weight_counts(i));
    }
    // Model-agnostic embedding validation: if descriptors declare embedding_dim, warn on mismatch.
    int64_t declared_embed = -1;
    for (int i = 0; i < num_blocks; ++i) {
        int64_t meta_embed = -1;
        if (parsed_descriptors[i].TryGetMetadataInt("embedding_dim", &meta_embed)) {
            declared_embed = meta_embed;
            break;
        }
    }
    if (declared_embed > 0 && declared_embed != d_embed) {
        LOG_FIRST_N(WARNING, 5)
            << "[FusedReasoningStack] embedding_dim mismatch: descriptor "
            << declared_embed << " vs tensor " << d_embed
            << ". Proceeding for model-agnostic use.";
    }
    const int num_float_states = initial_float_states.size();
    const int num_int_states = initial_int_states.size();

    std::vector<Tensor> final_float_state_tensors;
    final_float_state_tensors.reserve(num_float_states);
    for (int i = 0; i < num_float_states; ++i) {
        final_float_state_tensors.emplace_back(initial_float_states[i]);
    }
    std::vector<Tensor> final_int_state_tensors;
    final_int_state_tensors.reserve(num_int_states);
    for (int i = 0; i < num_int_states; ++i) {
        final_int_state_tensors.emplace_back(initial_int_states[i]);
    }

    int float_state_idx = 0;
    int int_state_idx = 0;
    int weight_idx = reasoning_weight_base;

    MatrixXf block_input_sequence;
    for (int i = 0; i < num_blocks; ++i) {
        const int num_weights_to_consume = block_weight_counts(i);
        const std::string& block_type = block_types(i);
        const BlockDescriptorInfo& descriptor = parsed_descriptors[i];
        block_input_sequence = current_sequence;

        int64_t meta_chunk_size = 0;
        int64_t meta_chunk_stride = 0;
        const bool has_chunk_size = descriptor.TryGetMetadataInt("chunk_size", &meta_chunk_size);
        const bool has_chunk_stride = descriptor.TryGetMetadataInt("chunk_stride", &meta_chunk_stride);
        if (has_chunk_size && has_chunk_stride && meta_chunk_stride > meta_chunk_size) {
            LOG_FIRST_N(WARNING, 10)
                << "[FusedReasoningStack] Skipping block index " << i
                << " due to chunk_stride (" << meta_chunk_stride
                << ") > chunk_size (" << meta_chunk_size << "). descriptor="
                << descriptor.raw_json;
            weight_idx += num_weights_to_consume;
            continue;
        }

        OP_REQUIRES(context, descriptor.weight_count == num_weights_to_consume,
                    errors::InvalidArgument(
                        "Descriptor weight count (", descriptor.weight_count,
                        ") does not match block_weight_counts entry (",
                        num_weights_to_consume, ") for block type '",
                        descriptor.type, "'"));

        if (block_type == "TimeCrystalSequenceBlock") {
            const TimeCrystalDriftMetadata drift_metadata =
                ParseTimeCrystalDriftMetadata(descriptor);
            std::vector<std::vector<HNNForwardState>> local_hnn_states(
                ctx.batch_size,
                std::vector<HNNForwardState>(ctx.seq_len_combined));
            TimeCrystalSequenceForward(ctx,
                                       current_sequence,
                                       initial_float_states,
                                       weights,
                                       float_state_idx,
                                       weight_idx,
                                       final_float_state_tensors,
                                       local_hnn_states);
            if (!context->status().ok()) {
                return;
            }
            if (drift_metadata.emit_drift) {
                Tensor drift_seq_tensor =
                    BuildTimeCrystalDriftTensor(ctx, local_hnn_states);
                MaybeWriteDriftState(drift_seq_tensor,
                                     drift_metadata.drift_state_slot,
                                     num_float_states,
                                     &final_float_state_tensors);
            }
        } else if (block_type == "SpatialBlock" ||
                   block_type == "ReasoningMamba2Block" ||
                   block_type == "WLAMBlock" ||
                   block_type == "MoELayer" ||
                   block_type == "QHDSpatialBlock" ||       // Phase 3.1: HD spatial with quantum paths
                   block_type == "HDTimeCrystalBlock" ||    // Phase 3.2: HD time crystal dynamics
                   block_type == "LatentReasoningBlock" ||  // Phase 3.3: COCONUT reasoning
                   block_type == "KalmanBlock" ||           // Phase 3.4: Neural Kalman filtering
                   block_type == "FusedQHDTimeCrystalBlock" ||  // F1 Phase 5.1: Fused QHD+TimeCrystal
                   block_type == "FusedWLAMMoEBlock") {         // F1 Phase 5.2: Fused WLAM+MoE
            MatrixXf transformed = StatelessBlockForward(
                ctx,
                descriptor,
                current_sequence,
                weights,
                weight_idx,
                num_weights_to_consume);
            if (!context->status().ok()) {
                return;
            }
            if (transformed.size() == 0) {
                continue;
            }
            current_sequence = transformed;
        } else {
            context->CtxFailure(errors::Unimplemented(
                "FusedReasoningStack forward kernel encountered unknown block type '",
                block_type, "'. Descriptor=", descriptor.raw_json));
            return;
        }
    }

    Map<MatrixXf, RowMajor>(output_sequence_tensor->flat<float>().data(), total_rows, d_embed) = current_sequence;

    for (int i = 0; i < num_float_states; ++i) {
        final_float_states_out->set(i, final_float_state_tensors[i]);
    }
    for (int i = 0; i < num_int_states; ++i) {
        final_int_states_out->set(i, final_int_state_tensors[i]);
    }
}

void ComputeFusedReasoningStackBackward(
    OpKernelContext* context,
    const Tensor& grad_output_tensor,
    const std::vector<const Tensor*>& grad_final_states,
    const Tensor& sequence_input_tensor,
    const Tensor& block_types_tensor,
    const Tensor& block_weight_counts_tensor,
    const Tensor& block_descriptors_tensor,
    const OpInputList& initial_states,
    const OpInputList& weights,
    Tensor* grad_seq_in_tensor,
    OpOutputList* grad_initial_states_out,
    OpOutputList* grad_weights_out,
    const std::vector<Tensor*>* manual_grad_weight_buffers) {
    
    VLOG(1) << "[FusedReasoningStackBackward] Starting backward pass for "
            << block_types_tensor.NumElements() << " blocks.";

    // --- 2. Dimensions and Setup ---
    const int64_t batch_size = sequence_input_tensor.dim_size(0);
    const int64_t seq_len_combined = sequence_input_tensor.dim_size(1);
    const int64_t d_embed = sequence_input_tensor.dim_size(2);
    const int64_t total_rows = batch_size * seq_len_combined;
    const auto block_types = block_types_tensor.flat<tstring>();
    const auto block_weight_counts = block_weight_counts_tensor.flat<int32>();
    const auto block_descriptors = block_descriptors_tensor.flat<tstring>();
    const int num_blocks = block_types.size();
    OP_REQUIRES(context, block_descriptors.size() == num_blocks,
                errors::InvalidArgument(
                    "block_descriptors length (", block_descriptors.size(),
                    ") must match number of block types (", num_blocks, ")"));
    std::vector<BlockDescriptorInfo> parsed_descriptors(num_blocks);
    int64_t total_reasoning_weights = 0;
    for (int i = 0; i < num_blocks; ++i) {
        OP_REQUIRES(context, block_weight_counts(i) >= 0,
                    errors::InvalidArgument("Block ", i,
                                            " weight count must be non-negative."));
        total_reasoning_weights += block_weight_counts(i);
    }
    OP_REQUIRES(context, weights.size() >= total_reasoning_weights,
                errors::InvalidArgument(
                    "Reasoning stack expects ", total_reasoning_weights,
                    " weight tensors but only ", weights.size(), " were provided."));
    const int reasoning_weight_base =
        static_cast<int>(weights.size() - total_reasoning_weights);
    for (int i = 0; i < num_blocks; ++i) {
        parsed_descriptors[i] = BuildShallowDescriptor(
            block_descriptors(i),
            std::string(block_types(i).data(), block_types(i).size()),
            block_weight_counts(i));
    }
    int num_states = initial_states.size();
    int num_weights = weights.size();

    // --- 3. Allocate and Initialize Outputs ---
    Map<const MatrixXf, RowMajor> grad_output_map(grad_output_tensor.flat<float>().data(), total_rows, d_embed);
    
    absl::Mutex stateless_grad_mutex; // Declare the mutex here
    
    std::vector<Tensor*> grad_initial_states_tensors;
    for(int i = 0; i < num_states; ++i) {
        Tensor* grad_s = nullptr;
        OP_REQUIRES_OK(context, grad_initial_states_out->allocate(i, initial_states[i].shape(), &grad_s));
        grad_s->flat<float>().setZero();
        grad_initial_states_tensors.push_back(grad_s);
    }



    std::vector<Tensor*> grad_weights_tensors;
    if (manual_grad_weight_buffers && !manual_grad_weight_buffers->empty()) {
        grad_weights_tensors = *manual_grad_weight_buffers;
        OP_REQUIRES(context, grad_weights_tensors.size() == num_weights,
                    errors::InvalidArgument(
                        "Manual gradient buffer count mismatch: expected ",
                        num_weights, " but received ",
                        grad_weights_tensors.size()));
        for (int i = 0; i < num_weights; ++i) {
            OP_REQUIRES(context, grad_weights_tensors[i] != nullptr,
                        errors::InvalidArgument(
                            "Manual gradient buffer at index ", i, " is null."));
        }
    } else {
        OP_REQUIRES(context, grad_weights_out != nullptr,
                    errors::InvalidArgument(
                        "grad_weights_out must not be null when no manual "
                        "gradient buffers are provided."));
        grad_weights_tensors.reserve(num_weights);
        for(int i = 0; i < num_weights; ++i) {
            Tensor* grad_w = nullptr;
            OP_REQUIRES_OK(context, grad_weights_out->allocate(i, weights[i].shape(), &grad_w));
            grad_w->flat<float>().setZero();
            grad_weights_tensors.push_back(grad_w);
        }
    }
    
    BlockContext ctx = {context, batch_size, seq_len_combined, d_embed};

    // Initialize the adjoint state as a mutable copy of the upstream gradient
    MatrixXf grad_adj_state = grad_output_map;

    // --- 4. Forward Pass Recomputation (to get intermediate states) ---
    // Duplicate the forward pass to get input_seq (x_l) for each block and HNN intermediates.
    std::vector<MatrixXf> forward_sequences;
    forward_sequences.push_back(Map<const MatrixXf, RowMajor>(sequence_input_tensor.flat<float>().data(), total_rows, d_embed));
    
    // --- FIX: Pre-allocate vector to avoid reallocations of non-copyable HNNForwardState ---
    int num_hnn_blocks = 0;
    for (int i = 0; i < num_blocks; ++i) {
        if (block_types(i) == "TimeCrystalSequenceBlock") {
            num_hnn_blocks++;
        }
    }
    std::vector<std::vector<std::vector<HNNForwardState>>> all_hnn_forward_states;
    all_hnn_forward_states.reserve(num_hnn_blocks);
    std::vector<int> hnn_block_indices; 

    int state_idx_fwd = 0;
    int weight_idx_fwd = reasoning_weight_base;
    
    for (int i = 0; i < num_blocks; ++i) {
        const int num_weights_to_consume = block_weight_counts(i);
        MatrixXf current_sequence_fwd = forward_sequences.back();
        const std::string& block_type = block_types(i);
        const BlockDescriptorInfo& descriptor = parsed_descriptors[i];
        MatrixXf next_sequence;

        int64_t meta_chunk_size = 0;
        int64_t meta_chunk_stride = 0;
        const bool has_chunk_size = descriptor.TryGetMetadataInt("chunk_size", &meta_chunk_size);
        const bool has_chunk_stride = descriptor.TryGetMetadataInt("chunk_stride", &meta_chunk_stride);
        if (has_chunk_size && has_chunk_stride && meta_chunk_stride > meta_chunk_size) {
            LOG_FIRST_N(WARNING, 10)
                << "[FusedReasoningStack] (recompute) Skipping block index " << i
                << " due to chunk_stride (" << meta_chunk_stride
                << ") > chunk_size (" << meta_chunk_size << "). descriptor="
                << descriptor.raw_json;
            weight_idx_fwd += num_weights_to_consume;
            forward_sequences.push_back(current_sequence_fwd);
            continue;
        }

        if (block_type == "TimeCrystalSequenceBlock") {
            hnn_block_indices.push_back(i);
            // Allocate space for intermediates (specific to TimeCrystal/HNN)
            all_hnn_forward_states.emplace_back(batch_size, std::vector<HNNForwardState>(seq_len_combined));

            const Tensor& h_padded_tensor = initial_states[state_idx_fwd];
            const Tensor& W1 = weights[weight_idx_fwd];
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
            const Tensor& W1_w = weights[weight_idx_fwd++]; 
            const Tensor& b1 = weights[weight_idx_fwd++];
            const Tensor& W2 = weights[weight_idx_fwd++]; 
            const Tensor& b2 = weights[weight_idx_fwd++];
            const Tensor& W3 = weights[weight_idx_fwd++]; 
            const Tensor& b3 = weights[weight_idx_fwd++]; // SCALAR
            const Tensor& epsilon_param = weights[weight_idx_fwd++]; // SCALAR
            const Tensor& W_out = weights[weight_idx_fwd++]; 
            const Tensor& b_out = weights[weight_idx_fwd++];

            // Use the same check as the forward pass for consistency
            OP_REQUIRES(context, b3.NumElements() == 1,
                        errors::InvalidArgument("HNN b3 bias must be a 1-element tensor. got ",
                                                b3.shape().DebugString()));
            OP_REQUIRES(context, epsilon_param.NumElements() == 1,
                        errors::InvalidArgument("HNN epsilon_param must be a 1-element tensor. got ",
                                                epsilon_param.shape().DebugString()));


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
            
            next_sequence.resize(total_rows, d_embed);

            saguaro::parallel::ForRange(
                0, static_cast<std::size_t>(batch_size), 1,
                [&](std::size_t range_begin, std::size_t range_end) {
                    for (std::size_t idx = range_begin; idx < range_end; ++idx) {
                        const int b = static_cast<int>(idx);
                        VectorXf q_t = Map<const VectorXf>(q_init_tensor.flat<float>().data() + b * D_state, D_state);
                        VectorXf p_t = Map<const VectorXf>(p_init_tensor.flat<float>().data() + b * D_state, D_state);

                        for (int l = 0; l < seq_len_combined; ++l) {
                            const int64_t row_idx = b * seq_len_combined + l;
                            const auto x_l_row = current_sequence_fwd.row(row_idx);
                            VectorXf x_l = x_l_row.transpose();

                            HNNForwardState& state = all_hnn_forward_states.back()[b][l];
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
            state_idx_fwd += 2;
        } else if (block_type == "SpatialBlock" ||
                   block_type == "ReasoningMamba2Block" ||
                   block_type == "WLAMBlock" ||
                   block_type == "MoELayer" ||
                   block_type == "QHDSpatialBlock" ||       // Phase 3.1: HD spatial with quantum paths
                   block_type == "HDTimeCrystalBlock" ||    // Phase 3.2: HD time crystal dynamics
                   block_type == "LatentReasoningBlock" ||  // Phase 3.3: COCONUT reasoning
                   block_type == "KalmanBlock") {           // Phase 3.4: Neural Kalman filtering
            MatrixXf transformed = StatelessBlockForward(
                ctx,
                descriptor,
                current_sequence_fwd,
                weights,
                weight_idx_fwd,
                num_weights_to_consume);
            if (!context->status().ok()) {
                return;
            }
            if (transformed.size() == 0) {
                forward_sequences.push_back(current_sequence_fwd);
                continue;
            }
            next_sequence = transformed;
        } else {
            context->CtxFailure(errors::Unimplemented(
                "FusedReasoningStack backward kernel encountered unknown block type '",
                block_type, "'. Descriptor=", descriptor.raw_json));
            return;
        }
        forward_sequences.push_back(next_sequence);
    }

    // --- 5. Backward Pass (Adjoint Sensitivity Method) ---
    int hnn_block_counter = hnn_block_indices.size() - 1;
    int state_idx_bwd = num_states;
    int weight_idx_bwd = num_weights;

    for (int i = num_blocks - 1; i >= 0; --i) {
        const MatrixXf& input_seq = forward_sequences[i]; 
        const int num_weights_consumed = block_weight_counts(i);
        const std::string& block_type = block_types(i); // Correct access
        const BlockDescriptorInfo& descriptor = parsed_descriptors[i];

        int64_t meta_chunk_size = 0;
        int64_t meta_chunk_stride = 0;
        const bool has_chunk_size = descriptor.TryGetMetadataInt("chunk_size", &meta_chunk_size);
        const bool has_chunk_stride = descriptor.TryGetMetadataInt("chunk_stride", &meta_chunk_stride);
        if (has_chunk_size && has_chunk_stride && meta_chunk_stride > meta_chunk_size) {
            weight_idx_bwd -= num_weights_consumed;
            continue;
        }

        if (block_type == "TimeCrystalSequenceBlock") {
            state_idx_bwd -= 2;
            weight_idx_bwd -= 9;
            
            const auto& hnn_states_for_block = all_hnn_forward_states[hnn_block_counter];
            
            const Tensor& W1 = weights[weight_idx_bwd]; 
            const int64_t D_state = (W1.dim_size(0) - d_embed) / 2;
            const int64_t D_in_hnn = W1.dim_size(0);
            const int64_t D_h = W1.dim_size(1);
            const int64_t D_output = weights[weight_idx_bwd+8].dim_size(0); 
            
            const float epsilon_param_val = weights[weight_idx_bwd+6].scalar<float>()();
            const float epsilon = std::min(1.0f, 0.01f + 0.99f * std::tanh(epsilon_param_val));
            const float b3_scalar = weights[weight_idx_bwd+5].scalar<float>()();

            Map<const MatrixXf> W1_map(W1.flat<float>().data(), D_in_hnn, D_h);
            Map<const VectorXf> b1_map(weights[weight_idx_bwd+1].flat<float>().data(), D_h);
            Map<const MatrixXf> W2_map(weights[weight_idx_bwd+2].flat<float>().data(), D_h, D_h);
            Map<const VectorXf> b2_map(weights[weight_idx_bwd+3].flat<float>().data(), D_h);
            Map<const MatrixXf> W3_map(weights[weight_idx_bwd+4].flat<float>().data(), D_h, 1);
            Map<const MatrixXf> W_out_map(weights[weight_idx_bwd+7].flat<float>().data(), 2 * D_state, D_output);

            MatrixXf grad_W1_acc = MatrixXf::Zero(D_in_hnn, D_h);
            VectorXf grad_b1_acc = VectorXf::Zero(D_h);
            MatrixXf grad_W2_acc = MatrixXf::Zero(D_h, D_h);
            VectorXf grad_b2_acc = VectorXf::Zero(D_h);
            MatrixXf grad_W3_acc = MatrixXf::Zero(D_h, 1);
            float grad_b3_acc = 0.0f;
            MatrixXf grad_W_out_acc = MatrixXf::Zero(2 * D_state, D_output);
            VectorXf grad_b_out_acc = VectorXf::Zero(D_output);
            float grad_epsilon_param_acc = 0.0f;
            absl::Mutex grad_accum_mutex;

            MatrixXf grad_input_from_hnn = MatrixXf::Zero(total_rows, d_embed);
            

            auto* worker_threads = context->device()->tensorflow_cpu_worker_threads();
            auto work_fn = [&](int64_t start, int64_t end) {
                for (int b = start; b < end; ++b) {
                VectorXf grad_q = Map<const VectorXf>(grad_final_states[state_idx_bwd]->flat<float>().data() + b * D_state, D_state);
                VectorXf grad_p = Map<const VectorXf>(grad_final_states[state_idx_bwd+1]->flat<float>().data() + b * D_state, D_state);

                MatrixXf thread_local_grad_W1 = MatrixXf::Zero(D_in_hnn, D_h); 
                MatrixXf thread_local_grad_W2 = MatrixXf::Zero(D_h, D_h); 
                MatrixXf thread_local_grad_W3 = MatrixXf::Zero(D_h, 1);
                VectorXf thread_local_grad_b1 = VectorXf::Zero(D_h); 
                VectorXf thread_local_grad_b2 = VectorXf::Zero(D_h);
                float thread_local_grad_b3 = 0.0f;
                MatrixXf thread_local_grad_W_out = MatrixXf::Zero(2 * D_state, D_output);
                VectorXf thread_local_grad_b_out = VectorXf::Zero(D_output);
                float thread_local_grad_epsilon_param = 0.0f;

                for (int l = seq_len_combined - 1; l >= 0; --l) {
                    const HNNForwardState& state = hnn_states_for_block[b][l];
                    Map<const VectorXf> grad_output_l(grad_adj_state.data() + (b * seq_len_combined + l) * d_embed, d_embed);
                    
                    VectorXf final_state_vec(2 * D_state); final_state_vec << state.q_next, state.p_next;
                    
                    thread_local_grad_W_out += final_state_vec * grad_output_l.transpose();
                    thread_local_grad_b_out += grad_output_l;
                    
                    VectorXf grad_final_state = W_out_map * grad_output_l;
                    grad_q += grad_final_state.head(D_state);
                    grad_p += grad_final_state.tail(D_state);
                    
                    VectorXf grad_p_half = grad_p;
                    float grad_H3_scalar = grad_p.dot(-0.5f * epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int3, W1_map, W2_map, W3_map, grad_H3_scalar, thread_local_grad_W1, thread_local_grad_b1, thread_local_grad_W2, thread_local_grad_b2, thread_local_grad_W3, thread_local_grad_b3);
                    VectorXf grad_z3 = state.dH_dz3 * grad_H3_scalar;
                    
                    grad_q += grad_z3.head(D_state);
                    grad_p_half += grad_z3.segment(D_state, D_state);
                    grad_input_from_hnn.row(b * seq_len_combined + l) += grad_z3.tail(d_embed).transpose();

                    float grad_H2_scalar = grad_q.dot(epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int2, W1_map, W2_map, W3_map, grad_H2_scalar, thread_local_grad_W1, thread_local_grad_b1, thread_local_grad_W2, thread_local_grad_b2, thread_local_grad_W3, thread_local_grad_b3);
                    VectorXf grad_z2 = state.dH_dz2 * grad_H2_scalar;
                    grad_q += grad_z2.head(D_state);
                    grad_p_half += grad_z2.segment(D_state, D_state);
                    grad_input_from_hnn.row(b * seq_len_combined + l) += grad_z2.tail(d_embed).transpose();
                    
                    VectorXf grad_p_t = grad_p_half;
                    float grad_H1_scalar = grad_p_half.dot(-0.5f * epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int1, W1_map, W2_map, W3_map, grad_H1_scalar, thread_local_grad_W1, thread_local_grad_b1, thread_local_grad_W2, thread_local_grad_b2, thread_local_grad_W3, thread_local_grad_b3);
                    VectorXf grad_z1 = state.dH_dz1 * grad_H1_scalar;

                    // Propagate adjoint state to the previous time step (q_t, p_t, x_t).
                    grad_q = grad_q + grad_z1.head(D_state);
                    grad_p = grad_p_t + grad_z1.segment(D_state, D_state);
                    grad_input_from_hnn.row(b * seq_len_combined + l) += grad_z1.tail(d_embed).transpose();
                    
                    // --- Adjoint of evolution_time (epsilon) ---
                    float grad_eps = 0;
                    grad_eps -= (grad_p.transpose() * state.dH_dz3.head(D_state) / 2.0f)(0,0);
                    grad_eps += (grad_q.transpose() * state.dH_dz2.segment(D_state, D_state))(0,0);
                    grad_eps -= (grad_p_half.transpose() * state.dH_dz1.head(D_state) / 2.0f)(0,0);
                    thread_local_grad_epsilon_param += grad_eps * (0.99f * (1.0f - std::pow(std::tanh(epsilon_param_val), 2)));
                }
                
                Map<VectorXf>(grad_initial_states_tensors[state_idx_bwd]->flat<float>().data() + b * D_state, D_state) = grad_q;
                Map<VectorXf>(grad_initial_states_tensors[state_idx_bwd+1]->flat<float>().data() + b * D_state, D_state) = grad_p;
                
                {
                    absl::MutexLock lock(&grad_accum_mutex);
                    grad_W1_acc += thread_local_grad_W1;
                    grad_b1_acc += thread_local_grad_b1;
                    grad_W2_acc += thread_local_grad_W2;
                    grad_b2_acc += thread_local_grad_b2;
                    grad_W3_acc += thread_local_grad_W3;
                    grad_b3_acc += thread_local_grad_b3;
                    grad_W_out_acc += thread_local_grad_W_out;
                    grad_b_out_acc += thread_local_grad_b_out;
                    grad_epsilon_param_acc += thread_local_grad_epsilon_param;
                }

            } // End of for (int b = start; b < end; ++b)
        }; // End of work_fn lambda
        Shard(worker_threads->num_threads, worker_threads->workers, batch_size, 1000, work_fn);

            // --- DEFINITIVE FIX: Account for the residual connection in the backward pass ---
            // The gradient flowing to the previous block's output is the sum of the gradient
            // from this block's logic AND the gradient from the residual path (which is just grad_adj_state).
            grad_adj_state = grad_input_from_hnn + grad_adj_state;
            hnn_block_counter--;

            Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+0]->flat<float>().data(), D_in_hnn, D_h).array() += grad_W1_acc.array();
            Map<VectorXf>(grad_weights_tensors[weight_idx_bwd+1]->flat<float>().data(), D_h).array() += grad_b1_acc.array();
            Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+2]->flat<float>().data(), D_h, D_h).array() += grad_W2_acc.array();
            Map<VectorXf>(grad_weights_tensors[weight_idx_bwd+3]->flat<float>().data(), D_h).array() += grad_b2_acc.array();
            Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+4]->flat<float>().data(), D_h, 1).array() += grad_W3_acc.array();
            grad_weights_tensors[weight_idx_bwd+5]->scalar<float>()() += grad_b3_acc;
            grad_weights_tensors[weight_idx_bwd+6]->scalar<float>()() += grad_epsilon_param_acc;
            Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+7]->flat<float>().data(), 2 * D_state, D_output).array() += grad_W_out_acc.array();
            Map<VectorXf>(grad_weights_tensors[weight_idx_bwd+8]->flat<float>().data(), D_output).array() += grad_b_out_acc.array();

        } else {
            if (block_type == "KalmanBlock") { state_idx_bwd -= 2; }

            weight_idx_bwd -= num_weights_consumed;
            
            // This will now correctly execute the FFN logic for num_weights_consumed = 2,
            // or perform the no-op pass-through for complex blocks.
            StatelessBlockBackward(context,
                                   descriptor,
                                   grad_adj_state,
                                   input_seq,
                                   weights,
                                   weight_idx_bwd,
                                   num_weights_consumed,
                                   grad_weights_tensors,
                                   stateless_grad_mutex);
            // For StatelessBlockBackward, the grad_adj_state is updated by (grad_output * W_map.transpose()) + grad_output;
            // This is handled inside StatelessBlockBackward.
        }
    }
    
    Map<MatrixXf, RowMajor>(grad_seq_in_tensor->flat<float>().data(), total_rows, d_embed) = grad_adj_state;
}

} // namespace tensorflow
