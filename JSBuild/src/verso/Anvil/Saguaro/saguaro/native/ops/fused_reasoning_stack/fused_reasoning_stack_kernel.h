// src/ops/fused_reasoning_stack/fused_reasoning_stack_kernel.h
// Copyright 2025 Verso Industries

#ifndef TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_KERNEL_H_
#define TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_KERNEL_H_

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "ops/fused_reasoning_stack/forward_kernel.h"


namespace tensorflow {

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
    OpOutputList* final_int_states_out);

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
    const std::vector<Tensor*>* manual_grad_weight_buffers = nullptr);

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_KERNEL_H_
