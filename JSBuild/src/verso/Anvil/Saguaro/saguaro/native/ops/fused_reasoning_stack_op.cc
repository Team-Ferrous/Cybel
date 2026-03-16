// src/ops/fused_reasoning_stack_op.cc
// Copyright 2025 Verso Industries
//
// This file contains the TensorFlow operator and kernel registrations for the
// FusedReasoningStack meta-operator. The actual compute logic is implemented
// in the included header files.

#include <utility>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "ops/fused_reasoning_stack/fused_reasoning_stack_kernel.h"
#include "common/edition_limits.h"

namespace tensorflow {

// =============================================================================
// Op Registration (Forward & Backward)
// =============================================================================

REGISTER_OP("FusedReasoningStack")
    .Input("sequence_input: float")     // Input 0: [B, L_combined, D_embed]
    .Input("training: bool")            // Input 1: [1]
    .Input("block_types: string")       // Input 2: [Num_Blocks]
    .Input("block_weight_counts: int32")      // Input 3: [Num_Blocks]
    .Input("block_descriptors: string")       // Input 4: [Num_Blocks]
    .Input("initial_float_states: N * float") // Following inputs
    .Input("initial_int_states: P * int32")
    .Input("weights: M * float")
    .Attr("N: int >= 0") 
    .Attr("P: int >= 0")
    .Attr("M: int >= 0")
    .Output("output_sequence: float")   // Output 0
    .Output("final_float_states: N * float")  // Output 1+
    .Output("final_int_states: P * int32")    // Output 1+N+
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        int n, p;
        TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
        TF_RETURN_IF_ERROR(c->GetAttr("P", &p));
        // initial_float_states starts after block descriptors (index 5)
        for (int i = 0; i < n; ++i) {
            c->set_output(i + 1, c->input(i + 5));
        }
        // final_int_states starts at output index 1+N
        // initial_int_states starts at input index 5+N
        for (int i = 0; i < p; ++i) {
            c->set_output(i + 1 + n, c->input(i + 5 + n));
        }
        return OkStatus();
    });

REGISTER_OP("FusedReasoningStackGrad")
    .Input("grad_output_sequence: float")
    .Input("grad_final_float_states: N * float")
    .Input("sequence_input: float")
    .Input("block_types: string")
    .Input("block_weight_counts: int32")
    .Input("block_descriptors: string")
    .Input("initial_float_states: N * float")
    .Input("initial_int_states: P * int32")
    .Input("weights: M * float")
    .Attr("N: int >= 0")
    .Attr("P: int >= 0")
    .Attr("M: int >= 0")
    .Output("grad_sequence_input: float")
    .Output("grad_initial_float_states: N * float")
    .Output("grad_weights: M * float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int n, p, m;
        TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
        TF_RETURN_IF_ERROR(c->GetAttr("P", &p));
        TF_RETURN_IF_ERROR(c->GetAttr("M", &m));

        const int grad_final_offset = 1 + n;
        const int sequence_input_index = grad_final_offset;
        const int block_types_index = sequence_input_index + 1;
        const int block_weight_counts_index = block_types_index + 1;
        const int block_descriptors_index = block_weight_counts_index + 1;
        const int initial_float_start = block_descriptors_index + 1;
        const int initial_int_start = initial_float_start + n;
        const int weights_start = initial_int_start + p;

        c->set_output(0, c->input(sequence_input_index));
        for (int i = 0; i < n; ++i) {
            c->set_output(i + 1, c->input(initial_float_start + i));
        }
        for (int i = 0; i < m; ++i) {
            c->set_output(1 + n + i, c->input(weights_start + i));
        }
        return OkStatus();
    });

// =============================================================================
// Kernel Registration
// =============================================================================

class FusedReasoningStackOpCpu : public OpKernel {
public:
    explicit FusedReasoningStackOpCpu(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& sequence_input_tensor = context->input(0);
        const Tensor& block_types_tensor = context->input(2);
        const Tensor& block_weight_counts_tensor = context->input(3);
        const Tensor& block_descriptors_tensor = context->input(4);
        
        // HighNoon Lite Edition: Enforce reasoning block limit (max 24)
        const int64_t num_blocks = block_types_tensor.dim_size(0);
        SAGUARO_CHECK_REASONING_BLOCKS(context, num_blocks);
        
        OpInputList initial_float_states;
        OP_REQUIRES_OK(context, context->input_list("initial_float_states", &initial_float_states));
        OpInputList initial_int_states;
        OP_REQUIRES_OK(context, context->input_list("initial_int_states", &initial_int_states));
        OpInputList weights;
        OP_REQUIRES_OK(context, context->input_list("weights", &weights));

        Tensor* output_sequence_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, sequence_input_tensor.shape(), &output_sequence_tensor));

        OpOutputList final_float_states_out;
        OP_REQUIRES_OK(context, context->output_list("final_float_states", &final_float_states_out));
        OpOutputList final_int_states_out;
        OP_REQUIRES_OK(context, context->output_list("final_int_states", &final_int_states_out));

        ComputeFusedReasoningStackForward(
            context,
            sequence_input_tensor,
            block_types_tensor,
            block_weight_counts_tensor,
            block_descriptors_tensor,
            initial_float_states,
            initial_int_states,
            weights,
            output_sequence_tensor,
            &final_float_states_out,
            &final_int_states_out
        );
    }
};


REGISTER_KERNEL_BUILDER(Name("FusedReasoningStack").Device(DEVICE_CPU), FusedReasoningStackOpCpu);

class FusedReasoningStackGradOp : public OpKernel {
public:
    explicit FusedReasoningStackGradOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("N", &n_));
        OP_REQUIRES_OK(context, context->GetAttr("P", &p_));
        OP_REQUIRES_OK(context, context->GetAttr("M", &m_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output_tensor = context->input(0);
        OP_REQUIRES(context, grad_output_tensor.dtype() == DT_FLOAT,
                    errors::InvalidArgument("grad_output_sequence must be DT_FLOAT"));

        OpInputList grad_final_float_states_list;
        OP_REQUIRES_OK(context,
                       context->input_list("grad_final_float_states", &grad_final_float_states_list));
        OP_REQUIRES(context, grad_final_float_states_list.size() == n_,
                    errors::InvalidArgument("Expected ", n_,
                                            " grad_final_float_states tensors, received ",
                                            grad_final_float_states_list.size()));

        const int sequence_input_index = 1 + n_;
        const Tensor& sequence_input_original = context->input(sequence_input_index);
        const Tensor& block_types_original = context->input(sequence_input_index + 1);
        const Tensor& block_weight_counts_original = context->input(sequence_input_index + 2);
        const Tensor& block_descriptors_original = context->input(sequence_input_index + 3);

        OpInputList initial_float_states_original;
        OP_REQUIRES_OK(context,
                       context->input_list("initial_float_states", &initial_float_states_original));
        OP_REQUIRES(context, initial_float_states_original.size() == n_,
                    errors::InvalidArgument("Expected ", n_,
                                            " initial_float_states tensors, received ",
                                            initial_float_states_original.size()));

        OpInputList initial_int_states_original;
        OP_REQUIRES_OK(context,
                       context->input_list("initial_int_states", &initial_int_states_original));
        OP_REQUIRES(context, initial_int_states_original.size() == p_,
                    errors::InvalidArgument("Expected ", p_,
                                            " initial_int_states tensors, received ",
                                            initial_int_states_original.size()));

        OpInputList weights_original;
        OP_REQUIRES_OK(context, context->input_list("weights", &weights_original));
        OP_REQUIRES(context, weights_original.size() == m_,
                    errors::InvalidArgument("Expected ", m_,
                                            " weight tensors, received ",
                                            weights_original.size()));

        std::vector<const Tensor*> grad_final_float_states_ptrs;
        grad_final_float_states_ptrs.reserve(n_);
        std::vector<Tensor> zero_grad_storage;
        zero_grad_storage.reserve(n_);
        for (int i = 0; i < n_; ++i) {
            const Tensor& grad_tensor = grad_final_float_states_list[i];
            if (grad_tensor.dtype() == DT_INVALID) {
                Tensor zero_grad_tensor;
                OP_REQUIRES_OK(context,
                               context->allocate_temp(DT_FLOAT,
                                                      initial_float_states_original[i].shape(),
                                                      &zero_grad_tensor));
                zero_grad_tensor.flat<float>().setZero();
                zero_grad_storage.push_back(std::move(zero_grad_tensor));
                grad_final_float_states_ptrs.push_back(&zero_grad_storage.back());
            } else {
                grad_final_float_states_ptrs.push_back(&grad_tensor);
            }
        }

        Tensor* grad_seq_in_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, sequence_input_original.shape(),
                                                &grad_seq_in_tensor));

        OpOutputList grad_initial_float_states_out;
        OP_REQUIRES_OK(context,
                       context->output_list("grad_initial_float_states",
                                            &grad_initial_float_states_out));

        OpOutputList grad_weights_out;
        OP_REQUIRES_OK(context, context->output_list("grad_weights", &grad_weights_out));

        ComputeFusedReasoningStackBackward(
            context,
            grad_output_tensor,
            grad_final_float_states_ptrs,
            sequence_input_original,
            block_types_original,
            block_weight_counts_original,
            block_descriptors_original,
            initial_float_states_original,
            weights_original,
            grad_seq_in_tensor,
            &grad_initial_float_states_out,
            &grad_weights_out,
            /*manual_grad_weight_buffers=*/nullptr);
    }

private:
    int n_;
    int p_;
    int m_;
};

REGISTER_KERNEL_BUILDER(Name("FusedReasoningStackGrad").Device(DEVICE_CPU), FusedReasoningStackGradOp);

} // namespace tensorflow
