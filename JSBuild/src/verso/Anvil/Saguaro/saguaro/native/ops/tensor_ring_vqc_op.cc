// saguaro.native/ops/tensor_ring_vqc_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// TensorFlow op registration for Tensor Ring VQC and Neural BP Mitigation.
// Phase 1005: Native implementation replacing Python version.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "tensor_ring_vqc_op.h"

#include <vector>

namespace tensorflow {

using namespace saguaro::tensor_ring;

// ============================================================================
// Op Registration: TensorRingContract
// ============================================================================

REGISTER_OP("TensorRingContract")
    .Input("cores: float32")        // [num_qubits, bond_dim, 2, bond_dim] flattened
    .Input("params: float32")       // [num_layers * num_qubits * 3]
    .Input("inputs: float32")       // [batch, features]
    .Attr("num_qubits: int = 8")
    .Attr("num_layers: int = 4")
    .Attr("bond_dim: int = 16")
    .Output("output: float32")      // [batch, num_qubits]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int num_qubits;
        TF_RETURN_IF_ERROR(c->GetAttr("num_qubits", &num_qubits));

        shape_inference::ShapeHandle inputs_shape = c->input(2);
        if (c->RankKnown(inputs_shape) && c->Rank(inputs_shape) == 2) {
            shape_inference::DimensionHandle batch = c->Dim(inputs_shape, 0);
            c->set_output(0, c->Matrix(batch, num_qubits));
        } else {
            c->set_output(0, c->UnknownShapeOfRank(2));
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Tensor Ring VQC forward pass with fused contraction.

Phase 1005 - Approximates VQC simulation using tensor ring decomposition.
Provides O(chi^3 * L * num_qubits) cost instead of O(2^n).

The cores tensor contains flattened tensor ring cores with shape
[num_qubits * bond_dim * 2 * bond_dim].

The params tensor contains variational rotation angles with shape
[num_layers * num_qubits * 3].

The inputs tensor contains input features with shape [batch, features].

The output tensor contains VQC output with shape [batch, num_qubits].
)doc");

// ============================================================================
// Op Registration: NeuralBPMitigationForward
// ============================================================================

REGISTER_OP("NeuralBPMitigationForward")
    .Input("inputs: float32")       // [batch, input_dim]
    .Input("weights_1: float32")    // [input_dim, hidden_dim]
    .Input("bias_1: float32")       // [hidden_dim]
    .Input("weights_2: float32")    // [hidden_dim, hidden_dim]
    .Input("bias_2: float32")       // [hidden_dim]
    .Input("weights_out: float32")  // [hidden_dim, output_dim]
    .Input("bias_out: float32")     // [output_dim]
    .Attr("hidden_dim: int = 64")
    .Output("output: float32")      // [batch, output_dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output shape is [batch, output_dim] where output_dim = bias_out size
        shape_inference::ShapeHandle inputs_shape = c->input(0);
        shape_inference::ShapeHandle bias_out_shape = c->input(6);

        if (c->RankKnown(inputs_shape) && c->Rank(inputs_shape) == 2 &&
            c->RankKnown(bias_out_shape) && c->Rank(bias_out_shape) == 1) {
            shape_inference::DimensionHandle batch = c->Dim(inputs_shape, 0);
            shape_inference::DimensionHandle output_dim = c->Dim(bias_out_shape, 0);
            c->set_output(0, c->Matrix(batch, output_dim));
        } else {
            c->set_output(0, c->UnknownShapeOfRank(2));
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Neural network forward pass for VQC barren plateau mitigation.

Phase 1006: Predicts VQC initial parameters that avoid barren plateau regions.
MLP with 2 hidden layers, ReLU activation, tanh output scaled by 0.1.

inputs: Circuit features [batch, input_dim].
output: Predicted initial rotation angles [batch, output_dim].
)doc");

// ============================================================================
// CPU Kernel: TensorRingContractOp
// ============================================================================

class TensorRingContractOp : public OpKernel {
public:
    explicit TensorRingContractOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_qubits", &num_qubits_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_layers", &num_layers_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bond_dim", &bond_dim_));

        LOG(INFO) << "[TensorRingContract] Initialized: qubits=" << num_qubits_
                  << ", layers=" << num_layers_ << ", bond_dim=" << bond_dim_;
    }

    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& cores_tensor = ctx->input(0);
        const Tensor& params_tensor = ctx->input(1);
        const Tensor& inputs_tensor = ctx->input(2);

        // Validate shapes
        const int expected_core_size = num_qubits_ * bond_dim_ * 2 * bond_dim_;
        OP_REQUIRES(ctx, cores_tensor.NumElements() >= expected_core_size,
                    errors::InvalidArgument(
                        "cores must have at least ", expected_core_size,
                        " elements, got ", cores_tensor.NumElements()));

        const int expected_params_size = num_layers_ * num_qubits_ * kRotationAnglesPerQubit;
        OP_REQUIRES(ctx, params_tensor.NumElements() >= expected_params_size,
                    errors::InvalidArgument(
                        "params must have at least ", expected_params_size,
                        " elements, got ", params_tensor.NumElements()));

        OP_REQUIRES(ctx, inputs_tensor.dims() == 2,
                    errors::InvalidArgument(
                        "inputs must be rank-2 [batch, features], got rank ",
                        inputs_tensor.dims()));

        const int batch_size = inputs_tensor.dim_size(0);
        const int input_features = inputs_tensor.dim_size(1);

        // Allocate output
        Tensor* output_tensor;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, num_qubits_}), &output_tensor));

        // Get data pointers
        const float* cores = cores_tensor.flat<float>().data();
        const float* params = params_tensor.flat<float>().data();
        const float* inputs = inputs_tensor.flat<float>().data();
        float* output = output_tensor->flat<float>().data();

        // Call SIMD-optimized contraction
        TensorRingContract(cores, params, inputs, output,
                          batch_size, num_qubits_, num_layers_,
                          bond_dim_, input_features);
    }

private:
    int num_qubits_;
    int num_layers_;
    int bond_dim_;
};

REGISTER_KERNEL_BUILDER(Name("TensorRingContract").Device(DEVICE_CPU),
                        TensorRingContractOp);

// ============================================================================
// CPU Kernel: NeuralBPMitigationForwardOp
// ============================================================================

class NeuralBPMitigationForwardOp : public OpKernel {
public:
    explicit NeuralBPMitigationForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hidden_dim", &hidden_dim_));
        LOG(INFO) << "[NeuralBPMitigationForward] Initialized: hidden_dim=" << hidden_dim_;
    }

    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& inputs_tensor = ctx->input(0);
        const Tensor& weights_1 = ctx->input(1);
        const Tensor& bias_1 = ctx->input(2);
        const Tensor& weights_2 = ctx->input(3);
        const Tensor& bias_2 = ctx->input(4);
        const Tensor& weights_out = ctx->input(5);
        const Tensor& bias_out = ctx->input(6);

        // Validate shapes
        OP_REQUIRES(ctx, inputs_tensor.dims() == 2,
                    errors::InvalidArgument(
                        "inputs must be rank-2 [batch, input_dim], got rank ",
                        inputs_tensor.dims()));

        const int batch_size = inputs_tensor.dim_size(0);
        const int input_dim = inputs_tensor.dim_size(1);
        const int output_dim = bias_out.dim_size(0);

        OP_REQUIRES(ctx, bias_1.dim_size(0) == hidden_dim_,
                    errors::InvalidArgument(
                        "bias_1 size must match hidden_dim (", hidden_dim_,
                        "), got ", bias_1.dim_size(0)));

        // Allocate output
        Tensor* output_tensor;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, output_dim}), &output_tensor));

        // Get data pointers
        const float* inputs = inputs_tensor.flat<float>().data();
        const float* w1 = weights_1.flat<float>().data();
        const float* b1 = bias_1.flat<float>().data();
        const float* w2 = weights_2.flat<float>().data();
        const float* b2 = bias_2.flat<float>().data();
        const float* wo = weights_out.flat<float>().data();
        const float* bo = bias_out.flat<float>().data();
        float* output = output_tensor->flat<float>().data();

        // Call SIMD-optimized forward pass
        NeuralBPMitigationForward(inputs, w1, b1, w2, b2, wo, bo, output,
                                  batch_size, input_dim, hidden_dim_, output_dim);
    }

private:
    int hidden_dim_;
};

REGISTER_KERNEL_BUILDER(Name("NeuralBPMitigationForward").Device(DEVICE_CPU),
                        NeuralBPMitigationForwardOp);

}  // namespace tensorflow
