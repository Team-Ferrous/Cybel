// saguaro.native/ops/quantum_norm_op.cc
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
 * @file quantum_norm_op.cc
 * @brief TensorFlow custom ops for unitary-preserving normalization.
 *
 * Phase 30 of Unified Quantum Architecture Enhancement.
 *
 * Registers the following ops:
 *   - UnitaryNormForward: x_norm = (x / ||x||₂) · scale + bias
 *   - UnitaryNormBackward: Gradient with orthogonal projection
 *   - RMSNormForward: RMS normalization variant
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "quantum_norm_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("UnitaryNormForward")
    .Input("input: float")
    .Input("scale: float")
    .Input("bias: float")
    .Output("output: float")
    .Output("norms: float")
    .Attr("eps: float = 1e-6")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));  // output same as input
        
        // norms shape: [batch * seq_len] = product of all dims except last
        shape_inference::ShapeHandle input_shape = c->input(0);
        int rank = c->Rank(input_shape);
        if (rank >= 2) {
            std::vector<shape_inference::DimensionHandle> norm_dims;
            for (int i = 0; i < rank - 1; ++i) {
                norm_dims.push_back(c->Dim(input_shape, i));
            }
            c->set_output(1, c->MakeShape(norm_dims));
        } else {
            c->set_output(1, c->MakeShape({1}));
        }
        return Status();
    })
    .Doc(R"doc(
Unitary-preserving normalization via projection to unit hypersphere.
Computes output = (x / ||x||) * scale + bias with Stiefel manifold geometry.

input: Input tensor with shape [batch, seq_len, dim].
scale: Learnable scale vector with shape [dim].
bias: Learnable bias vector with shape [dim].
eps: Epsilon for numerical stability.
output: Normalized output with shape [batch, seq_len, dim].
norms: Cached L2 norms for backward with shape [batch, seq_len].
)doc");

REGISTER_OP("UnitaryNormBackward")
    .Input("grad_output: float")
    .Input("input: float")
    .Input("scale: float")
    .Input("norms: float")
    .Output("grad_input: float")
    .Output("grad_scale: float")
    .Output("grad_bias: float")
    .Attr("eps: float = 1e-6")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_input same as input
        c->set_output(1, c->input(2));  // grad_scale same as scale
        c->set_output(2, c->input(2));  // grad_bias same as scale (both [dim])
        return Status();
    })
    .Doc(R"doc(
Backward pass for unitary normalization with orthogonal projection.
Gradients use orthogonal complement projection for Stiefel manifold optimization.

grad_output: Gradient w.r.t. output.
input: Original input tensor.
scale: Scale parameter.
norms: Cached L2 norms from forward.
grad_input: Gradient w.r.t. input.
grad_scale: Gradient w.r.t. scale.
grad_bias: Gradient w.r.t. bias.
)doc");

REGISTER_OP("RMSNormForward")
    .Input("input: float")
    .Input("scale: float")
    .Output("output: float")
    .Attr("eps: float = 1e-6")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
RMS normalization: output = x / RMS(x) · scale

More efficient variant without mean centering.

input: Input tensor [batch, seq_len, dim]
scale: Learnable scale [dim]
output: Normalized output [batch, seq_len, dim]
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class UnitaryNormForwardOp : public OpKernel {
public:
    explicit UnitaryNormForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("eps", &eps_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const Tensor& scale = context->input(1);
        const Tensor& bias = context->input(2);
        
        const TensorShape& input_shape = input.shape();
        OP_REQUIRES(context, input_shape.dims() >= 2,
            errors::InvalidArgument("Input must be at least 2D"));
        
        const int dim = input_shape.dim_size(input_shape.dims() - 1);
        OP_REQUIRES(context, scale.dim_size(0) == dim,
            errors::InvalidArgument("Scale must have size dim"));
        OP_REQUIRES(context, bias.dim_size(0) == dim,
            errors::InvalidArgument("Bias must have size dim"));
        
        // Compute number of vectors
        int64_t num_vectors = 1;
        for (int i = 0; i < input_shape.dims() - 1; ++i) {
            num_vectors *= input_shape.dim_size(i);
        }
        
        // Allocate outputs
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        
        // Norms shape: all dims except last
        TensorShape norms_shape;
        for (int i = 0; i < input_shape.dims() - 1; ++i) {
            norms_shape.AddDim(input_shape.dim_size(i));
        }
        Tensor* norms = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, norms_shape, &norms));
        
        const float* input_data = input.flat<float>().data();
        const float* scale_data = scale.flat<float>().data();
        const float* bias_data = bias.flat<float>().data();
        float* output_data = output->flat<float>().data();
        float* norms_data = norms->flat<float>().data();
        
        // For 2D input [seq_len, dim]: batch=1, seq_len=seq_len
        // For 3D input [batch, seq_len, dim]: as expected
        int batch_size = 1;
        int seq_len = num_vectors;
        if (input_shape.dims() == 3) {
            batch_size = input_shape.dim_size(0);
            seq_len = input_shape.dim_size(1);
        }
        
        saguaro::ops::quantum_norm::UnitaryNormForward(
            input_data, scale_data, bias_data, output_data, norms_data,
            batch_size, seq_len, dim, eps_);
    }

private:
    float eps_;
};

REGISTER_KERNEL_BUILDER(
    Name("UnitaryNormForward").Device(DEVICE_CPU),
    UnitaryNormForwardOp);

class UnitaryNormBackwardOp : public OpKernel {
public:
    explicit UnitaryNormBackwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("eps", &eps_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output = context->input(0);
        const Tensor& input = context->input(1);
        const Tensor& scale = context->input(2);
        const Tensor& norms = context->input(3);
        
        const TensorShape& input_shape = input.shape();
        const int dim = input_shape.dim_size(input_shape.dims() - 1);
        
        // Compute number of vectors
        int64_t num_vectors = 1;
        for (int i = 0; i < input_shape.dims() - 1; ++i) {
            num_vectors *= input_shape.dim_size(i);
        }
        
        // Allocate outputs
        Tensor* grad_input = nullptr;
        Tensor* grad_scale = nullptr;
        Tensor* grad_bias = nullptr;
        
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
        OP_REQUIRES_OK(context, context->allocate_output(1, scale.shape(), &grad_scale));
        OP_REQUIRES_OK(context, context->allocate_output(2, scale.shape(), &grad_bias));
        
        const float* grad_output_data = grad_output.flat<float>().data();
        const float* input_data = input.flat<float>().data();
        const float* scale_data = scale.flat<float>().data();
        const float* norms_data = norms.flat<float>().data();
        float* grad_input_data = grad_input->flat<float>().data();
        float* grad_scale_data = grad_scale->flat<float>().data();
        float* grad_bias_data = grad_bias->flat<float>().data();
        
        int batch_size = 1;
        int seq_len = num_vectors;
        if (input_shape.dims() == 3) {
            batch_size = input_shape.dim_size(0);
            seq_len = input_shape.dim_size(1);
        }
        
        saguaro::ops::quantum_norm::UnitaryNormBackward(
            grad_output_data, input_data, scale_data, norms_data,
            grad_input_data, grad_scale_data, grad_bias_data,
            batch_size, seq_len, dim, eps_);
    }

private:
    float eps_;
};

REGISTER_KERNEL_BUILDER(
    Name("UnitaryNormBackward").Device(DEVICE_CPU),
    UnitaryNormBackwardOp);

class RMSNormForwardOp : public OpKernel {
public:
    explicit RMSNormForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("eps", &eps_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const Tensor& scale = context->input(1);
        
        const TensorShape& input_shape = input.shape();
        const int dim = input_shape.dim_size(input_shape.dims() - 1);
        
        int64_t num_vectors = 1;
        for (int i = 0; i < input_shape.dims() - 1; ++i) {
            num_vectors *= input_shape.dim_size(i);
        }
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        
        const float* input_data = input.flat<float>().data();
        const float* scale_data = scale.flat<float>().data();
        float* output_data = output->flat<float>().data();
        
        int batch_size = 1;
        int seq_len = num_vectors;
        if (input_shape.dims() == 3) {
            batch_size = input_shape.dim_size(0);
            seq_len = input_shape.dim_size(1);
        }
        
        saguaro::ops::quantum_norm::RMSNormForward(
            input_data, scale_data, output_data,
            batch_size, seq_len, dim, eps_);
    }

private:
    float eps_;
};

REGISTER_KERNEL_BUILDER(
    Name("RMSNormForward").Device(DEVICE_CPU),
    RMSNormForwardOp);
