// saguaro.native/ops/quantum_position_encoding_op.cc
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
 * @file quantum_position_encoding_op.cc
 * @brief TensorFlow custom ops for Floquet position encoding.
 *
 * Phase 27 of Unified Quantum Architecture Enhancement.
 *
 * Registers the following ops:
 *   - FloquetPositionEncodingForward: Apply time-crystal dynamics
 *   - FloquetPositionEncodingBackward: Gradient computation
 *   - InitFloquetAngles: Initialize with frequency-scaled defaults
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "quantum_position_encoding_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("FloquetPositionEncodingForward")
    .Input("base_embedding: float")
    .Input("floquet_angles: float")
    .Output("output: float")
    .Attr("max_position: int = 100000")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Floquet time-crystal position encoding.

Applies learnable SU(2) rotations to pairs of dimensions, simulating
discrete time-crystal dynamics for position-dependent transformations.

base_embedding: Input embedding [batch, seq_len, dim]
floquet_angles: Rotation parameters [dim/2, 3] (θ, φ, ω per qubit)
max_position: Maximum expected position (for scaling)
output: Position-encoded embedding [batch, seq_len, dim]
)doc");

REGISTER_OP("FloquetPositionEncodingBackward")
    .Input("grad_output: float")
    .Input("base_embedding: float")
    .Input("floquet_angles: float")
    .Output("grad_angles: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(2));  // Same shape as floquet_angles
        return Status();
    })
    .Doc(R"doc(
Backward pass for Floquet position encoding.

Computes gradients w.r.t. floquet_angles.

grad_output: Gradient w.r.t. output
base_embedding: Original input
floquet_angles: Current angles
grad_angles: Gradient w.r.t. angles
)doc");

REGISTER_OP("InitFloquetAngles")
    .Input("num_qubits: int32")
    .Output("angles: float")
    .Attr("base_frequency: float = 10000.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output shape: [num_qubits, 3]
        c->set_output(0, c->UnknownShape());
        return Status();
    })
    .Doc(R"doc(
Initialize Floquet angles with frequency-scaled defaults.

Similar to sinusoidal position encoding, low-indexed qubits get
lower frequencies (slower variation) and high-indexed qubits get
higher frequencies (faster variation).

num_qubits: Number of qubit pairs (dim/2)
base_frequency: Base frequency for scaling
angles: Initialized angles [num_qubits, 3]
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class FloquetPositionEncodingForwardOp : public OpKernel {
public:
    explicit FloquetPositionEncodingForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("max_position", &max_position_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& base_embedding = context->input(0);
        const Tensor& floquet_angles = context->input(1);
        
        const TensorShape& emb_shape = base_embedding.shape();
        OP_REQUIRES(context, emb_shape.dims() >= 2,
            errors::InvalidArgument("base_embedding must be at least 2D"));
        
        int batch_size = 1;
        int seq_len;
        int dim;
        
        if (emb_shape.dims() == 2) {
            seq_len = emb_shape.dim_size(0);
            dim = emb_shape.dim_size(1);
        } else {
            batch_size = emb_shape.dim_size(0);
            seq_len = emb_shape.dim_size(1);
            dim = emb_shape.dim_size(2);
        }
        
        OP_REQUIRES(context, dim % 2 == 0,
            errors::InvalidArgument("Dimension must be even for qubit pairs"));
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, emb_shape, &output));
        
        const float* emb_data = base_embedding.flat<float>().data();
        const float* angles_data = floquet_angles.flat<float>().data();
        float* output_data = output->flat<float>().data();
        
        saguaro::ops::quantum_position::FloquetPositionEncodingForwardF32(
            emb_data, angles_data, output_data,
            batch_size, seq_len, dim);
    }

private:
    int max_position_;
};

REGISTER_KERNEL_BUILDER(
    Name("FloquetPositionEncodingForward").Device(DEVICE_CPU),
    FloquetPositionEncodingForwardOp);

class FloquetPositionEncodingBackwardOp : public OpKernel {
public:
    explicit FloquetPositionEncodingBackwardOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output = context->input(0);
        const Tensor& base_embedding = context->input(1);
        const Tensor& floquet_angles = context->input(2);
        
        const TensorShape& emb_shape = base_embedding.shape();
        
        int batch_size = 1;
        int seq_len;
        int dim;
        
        if (emb_shape.dims() == 2) {
            seq_len = emb_shape.dim_size(0);
            dim = emb_shape.dim_size(1);
        } else {
            batch_size = emb_shape.dim_size(0);
            seq_len = emb_shape.dim_size(1);
            dim = emb_shape.dim_size(2);
        }
        
        Tensor* grad_angles = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, floquet_angles.shape(), &grad_angles));
        
        const float* grad_out_data = grad_output.flat<float>().data();
        const float* emb_data = base_embedding.flat<float>().data();
        const float* angles_data = floquet_angles.flat<float>().data();
        float* grad_angles_data = grad_angles->flat<float>().data();
        
        saguaro::ops::quantum_position::FloquetPositionEncodingBackward(
            grad_out_data, emb_data, angles_data, grad_angles_data,
            batch_size, seq_len, dim);
    }
};

REGISTER_KERNEL_BUILDER(
    Name("FloquetPositionEncodingBackward").Device(DEVICE_CPU),
    FloquetPositionEncodingBackwardOp);

class InitFloquetAnglesOp : public OpKernel {
public:
    explicit InitFloquetAnglesOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("base_frequency", &base_frequency_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& num_qubits_tensor = context->input(0);
        
        int num_qubits = num_qubits_tensor.scalar<int32_t>()();
        
        Tensor* angles = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
            TensorShape({num_qubits, 3}), &angles));
        
        float* angles_data = angles->flat<float>().data();
        
        saguaro::ops::quantum_position::InitFloquetAngles(
            angles_data, num_qubits, base_frequency_);
    }

private:
    float base_frequency_;
};

REGISTER_KERNEL_BUILDER(
    Name("InitFloquetAngles").Device(DEVICE_CPU),
    InitFloquetAnglesOp);
