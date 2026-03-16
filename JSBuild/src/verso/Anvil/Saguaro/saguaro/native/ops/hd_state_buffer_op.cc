// saguaro.native/ops/hd_state_buffer_op.cc
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
 * @file hd_state_buffer_op.cc
 * @brief TensorFlow ops for HD State Buffer (optimizer state compression).
 *
 * Registers the following ops:
 * - HDStateEncode: Compress optimizer state to HD representation
 * - HDStateDecode: Decompress HD representation back to full state
 * - HDStateEncodeGrad: Gradient op for encoding
 * - HDStateDecodeGrad: Gradient op for decoding
 * - HDStateGenerateProjection: Generate random projection matrix
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "hd_state_buffer_op.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("HDStateEncode")
    .Input("state: float32")
    .Input("projection: float32")
    .Output("compressed: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle state_shape, proj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &state_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &proj_shape));
        
        DimensionHandle compressed_size = c->Dim(proj_shape, 1);
        c->set_output(0, c->Vector(compressed_size));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Encode optimizer state to HD compressed representation.

state: Input state tensor [param_size].
projection: Projection matrix [param_size, compressed_size].
compressed: Compressed state [compressed_size].
)doc");

REGISTER_OP("HDStateDecode")
    .Input("compressed: float32")
    .Input("projection: float32")
    .Output("state: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle proj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &proj_shape));
        
        DimensionHandle param_size = c->Dim(proj_shape, 0);
        c->set_output(0, c->Vector(param_size));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Decode HD compressed representation back to full state.

compressed: Compressed state [compressed_size].
projection: Projection matrix [param_size, compressed_size].
state: Reconstructed state [param_size].
)doc");

REGISTER_OP("HDStateBatchEncode")
    .Input("states: float32")
    .Input("projection: float32")
    .Output("compressed: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle states_shape, proj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &states_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &proj_shape));
        
        DimensionHandle num_states = c->Dim(states_shape, 0);
        DimensionHandle compressed_size = c->Dim(proj_shape, 1);
        c->set_output(0, c->Matrix(num_states, compressed_size));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Batch encode multiple optimizer states to HD compressed representation.

states: Input states [num_states, param_size].
projection: Projection matrix [param_size, compressed_size].
compressed: Compressed states [num_states, compressed_size].
)doc");

REGISTER_OP("HDStateBatchDecode")
    .Input("compressed: float32")
    .Input("projection: float32")
    .Output("states: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle comp_shape, proj_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &comp_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &proj_shape));
        
        DimensionHandle num_states = c->Dim(comp_shape, 0);
        DimensionHandle param_size = c->Dim(proj_shape, 0);
        c->set_output(0, c->Matrix(num_states, param_size));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Batch decode HD compressed representation back to full states.

compressed: Compressed states [num_states, compressed_size].
projection: Projection matrix [param_size, compressed_size].
states: Reconstructed states [num_states, param_size].
)doc");

REGISTER_OP("HDStateGenerateProjection")
    .Attr("param_size: int")
    .Attr("compressed_size: int")
    .Attr("sparse: bool = true")
    .Attr("sparse_density: int = 3")
    .Attr("seed: int = 42")
    .Output("projection: float32")
    .SetShapeFn([](InferenceContext* c) {
        int param_size, compressed_size;
        TF_RETURN_IF_ERROR(c->GetAttr("param_size", &param_size));
        TF_RETURN_IF_ERROR(c->GetAttr("compressed_size", &compressed_size));
        c->set_output(0, c->Matrix(param_size, compressed_size));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Generate random projection matrix for HD state compression.

param_size: Original parameter size.
compressed_size: Target compressed size.
sparse: Use sparse random projection (faster, default true).
sparse_density: Non-zeros per row for sparse projection.
seed: Random seed for reproducibility.
projection: Generated projection matrix [param_size, compressed_size].
)doc");

// =============================================================================
// Kernel Implementations
// =============================================================================

class HDStateEncodeOp : public OpKernel {
public:
    explicit HDStateEncodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& state_tensor = ctx->input(0);
        const Tensor& proj_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, state_tensor.dims() == 1,
            errors::InvalidArgument("State must be 1D"));
        OP_REQUIRES(ctx, proj_tensor.dims() == 2,
            errors::InvalidArgument("Projection must be 2D"));
        
        const int param_size = state_tensor.dim_size(0);
        const int compressed_size = proj_tensor.dim_size(1);
        
        OP_REQUIRES(ctx, proj_tensor.dim_size(0) == param_size,
            errors::InvalidArgument(
                "Projection dim 0 must match state size: ",
                proj_tensor.dim_size(0), " vs ", param_size));
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({compressed_size}), &output_tensor));
        
        const float* state = state_tensor.flat<float>().data();
        const float* projection = proj_tensor.flat<float>().data();
        float* compressed = output_tensor->flat<float>().data();
        
        saguaro::hd_state::HDStateEncode(state, projection, compressed, param_size, compressed_size);
    }
};

class HDStateDecodeOp : public OpKernel {
public:
    explicit HDStateDecodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& comp_tensor = ctx->input(0);
        const Tensor& proj_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, comp_tensor.dims() == 1,
            errors::InvalidArgument("Compressed must be 1D"));
        OP_REQUIRES(ctx, proj_tensor.dims() == 2,
            errors::InvalidArgument("Projection must be 2D"));
        
        const int param_size = proj_tensor.dim_size(0);
        const int compressed_size = proj_tensor.dim_size(1);
        
        OP_REQUIRES(ctx, comp_tensor.dim_size(0) == compressed_size,
            errors::InvalidArgument(
                "Compressed size must match projection dim 1: ",
                comp_tensor.dim_size(0), " vs ", compressed_size));
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({param_size}), &output_tensor));
        
        const float* compressed = comp_tensor.flat<float>().data();
        const float* projection = proj_tensor.flat<float>().data();
        float* state = output_tensor->flat<float>().data();
        
        saguaro::hd_state::HDStateDecode(compressed, projection, state, param_size, compressed_size);
    }
};

class HDStateBatchEncodeOp : public OpKernel {
public:
    explicit HDStateBatchEncodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& states_tensor = ctx->input(0);
        const Tensor& proj_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, states_tensor.dims() == 2,
            errors::InvalidArgument("States must be 2D"));
        OP_REQUIRES(ctx, proj_tensor.dims() == 2,
            errors::InvalidArgument("Projection must be 2D"));
        
        const int num_states = states_tensor.dim_size(0);
        const int param_size = states_tensor.dim_size(1);
        const int compressed_size = proj_tensor.dim_size(1);
        
        OP_REQUIRES(ctx, proj_tensor.dim_size(0) == param_size,
            errors::InvalidArgument(
                "Projection dim 0 must match state size"));
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            TensorShape({num_states, compressed_size}), &output_tensor));
        
        const float* states = states_tensor.flat<float>().data();
        const float* projection = proj_tensor.flat<float>().data();
        float* compressed = output_tensor->flat<float>().data();
        
        saguaro::hd_state::HDStateBatchEncode(
            states, projection, compressed,
            num_states, param_size, compressed_size
        );
    }
};

class HDStateBatchDecodeOp : public OpKernel {
public:
    explicit HDStateBatchDecodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& comp_tensor = ctx->input(0);
        const Tensor& proj_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, comp_tensor.dims() == 2,
            errors::InvalidArgument("Compressed must be 2D"));
        OP_REQUIRES(ctx, proj_tensor.dims() == 2,
            errors::InvalidArgument("Projection must be 2D"));
        
        const int num_states = comp_tensor.dim_size(0);
        const int param_size = proj_tensor.dim_size(0);
        const int compressed_size = proj_tensor.dim_size(1);
        
        OP_REQUIRES(ctx, comp_tensor.dim_size(1) == compressed_size,
            errors::InvalidArgument(
                "Compressed dim 1 must match projection dim 1"));
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({num_states, param_size}), &output_tensor));
        
        const float* compressed = comp_tensor.flat<float>().data();
        const float* projection = proj_tensor.flat<float>().data();
        float* states = output_tensor->flat<float>().data();
        
        saguaro::hd_state::HDStateBatchDecode(
            compressed, projection, states,
            num_states, param_size, compressed_size
        );
    }
};

class HDStateGenerateProjectionOp : public OpKernel {
public:
    explicit HDStateGenerateProjectionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("param_size", &param_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("compressed_size", &compressed_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse", &sparse_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_density", &sparse_density_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    }

    void Compute(OpKernelContext* ctx) override {
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({param_size_, compressed_size_}), &output_tensor));
        
        float* projection = output_tensor->flat<float>().data();
        
        saguaro::hd_state::HDStateConfig config;
        config.sparse_density = sparse_density_;
        config.seed = static_cast<uint64_t>(seed_);
        config.use_sparse_projection = sparse_;
        
        if (sparse_) {
            saguaro::hd_state::generate_sparse_projection(
                projection, param_size_, compressed_size_, config);
        } else {
            saguaro::hd_state::generate_dense_projection(
                projection, param_size_, compressed_size_, config);
        }
    }

private:
    int param_size_;
    int compressed_size_;
    bool sparse_;
    int sparse_density_;
    int seed_;
};

// =============================================================================
// Kernel Registrations
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HDStateEncode").Device(DEVICE_CPU),
    HDStateEncodeOp);

REGISTER_KERNEL_BUILDER(
    Name("HDStateDecode").Device(DEVICE_CPU),
    HDStateDecodeOp);

REGISTER_KERNEL_BUILDER(
    Name("HDStateBatchEncode").Device(DEVICE_CPU),
    HDStateBatchEncodeOp);

REGISTER_KERNEL_BUILDER(
    Name("HDStateBatchDecode").Device(DEVICE_CPU),
    HDStateBatchDecodeOp);

REGISTER_KERNEL_BUILDER(
    Name("HDStateGenerateProjection").Device(DEVICE_CPU),
    HDStateGenerateProjectionOp);

}  // namespace tensorflow
