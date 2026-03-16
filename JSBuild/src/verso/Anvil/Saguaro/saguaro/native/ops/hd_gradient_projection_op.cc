// saguaro.native/ops/hd_gradient_projection_op.cc
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
 * @file hd_gradient_projection_op.cc
 * @brief TensorFlow ops for HD gradient projection (GaLore replacement).
 *
 * Registers:
 * - HDGradientProject: Compress gradient via SRHT
 * - HDGradientReconstruct: Decompress gradient via SRHT^T
 * - HDGradientGenerateProjection: Generate projection parameters
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "hd_gradient_projection_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("HDGradientProject")
    .Input("gradient: float32")
    .Input("signs: float32")
    .Input("indices: int32")
    .Output("compressed: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle indices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &indices_shape));
        
        DimensionHandle rank = c->Dim(indices_shape, 0);
        
        ShapeHandle grad_shape = c->input(0);
        if (c->Rank(grad_shape) == 1) {
            c->set_output(0, c->Vector(rank));
        } else if (c->Rank(grad_shape) == 2) {
            DimensionHandle batch = c->Dim(grad_shape, 0);
            c->set_output(0, c->Matrix(batch, rank));
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Project gradient to low-rank space via SRHT.

gradient: Input gradient [param_size] or [batch, param_size].
signs: Random sign flips [padded_dim].
indices: Subsampling indices [rank].
compressed: Compressed gradient [rank] or [batch, rank].
)doc");

REGISTER_OP("HDGradientReconstruct")
    .Input("compressed: float32")
    .Input("signs: float32")
    .Input("indices: int32")
    .Attr("param_size: int")
    .Output("gradient: float32")
    .SetShapeFn([](InferenceContext* c) {
        int param_size;
        TF_RETURN_IF_ERROR(c->GetAttr("param_size", &param_size));
        
        ShapeHandle comp_shape = c->input(0);
        if (c->Rank(comp_shape) == 1) {
            c->set_output(0, c->Vector(param_size));
        } else if (c->Rank(comp_shape) == 2) {
            DimensionHandle batch = c->Dim(comp_shape, 0);
            c->set_output(0, c->Matrix(batch, param_size));
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Reconstruct gradient from low-rank space via SRHT^T.

compressed: Compressed gradient [rank] or [batch, rank].
signs: Random sign flips [padded_dim].
indices: Subsampling indices [rank].
param_size: Original parameter size.
gradient: Reconstructed gradient [param_size] or [batch, param_size].
)doc");

REGISTER_OP("HDGradientGenerateProjection")
    .Attr("param_size: int")
    .Attr("rank: int")
    .Attr("seed: int = 314159")
    .Output("signs: float32")
    .Output("indices: int32")
    .SetShapeFn([](InferenceContext* c) {
        int param_size, rank;
        TF_RETURN_IF_ERROR(c->GetAttr("param_size", &param_size));
        TF_RETURN_IF_ERROR(c->GetAttr("rank", &rank));
        
        // Pad to power of 2
        int padded = 1;
        while (padded < param_size) padded <<= 1;
        
        c->set_output(0, c->Vector(padded));
        c->set_output(1, c->Vector(rank));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Generate projection parameters for HD gradient compression.

param_size: Original parameter size.
rank: Target compressed rank.
seed: Random seed for reproducibility.
signs: Random sign flips [padded_dim].
indices: Subsampling indices [rank].
)doc");

// =============================================================================
// Kernel Implementations
// =============================================================================

class HDGradientProjectOp : public OpKernel {
public:
    explicit HDGradientProjectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_tensor = ctx->input(0);
        const Tensor& signs_tensor = ctx->input(1);
        const Tensor& indices_tensor = ctx->input(2);
        
        const int rank = indices_tensor.dim_size(0);
        const int is_batched = grad_tensor.dims() == 2;
        
        const float* signs = signs_tensor.flat<float>().data();
        const int* indices = indices_tensor.flat<int>().data();
        
        if (is_batched) {
            const int batch_size = grad_tensor.dim_size(0);
            const int param_size = grad_tensor.dim_size(1);
            
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({batch_size, rank}), &output_tensor));
            
            const float* grad = grad_tensor.flat<float>().data();
            float* compressed = output_tensor->flat<float>().data();
            
            saguaro::hd_gradient::HDGradientBatchProject(
                grad, compressed, signs, indices,
                batch_size, param_size, rank
            );
        } else {
            const int param_size = grad_tensor.dim_size(0);
            
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({rank}), &output_tensor));
            
            const float* grad = grad_tensor.flat<float>().data();
            float* compressed = output_tensor->flat<float>().data();
            
            saguaro::hd_gradient::HDGradientProject(
                grad, compressed, signs, indices, param_size, rank
            );
        }
    }
};

class HDGradientReconstructOp : public OpKernel {
public:
    explicit HDGradientReconstructOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("param_size", &param_size_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& comp_tensor = ctx->input(0);
        const Tensor& signs_tensor = ctx->input(1);
        const Tensor& indices_tensor = ctx->input(2);
        
        const int rank = indices_tensor.dim_size(0);
        const int is_batched = comp_tensor.dims() == 2;
        
        const float* signs = signs_tensor.flat<float>().data();
        const int* indices = indices_tensor.flat<int>().data();
        
        if (is_batched) {
            const int batch_size = comp_tensor.dim_size(0);
            
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({batch_size, param_size_}), &output_tensor));
            
            const float* compressed = comp_tensor.flat<float>().data();
            float* grad = output_tensor->flat<float>().data();
            
            saguaro::hd_gradient::HDGradientBatchReconstruct(
                compressed, grad, signs, indices,
                batch_size, param_size_, rank
            );
        } else {
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({param_size_}), &output_tensor));
            
            const float* compressed = comp_tensor.flat<float>().data();
            float* grad = output_tensor->flat<float>().data();
            
            saguaro::hd_gradient::HDGradientReconstruct(
                compressed, grad, signs, indices, param_size_, rank
            );
        }
    }

private:
    int param_size_;
};

class HDGradientGenerateProjectionOp : public OpKernel {
public:
    explicit HDGradientGenerateProjectionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("param_size", &param_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Compute padded dimension
        int padded = 1;
        while (padded < param_size_) padded <<= 1;
        
        Tensor* signs_tensor = nullptr;
        Tensor* indices_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({padded}), &signs_tensor));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1,
            TensorShape({rank_}), &indices_tensor));
        
        float* signs = signs_tensor->flat<float>().data();
        int* indices = indices_tensor->flat<int>().data();
        
        saguaro::hd_gradient::generate_random_signs(signs, padded, seed_);
        saguaro::hd_gradient::generate_subsampling_indices(indices, rank_, padded, seed_ + 1);
    }

private:
    int param_size_;
    int rank_;
    int seed_;
};

// =============================================================================
// Kernel Registrations
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HDGradientProject").Device(DEVICE_CPU),
    HDGradientProjectOp);

REGISTER_KERNEL_BUILDER(
    Name("HDGradientReconstruct").Device(DEVICE_CPU),
    HDGradientReconstructOp);

REGISTER_KERNEL_BUILDER(
    Name("HDGradientGenerateProjection").Device(DEVICE_CPU),
    HDGradientGenerateProjectionOp);

}  // namespace tensorflow
