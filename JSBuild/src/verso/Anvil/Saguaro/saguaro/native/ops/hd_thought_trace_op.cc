// saguaro.native/ops/hd_thought_trace_op.cc
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
 * @file hd_thought_trace_op.cc
 * @brief TensorFlow ops for HD thought trace (COCONUT integration).
 *
 * Registers:
 * - HDThoughtTraceUpdate: Add thought to trace via holographic binding
 * - HDThoughtTraceRetrieve: Retrieve specific step via unbinding
 * - HDThoughtPathScores: Score candidate paths for BFS selection
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "hd_thought_trace_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("HDThoughtTraceUpdate")
    .Input("trace: float32")
    .Input("thought: float32")
    .Attr("step: int")
    .Output("updated_trace: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Update thought trace with new thought via holographic binding.

trace: Current thought trace [batch, hd_dim].
thought: New thought to add [batch, hd_dim].
step: Current step index.
updated_trace: Updated trace [batch, hd_dim].
)doc");

REGISTER_OP("HDThoughtTraceRetrieve")
    .Input("trace: float32")
    .Attr("step: int")
    .Attr("epsilon: float = 1e-8")
    .Output("retrieved: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Retrieve specific step from thought trace via holographic unbinding.

trace: Thought trace [batch, hd_dim].
step: Step index to retrieve.
epsilon: Numerical stability constant.
retrieved: Retrieved thought [batch, hd_dim].
)doc");

REGISTER_OP("HDThoughtPathScores")
    .Input("paths: float32")
    .Input("target: float32")
    .Output("scores: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle paths_shape = c->input(0);
        DimensionHandle batch = c->Dim(paths_shape, 0);
        DimensionHandle num_paths = c->Dim(paths_shape, 1);
        c->set_output(0, c->Matrix(batch, num_paths));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compute similarity scores between path candidates and target.

paths: Candidate paths [batch, num_paths, hd_dim].
target: Target thought [batch, hd_dim].
scores: Similarity scores [batch, num_paths].
)doc");

REGISTER_OP("HDThoughtTraceUpdateGrad")
    .Input("grad_trace: float32")
    .Input("thought: float32")
    .Attr("step: int")
    .Output("grad_thought: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Gradient of HDThoughtTraceUpdate w.r.t. thought.

grad_trace: Gradient w.r.t. updated trace [batch, hd_dim].
thought: Forward pass thought [batch, hd_dim].
step: Step index.
grad_thought: Gradient w.r.t. thought [batch, hd_dim].
)doc");

// =============================================================================
// Kernel Implementations
// =============================================================================

class HDThoughtTraceUpdateOp : public OpKernel {
public:
    explicit HDThoughtTraceUpdateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("step", &step_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& trace_tensor = ctx->input(0);
        const Tensor& thought_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, trace_tensor.dims() == 2,
            errors::InvalidArgument("Trace must be 2D"));
        OP_REQUIRES(ctx, thought_tensor.dims() == 2,
            errors::InvalidArgument("Thought must be 2D"));
        
        const int batch_size = trace_tensor.dim_size(0);
        const int hd_dim = trace_tensor.dim_size(1);
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, trace_tensor.shape(), &output_tensor));
        
        // Copy trace to output first
        std::memcpy(
            output_tensor->flat<float>().data(),
            trace_tensor.flat<float>().data(),
            trace_tensor.NumElements() * sizeof(float)
        );
        
        // Update in place
        saguaro::hd_thought::HDThoughtTraceUpdate(
            output_tensor->flat<float>().data(),
            thought_tensor.flat<float>().data(),
            step_, batch_size, hd_dim
        );
    }

private:
    int step_;
};

class HDThoughtTraceRetrieveOp : public OpKernel {
public:
    explicit HDThoughtTraceRetrieveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("step", &step_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& trace_tensor = ctx->input(0);
        
        const int batch_size = trace_tensor.dim_size(0);
        const int hd_dim = trace_tensor.dim_size(1);
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, trace_tensor.shape(), &output_tensor));
        
        saguaro::hd_thought::HDThoughtTraceRetrieve(
            trace_tensor.flat<float>().data(),
            output_tensor->flat<float>().data(),
            step_, batch_size, hd_dim, epsilon_
        );
    }

private:
    int step_;
    float epsilon_;
};

class HDThoughtPathScoresOp : public OpKernel {
public:
    explicit HDThoughtPathScoresOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& paths_tensor = ctx->input(0);
        const Tensor& target_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, paths_tensor.dims() == 3,
            errors::InvalidArgument("Paths must be 3D"));
        OP_REQUIRES(ctx, target_tensor.dims() == 2,
            errors::InvalidArgument("Target must be 2D"));
        
        const int batch_size = paths_tensor.dim_size(0);
        const int num_paths = paths_tensor.dim_size(1);
        const int hd_dim = paths_tensor.dim_size(2);
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({batch_size, num_paths}), &output_tensor));
        
        saguaro::hd_thought::HDThoughtPathScores(
            paths_tensor.flat<float>().data(),
            target_tensor.flat<float>().data(),
            output_tensor->flat<float>().data(),
            batch_size, num_paths, hd_dim
        );
    }
};

class HDThoughtTraceUpdateGradOp : public OpKernel {
public:
    explicit HDThoughtTraceUpdateGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("step", &step_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_tensor = ctx->input(0);
        const Tensor& thought_tensor = ctx->input(1);
        
        const int batch_size = thought_tensor.dim_size(0);
        const int hd_dim = thought_tensor.dim_size(1);
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, thought_tensor.shape(), &output_tensor));
        
        saguaro::hd_thought::HDThoughtTraceUpdateGrad(
            grad_tensor.flat<float>().data(),
            thought_tensor.flat<float>().data(),
            output_tensor->flat<float>().data(),
            step_, batch_size, hd_dim
        );
    }

private:
    int step_;
};

// =============================================================================
// Kernel Registrations
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HDThoughtTraceUpdate").Device(DEVICE_CPU),
    HDThoughtTraceUpdateOp);

REGISTER_KERNEL_BUILDER(
    Name("HDThoughtTraceRetrieve").Device(DEVICE_CPU),
    HDThoughtTraceRetrieveOp);

REGISTER_KERNEL_BUILDER(
    Name("HDThoughtPathScores").Device(DEVICE_CPU),
    HDThoughtPathScoresOp);

REGISTER_KERNEL_BUILDER(
    Name("HDThoughtTraceUpdateGrad").Device(DEVICE_CPU),
    HDThoughtTraceUpdateGradOp);

}  // namespace tensorflow
