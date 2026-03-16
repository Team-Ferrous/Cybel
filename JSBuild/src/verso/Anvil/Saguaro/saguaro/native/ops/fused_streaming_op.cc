// saguaro.native/ops/fused_streaming_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// TensorFlow Ops for KV-free streaming inference.

#include "fused_streaming_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("FusedStreamingCompress")
    .Input("state: float32")  // [batch, state_dim]
    .Output("compressed: float32")  // [batch, target_dim]
    .Attr("target_dim: int")
    .SetShapeFn([](InferenceContext* c) {
        int target_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("target_dim", &target_dim));
        ShapeHandle state = c->input(0);
        auto batch = c->Dim(state, 0);
        c->set_output(0, c->MakeShape({batch, target_dim}));
        return Status();
    })
    .Doc("Compress streaming state to target dimension.");

REGISTER_OP("FusedStreamingUpdate")
    .Input("old_state: float32")  // [batch, state_dim]
    .Input("new_state: float32")  // [batch, state_dim]
    .Output("updated: float32")  // [batch, state_dim]
    .Attr("alpha: float = 0.9")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc("Update streaming state with EMA.");

class FusedStreamingCompressOp : public OpKernel {
 public:
    explicit FusedStreamingCompressOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("target_dim", &target_dim_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& state = ctx->input(0);
        const int64_t batch_size = state.dim_size(0);
        const int64_t state_dim = state.dim_size(1);

        Tensor* compressed = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, target_dim_}), &compressed));

        saguaro::ops::streaming_compress_state(
            state.flat<float>().data(),
            compressed->flat<float>().data(),
            batch_size, state_dim, target_dim_);
    }

 private:
    int target_dim_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedStreamingCompress").Device(DEVICE_CPU),
    FusedStreamingCompressOp);

class FusedStreamingUpdateOp : public OpKernel {
 public:
    explicit FusedStreamingUpdateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& old_state = ctx->input(0);
        const Tensor& new_state = ctx->input(1);
        
        const int64_t batch_size = old_state.dim_size(0);
        const int64_t state_dim = old_state.dim_size(1);

        Tensor* updated = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, old_state.shape(), &updated));

        // Copy old state to output
        std::copy(old_state.flat<float>().data(),
                  old_state.flat<float>().data() + batch_size * state_dim,
                  updated->flat<float>().data());

        // Update in-place
        saguaro::ops::streaming_update_state(
            updated->flat<float>().data(),
            new_state.flat<float>().data(),
            batch_size, state_dim, alpha_);
    }

 private:
    float alpha_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedStreamingUpdate").Device(DEVICE_CPU),
    FusedStreamingUpdateOp);

}  // namespace tensorflow
