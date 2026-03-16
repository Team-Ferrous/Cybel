// saguaro.native/ops/quantum_advanced_ops.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantum_advanced_ops.h"
using namespace tensorflow;

REGISTER_OP("NQSDecoder")
    .Input("visible: float").Input("weights: float").Input("bias: float")
    .Output("hidden: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        auto h_dim = c->Dim(c->input(2), 0);
        c->set_output(0, c->MakeShape({batch, h_dim})); return Status();
    }).Doc("Phase 73: Neural Quantum State decoder.");

class NQSDecoderOp : public OpKernel {
 public:
  explicit NQSDecoderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    auto v = ctx->input(0), w = ctx->input(1), b = ctx->input(2);
    int batch = v.dim_size(0), v_dim = v.dim_size(1), h_dim = b.dim_size(0);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, h_dim}, &out));
    saguaro::qadvanced::NQSDecoder(v.flat<float>().data(), w.flat<float>().data(),
        b.flat<float>().data(), out->flat<float>().data(), batch, v_dim, h_dim);
  }
};
REGISTER_KERNEL_BUILDER(Name("NQSDecoder").Device(DEVICE_CPU), NQSDecoderOp);

REGISTER_OP("QCOTReason")
    .Input("thought: float").Input("reasoning_weights: float")
    .Output("next_thought: float").Attr("steps: int = 3")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); return Status();
    }).Doc("Phase 79: Quantum chain-of-thought reasoning.");

class QCOTReasonOp : public OpKernel {
 public:
  explicit QCOTReasonOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("steps", &steps_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto thought = ctx->input(0), weights = ctx->input(1);
    int batch = thought.dim_size(0), dim = thought.dim_size(1);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, thought.shape(), &out));
    saguaro::qadvanced::QCOTReason(thought.flat<float>().data(), weights.flat<float>().data(),
        out->flat<float>().data(), batch, dim, steps_);
  }
 private:
  int steps_;
};
REGISTER_KERNEL_BUILDER(Name("QCOTReason").Device(DEVICE_CPU), QCOTReasonOp);

REGISTER_OP("WaveformAttention")
    .Input("input: float").Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        auto dim = c->Dim(c->input(0), 2);
        c->set_output(0, c->MakeShape({batch, dim})); return Status();
    }).Doc("Phase 80: Waveform-based attention pooling.");

class WaveformAttentionOp : public OpKernel {
 public:
  explicit WaveformAttentionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    auto input = ctx->input(0);
    int batch = input.dim_size(0), seq = input.dim_size(1), dim = input.dim_size(2);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, dim}, &out));
    saguaro::qadvanced::WaveformAttention(input.flat<float>().data(),
        out->flat<float>().data(), batch, seq, dim);
  }
};
REGISTER_KERNEL_BUILDER(Name("WaveformAttention").Device(DEVICE_CPU), WaveformAttentionOp);

REGISTER_OP("ComputeCoherence")
    .Input("state: float").Output("coherence: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar()); return Status();
    }).Doc("Phase 84: Compute coherence metric for training loop.");

class ComputeCoherenceOp : public OpKernel {
 public:
  explicit ComputeCoherenceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    auto state = ctx->input(0);
    int dim = state.NumElements();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {}, &out));
    out->scalar<float>()() = saguaro::qadvanced::ComputeCoherenceMetric(state.flat<float>().data(), dim);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeCoherence").Device(DEVICE_CPU), ComputeCoherenceOp);
