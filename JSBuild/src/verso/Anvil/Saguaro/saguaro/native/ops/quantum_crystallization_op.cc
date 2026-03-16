// saguaro.native/ops/quantum_crystallization_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantum_crystallization_op.h"
using namespace tensorflow;

REGISTER_OP("CrystallizeMemory")
    .Input("knowledge: float").Input("importance: float")
    .Output("crystal: float").Attr("threshold: float = 0.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); return Status();
    }).Doc("Phase 65: Crystallize important knowledge for persistence.");

class CrystallizeMemoryOp : public OpKernel {
 public:
  explicit CrystallizeMemoryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &thresh_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto k = ctx->input(0), imp = ctx->input(1);
    int batch = k.dim_size(0), dim = k.dim_size(1);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, k.shape(), &out));
    saguaro::qcrystal::CrystallizeMemory(k.flat<float>().data(), imp.flat<float>().data(),
        out->flat<float>().data(), thresh_, batch, dim);
  }
 private:
  float thresh_;
};
REGISTER_KERNEL_BUILDER(Name("CrystallizeMemory").Device(DEVICE_CPU), CrystallizeMemoryOp);

REGISTER_OP("RetrieveFromCrystal")
    .Input("crystal: float").Input("query: float")
    .Output("retrieved: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1)); return Status();
    }).Doc("Phase 65: Retrieve from crystallized memory.");

class RetrieveFromCrystalOp : public OpKernel {
 public:
  explicit RetrieveFromCrystalOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    auto cr = ctx->input(0), q = ctx->input(1);
    int batch = q.dim_size(0), dim = q.dim_size(1);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, q.shape(), &out));
    saguaro::qcrystal::RetrieveFromCrystal(cr.flat<float>().data(), q.flat<float>().data(),
        out->flat<float>().data(), batch, dim);
  }
};
REGISTER_KERNEL_BUILDER(Name("RetrieveFromCrystal").Device(DEVICE_CPU), RetrieveFromCrystalOp);
