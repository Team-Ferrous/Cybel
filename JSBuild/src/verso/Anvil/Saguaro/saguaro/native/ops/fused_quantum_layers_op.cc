// saguaro.native/ops/fused_quantum_layers_op.cc
#include "fused_quantum_layers_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
using shape_inference::InferenceContext;

REGISTER_OP("FusedQuantumRotation")
    .Input("angles: float32")
    .Output("matrix_real: float32")
    .Output("matrix_imag: float32")
    .SetShapeFn([](InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        c->set_output(0, c->MakeShape({batch, 2, 2}));
        c->set_output(1, c->MakeShape({batch, 2, 2}));
        return Status();
    })
    .Doc("Compute quantum rotation matrices.");

class FusedQuantumRotationOp : public OpKernel {
 public:
    explicit FusedQuantumRotationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& angles = ctx->input(0);
        const int64_t batch = angles.dim_size(0);
        
        Tensor* real = nullptr;
        Tensor* imag = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch, 2, 2}), &real));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({batch, 2, 2}), &imag));
        
        const float* ang = angles.flat<float>().data();
        float* r = real->flat<float>().data();
        float* i = imag->flat<float>().data();
        
        for (int64_t b = 0; b < batch; ++b) {
            saguaro::ops::quantum_rotation_z(ang[b], r + b * 4, i + b * 4);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedQuantumRotation").Device(DEVICE_CPU), FusedQuantumRotationOp);
} // namespace tensorflow
