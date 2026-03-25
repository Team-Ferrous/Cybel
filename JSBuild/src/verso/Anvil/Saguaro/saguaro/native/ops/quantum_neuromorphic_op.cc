// saguaro.native/ops/quantum_neuromorphic_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantum_neuromorphic_op.h"
using namespace tensorflow;

REGISTER_OP("SpikingQuantumNeuron")
    .Input("input: float").Input("membrane_potential: float")
    .Output("spikes: float").Output("new_potential: float")
    .Attr("threshold: float = 1.0").Attr("tau: float = 10.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); c->set_output(1, c->input(1)); return Status();
    }).Doc("Phase 68: Spiking quantum neuron with leaky integrate-and-fire.");

class SpikingQuantumNeuronOp : public OpKernel {
 public:
  explicit SpikingQuantumNeuronOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &thresh_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tau", &tau_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto inp = ctx->input(0), mp = ctx->input(1);
    int batch = inp.dim_size(0), neurons = inp.dim_size(1);
    Tensor *spikes, *new_mp;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, inp.shape(), &spikes));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, mp.shape(), &new_mp));
    saguaro::neuromorphic::SpikingQuantumNeuron(inp.flat<float>().data(), mp.flat<float>().data(),
        spikes->flat<float>().data(), new_mp->flat<float>().data(), thresh_, tau_, batch, neurons);
  }
 private:
  float thresh_, tau_;
};
REGISTER_KERNEL_BUILDER(Name("SpikingQuantumNeuron").Device(DEVICE_CPU), SpikingQuantumNeuronOp);
