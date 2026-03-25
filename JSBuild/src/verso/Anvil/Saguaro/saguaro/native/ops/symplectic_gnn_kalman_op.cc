// saguaro.native/ops/symplectic_gnn_kalman_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "symplectic_gnn_kalman_op.h"

using namespace tensorflow;

REGISTER_OP("SymplecticGNNKalman")
    .Input("node_q: float")
    .Input("node_p: float")
    .Input("edges: int32")
    .Input("observations: float")
    .Input("kalman_gain: float")
    .Output("output_q: float")
    .Output("output_p: float")
    .Attr("dt: float = 0.01")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status();
    })
    .Doc("Phase 58: Symplectic GNN-Kalman for Hamiltonian dynamics prediction.");

class SymplecticGNNKalmanOp : public OpKernel {
 public:
  explicit SymplecticGNNKalmanOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dt", &dt_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto node_q = ctx->input(0);
    auto node_p = ctx->input(1);
    auto edges = ctx->input(2);
    auto obs = ctx->input(3);
    auto gain = ctx->input(4);
    
    int batch = node_q.dim_size(0), num_nodes = node_q.dim_size(1), dim = node_q.dim_size(2);
    int num_edges = edges.dim_size(0);
    
    Tensor* out_q = nullptr;
    Tensor* out_p = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, node_q.shape(), &out_q));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, node_p.shape(), &out_p));
    
    std::copy_n(node_q.flat<float>().data(), batch * num_nodes * dim, out_q->flat<float>().data());
    std::copy_n(node_p.flat<float>().data(), batch * num_nodes * dim, out_p->flat<float>().data());
    
    saguaro::sgkf::SymplecticGNNKalman(out_q->flat<float>().data(), out_p->flat<float>().data(),
        edges.flat<int>().data(), num_edges, obs.flat<float>().data(),
        gain.flat<float>().data(), dt_, batch, num_nodes, dim);
  }
 private:
  float dt_;
};
REGISTER_KERNEL_BUILDER(Name("SymplecticGNNKalman").Device(DEVICE_CPU), SymplecticGNNKalmanOp);
