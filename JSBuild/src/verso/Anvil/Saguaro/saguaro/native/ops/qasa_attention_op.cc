// saguaro.native/ops/qasa_attention_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "qasa_attention_op.h"

using namespace tensorflow;

REGISTER_OP("QASAAttention")
    .Input("queries: float")
    .Input("keys: float")
    .Input("values: float")
    .Input("vqc_params: float")
    .Output("output: float")
    .Attr("num_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("entanglement_strength: float = 0.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc("Phase 53/119: Quantum Adaptive Self-Attention with VQC scoring.");

// Gradient op for QASA - computes gradients w.r.t. VQC params via finite differences
REGISTER_OP("QASAAttentionGrad")
    .Input("queries: float")
    .Input("keys: float")
    .Input("values: float")
    .Input("vqc_params: float")
    .Input("grad_output: float")
    .Output("grad_vqc_params: float")
    .Attr("num_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("entanglement_strength: float = 0.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(3));  // Same shape as vqc_params
        return Status();
    })
    .Doc("Gradient for QASAAttention - numerical differentiation on VQC params.");

class QASAAttentionOp : public OpKernel {
 public:
  explicit QASAAttentionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_qubits", &config_.num_qubits));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_layers", &config_.vqc_layers));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("entanglement_strength", &config_.entanglement_strength));
  }
  void Compute(OpKernelContext* ctx) override {
    auto queries = ctx->input(0);
    auto keys = ctx->input(1);
    auto values = ctx->input(2);
    auto vqc_params = ctx->input(3);
    int batch = queries.dim_size(0), heads = queries.dim_size(1);
    int seq = queries.dim_size(2), head_dim = queries.dim_size(3);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, queries.shape(), &output));
    saguaro::qasa::QASAAttention(queries.flat<float>().data(), keys.flat<float>().data(),
        values.flat<float>().data(), vqc_params.flat<float>().data(),
        output->flat<float>().data(), config_, batch, heads, seq, head_dim);
  }
 private:
  saguaro::qasa::QASAConfig config_;
};
REGISTER_KERNEL_BUILDER(Name("QASAAttention").Device(DEVICE_CPU), QASAAttentionOp);

class QASAAttentionGradOp : public OpKernel {
 public:
  explicit QASAAttentionGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_qubits", &config_.num_qubits));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_layers", &config_.vqc_layers));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("entanglement_strength", &config_.entanglement_strength));
  }

  void Compute(OpKernelContext* ctx) override {
    auto queries = ctx->input(0);
    auto keys = ctx->input(1);
    auto values = ctx->input(2);
    auto vqc_params = ctx->input(3);
    auto grad_output = ctx->input(4);

    int batch = queries.dim_size(0), heads = queries.dim_size(1);
    int seq = queries.dim_size(2), head_dim = queries.dim_size(3);
    int num_params = vqc_params.dim_size(0);

    // Allocate output gradient
    Tensor* grad_vqc = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, vqc_params.shape(), &grad_vqc));

    // Get flat data pointers
    const float* q_ptr = queries.flat<float>().data();
    const float* k_ptr = keys.flat<float>().data();
    const float* v_ptr = values.flat<float>().data();
    const float* grad_out_ptr = grad_output.flat<float>().data();
    std::vector<float> params(vqc_params.flat<float>().data(),
                               vqc_params.flat<float>().data() + num_params);
    float* grad_ptr = grad_vqc->flat<float>().data();

    // Finite difference epsilon
    const float eps = 1e-4f;
    const int output_size = batch * heads * seq * head_dim;

    // Allocate temp buffers for forward/backward perturbation
    std::vector<float> out_plus(output_size), out_minus(output_size);

    // Compute gradient via central finite differences
    #pragma omp parallel for
    for (int p = 0; p < num_params; ++p) {
      std::vector<float> params_plus = params;
      std::vector<float> params_minus = params;
      params_plus[p] += eps;
      params_minus[p] -= eps;

      std::vector<float> local_out_plus(output_size), local_out_minus(output_size);

      saguaro::qasa::QASAAttention(q_ptr, k_ptr, v_ptr, params_plus.data(),
          local_out_plus.data(), config_, batch, heads, seq, head_dim);
      saguaro::qasa::QASAAttention(q_ptr, k_ptr, v_ptr, params_minus.data(),
          local_out_minus.data(), config_, batch, heads, seq, head_dim);

      // Compute gradient contribution: sum over (grad_out * d_out/d_param)
      float grad_sum = 0.0f;
      for (int i = 0; i < output_size; ++i) {
        float d_out_d_param = (local_out_plus[i] - local_out_minus[i]) / (2.0f * eps);
        grad_sum += grad_out_ptr[i] * d_out_d_param;
      }
      grad_ptr[p] = grad_sum;
    }
  }

 private:
  saguaro::qasa::QASAConfig config_;
};
REGISTER_KERNEL_BUILDER(Name("QASAAttentionGrad").Device(DEVICE_CPU), QASAAttentionGradOp);

