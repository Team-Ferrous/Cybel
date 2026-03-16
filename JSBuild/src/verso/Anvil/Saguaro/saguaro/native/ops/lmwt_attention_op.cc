// saguaro.native/ops/lmwt_attention_op.cc
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
 * @file lmwt_attention_op.cc
 * @brief Phase 41: LMWT TensorFlow custom operations.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "lmwt_attention_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: LMWTForward
// =============================================================================

REGISTER_OP("LMWTForward")
    .Input("x: float")                    // [batch, seq_len, dim]
    .Input("alpha: float")                // [num_scales]
    .Input("beta: float")                 // [num_scales]
    .Output("output: float")              // [batch, seq_len, dim]
    .Attr("num_scales: int = 4")
    .Attr("num_heads: int = 8")
    .Attr("learn_filters: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 41: Learnable Multi-Scale Wavelet Transformer Forward.

Applies learnable wavelet decomposition, cross-scale attention,
and reconstruction:
1. Multi-scale Haar decomposition with learnable α, β
2. Cross-scale attention between low and high frequencies
3. Learnable reconstruction back to sequence

x: Input sequence [batch, seq_len, dim]
alpha: Learnable low-pass filter parameters [num_scales]
beta: Learnable high-pass filter parameters [num_scales]

output: Transformed sequence [batch, seq_len, dim]
)doc");

class LMWTForwardOp : public OpKernel {
 public:
  explicit LMWTForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_scales", &config_.num_scales));
    OP_REQUIRES_OK(context, context->GetAttr("num_heads", &config_.num_heads));
    OP_REQUIRES_OK(context, context->GetAttr("learn_filters", &config_.learn_filters));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& alpha = context->input(1);
    const Tensor& beta = context->input(2);

    OP_REQUIRES(context, x.dims() == 3,
                errors::InvalidArgument("x must be 3D [batch, seq, dim]"));

    const int batch_size = x.dim_size(0);
    const int seq_len = x.dim_size(1);
    const int dim = x.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &output));

    saguaro::lmwt::LMWTForward(
        x.flat<float>().data(),
        output->flat<float>().data(),
        alpha.flat<float>().data(),
        beta.flat<float>().data(),
        config_,
        batch_size, seq_len, dim
    );
  }

 private:
  saguaro::lmwt::LMWTConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("LMWTForward").Device(DEVICE_CPU), LMWTForwardOp);

// =============================================================================
// OP REGISTRATION: LearnableHaarDecompose
// =============================================================================

REGISTER_OP("LearnableHaarDecompose")
    .Input("x: float")                    // [batch, seq_len, dim]
    .Input("alpha: float")                // scalar
    .Input("beta: float")                 // scalar
    .Output("low: float")                 // [batch, seq_len/2, dim]
    .Output("high: float")                // [batch, seq_len/2, dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle x = c->input(0);
        if (c->RankKnown(x) && c->Rank(x) == 3) {
            auto batch = c->Dim(x, 0);
            auto seq = c->Dim(x, 1);
            auto dim = c->Dim(x, 2);
            // Output is half length
            c->set_output(0, c->MakeShape({batch, c->UnknownDim(), dim}));
            c->set_output(1, c->MakeShape({batch, c->UnknownDim(), dim}));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Learnable Haar wavelet decomposition step.

low[i] = α * (x[2i] + x[2i+1])
high[i] = β * (x[2i] - x[2i+1])

x: Input signal [batch, seq_len, dim]
alpha: Low-pass filter parameter (learnable)
beta: High-pass filter parameter (learnable)

low: Low-frequency coefficients [batch, seq_len/2, dim]
high: High-frequency coefficients [batch, seq_len/2, dim]
)doc");

class LearnableHaarDecomposeOp : public OpKernel {
 public:
  explicit LearnableHaarDecomposeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& alpha_t = context->input(1);
    const Tensor& beta_t = context->input(2);

    const int batch_size = x.dim_size(0);
    const int seq_len = x.dim_size(1);
    const int dim = x.dim_size(2);
    const int half_len = seq_len / 2;

    OP_REQUIRES(context, seq_len % 2 == 0,
                errors::InvalidArgument("seq_len must be even for Haar decomposition"));

    Tensor* low = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, half_len, dim}), &low));
    
    Tensor* high = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, half_len, dim}), &high));

    float alpha = alpha_t.scalar<float>()();
    float beta = beta_t.scalar<float>()();

    saguaro::lmwt::LearnableHaarDecompose(
        x.flat<float>().data(),
        low->flat<float>().data(),
        high->flat<float>().data(),
        alpha, beta,
        batch_size, seq_len, dim
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("LearnableHaarDecompose").Device(DEVICE_CPU),
                        LearnableHaarDecomposeOp);

// =============================================================================
// OP REGISTRATION: LearnableHaarReconstruct
// =============================================================================

REGISTER_OP("LearnableHaarReconstruct")
    .Input("low: float")                  // [batch, seq_len/2, dim]
    .Input("high: float")                 // [batch, seq_len/2, dim]
    .Input("alpha: float")                // scalar
    .Input("beta: float")                 // scalar
    .Output("x: float")                   // [batch, seq_len, dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle low = c->input(0);
        if (c->RankKnown(low) && c->Rank(low) == 3) {
            auto batch = c->Dim(low, 0);
            auto dim = c->Dim(low, 2);
            c->set_output(0, c->MakeShape({batch, c->UnknownDim(), dim}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Learnable Haar wavelet reconstruction step.

x[2i] = (low[i] / α + high[i] / β) / 2
x[2i+1] = (low[i] / α - high[i] / β) / 2

low: Low-frequency coefficients [batch, seq_len/2, dim]
high: High-frequency coefficients [batch, seq_len/2, dim]
alpha: Low-pass filter parameter
beta: High-pass filter parameter

x: Reconstructed signal [batch, seq_len, dim]
)doc");

class LearnableHaarReconstructOp : public OpKernel {
 public:
  explicit LearnableHaarReconstructOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& low = context->input(0);
    const Tensor& high = context->input(1);
    const Tensor& alpha_t = context->input(2);
    const Tensor& beta_t = context->input(3);

    const int batch_size = low.dim_size(0);
    const int half_len = low.dim_size(1);
    const int dim = low.dim_size(2);
    const int seq_len = half_len * 2;

    Tensor* x = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len, dim}), &x));

    float alpha = alpha_t.scalar<float>()();
    float beta = beta_t.scalar<float>()();

    saguaro::lmwt::LearnableHaarReconstruct(
        low.flat<float>().data(),
        high.flat<float>().data(),
        x->flat<float>().data(),
        alpha, beta,
        batch_size, half_len, dim
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("LearnableHaarReconstruct").Device(DEVICE_CPU),
                        LearnableHaarReconstructOp);

// =============================================================================
// OP REGISTRATION: CrossScaleAttention
// =============================================================================

REGISTER_OP("CrossScaleAttention")
    .Input("coeff_low: float")            // [batch, len, dim]
    .Input("coeff_high: float")           // [batch, len, dim]
    .Output("output: float")              // [batch, len, dim]
    .Attr("num_heads: int = 8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Cross-scale attention between wavelet coefficients.

Low frequency attends to high frequency for multi-resolution fusion.

coeff_low: Low frequency coefficients [batch, len, dim]
coeff_high: High frequency coefficients [batch, len, dim]

output: Attended output [batch, len, dim]
)doc");

class CrossScaleAttentionOp : public OpKernel {
 public:
  explicit CrossScaleAttentionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_heads", &num_heads_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& low = context->input(0);
    const Tensor& high = context->input(1);

    const int batch_size = low.dim_size(0);
    const int len = low.dim_size(1);
    const int dim = low.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, low.shape(), &output));

    saguaro::lmwt::CrossScaleAttention(
        low.flat<float>().data(),
        high.flat<float>().data(),
        output->flat<float>().data(),
        batch_size, len, dim, num_heads_
    );
  }

 private:
  int num_heads_;
};

REGISTER_KERNEL_BUILDER(Name("CrossScaleAttention").Device(DEVICE_CPU),
                        CrossScaleAttentionOp);

// =============================================================================
// PHASE 88: LEARNABLE FILTER BANK DWT
// =============================================================================

REGISTER_OP("LearnableFilterBankDWT")
    .Input("x: float")                    // [batch, seq_len, dim]
    .Input("low_pass: float")             // [kernel_size]
    .Input("high_pass: float")            // [kernel_size] (or derived via QMF)
    .Output("low: float")                 // [batch, seq_len/2, dim]
    .Output("high: float")                // [batch, seq_len/2, dim]
    .Attr("kernel_size: int")
    .Attr("enforce_qmf: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle x = c->input(0);
        if (c->RankKnown(x) && c->Rank(x) == 3) {
            auto batch = c->Dim(x, 0);
            auto dim = c->Dim(x, 2);
            c->set_output(0, c->MakeShape({batch, c->UnknownDim(), dim}));
            c->set_output(1, c->MakeShape({batch, c->UnknownDim(), dim}));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 88: Learnable Filter Bank DWT decomposition.

Uses learnable convolution filters for multi-resolution analysis:
  low[i] = sum_k(low_pass[k] * x[2i + k])
  high[i] = sum_k(high_pass[k] * x[2i + k])

When enforce_qmf=true, high_pass is derived from low_pass via QMF constraint.

x: Input signal [batch, seq_len, dim]
low_pass: Learnable low-pass filter [kernel_size]
high_pass: High-pass filter [kernel_size] (ignored if enforce_qmf=true)

low: Low-frequency coefficients [batch, seq_len/2, dim]
high: High-frequency coefficients [batch, seq_len/2, dim]
)doc");

class LearnableFilterBankDWTOp : public OpKernel {
 public:
  explicit LearnableFilterBankDWTOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size_));
    OP_REQUIRES_OK(context, context->GetAttr("enforce_qmf", &enforce_qmf_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& low_pass_t = context->input(1);
    const Tensor& high_pass_t = context->input(2);

    const int batch_size = x.dim_size(0);
    const int seq_len = x.dim_size(1);
    const int dim = x.dim_size(2);
    const int half_len = seq_len / 2;

    OP_REQUIRES(context, seq_len % 2 == 0,
                errors::InvalidArgument("seq_len must be even"));

    Tensor* low = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, half_len, dim}), &low));
    
    Tensor* high = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, half_len, dim}), &high));

    const float* low_pass = low_pass_t.flat<float>().data();
    std::vector<float> high_pass_buf(kernel_size_);
    const float* high_pass;
    
    if (enforce_qmf_) {
      saguaro::lmwt_v2::ApplyQMFConstraint(low_pass, high_pass_buf.data(), kernel_size_);
      high_pass = high_pass_buf.data();
    } else {
      high_pass = high_pass_t.flat<float>().data();
    }

    saguaro::lmwt_v2::LearnableFilterBankDecompose(
        x.flat<float>().data(),
        low->flat<float>().data(), high->flat<float>().data(),
        low_pass, high_pass,
        batch_size, seq_len, dim, kernel_size_
    );
  }

 private:
  int kernel_size_;
  bool enforce_qmf_;
};

REGISTER_KERNEL_BUILDER(Name("LearnableFilterBankDWT").Device(DEVICE_CPU),
                        LearnableFilterBankDWTOp);

// =============================================================================
// PHASE 88: LEARNABLE FILTER BANK IWT
// =============================================================================

REGISTER_OP("LearnableFilterBankIWT")
    .Input("low: float")                  // [batch, half_len, dim]
    .Input("high: float")                 // [batch, half_len, dim]
    .Input("synth_low: float")            // [kernel_size]
    .Input("synth_high: float")           // [kernel_size]
    .Output("x: float")                   // [batch, seq_len, dim]
    .Attr("kernel_size: int")
    .Attr("enforce_qmf: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle low = c->input(0);
        if (c->RankKnown(low) && c->Rank(low) == 3) {
            auto batch = c->Dim(low, 0);
            auto dim = c->Dim(low, 2);
            c->set_output(0, c->MakeShape({batch, c->UnknownDim(), dim}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 88: Learnable Filter Bank IWT reconstruction.

Reconstructs signal from wavelet coefficients using synthesis filters.

low: Low-frequency coefficients [batch, half_len, dim]
high: High-frequency coefficients [batch, half_len, dim]
synth_low: Synthesis low-pass filter [kernel_size]
synth_high: Synthesis high-pass filter [kernel_size]

x: Reconstructed signal [batch, seq_len, dim]
)doc");

class LearnableFilterBankIWTOp : public OpKernel {
 public:
  explicit LearnableFilterBankIWTOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &kernel_size_));
    OP_REQUIRES_OK(context, context->GetAttr("enforce_qmf", &enforce_qmf_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& low = context->input(0);
    const Tensor& high = context->input(1);
    const Tensor& synth_low_t = context->input(2);
    const Tensor& synth_high_t = context->input(3);

    const int batch_size = low.dim_size(0);
    const int half_len = low.dim_size(1);
    const int dim = low.dim_size(2);
    const int seq_len = half_len * 2;

    Tensor* x = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len, dim}), &x));

    const float* synth_low = synth_low_t.flat<float>().data();
    std::vector<float> synth_high_buf(kernel_size_);
    const float* synth_high;
    
    if (enforce_qmf_) {
      saguaro::lmwt_v2::ApplyQMFConstraint(synth_low, synth_high_buf.data(), kernel_size_);
      synth_high = synth_high_buf.data();
    } else {
      synth_high = synth_high_t.flat<float>().data();
    }

    saguaro::lmwt_v2::LearnableFilterBankReconstruct(
        low.flat<float>().data(), high.flat<float>().data(),
        x->flat<float>().data(),
        synth_low, synth_high,
        batch_size, half_len, dim, kernel_size_
    );
  }

 private:
  int kernel_size_;
  bool enforce_qmf_;
};

REGISTER_KERNEL_BUILDER(Name("LearnableFilterBankIWT").Device(DEVICE_CPU),
                        LearnableFilterBankIWTOp);

// =============================================================================
// PHASE 88: WAVELET MOE ROUTING BIAS
// =============================================================================

REGISTER_OP("WaveletMoERoutingBias")
    .Input("coeff_low: float")            // [batch, len, dim]
    .Input("coeff_high: float")           // [batch, len, dim]
    .Output("routing_bias: float")        // [batch, len, num_experts]
    .Attr("num_experts: int")
    .Attr("bias_scale: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int num_experts;
        c->GetAttr("num_experts", &num_experts);
        shape_inference::ShapeHandle low = c->input(0);
        if (c->RankKnown(low) && c->Rank(low) == 3) {
            auto batch = c->Dim(low, 0);
            auto len = c->Dim(low, 1);
            c->set_output(0, c->MakeShape({batch, len, num_experts}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 88: Wavelet-domain MoE routing bias.

Computes frequency-based routing bias for expert selection:
- High-frequency tokens → detail/syntax experts
- Low-frequency tokens → semantic/reasoning experts

coeff_low: Low frequency wavelet coefficients [batch, len, dim]
coeff_high: High frequency wavelet coefficients [batch, len, dim]

routing_bias: Expert routing bias [batch, len, num_experts]
)doc");

class WaveletMoERoutingBiasOp : public OpKernel {
 public:
  explicit WaveletMoERoutingBiasOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_experts", &num_experts_));
    OP_REQUIRES_OK(context, context->GetAttr("bias_scale", &bias_scale_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& low = context->input(0);
    const Tensor& high = context->input(1);

    const int batch_size = low.dim_size(0);
    const int len = low.dim_size(1);
    const int dim = low.dim_size(2);

    Tensor* routing_bias = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, len, num_experts_}), &routing_bias));

    saguaro::lmwt_v2::WaveletMoERoutingBias(
        low.flat<float>().data(), high.flat<float>().data(),
        routing_bias->flat<float>().data(),
        nullptr,  // freq_proj not used in simplified version
        batch_size, len, dim, num_experts_, bias_scale_
    );
  }

 private:
  int num_experts_;
  float bias_scale_;
};

REGISTER_KERNEL_BUILDER(Name("WaveletMoERoutingBias").Device(DEVICE_CPU),
                        WaveletMoERoutingBiasOp);

// =============================================================================
// PHASE 88: CROSS-SCALE LINEAR ATTENTION
// =============================================================================

REGISTER_OP("CrossScaleLinearAttention")
    .Input("coeff_low: float")            // [batch, len, dim]
    .Input("coeff_high: float")           // [batch, len, dim]
    .Input("gate_weight: float")          // [dim]
    .Output("output: float")              // [batch, len, dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 88: Cross-scale linear attention.

Fuses low and high frequency wavelet coefficients using O(n) linear attention.

coeff_low: Low frequency coefficients (queries) [batch, len, dim]
coeff_high: High frequency coefficients (keys/values) [batch, len, dim]
gate_weight: Learned gate for fusion [dim]

output: Fused output [batch, len, dim]
)doc");

class CrossScaleLinearAttentionOp : public OpKernel {
 public:
  explicit CrossScaleLinearAttentionOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& low = context->input(0);
    const Tensor& high = context->input(1);
    const Tensor& gate = context->input(2);

    const int batch_size = low.dim_size(0);
    const int len = low.dim_size(1);
    const int dim = low.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, low.shape(), &output));

    saguaro::lmwt_v2::CrossScaleLinearAttention(
        low.flat<float>().data(), high.flat<float>().data(),
        output->flat<float>().data(),
        gate.flat<float>().data(),
        batch_size, len, dim
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("CrossScaleLinearAttention").Device(DEVICE_CPU),
                        CrossScaleLinearAttentionOp);

// =============================================================================
// PHASE 88: FULL LMWTv2 FORWARD
// =============================================================================

REGISTER_OP("LMWTv2Forward")
    .Input("x: float")                    // [batch, seq_len, dim]
    .Input("low_pass_filters: float")     // [num_levels, kernel_size]
    .Input("synth_filters: float")        // [num_levels, kernel_size]
    .Input("gate_weights: float")         // [num_levels, dim]
    .Output("output: float")              // [batch, seq_len, dim]
    .Attr("num_levels: int = 4")
    .Attr("kernel_size: int = 5")
    .Attr("enforce_qmf: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 88: Full LMWTv2 forward pass.

Multi-scale learnable wavelet transform with cross-scale attention:
1. Learnable filter bank DWT decomposition
2. Cross-scale linear attention at each level
3. Learnable IWT reconstruction

x: Input sequence [batch, seq_len, dim]
low_pass_filters: Low-pass filters per level [num_levels, kernel_size]
synth_filters: Synthesis filters per level [num_levels, kernel_size]
gate_weights: Cross-scale gate per level [num_levels, dim]

output: Transformed sequence [batch, seq_len, dim]
)doc");

class LMWTv2ForwardOp : public OpKernel {
 public:
  explicit LMWTv2ForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_levels", &config_.num_levels));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_size", &config_.kernel_size));
    OP_REQUIRES_OK(context, context->GetAttr("enforce_qmf", &config_.enforce_qmf));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& low_pass = context->input(1);
    const Tensor& synth = context->input(2);
    const Tensor& gates = context->input(3);

    const int batch_size = x.dim_size(0);
    const int seq_len = x.dim_size(1);
    const int dim = x.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &output));

    saguaro::lmwt_v2::LMWTv2Forward(
        x.flat<float>().data(),
        output->flat<float>().data(),
        low_pass.flat<float>().data(),
        synth.flat<float>().data(),
        gates.flat<float>().data(),
        config_,
        batch_size, seq_len, dim
    );
  }

 private:
  saguaro::lmwt_v2::LMWTv2Config config_;
};

REGISTER_KERNEL_BUILDER(Name("LMWTv2Forward").Device(DEVICE_CPU), LMWTv2ForwardOp);

