// saguaro.native/ops/quantum_measurement_dropout_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "quantum_measurement_dropout_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: QuantumMeasurementDropout
// =============================================================================

REGISTER_OP("QuantumMeasurementDropout")
    .Input("input: float")                // [batch, seq, dim]
    .Output("output: float")              // [batch, seq, dim]
    .Attr("drop_rate: float = 0.1")
    .Attr("seed: int = 42")
    .Attr("training: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 47: Quantum Measurement Dropout.

Randomly measures selected positions, collapsing their quantum state.
Creates ensemble effect through varying effective circuit depths.

input: Input activations [batch, seq, dim]

output: Dropped activations [batch, seq, dim]
)doc");

class QuantumMeasurementDropoutOp : public OpKernel {
 public:
  explicit QuantumMeasurementDropoutOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("drop_rate", &config_.drop_rate));
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const int batch_size = input.dim_size(0);
    const int seq_len = input.dim_size(1);
    const int dim = input.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    if (training_) {
      saguaro::qmd::QuantumMeasurementDropout(
          input.flat<float>().data(),
          output->flat<float>().data(),
          nullptr,  // Generate mask internally
          config_,
          batch_size, seq_len, dim
      );
    } else {
      // Inference: pass through
      std::copy_n(input.flat<float>().data(), batch_size * seq_len * dim,
                  output->flat<float>().data());
    }
  }

 private:
  saguaro::qmd::QMDropoutConfig config_;
  bool training_;
};

REGISTER_KERNEL_BUILDER(Name("QuantumMeasurementDropout").Device(DEVICE_CPU),
                        QuantumMeasurementDropoutOp);

// =============================================================================
// OP REGISTRATION: SoftQuantumDropout
// =============================================================================

REGISTER_OP("SoftQuantumDropout")
    .Input("input: float")                // [batch, seq, dim]
    .Input("softening_params: float")     // [dim]
    .Output("output: float")              // [batch, seq, dim]
    .Attr("drop_rate: float = 0.1")
    .Attr("temperature: float = 1.0")
    .Attr("seed: int = 42")
    .Attr("training: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 47: Soft Quantum Dropout with learned softening.

Uses soft measurement operator: M_soft = (1-σ)·I + σ·|0⟩⟨0|

input: Input activations [batch, seq, dim]
softening_params: Learnable softening parameters [dim]

output: Softly dropped activations [batch, seq, dim]
)doc");

class SoftQuantumDropoutOp : public OpKernel {
 public:
  explicit SoftQuantumDropoutOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("drop_rate", &config_.drop_rate));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
    config_.use_soft_dropout = true;
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& softening_params = context->input(1);

    const int batch_size = input.dim_size(0);
    const int seq_len = input.dim_size(1);
    const int dim = input.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    if (training_) {
      saguaro::qmd::SoftQuantumDropout(
          input.flat<float>().data(),
          output->flat<float>().data(),
          softening_params.flat<float>().data(),
          temperature_,
          config_,
          batch_size, seq_len, dim
      );
    } else {
      std::copy_n(input.flat<float>().data(), batch_size * seq_len * dim,
                  output->flat<float>().data());
    }
  }

 private:
  saguaro::qmd::QMDropoutConfig config_;
  float temperature_;
  bool training_;
};

REGISTER_KERNEL_BUILDER(Name("SoftQuantumDropout").Device(DEVICE_CPU),
                        SoftQuantumDropoutOp);

// =============================================================================
// OP REGISTRATION: SoftQuantumDropoutGrad
// =============================================================================

REGISTER_OP("SoftQuantumDropoutGrad")
    .Input("grad_output: float")          // [batch, seq, dim]
    .Input("input: float")                // [batch, seq, dim]
    .Input("softening_params: float")     // [dim]
    .Output("grad_input: float")          // [batch, seq, dim]
    .Output("grad_params: float")         // [dim]
    .Attr("drop_rate: float = 0.1")
    .Attr("temperature: float = 1.0")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        return Status();
    })
    .Doc(R"doc(
Phase 47: Gradient for Soft Quantum Dropout.

grad_output: Gradient from output [batch, seq, dim]
input: Original input [batch, seq, dim]
softening_params: Learnable parameters [dim]

grad_input: Gradient to input [batch, seq, dim]
grad_params: Gradient to softening params [dim]
)doc");

class SoftQuantumDropoutGradOp : public OpKernel {
 public:
  explicit SoftQuantumDropoutGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("drop_rate", &config_.drop_rate));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
    config_.use_soft_dropout = true;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_output = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& softening_params = context->input(2);

    const int batch_size = input.dim_size(0);
    const int seq_len = input.dim_size(1);
    const int dim = input.dim_size(2);

    Tensor* grad_input = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &grad_input));
    
    Tensor* grad_params = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, softening_params.shape(), &grad_params));

    saguaro::qmd::SoftQuantumDropoutGrad(
        grad_output.flat<float>().data(),
        input.flat<float>().data(),
        softening_params.flat<float>().data(),
        grad_input->flat<float>().data(),
        grad_params->flat<float>().data(),
        temperature_,
        config_,
        batch_size, seq_len, dim
    );
  }

 private:
  saguaro::qmd::QMDropoutConfig config_;
  float temperature_;
};

REGISTER_KERNEL_BUILDER(Name("SoftQuantumDropoutGrad").Device(DEVICE_CPU),
                        SoftQuantumDropoutGradOp);

// =============================================================================
// OP REGISTRATION: EntanglingDropout
// =============================================================================

REGISTER_OP("EntanglingDropout")
    .Input("input: float")                // [batch, seq, dim]
    .Output("output: float")              // [batch, seq, dim]
    .Attr("drop_rate: float = 0.1")
    .Attr("seed: int = 42")
    .Attr("training: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 47: Entangling Gate Dropout.

Randomly skips entangling operations between feature dimensions,
creating varying effective circuit depths.

input: Input after local gates [batch, seq, dim]

output: Output with entangling dropout [batch, seq, dim]
)doc");

class EntanglingDropoutOp : public OpKernel {
 public:
  explicit EntanglingDropoutOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("drop_rate", &config_.drop_rate));
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
    config_.entangling_dropout = true;
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const int batch_size = input.dim_size(0);
    const int seq_len = input.dim_size(1);
    const int dim = input.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));

    if (training_) {
      saguaro::qmd::EntanglingDropout(
          input.flat<float>().data(),
          output->flat<float>().data(),
          nullptr,  // Generate mask internally
          config_,
          batch_size, seq_len, dim
      );
    } else {
      std::copy_n(input.flat<float>().data(), batch_size * seq_len * dim,
                  output->flat<float>().data());
    }
  }

 private:
  saguaro::qmd::QMDropoutConfig config_;
  bool training_;
};

REGISTER_KERNEL_BUILDER(Name("EntanglingDropout").Device(DEVICE_CPU),
                        EntanglingDropoutOp);
