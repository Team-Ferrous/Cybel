// saguaro.native/ops/tt_floquet_decomposition_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// UQHA Priority 3: TensorFlow ops for TT-Floquet Decomposition.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tt_floquet_decomposition.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: TTFloquetForward
// =============================================================================

REGISTER_OP("TTFloquetForward")
    .Input("hd_input: float")            // [batch, seq_len, hd_dim]
    .Input("floquet_energies: float")    // [floquet_modes, hd_dim]
    .Input("drive_weights: float")       // [floquet_modes]
    .Input("coupling_matrix: float")     // [floquet_modes, floquet_modes]
    .Output("hd_output: float")          // [batch, seq_len, hd_dim]
    .Output("compression_ratio: float")  // Scalar
    .Attr("max_tt_rank: int = 8")
    .Attr("drive_frequency: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));  // Same shape as input
        c->set_output(1, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
UQHA Priority 3: TT-Floquet Forward pass.

Memory-efficient Floquet evolution using Tensor-Train decomposition.
Reduces memory from O(modes * hd_dim) to O(modes * r^2) for large hd_dim.

hd_input: Input HD bundles [batch, seq_len, hd_dim]
floquet_energies: Quasi-energies for each mode [floquet_modes, hd_dim]
drive_weights: Drive coupling weights [floquet_modes]
coupling_matrix: DTC mode coupling [floquet_modes, floquet_modes]
max_tt_rank: Maximum TT rank for compression
drive_frequency: Floquet drive frequency

hd_output: Output HD bundles [batch, seq_len, hd_dim]
compression_ratio: Achieved compression ratio (dense/TT memory)
)doc");

class TTFloquetForwardOp : public OpKernel {
 public:
  explicit TTFloquetForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_tt_rank", &max_tt_rank_));
    OP_REQUIRES_OK(context, context->GetAttr("drive_frequency", &drive_frequency_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& hd_input = context->input(0);
    const Tensor& floquet_energies = context->input(1);
    const Tensor& drive_weights = context->input(2);
    const Tensor& coupling_matrix = context->input(3);

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);
    const int hd_dim = hd_input.dim_size(2);
    const int floquet_modes = floquet_energies.dim_size(0);

    // Configure TT-Floquet
    saguaro::tt_floquet::TTFloquetConfig config;
    config.hd_dim = hd_dim;
    config.floquet_modes = floquet_modes;
    config.max_tt_rank = max_tt_rank_;
    config.drive_frequency = drive_frequency_;

    // Allocate output
    Tensor* hd_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, hd_input.shape(), &hd_output));

    // Run TT-Floquet forward
    saguaro::tt_floquet::TTFloquetForward(
        hd_input.flat<float>().data(),
        floquet_energies.flat<float>().data(),
        drive_weights.flat<float>().data(),
        coupling_matrix.flat<float>().data(),
        hd_output->flat<float>().data(),
        config,
        batch_size,
        seq_len
    );

    // Calculate and output compression ratio
    Tensor* ratio_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &ratio_out));
    
    saguaro::tt_floquet::TTFloquetDecomposition tt_temp;
    tt_temp.init(config);
    ratio_out->scalar<float>()() = tt_temp.getCompressionRatio();
  }

 private:
  int max_tt_rank_;
  float drive_frequency_;
};

REGISTER_KERNEL_BUILDER(Name("TTFloquetForward").Device(DEVICE_CPU), TTFloquetForwardOp);

// =============================================================================
// OP REGISTRATION: TTFloquetCompressionStats
// =============================================================================

REGISTER_OP("TTFloquetCompressionStats")
    .Input("hd_dim: int32")
    .Input("floquet_modes: int32")
    .Output("dense_bytes: int64")
    .Output("tt_bytes: int64")
    .Output("compression_ratio: float")
    .Attr("max_tt_rank: int = 8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        c->set_output(2, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Get TT-Floquet compression statistics.

Calculates memory usage for dense vs TT-compressed Floquet coefficients.

hd_dim: HD embedding dimension
floquet_modes: Number of Floquet modes

dense_bytes: Memory for dense storage
tt_bytes: Memory for TT-compressed storage
compression_ratio: dense_bytes / tt_bytes
)doc");

class TTFloquetCompressionStatsOp : public OpKernel {
 public:
  explicit TTFloquetCompressionStatsOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_tt_rank", &max_tt_rank_));
  }

  void Compute(OpKernelContext* context) override {
    int hd_dim = context->input(0).scalar<int>()();
    int floquet_modes = context->input(1).scalar<int>()();

    saguaro::tt_floquet::TTFloquetConfig config;
    config.hd_dim = hd_dim;
    config.floquet_modes = floquet_modes;
    config.max_tt_rank = max_tt_rank_;

    saguaro::tt_floquet::TTFloquetDecomposition tt;
    tt.init(config);

    Tensor* dense_out = nullptr;
    Tensor* tt_out = nullptr;
    Tensor* ratio_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &dense_out));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &tt_out));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}), &ratio_out));

    dense_out->scalar<int64_t>()() = tt.getDenseMemoryUsage();
    tt_out->scalar<int64_t>()() = tt.getMemoryUsage();
    ratio_out->scalar<float>()() = tt.getCompressionRatio();
  }

 private:
  int max_tt_rank_;
};

REGISTER_KERNEL_BUILDER(Name("TTFloquetCompressionStats").Device(DEVICE_CPU), TTFloquetCompressionStatsOp);
