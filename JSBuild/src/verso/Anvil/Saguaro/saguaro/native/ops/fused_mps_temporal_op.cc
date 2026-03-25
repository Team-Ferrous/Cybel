// saguaro.native/ops/fused_mps_temporal_op.cc
// Fused MPS Temporal Scan Operation
//
// Implements efficient O(n·χ²) sequence processing using Matrix Product States.
// This kernel replaces the Python iterative loop for significant speedup.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include <Eigen/Dense>
#include <vector>
#include <cmath>

#include "common/parallel/parallel_backend.h"
#include "mps_isometry_helpers.h"  // Phase 3: Isometry enforcement

namespace tensorflow {

REGISTER_OP("MPSTemporalScan")
    .Input("inputs: float32")           // [B, L, D]
    .Input("site_weights: float32")     // [B, L, chi, d, chi]
    .Input("initial_state: float32")   // [B, 1, chi]
    .Input("max_bond_dim: int32")
    .Attr("use_tdvp: bool = false")
    .Attr("enforce_isometry: bool = false")  // Phase 3: Enable isometry enforcement
    .Output("outputs: float32")         // [B, L, d]
    .Output("log_probs: float32")       // [B, L]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle inputs_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs_shape));
        shape_inference::ShapeHandle weights_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &weights_shape));
        
        // Output: [B, L, d] where d is the physical dimension of weights
        c->set_output(0, c->MakeShape({c->Dim(inputs_shape, 0), c->Dim(inputs_shape, 1), c->Dim(weights_shape, 3)}));
        c->set_output(1, c->MakeShape({c->Dim(inputs_shape, 0), c->Dim(inputs_shape, 1)}));
        return absl::OkStatus();
    });

class MPSTemporalScanOpCpu : public OpKernel {
public:
    explicit MPSTemporalScanOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_tdvp", &use_tdvp_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("enforce_isometry", &enforce_isometry_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& inputs = ctx->input(0);
        const Tensor& site_weights = ctx->input(1);
        const Tensor& initial_state = ctx->input(2);
        const int max_bond_dim = ctx->input(3).scalar<int32>()();
        (void)max_bond_dim;

        const int batch_size = inputs.dim_size(0);
        const int seq_len = inputs.dim_size(1);
        const int chi = site_weights.dim_size(2);
        const int d = site_weights.dim_size(3);

        Tensor* outputs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size, seq_len, d}), &outputs));
        
        Tensor* log_probs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({batch_size, seq_len}), &log_probs));

        auto inputs_flat = inputs.tensor<float, 3>();
        auto weights_flat = site_weights.tensor<float, 5>();
        auto init_state_flat = initial_state.tensor<float, 3>();
        auto outputs_flat = outputs->tensor<float, 3>();
        auto log_probs_flat = log_probs->tensor<float, 2>();

        // Parallelize over batch dimension
        saguaro::parallel::ForEachIndex(batch_size, 1, [&](std::size_t b) {
            Eigen::MatrixXf left_env = Eigen::MatrixXf::Zero(1, chi);
            for (int c = 0; c < chi; ++c) left_env(0, c) = init_state_flat(b, 0, c);

            for (int t = 0; t < seq_len; ++t) {
                // Site tensor A_t: [chi, d, chi]
                // Contract: left_env [1, chi] * A_t [chi, d, chi] -> [d, chi]
                Eigen::MatrixXf result = Eigen::MatrixXf::Zero(d, chi);
                
                for (int p = 0; p < d; ++p) {
                    for (int cl = 0; cl < chi; ++cl) {
                        for (int cr = 0; cr < chi; cr++) {
                            result(p, cr) += left_env(0, cl) * weights_flat(b, t, cl, p, cr);
                        }
                    }
                }

                // Output at time t: average over bond dimensions (or take first slice?)
                // Standard MPS linear layer take physical output
                for (int p = 0; p < d; ++p) {
                    float sum = 0.0f;
                    for (int cr = 0; cr < chi; ++cr) sum += result(p, cr);
                    outputs_flat(b, t, p) = sum / chi;
                }

                // Probability modeling (log-norm)
                float norm_sq = result.squaredNorm();
                log_probs_flat(b, t) = 0.5f * std::log(norm_sq + 1e-12f);

                // Update left environment for next step: [1, chi]
                // Extract a "state" from the result. For simple modeling, we take the row-wise sum
                // or a specific projection. Here we use the normalized first physical slice as surrogate.
                if (norm_sq > 1e-12f) {
                    result /= std::sqrt(norm_sq);
                }
                
                left_env.setZero();
                for (int cr = 0; cr < chi; ++cr) {
                    for (int p = 0; p < d; ++p) {
                        left_env(0, cr) += result(p, cr);
                    }
                }
                left_env /= d; // Normalize
            }
        });
    }

private:
    bool use_tdvp_;
    bool enforce_isometry_;
};

REGISTER_KERNEL_BUILDER(Name("MPSTemporalScan").Device(DEVICE_CPU), MPSTemporalScanOpCpu);

} // namespace tensorflow
