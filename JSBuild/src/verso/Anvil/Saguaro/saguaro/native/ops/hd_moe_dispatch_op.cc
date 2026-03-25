// saguaro.native/ops/hd_moe_dispatch_op.cc
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
 * @file hd_moe_dispatch_op.cc
 * @brief Phase 200+: HD MoE Dispatch TensorFlow custom operations.
 *
 * SAGUARO_UPGRADE_ROADMAP.md Phase 2.2 - Block-level HD integration.
 *
 * Implements holographic routing for HD-space Mixture-of-Experts:
 *   - Circular correlation-based similarity for expert selection
 *   - O(D) routing via cosine similarity (O(D log D) with FFT future)
 *   - Replaces attention-based collapse with geometric routing
 *
 * This op is used by HDMoEBlock for holographic expert routing in HD space.
 * See HD_SUPERPOSED_EXPERT_UNIFICATION.md for architecture details.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "fused_superposition_moe/holographic_routing.h"
#include "common/edition_limits.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: HDMoEDispatchForward
// =============================================================================

REGISTER_OP("HDMoEDispatchForward")
    .Input("hd_input: float")           // [batch, hd_dim]
    .Input("expert_bases: float")       // [num_experts, hd_dim]
    .Input("expert_weights: float")     // [num_experts, hd_dim]
    .Output("hd_output: float")         // [batch, hd_dim]
    .Output("routing_weights: float")   // [batch, num_experts]
    .Attr("hd_dim: int = 4096")
    .Attr("num_experts: int = 8")
    .Attr("top_k: int = 2")
    .Attr("temperature: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output shape same as input
        c->set_output(0, c->input(0));
        
        // Routing weights: [batch, num_experts]
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
        
        int num_experts;
        TF_RETURN_IF_ERROR(c->GetAttr("num_experts", &num_experts));
        
        c->set_output(1, c->MakeShape({
            c->Dim(input_shape, 0),
            c->MakeDim(num_experts)
        }));
        
        return OkStatus();
    })
    .Doc(R"doc(
HD MoE Dispatch Forward Pass - Holographic routing for expert selection.

Phase 200+: Block-integrated HD streaming. Uses circular correlation
similarity for geometric expert routing in hyperdimensional space.

hd_input: HD bundle input [batch, hd_dim]
expert_bases: Expert signature vectors for routing [num_experts, hd_dim]
expert_weights: Expert transformation weights [num_experts, hd_dim]

hd_output: Routed output [batch, hd_dim]
routing_weights: Expert selection weights [batch, num_experts]
)doc");

// =============================================================================
// KERNEL: HDMoEDispatchForward
// =============================================================================

class HDMoEDispatchForwardOp : public OpKernel {
 public:
  explicit HDMoEDispatchForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &hd_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("num_experts", &num_experts_));
    OP_REQUIRES_OK(context, context->GetAttr("top_k", &top_k_));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
    
    OP_REQUIRES(context, hd_dim_ > 0,
                errors::InvalidArgument("hd_dim must be positive"));
    OP_REQUIRES(context, num_experts_ > 0,
                errors::InvalidArgument("num_experts must be positive"));
    OP_REQUIRES(context, top_k_ > 0 && top_k_ <= num_experts_,
                errors::InvalidArgument("top_k must be in [1, num_experts]"));
  }

  void Compute(OpKernelContext* context) override {
    SAGUARO_SECURITY_HEARTBEAT();
    
    // Get input tensors
    const Tensor& hd_input = context->input(0);
    const Tensor& expert_bases = context->input(1);
    const Tensor& expert_weights = context->input(2);

    // Validate shapes
    OP_REQUIRES(context, hd_input.dims() == 2,
                errors::InvalidArgument("hd_input must be 2D [batch, hd_dim]"));
    OP_REQUIRES(context, expert_bases.dims() == 2,
                errors::InvalidArgument("expert_bases must be 2D [num_experts, hd_dim]"));
    OP_REQUIRES(context, expert_weights.dims() == 2,
                errors::InvalidArgument("expert_weights must be 2D [num_experts, hd_dim]"));

    const int batch_size = hd_input.dim_size(0);
    const int hd_dim = hd_input.dim_size(1);
    const int num_experts = expert_bases.dim_size(0);

    OP_REQUIRES(context, hd_dim == hd_dim_,
                errors::InvalidArgument("hd_dim mismatch"));
    OP_REQUIRES(context, num_experts == num_experts_,
                errors::InvalidArgument("num_experts mismatch"));
    OP_REQUIRES(context, expert_bases.dim_size(1) == hd_dim,
                errors::InvalidArgument("expert_bases dim 1 must equal hd_dim"));
    OP_REQUIRES(context, expert_weights.dim_size(0) == num_experts,
                errors::InvalidArgument("expert_weights dim 0 must equal num_experts"));
    OP_REQUIRES(context, expert_weights.dim_size(1) == hd_dim,
                errors::InvalidArgument("expert_weights dim 1 must equal hd_dim"));

    // Allocate outputs
    Tensor* hd_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, hd_input.shape(), &hd_output));

    Tensor* routing_weights_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, num_experts}), &routing_weights_tensor));

    // Get data pointers
    const float* input_ptr = hd_input.flat<float>().data();
    const float* bases_ptr = expert_bases.flat<float>().data();
    const float* weights_ptr = expert_weights.flat<float>().data();
    float* output_ptr = hd_output->flat<float>().data();
    float* routing_ptr = routing_weights_tensor->flat<float>().data();

    // Configure holographic routing
    saguaro::hd_routing::HolographicRoutingConfig config;
    config.hd_dim = hd_dim;
    config.superposition_dim = num_experts;  // Reuse routing for experts
    config.temperature = temperature_;

    // Process each batch element
    for (int b = 0; b < batch_size; ++b) {
      const float* x_b = input_ptr + b * hd_dim;
      float* y_b = output_ptr + b * hd_dim;
      float* rw_b = routing_ptr + b * num_experts;

      // Compute holographic similarity for each expert
      std::vector<float> scores(num_experts);
      for (int e = 0; e < num_experts; ++e) {
        const float* base_e = bases_ptr + e * hd_dim;
        scores[e] = saguaro::hd_routing::holographic_similarity(
            x_b, base_e, hd_dim);
      }

      // Apply softmax to get routing weights
      saguaro::hd_routing::softmax_path_scores(
          scores.data(), rw_b, num_experts, temperature_);

      // Apply top-k sparsification if needed
      if (top_k_ < num_experts) {
        // Find top-k indices
        std::vector<std::pair<float, int>> sorted_scores(num_experts);
        for (int e = 0; e < num_experts; ++e) {
          sorted_scores[e] = {rw_b[e], e};
        }
        std::sort(sorted_scores.begin(), sorted_scores.end(),
                  [](const auto& a, const auto& b) { 
                    return a.first > b.first; 
                  });

        // Zero out non-top-k and renormalize
        float sum_topk = 0.0f;
        for (int i = 0; i < top_k_; ++i) {
          sum_topk += sorted_scores[i].first;
        }
        for (int i = top_k_; i < num_experts; ++i) {
          rw_b[sorted_scores[i].second] = 0.0f;
        }
        if (sum_topk > 1e-8f) {
          float inv_sum = 1.0f / sum_topk;
          for (int i = 0; i < top_k_; ++i) {
            rw_b[sorted_scores[i].second] *= inv_sum;
          }
        }
      }

      // Compute weighted combination with HD binding
      std::fill(y_b, y_b + hd_dim, 0.0f);
      std::vector<float> bound_temp(hd_dim);
      
      for (int e = 0; e < num_experts; ++e) {
        if (rw_b[e] < 1e-8f) continue;  // Skip zero-weight experts
        
        const float* weight_e = weights_ptr + e * hd_dim;
        
        // HD binding: x ⊗ expert_weight
        saguaro::hd_routing::simd_hadamard_product(
            bound_temp.data(), x_b, weight_e, hd_dim);
        
        // Weighted accumulation
        saguaro::hd_routing::simd_weighted_accumulate(
            y_b, bound_temp.data(), rw_b[e], hd_dim);
      }
    }
  }

 private:
  int hd_dim_;
  int num_experts_;
  int top_k_;
  float temperature_;
};

REGISTER_KERNEL_BUILDER(Name("HDMoEDispatchForward").Device(DEVICE_CPU),
                        HDMoEDispatchForwardOp);

// =============================================================================
// OP REGISTRATION: HDMoEDispatchBackward
// =============================================================================

REGISTER_OP("HDMoEDispatchBackward")
    .Input("grad_output: float")        // [batch, hd_dim]
    .Input("hd_input: float")           // [batch, hd_dim]
    .Input("expert_bases: float")       // [num_experts, hd_dim]
    .Input("expert_weights: float")     // [num_experts, hd_dim]
    .Input("routing_weights: float")    // [batch, num_experts]
    .Output("grad_input: float")        // [batch, hd_dim]
    .Output("grad_bases: float")        // [num_experts, hd_dim]
    .Output("grad_weights: float")      // [num_experts, hd_dim]
    .Attr("hd_dim: int = 4096")
    .Attr("num_experts: int = 8")
    .Attr("temperature: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_input same as hd_input
        c->set_output(1, c->input(2));  // grad_bases same as expert_bases
        c->set_output(2, c->input(3));  // grad_weights same as expert_weights
        return OkStatus();
    })
    .Doc("HD MoE Dispatch Backward Pass - Gradient computation for holographic routing.");

// =============================================================================
// KERNEL: HDMoEDispatchBackward
// =============================================================================

class HDMoEDispatchBackwardOp : public OpKernel {
 public:
  explicit HDMoEDispatchBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &hd_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("num_experts", &num_experts_));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& grad_output = context->input(0);
    const Tensor& hd_input = context->input(1);
    const Tensor& expert_bases = context->input(2);
    const Tensor& expert_weights = context->input(3);
    const Tensor& routing_weights = context->input(4);

    const int batch_size = hd_input.dim_size(0);
    const int hd_dim = hd_input.dim_size(1);
    const int num_experts = expert_bases.dim_size(0);

    // Allocate gradient tensors
    Tensor* grad_input = nullptr;
    Tensor* grad_bases = nullptr;
    Tensor* grad_weights = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(
        0, hd_input.shape(), &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, expert_bases.shape(), &grad_bases));
    OP_REQUIRES_OK(context, context->allocate_output(
        2, expert_weights.shape(), &grad_weights));

    // Get data pointers
    const float* grad_out_ptr = grad_output.flat<float>().data();
    const float* input_ptr = hd_input.flat<float>().data();
    const float* bases_ptr = expert_bases.flat<float>().data();
    const float* weights_ptr = expert_weights.flat<float>().data();
    const float* routing_ptr = routing_weights.flat<float>().data();
    float* grad_in_ptr = grad_input->flat<float>().data();
    float* grad_bases_ptr = grad_bases->flat<float>().data();
    float* grad_weights_ptr = grad_weights->flat<float>().data();

    // Zero initialize gradients
    std::fill(grad_in_ptr, grad_in_ptr + batch_size * hd_dim, 0.0f);
    std::fill(grad_bases_ptr, grad_bases_ptr + num_experts * hd_dim, 0.0f);
    std::fill(grad_weights_ptr, grad_weights_ptr + num_experts * hd_dim, 0.0f);

    // Compute gradients for each batch element
    for (int b = 0; b < batch_size; ++b) {
      const float* grad_y = grad_out_ptr + b * hd_dim;
      const float* x_b = input_ptr + b * hd_dim;
      const float* rw_b = routing_ptr + b * num_experts;
      float* grad_x = grad_in_ptr + b * hd_dim;

      for (int e = 0; e < num_experts; ++e) {
        if (rw_b[e] < 1e-8f) continue;
        
        const float* weight_e = weights_ptr + e * hd_dim;
        float* grad_w_e = grad_weights_ptr + e * hd_dim;
        float w_e = rw_b[e];

        for (int d = 0; d < hd_dim; ++d) {
          float grad_bound = grad_y[d] * w_e;
          
          // d(x ⊗ weight)/d(x) = weight
          grad_x[d] += grad_bound * weight_e[d];
          
          // d(x ⊗ weight)/d(weight) = x
          grad_w_e[d] += grad_bound * x_b[d];
        }
      }
    }

    // Gradient through routing is approximated (straight-through for bases)
    // Full implementation would require Jacobian of softmax × holographic similarity
  }

 private:
  int hd_dim_;
  int num_experts_;
  float temperature_;
};

REGISTER_KERNEL_BUILDER(Name("HDMoEDispatchBackward").Device(DEVICE_CPU),
                        HDMoEDispatchBackwardOp);

// =============================================================================
// NOTE: HolographicSimilarity op is defined in hd_holographic_similarity_op.cc
// to avoid duplicate registration when compiled into consolidated binary.
// =============================================================================
