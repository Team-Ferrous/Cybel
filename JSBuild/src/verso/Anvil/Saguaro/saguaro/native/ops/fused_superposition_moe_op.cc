// saguaro.native/ops/fused_superposition_moe_op.cc
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
//
// ============================================================================
// UNIFIED HD-SUPERPOSED EXPERT OPERATOR (v2.0)
//
// This replaces the legacy FusedSuperpositionMoe with holographic routing.
// Breaking change: Q/K/V/O collapse weights replaced with path_bases/path_weights.
// ============================================================================

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/edition_limits.h"
#include "fused_superposition_moe/forward_kernel.h"
#include "fused_superposition_moe/backward_kernel.h"

namespace tensorflow {

// =============================================================================
// UNIFIED HD-SUPERPOSED EXPERT OP REGISTRATION (v2.0)
// =============================================================================

REGISTER_OP("FusedSuperpositionMoe")
    .Input("tokens: float")              // Shape: [B, d_model]
    .Input("ffn1_cores: float")          // Shape: [total_core_elements] (flattened TT cores)
    .Input("ffn2_cores: float")          // Shape: [total_core_elements]
    .Input("path_bases: float")          // Shape: [K, d_model] - Holographic routing bases
    .Input("path_weights: float")        // Shape: [K, d_model] - Transformation weights
    .Input("hd_input_proj: float")       // Shape: [d_model, hd_dim] or empty if no projection
    .Input("hd_output_proj: float")      // Shape: [hd_dim, d_model] or empty if no projection
    .Attr("input_dims: list(int)")
    .Attr("output_dims_ffn1: list(int)")
    .Attr("output_dims_ffn2: list(int)")
    .Attr("tt_ranks: list(int)")
    .Attr("superposition_dim: int")
    .Attr("micro_batch_size: int = 32")
    .Attr("hd_dim: int = 4096")
    .Attr("use_hd_projection: bool = false")
    .Attr("routing_temperature: float = 1.0")
    .Output("output: float")             // Shape: [B, d_model]
    .Output("routing_weights: float")    // Shape: [B, K] - For visualization/debugging
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output 0: same as input tokens
        c->set_output(0, c->input(0));
        
        // Output 1: routing weights [batch, K]
        int superposition_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("superposition_dim", &superposition_dim));
        shape_inference::DimensionHandle batch_dim = c->Dim(c->input(0), 0);
        c->set_output(1, c->MakeShape({batch_dim, superposition_dim}));
        
        return OkStatus();
    });

REGISTER_OP("FusedSuperpositionMoeGrad")
    .Input("grad_output: float")         // Shape: [B, d_model]
    .Input("tokens: float")              // Shape: [B, d_model]
    .Input("ffn1_cores: float")          // Shape: [total_core_elements]
    .Input("ffn2_cores: float")          // Shape: [total_core_elements]
    .Input("path_bases: float")          // Shape: [K, d_model]
    .Input("path_weights: float")        // Shape: [K, d_model]
    .Input("hd_input_proj: float")       // Shape: [d_model, hd_dim]
    .Input("hd_output_proj: float")      // Shape: [hd_dim, d_model]
    .Input("routing_weights: float")     // Shape: [B, K] - Cached from forward
    .Attr("input_dims: list(int)")
    .Attr("output_dims_ffn1: list(int)")
    .Attr("output_dims_ffn2: list(int)")
    .Attr("tt_ranks: list(int)")
    .Attr("superposition_dim: int")
    .Attr("micro_batch_size: int = 32")
    .Attr("hd_dim: int = 4096")
    .Attr("use_hd_projection: bool = false")
    .Attr("routing_temperature: float = 1.0")
    .Output("grad_tokens: float")        // Shape: [B, d_model]
    .Output("grad_ffn1_cores: float")    // Shape: [total_core_elements]
    .Output("grad_ffn2_cores: float")    // Shape: [total_core_elements]
    .Output("grad_path_bases: float")    // Shape: [K, d_model]
    .Output("grad_path_weights: float")  // Shape: [K, d_model]
    .Output("grad_hd_input_proj: float") // Shape: [d_model, hd_dim]
    .Output("grad_hd_output_proj: float")// Shape: [hd_dim, d_model]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));   // grad_tokens same as tokens
        c->set_output(1, c->input(2));   // grad_ffn1_cores same as ffn1_cores
        c->set_output(2, c->input(3));   // grad_ffn2_cores same as ffn2_cores
        c->set_output(3, c->input(4));   // grad_path_bases same as path_bases
        c->set_output(4, c->input(5));   // grad_path_weights same as path_weights
        c->set_output(5, c->input(6));   // grad_hd_input_proj same as hd_input_proj
        c->set_output(6, c->input(7));   // grad_hd_output_proj same as hd_output_proj
        return OkStatus();
    });

// =============================================================================
// KERNEL REGISTRATION
// =============================================================================

REGISTER_KERNEL_BUILDER(Name("FusedSuperpositionMoe").Device(DEVICE_CPU), FusedSuperpositionMoeOpCpu);
REGISTER_KERNEL_BUILDER(Name("FusedSuperpositionMoeGrad").Device(DEVICE_CPU), FusedSuperpositionMoeGradOpCpu);

} // namespace tensorflow