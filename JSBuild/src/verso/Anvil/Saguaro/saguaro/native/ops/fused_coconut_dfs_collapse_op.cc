// saguaro.native/ops/fused_coconut_dfs_collapse_op.cc
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
 * @file fused_coconut_dfs_collapse_op.cc
 * @brief Phase 87: Adaptive BFS→DFS Collapse for CoCoNut.
 *
 * When path confidence exceeds a threshold, collapse from breadth-first
 * exploration to depth-first refinement on the best path. This saves
 * compute when a clear winner emerges early.
 *
 * Also flags high-confidence paths for crystallization (freezing as
 * reusable reasoning primitives).
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedCoconutDFSCollapse")
    .Input("path_states: float32")           // [batch, num_paths, dim]
    .Input("path_amplitudes: float32")        // [batch, num_paths]
    .Output("collapsed_state: float32")       // [batch, dim]
    .Output("should_crystallize: bool")       // [batch] - flags for crystallization
    .Output("best_path_index: int32")         // [batch] - index of selected path
    .Output("confidence: float32")            // [batch] - collapse confidence
    .Attr("collapse_threshold: float = 0.8")
    .Attr("crystallize_threshold: float = 0.9")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle path_shape = c->input(0);
        int64_t batch = c->Value(c->Dim(path_shape, 0));
        int64_t dim = c->Value(c->Dim(path_shape, 2));
        c->set_output(0, c->MakeShape({batch, dim}));     // collapsed_state
        c->set_output(1, c->MakeShape({batch}));          // should_crystallize
        c->set_output(2, c->MakeShape({batch}));          // best_path_index
        c->set_output(3, c->MakeShape({batch}));          // confidence
        return Status();
    })
    .Doc(R"doc(
Phase 87: Adaptive BFS→DFS Collapse.

When the best path's amplitude exceeds collapse_threshold, select that path
and discard others. When amplitude exceeds crystallize_threshold, flag
the path for crystallization as a reusable reasoning primitive.

path_states: Current path states [batch, num_paths, dim]
path_amplitudes: Path quality scores [batch, num_paths] (sum to 1)
collapsed_state: Best path state [batch, dim]
should_crystallize: Whether to freeze this path [batch]
best_path_index: Index of selected path [batch]
confidence: Confidence score for collapse decision [batch]
)doc");

// =============================================================================
// FORWARD KERNEL
// =============================================================================

class FusedCoconutDFSCollapseOp : public OpKernel {
 public:
  explicit FusedCoconutDFSCollapseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("collapse_threshold", &collapse_threshold_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("crystallize_threshold", &crystallize_threshold_));
  }

  void Compute(OpKernelContext* ctx) override {
    // Get inputs
    const Tensor& path_states = ctx->input(0);
    const Tensor& path_amplitudes = ctx->input(1);

    const int64_t batch_size = path_states.dim_size(0);
    const int64_t num_paths = path_states.dim_size(1);
    const int64_t dim = path_states.dim_size(2);

    // Allocate outputs
    Tensor* collapsed_state = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, TensorShape({batch_size, dim}), &collapsed_state));

    Tensor* should_crystallize = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        1, TensorShape({batch_size}), &should_crystallize));

    Tensor* best_path_index = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        2, TensorShape({batch_size}), &best_path_index));

    Tensor* confidence = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        3, TensorShape({batch_size}), &confidence));

    // Get raw pointers
    const float* paths_data = path_states.flat<float>().data();
    const float* amps_data = path_amplitudes.flat<float>().data();
    float* collapsed_data = collapsed_state->flat<float>().data();
    bool* crystallize_data = should_crystallize->flat<bool>().data();
    int32_t* index_data = best_path_index->flat<int32_t>().data();
    float* conf_data = confidence->flat<float>().data();

    // Process each batch element
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
      const float* batch_amps = amps_data + b * num_paths;
      
      // Find best path (highest amplitude)
      int32_t best_idx = 0;
      float best_amp = batch_amps[0];
      for (int64_t p = 1; p < num_paths; ++p) {
        if (batch_amps[p] > best_amp) {
          best_amp = batch_amps[p];
          best_idx = static_cast<int32_t>(p);
        }
      }

      // Store results
      index_data[b] = best_idx;
      conf_data[b] = best_amp;
      crystallize_data[b] = (best_amp >= crystallize_threshold_);

      // Copy best path to collapsed state
      const float* best_path = paths_data + (b * num_paths + best_idx) * dim;
      float* out_row = collapsed_data + b * dim;
      
      // If confidence is high enough for collapse, use only best path
      // Otherwise, use weighted average of all paths
      if (best_amp >= collapse_threshold_) {
        // DFS collapse: use only best path
        std::copy(best_path, best_path + dim, out_row);
      } else {
        // BFS: weighted average of all paths
        std::fill(out_row, out_row + dim, 0.0f);
        for (int64_t p = 0; p < num_paths; ++p) {
          float amp = batch_amps[p];
          const float* path = paths_data + (b * num_paths + p) * dim;
          for (int64_t d = 0; d < dim; ++d) {
            out_row[d] += amp * path[d];
          }
        }
      }
    }
  }

 private:
  float collapse_threshold_;
  float crystallize_threshold_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedCoconutDFSCollapse").Device(DEVICE_CPU),
    FusedCoconutDFSCollapseOp);

}  // namespace tensorflow
