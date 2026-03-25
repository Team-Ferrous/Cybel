// saguaro.native/ops/fused_coconut_crystallize_op.cc
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
 * @file fused_coconut_crystallize_op.cc
 * @brief Phase 87: Thought Crystallization for CoCoNut.
 *
 * Freezes high-confidence reasoning paths as reusable primitives.
 * Integrates with existing quantum crystallization infrastructure
 * from Phase 65/83.
 *
 * Crystal store uses a fixed-capacity ring buffer with LRU eviction
 * when max_crystals is reached.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "quantum_crystallization_op.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// OP REGISTRATION: Crystallize Thought Path
// =============================================================================

REGISTER_OP("FusedCoconutCrystallize")
    .Input("thought_path: float32")          // [batch, dim]
    .Input("confidence: float32")             // [batch]
    .Input("crystal_store: float32")         // [max_crystals, dim]
    .Input("crystal_ages: int32")            // [max_crystals] - for LRU eviction
    .Output("updated_store: float32")        // [max_crystals, dim]
    .Output("updated_ages: int32")           // [max_crystals]
    .Output("crystal_indices: int32")        // [batch] - where each was stored (-1 if not)
    .Attr("crystallize_threshold: float = 0.9")
    .Attr("max_crystals: int = 64")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(2));   // updated_store same shape
        c->set_output(1, c->input(3));   // updated_ages same shape
        c->set_output(2, c->MakeShape({c->Dim(c->input(0), 0)})); // crystal_indices
        return Status();
    })
    .Doc(R"doc(
Phase 87: Crystallize high-confidence thought paths.

Stores thought paths that exceed crystallize_threshold into a persistent
crystal store. Uses LRU eviction when store is full.

thought_path: Thought path to potentially crystallize [batch, dim]
confidence: Confidence score for each path [batch]
crystal_store: Existing crystal store [max_crystals, dim]
crystal_ages: Age counter for LRU eviction [max_crystals]
updated_store: Updated crystal store [max_crystals, dim]
updated_ages: Updated age counters [max_crystals]
crystal_indices: Index where each path was stored, -1 if not crystallized [batch]
)doc");

// =============================================================================
// OP REGISTRATION: Retrieve From Crystals
// =============================================================================

REGISTER_OP("FusedCoconutRetrieve")
    .Input("query: float32")                 // [batch, dim]
    .Input("crystal_store: float32")         // [max_crystals, dim]
    .Input("crystal_valid: bool")            // [max_crystals] - which slots are valid
    .Output("retrieved: float32")            // [batch, dim]
    .Output("similarity: float32")           // [batch] - best match similarity
    .Attr("top_k: int = 1")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));  // retrieved same shape as query
        c->set_output(1, c->MakeShape({c->Dim(c->input(0), 0)})); // similarity
        return Status();
    })
    .Doc(R"doc(
Phase 87: Retrieve crystallized reasoning paths.

Finds the most similar crystallized path to the query and returns it.
Uses cosine similarity for matching.

query: Query embedding to match [batch, dim]
crystal_store: Crystal store [max_crystals, dim]
crystal_valid: Which crystal slots are valid [max_crystals]
retrieved: Best matching crystal [batch, dim]
similarity: Cosine similarity of match [batch]
)doc");

// =============================================================================
// CRYSTALLIZE KERNEL
// =============================================================================

class FusedCoconutCrystallizeOp : public OpKernel {
 public:
  explicit FusedCoconutCrystallizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("crystallize_threshold", &threshold_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_crystals", &max_crystals_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& thought_path = ctx->input(0);
    const Tensor& confidence = ctx->input(1);
    const Tensor& crystal_store = ctx->input(2);
    const Tensor& crystal_ages = ctx->input(3);

    const int64_t batch_size = thought_path.dim_size(0);
    const int64_t dim = thought_path.dim_size(1);

    // Allocate outputs (copy inputs first, then modify)
    Tensor* updated_store = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, crystal_store.shape(), &updated_store));
    std::copy(crystal_store.flat<float>().data(),
              crystal_store.flat<float>().data() + crystal_store.NumElements(),
              updated_store->flat<float>().data());

    Tensor* updated_ages = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, crystal_ages.shape(), &updated_ages));
    std::copy(crystal_ages.flat<int32>().data(),
              crystal_ages.flat<int32>().data() + crystal_ages.NumElements(),
              updated_ages->flat<int32>().data());

    Tensor* crystal_indices = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        2, TensorShape({batch_size}), &crystal_indices));

    const float* path_data = thought_path.flat<float>().data();
    const float* conf_data = confidence.flat<float>().data();
    float* store_data = updated_store->flat<float>().data();
    int32_t* ages_data = updated_ages->flat<int32>().data();
    int32_t* indices_data = crystal_indices->flat<int32>().data();

    // Increment all ages
    for (int64_t i = 0; i < max_crystals_; ++i) {
      ages_data[i]++;
    }

    // Process each batch element
    for (int64_t b = 0; b < batch_size; ++b) {
      if (conf_data[b] < threshold_) {
        indices_data[b] = -1;  // Not crystallized
        continue;
      }

      // Find oldest slot (LRU eviction)
      int32_t oldest_idx = 0;
      int32_t oldest_age = ages_data[0];
      for (int64_t i = 1; i < max_crystals_; ++i) {
        if (ages_data[i] > oldest_age) {
          oldest_age = ages_data[i];
          oldest_idx = static_cast<int32_t>(i);
        }
      }

      // Store path in oldest slot
      const float* src = path_data + b * dim;
      float* dst = store_data + oldest_idx * dim;
      std::copy(src, src + dim, dst);
      ages_data[oldest_idx] = 0;  // Reset age
      indices_data[b] = oldest_idx;
    }
  }

 private:
  float threshold_;
  int64_t max_crystals_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedCoconutCrystallize").Device(DEVICE_CPU),
    FusedCoconutCrystallizeOp);

// =============================================================================
// RETRIEVE KERNEL
// =============================================================================

class FusedCoconutRetrieveOp : public OpKernel {
 public:
  explicit FusedCoconutRetrieveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_k", &top_k_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& query = ctx->input(0);
    const Tensor& crystal_store = ctx->input(1);
    const Tensor& crystal_valid = ctx->input(2);

    const int64_t batch_size = query.dim_size(0);
    const int64_t dim = query.dim_size(1);
    const int64_t num_crystals = crystal_store.dim_size(0);

    Tensor* retrieved = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &retrieved));

    Tensor* similarity = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        1, TensorShape({batch_size}), &similarity));

    const float* q_data = query.flat<float>().data();
    const float* store_data = crystal_store.flat<float>().data();
    const bool* valid_data = crystal_valid.flat<bool>().data();
    float* ret_data = retrieved->flat<float>().data();
    float* sim_data = similarity->flat<float>().data();

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
      const float* qv = q_data + b * dim;
      
      // Compute query norm
      float q_norm_sq = 0.0f;
      for (int64_t d = 0; d < dim; ++d) {
        q_norm_sq += qv[d] * qv[d];
      }
      float q_norm = std::sqrt(q_norm_sq + 1e-8f);

      // Find best matching crystal
      int64_t best_idx = -1;
      float best_sim = -1.0f;

      for (int64_t c = 0; c < num_crystals; ++c) {
        if (!valid_data[c]) continue;

        const float* cv = store_data + c * dim;
        float dot = 0.0f;
        float c_norm_sq = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
          dot += qv[d] * cv[d];
          c_norm_sq += cv[d] * cv[d];
        }
        float c_norm = std::sqrt(c_norm_sq + 1e-8f);
        float sim = dot / (q_norm * c_norm);

        if (sim > best_sim) {
          best_sim = sim;
          best_idx = c;
        }
      }

      sim_data[b] = best_sim;
      float* out = ret_data + b * dim;

      if (best_idx >= 0) {
        const float* best = store_data + best_idx * dim;
        std::copy(best, best + dim, out);
      } else {
        // No valid crystals, return zeros
        std::fill(out, out + dim, 0.0f);
      }
    }
  }

 private:
  int64_t top_k_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedCoconutRetrieve").Device(DEVICE_CPU),
    FusedCoconutRetrieveOp);

}  // namespace tensorflow
