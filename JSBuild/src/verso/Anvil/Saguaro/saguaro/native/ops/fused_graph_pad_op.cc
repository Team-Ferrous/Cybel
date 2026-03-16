// src/ops/fused_graph_pad_op.cc
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
// FusedGraphPad: Fast padding for variable-length molecular graphs
//
// This operation pads variable-length graph node features to a fixed maximum
// length with masking for downstream GNN processing. Essential for batching
// molecules/emitter arrays of different sizes.
//
// Use Case:
//   - Variable number of emitters (N=1 to 16) → fixed array size
//   - Variable molecular graphs (QM9 dataset) → batch processing
//   - Permutation-invariant node features with validity masks
//
// Implementation:
//   - SIMD zero-fill for efficient padding
//   - Validity mask generation for attention masking
//   - Handles jagged tensor batches (different N per batch element)
//
// Phase 11 SIMD Compliance: FULL
//   - SIMD memset for zero-fill (AVX512/AVX2/NEON)
//   - Cache-friendly memory layout
//
// Target Performance: <1ms per batch on 32-core CPU (batch_size=32, max_nodes=128)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

#include "common/parallel/parallel_backend.h"

#include <algorithm>
#include <cstring>
#include <vector>

// Phase 11: SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define GRAPH_PAD_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define GRAPH_PAD_AVX2 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define GRAPH_PAD_NEON 1
#endif

namespace tensorflow {

using shape_inference::InferenceContext;

// =============================================================================
// 1. Op Registration
// =============================================================================

REGISTER_OP("FusedGraphPad")
    .Input("node_features: float")           // [B, max_N, F] with variable N
    .Input("node_counts: int32")             // [B] actual node count per graph
    .Attr("target_length: int")              // Fixed output length (e.g., 128)
    .Output("padded_features: float")        // [B, target_length, F]
    .Output("padding_mask: bool")            // [B, target_length] (true=valid, false=padding)
    .SetShapeFn([](InferenceContext* c) {
        shape_inference::ShapeHandle input_shape = c->input(0);
        shape_inference::DimensionHandle batch_dim = c->Dim(input_shape, 0);
        shape_inference::DimensionHandle feature_dim = c->Dim(input_shape, 2);

        int64 target_length;
        TF_RETURN_IF_ERROR(c->GetAttr("target_length", &target_length));

        // Padded features output
        c->set_output(0, c->MakeShape({
            batch_dim,
            c->MakeDim(target_length),
            feature_dim
        }));

        // Padding mask output
        c->set_output(1, c->MakeShape({
            batch_dim,
            c->MakeDim(target_length)
        }));

        return OkStatus();
    });

// =============================================================================
// 2. CPU Implementation
// =============================================================================

namespace hpc {
namespace cpu {

// =============================================================================
// SIMD Helper: Fast zero-fill (memset alternative)
// =============================================================================

inline void ZeroFill_SIMD(float* data, int64_t size) {
    int64_t i = 0;
#if defined(GRAPH_PAD_AVX512)
    __m512 zero = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        _mm512_storeu_ps(data + i, zero);
    }
#elif defined(GRAPH_PAD_AVX2)
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        _mm256_storeu_ps(data + i, zero);
    }
#elif defined(GRAPH_PAD_NEON)
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        vst1q_f32(data + i, zero);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        data[i] = 0.0f;
    }
}

// =============================================================================
// Graph Padding with Masking
//
// For each batch element:
//   1. Copy actual node features to output
//   2. Zero-fill padding nodes
//   3. Generate validity mask (true for actual nodes, false for padding)
// =============================================================================

inline void GraphPadForwardCpuImpl(
    const float* node_features,
    const int32_t* node_counts,
    float* padded_features,
    bool* padding_mask,
    int64_t batch_size,
    int max_input_nodes,
    int feature_dim,
    int target_length
) {
    // Parallel over batch
    saguaro::parallel::ForRange(
        0, batch_size, 1,
        [&](size_t b_begin, size_t b_end) {
            for (size_t b = b_begin; b < b_end; ++b) {
                int32_t num_nodes = node_counts[b];

                // Clamp to valid range [0, min(max_input_nodes, target_length)]
                num_nodes = std::max(0, std::min(num_nodes,
                    static_cast<int32_t>(std::min(max_input_nodes, target_length))));

                const float* batch_input = node_features + b * max_input_nodes * feature_dim;
                float* batch_output = padded_features + b * target_length * feature_dim;
                bool* batch_mask = padding_mask + b * target_length;

                // Step 1: Copy actual node features
                for (int n = 0; n < num_nodes; ++n) {
                    const float* src = batch_input + n * feature_dim;
                    float* dst = batch_output + n * feature_dim;
                    std::memcpy(dst, src, feature_dim * sizeof(float));

                    // Mark as valid node
                    batch_mask[n] = true;
                }

                // Step 2: Zero-fill padding nodes
                if (num_nodes < target_length) {
                    int padding_count = target_length - num_nodes;
                    float* padding_start = batch_output + num_nodes * feature_dim;

                    ZeroFill_SIMD(padding_start, padding_count * feature_dim);

                    // Mark as padding nodes
                    for (int n = num_nodes; n < target_length; ++n) {
                        batch_mask[n] = false;
                    }
                }

                // Step 3: If input has more nodes than target, truncate
                // (Validity mask already set for [0, target_length) range)
            }
        });
}

} // namespace cpu
} // namespace hpc

// =============================================================================
// 3. TensorFlow Kernel Implementation
// =============================================================================

class FusedGraphPadOp : public OpKernel {
 public:
  explicit FusedGraphPadOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("target_length", &target_length_));
    OP_REQUIRES(context, target_length_ > 0,
                errors::InvalidArgument("target_length must be positive"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& node_features = context->input(0);
    const Tensor& node_counts = context->input(1);

    // Validate shapes
    OP_REQUIRES(context, node_features.dims() == 3,
                errors::InvalidArgument("node_features must be 3D [batch, max_N, F]"));
    OP_REQUIRES(context, node_counts.dims() == 1,
                errors::InvalidArgument("node_counts must be 1D [batch]"));
    OP_REQUIRES(context, node_features.dim_size(0) == node_counts.dim_size(0),
                errors::InvalidArgument("batch dimensions must match"));

    const int64 batch_size = node_features.dim_size(0);
    const int max_input_nodes = static_cast<int>(node_features.dim_size(1));
    const int feature_dim = static_cast<int>(node_features.dim_size(2));

    // Allocate outputs
    Tensor* padded_features = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, target_length_, feature_dim}),
        &padded_features));

    Tensor* padding_mask = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, target_length_}),
        &padding_mask));

    // Get data pointers
    const float* features_data = node_features.flat<float>().data();
    const int32_t* counts_data = node_counts.flat<int32>().data();
    float* padded_data = padded_features->flat<float>().data();
    bool* mask_data = padding_mask->flat<bool>().data();

    // Call CPU implementation
    hpc::cpu::GraphPadForwardCpuImpl(
        features_data,
        counts_data,
        padded_data,
        mask_data,
        batch_size,
        max_input_nodes,
        feature_dim,
        target_length_
    );
  }

 private:
  int64 target_length_;
};

REGISTER_KERNEL_BUILDER(Name("FusedGraphPad").Device(DEVICE_CPU), FusedGraphPadOp);

} // namespace tensorflow
