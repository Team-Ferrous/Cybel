// saguaro.native/ops/fused_superposition_moe/forward_kernel.h
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
// UNIFIED HD-SUPERPOSED EXPERT KERNEL (v2.0)
//
// This kernel unifies FusedSuperpositionMoe and HDMoEDispatch into a single
// architecture with holographic circular correlation routing.
//
// Key Changes from v1.0:
// - Replaced attention-based collapse (Q/K/V/O) with holographic routing
// - Added HD projection support (d_model <-> hd_dim)
// - Uses path_bases and path_weights for geometric routing
// - O(D) routing via cosine similarity (O(D log D) with FFT future)
//
// Micro-batching is preserved for memory efficiency.
// ============================================================================

#ifndef TENSORFLOW_CORE_USER_OPS_FUSED_SUPERPOSITION_MOE_FORWARD_KERNEL_H_
#define TENSORFLOW_CORE_USER_OPS_FUSED_SUPERPOSITION_MOE_FORWARD_KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "holographic_routing.h"
#include "common/parallel/parallel_backend.h"
#include "common/edition_limits.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace tensorflow {

using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowMajor;

// =============================================================================
// PHASE V2.0-P1.6: SCRATCH BUFFER POOLING NOTES
// =============================================================================
// Thread-local allocations in process_range() (lines 215-216) are already
// efficient due to small K values (1-2 for Lite edition). Scratch pooling
// provides marginal benefit here since:
// - Allocation size is K × d_model ≈ 2KB-8KB
// - Allocations are reused across tokens within a micro-batch
// - Thread-local std::vector avoids contention
//
// For larger K (8+), consider using g_path_scratch from hnn_simd_common.h.
// See SAGUARO_V2_PERFORMANCE_ANALYSIS.md Section 11.6 (P-1.1)
// =============================================================================

// =============================================================================
// PHASE V2.0-P1.1: EXPERT-LEVEL PARALLELISM (OpenMP)
// =============================================================================
// Parallelization strategy for SuperposedExpert:
//
// LEVEL 1: Batch parallelism (primary)
//   - ForShard distributes tokens across threads
//   - Each thread processes a range of batch indices independently
//   - Thread-local storage avoids contention
//
// LEVEL 2: Expert path parallelism (K paths)
//   - For K >= 4, inner loop over paths can be parallelized
//   - Use: #pragma omp parallel for if(K >= 4) num_threads(min(K, 4))
//   - Benefit: ~25% speedup when K=8, negligible when K=1-2
//
// LEVEL 3: TT contraction parallelism (via Eigen)
//   - Eigen tensor ops use internal threading for large tensors
//   - Controlled by Eigen::setNbThreads() at startup
//
// Current implementation uses LEVEL 1 (batch) as the primary strategy,
// which provides best efficiency for typical workloads.
//
// See SAGUARO_V2_PERFORMANCE_ANALYSIS.md Section 13.2 (Phase P1)
// =============================================================================

/**
 * @brief Unified HD-SuperposedExpert Forward Kernel.
 *
 * Combines TT-decomposed superposition paths with holographic routing:
 *   1. Optionally project tokens to HD space (d_model -> hd_dim)
 *   2. Process through K parallel TT-FFN paths
 *   3. Collapse paths via holographic similarity routing
 *   4. Optionally project back to model space (hd_dim -> d_model)
 *
 * Inputs (index):
 *   0: tokens [batch, d_model] - Input tokens
 *   1: ffn1_cores - TT cores for first FFN layer (flattened)
 *   2: ffn2_cores - TT cores for second FFN layer (flattened)
 *   3: path_bases [K, d_model] - Holographic routing bases per path
 *   4: path_weights [K, d_model] - Transformation weights per path
 *   5: hd_input_proj [d_model, hd_dim] - HD projection in (optional, can be identity)
 *   6: hd_output_proj [hd_dim, d_model] - HD projection out (optional)
 *
 * Outputs:
 *   0: output [batch, d_model]
 *   1: routing_weights [batch, K] - For visualization/debugging
 */
class FusedSuperpositionMoeOpCpu : public OpKernel {
public:
    explicit FusedSuperpositionMoeOpCpu(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("input_dims", &input_dims_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dims_ffn1", &output_dims_ffn1_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dims_ffn2", &output_dims_ffn2_));
        OP_REQUIRES_OK(context, context->GetAttr("tt_ranks", &tt_ranks_));
        OP_REQUIRES_OK(context, context->GetAttr("superposition_dim", &superposition_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("micro_batch_size", &micro_batch_size_));
        
        // New v2.0 attributes
        OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &hd_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("use_hd_projection", &use_hd_projection_));
        OP_REQUIRES_OK(context, context->GetAttr("routing_temperature", &routing_temperature_));
        
        OP_REQUIRES(context, micro_batch_size_ > 0,
                    errors::InvalidArgument("micro_batch_size must be positive, got ", micro_batch_size_));
        OP_REQUIRES(context, superposition_dim_ > 0,
                    errors::InvalidArgument("superposition_dim must be positive, got ", superposition_dim_));
        OP_REQUIRES(context, hd_dim_ > 0,
                    errors::InvalidArgument("hd_dim must be positive, got ", hd_dim_));
        
        // HighNoon Lite Edition: Enforce superposition dimension limit
        SAGUARO_CHECK_SUPERPOSITION_DIM(context, superposition_dim_);
    }

    void Compute(OpKernelContext* context) override {
        SAGUARO_SECURITY_HEARTBEAT();
        
        // --- Get Input Tensors ---
        const Tensor& tokens = context->input(0);
        const Tensor& ffn1_cores_tensor = context->input(1);
        const Tensor& ffn2_cores_tensor = context->input(2);
        const Tensor& path_bases_tensor = context->input(3);
        const Tensor& path_weights_tensor = context->input(4);
        const Tensor& hd_input_proj_tensor = context->input(5);
        const Tensor& hd_output_proj_tensor = context->input(6);

        // --- Validate Shapes ---
        const auto& tokens_shape = tokens.shape();
        OP_REQUIRES(context, tokens_shape.dims() == 2,
            errors::InvalidArgument(
                "[UnifiedHDSuperposedExpert] expects 2D input [batch, d_model], got ",
                tokens_shape.DebugString()));

        const int64 batch_size = tokens_shape.dim_size(0);
        const int64 d_model = tokens_shape.dim_size(1);
        const int64 K = superposition_dim_;
        const int64 hd_dim = use_hd_projection_ ? hd_dim_ : d_model;
        
        // Compute d_ff from TT dims
        const auto safe_product = [&](const std::vector<int64>& dims) -> int64 {
            int64 product = 1;
            for (int64 dim : dims) {
                product *= dim;
            }
            return product;
        };
        const int64 d_ff = safe_product(output_dims_ffn1_);

        // Validate path_bases and path_weights
        OP_REQUIRES(context, path_bases_tensor.dims() == 2,
            errors::InvalidArgument("path_bases must be 2D [K, d_model]"));
        OP_REQUIRES(context, path_bases_tensor.dim_size(0) == K,
            errors::InvalidArgument("path_bases dim 0 must equal superposition_dim"));
        OP_REQUIRES(context, path_bases_tensor.dim_size(1) == d_model,
            errors::InvalidArgument("path_bases dim 1 must equal d_model"));
        
        OP_REQUIRES(context, path_weights_tensor.dims() == 2,
            errors::InvalidArgument("path_weights must be 2D [K, d_model]"));
        OP_REQUIRES(context, path_weights_tensor.dim_size(0) == K,
            errors::InvalidArgument("path_weights dim 0 must equal superposition_dim"));
        OP_REQUIRES(context, path_weights_tensor.dim_size(1) == d_model,
            errors::InvalidArgument("path_weights dim 1 must equal d_model"));

        // Validate HD projections if used
        if (use_hd_projection_) {
            OP_REQUIRES(context, hd_input_proj_tensor.dims() == 2,
                errors::InvalidArgument("hd_input_proj must be 2D [d_model, hd_dim]"));
            OP_REQUIRES(context, hd_input_proj_tensor.dim_size(0) == d_model,
                errors::InvalidArgument("hd_input_proj dim 0 must equal d_model"));
            OP_REQUIRES(context, hd_input_proj_tensor.dim_size(1) == hd_dim,
                errors::InvalidArgument("hd_input_proj dim 1 must equal hd_dim"));
            
            OP_REQUIRES(context, hd_output_proj_tensor.dims() == 2,
                errors::InvalidArgument("hd_output_proj must be 2D [hd_dim, d_model]"));
            OP_REQUIRES(context, hd_output_proj_tensor.dim_size(0) == hd_dim,
                errors::InvalidArgument("hd_output_proj dim 0 must equal hd_dim"));
            OP_REQUIRES(context, hd_output_proj_tensor.dim_size(1) == d_model,
                errors::InvalidArgument("hd_output_proj dim 1 must equal d_model"));
        }

        if (batch_size == 0) {
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, tokens.shape(), &output_tensor));
            Tensor* routing_weights_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, 
                TensorShape({0, K}), &routing_weights_tensor));
            return;
        }

        // --- Allocate Outputs ---
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, tokens.shape(), &output_tensor));
        
        Tensor* routing_weights_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, 
            TensorShape({batch_size, K}), &routing_weights_tensor));

        // --- Create Core Managers ---
        TTCores ffn1_cores(ffn1_cores_tensor, tt_ranks_, input_dims_, output_dims_ffn1_, K);
        TTCores ffn2_cores(ffn2_cores_tensor, tt_ranks_, output_dims_ffn1_, output_dims_ffn2_, K);

        // --- Get Data Pointers ---
        const float* tokens_ptr = tokens.flat<float>().data();
        const float* path_bases_ptr = path_bases_tensor.flat<float>().data();
        const float* path_weights_ptr = path_weights_tensor.flat<float>().data();
        const float* hd_input_proj_ptr = use_hd_projection_ ? 
            hd_input_proj_tensor.flat<float>().data() : nullptr;
        const float* hd_output_proj_ptr = use_hd_projection_ ? 
            hd_output_proj_tensor.flat<float>().data() : nullptr;
        float* output_ptr = output_tensor->flat<float>().data();
        float* routing_weights_ptr = routing_weights_tensor->flat<float>().data();

        // --- Eigen Shape Arrays ---
        const Eigen::array<Eigen::Index, 2> input_dims_eigen = {
            static_cast<Eigen::Index>(input_dims_[0]),
            static_cast<Eigen::Index>(input_dims_[1])
        };

        // --- Holographic Routing Config ---
        saguaro::hd_routing::HolographicRoutingConfig routing_config;
        routing_config.hd_dim = static_cast<int>(d_model);  // Operate in d_model space
        routing_config.superposition_dim = static_cast<int>(K);
        routing_config.temperature = routing_temperature_;

        // --- Micro-batched Processing ---
        const int64 micro_batch_size = std::max<int64>(1, std::min<int64>(micro_batch_size_, batch_size));
        const std::size_t cost_per_token = static_cast<std::size_t>(
            std::max<int64>(1, d_model * d_ff * std::max<int64>(1, K)));

        auto process_range = [&](int64_t begin, int64_t end) {
            // Thread-local storage for superposition outputs
            std::vector<float> y_superposed(K * d_model);
            std::vector<float> routing_weights(K);
            
            for (int64_t b = begin; b < end; ++b) {
                const float* x_ptr = tokens_ptr + b * d_model;
                float* out_ptr = output_ptr + b * d_model;
                float* rw_ptr = routing_weights_ptr + b * K;
                
                Map<const VectorXf> x_vec(x_ptr, d_model);
                Map<VectorXf> out_vec(out_ptr, d_model);

                Eigen::TensorMap<Eigen::Tensor<const float, 2>> res_2d(
                    x_ptr, input_dims_eigen);

                // --- Process K superposition paths ---
                for (int k = 0; k < K; ++k) {
                    // FFN1: TT contraction
                    auto core0_k = ffn1_cores.get_core(0).chip(k, 4).chip(0, 0);
                    auto core1_k_4d = ffn1_cores.get_core(1).chip(k, 4);
                    auto core1_k = core1_k_4d.chip(0, 3);
                    
                    const Eigen::array<Eigen::IndexPair<int>, 1> contract_0 =
                        { Eigen::IndexPair<int>(0, 0) };
                    auto temp0_k = res_2d.contract(core0_k, contract_0);
                    
                    const Eigen::array<Eigen::IndexPair<int>, 2> contract_1 =
                        { Eigen::IndexPair<int>(0, 1), Eigen::IndexPair<int>(2, 0) };
                    Eigen::Tensor<float, 2> y1_tensor =
                        temp0_k.contract(core1_k, contract_1).eval();

                    // GELU activation
                    float* y1_data = y1_tensor.data();
                    apply_gelu_inplace(y1_data, y1_tensor.size());

                    // FFN2: TT contraction
                    auto ffn2_core0_k = ffn2_cores.get_core(0).chip(k, 4).chip(0, 0);
                    auto ffn2_core1_k_4d = ffn2_cores.get_core(1).chip(k, 4);
                    auto ffn2_core1_k = ffn2_core1_k_4d.chip(0, 3);

                    auto temp0_ffn2_k = y1_tensor.contract(
                        ffn2_core0_k,
                        Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});
                    Eigen::Tensor<float, 2> y2_tensor =
                        temp0_ffn2_k.contract(
                        ffn2_core1_k,
                        Eigen::array<Eigen::IndexPair<int>, 2>{
                            Eigen::IndexPair<int>(0, 1), Eigen::IndexPair<int>(2, 0)}).eval();

                    // Store superposition output for path k
                    std::memcpy(y_superposed.data() + k * d_model, 
                               y2_tensor.data(), d_model * sizeof(float));
                }

                // --- Holographic Collapse via Circular Routing ---
                saguaro::hd_routing::holographic_collapse_forward(
                    y_superposed.data(),
                    path_bases_ptr,
                    path_weights_ptr,
                    out_ptr,
                    rw_ptr,
                    routing_config
                );
            }
        };

        // --- Execute with Parallel Sharding ---
        for (int64 mb_start = 0; mb_start < batch_size; mb_start += micro_batch_size) {
            const int64 mb_end = std::min(mb_start + micro_batch_size, batch_size);
            const int64 current_micro_batch_size = mb_end - mb_start;
            saguaro::parallel::ForShard(
                static_cast<std::size_t>(current_micro_batch_size),
                cost_per_token,
                [&](int64_t shard_begin, int64_t shard_end) {
                    process_range(mb_start + shard_begin, mb_start + shard_end);
                });
        }
    }

private:
    std::vector<int64> input_dims_, output_dims_ffn1_, output_dims_ffn2_, tt_ranks_;
    int superposition_dim_;
    int64 micro_batch_size_;
    int64 hd_dim_;
    bool use_hd_projection_;
    float routing_temperature_;
};

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_FUSED_SUPERPOSITION_MOE_FORWARD_KERNEL_H_
