// src/ops/fused_superposition_moe/backward_kernel.h
// Copyright 2025 Verso Industries
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
// UNIFIED HD-SUPERPOSED EXPERT BACKWARD KERNEL (v2.0)
//
// Updated to match v2.0 forward kernel with holographic routing.
// Breaking change: Q/K/V/O collapse weights replaced with path_bases/path_weights.
//
// Key Changes:
// - Inputs now match forward: tokens, ffn1_cores, ffn2_cores, path_bases,
//   path_weights, hd_input_proj, hd_output_proj, routing_weights
// - Uses holographic_collapse_backward for routing gradient
// - Micro-batching preserved for memory efficiency
// ============================================================================


#ifndef TENSORFLOW_CORE_USER_OPS_FUSED_SUPERPOSITION_MOE_BACKWARD_KERNEL_H_
#define TENSORFLOW_CORE_USER_OPS_FUSED_SUPERPOSITION_MOE_BACKWARD_KERNEL_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "helpers.h"
#include "holographic_routing.h"
#include "common/parallel/parallel_backend.h"
#include "absl/synchronization/mutex.h"
#include <atomic>
#include <algorithm>
#include <vector>
#include <cstring>

namespace tensorflow {

using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowMajor;
using Eigen::Matrix;
using Eigen::Dynamic;

/**
 * @brief Backward kernel for Unified HD-SuperposedExpert (v2.0).
 *
 * Inputs (index):
 *   0: grad_output [B, d_model]
 *   1: tokens [B, d_model]
 *   2: ffn1_cores [total_core_elements]
 *   3: ffn2_cores [total_core_elements]
 *   4: path_bases [K, d_model]
 *   5: path_weights [K, d_model]
 *   6: hd_input_proj [d_model, hd_dim] (optional)
 *   7: hd_output_proj [hd_dim, d_model] (optional)
 *   8: routing_weights [B, K] (cached from forward)
 *
 * Outputs:
 *   0: grad_tokens [B, d_model]
 *   1: grad_ffn1_cores [total_core_elements]
 *   2: grad_ffn2_cores [total_core_elements]
 *   3: grad_path_bases [K, d_model]
 *   4: grad_path_weights [K, d_model]
 *   5: grad_hd_input_proj [d_model, hd_dim]
 *   6: grad_hd_output_proj [hd_dim, d_model]
 */
class FusedSuperpositionMoeGradOpCpu : public OpKernel {
public:
    explicit FusedSuperpositionMoeGradOpCpu(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("input_dims", &input_dims_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dims_ffn1", &output_dims_ffn1_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dims_ffn2", &output_dims_ffn2_));
        OP_REQUIRES_OK(context, context->GetAttr("tt_ranks", &tt_ranks_));
        OP_REQUIRES_OK(context, context->GetAttr("superposition_dim", &superposition_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("micro_batch_size", &micro_batch_size_));
        OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &hd_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("use_hd_projection", &use_hd_projection_));
        OP_REQUIRES_OK(context, context->GetAttr("routing_temperature", &routing_temperature_));
        OP_REQUIRES(context, micro_batch_size_ > 0,
                    errors::InvalidArgument("micro_batch_size must be positive, got ", micro_batch_size_));
    }

    void Compute(OpKernelContext* context) override {
        // --- 1. Get All Input Tensors (v2.0 API) ---
        const auto& grad_output = context->input(0);
        const auto& tokens = context->input(1);
        const auto& ffn1_cores_tensor = context->input(2);
        const auto& ffn2_cores_tensor = context->input(3);
        const auto& path_bases_tensor = context->input(4);
        const auto& path_weights_tensor = context->input(5);
        const auto& hd_input_proj_tensor = context->input(6);
        const auto& hd_output_proj_tensor = context->input(7);
        const auto& routing_weights_tensor = context->input(8);

        // --- 2. Get Dimensions ---
        const auto& tokens_shape = tokens.shape();
        OP_REQUIRES(context, tokens_shape.dims() == 2,
            errors::InvalidArgument("FusedSuperpositionMoeGradOpCpu expects a 2D input tensor (batch, features), but got shape ",
            tokens_shape.DebugString(), ". The Python layer should have flattened the input."));
        const int64 batch_size = tokens_shape.dim_size(0);
        const int64 d_model = tokens_shape.dim_size(1);
        const int64 K = superposition_dim_;
        const int64 hd_dim = use_hd_projection_ ? hd_dim_ : d_model;

        int64 d_ff = 1;
        for (int64 dim : output_dims_ffn1_) d_ff *= dim;

        // --- 3. Allocate Output Gradient Tensors ---
        Tensor* grad_tokens;
        OP_REQUIRES_OK(context, context->allocate_output(0, tokens.shape(), &grad_tokens));
        Tensor* grad_ffn1_cores;
        OP_REQUIRES_OK(context, context->allocate_output(1, ffn1_cores_tensor.shape(), &grad_ffn1_cores));
        Tensor* grad_ffn2_cores;
        OP_REQUIRES_OK(context, context->allocate_output(2, ffn2_cores_tensor.shape(), &grad_ffn2_cores));
        Tensor* grad_path_bases;
        OP_REQUIRES_OK(context, context->allocate_output(3, path_bases_tensor.shape(), &grad_path_bases));
        Tensor* grad_path_weights;
        OP_REQUIRES_OK(context, context->allocate_output(4, path_weights_tensor.shape(), &grad_path_weights));
        Tensor* grad_hd_input_proj;
        OP_REQUIRES_OK(context, context->allocate_output(5, hd_input_proj_tensor.shape(), &grad_hd_input_proj));
        Tensor* grad_hd_output_proj;
        OP_REQUIRES_OK(context, context->allocate_output(6, hd_output_proj_tensor.shape(), &grad_hd_output_proj));

        // Zero-initialize all outputs
        grad_tokens->flat<float>().setZero();
        grad_ffn1_cores->flat<float>().setZero();
        grad_ffn2_cores->flat<float>().setZero();
        grad_path_bases->flat<float>().setZero();
        grad_path_weights->flat<float>().setZero();
        grad_hd_input_proj->flat<float>().setZero();
        grad_hd_output_proj->flat<float>().setZero();

        if (batch_size == 0) {
            return;
        }

        // --- 4. Prepare Data Pointers ---
        const float* tokens_ptr = tokens.flat<float>().data();
        const float* path_bases_ptr = path_bases_tensor.flat<float>().data();
        const float* path_weights_ptr = path_weights_tensor.flat<float>().data();
        const float* routing_weights_ptr = routing_weights_tensor.flat<float>().data();

        float* grad_path_bases_ptr = grad_path_bases->flat<float>().data();
        float* grad_path_weights_ptr = grad_path_weights->flat<float>().data();

        // --- 5. Prepare TT Core Managers ---
        TTCores ffn1_cores(ffn1_cores_tensor, tt_ranks_, input_dims_, output_dims_ffn1_, K);
        TTCores ffn2_cores(ffn2_cores_tensor, tt_ranks_, output_dims_ffn1_, output_dims_ffn2_, K);

        // --- 6. Holographic Routing Config ---
        saguaro::hd_routing::HolographicRoutingConfig routing_config;
        routing_config.hd_dim = static_cast<int>(d_model);
        routing_config.superposition_dim = static_cast<int>(K);
        routing_config.temperature = routing_temperature_;

        // --- 7. Micro-batched Backward Pass ---
        const int64 micro_batch_size = micro_batch_size_;
        const Eigen::DSizes<Eigen::Index, 2> ffn1_dims(
            static_cast<Eigen::Index>(output_dims_ffn1_[0]),
            static_cast<Eigen::Index>(output_dims_ffn1_[1]));
        const Eigen::DSizes<Eigen::Index, 2> ffn2_dims(
            static_cast<Eigen::Index>(output_dims_ffn2_[0]),
            static_cast<Eigen::Index>(output_dims_ffn2_[1]));
        const Eigen::DSizes<Eigen::Index, 1> vec_dims_dff(d_ff);
        const Eigen::array<Eigen::Index, 2> input_dims_eigen = {
            static_cast<Eigen::Index>(input_dims_[0]),
            static_cast<Eigen::Index>(input_dims_[1])
        };

        for (int64 mb_start = 0; mb_start < batch_size; mb_start += micro_batch_size) {
            const int64 mb_end = std::min(mb_start + micro_batch_size, batch_size);
            const int64 current_micro_batch_size = mb_end - mb_start;

            auto shard_fn = [&](int64_t start, int64_t end) {
                // Thread-local storage
                std::vector<float> y_superposed(K * d_model);
                std::vector<float> grad_y_superposed(K * d_model);
                std::vector<float> local_grad_path_weights(K * d_model, 0.0f);
                std::vector<float> local_grad_path_bases(K * d_model, 0.0f);

                for (int64_t b_idx = start; b_idx < end; ++b_idx) {
                    const int64 b = mb_start + b_idx;

                    // --- 7a. Recompute Forward Pass Intermediates ---
                    const float* x_ptr = tokens_ptr + b * d_model;
                    Map<const VectorXf> x_vec(x_ptr, d_model);
                    Eigen::TensorMap<Eigen::Tensor<const float, 2>> res_2d(
                        x_ptr, input_dims_eigen);

                    Eigen::Tensor<float, 2> y1_pre_gelu(d_ff, K);
                    for (int k = 0; k < K; ++k) {
                        auto core0_k = ffn1_cores.get_core(0).chip(k, 4).chip(0, 0);
                        auto core1_k_4d = ffn1_cores.get_core(1).chip(k, 4);
                        auto core1_k = core1_k_4d.chip(0, 3);
                        Eigen::array<Eigen::IndexPair<int>, 1> contract_0 = { Eigen::IndexPair<int>(0, 0) };
                        auto temp0_k = res_2d.contract(core0_k, contract_0);
                        Eigen::array<Eigen::IndexPair<int>, 2> contract_1 = { Eigen::IndexPair<int>(0, 1), Eigen::IndexPair<int>(2, 0) };
                        y1_pre_gelu.chip(k, 1) = temp0_k.contract(core1_k, contract_1).reshape(vec_dims_dff);
                    }

                    // GELU activation
                    Eigen::Tensor<float, 2> y1_superposed = y1_pre_gelu;
                    apply_gelu_inplace(y1_superposed.data(), y1_superposed.size());

                    // FFN2 pass
                    for (int k = 0; k < K; ++k) {
                        Eigen::Tensor<float, 1> y1_k_vec = y1_superposed.chip(k, 1);
                        Eigen::Tensor<float, 2> y1_k_2d = y1_k_vec.reshape(ffn1_dims);
                        auto ffn2_core0_k = ffn2_cores.get_core(0).chip(k, 4).chip(0, 0);
                        auto ffn2_core1_k_4d = ffn2_cores.get_core(1).chip(k, 4);
                        auto ffn2_core1_k = ffn2_core1_k_4d.chip(0, 3);
                        auto temp0_ffn2_k = y1_k_2d.contract(ffn2_core0_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0,0)});
                        Eigen::Tensor<float, 1> y2_k = temp0_ffn2_k.contract(ffn2_core1_k,
                            Eigen::array<Eigen::IndexPair<int>, 2>{
                                Eigen::IndexPair<int>(0,1), Eigen::IndexPair<int>(2,0)})
                            .reshape(Eigen::DSizes<Eigen::Index,1>(d_model));
                        std::memcpy(y_superposed.data() + k * d_model, y2_k.data(), d_model * sizeof(float));
                    }

                    // --- 7b. Backward Through Holographic Collapse ---
                    const float* grad_out_ptr = grad_output.matrix<float>().data() + b * d_model;
                    const float* rw_ptr = routing_weights_ptr + b * K;

                    saguaro::hd_routing::holographic_collapse_backward(
                        grad_out_ptr,
                        y_superposed.data(),
                        path_bases_ptr,
                        path_weights_ptr,
                        rw_ptr,
                        grad_y_superposed.data(),
                        local_grad_path_weights.data(),
                        local_grad_path_bases.data(),
                        routing_config
                    );

                    // Accumulate path_weights and path_bases gradients (thread-safe)
                    {
                        absl::MutexLock l(&mu_);
                        for (int64 i = 0; i < K * d_model; ++i) {
                            grad_path_weights_ptr[i] += local_grad_path_weights[i];
                            grad_path_bases_ptr[i] += local_grad_path_bases[i];
                        }
                    }

                    // --- 7c. Backward Through FFN2 ---
                    Eigen::Tensor<float, 2> grad_y1_superposed(d_ff, K);
                    grad_y1_superposed.setZero();
                    for (int k = 0; k < K; ++k) {
                        Eigen::TensorMap<const Eigen::Tensor<float, 1>> grad_y2_k_vec(
                            grad_y_superposed.data() + k * d_model, d_model);
                        Eigen::Tensor<float, 2> grad_Y2_k = grad_y2_k_vec.reshape(ffn2_dims);
                        Eigen::Tensor<float, 1> y1_k_vec = y1_superposed.chip(k, 1);
                        Eigen::Tensor<float, 2> y1_k_2d = y1_k_vec.reshape(ffn1_dims);
                        auto D0_k = ffn2_cores.get_core(0).chip(k, 4).chip(0, 0);
                        auto D1_k_4d = ffn2_cores.get_core(1).chip(k, 4);
                        auto D1_k = D1_k_4d.chip(0, 3);
                        Eigen::Tensor<float, 3> T1_k = y1_k_2d.contract(D0_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0,0)});
                        Eigen::Tensor<float, 3> g_T1_k = grad_Y2_k.contract(D1_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 2)})
                            .shuffle(Eigen::array<int, 3>{2, 0, 1});
                        Eigen::Tensor<float, 3> g_D1_k = T1_k.contract(grad_Y2_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)})
                            .shuffle(Eigen::array<int, 3>{1, 0, 2});
                        Eigen::Tensor<float, 2> g_y1_k_2d = g_T1_k.contract(D0_k,
                            Eigen::array<Eigen::IndexPair<int>, 2>{
                                Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)})
                            .shuffle(Eigen::array<int, 2>{1, 0});
                        Eigen::Tensor<float, 3> g_D0_k = y1_k_2d.contract(g_T1_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1,0)});
                        grad_y1_superposed.chip(k, 1) = g_y1_k_2d.reshape(vec_dims_dff);

                        {
                            absl::MutexLock l(&mu_);
                            auto grad_ffn2_core0 = Eigen::TensorMap<Eigen::Tensor<float, 5>>(
                                grad_ffn2_cores->flat<float>().data(),
                                tt_ranks_[0], output_dims_ffn1_[0], output_dims_ffn2_[0], tt_ranks_[1], K);
                            grad_ffn2_core0.chip(k, 4).chip(0, 0) += g_D0_k;
                            int64 ffn2_core0_total_size = tt_ranks_[0] * output_dims_ffn1_[0] * output_dims_ffn2_[0] * tt_ranks_[1] * K;
                            float* grad_ffn2_core1_ptr = grad_ffn2_cores->flat<float>().data() + ffn2_core0_total_size;
                            auto grad_ffn2_core1 = Eigen::TensorMap<Eigen::Tensor<float, 5>>(
                                grad_ffn2_core1_ptr,
                                tt_ranks_[1], output_dims_ffn1_[1], output_dims_ffn2_[1], tt_ranks_[2], K);
                            grad_ffn2_core1.chip(k, 4).chip(0, 3) += g_D1_k;
                        }
                    }

                    // --- 7d. Backward Through GELU ---
                    apply_gelu_grad_inplace(grad_y1_superposed.data(), y1_pre_gelu.data(), y1_pre_gelu.size());

                    // --- 7e. Backward Through FFN1 ---
                    Eigen::Tensor<float, 2> grad_X_2d(input_dims_[0], input_dims_[1]);
                    grad_X_2d.setZero();
                    for (int k = 0; k < K; ++k) {
                        Eigen::Tensor<float, 1> grad_y1_pre_gelu_k_vec = grad_y1_superposed.chip(k, 1);
                        Eigen::Tensor<float, 2> grad_Y1_k = grad_y1_pre_gelu_k_vec.reshape(ffn1_dims);
                        auto C0_k = ffn1_cores.get_core(0).chip(k, 4).chip(0, 0);
                        auto C1_k_4d = ffn1_cores.get_core(1).chip(k, 4);
                        auto C1_k = C1_k_4d.chip(0, 3);
                        Eigen::Tensor<float, 3> T0_k = res_2d.contract(C0_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});
                        Eigen::Tensor<float, 3> g_T0_k = grad_Y1_k.contract(C1_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 2)})
                            .shuffle(Eigen::array<int, 3>{2, 0, 1});
                        Eigen::Tensor<float, 3> g_C1_k = T0_k.contract(grad_Y1_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)})
                            .shuffle(Eigen::array<int, 3>{1, 0, 2});
                        Eigen::Tensor<float, 2> g_X_2d_k = g_T0_k.contract(C0_k,
                            Eigen::array<Eigen::IndexPair<int>, 2>{
                                Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(2,2)})
                            .shuffle(Eigen::array<int, 2>{1, 0});
                        Eigen::Tensor<float, 3> g_C0_k = res_2d.contract(g_T0_k,
                            Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1,0)});
                        grad_X_2d += g_X_2d_k;

                        {
                            absl::MutexLock l(&mu_);
                            auto grad_ffn1_core0 = Eigen::TensorMap<Eigen::Tensor<float, 5>>(
                                grad_ffn1_cores->flat<float>().data(),
                                tt_ranks_[0], input_dims_[0], output_dims_ffn1_[0], tt_ranks_[1], K);
                            grad_ffn1_core0.chip(k, 4).chip(0, 0) += g_C0_k;
                            int64 ffn1_core0_total_size = tt_ranks_[0] * input_dims_[0] * output_dims_ffn1_[0] * tt_ranks_[1] * K;
                            float* grad_ffn1_core1_ptr = grad_ffn1_cores->flat<float>().data() + ffn1_core0_total_size;
                            auto grad_ffn1_core1 = Eigen::TensorMap<Eigen::Tensor<float, 5>>(
                                grad_ffn1_core1_ptr,
                                tt_ranks_[1], input_dims_[1], output_dims_ffn1_[1], tt_ranks_[2], K);
                            grad_ffn1_core1.chip(k, 4).chip(0, 3) += g_C1_k;
                        }
                    }

                    // Write token gradients
                    {
                        absl::MutexLock l(&mu_);
                        Map<VectorXf> grad_x_vec(grad_X_2d.data(), d_model);
                        for (int i = 0; i < d_model; ++i) {
                            grad_tokens->matrix<float>()(b, i) += grad_x_vec(i);
                        }
                    }

                    // Reset local accumulators
                    std::fill(local_grad_path_weights.begin(), local_grad_path_weights.end(), 0.0f);
                    std::fill(local_grad_path_bases.begin(), local_grad_path_bases.end(), 0.0f);
                }
            };

            const int64 cost_per_unit = d_model * d_ff * K;
            saguaro::parallel::ForShard(
                static_cast<std::size_t>(current_micro_batch_size),
                static_cast<std::size_t>(cost_per_unit),
                shard_fn);
        }
    }

private:
    std::vector<int64> input_dims_, output_dims_ffn1_, output_dims_ffn2_, tt_ranks_;
    int superposition_dim_;
    int64 micro_batch_size_;
    int64 hd_dim_;
    bool use_hd_projection_;
    float routing_temperature_;
    mutable absl::Mutex mu_;
};


} // namespace tensorflow


#endif // TENSORFLOW_CORE_USER_OPS_FUSED_SUPERPOSITION_MOE_BACKWARD_KERNEL_H_
