// saguaro.native/ops/fused_self_consistency_op.cc
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
 * @file fused_self_consistency_op.cc
 * @brief Self-Consistency Verification TensorFlow Op.
 *
 * Implements DeepSeek-R1 style self-consistency verification for reasoning:
 *   - Multiple path verification
 *   - Pairwise agreement computation
 *   - Consistency-weighted aggregation
 *   - Threshold-based confidence gating
 *
 * Complexity: O(P² × d) per position, where P = num_paths, d = dim
 */

#include "fused_self_consistency_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <algorithm>
#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedSelfConsistency")
    .Input("paths: float32")  // [batch, seq_len, num_paths, dim]
    .Input("verification_weights: float32")  // [num_heads, dim, head_dim]
    .Input("aggregation_weight: float32")  // [total_head_dim, dim]
    .Input("aggregation_bias: float32")  // [dim]
    .Input("norm_gamma: float32")  // [dim]
    .Input("norm_beta: float32")  // [dim]
    .Output("output: float32")  // [batch, seq_len, dim]
    .Output("confidence: float32")  // [batch, seq_len]
    .Attr("num_verification_heads: int = 4")
    .Attr("threshold: float = 0.5")
    .Attr("streaming_chunk_size: int = 0")
    .SetShapeFn([](InferenceContext* c) {
        // paths: [batch, seq_len, num_paths, dim]
        ShapeHandle paths = c->input(0);
        auto batch = c->Dim(paths, 0);
        auto seq_len = c->Dim(paths, 1);
        auto dim = c->Dim(paths, 3);
        
        c->set_output(0, c->MakeShape({batch, seq_len, dim}));
        c->set_output(1, c->MakeShape({batch, seq_len}));
        return Status();
    })
    .Doc(R"doc(
Fused Self-Consistency Verification with SIMD optimization.
Verifies reasoning paths and produces confidence-weighted output.
)doc");

REGISTER_OP("FusedSelfConsistencyGrad")
    .Input("grad_output: float32")
    .Input("grad_confidence: float32")
    .Input("paths: float32")
    .Input("verification_weights: float32")
    .Input("aggregation_weight: float32")
    .Input("aggregation_bias: float32")
    .Input("norm_gamma: float32")
    .Input("norm_beta: float32")
    .Attr("num_verification_heads: int = 4")
    .Attr("threshold: float = 0.5")
    .Attr("streaming_chunk_size: int = 0")
    .Output("grad_paths: float32")
    .Output("grad_verification_weights: float32")
    .Output("grad_aggregation_weight: float32")
    .Output("grad_aggregation_bias: float32")
    .Output("grad_norm_gamma: float32")
    .Output("grad_norm_beta: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(2));  // grad_paths
        c->set_output(1, c->input(3));  // grad_verification_weights
        c->set_output(2, c->input(4));  // grad_aggregation_weight
        c->set_output(3, c->input(5));  // grad_aggregation_bias
        c->set_output(4, c->input(6));  // grad_norm_gamma
        c->set_output(5, c->input(7));  // grad_norm_beta
        return Status();
    });

// =============================================================================
// FORWARD KERNEL
// =============================================================================

namespace {

// Matrix multiply: C[M,N] = A[M,K] @ B[K,N]
void matmul(const float* A, const float* B, float* C,
            int64_t M, int64_t K, int64_t N) {
    #pragma omp parallel for
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

void add_bias(float* C, const float* bias, int64_t rows, int64_t cols) {
    #pragma omp parallel for
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            C[i * cols + j] += bias[j];
        }
    }
}

}  // anonymous namespace

class FusedSelfConsistencyOp : public OpKernel {
 public:
    explicit FusedSelfConsistencyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_verification_heads", &num_verification_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("streaming_chunk_size", &streaming_chunk_size_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& paths = ctx->input(0);
        const Tensor& verification_weights = ctx->input(1);
        const Tensor& aggregation_weight = ctx->input(2);
        const Tensor& aggregation_bias = ctx->input(3);
        const Tensor& norm_gamma = ctx->input(4);
        const Tensor& norm_beta = ctx->input(5);

        // Get dimensions
        const int64_t batch_size = paths.dim_size(0);
        const int64_t seq_len = paths.dim_size(1);
        const int64_t num_paths = paths.dim_size(2);
        const int64_t dim = paths.dim_size(3);
        const int64_t num_heads = verification_weights.dim_size(0);
        const int64_t head_dim = verification_weights.dim_size(2);
        const int64_t total_head_dim = num_heads * head_dim;
        int64_t chunk_size = streaming_chunk_size_ > 0 ? streaming_chunk_size_ : seq_len;
        if (chunk_size <= 0) {
            chunk_size = seq_len;
        }
        if (chunk_size > seq_len) {
            chunk_size = seq_len;
        }
        const int64_t max_batch_chunk = batch_size * chunk_size;

        OP_REQUIRES(ctx, num_heads == num_verification_heads_,
                    errors::InvalidArgument("num_verification_heads mismatch: ",
                                            num_heads, " vs ", num_verification_heads_));
        OP_REQUIRES(ctx, aggregation_weight.dim_size(0) == total_head_dim,
                    errors::InvalidArgument("aggregation_weight first dim mismatch: ",
                                            aggregation_weight.dim_size(0), " vs ", total_head_dim));
        OP_REQUIRES(ctx, aggregation_weight.dim_size(1) == dim,
                    errors::InvalidArgument("aggregation_weight second dim mismatch: ",
                                            aggregation_weight.dim_size(1), " vs ", dim));

        // Allocate outputs
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, seq_len, dim}), &output));
        
        Tensor* confidence = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            1, TensorShape({batch_size, seq_len}), &confidence));

        // Get raw pointers
        const float* paths_data = paths.flat<float>().data();
        const float* vw = verification_weights.flat<float>().data();
        const float* agg_w = aggregation_weight.flat<float>().data();
        const float* agg_b = aggregation_bias.flat<float>().data();
        const float* ng = norm_gamma.flat<float>().data();
        const float* nb = norm_beta.flat<float>().data();
        float* out_data = output->flat<float>().data();
        float* conf_data = confidence->flat<float>().data();

        // Allocate work buffers for streaming chunks
        std::vector<float> paths_chunk(max_batch_chunk * num_paths * dim);
        std::vector<float> normalized_paths(max_batch_chunk * num_paths * dim);
        std::vector<float> agreement(max_batch_chunk * num_paths * num_paths);
        std::vector<float> consistency(max_batch_chunk);
        std::vector<float> path_weights(max_batch_chunk * num_paths);
        std::vector<float> verified_paths(max_batch_chunk * num_paths * total_head_dim);
        std::vector<float> weighted_output(max_batch_chunk * total_head_dim);
        std::vector<float> out_chunk(max_batch_chunk * dim);
        std::vector<float> conf_chunk(max_batch_chunk);

        for (int64_t start = 0; start < seq_len; start += chunk_size) {
            const int64_t chunk_len = std::min<int64_t>(chunk_size, seq_len - start);
            const int64_t batch_chunk = batch_size * chunk_len;

            // Step 1: Copy and L2 normalize paths
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* src = paths_data + (b * seq_len + start) * num_paths * dim;
                float* dst = paths_chunk.data() + b * chunk_len * num_paths * dim;
                std::memcpy(dst, src, chunk_len * num_paths * dim * sizeof(float));
            }
            std::copy(paths_chunk.begin(),
                      paths_chunk.begin() + batch_chunk * num_paths * dim,
                      normalized_paths.begin());

            // Normalize each path vector
            #pragma omp parallel for
            for (int64_t idx = 0; idx < batch_chunk * num_paths; ++idx) {
                float* path = normalized_paths.data() + idx * dim;
                float norm_sq = 0.0f;
                for (int64_t d = 0; d < dim; ++d) {
                    norm_sq += path[d] * path[d];
                }
                float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-8f);
                for (int64_t d = 0; d < dim; ++d) {
                    path[d] *= inv_norm;
                }
            }

            // Step 2: Compute pairwise agreement
            saguaro::ops::self_consistency_pairwise_agreement(
                normalized_paths.data(), agreement.data(),
                batch_size, chunk_len, num_paths, dim);

            // Step 3: Compute consistency scores
            saguaro::ops::self_consistency_compute_score(
                agreement.data(), consistency.data(),
                batch_size, chunk_len, num_paths);

            // Step 4: Compute path weights via softmax
            #pragma omp parallel for
            for (int64_t idx = 0; idx < batch_chunk; ++idx) {
                const int64_t agree_base = idx * num_paths * num_paths;
                const int64_t weight_base = idx * num_paths;
                float cons = consistency[idx];

                // Use first row of agreement scaled by consistency
                for (int64_t p = 0; p < num_paths; ++p) {
                    path_weights[weight_base + p] = cons * agreement[agree_base + p];
                }
            }

            saguaro::ops::self_consistency_softmax(
                path_weights.data(), path_weights.data(),
                batch_chunk, num_paths);

            // Step 5: Weighted combination of paths
            // Step 5: Verification projections (multi-head) + weighted combination
            // verified_paths shape: [batch_chunk, num_paths, total_head_dim]
            for (int64_t idx = 0; idx < batch_chunk * num_paths; ++idx) {
                const float* path = paths_chunk.data() + idx * dim;
                float* verified = verified_paths.data() + idx * total_head_dim;

                for (int64_t h = 0; h < num_heads; ++h) {
                    const float* w_head = vw + h * dim * head_dim;
                    float* out_head = verified + h * head_dim;
                    for (int64_t j = 0; j < head_dim; ++j) {
                        float sum = 0.0f;
                        for (int64_t d = 0; d < dim; ++d) {
                            sum += path[d] * w_head[d * head_dim + j];
                        }
                        out_head[j] = sum;
                    }
                }
            }

            saguaro::ops::self_consistency_weighted_combine(
                verified_paths.data(), path_weights.data(), weighted_output.data(),
                batch_size, chunk_len, num_paths, total_head_dim);

            // Step 6: Aggregation projection
            matmul(weighted_output.data(), agg_w, out_chunk.data(),
                   batch_chunk, total_head_dim, dim);
            add_bias(out_chunk.data(), agg_b, batch_chunk, dim);

            // Step 7: Layer normalization
            saguaro::ops::self_consistency_layer_norm(
                out_chunk.data(), ng, nb, out_chunk.data(),
                batch_chunk, dim);

            // Step 8: Threshold gating for confidence
            saguaro::ops::self_consistency_threshold_gate(
                consistency.data(), conf_chunk.data(),
                batch_chunk, threshold_);

            // Scatter output and confidence
            for (int64_t b = 0; b < batch_size; ++b) {
                float* out_dst = out_data + (b * seq_len + start) * dim;
                const float* out_src = out_chunk.data() + b * chunk_len * dim;
                std::memcpy(out_dst, out_src, chunk_len * dim * sizeof(float));

                float* conf_dst = conf_data + (b * seq_len + start);
                const float* conf_src = conf_chunk.data() + b * chunk_len;
                std::memcpy(conf_dst, conf_src, chunk_len * sizeof(float));
            }
        }
    }

 private:
    int num_verification_heads_;
    float threshold_;
    int streaming_chunk_size_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedSelfConsistency").Device(DEVICE_CPU),
    FusedSelfConsistencyOp);

// =============================================================================
// GRADIENT KERNEL
// =============================================================================

class FusedSelfConsistencyGradOp : public OpKernel {
 public:
    explicit FusedSelfConsistencyGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_verification_heads", &num_verification_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("streaming_chunk_size", &streaming_chunk_size_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        const Tensor& grad_confidence = ctx->input(1);
        const Tensor& paths = ctx->input(2);
        const Tensor& verification_weights = ctx->input(3);
        const Tensor& aggregation_weight = ctx->input(4);
        const Tensor& aggregation_bias = ctx->input(5);
        const Tensor& norm_gamma = ctx->input(6);
        const Tensor& norm_beta = ctx->input(7);

        const int64_t batch_size = paths.dim_size(0);
        const int64_t seq_len = paths.dim_size(1);
        const int64_t num_paths = paths.dim_size(2);
        const int64_t dim = paths.dim_size(3);
        const int64_t num_heads = verification_weights.dim_size(0);
        const int64_t head_dim = verification_weights.dim_size(2);
        const int64_t total_head_dim = num_heads * head_dim;
        int64_t chunk_size = streaming_chunk_size_ > 0 ? streaming_chunk_size_ : seq_len;
        if (chunk_size <= 0) {
            chunk_size = seq_len;
        }
        if (chunk_size > seq_len) {
            chunk_size = seq_len;
        }
        const int64_t max_batch_chunk = batch_size * chunk_size;

        OP_REQUIRES(ctx, num_heads == num_verification_heads_,
                    errors::InvalidArgument("num_verification_heads mismatch: ",
                                            num_heads, " vs ", num_verification_heads_));
        OP_REQUIRES(ctx, aggregation_weight.dim_size(0) == total_head_dim,
                    errors::InvalidArgument("aggregation_weight first dim mismatch: ",
                                            aggregation_weight.dim_size(0), " vs ", total_head_dim));
        OP_REQUIRES(ctx, aggregation_weight.dim_size(1) == dim,
                    errors::InvalidArgument("aggregation_weight second dim mismatch: ",
                                            aggregation_weight.dim_size(1), " vs ", dim));

        // Allocate gradients
        Tensor* grad_paths = nullptr;
        Tensor* grad_verification_weights = nullptr;
        Tensor* grad_aggregation_weight = nullptr;
        Tensor* grad_aggregation_bias = nullptr;
        Tensor* grad_norm_gamma = nullptr;
        Tensor* grad_norm_beta = nullptr;

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, paths.shape(), &grad_paths));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, verification_weights.shape(), 
                                                  &grad_verification_weights));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, aggregation_weight.shape(), 
                                                  &grad_aggregation_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, TensorShape({dim}), 
                                                  &grad_aggregation_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, TensorShape({dim}), 
                                                  &grad_norm_gamma));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(5, TensorShape({dim}), 
                                                  &grad_norm_beta));

        // Get raw pointers
        const float* grad_out = grad_output.flat<float>().data();
        const float* paths_data = paths.flat<float>().data();
        const float* vw = verification_weights.flat<float>().data();
        const float* agg_w = aggregation_weight.flat<float>().data();
        const float* ng = norm_gamma.flat<float>().data();
        float* grad_paths_data = grad_paths->flat<float>().data();
        float* grad_agg_w = grad_aggregation_weight->flat<float>().data();
        float* grad_agg_b = grad_aggregation_bias->flat<float>().data();
        float* grad_ng = grad_norm_gamma->flat<float>().data();
        float* grad_nb = grad_norm_beta->flat<float>().data();
        float* grad_vw = grad_verification_weights->flat<float>().data();

        // Initialize gradients to zero
        std::fill(grad_paths_data, grad_paths_data + paths.NumElements(), 0.0f);
        std::fill(grad_agg_w, grad_agg_w + aggregation_weight.NumElements(), 0.0f);
        std::fill(grad_agg_b, grad_agg_b + dim, 0.0f);
        std::fill(grad_ng, grad_ng + dim, 0.0f);
        std::fill(grad_nb, grad_nb + dim, 0.0f);
        std::fill(grad_vw, grad_vw + verification_weights.NumElements(), 0.0f);

        // Recompute forward pass values needed for gradient (chunked)
        std::vector<float> paths_chunk(max_batch_chunk * num_paths * dim);
        std::vector<float> normalized_paths(max_batch_chunk * num_paths * dim);
        std::vector<float> agreement(max_batch_chunk * num_paths * num_paths);
        std::vector<float> consistency(max_batch_chunk);
        std::vector<float> path_weights(max_batch_chunk * num_paths);
        std::vector<float> verified_paths(max_batch_chunk * num_paths * total_head_dim);
        std::vector<float> weighted_output(max_batch_chunk * total_head_dim);
        std::vector<float> pre_norm_output(max_batch_chunk * dim);
        std::vector<float> grad_out_chunk(max_batch_chunk * dim);
        std::vector<float> grad_pre_norm(max_batch_chunk * dim);
        std::vector<float> grad_weighted(max_batch_chunk * total_head_dim);
        std::vector<float> grad_agg_w_chunk(total_head_dim * dim, 0.0f);

        for (int64_t start = 0; start < seq_len; start += chunk_size) {
            const int64_t chunk_len = std::min<int64_t>(chunk_size, seq_len - start);
            const int64_t batch_chunk = batch_size * chunk_len;

            // Recompute: L2 normalize paths
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* src = paths_data + (b * seq_len + start) * num_paths * dim;
                float* dst = paths_chunk.data() + b * chunk_len * num_paths * dim;
                std::memcpy(dst, src, chunk_len * num_paths * dim * sizeof(float));
            }
            std::copy(paths_chunk.begin(),
                      paths_chunk.begin() + batch_chunk * num_paths * dim,
                      normalized_paths.begin());
            #pragma omp parallel for
            for (int64_t i = 0; i < batch_chunk * num_paths; ++i) {
                float* path = normalized_paths.data() + i * dim;
                float norm_sq = 0.0f;
                for (int64_t d = 0; d < dim; ++d) norm_sq += path[d] * path[d];
                float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-8f);
                for (int64_t d = 0; d < dim; ++d) path[d] *= inv_norm;
            }

            // Recompute: pairwise agreement and consistency
            saguaro::ops::self_consistency_pairwise_agreement(
                normalized_paths.data(), agreement.data(),
                batch_size, chunk_len, num_paths, dim);
            saguaro::ops::self_consistency_compute_score(
                agreement.data(), consistency.data(),
                batch_size, chunk_len, num_paths);

            // Recompute: path weights
            #pragma omp parallel for
            for (int64_t idx = 0; idx < batch_chunk; ++idx) {
                const int64_t agree_base = idx * num_paths * num_paths;
                const int64_t weight_base = idx * num_paths;
                float cons = consistency[idx];
                for (int64_t p = 0; p < num_paths; ++p) {
                    path_weights[weight_base + p] = cons * agreement[agree_base + p];
                }
            }
            saguaro::ops::self_consistency_softmax(
                path_weights.data(), path_weights.data(),
                batch_chunk, num_paths);

            // Recompute: verification projections + weighted combination
            for (int64_t idx = 0; idx < batch_chunk * num_paths; ++idx) {
                const float* path = paths_chunk.data() + idx * dim;
                float* verified = verified_paths.data() + idx * total_head_dim;

                for (int64_t h = 0; h < num_heads; ++h) {
                    const float* w_head = vw + h * dim * head_dim;
                    float* out_head = verified + h * head_dim;
                    for (int64_t j = 0; j < head_dim; ++j) {
                        float sum = 0.0f;
                        for (int64_t d = 0; d < dim; ++d) {
                            sum += path[d] * w_head[d * head_dim + j];
                        }
                        out_head[j] = sum;
                    }
                }
            }

            saguaro::ops::self_consistency_weighted_combine(
                verified_paths.data(), path_weights.data(), weighted_output.data(),
                batch_size, chunk_len, num_paths, total_head_dim);

            // Recompute: aggregation projection (store pre-norm for gradient)
            matmul(weighted_output.data(), agg_w, pre_norm_output.data(),
                   batch_chunk, total_head_dim, dim);
            add_bias(pre_norm_output.data(),
                     aggregation_bias.flat<float>().data(),
                     batch_chunk, dim);

            // ============ BACKWARD PASS ============

            // Step 1: Gradient through layer norm
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* src = grad_out + (b * seq_len + start) * dim;
                float* dst = grad_out_chunk.data() + b * chunk_len * dim;
                std::memcpy(dst, src, chunk_len * dim * sizeof(float));
            }
            saguaro::ops::self_consistency_layer_norm_backward(
                grad_out_chunk.data(), pre_norm_output.data(), ng,
                grad_pre_norm.data(), grad_ng, grad_nb,
                batch_chunk, dim);

            // Step 2: Gradient through aggregation bias
            #pragma omp parallel for
            for (int64_t d = 0; d < dim; ++d) {
                float sum = 0.0f;
                for (int64_t i = 0; i < batch_chunk; ++i) {
                    sum += grad_pre_norm[i * dim + d];
                }
                grad_agg_b[d] += sum;
            }

            // Step 3: Gradient through aggregation matmul
            // grad_weighted = grad_pre_norm @ agg_w^T
            #pragma omp parallel for
            for (int64_t i = 0; i < batch_chunk; ++i) {
                for (int64_t j = 0; j < total_head_dim; ++j) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < dim; ++k) {
                        sum += grad_pre_norm[i * dim + k] * agg_w[j * dim + k];
                    }
                    grad_weighted[i * total_head_dim + j] = sum;
                }
            }

            // grad_agg_w = weighted_output^T @ grad_pre_norm
            std::fill(grad_agg_w_chunk.begin(), grad_agg_w_chunk.end(), 0.0f);
            #pragma omp parallel for
            for (int64_t i = 0; i < total_head_dim; ++i) {
                for (int64_t j = 0; j < dim; ++j) {
                    float sum = 0.0f;
                    for (int64_t k = 0; k < batch_chunk; ++k) {
                        sum += weighted_output[k * total_head_dim + i] * grad_pre_norm[k * dim + j];
                    }
                    grad_agg_w_chunk[i * dim + j] = sum;
                }
            }
            for (int64_t i = 0; i < total_head_dim * dim; ++i) {
                grad_agg_w[i] += grad_agg_w_chunk[i];
            }

            // Step 4: Gradient through weighted combination + verification projections
#ifdef _OPENMP
            const int max_threads = omp_get_max_threads();
            std::vector<std::vector<float>> grad_vw_local(
                max_threads, std::vector<float>(verification_weights.NumElements(), 0.0f));
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                float* grad_vw_thread = grad_vw_local[tid].data();
                #pragma omp for
                for (int64_t b = 0; b < batch_size; ++b) {
                    for (int64_t s = 0; s < chunk_len; ++s) {
                        const int64_t pos_idx = b * chunk_len + s;
                        const float* gw = grad_weighted.data() + pos_idx * total_head_dim;
                        const float* pw = path_weights.data() + pos_idx * num_paths;

                        float* gp = grad_paths_data +
                            ((b * seq_len + (start + s)) * num_paths) * dim;
                        const float* path_src = paths_chunk.data() +
                            (b * chunk_len + s) * num_paths * dim;

                        for (int64_t p = 0; p < num_paths; ++p) {
                            const float* path = path_src + p * dim;
                            float* gp_path = gp + p * dim;
                            const float path_weight = pw[p];

                            for (int64_t h = 0; h < num_heads; ++h) {
                                const float* w_head = vw + h * dim * head_dim;
                                float* grad_w_head = grad_vw_thread + h * dim * head_dim;
                                const float* gw_head = gw + h * head_dim;

                                for (int64_t j = 0; j < head_dim; ++j) {
                                    const float gv = path_weight * gw_head[j];
                                    for (int64_t d = 0; d < dim; ++d) {
                                        grad_w_head[d * head_dim + j] += path[d] * gv;
                                        gp_path[d] += w_head[d * head_dim + j] * gv;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for (int t = 0; t < max_threads; ++t) {
                const float* local = grad_vw_local[t].data();
                for (int64_t i = 0; i < verification_weights.NumElements(); ++i) {
                    grad_vw[i] += local[i];
                }
            }
#else
            for (int64_t b = 0; b < batch_size; ++b) {
                for (int64_t s = 0; s < chunk_len; ++s) {
                    const int64_t pos_idx = b * chunk_len + s;
                    const float* gw = grad_weighted.data() + pos_idx * total_head_dim;
                    const float* pw = path_weights.data() + pos_idx * num_paths;

                    float* gp = grad_paths_data +
                        ((b * seq_len + (start + s)) * num_paths) * dim;
                    const float* path_src = paths_chunk.data() +
                        (b * chunk_len + s) * num_paths * dim;

                    for (int64_t p = 0; p < num_paths; ++p) {
                        const float* path = path_src + p * dim;
                        float* gp_path = gp + p * dim;
                        const float path_weight = pw[p];

                        for (int64_t h = 0; h < num_heads; ++h) {
                            const float* w_head = vw + h * dim * head_dim;
                            float* grad_w_head = grad_vw + h * dim * head_dim;
                            const float* gw_head = gw + h * head_dim;

                            for (int64_t j = 0; j < head_dim; ++j) {
                                const float gv = path_weight * gw_head[j];
                                for (int64_t d = 0; d < dim; ++d) {
                                    grad_w_head[d * head_dim + j] += path[d] * gv;
                                    gp_path[d] += w_head[d * head_dim + j] * gv;
                                }
                            }
                        }
                    }
                }
            }
#endif
        }
    }

 private:
    int num_verification_heads_;
    float threshold_;
    int streaming_chunk_size_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedSelfConsistencyGrad").Device(DEVICE_CPU),
    FusedSelfConsistencyGradOp);

}  // namespace tensorflow
