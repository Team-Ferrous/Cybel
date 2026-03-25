// saguaro.native/ops/fused_superposition_slots_op.cc
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
 * @file fused_superposition_slots_op.cc
 * @brief TensorFlow custom ops for quantum-inspired slot superposition.
 *
 * Enhancement 4: Quantum-Inspired Slot Superposition
 *
 * Two main operations:
 * 1. SuperpositionCollapseRead: Collapse superposition via query attention
 * 2. SuperpositionWrite: Gated update to all superposition dimensions
 */

#include "fused_superposition_slots_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <cmath>
#include <vector>

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// OP REGISTRATION: SuperpositionCollapseRead
// =============================================================================

REGISTER_OP("SuperpositionCollapseRead")
    .Input("query: float32")           // [batch, query_dim]
    .Input("buffer: float32")          // [batch, num_slots, superposition_dim, bus_dim]
    .Input("collapse_weight: float32") // [query_dim, bus_dim]
    .Input("collapse_bias: float32")   // [bus_dim]
    .Output("collapsed_slots: float32")// [batch, num_slots, bus_dim]
    .Attr("num_slots: int")
    .Attr("superposition_dim: int")
    .Attr("bus_dim: int")
    .Attr("temperature: float = 1.0")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle query_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &query_shape));
        DimensionHandle batch_dim = c->Dim(query_shape, 0);
        int64_t num_slots, bus_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("num_slots", &num_slots));
        TF_RETURN_IF_ERROR(c->GetAttr("bus_dim", &bus_dim));
        c->set_output(0, c->MakeShape({batch_dim, num_slots, bus_dim}));
        return OkStatus();
    })
    .Doc(R"doc(
Collapse superposition buffer via query-conditioned attention.

For each slot, computes attention weights over superposition dimension
and returns weighted sum as collapsed slot values.

query: Query tensor [batch, query_dim]
buffer: Superposition buffer [batch, num_slots, superposition_dim, bus_dim]
collapse_weight: Projection weights [query_dim, bus_dim]
collapse_bias: Projection bias [bus_dim]
collapsed_slots: Resulting slots [batch, num_slots, bus_dim]
)doc");

// =============================================================================
// OP REGISTRATION: SuperpositionWrite
// =============================================================================

REGISTER_OP("SuperpositionWrite")
    .Input("content: float32")         // [batch, bus_dim]
    .Input("gate: float32")            // [batch, num_slots]
    .Input("buffer: float32")          // [batch, num_slots, superposition_dim, bus_dim]
    .Output("buffer_new: float32")     // [batch, num_slots, superposition_dim, bus_dim]
    .Attr("num_slots: int")
    .Attr("superposition_dim: int")
    .Attr("bus_dim: int")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(2));
        return OkStatus();
    })
    .Doc(R"doc(
Gated write to all superposition dimensions.

Updates buffer with blend: new = old * (1 - gate) + content * gate

content: Content to write [batch, bus_dim]
gate: Per-slot write gate [batch, num_slots] (0-1 values)
buffer: Input buffer [batch, num_slots, superposition_dim, bus_dim]
buffer_new: Updated buffer [batch, num_slots, superposition_dim, bus_dim]
)doc");

// =============================================================================
// KERNEL: SuperpositionCollapseRead
// =============================================================================

class SuperpositionCollapseReadOp : public OpKernel {
 public:
    explicit SuperpositionCollapseReadOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_slots", &num_slots_));
        OP_REQUIRES_OK(context, context->GetAttr("superposition_dim", &superdim_));
        OP_REQUIRES_OK(context, context->GetAttr("bus_dim", &bus_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& query = context->input(0);
        const Tensor& buffer = context->input(1);
        const Tensor& collapse_weight = context->input(2);
        const Tensor& collapse_bias = context->input(3);

        const int64_t batch = query.dim_size(0);
        const int64_t query_dim = query.dim_size(1);

        // Allocate output
        Tensor* collapsed = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, TensorShape({batch, num_slots_, bus_dim_}), &collapsed));

        // Get data pointers
        const float* q_data = query.flat<float>().data();
        const float* buf_data = buffer.flat<float>().data();
        const float* w_data = collapse_weight.flat<float>().data();
        const float* b_data = collapse_bias.flat<float>().data();
        float* out_data = collapsed->flat<float>().data();

        // Temp buffers
        std::vector<float> proj_query(bus_dim_);
        std::vector<float> scores(superdim_);

        // Process each batch
        for (int64_t b = 0; b < batch; ++b) {
            const float* q_batch = q_data + b * query_dim;

            // Project query: q_proj = query @ weight + bias
            for (int64_t d = 0; d < bus_dim_; ++d) {
                float sum = b_data[d];
                for (int64_t i = 0; i < query_dim; ++i) {
                    sum += q_batch[i] * w_data[i * bus_dim_ + d];
                }
                proj_query[d] = sum;
            }

            // Process each slot
            for (int64_t s = 0; s < num_slots_; ++s) {
                const float* slot_buf = buf_data +
                    ((b * num_slots_ + s) * superdim_) * bus_dim_;
                float* slot_out = out_data +
                    (b * num_slots_ + s) * bus_dim_;

                // Compute attention scores over superposition dim
                float scale = 1.0f / std::sqrt(static_cast<float>(bus_dim_));
                for (int64_t d = 0; d < superdim_; ++d) {
                    const float* super_vec = slot_buf + d * bus_dim_;
                    scores[d] = saguaro::ops::simd_dot(
                        proj_query.data(), super_vec, bus_dim_
                    ) * scale;
                }

                // Softmax with temperature
                saguaro::ops::superposition_softmax(
                    scores.data(), superdim_, temperature_
                );

                // Weighted sum of superposition vectors
                saguaro::ops::simd_weighted_sum(
                    slot_buf, scores.data(), slot_out, superdim_, bus_dim_
                );
            }
        }
    }

 private:
    int64_t num_slots_;
    int64_t superdim_;
    int64_t bus_dim_;
    float temperature_;
};

REGISTER_KERNEL_BUILDER(
    Name("SuperpositionCollapseRead").Device(DEVICE_CPU),
    SuperpositionCollapseReadOp);

// =============================================================================
// KERNEL: SuperpositionWrite
// =============================================================================

class SuperpositionWriteOp : public OpKernel {
 public:
    explicit SuperpositionWriteOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_slots", &num_slots_));
        OP_REQUIRES_OK(context, context->GetAttr("superposition_dim", &superdim_));
        OP_REQUIRES_OK(context, context->GetAttr("bus_dim", &bus_dim_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& content = context->input(0);
        const Tensor& gate = context->input(1);
        const Tensor& buffer = context->input(2);

        const int64_t batch = content.dim_size(0);

        // Allocate output (copy of input, then modify)
        Tensor* buffer_new = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, buffer.shape(), &buffer_new));

        const float* c_data = content.flat<float>().data();
        const float* g_data = gate.flat<float>().data();
        const float* buf_data = buffer.flat<float>().data();
        float* out_data = buffer_new->flat<float>().data();

        // Process each batch, slot, superposition dim
        for (int64_t b = 0; b < batch; ++b) {
            const float* c_batch = c_data + b * bus_dim_;

            for (int64_t s = 0; s < num_slots_; ++s) {
                float g = g_data[b * num_slots_ + s];
                float one_minus_g = 1.0f - g;

                for (int64_t d = 0; d < superdim_; ++d) {
                    int64_t offset = ((b * num_slots_ + s) * superdim_ + d) * bus_dim_;
                    const float* old_vec = buf_data + offset;
                    float* new_vec = out_data + offset;

                    // Blended update: new = old * (1-g) + content * g
                    saguaro::ops::simd_scaled_add(
                        old_vec, one_minus_g,
                        c_batch, g,
                        new_vec, bus_dim_
                    );
                }
            }
        }
    }

 private:
    int64_t num_slots_;
    int64_t superdim_;
    int64_t bus_dim_;
};

REGISTER_KERNEL_BUILDER(
    Name("SuperpositionWrite").Device(DEVICE_CPU),
    SuperpositionWriteOp);

// =============================================================================
// GRADIENT OP REGISTRATION: SuperpositionCollapseReadGrad
// =============================================================================

REGISTER_OP("SuperpositionCollapseReadGrad")
    .Input("grad_collapsed: float32")  // [batch, num_slots, bus_dim]
    .Input("query: float32")           // [batch, query_dim]
    .Input("buffer: float32")          // [batch, num_slots, superposition_dim, bus_dim]
    .Input("collapse_weight: float32") // [query_dim, bus_dim]
    .Input("collapse_bias: float32")   // [bus_dim]
    .Output("grad_query: float32")
    .Output("grad_buffer: float32")
    .Output("grad_weight: float32")
    .Output("grad_bias: float32")
    .Attr("num_slots: int")
    .Attr("superposition_dim: int")
    .Attr("bus_dim: int")
    .Attr("temperature: float = 1.0")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_query
        c->set_output(1, c->input(2));  // grad_buffer
        c->set_output(2, c->input(3));  // grad_weight
        c->set_output(3, c->input(4));  // grad_bias
        return OkStatus();
    });

// =============================================================================
// KERNEL: SuperpositionCollapseReadGrad
// =============================================================================

class SuperpositionCollapseReadGradOp : public OpKernel {
 public:
    explicit SuperpositionCollapseReadGradOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_slots", &num_slots_));
        OP_REQUIRES_OK(context, context->GetAttr("superposition_dim", &superdim_));
        OP_REQUIRES_OK(context, context->GetAttr("bus_dim", &bus_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_collapsed = context->input(0);
        const Tensor& query = context->input(1);
        const Tensor& buffer = context->input(2);
        const Tensor& collapse_weight = context->input(3);
        const Tensor& collapse_bias = context->input(4);

        const int64_t batch = query.dim_size(0);
        const int64_t query_dim = query.dim_size(1);

        // Allocate outputs
        Tensor* grad_query = nullptr;
        Tensor* grad_buffer = nullptr;
        Tensor* grad_weight = nullptr;
        Tensor* grad_bias = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, query.shape(), &grad_query));
        OP_REQUIRES_OK(context, context->allocate_output(1, buffer.shape(), &grad_buffer));
        OP_REQUIRES_OK(context, context->allocate_output(2, collapse_weight.shape(), &grad_weight));
        OP_REQUIRES_OK(context, context->allocate_output(3, collapse_bias.shape(), &grad_bias));

        // Zero initialize outputs
        auto gq_flat = grad_query->flat<float>();
        auto gb_flat = grad_buffer->flat<float>();
        auto gw_flat = grad_weight->flat<float>();
        auto gbias_flat = grad_bias->flat<float>();

        std::fill_n(gq_flat.data(), gq_flat.size(), 0.0f);
        std::fill_n(gb_flat.data(), gb_flat.size(), 0.0f);
        std::fill_n(gw_flat.data(), gw_flat.size(), 0.0f);
        std::fill_n(gbias_flat.data(), gbias_flat.size(), 0.0f);

        // Get data pointers
        const float* grad_c = grad_collapsed.flat<float>().data();
        const float* q_data = query.flat<float>().data();
        const float* buf_data = buffer.flat<float>().data();
        const float* w_data = collapse_weight.flat<float>().data();

        float* gq_data = gq_flat.data();
        float* gb_data = gb_flat.data();
        float* gw_data = gw_flat.data();
        float* gbias_data = gbias_flat.data();

        // Temp buffers
        std::vector<float> proj_query(bus_dim_);
        std::vector<float> scores(superdim_);

        // Forward recompute + backward pass
        for (int64_t b = 0; b < batch; ++b) {
            const float* q_batch = q_data + b * query_dim;
            float* gq_batch = gq_data + b * query_dim;

            // Recompute projected query
            for (int64_t d = 0; d < bus_dim_; ++d) {
                float sum = 0.0f;
                for (int64_t i = 0; i < query_dim; ++i) {
                    sum += q_batch[i] * w_data[i * bus_dim_ + d];
                }
                proj_query[d] = sum;
            }

            for (int64_t s = 0; s < num_slots_; ++s) {
                const float* slot_buf = buf_data +
                    ((b * num_slots_ + s) * superdim_) * bus_dim_;
                float* slot_gb = gb_data +
                    ((b * num_slots_ + s) * superdim_) * bus_dim_;
                const float* grad_slot = grad_c + (b * num_slots_ + s) * bus_dim_;

                // Recompute scores + softmax
                float scale = 1.0f / std::sqrt(static_cast<float>(bus_dim_));
                for (int64_t d = 0; d < superdim_; ++d) {
                    const float* super_vec = slot_buf + d * bus_dim_;
                    scores[d] = saguaro::ops::simd_dot(
                        proj_query.data(), super_vec, bus_dim_
                    ) * scale;
                }
                saguaro::ops::superposition_softmax(scores.data(), superdim_, temperature_);

                // Backward through weighted sum: grad_buffer contribution
                for (int64_t d = 0; d < superdim_; ++d) {
                    float* super_gb = slot_gb + d * bus_dim_;
                    for (int64_t v = 0; v < bus_dim_; ++v) {
                        super_gb[v] += scores[d] * grad_slot[v];
                    }
                }

                // Backward through softmax + attention (simplified)
                // This is an approximation; full gradient needs jacobian
                // For now, use chain rule approximation
                std::vector<float> grad_scores(superdim_, 0.0f);
                for (int64_t d = 0; d < superdim_; ++d) {
                    const float* super_vec = slot_buf + d * bus_dim_;
                    grad_scores[d] = saguaro::ops::simd_dot(
                        grad_slot, super_vec, bus_dim_
                    );
                }

                // Softmax gradient: ds = s * (dL - sum(s * dL))
                float sum_s_ds = 0.0f;
                for (int64_t d = 0; d < superdim_; ++d) {
                    sum_s_ds += scores[d] * grad_scores[d];
                }
                for (int64_t d = 0; d < superdim_; ++d) {
                    grad_scores[d] = scores[d] * (grad_scores[d] - sum_s_ds) * scale / temperature_;
                }

                // Backward through attention to query
                std::vector<float> grad_proj_q(bus_dim_, 0.0f);
                for (int64_t d = 0; d < superdim_; ++d) {
                    const float* super_vec = slot_buf + d * bus_dim_;
                    for (int64_t v = 0; v < bus_dim_; ++v) {
                        grad_proj_q[v] += grad_scores[d] * super_vec[v];
                    }
                }

                // Backward through projection to query
                for (int64_t i = 0; i < query_dim; ++i) {
                    for (int64_t d = 0; d < bus_dim_; ++d) {
                        gq_batch[i] += grad_proj_q[d] * w_data[i * bus_dim_ + d];
                        gw_data[i * bus_dim_ + d] += grad_proj_q[d] * q_batch[i];
                    }
                }

                // Bias gradient
                for (int64_t d = 0; d < bus_dim_; ++d) {
                    gbias_data[d] += grad_proj_q[d];
                }
            }
        }
    }

 private:
    int64_t num_slots_;
    int64_t superdim_;
    int64_t bus_dim_;
    float temperature_;
};

REGISTER_KERNEL_BUILDER(
    Name("SuperpositionCollapseReadGrad").Device(DEVICE_CPU),
    SuperpositionCollapseReadGradOp);

}  // namespace tensorflow
