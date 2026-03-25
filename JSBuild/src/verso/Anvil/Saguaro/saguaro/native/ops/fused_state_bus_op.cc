// saguaro.native/ops/fused_state_bus_op.cc
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
 * @file fused_state_bus_op.cc
 * @brief Fused State Bus TensorFlow custom op.
 *
 * Implements read (attention over slots) and gated write for a fixed-size
 * cross-block communication bus. Scalar/AVX2-safe implementation only.
 */

#include "fused_state_bus_op.h"

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
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedStateBus")
    .Input("query: float32")              // [batch, dim]
    .Input("write_value: float32")        // [batch, dim]
    .Input("slots: float32")              // [batch, num_slots, bus_dim]
    .Input("read_query_weight: float32")  // [dim, bus_dim]
    .Input("read_query_bias: float32")    // [bus_dim]
    .Input("write_gate_weight: float32")  // [dim, num_slots]
    .Input("write_gate_bias: float32")    // [num_slots]
    .Input("write_value_weight: float32") // [dim, bus_dim]
    .Input("write_value_bias: float32")   // [bus_dim]
    .Output("context: float32")           // [batch, bus_dim]
    .Output("slots_new: float32")         // [batch, num_slots, bus_dim]
    .Attr("num_slots: int")
    .Attr("bus_dim: int")
    .Attr("write_enabled: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle query_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &query_shape));
        DimensionHandle batch_dim = c->Dim(query_shape, 0);
        int64_t bus_dim = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("bus_dim", &bus_dim));

        c->set_output(0, c->MakeShape({batch_dim, bus_dim}));
        c->set_output(1, c->input(2));
        return OkStatus();
    })
    .Doc(R"doc(
Fused State Bus operation.
Performs attention-based read over fixed slots and gated write updates.
)doc");

REGISTER_OP("FusedStateBusGrad")
    .Input("grad_context: float32")
    .Input("grad_slots_new: float32")
    .Input("query: float32")
    .Input("write_value: float32")
    .Input("slots: float32")
    .Input("read_query_weight: float32")
    .Input("read_query_bias: float32")
    .Input("write_gate_weight: float32")
    .Input("write_gate_bias: float32")
    .Input("write_value_weight: float32")
    .Input("write_value_bias: float32")
    .Output("grad_query: float32")
    .Output("grad_write_value: float32")
    .Output("grad_slots: float32")
    .Output("grad_read_query_weight: float32")
    .Output("grad_read_query_bias: float32")
    .Output("grad_write_gate_weight: float32")
    .Output("grad_write_gate_bias: float32")
    .Output("grad_write_value_weight: float32")
    .Output("grad_write_value_bias: float32")
    .Attr("num_slots: int")
    .Attr("bus_dim: int")
    .Attr("write_enabled: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(2));
        c->set_output(1, c->input(3));
        c->set_output(2, c->input(4));
        c->set_output(3, c->input(5));
        c->set_output(4, c->input(6));
        c->set_output(5, c->input(7));
        c->set_output(6, c->input(8));
        c->set_output(7, c->input(9));
        c->set_output(8, c->input(10));
        return OkStatus();
    });

// =============================================================================
// FORWARD KERNEL
// =============================================================================

class FusedStateBusOp : public OpKernel {
 public:
    explicit FusedStateBusOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_slots", &num_slots_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bus_dim", &bus_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("write_enabled", &write_enabled_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& query = ctx->input(0);
        const Tensor& write_value = ctx->input(1);
        const Tensor& slots = ctx->input(2);
        const Tensor& read_query_weight = ctx->input(3);
        const Tensor& read_query_bias = ctx->input(4);
        const Tensor& write_gate_weight = ctx->input(5);
        const Tensor& write_gate_bias = ctx->input(6);
        const Tensor& write_value_weight = ctx->input(7);
        const Tensor& write_value_bias = ctx->input(8);

        const int64_t batch_size = query.dim_size(0);
        const int64_t query_dim = query.dim_size(1);
        const int64_t slots_dim0 = slots.dim_size(0);
        const int64_t slots_dim1 = slots.dim_size(1);
        const int64_t slots_dim2 = slots.dim_size(2);

        OP_REQUIRES(ctx, slots_dim0 == batch_size,
                    errors::InvalidArgument("slots batch mismatch"));
        OP_REQUIRES(ctx, slots_dim1 == num_slots_,
                    errors::InvalidArgument("slots num_slots mismatch"));
        OP_REQUIRES(ctx, slots_dim2 == bus_dim_,
                    errors::InvalidArgument("slots bus_dim mismatch"));
        OP_REQUIRES(ctx, read_query_weight.dim_size(0) == query_dim,
                    errors::InvalidArgument("read_query_weight dim mismatch"));
        OP_REQUIRES(ctx, read_query_weight.dim_size(1) == bus_dim_,
                    errors::InvalidArgument("read_query_weight bus_dim mismatch"));
        OP_REQUIRES(ctx, write_value_weight.dim_size(0) == query_dim,
                    errors::InvalidArgument("write_value_weight dim mismatch"));
        OP_REQUIRES(ctx, write_value_weight.dim_size(1) == bus_dim_,
                    errors::InvalidArgument("write_value_weight bus_dim mismatch"));
        OP_REQUIRES(ctx, write_gate_weight.dim_size(0) == query_dim,
                    errors::InvalidArgument("write_gate_weight dim mismatch"));
        OP_REQUIRES(ctx, write_gate_weight.dim_size(1) == num_slots_,
                    errors::InvalidArgument("write_gate_weight num_slots mismatch"));

        Tensor* context = nullptr;
        Tensor* slots_new = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, bus_dim_}), &context));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, slots.shape(), &slots_new));

        const float* query_data = query.flat<float>().data();
        const float* write_data = write_value.flat<float>().data();
        const float* slots_data = slots.flat<float>().data();
        const float* rq_w = read_query_weight.flat<float>().data();
        const float* rq_b = read_query_bias.flat<float>().data();
        const float* wg_w = write_gate_weight.flat<float>().data();
        const float* wg_b = write_gate_bias.flat<float>().data();
        const float* wv_w = write_value_weight.flat<float>().data();
        const float* wv_b = write_value_bias.flat<float>().data();

        float* context_data = context->flat<float>().data();
        float* slots_new_data = slots_new->flat<float>().data();

        std::vector<float> projected_query(batch_size * bus_dim_);
        std::vector<float> write_content;
        std::vector<float> write_gate;
        if (write_enabled_) {
            write_content.resize(batch_size * bus_dim_);
            write_gate.resize(batch_size * num_slots_);
        }

        // Project queries
        for (int64_t b = 0; b < batch_size; ++b) {
            const float* query_row = query_data + b * query_dim;
            float* q_out = projected_query.data() + b * bus_dim_;
            for (int64_t d = 0; d < bus_dim_; ++d) {
                float sum = rq_b[d];
                for (int64_t k = 0; k < query_dim; ++k) {
                    sum += query_row[k] * rq_w[k * bus_dim_ + d];
                }
                q_out[d] = sum;
            }
        }

        if (write_enabled_) {
            // Project write content
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* write_row = write_data + b * query_dim;
                float* content_out = write_content.data() + b * bus_dim_;
                for (int64_t d = 0; d < bus_dim_; ++d) {
                    float sum = wv_b[d];
                    for (int64_t k = 0; k < query_dim; ++k) {
                        sum += write_row[k] * wv_w[k * bus_dim_ + d];
                    }
                    content_out[d] = sum;
                }
            }

            // Compute write gates
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* write_row = write_data + b * query_dim;
                float* gate_out = write_gate.data() + b * num_slots_;
                for (int64_t s = 0; s < num_slots_; ++s) {
                    float sum = wg_b[s];
                    for (int64_t k = 0; k < query_dim; ++k) {
                        sum += write_row[k] * wg_w[k * num_slots_ + s];
                    }
                    gate_out[s] = sum;
                }
            }
            saguaro::ops::statebus_sigmoid_inplace(
                write_gate.data(), static_cast<int64_t>(write_gate.size()));
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(bus_dim_));

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* q = projected_query.data() + b * bus_dim_;
            const float* slots_row = slots_data + b * num_slots_ * bus_dim_;
            float* context_row = context_data + b * bus_dim_;

            std::vector<float> scores(num_slots_);
            for (int64_t s = 0; s < num_slots_; ++s) {
                const float* slot = slots_row + s * bus_dim_;
                scores[s] = saguaro::ops::statebus_dot(q, slot, bus_dim_) * scale;
            }
            saguaro::ops::statebus_softmax(scores.data(), num_slots_);

            for (int64_t d = 0; d < bus_dim_; ++d) {
                context_row[d] = 0.0f;
            }
            for (int64_t s = 0; s < num_slots_; ++s) {
                const float weight = scores[s];
                const float* slot = slots_row + s * bus_dim_;
                for (int64_t d = 0; d < bus_dim_; ++d) {
                    context_row[d] += weight * slot[d];
                }
            }

            float* slots_new_row = slots_new_data + b * num_slots_ * bus_dim_;
            if (!write_enabled_) {
                std::copy(slots_row, slots_row + num_slots_ * bus_dim_, slots_new_row);
                continue;
            }

            const float* content_row = write_content.data() + b * bus_dim_;
            const float* gate_row = write_gate.data() + b * num_slots_;
            for (int64_t s = 0; s < num_slots_; ++s) {
                const float gate = gate_row[s];
                const float* slot = slots_row + s * bus_dim_;
                float* slot_out = slots_new_row + s * bus_dim_;
                for (int64_t d = 0; d < bus_dim_; ++d) {
                    slot_out[d] = slot[d] * (1.0f - gate) + content_row[d] * gate;
                }
            }
        }
    }

 private:
    int num_slots_;
    int bus_dim_;
    bool write_enabled_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedStateBus").Device(DEVICE_CPU),
    FusedStateBusOp);

// =============================================================================
// GRADIENT KERNEL
// =============================================================================

class FusedStateBusGradOp : public OpKernel {
 public:
    explicit FusedStateBusGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& query = ctx->input(2);
        const Tensor& write_value = ctx->input(3);
        const Tensor& slots = ctx->input(4);
        const Tensor& read_query_weight = ctx->input(5);
        const Tensor& read_query_bias = ctx->input(6);
        const Tensor& write_gate_weight = ctx->input(7);
        const Tensor& write_gate_bias = ctx->input(8);
        const Tensor& write_value_weight = ctx->input(9);
        const Tensor& write_value_bias = ctx->input(10);

        Tensor* grad_query = nullptr;
        Tensor* grad_write_value = nullptr;
        Tensor* grad_slots = nullptr;
        Tensor* grad_read_query_weight = nullptr;
        Tensor* grad_read_query_bias = nullptr;
        Tensor* grad_write_gate_weight = nullptr;
        Tensor* grad_write_gate_bias = nullptr;
        Tensor* grad_write_value_weight = nullptr;
        Tensor* grad_write_value_bias = nullptr;

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &grad_query));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, write_value.shape(), &grad_write_value));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, slots.shape(), &grad_slots));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, read_query_weight.shape(), &grad_read_query_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, read_query_bias.shape(), &grad_read_query_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(5, write_gate_weight.shape(), &grad_write_gate_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(6, write_gate_bias.shape(), &grad_write_gate_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(7, write_value_weight.shape(), &grad_write_value_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(8, write_value_bias.shape(), &grad_write_value_bias));

        std::fill(grad_query->flat<float>().data(),
                  grad_query->flat<float>().data() + grad_query->NumElements(), 0.0f);
        std::fill(grad_write_value->flat<float>().data(),
                  grad_write_value->flat<float>().data() + grad_write_value->NumElements(), 0.0f);
        std::fill(grad_slots->flat<float>().data(),
                  grad_slots->flat<float>().data() + grad_slots->NumElements(), 0.0f);
        std::fill(grad_read_query_weight->flat<float>().data(),
                  grad_read_query_weight->flat<float>().data()
                      + grad_read_query_weight->NumElements(), 0.0f);
        std::fill(grad_read_query_bias->flat<float>().data(),
                  grad_read_query_bias->flat<float>().data()
                      + grad_read_query_bias->NumElements(), 0.0f);
        std::fill(grad_write_gate_weight->flat<float>().data(),
                  grad_write_gate_weight->flat<float>().data()
                      + grad_write_gate_weight->NumElements(), 0.0f);
        std::fill(grad_write_gate_bias->flat<float>().data(),
                  grad_write_gate_bias->flat<float>().data()
                      + grad_write_gate_bias->NumElements(), 0.0f);
        std::fill(grad_write_value_weight->flat<float>().data(),
                  grad_write_value_weight->flat<float>().data()
                      + grad_write_value_weight->NumElements(), 0.0f);
        std::fill(grad_write_value_bias->flat<float>().data(),
                  grad_write_value_bias->flat<float>().data()
                      + grad_write_value_bias->NumElements(), 0.0f);
    }
};

REGISTER_KERNEL_BUILDER(
    Name("FusedStateBusGrad").Device(DEVICE_CPU),
    FusedStateBusGradOp);

}  // namespace tensorflow
