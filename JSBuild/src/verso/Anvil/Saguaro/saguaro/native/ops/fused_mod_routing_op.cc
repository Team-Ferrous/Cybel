// saguaro.native/ops/fused_mod_routing_op.cc
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
 * @file fused_mod_routing_op.cc
 * @brief TensorFlow custom ops for Mixture-of-Depths routing.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "fused_mod_routing_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// MOD ROUTE OP - Computes routing decisions
// =============================================================================

REGISTER_OP("MoDRoute")
    .Input("hidden: float32")           // [batch, seq_len, hidden_dim]
    .Input("router_weight: float32")    // [hidden_dim]
    .Input("router_bias: float32")      // scalar []
    .Attr("capacity_factor: float = 0.5")
    .Attr("use_auxiliary_loss: bool = true")
    .Attr("aux_loss_weight: float = 0.01")
    .Output("router_probs: float32")    // [batch, seq_len]
    .Output("selected_mask: int32")     // [batch, seq_len] - 1 for selected, 0 for skip
    .Output("auxiliary_loss: float32")  // [batch]
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle hidden_shape = c->input(0);
        DimensionHandle batch = c->Dim(hidden_shape, 0);
        DimensionHandle seq_len = c->Dim(hidden_shape, 1);
        
        c->set_output(0, c->MakeShape({batch, seq_len}));
        c->set_output(1, c->MakeShape({batch, seq_len}));
        c->set_output(2, c->MakeShape({batch}));
        return OkStatus();
    })
    .Doc(R"doc(\nMixture-of-Depths routing decision.\n\nComputes which tokens should be processed vs skipped at each layer.\nUses top-k selection based on router scores to maintain capacity budget.\n\nhidden: Input hidden states [batch, seq_len, hidden_dim]\nrouter_weight: Router projection weights [hidden_dim]\nrouter_bias: Router bias scalar\ncapacity_factor: Fraction of tokens to process (default 0.5)\nuse_auxiliary_loss: Whether to compute load balancing loss\naux_loss_weight: Weight for auxiliary loss\nrouter_probs: Per-token routing probabilities [batch, seq_len]\nselected_mask: Binary mask (1=process, 0=skip) [batch, seq_len]\nauxiliary_loss: Load balancing loss per batch [batch]\n)doc");

class MoDRouteOp : public OpKernel {
public:
    explicit MoDRouteOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("capacity_factor", &capacity_factor_));
        OP_REQUIRES_OK(context, context->GetAttr("use_auxiliary_loss", &use_auxiliary_loss_));
        OP_REQUIRES_OK(context, context->GetAttr("aux_loss_weight", &aux_loss_weight_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& hidden = context->input(0);
        const Tensor& router_weight = context->input(1);
        const Tensor& router_bias = context->input(2);

        OP_REQUIRES(context, hidden.dims() == 3,
            errors::InvalidArgument("Hidden must be 3D [batch, seq, dim]"));

        const int batch = hidden.dim_size(0);
        const int seq_len = hidden.dim_size(1);
        const int hidden_dim = hidden.dim_size(2);

        // Allocate outputs
        Tensor* router_probs = nullptr;
        Tensor* selected_mask = nullptr;
        Tensor* auxiliary_loss = nullptr;
        
        OP_REQUIRES_OK(context, context->allocate_output(0, {batch, seq_len}, &router_probs));
        OP_REQUIRES_OK(context, context->allocate_output(1, {batch, seq_len}, &selected_mask));
        OP_REQUIRES_OK(context, context->allocate_output(2, {batch}, &auxiliary_loss));

        const float* hidden_data = hidden.flat<float>().data();
        const float* weight_data = router_weight.flat<float>().data();
        float bias = router_bias.scalar<float>()();
        
        float* probs_data = router_probs->flat<float>().data();
        int32_t* mask_data = selected_mask->flat<int32_t>().data();
        float* loss_data = auxiliary_loss->flat<float>().data();

        saguaro::ops::MoDConfig config;
        config.capacity_factor = capacity_factor_;
        config.use_auxiliary_loss = use_auxiliary_loss_;
        config.aux_loss_weight = aux_loss_weight_;

        // Process each batch
        for (int b = 0; b < batch; ++b) {
            const float* h = hidden_data + b * seq_len * hidden_dim;
            float* p = probs_data + b * seq_len;
            int32_t* m = mask_data + b * seq_len;

            // Compute routing for this batch
            auto result = saguaro::ops::mod_route_forward(
                h, weight_data, bias, seq_len, hidden_dim, config);

            // Copy results
            std::copy(result.router_probs.begin(), result.router_probs.end(), p);
            std::copy(result.route_weights.begin(), result.route_weights.end(), m);
            loss_data[b] = result.auxiliary_loss;
        }
    }

private:
    float capacity_factor_;
    bool use_auxiliary_loss_;
    float aux_loss_weight_;
};

REGISTER_KERNEL_BUILDER(Name("MoDRoute").Device(DEVICE_CPU), MoDRouteOp);

// =============================================================================
// MOD GATHER OP - Gathers selected tokens
// =============================================================================

REGISTER_OP("MoDGather")
    .Input("hidden: float32")           // [batch, seq_len, hidden_dim]
    .Input("selected_mask: int32")      // [batch, seq_len]
    .Output("gathered: float32")        // [batch, capacity, hidden_dim]
    .Output("gather_indices: int32")    // [batch, capacity]
    .Output("num_selected: int32")      // [batch]
    .Attr("capacity: int")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle hidden_shape = c->input(0);
        DimensionHandle batch = c->Dim(hidden_shape, 0);
        DimensionHandle hidden_dim = c->Dim(hidden_shape, 2);
        
        int64_t capacity;
        c->GetAttr("capacity", &capacity);
        
        c->set_output(0, c->MakeShape({batch, capacity, hidden_dim}));
        c->set_output(1, c->MakeShape({batch, capacity}));
        c->set_output(2, c->MakeShape({batch}));
        return OkStatus();
    });

class MoDGatherOp : public OpKernel {
public:
    explicit MoDGatherOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("capacity", &capacity_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& hidden = context->input(0);
        const Tensor& selected_mask = context->input(1);

        const int batch = hidden.dim_size(0);
        const int seq_len = hidden.dim_size(1);
        const int hidden_dim = hidden.dim_size(2);

        Tensor* gathered = nullptr;
        Tensor* gather_indices = nullptr;
        Tensor* num_selected = nullptr;
        
        OP_REQUIRES_OK(context, context->allocate_output(0, {batch, capacity_, hidden_dim}, &gathered));
        OP_REQUIRES_OK(context, context->allocate_output(1, {batch, capacity_}, &gather_indices));
        OP_REQUIRES_OK(context, context->allocate_output(2, {batch}, &num_selected));

        const float* hidden_data = hidden.flat<float>().data();
        const int32_t* mask_data = selected_mask.flat<int32_t>().data();
        
        float* gathered_data = gathered->flat<float>().data();
        int32_t* indices_data = gather_indices->flat<int32_t>().data();
        int32_t* num_sel_data = num_selected->flat<int32_t>().data();

        // Initialize to zero
        std::fill_n(gathered_data, batch * capacity_ * hidden_dim, 0.0f);
        std::fill_n(indices_data, batch * capacity_, -1);

        for (int b = 0; b < batch; ++b) {
            const float* h = hidden_data + b * seq_len * hidden_dim;
            const int32_t* m = mask_data + b * seq_len;
            float* g = gathered_data + b * capacity_ * hidden_dim;
            int32_t* idx = indices_data + b * capacity_;

            int count = 0;
            for (int s = 0; s < seq_len && count < capacity_; ++s) {
                if (m[s] > 0) {
                    idx[count] = s;
                    std::copy(h + s * hidden_dim, h + (s + 1) * hidden_dim, 
                              g + count * hidden_dim);
                    count++;
                }
            }
            num_sel_data[b] = count;
        }
    }

private:
    int64_t capacity_;
};

REGISTER_KERNEL_BUILDER(Name("MoDGather").Device(DEVICE_CPU), MoDGatherOp);

// =============================================================================
// MOD SCATTER OP - Scatters processed tokens back
// =============================================================================

REGISTER_OP("MoDScatter")
    .Input("original: float32")         // [batch, seq_len, hidden_dim]
    .Input("processed: float32")        // [batch, capacity, hidden_dim]
    .Input("gather_indices: int32")     // [batch, capacity]
    .Input("router_probs: float32")     // [batch, seq_len]
    .Input("num_selected: int32")       // [batch]
    .Output("output: float32")          // [batch, seq_len, hidden_dim]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return OkStatus();
    });

class MoDScatterOp : public OpKernel {
public:
    explicit MoDScatterOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& original = context->input(0);
        const Tensor& processed = context->input(1);
        const Tensor& gather_indices = context->input(2);
        const Tensor& router_probs = context->input(3);
        const Tensor& num_selected = context->input(4);

        const int batch = original.dim_size(0);
        const int seq_len = original.dim_size(1);
        const int hidden_dim = original.dim_size(2);
        const int capacity = processed.dim_size(1);

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, original.shape(), &output));

        const float* orig_data = original.flat<float>().data();
        const float* proc_data = processed.flat<float>().data();
        const int32_t* idx_data = gather_indices.flat<int32_t>().data();
        const float* probs_data = router_probs.flat<float>().data();
        const int32_t* num_sel_data = num_selected.flat<int32_t>().data();
        float* out_data = output->flat<float>().data();

        // Copy original to output first (for tokens that weren't selected)
        std::copy(orig_data, orig_data + batch * seq_len * hidden_dim, out_data);

        // Scatter processed tokens
        for (int b = 0; b < batch; ++b) {
            const float* proc = proc_data + b * capacity * hidden_dim;
            const int32_t* idx = idx_data + b * capacity;
            const float* probs = probs_data + b * seq_len;
            float* out = out_data + b * seq_len * hidden_dim;
            int num_sel = num_sel_data[b];

            saguaro::ops::mod_scatter_add(
                proc, idx, out, probs, num_sel, hidden_dim);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("MoDScatter").Device(DEVICE_CPU), MoDScatterOp);

}  // namespace tensorflow
