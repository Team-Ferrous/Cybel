// saguaro.native/ops/cayley_transform_op.cc
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
 * @file cayley_transform_op.cc
 * @brief TensorFlow op registration for CayleyDense layer.
 *
 * Registers custom ops for Cayley-parameterized orthogonal dense layer:
 * - CayleyDenseForward: Forward pass with cached weights
 * - CayleyDenseBackward: Gradient computation
 *
 * SUPERSEDES: Pure Python _cayley_transform() in cayley_weights.py
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "cayley_transform_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: CayleyDenseForward
// =============================================================================

REGISTER_OP("CayleyDenseForward")
    .Input("input: float")              // [batch, input_dim]
    .Input("skew_params: float")        // [n*(n-1)/2]
    .Input("proj_weight: float")        // [input_dim, output_dim] or empty for square
    .Input("bias: float")               // [output_dim] or empty
    .Output("output: float")            // [batch, output_dim]
    .Attr("input_dim: int")
    .Attr("output_dim: int")
    .Attr("use_bias: bool = true")
    .Attr("training: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int output_dim;
        c->GetAttr("output_dim", &output_dim);
        
        shape_inference::ShapeHandle input_shape = c->input(0);
        if (c->RankKnown(input_shape) && c->Rank(input_shape) == 2) {
            auto batch = c->Dim(input_shape, 0);
            c->set_output(0, c->MakeShape({batch, output_dim}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
CayleyDense Forward Pass.

Computes dense layer with Cayley-parameterized orthogonal weights.
For square matrices: W = (I - A)(I + A)^{-1} where A is skew-symmetric.
For rectangular: Uses projection weight directly.

Guarantees orthogonality for square case, improving gradient stability.
)doc");

// =============================================================================
// KERNEL: CayleyDenseForward
// =============================================================================

class CayleyDenseForwardOp : public OpKernel {
 public:
    explicit CayleyDenseForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("input_dim", &input_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dim", &output_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("use_bias", &use_bias_));
        OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
        
        // Pre-allocate cache for inference
        if (!training_) {
            int64_t min_dim = std::min(input_dim_, output_dim_);
            cached_W_.resize(min_dim * min_dim, 0.0f);
            cache_valid_ = false;
        }
    }

    void Compute(OpKernelContext* context) override {
        // Get input tensors
        const Tensor& input = context->input(0);
        const Tensor& skew_params = context->input(1);
        const Tensor& proj_weight = context->input(2);
        const Tensor& bias = context->input(3);

        // Validate shapes
        OP_REQUIRES(context, input.dims() == 2,
                    errors::InvalidArgument("input must be 2D [batch, input_dim]"));
        
        const int64_t batch_size = input.dim_size(0);
        const int64_t actual_input_dim = input.dim_size(1);
        
        OP_REQUIRES(context, actual_input_dim == input_dim_,
                    errors::InvalidArgument("input_dim mismatch: ", 
                                            actual_input_dim, " vs ", input_dim_));

        // Allocate output
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
            0, TensorShape({batch_size, output_dim_}), &output));

        // Get data pointers
        const float* input_data = input.flat<float>().data();
        const float* skew_data = skew_params.flat<float>().data();
        const float* proj_data = proj_weight.NumElements() > 0 ? 
                                  proj_weight.flat<float>().data() : nullptr;
        const float* bias_data = (use_bias_ && bias.NumElements() > 0) ? 
                                  bias.flat<float>().data() : nullptr;
        float* output_data = output->flat<float>().data();

        // Use cached weight for inference
        float* cache_ptr = (!training_ && !cached_W_.empty()) ? 
                            cached_W_.data() : nullptr;

        // Call optimized kernel
        saguaro::ops::cayley::cayley_dense_forward(
            input_data,
            skew_data,
            proj_data,
            bias_data,
            output_data,
            batch_size,
            input_dim_,
            output_dim_,
            training_,
            cache_ptr
        );
    }

 private:
    int64_t input_dim_;
    int64_t output_dim_;
    bool use_bias_;
    bool training_;
    std::vector<float> cached_W_;
    bool cache_valid_ = false;
};

REGISTER_KERNEL_BUILDER(Name("CayleyDenseForward").Device(DEVICE_CPU),
                        CayleyDenseForwardOp);

// =============================================================================
// OP REGISTRATION: CayleyDenseBackward
// =============================================================================

REGISTER_OP("CayleyDenseBackward")
    .Input("grad_output: float")        // [batch, output_dim]
    .Input("input: float")              // [batch, input_dim]
    .Input("skew_params: float")        // [n*(n-1)/2]
    .Input("proj_weight: float")        // [input_dim, output_dim] or empty
    .Output("grad_input: float")        // [batch, input_dim]
    .Output("grad_skew_params: float")  // [n*(n-1)/2]
    .Output("grad_proj_weight: float")  // [input_dim, output_dim] or empty
    .Output("grad_bias: float")         // [output_dim]
    .Attr("input_dim: int")
    .Attr("output_dim: int")
    .Attr("use_bias: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // grad_input same as input
        c->set_output(0, c->input(1));
        // grad_skew_params same as skew_params
        c->set_output(1, c->input(2));
        // grad_proj_weight same as proj_weight
        c->set_output(2, c->input(3));
        
        int output_dim;
        c->GetAttr("output_dim", &output_dim);
        // grad_bias
        c->set_output(3, c->MakeShape({output_dim}));
        
        return Status();
    })
    .Doc("CayleyDense Backward Pass - Gradient computation for all parameters.");

// =============================================================================
// KERNEL: CayleyDenseBackward
// =============================================================================

class CayleyDenseBackwardOp : public OpKernel {
 public:
    explicit CayleyDenseBackwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("input_dim", &input_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dim", &output_dim_));
        OP_REQUIRES_OK(context, context->GetAttr("use_bias", &use_bias_));
    }

    void Compute(OpKernelContext* context) override {
        // Get input tensors
        const Tensor& grad_output = context->input(0);
        const Tensor& input = context->input(1);
        const Tensor& skew_params = context->input(2);
        const Tensor& proj_weight = context->input(3);

        const int64_t batch_size = input.dim_size(0);

        // Allocate gradient tensors
        Tensor* grad_input = nullptr;
        Tensor* grad_skew = nullptr;
        Tensor* grad_proj = nullptr;
        Tensor* grad_bias = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(
            0, input.shape(), &grad_input));
        OP_REQUIRES_OK(context, context->allocate_output(
            1, skew_params.shape(), &grad_skew));
        
        // For rectangular case, allocate proj gradient
        if (proj_weight.NumElements() > 0) {
            OP_REQUIRES_OK(context, context->allocate_output(
                2, proj_weight.shape(), &grad_proj));
        } else {
            OP_REQUIRES_OK(context, context->allocate_output(
                2, TensorShape({0}), &grad_proj));
        }
        
        OP_REQUIRES_OK(context, context->allocate_output(
            3, TensorShape({output_dim_}), &grad_bias));

        // Get data pointers
        const float* grad_out_data = grad_output.flat<float>().data();
        const float* input_data = input.flat<float>().data();
        const float* skew_data = skew_params.flat<float>().data();
        const float* proj_data = proj_weight.NumElements() > 0 ?
                                  proj_weight.flat<float>().data() : nullptr;

        float* grad_input_data = grad_input->flat<float>().data();
        float* grad_skew_data = grad_skew->flat<float>().data();
        float* grad_proj_data = grad_proj->NumElements() > 0 ?
                                 grad_proj->flat<float>().data() : nullptr;
        float* grad_bias_data = use_bias_ ? grad_bias->flat<float>().data() : nullptr;

        // Call optimized kernel
        saguaro::ops::cayley::cayley_dense_backward(
            grad_out_data,
            input_data,
            skew_data,
            proj_data,
            grad_input_data,
            grad_skew_data,
            grad_proj_data,
            grad_bias_data,
            batch_size,
            input_dim_,
            output_dim_
        );
    }

 private:
    int64_t input_dim_;
    int64_t output_dim_;
    bool use_bias_;
};

REGISTER_KERNEL_BUILDER(Name("CayleyDenseBackward").Device(DEVICE_CPU),
                        CayleyDenseBackwardOp);
