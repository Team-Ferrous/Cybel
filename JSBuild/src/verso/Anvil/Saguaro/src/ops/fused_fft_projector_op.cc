// saguaro/native/ops/fused_fft_projector_op.cc
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "fused_fft_projector_op.h"

using namespace tensorflow;

REGISTER_OP("FFTProjectorForward")
    .Input("state: float")           // [total_paths, dim] or [total_paths, 2, dim] if persistent
    .Input("freq_weights_1: float")  // [2, dim]
    .Input("bias_1: float")          // [dim]
    .Input("freq_weights_2: float")  // [2, dim]
    .Input("bias_2: float")          // [dim]
    .Input("norm_gamma: float")      // [dim]
    .Input("norm_beta: float")       // [dim]
    .Output("output: float")         // [total_paths, dim]
    .Attr("dim: int")
    .Attr("input_is_freq: bool = false")
    .Attr("output_is_freq: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int64_t dim;
        c->GetAttr("dim", &dim);
        bool in_f, out_f;
        c->GetAttr("input_is_freq", &in_f);
        c->GetAttr("output_is_freq", &out_f);

        shape_inference::ShapeHandle input = c->input(0);
        if (out_f && !in_f) {
            // [B, D] -> [B, 2, D] (interpreted as 2*D)
            c->set_output(0, c->Matrix(c->Dim(input, 0), 2 * dim));
        } else if (!out_f && in_f) {
            // [B, 2*D] -> [B, D]
            c->set_output(0, c->Matrix(c->Dim(input, 0), dim));
        } else {
            c->set_output(0, input);
        }
        return Status();
    });

class FFTProjectorForwardOp : public OpKernel {
 public:
  explicit FFTProjectorForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
    OP_REQUIRES_OK(context, context->GetAttr("input_is_freq", &input_is_freq_));
    OP_REQUIRES_OK(context, context->GetAttr("output_is_freq", &output_is_freq_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& state = context->input(0);
    const Tensor& fw1 = context->input(1);
    const Tensor& b1 = context->input(2);
    const Tensor& fw2 = context->input(3);
    const Tensor& b2 = context->input(4);
    const Tensor& gamma = context->input(5);
    const Tensor& beta = context->input(6);

    const int64_t total_paths = state.dim_size(0);
    
    TensorShape out_shape = state.shape();
    if (output_is_freq_ && !input_is_freq_) {
        out_shape = TensorShape({total_paths, 2 * dim_});
    } else if (!output_is_freq_ && input_is_freq_) {
        out_shape = TensorShape({total_paths, dim_});
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    
    float* out_ptr = output->flat<float>().data();
    const float* in_ptr = state.flat<float>().data();

    // The kernel is currently written to be mostly in-place if shapes match.
    // If not, we might need to handle it. 
    // BUT the fft_projector_forward takes a 'state' ptr which it assumes 
    // is large enough or it uses scratch.
    
    // For simplicity, let's copy input to output if they differ in content but not ptr?
    // Actually, let's just use the output as the scratch space and copy input there first.
    if (input_is_freq_ && !output_is_freq_) {
        // [B, 2*D] -> [B, D]
        // We can't use output as in-place because it's smaller.
        // We need a temp buffer for freq.
        std::vector<float> temp_freq(total_paths * 2 * dim_);
        std::memcpy(temp_freq.data(), in_ptr, state.NumElements() * sizeof(float));
        
        saguaro::ops::fft_projector_forward(
            temp_freq.data(),
            fw1.flat<float>().data(),
            b1.flat<float>().data(),
            fw2.flat<float>().data(),
            b2.flat<float>().data(),
            gamma.flat<float>().data(),
            beta.flat<float>().data(),
            total_paths,
            dim_,
            true, // input is freq
            false // output is spatial
        );
        // Copy spatial result to output
        for (int64_t i = 0; i < total_paths; ++i) {
            std::memcpy(out_ptr + i * dim_, temp_freq.data() + i * dim_, dim_ * sizeof(float));
        }
    } else if (!input_is_freq_ && output_is_freq_) {
        // [B, D] -> [B, 2*D]
        // Use output as freq buffer, copy spatial input into its real parts
        for (int64_t i = 0; i < total_paths; ++i) {
            for (int64_t d = 0; d < dim_; ++d) {
                out_ptr[i * 2 * dim_ + 2 * d] = in_ptr[i * dim_ + d];
                out_ptr[i * 2 * dim_ + 2 * d + 1] = 0.0f;
            }
        }
        saguaro::ops::fft_projector_forward(
            out_ptr,
            fw1.flat<float>().data(),
            b1.flat<float>().data(),
            fw2.flat<float>().data(),
            b2.flat<float>().data(),
            gamma.flat<float>().data(),
            beta.flat<float>().data(),
            total_paths,
            dim_,
            true, // now interpreted as freq
            true // output as freq
        );
    } else {
        // Shapes match (D->D or 2D->2D)
        std::memcpy(out_ptr, in_ptr, state.NumElements() * sizeof(float));
        saguaro::ops::fft_projector_forward(
            out_ptr,
            fw1.flat<float>().data(),
            b1.flat<float>().data(),
            fw2.flat<float>().data(),
            b2.flat<float>().data(),
            gamma.flat<float>().data(),
            beta.flat<float>().data(),
            total_paths,
            dim_,
            input_is_freq_,
            output_is_freq_
        );
    }
  }

 private:
  int dim_;
  bool input_is_freq_;
  bool output_is_freq_;
};

REGISTER_KERNEL_BUILDER(Name("FFTProjectorForward").Device(DEVICE_CPU), FFTProjectorForwardOp);
