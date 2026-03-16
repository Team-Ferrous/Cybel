// src/ops/fused_depthwise_conv1d.cc
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
// This custom operator implements a 1D depthwise convolution with 'causal'
// padding and its corresponding gradient for CPU execution. This is necessary
// to overcome the limitation in TensorFlow where gradients for depthwise
// convolutions are not supported on CPU.
//
// PRODUCTION-READY REFACTOR (2025-10-05):
// - Formatted to align with project standards for custom C++ operators.
// - Wrapped the entire implementation in the `tensorflow` namespace to ensure
//   proper symbol registration and prevent linkage errors in modern TF versions.
// - Added extensive comments to clarify the logic of both the forward and
//   backward passes.
// - Simplified convolution loop for clarity without changing the causal logic.
// - ADDED: Robust OP_REQUIRES checks to validate input tensor shapes and prevent
//   crashes from invalid inputs, providing clear error messages instead.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <algorithm>
#include <vector>
#include <atomic> // Use std::atomic as tbb::atomic is deprecated

// Phase 11 SIMD Guards
#if defined(__AVX512F__)
  #include <immintrin.h>  // AVX512 intrinsics
#elif defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>  // AVX2 intrinsics
#elif defined(__ARM_NEON)
  #include <arm_neon.h>   // NEON intrinsics
#endif

#include "common/parallel/parallel_backend.h"
#include "ops/common/op_validation.h"

namespace tensorflow {

namespace {
inline void AtomicAddFloat(float* target, float value) {
    auto* atomic_ptr = reinterpret_cast<std::atomic<float>*>(target);
    float expected = atomic_ptr->load(std::memory_order_relaxed);
    while (!atomic_ptr->compare_exchange_weak(
        expected, expected + value,
        std::memory_order_relaxed,
        std::memory_order_relaxed)) {
        // Retry until the value is updated atomically.
    }
}
}  // namespace

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// 1. FORWARD PASS OPERATOR
// =============================================================================

REGISTER_OP("FusedDepthwiseConv1D")
    .Input("input: float")   // Shape: [batch_size, in_width, in_channels]
    .Input("filter: float")  // Shape: [filter_width, 1, in_channels]
    .Input("bias: float")    // Shape: [in_channels]
    .Attr("stride: int")
    .Attr("padding: string")
    .Output("output: float")
    .SetShapeFn([](InferenceContext* c) {
        using shape_inference::DimensionHandle;
        ShapeHandle input;
        ShapeHandle filter;
        ShapeHandle bias;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &filter));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &bias));

        DimensionHandle channels = c->Dim(input, 2);
        TF_RETURN_IF_ERROR(c->Merge(channels, c->Dim(filter, 2), &channels));
        TF_RETURN_IF_ERROR(c->Merge(channels, c->Dim(bias, 0), &channels));

        DimensionHandle in_width = c->Dim(input, 1);
        DimensionHandle filter_width = c->Dim(filter, 0);

        int stride;
        string padding;
        TF_RETURN_IF_ERROR(c->GetAttr("stride", &stride));
        TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));
        if (stride <= 0) {
            return errors::InvalidArgument("stride must be positive, got ", stride);
        }

        DimensionHandle out_width = c->UnknownDim();
        if (c->ValueKnown(in_width) && c->ValueKnown(filter_width)) {
            const int64_t in_val = c->Value(in_width);
            const int64_t filter_val = c->Value(filter_width);
            int64_t computed = -1;
            if (padding == "VALID") {
                computed = (in_val - filter_val) / stride + 1;
            } else if (padding == "SAME") {
                computed = (in_val + stride - 1) / stride;
            } else {
                return errors::InvalidArgument(
                    "Padding must be 'VALID' or 'SAME', got ", padding);
            }
            if (computed < 0) computed = 0;
            out_width = c->MakeDim(computed);
        }

        std::vector<DimensionHandle> dims = {c->Dim(input, 0), out_width, channels};
        c->set_output(0, c->MakeShape(dims));
        return OkStatus();
    });

class FusedDepthwiseConv1DOp : public OpKernel {
public:
    explicit FusedDepthwiseConv1DOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override {
        // --- 1. Get Input Tensors ---
        const Tensor& input_tensor = context->input(0);
        const Tensor& filter_tensor = context->input(1);
        const Tensor& bias_tensor = context->input(2);

        // --- 2. Get Dimensions & Add Defensive Shape Checks ---
        OP_REQUIRES(context, input_tensor.dims() == 3,
            errors::InvalidArgument("Input must be a 3D tensor, but got shape ", input_tensor.shape().DebugString()));
        OP_REQUIRES(context, filter_tensor.dims() == 3,
            errors::InvalidArgument("Filter must be a 3D tensor, but got shape ", filter_tensor.shape().DebugString()));
        OP_REQUIRES(context, bias_tensor.dims() == 1,
            errors::InvalidArgument("Bias must be a 1D tensor, but got shape ", bias_tensor.shape().DebugString()));

        const int64_t batch_size = input_tensor.dim_size(0);
        const int64_t in_width = input_tensor.dim_size(1);
        const int64_t channels = input_tensor.dim_size(2);
        const int64_t filter_width = filter_tensor.dim_size(0);

        OP_REQUIRES_OK(context, RequireRankAndDimSize(context, filter_tensor, 3, 1, 1, "filter"));
        OP_REQUIRES_OK(context, RequireRankAndDimSize(context, filter_tensor, 3, 2, channels, "filter"));
        OP_REQUIRES(context, filter_tensor.dim_size(2) == channels,
            errors::InvalidArgument("Input channels (dim 2) and filter channels must match. Got ", channels, " and ", filter_tensor.dim_size(2)));
        OP_REQUIRES(context, bias_tensor.dim_size(0) == channels,
            errors::InvalidArgument("Bias size must match input channels. Got ", bias_tensor.dim_size(0), " and ", channels));
        OP_REQUIRES(context, in_width > 0, errors::InvalidArgument("Input width must be positive."));

        // Determine output width based on padding
        int64_t out_width;
        if (padding_ == "VALID") {
            out_width = (in_width - filter_width) / stride_ + 1;
        } else if (padding_ == "SAME") {
            out_width = (in_width + stride_ - 1) / stride_;
        } else {
            OP_REQUIRES(context, false, errors::InvalidArgument("Padding must be 'VALID' or 'SAME', but got ", padding_));
        }
        OP_REQUIRES(context, out_width > 0, errors::InvalidArgument("Calculated output width is not positive."));

        // --- 3. Allocate Output Tensor ---
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size, out_width, channels}), &output_tensor));

        // --- 4. Get Data Pointers (TTypes) ---
        auto input = input_tensor.tensor<float, 3>();
        auto filter = filter_tensor.tensor<float, 3>();
        auto bias = bias_tensor.vec<float>();
        auto output = output_tensor->tensor<float, 3>();

        // --- 5. Perform Causal Depthwise Convolution ---
        // Use the shared parallel backend for portability across vendors.
        const int64_t total_tasks = batch_size * channels;
        const std::size_t cost_per_task =
            static_cast<std::size_t>(std::max<int64_t>(1, in_width * filter_width));

        saguaro::parallel::ForShard(
            static_cast<std::size_t>(total_tasks),
            cost_per_task,
            [&](int64_t start, int64_t limit) {
            for (int64_t task_idx = start; task_idx < limit; ++task_idx) {
                int64_t b = task_idx / channels;
                int64_t c = task_idx % channels;

                for (int i = 0; i < out_width; ++i) {
                    float total_sum = bias(c);
                    int64_t input_start_idx = i * stride_; // For VALID/SAME padding
                    
#if defined(__AVX512F__)
                    __m512 sum_vec_512 = _mm512_setzero_ps();
                    const __m512i lane_offsets_512 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                                                                       7, 6, 5, 4, 3, 2, 1, 0);
                    const __m512i zero_vec_512 = _mm512_setzero_si512();
                    const __m512i width_limit_vec_512 = _mm512_set1_epi32(static_cast<int>(in_width));
                    int k_512 = 0;
                    // Process 16 filter taps at a time using masked gathers for boundary safety.
                    for (; k_512 <= filter_width - 16; k_512 += 16) {
                        const int current_input_idx = static_cast<int>(input_start_idx - k_512);
                        const __m512i base_index_vec = _mm512_set1_epi32(current_input_idx);
                        const __m512i gather_indices = _mm512_sub_epi32(base_index_vec, lane_offsets_512);
                        const __mmask16 ge_mask = _mm512_cmp_epi32_mask(gather_indices, zero_vec_512, _MM_CMPINT_GE);
                        const __mmask16 lt_mask = _mm512_cmp_epi32_mask(gather_indices, width_limit_vec_512, _MM_CMPINT_LT);
                        const __mmask16 lane_mask = ge_mask & lt_mask;
                        if (lane_mask == 0) {
                            continue;
                        }
                        __m512 filter_vals_512 = _mm512_loadu_ps(&filter(k_512, 0, c));
                        __m512 input_vals_512 = _mm512_mask_i32gather_ps(
                            _mm512_setzero_ps(),
                            lane_mask,
                            gather_indices,
                            &input(b, 0, c),
                            sizeof(float));
                        sum_vec_512 = _mm512_fmadd_ps(filter_vals_512, input_vals_512, sum_vec_512);
                    }
                    // Horizontal sum for AVX512
                    total_sum += _mm512_reduce_add_ps(sum_vec_512);

                    // Scalar loop for the remainder of filter_width
                    for (; k_512 < filter_width; ++k_512) {
                        int64_t idx = input_start_idx - k_512;
                        if (idx >= 0 && idx < in_width) {
                            total_sum += input(b, idx, c) * filter(k_512, 0, c);
                        }
                    }
#elif defined(__AVX2__) && defined(__FMA__)
                    __m256 sum_vec = _mm256_setzero_ps();
                    const __m256i lane_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
                    const __m256i minus_one = _mm256_set1_epi32(-1);
                    const __m256i width_limit_vec = _mm256_set1_epi32(static_cast<int>(in_width));
                    int k = 0;
                    // Process 8 elements at a time
                    for (; k <= filter_width - 8; k += 8) {
                        const int current_input_idx = static_cast<int>(input_start_idx - k);
                        const __m256i base_index_vec = _mm256_set1_epi32(current_input_idx);
                        const __m256i gather_indices = _mm256_sub_epi32(base_index_vec, lane_offsets);
                        const __m256i ge_mask = _mm256_cmpgt_epi32(gather_indices, minus_one);
                        const __m256i lt_mask = _mm256_cmpgt_epi32(width_limit_vec, gather_indices);
                        const __m256i lane_mask = _mm256_and_si256(ge_mask, lt_mask);
                        if (_mm256_testz_si256(lane_mask, lane_mask)) {
                            continue;
                        }
                        __m256 filter_vals = _mm256_loadu_ps(&filter(k, 0, c));
                        __m256 lane_mask_ps = _mm256_castsi256_ps(lane_mask);
                        __m256 input_vals = _mm256_mask_i32gather_ps(
                            _mm256_setzero_ps(),
                            &input(b, 0, c),
                            gather_indices,
                            lane_mask_ps,
                            sizeof(float));
                        sum_vec = _mm256_fmadd_ps(filter_vals, input_vals, sum_vec);
                    }
                    // Horizontal sum for AVX2
                    __m128 vlow  = _mm256_castps256_ps128(sum_vec);
                    __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
                    vlow = _mm_add_ps(vlow, vhigh);
                    __m128 shuf = _mm_movehdup_ps(vlow);
                    __m128 sums = _mm_add_ps(vlow, shuf);
                    shuf = _mm_movehl_ps(shuf, sums);
                    sums = _mm_add_ss(sums, shuf);
                    total_sum += _mm_cvtss_f32(sums);

                    // Scalar loop for the remainder of filter_width
                    for (; k < filter_width; ++k) {
                        int64_t idx = input_start_idx - k;
                        if (idx >= 0 && idx < in_width) {
                            total_sum += input(b, idx, c) * filter(k, 0, c);
                        }
                    }
#elif defined(__ARM_NEON)
                    // NEON: Process 4 elements at a time
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    int k = 0;
                    for (; k <= filter_width - 4; k += 4) {
                        // Manual bounds check for NEON (no masked gather)
                        float filter_vals[4], input_vals[4];
                        for (int lane = 0; lane < 4; ++lane) {
                            filter_vals[lane] = filter(k + lane, 0, c);
                            int64_t idx = input_start_idx - (k + lane);
                            input_vals[lane] = (idx >= 0 && idx < in_width) ? input(b, idx, c) : 0.0f;
                        }
                        float32x4_t filter_v = vld1q_f32(filter_vals);
                        float32x4_t input_v = vld1q_f32(input_vals);
                        sum_vec = vmlaq_f32(sum_vec, filter_v, input_v);  // FMA: sum += filter * input
                    }
                    // Horizontal sum for NEON
                    float32x2_t sum_low = vget_low_f32(sum_vec);
                    float32x2_t sum_high = vget_high_f32(sum_vec);
                    float32x2_t sum_pair = vadd_f32(sum_low, sum_high);
                    total_sum += vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

                    // Scalar remainder
                    for (; k < filter_width; ++k) {
                        int64_t idx = input_start_idx - k;
                        if (idx >= 0 && idx < in_width) {
                            total_sum += input(b, idx, c) * filter(k, 0, c);
                        }
                    }
#else
                    // Scalar fallback (no SIMD available)
                    for (int k = 0; k < filter_width; ++k) {
                        int64_t idx = input_start_idx - k;
                        if (idx >= 0 && idx < in_width) {
                            total_sum += input(b, idx, c) * filter(k, 0, c);
                        }
                    }
#endif
                    output(b, i, c) = total_sum;
                }
            }
        });
    }
private:
    int stride_;
    std::string padding_;
}; // Close FusedDepthwiseConv1DOp class
REGISTER_KERNEL_BUILDER(Name("FusedDepthwiseConv1D").Device(DEVICE_CPU), FusedDepthwiseConv1DOp);


// =============================================================================
// 2. BACKWARD PASS (GRADIENT) OPERATOR
// =============================================================================

REGISTER_OP("FusedDepthwiseConv1DGrad")
    .Input("grad_output: float")
    .Input("input: float")
    .Input("filter: float")
    .Input("stride: int32")
    .Input("padding: string")
    .Output("grad_input: float")
    .Output("grad_filter: float")
    .Output("grad_bias: float")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input;
        ShapeHandle filter;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &filter));
        c->set_output(0, input);
        c->set_output(1, filter);
        c->set_output(2, c->Vector(c->Dim(filter, 2)));
        return OkStatus();
    });

class FusedDepthwiseConv1DGradOp : public OpKernel {
 public:
    explicit FusedDepthwiseConv1DGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output_tensor = context->input(0);
        const Tensor& input_tensor = context->input(1);
        const Tensor& filter_tensor = context->input(2);
        const Tensor& stride_tensor = context->input(3);
        const Tensor& padding_tensor = context->input(4);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(stride_tensor.shape()),
                    errors::InvalidArgument("stride tensor must be scalar"));
        const int32 stride = stride_tensor.scalar<int32>()();
        OP_REQUIRES(context, stride > 0,
                    errors::InvalidArgument("stride must be positive, got ", stride));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(padding_tensor.shape()),
                    errors::InvalidArgument("padding tensor must be scalar"));

        const int64_t batch_size = input_tensor.dim_size(0);
        const int64_t in_width = input_tensor.dim_size(1);
        const int64_t channels = input_tensor.dim_size(2);
        const int64_t filter_width = filter_tensor.dim_size(0);
        const int64_t out_width = grad_output_tensor.dim_size(1);

        Tensor* grad_input_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &grad_input_tensor));
        Tensor* grad_filter_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, filter_tensor.shape(), &grad_filter_tensor));
        Tensor* grad_bias_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({channels}), &grad_bias_tensor));

        auto grad_input = grad_input_tensor->tensor<float, 3>();
        auto grad_filter = grad_filter_tensor->tensor<float, 3>();
        auto grad_bias = grad_bias_tensor->vec<float>();
        grad_input.setZero();
        grad_filter.setZero();
        grad_bias.setZero();

        auto grad_output = grad_output_tensor.tensor<float, 3>();
        auto input = input_tensor.tensor<float, 3>();
        auto filter = filter_tensor.tensor<float, 3>();

        const int64_t total_tasks = batch_size * channels;
        const std::size_t cost_per_task =
            static_cast<std::size_t>(std::max<int64_t>(1, out_width * filter_width));

        saguaro::parallel::ForShard(
            static_cast<std::size_t>(total_tasks), cost_per_task,
            [&](int64_t start, int64_t limit) {
                for (int64_t task_idx = start; task_idx < limit; ++task_idx) {
                    const int64_t b = task_idx / channels;
                    const int64_t c = task_idx % channels;
                    for (int64_t i = 0; i < out_width; ++i) {
                        const float grad_out_val = grad_output(b, i, c);
                        if (grad_out_val == 0.0f) {
                            continue;
                        }
                        AtomicAddFloat(&grad_bias(c), grad_out_val);
                        const int64_t input_start_idx = i * stride;

                        // Phase 11: Vectorize gradient accumulation
#if defined(__AVX512F__)
                        // AVX512: Process 16 filter taps at a time
                        int64_t k = 0;
                        __m512 grad_out_vec_512 = _mm512_set1_ps(grad_out_val);
                        for (; k <= filter_width - 16; k += 16) {
                            // Bounds check for 16 elements
                            bool all_valid = true;
                            for (int lane = 0; lane < 16; ++lane) {
                                int64_t idx = input_start_idx - (k + lane);
                                if (idx < 0 || idx >= in_width) {
                                    all_valid = false;
                                    break;
                                }
                            }
                            if (!all_valid) {
                                // Fall through to scalar for boundary cases
                                break;
                            }
                            // Load filter and input values
                            __m512 filter_vals_512 = _mm512_loadu_ps(&filter(k, 0, c));
                            __m512 input_vals_512 = _mm512_set_ps(
                                input(b, input_start_idx - (k+15), c), input(b, input_start_idx - (k+14), c),
                                input(b, input_start_idx - (k+13), c), input(b, input_start_idx - (k+12), c),
                                input(b, input_start_idx - (k+11), c), input(b, input_start_idx - (k+10), c),
                                input(b, input_start_idx - (k+9), c), input(b, input_start_idx - (k+8), c),
                                input(b, input_start_idx - (k+7), c), input(b, input_start_idx - (k+6), c),
                                input(b, input_start_idx - (k+5), c), input(b, input_start_idx - (k+4), c),
                                input(b, input_start_idx - (k+3), c), input(b, input_start_idx - (k+2), c),
                                input(b, input_start_idx - (k+1), c), input(b, input_start_idx - k, c)
                            );
                            // Compute gradients
                            __m512 grad_input_contrib = _mm512_mul_ps(grad_out_vec_512, filter_vals_512);
                            __m512 grad_filter_contrib = _mm512_mul_ps(grad_out_vec_512, input_vals_512);
                            // Store grad_input (non-atomic, single writer per batch/channel)
                            for (int lane = 0; lane < 16; ++lane) {
                                int64_t idx = input_start_idx - (k + lane);
                                grad_input(b, idx, c) += ((float*)&grad_input_contrib)[15 - lane];
                            }
                            // Store grad_filter (atomic, multiple writers)
                            for (int lane = 0; lane < 16; ++lane) {
                                AtomicAddFloat(&grad_filter(k + lane, 0, c), ((float*)&grad_filter_contrib)[15 - lane]);
                            }
                        }
                        // Scalar remainder
                        for (; k < filter_width; ++k) {
                            const int64_t idx = input_start_idx - k;
                            if (idx < 0 || idx >= in_width) {
                                continue;
                            }
                            grad_input(b, idx, c) += grad_out_val * filter(k, 0, c);
                            AtomicAddFloat(&grad_filter(k, 0, c), grad_out_val * input(b, idx, c));
                        }
#elif defined(__AVX2__) && defined(__FMA__)
                        // AVX2: Process 8 filter taps at a time
                        int64_t k = 0;
                        __m256 grad_out_vec = _mm256_set1_ps(grad_out_val);
                        for (; k <= filter_width - 8; k += 8) {
                            // Bounds check for 8 elements
                            bool all_valid = true;
                            for (int lane = 0; lane < 8; ++lane) {
                                int64_t idx = input_start_idx - (k + lane);
                                if (idx < 0 || idx >= in_width) {
                                    all_valid = false;
                                    break;
                                }
                            }
                            if (!all_valid) {
                                break;
                            }
                            // Load filter and input values
                            __m256 filter_vals = _mm256_loadu_ps(&filter(k, 0, c));
                            __m256 input_vals = _mm256_set_ps(
                                input(b, input_start_idx - (k+7), c), input(b, input_start_idx - (k+6), c),
                                input(b, input_start_idx - (k+5), c), input(b, input_start_idx - (k+4), c),
                                input(b, input_start_idx - (k+3), c), input(b, input_start_idx - (k+2), c),
                                input(b, input_start_idx - (k+1), c), input(b, input_start_idx - k, c)
                            );
                            // Compute gradients
                            __m256 grad_input_contrib = _mm256_mul_ps(grad_out_vec, filter_vals);
                            __m256 grad_filter_contrib = _mm256_mul_ps(grad_out_vec, input_vals);
                            // Store grad_input (non-atomic)
                            for (int lane = 0; lane < 8; ++lane) {
                                int64_t idx = input_start_idx - (k + lane);
                                grad_input(b, idx, c) += ((float*)&grad_input_contrib)[7 - lane];
                            }
                            // Store grad_filter (atomic)
                            for (int lane = 0; lane < 8; ++lane) {
                                AtomicAddFloat(&grad_filter(k + lane, 0, c), ((float*)&grad_filter_contrib)[7 - lane]);
                            }
                        }
                        // Scalar remainder
                        for (; k < filter_width; ++k) {
                            const int64_t idx = input_start_idx - k;
                            if (idx < 0 || idx >= in_width) {
                                continue;
                            }
                            grad_input(b, idx, c) += grad_out_val * filter(k, 0, c);
                            AtomicAddFloat(&grad_filter(k, 0, c), grad_out_val * input(b, idx, c));
                        }
#elif defined(__ARM_NEON)
                        // NEON: Process 4 filter taps at a time
                        int64_t k = 0;
                        float32x4_t grad_out_vec = vdupq_n_f32(grad_out_val);
                        for (; k <= filter_width - 4; k += 4) {
                            // Bounds check for 4 elements
                            bool all_valid = true;
                            for (int lane = 0; lane < 4; ++lane) {
                                int64_t idx = input_start_idx - (k + lane);
                                if (idx < 0 || idx >= in_width) {
                                    all_valid = false;
                                    break;
                                }
                            }
                            if (!all_valid) {
                                break;
                            }
                            // Load filter and input values
                            float filter_vals[4], input_vals[4];
                            for (int lane = 0; lane < 4; ++lane) {
                                filter_vals[lane] = filter(k + lane, 0, c);
                                input_vals[3 - lane] = input(b, input_start_idx - (k + lane), c);
                            }
                            float32x4_t filter_v = vld1q_f32(filter_vals);
                            float32x4_t input_v = vld1q_f32(input_vals);
                            // Compute gradients
                            float32x4_t grad_input_contrib = vmulq_f32(grad_out_vec, filter_v);
                            float32x4_t grad_filter_contrib = vmulq_f32(grad_out_vec, input_v);
                            // Store grad_input (non-atomic)
                            float grad_input_buf[4];
                            vst1q_f32(grad_input_buf, grad_input_contrib);
                            for (int lane = 0; lane < 4; ++lane) {
                                int64_t idx = input_start_idx - (k + lane);
                                grad_input(b, idx, c) += grad_input_buf[lane];
                            }
                            // Store grad_filter (atomic)
                            float grad_filter_buf[4];
                            vst1q_f32(grad_filter_buf, grad_filter_contrib);
                            for (int lane = 0; lane < 4; ++lane) {
                                AtomicAddFloat(&grad_filter(k + lane, 0, c), grad_filter_buf[3 - lane]);
                            }
                        }
                        // Scalar remainder
                        for (; k < filter_width; ++k) {
                            const int64_t idx = input_start_idx - k;
                            if (idx < 0 || idx >= in_width) {
                                continue;
                            }
                            grad_input(b, idx, c) += grad_out_val * filter(k, 0, c);
                            AtomicAddFloat(&grad_filter(k, 0, c), grad_out_val * input(b, idx, c));
                        }
#else
                        // Scalar fallback
                        for (int64_t k = 0; k < filter_width; ++k) {
                            const int64_t idx = input_start_idx - k;
                            if (idx < 0 || idx >= in_width) {
                                continue;
                            }
                            grad_input(b, idx, c) += grad_out_val * filter(k, 0, c);
                            AtomicAddFloat(&grad_filter(k, 0, c),
                                           grad_out_val * input(b, idx, c));
                        }
#endif
                    }
                }
            });
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedDepthwiseConv1DGrad").Device(DEVICE_CPU), FusedDepthwiseConv1DGradOp);


} // namespace tensorflow
