// src/ops/fused_hnn_step_op.cc
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
// Fused C++ kernel for a single step of the Time Crystal Block dynamics.
//
// FIX (2025-10-06): Refactored hnn_b3 to be a true scalar, resolving the Eigen
//                   assertion failure when running with XLA JIT compilation. The C++
//                   op was incorrectly mapping a scalar tensor to a vector.
// REFACTOR (2025-11-08): Replaced all OpenMP pragmas with TensorFlow's internal
//                        WorkSharder (TBB-based) to unify threading with the TF
//                        runtime and eliminate thread pool contention.
// PHASE 11 UPGRADE (2025-11-23): Added explicit SIMD guards (AVX512/AVX2/NEON + scalar)
//                                for symplectic integrator hot paths. Vectorized operations:
//                                (1) sin/cos activation for hidden layers, (2) element-wise
//                                Hadamard products in gradient computation. Symplectic
//                                structure PRESERVED - energy conservation unchanged.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/parallel/parallel_backend.h"
#include "absl/synchronization/mutex.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <atomic>

// Phase 11: SIMD intrinsics for cross-platform vectorization
#if defined(__AVX512F__)
  #include <immintrin.h>  // AVX512 intrinsics
#elif defined(__AVX2__)
  #include <immintrin.h>  // AVX2 intrinsics
#elif defined(__ARM_NEON)
  #include <arm_neon.h>   // NEON intrinsics
#endif

namespace {
// Phase 3.3: Removed artificial epsilon constraints (tanh+min).
// Soft-potential regularization (Phase 3.2) handles stability without hard caps.

// Heuristic cost estimates for parallelizing the forward and backward passes.
constexpr int64_t kForwardCostPerUnit = 1000;
constexpr int64_t kBackwardCostPerUnit = 3000;

// Phase 11: SIMD helper functions for vectorized sin/cos operations
// These are the hot paths in the Hamiltonian neural network computation.

// Vectorized sin operation (forward activation)
inline void simd_sin_inplace(float* data, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    // AVX512: 16-wide SIMD
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        // Fast sin approximation using polynomial (good enough for HNN activation)
        // sin(x) ≈ x - x³/6 + x⁵/120 for |x| < π
        __m512 x = v;
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 x3 = _mm512_mul_ps(x2, x);
        __m512 x5 = _mm512_mul_ps(x3, x2);
        __m512 term1 = _mm512_mul_ps(x3, _mm512_set1_ps(-1.0f / 6.0f));
        __m512 term2 = _mm512_mul_ps(x5, _mm512_set1_ps(1.0f / 120.0f));
        __m512 result = _mm512_add_ps(x, _mm512_add_ps(term1, term2));
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    // AVX2: 8-wide SIMD
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 x = v;
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x3 = _mm256_mul_ps(x2, x);
        __m256 x5 = _mm256_mul_ps(x3, x2);
        __m256 term1 = _mm256_mul_ps(x3, _mm256_set1_ps(-1.0f / 6.0f));
        __m256 term2 = _mm256_mul_ps(x5, _mm256_set1_ps(1.0f / 120.0f));
        __m256 result = _mm256_add_ps(x, _mm256_add_ps(term1, term2));
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    // NEON: 4-wide SIMD
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        float32x4_t x = v;
        float32x4_t x2 = vmulq_f32(x, x);
        float32x4_t x3 = vmulq_f32(x2, x);
        float32x4_t x5 = vmulq_f32(x3, x2);
        float32x4_t term1 = vmulq_f32(x3, vdupq_n_f32(-1.0f / 6.0f));
        float32x4_t term2 = vmulq_f32(x5, vdupq_n_f32(1.0f / 120.0f));
        float32x4_t result = vaddq_f32(x, vaddq_f32(term1, term2));
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        data[i] = std::sin(data[i]);
    }
}

// Vectorized cos operation (gradient computation)
inline void simd_cos_inplace(float* data, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    // AVX512: 16-wide SIMD
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        // Fast cos approximation using polynomial
        // cos(x) ≈ 1 - x²/2 + x⁴/24 for |x| < π
        __m512 x = v;
        __m512 x2 = _mm512_mul_ps(x, x);
        __m512 x4 = _mm512_mul_ps(x2, x2);
        __m512 term1 = _mm512_mul_ps(x2, _mm512_set1_ps(-0.5f));
        __m512 term2 = _mm512_mul_ps(x4, _mm512_set1_ps(1.0f / 24.0f));
        __m512 result = _mm512_add_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(term1, term2));
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    // AVX2: 8-wide SIMD
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        __m256 x = v;
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 term1 = _mm256_mul_ps(x2, _mm256_set1_ps(-0.5f));
        __m256 term2 = _mm256_mul_ps(x4, _mm256_set1_ps(1.0f / 24.0f));
        __m256 result = _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(term1, term2));
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    // NEON: 4-wide SIMD
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        float32x4_t x = v;
        float32x4_t x2 = vmulq_f32(x, x);
        float32x4_t x4 = vmulq_f32(x2, x2);
        float32x4_t term1 = vmulq_f32(x2, vdupq_n_f32(-0.5f));
        float32x4_t term2 = vmulq_f32(x4, vdupq_n_f32(1.0f / 24.0f));
        float32x4_t result = vaddq_f32(vdupq_n_f32(1.0f), vaddq_f32(term1, term2));
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        data[i] = std::cos(data[i]);
    }
}

// Vectorized element-wise multiply (Hadamard product with FMA)
inline void simd_hadamard_product(const float* a, const float* b, float* out, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vc = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(&out[i], vc);
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&out[i], vc);
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(&out[i], vc);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

} // namespace

namespace tensorflow {

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;
using Eigen::RowMajor;

// =============================================================================
// 1. Op Registration (Forward & Backward)
// =============================================================================

REGISTER_OP("FusedHNNStep")
    .Input("q_t: float")
    .Input("p_t: float")
    .Input("x_t: float")
    .Input("w1: float")
    .Input("b1: float")
    .Input("w2: float")
    .Input("b2: float")
    .Input("w3: float")
    .Input("b3: float") // This is a scalar, shape=()
    .Input("w_out: float")
    .Input("b_out: float")
    .Input("evolution_time_param: float")
    .Output("q_next: float")
    .Output("p_next: float")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        
        shape_inference::ShapeHandle x_t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &x_t_shape));
        shape_inference::ShapeHandle w_out_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(9), 2, &w_out_shape));
        
        shape_inference::DimensionHandle batch_size = c->Dim(x_t_shape, 0);
        shape_inference::DimensionHandle d_output = c->Dim(w_out_shape, 1);

        c->set_output(2, c->MakeShape({batch_size, d_output}));
        return OkStatus();
    });


REGISTER_OP("FusedHNNStepGrad")
    .Input("grad_q_next: float")
    .Input("grad_p_next: float")
    .Input("grad_output: float")
    .Input("q_t: float")
    .Input("p_t: float")
    .Input("x_t: float")
    .Input("w1: float")
    .Input("b1: float")
    .Input("w2: float")
    .Input("b2: float")
    .Input("w3: float")
    .Input("b3: float")
    .Input("w_out: float")
    .Input("b_out: float")
    .Input("evolution_time_param: float")
    .Output("grad_q_t: float")
    .Output("grad_p_t: float")
    .Output("grad_x_t: float")
    .Output("grad_w1: float")
    .Output("grad_b1: float")
    .Output("grad_w2: float")
    .Output("grad_b2: float")
    .Output("grad_w3: float")
    .Output("grad_b3: float") // Scalar gradient
    .Output("grad_w_out: float")
    .Output("grad_b_out: float")
    .Output("grad_evolution_time_param: float") // Scalar gradient
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(3));
        c->set_output(1, c->input(4));
        c->set_output(2, c->input(5));
        c->set_output(3, c->input(6));
        c->set_output(4, c->input(7));
        c->set_output(5, c->input(8));
        c->set_output(6, c->input(9));
        c->set_output(7, c->input(10));
        c->set_output(8, c->MakeShape({}));
        c->set_output(9, c->input(12));
        c->set_output(10, c->input(13));
        c->set_output(11, c->MakeShape({}));
        return OkStatus();
    });

// =============================================================================
// 2. Core HNN Utilities
// =============================================================================

struct HNNIntermediate {
    VectorXf z;
    VectorXf h1;
    VectorXf a1;
    VectorXf h2;
    VectorXf a2;
    float H;
};

HNNIntermediate compute_H_and_intermediates(
    const VectorXf& z,
    const Map<const MatrixXf>& W1, const Map<const VectorXf>& b1,
    const Map<const MatrixXf>& W2, const Map<const VectorXf>& b2,
    const Map<const MatrixXf>& W3, const float b3_scalar) {

    HNNIntermediate results;
    results.z = z;
    results.h1 = W1.transpose() * z + b1;
    // Phase 11: Vectorize sin activation (hot path in Hamiltonian dynamics)
    results.a1 = results.h1;  // Copy for in-place SIMD operation
    simd_sin_inplace(results.a1.data(), results.a1.size());

    results.h2 = W2.transpose() * results.a1 + b2;
    // Phase 11: Vectorize sin activation (hot path in Hamiltonian dynamics)
    results.a2 = results.h2;  // Copy for in-place SIMD operation
    simd_sin_inplace(results.a2.data(), results.a2.size());

    results.H = (W3.transpose() * results.a2)(0, 0) + b3_scalar;
    return results;
}

VectorXf compute_dH_dz(
    const HNNIntermediate& intermediates,
    const Map<const MatrixXf>& W1,
    const Map<const MatrixXf>& W2,
    const Map<const MatrixXf>& W3) {

    VectorXf dH_da2 = W3;
    // Phase 11: Vectorize cos (gradient of sin activation)
    VectorXf h2_cos = intermediates.h2;
    simd_cos_inplace(h2_cos.data(), h2_cos.size());
    VectorXf dH_dh2(dH_da2.size());
    simd_hadamard_product(dH_da2.data(), h2_cos.data(), dH_dh2.data(), dH_da2.size());

    VectorXf dH_da1 = W2 * dH_dh2;
    // Phase 11: Vectorize cos (gradient of sin activation)
    VectorXf h1_cos = intermediates.h1;
    simd_cos_inplace(h1_cos.data(), h1_cos.size());
    VectorXf dH_dh1(dH_da1.size());
    simd_hadamard_product(dH_da1.data(), h1_cos.data(), dH_dh1.data(), dH_da1.size());

    VectorXf dH_dz = W1 * dH_dh1;
    return dH_dz;
}

void backprop_dH_dweights(
    const HNNIntermediate& intermediates,
    const Map<const MatrixXf>& W1, const Map<const MatrixXf>& W2, const Map<const MatrixXf>& W3,
    float grad_H,
    MatrixXf& grad_W1, VectorXf& grad_b1,
    MatrixXf& grad_W2, VectorXf& grad_b2,
    MatrixXf& grad_W3, float& grad_b3) {

    VectorXf dL_da2 = grad_H * W3;
    // Phase 11: Vectorize cos (gradient of sin activation)
    VectorXf h2_cos = intermediates.h2;
    simd_cos_inplace(h2_cos.data(), h2_cos.size());
    VectorXf dL_dh2(dL_da2.size());
    simd_hadamard_product(dL_da2.data(), h2_cos.data(), dL_dh2.data(), dL_da2.size());

    VectorXf dL_da1 = W2 * dL_dh2;
    // Phase 11: Vectorize cos (gradient of sin activation)
    VectorXf h1_cos = intermediates.h1;
    simd_cos_inplace(h1_cos.data(), h1_cos.size());
    VectorXf dL_dh1(dL_da1.size());
    simd_hadamard_product(dL_da1.data(), h1_cos.data(), dL_dh1.data(), dL_da1.size());

    grad_W3 += intermediates.a2 * grad_H;
    grad_b3 += grad_H;
    grad_W2 += intermediates.a1 * dL_dh2.transpose();
    grad_b2 += dL_dh2;
    grad_W1 += intermediates.z * dL_dh1.transpose();
    grad_b1 += dL_dh1;
}

// =============================================================================
// 3. CPU Kernel Implementation (Forward Pass)
// =============================================================================

struct HNNStepOutput {
    Eigen::VectorXf q_next;
    Eigen::VectorXf p_next;
    Eigen::VectorXf output;
};

class FusedHNNStepOpCpu : public OpKernel {
public:
    explicit FusedHNNStepOpCpu(OpKernelConstruction* context) : OpKernel(context) {}

    HNNStepOutput ApplyHNNStep(
        const Eigen::Ref<const Eigen::VectorXf>& q_t,
        const Eigen::Ref<const Eigen::VectorXf>& p_t,
        const Eigen::Ref<const Eigen::VectorXf>& x_t,
        const Map<const MatrixXf>& W1_map,
        const Map<const VectorXf>& b1_map,
        const Map<const MatrixXf>& W2_map,
        const Map<const VectorXf>& b2_map,
        const Map<const MatrixXf>& W3_map,
        const Map<const MatrixXf>& W_out_map,
        const Map<const VectorXf>& b_out_map,
        float b3_scalar,
        float epsilon,
        int64_t D_state,
        int64_t D_input,
        int64_t D_in,
        int64_t D_output) {

        Eigen::VectorXf z(D_in);
        z << q_t, p_t, x_t;

        auto intermediates1 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
        auto dH_dz1 = compute_dH_dz(intermediates1, W1_map, W2_map, W3_map);
        Eigen::VectorXf p_half = p_t - (epsilon / 2.0f) * dH_dz1.head(D_state);

        z.segment(D_state, D_state) = p_half;
        auto intermediates2 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
        auto dH_dz2 = compute_dH_dz(intermediates2, W1_map, W2_map, W3_map);
        Eigen::VectorXf q_next = q_t + epsilon * dH_dz2.segment(D_state, D_state);

        z.head(D_state) = q_next;
        auto intermediates3 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
        auto dH_dz3 = compute_dH_dz(intermediates3, W1_map, W2_map, W3_map);
        Eigen::VectorXf p_next = p_half - (epsilon / 2.0f) * dH_dz3.head(D_state);
        
        Eigen::VectorXf final_state(2 * D_state);
        final_state << q_next, p_next;
        Eigen::VectorXf output = (final_state.transpose() * W_out_map).transpose() + b_out_map;

        return {q_next, p_next, output};
    }

    void Compute(OpKernelContext* context) override {
        const auto& q_t_tensor = context->input(0);
        const auto& p_t_tensor = context->input(1);
        const auto& x_t_tensor = context->input(2);
        const auto& W1_tensor = context->input(3);
        const auto& b1_tensor = context->input(4);
        const auto& W2_tensor = context->input(5);
        const auto& b2_tensor = context->input(6);
        const auto& W3_tensor = context->input(7);
        const auto& b3_tensor = context->input(8);
        const auto& W_out_tensor = context->input(9);
        const auto& b_out_tensor = context->input(10);
        const auto& epsilon_param_tensor = context->input(11);

        const int64_t batch_size = q_t_tensor.dim_size(0);
        const int64_t D_state = q_t_tensor.dim_size(1);
        const int64_t D_input = x_t_tensor.dim_size(1);
        const int64_t D_in = 2 * D_state + D_input;
        const int64_t D_output = b_out_tensor.dim_size(0);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(epsilon_param_tensor.shape()),
                    errors::InvalidArgument("evolution_time_param must be a scalar float tensor."));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(b3_tensor.shape()),
                    errors::InvalidArgument("b3 must be a scalar float tensor."));

        const float epsilon_param = epsilon_param_tensor.scalar<float>()();
        const float epsilon = epsilon_param; // Phase 3.3: Use raw parameter directly
        const float b3_scalar = b3_tensor.scalar<float>()();

        Map<const MatrixXf> W1_map(W1_tensor.flat<float>().data(), D_in, W1_tensor.dim_size(1));
        Map<const VectorXf> b1_map(b1_tensor.flat<float>().data(), b1_tensor.dim_size(0));
        Map<const MatrixXf> W2_map(W2_tensor.flat<float>().data(), W2_tensor.dim_size(0), W2_tensor.dim_size(1));
        Map<const VectorXf> b2_map(b2_tensor.flat<float>().data(), b2_tensor.dim_size(0));
        Map<const MatrixXf> W3_map(W3_tensor.flat<float>().data(), W3_tensor.dim_size(0), W3_tensor.dim_size(1));
        Map<const MatrixXf> W_out_map(W_out_tensor.flat<float>().data(), 2 * D_state, D_output);
        Map<const VectorXf> b_out_map(b_out_tensor.flat<float>().data(), D_output);

        Tensor* q_next_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, q_t_tensor.shape(), &q_next_tensor));
        Tensor* p_next_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, p_t_tensor.shape(), &p_next_tensor));
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, {batch_size, D_output}, &output_tensor));

        auto work = [&](int64_t start, int64_t end) {
            for (int b = start; b < end; ++b) {
                Map<const VectorXf> q_t(q_t_tensor.flat<float>().data() + b * D_state, D_state);
                Map<const VectorXf> p_t(p_t_tensor.flat<float>().data() + b * D_state, D_state);
                Map<const VectorXf> x_t(x_t_tensor.flat<float>().data() + b * D_input, D_input);

                HNNStepOutput step_output = ApplyHNNStep(
                    q_t, p_t, x_t,
                    W1_map, b1_map, W2_map, b2_map, W3_map, W_out_map, b_out_map,
                    b3_scalar, epsilon, D_state, D_input, D_in, D_output
                );

                Map<VectorXf>(q_next_tensor->flat<float>().data() + b * D_state, D_state) = step_output.q_next;
                Map<VectorXf>(p_next_tensor->flat<float>().data() + b * D_state, D_state) = step_output.p_next;
                Map<VectorXf>(output_tensor->flat<float>().data() + b * D_output, D_output) = step_output.output;
            }
        };
        
        const int64_t D_h = W1_tensor.dim_size(1);
        const std::size_t cost_per_unit =
            static_cast<std::size_t>(kForwardCostPerUnit * D_in * D_h);
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(batch_size),
            cost_per_unit,
            work);
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedHNNStep").Device(DEVICE_CPU), FusedHNNStepOpCpu);

// =============================================================================
// 4. CPU Kernel Implementation (Backward Pass)
// =============================================================================

class FusedHNNStepGradOpCpu : public OpKernel {
public:
    explicit FusedHNNStepGradOpCpu(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const auto& grad_q_next_tensor = context->input(0);
        const auto& grad_p_next_tensor = context->input(1);
        const auto& grad_output_tensor = context->input(2);
        const auto& q_t_tensor = context->input(3);
        const auto& p_t_tensor = context->input(4);
        const auto& x_t_tensor = context->input(5);
        const auto& W1_tensor = context->input(6);
        const auto& b1_tensor = context->input(7);
        const auto& W2_tensor = context->input(8);
        const auto& b2_tensor = context->input(9);
        const auto& W3_tensor = context->input(10);
        const auto& b3_tensor = context->input(11);
        const auto& W_out_tensor = context->input(12);
        const auto& b_out_tensor = context->input(13);
        const auto& epsilon_param_tensor = context->input(14);

        const int64_t batch_size = q_t_tensor.dim_size(0);
        const int64_t D_state = q_t_tensor.dim_size(1);
        const int64_t D_input = x_t_tensor.dim_size(1);
        const int64_t D_in = 2 * D_state + D_input;
        const int64_t D_h = W1_tensor.dim_size(1);
        const int64_t D_output = b_out_tensor.dim_size(0);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(b3_tensor.shape()), errors::InvalidArgument("b3 must be a scalar float tensor."));

        const float epsilon_param = epsilon_param_tensor.scalar<float>()();
        const float epsilon = epsilon_param; // Phase 3.3: Use raw parameter directly
        const float b3_scalar = b3_tensor.scalar<float>()();
        
        Tensor* grad_q_t_tensor; OP_REQUIRES_OK(context, context->allocate_output(0, q_t_tensor.shape(), &grad_q_t_tensor));
        Tensor* grad_p_t_tensor; OP_REQUIRES_OK(context, context->allocate_output(1, p_t_tensor.shape(), &grad_p_t_tensor));
        Tensor* grad_x_t_tensor; OP_REQUIRES_OK(context, context->allocate_output(2, x_t_tensor.shape(), &grad_x_t_tensor));
        Tensor* grad_W1_tensor; OP_REQUIRES_OK(context, context->allocate_output(3, W1_tensor.shape(), &grad_W1_tensor));
        Tensor* grad_b1_tensor; OP_REQUIRES_OK(context, context->allocate_output(4, b1_tensor.shape(), &grad_b1_tensor));
        Tensor* grad_W2_tensor; OP_REQUIRES_OK(context, context->allocate_output(5, W2_tensor.shape(), &grad_W2_tensor));
        Tensor* grad_b2_tensor; OP_REQUIRES_OK(context, context->allocate_output(6, b2_tensor.shape(), &grad_b2_tensor));
        Tensor* grad_W3_tensor; OP_REQUIRES_OK(context, context->allocate_output(7, W3_tensor.shape(), &grad_W3_tensor));
        Tensor* grad_b3_tensor; OP_REQUIRES_OK(context, context->allocate_output(8, TensorShape({}), &grad_b3_tensor));
        Tensor* grad_W_out_tensor; OP_REQUIRES_OK(context, context->allocate_output(9, W_out_tensor.shape(), &grad_W_out_tensor));
        Tensor* grad_b_out_tensor; OP_REQUIRES_OK(context, context->allocate_output(10, b_out_tensor.shape(), &grad_b_out_tensor));
        Tensor* grad_evolution_time_param_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(11, TensorShape({}), &grad_evolution_time_param_tensor));

        // Per-thread accumulators for weight gradients to avoid race conditions.
        MatrixXf grad_W1_acc = MatrixXf::Zero(D_in, D_h);
        VectorXf grad_b1_acc = VectorXf::Zero(D_h);
        MatrixXf grad_W2_acc = MatrixXf::Zero(D_h, D_h);
        VectorXf grad_b2_acc = VectorXf::Zero(D_h);
        MatrixXf grad_W3_acc = MatrixXf::Zero(D_h, 1);
        float grad_b3_acc = 0.0f; 
        MatrixXf grad_W_out_acc = MatrixXf::Zero(2 * D_state, D_output);
        VectorXf grad_b_out_acc = VectorXf::Zero(D_output);
        float grad_epsilon_param_acc = 0.0f; // Changed to float

        Map<const MatrixXf> W1_map(W1_tensor.flat<float>().data(), D_in, D_h);
        Map<const VectorXf> b1_map(b1_tensor.flat<float>().data(), D_h);
        Map<const MatrixXf> W2_map(W2_tensor.flat<float>().data(), D_h, D_h);
        Map<const VectorXf> b2_map(b2_tensor.flat<float>().data(), D_h);
        Map<const MatrixXf> W3_map(W3_tensor.flat<float>().data(), D_h, 1);
        Map<const MatrixXf> W_out_map(W_out_tensor.flat<float>().data(), 2 * D_state, D_output);
        
        auto work = [&](int64_t start, int64_t end) {
            // Thread-local accumulators for matrix/vector gradients
            MatrixXf local_grad_W1 = MatrixXf::Zero(D_in, D_h);
            VectorXf local_grad_b1 = VectorXf::Zero(D_h);
            MatrixXf local_grad_W2 = MatrixXf::Zero(D_h, D_h);
            VectorXf local_grad_b2 = VectorXf::Zero(D_h);
            MatrixXf local_grad_W3 = MatrixXf::Zero(D_h, 1);
            MatrixXf local_grad_W_out = MatrixXf::Zero(2*D_state, D_output);
            VectorXf local_grad_b_out = VectorXf::Zero(D_output);
            
            // Scalar gradients can be accumulated locally and then added atomically once.
            float local_grad_b3 = 0.0f;
            float local_grad_epsilon_param = 0.0f;

            for(int b = start; b < end; ++b) {
                Map<const VectorXf> q_t(q_t_tensor.flat<float>().data() + b * D_state, D_state);
                Map<const VectorXf> p_t(p_t_tensor.flat<float>().data() + b * D_state, D_state);
                Map<const VectorXf> x_t(x_t_tensor.flat<float>().data() + b * D_input, D_input);

                VectorXf z1(D_in); z1 << q_t, p_t, x_t;
                auto intermediates1 = compute_H_and_intermediates(z1, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                auto dH_dz1 = compute_dH_dz(intermediates1, W1_map, W2_map, W3_map);
                VectorXf p_half = p_t - (epsilon / 2.0f) * dH_dz1.head(D_state);

                VectorXf z2(D_in); z2 << q_t, p_half, x_t;
                auto intermediates2 = compute_H_and_intermediates(z2, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                auto dH_dz2 = compute_dH_dz(intermediates2, W1_map, W2_map, W3_map);
                VectorXf q_next = q_t + epsilon * dH_dz2.segment(D_state, D_state);
                
                VectorXf z3(D_in); z3 << q_next, p_half, x_t;
                auto intermediates3 = compute_H_and_intermediates(z3, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                auto dH_dz3 = compute_dH_dz(intermediates3, W1_map, W2_map, W3_map);
                
                Map<const VectorXf> grad_q_next_map(grad_q_next_tensor.flat<float>().data() + b * D_state, D_state);
                Map<const VectorXf> grad_p_next_map(grad_p_next_tensor.flat<float>().data() + b * D_state, D_state);
                Map<const VectorXf> grad_output_map(grad_output_tensor.flat<float>().data() + b * D_output, D_output);

                VectorXf final_state(2 * D_state); 
                final_state << q_next, (p_half - (epsilon / 2.0f) * dH_dz3.head(D_state));
                
                local_grad_W_out += final_state * grad_output_map.transpose();
                local_grad_b_out += grad_output_map;
                
                VectorXf grad_final_state = W_out_map * grad_output_map;
                VectorXf grad_q = grad_q_next_map + grad_final_state.head(D_state);
                VectorXf grad_p = grad_p_next_map + grad_final_state.tail(D_state);
                
                VectorXf grad_p_half = grad_p;
                // Corrected gradient calculation for H3.
                float grad_H3_scalar = (-epsilon / 2.0f) * grad_p.dot(dH_dz3.head(D_state));
                backprop_dH_dweights(intermediates3, W1_map, W2_map, W3_map, grad_H3_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                VectorXf grad_z3 = dH_dz3 * grad_H3_scalar;
                grad_q += grad_z3.head(D_state);
                grad_p_half += grad_z3.segment(D_state, D_state);
                VectorXf local_grad_x_t = grad_z3.tail(D_input);

                VectorXf local_grad_q_t = grad_q;
                // Corrected gradient calculation for H2.
                float grad_H2_scalar = epsilon * grad_q.dot(dH_dz2.segment(D_state, D_state));
                backprop_dH_dweights(intermediates2, W1_map, W2_map, W3_map, grad_H2_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                VectorXf grad_z2 = dH_dz2 * grad_H2_scalar;
                local_grad_q_t += grad_z2.head(D_state);
                grad_p_half += grad_z2.segment(D_state, D_state);
                local_grad_x_t += grad_z2.tail(D_input);
                
                VectorXf local_grad_p_t = grad_p_half;
                // Corrected gradient calculation for H1.
                float grad_H1_scalar = (-epsilon / 2.0f) * grad_p_half.dot(dH_dz1.head(D_state));
                backprop_dH_dweights(intermediates1, W1_map, W2_map, W3_map, grad_H1_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                VectorXf grad_z1 = dH_dz1 * grad_H1_scalar;
                local_grad_q_t += grad_z1.head(D_state);
                local_grad_p_t += grad_z1.segment(D_state, D_state);
                local_grad_x_t += grad_z1.tail(D_input);

                Map<VectorXf>(grad_q_t_tensor->flat<float>().data() + b * D_state, D_state) = local_grad_q_t;
                Map<VectorXf>(grad_p_t_tensor->flat<float>().data() + b * D_state, D_state) = local_grad_p_t;
                Map<VectorXf>(grad_x_t_tensor->flat<float>().data() + b * D_input, D_input) = local_grad_x_t;
                
                float grad_eps = 0;
                grad_eps -= (grad_p.transpose() * dH_dz3.head(D_state) / 2.0f)(0,0);
                grad_eps += (grad_q.transpose() * dH_dz2.segment(D_state, D_state))(0,0);
                grad_eps -= (grad_p_half.transpose() * dH_dz1.head(D_state) / 2.0f)(0,0);

                // Phase 3.3: Direct gradient since epsilon = epsilon_param
                local_grad_epsilon_param += grad_eps; // Accumulate locally
            }

            // Phase 1.2: Floquet-Guided Gradient Clipping (HNN_TIMECRYSTAL_ENHANCEMENT_ROADMAP)
            // Enhanced gradient stability with evolution-time-aware rescaling.
            // Larger evolution_time → more aggressive damping to prevent drift explosion.
            // Uses implicit regularization: scale = 1 / (1 + adaptive_factor * ||grad||²)
            float grad_norm_sq = 0.0f;
            for (int i = 0; i < D_in * D_h; ++i) {
                grad_norm_sq += local_grad_W1.data()[i] * local_grad_W1.data()[i];
            }
            
            // Phase 1.2: Evolution-time-aware stability factor
            // Base factor: 1e-6, boosted by evolution time magnitude
            // sin(epsilon * phase_scale) provides periodic modulation aligned with Floquet dynamics
            // This prevents gradient explosion at large timesteps while allowing normal flow at small steps
            constexpr float BASE_STABILITY = 1e-6f;
            constexpr float PHASE_SCALE = 100.0f;  // Approximate Floquet phase scaling
            float phase_factor = 1.0f + 0.5f * std::abs(std::sin(epsilon * PHASE_SCALE));
            float adaptive_stability = BASE_STABILITY * phase_factor * (1.0f + 10.0f * epsilon);
            float stability_factor = 1.0f / (1.0f + adaptive_stability * grad_norm_sq);
            
            local_grad_W1 *= stability_factor;
            local_grad_b1 *= stability_factor;
            local_grad_W2 *= stability_factor;
            local_grad_b2 *= stability_factor;
            local_grad_W3 *= stability_factor;
            local_grad_b3 *= stability_factor;
            local_grad_W_out *= stability_factor;
            local_grad_b_out *= stability_factor;

            // Update global accumulators with thread-local results
            {
                absl::MutexLock lock(&mu_);
                grad_W1_acc += local_grad_W1;
                grad_b1_acc += local_grad_b1;
                grad_W2_acc += local_grad_W2;
                grad_b2_acc += local_grad_b2;
                grad_W3_acc += local_grad_W3;
                grad_b3_acc += local_grad_b3;
                grad_W_out_acc += local_grad_W_out;
                grad_b_out_acc += local_grad_b_out;
                grad_epsilon_param_acc += local_grad_epsilon_param; // Accumulate globally
            }
        };

        const std::size_t cost_per_unit =
            static_cast<std::size_t>(kBackwardCostPerUnit * D_in * D_h);
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(batch_size),
            cost_per_unit,
            work);

        // Write Final Accumulated Gradients to Output Tensors
        Map<MatrixXf>(grad_W1_tensor->flat<float>().data(), D_in, D_h) = grad_W1_acc;
        Map<VectorXf>(grad_b1_tensor->flat<float>().data(), D_h) = grad_b1_acc;
        Map<MatrixXf>(grad_W2_tensor->flat<float>().data(), D_h, D_h) = grad_W2_acc;
        Map<VectorXf>(grad_b2_tensor->flat<float>().data(), D_h) = grad_b2_acc;
        Map<MatrixXf>(grad_W3_tensor->flat<float>().data(), D_h, 1) = grad_W3_acc;
        grad_b3_tensor->scalar<float>()() = grad_b3_acc;
        Map<MatrixXf>(grad_W_out_tensor->flat<float>().data(), 2 * D_state, D_output) = grad_W_out_acc;
        Map<VectorXf>(grad_b_out_tensor->flat<float>().data(), D_output) = grad_b_out_acc;
        // grad_evolution_time_param_tensor is already allocated at output index 11.
        // We just need to assign the accumulated gradient to it.
        grad_evolution_time_param_tensor->scalar<float>()() = grad_epsilon_param_acc;
    }
private:
    absl::Mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("FusedHNNStepGrad").Device(DEVICE_CPU), FusedHNNStepGradOpCpu);

} // namespace tensorflow
