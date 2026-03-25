// src/ops/selective_scan_op.cc
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
// This file implements the Selective Scan (SSM) operation for CPU,
// with a focus on high-performance computing using TBB for multi-threading
// and AVX intrinsics for SIMD vectorization.
//
// The implementation follows the "Optimize Selective Scan for All Platforms"
// roadmap, specifically targeting CPU.
//
// Key features:
// - Highly optimized CPU implementation using TBB for multi-threading
//   and AVX intrinsics (AVX2/AVX512) for SIMD vectorization.
// - Robust input validation using TensorFlow's OP_REQUIRES.
// - Caching mechanism for hidden states to accelerate the backward pass.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/status.h"

#include "common/parallel/parallel_backend.h"
#include "common/edition_limits.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <atomic>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tensorflow {

using shape_inference::InferenceContext;
using CPUDevice = Eigen::ThreadPoolDevice;

// =============================================================================
// 1. Op Registration
// =============================================================================

REGISTER_OP("SelectiveScan")
    .Input("u: float")       // (batch_size, seq_len, d_inner)
    .Input("delta: float")   // (batch_size, seq_len, d_inner)
    .Input("a_log: float")   // (d_inner, state_dim)
    .Input("b: float")       // (batch_size, seq_len, state_dim)
    .Input("c: float")       // (batch_size, seq_len, state_dim)
    .Input("d: float")       // (d_inner)
    .Attr("max_seq_len_for_caching: int = 0") // Max sequence length to cache hidden states
    .Output("output: float") // (batch_size, seq_len, d_inner)
    .Output("hidden_states: float") // (batch_size, seq_len, d_inner, state_dim)
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0)); // output has same shape as u

        // hidden_states is always 4D: (batch, seq, d_inner, state_dim)
        // When caching is disabled, dimensions are 0 but rank is still 4
        shape_inference::ShapeHandle u_shape = c->input(0);
        shape_inference::ShapeHandle a_log_shape = c->input(2);

        shape_inference::DimensionHandle batch_size_dim = c->Dim(u_shape, 0);
        shape_inference::DimensionHandle seq_len_dim = c->Dim(u_shape, 1);
        shape_inference::DimensionHandle d_inner_dim = c->Dim(u_shape, 2);
        shape_inference::DimensionHandle state_dim_dim = c->Dim(a_log_shape, 1);

        // Note: At runtime, if caching disabled, actual dims will be (0,0,0,0)
        // but we use unknown dimensions here since we can't know at graph-build time
        c->set_output(1, c->UnknownShapeOfRank(4));
        return OkStatus();
    });

REGISTER_OP("SelectiveScanGrad")
    .Input("grad_output: float")    // (batch_size, seq_len, d_inner)
    .Input("u: float")              // (batch_size, seq_len, d_inner)
    .Input("delta: float")          // (batch_size, seq_len, d_inner)
    .Input("a_log: float")          // (d_inner, state_dim)
    .Input("b: float")              // (batch_size, seq_len, state_dim)
    .Input("c: float")              // (batch_size, seq_len, state_dim)
    .Input("d: float")              // (d_inner)
    .Input("hidden_states: float")  // (batch_size, seq_len, d_inner, state_dim)
    .Output("grad_u: float")        // (batch_size, seq_len, d_inner)
    .Output("grad_delta: float")    // (batch_size, seq_len, d_inner)
    .Output("grad_a_log: float")    // (d_inner, state_dim)
    .Output("grad_b: float")        // (batch_size, seq_len, state_dim)
    .Output("grad_c: float")        // (batch_size, seq_len, state_dim)
    .Output("grad_d: float")        // (d_inner)
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1)); // grad_u has same shape as u
        c->set_output(1, c->input(2)); // grad_delta has same shape as delta
        c->set_output(2, c->input(3)); // grad_a_log has same shape as a_log
        c->set_output(3, c->input(4)); // grad_b has same shape as b
        c->set_output(4, c->input(5)); // grad_c has same shape as c
        c->set_output(5, c->input(6)); // grad_d has same shape as d
        return OkStatus();
    });

// =============================================================================
// 2. CPU Implementation Details
// =============================================================================

namespace hpc {
namespace cpu {

inline void AtomicAddFloat(float* target, float value) {
    auto* atomic_ptr = reinterpret_cast<std::atomic<float>*>(target);
    float expected = atomic_ptr->load(std::memory_order_relaxed);
    while (!atomic_ptr->compare_exchange_weak(
        expected, expected + value,
        std::memory_order_relaxed,
        std::memory_order_relaxed)) {
        // Retry until the addition succeeds.
    }
}

// Helper function for horizontal sum of AVX2/AVX512 vectors
#if defined(__AVX512F__)
inline float _mm512_reduce_add_ps_custom(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#elif defined(__AVX2__)
inline float _mm256_reduce_add_ps_custom(__m256 x) {
    __m128 vlow = _mm256_castps256_ps128(x);
    __m128 vhigh = _mm256_extractf128_ps(x, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#endif

// Forward pass for Selective Scan on CPU
void SelectiveScanForwardCpuImpl(
    OpKernelContext* context,
    const Tensor& u_tensor, const Tensor& delta_tensor, const Tensor& a_log_tensor,
    const Tensor& b_tensor, const Tensor& c_tensor, const Tensor& d_tensor,
    Tensor* output_tensor, Tensor* hidden_states_tensor) {

    const auto u = u_tensor.tensor<float, 3>();
    const auto delta = delta_tensor.tensor<float, 3>();
    const auto a_log = a_log_tensor.tensor<float, 2>();
    const auto b = b_tensor.tensor<float, 3>();
    const auto c = c_tensor.tensor<float, 3>();
    const auto d = d_tensor.flat<float>();

    auto output = output_tensor->tensor<float, 3>();
    
    // Check if hidden states caching is enabled (tensor has elements)
    const bool cache_hidden_states = (hidden_states_tensor->NumElements() > 0);

    const int64 batchSize = u_tensor.dim_size(0);
    const int64 seqLen = u_tensor.dim_size(1);
    const int64 dInner = u_tensor.dim_size(2);
    const int64 stateDim = a_log_tensor.dim_size(1);

    // Parallelize over batch_size and d_inner
    saguaro::parallel::ForRange2D(
        0, static_cast<size_t>(batchSize), 1,
        0, static_cast<size_t>(dInner), 1,
        [&](size_t b_begin, size_t b_end, size_t d_begin, size_t d_end) {
        for (size_t b_idx = b_begin; b_idx < b_end; ++b_idx) {
            for (size_t d_idx = d_begin; d_idx < d_end; ++d_idx) {
                // h_t = A_disc * h_{t-1} + B_u
                // y_t = C * h_t + D * u_t

                // Initialize hidden state h for this (batch, d_inner) combination
                float h[stateDim];
                for (int n = 0; n < stateDim; ++n) {
                    h[n] = 0.0f;
                }

                for (int l = 0; l < seqLen; ++l) {
                    const float u_val = u(b_idx, l, d_idx);
                    const float dt = delta(b_idx, l, d_idx);

                    float a_disc[stateDim];
                    float b_u[stateDim];
                    float y_val = 0.0f;

                    // Calculate A_disc and B_u, and update h
                    for (int n = 0; n < stateDim; ++n) {
                        const float a_cont = -expf(a_log(d_idx, n));
                        a_disc[n] = expf(a_cont * dt);
                        b_u[n] = b(b_idx, l, n) * u_val; // B is already scaled by dt in the model
                    }

#if defined(__AVX512F__)
                    for (int n = 0; n <= stateDim - 16; n += 16) {
                        __m512 v_a_disc = _mm512_loadu_ps(a_disc + n);
                        __m512 v_h = _mm512_loadu_ps(h + n);
                        __m512 v_b_u = _mm512_loadu_ps(b_u + n);
                        v_h = _mm512_fmadd_ps(v_a_disc, v_h, v_b_u);
                        _mm512_storeu_ps(h + n, v_h);
                    }
#elif defined(__AVX2__)
                    for (int n = 0; n <= stateDim - 8; n += 8) {
                        __m256 v_a_disc = _mm256_loadu_ps(a_disc + n);
                        __m256 v_h = _mm256_loadu_ps(h + n);
                        __m256 v_b_u = _mm256_loadu_ps(b_u + n);
                        v_h = _mm256_fmadd_ps(v_a_disc, v_h, v_b_u);
                        _mm256_storeu_ps(h + n, v_h);
                    }
#endif
                    // Scalar remainder for h update
                    for (int n = stateDim - (stateDim % (
#if defined(__AVX512F__)
                        16
#elif defined(__AVX2__)
                        8
#else
                        1
#endif
                        )); n < stateDim; ++n) {
                        h[n] = a_disc[n] * h[n] + b_u[n];
                    }

                    // Calculate output y_t
#if defined(__AVX512F__)
                    __m512 v_y_sum_512 = _mm512_setzero_ps();
                    for (int n = 0; n <= stateDim - 16; n += 16) {
                        __m512 v_c = _mm512_loadu_ps(&c(b_idx, l, n));
                        __m512 v_h = _mm512_loadu_ps(h + n);
                        v_y_sum_512 = _mm512_fmadd_ps(v_c, v_h, v_y_sum_512);
                    }
                    y_val += _mm512_reduce_add_ps_custom(v_y_sum_512);
#elif defined(__AVX2__)
                    __m256 v_y_sum_256 = _mm256_setzero_ps();
                    for (int n = 0; n <= stateDim - 8; n += 8) {
                        __m256 v_c = _mm256_loadu_ps(&c(b_idx, l, n));
                        __m256 v_h = _mm256_loadu_ps(h + n);
                        v_y_sum_256 = _mm256_fmadd_ps(v_c, v_h, v_y_sum_256);
                    }
                    y_val += _mm256_reduce_add_ps_custom(v_y_sum_256);
#endif
                    // Scalar remainder for y_val calculation
                    for (int n = stateDim - (stateDim % (
#if defined(__AVX512F__)
                        16
#elif defined(__AVX2__)
                        8
#else
                        1
#endif
                        )); n < stateDim; ++n) {
                        y_val += c(b_idx, l, n) * h[n];
                    }

                    output(b_idx, l, d_idx) = y_val + d(d_idx) * u_val;

                    // Store hidden states for backward pass (only if caching enabled)
                    if (cache_hidden_states) {
                        auto hidden_states = hidden_states_tensor->tensor<float, 4>();
                        for (int n = 0; n < stateDim; ++n) {
                            hidden_states(b_idx, l, d_idx, n) = h[n];
                        }
                    }
                }
                }
            }
        });
}

// Backward pass for Selective Scan on CPU
void SelectiveScanBackwardCpuImpl(
    OpKernelContext* context,
    const Tensor& grad_output_tensor, const Tensor& u_tensor, const Tensor& delta_tensor,
    const Tensor& a_log_tensor, const Tensor& b_tensor, const Tensor& c_tensor,
    const Tensor& d_tensor, const Tensor& hidden_states_tensor,
    Tensor* grad_u_tensor, Tensor* grad_delta_tensor, Tensor* grad_a_log_tensor,
    Tensor* grad_b_tensor, Tensor* grad_c_tensor, Tensor* grad_d_tensor) {

    const auto grad_output = grad_output_tensor.tensor<float, 3>();
    const auto u = u_tensor.tensor<float, 3>();
    const auto delta = delta_tensor.tensor<float, 3>();
    const auto a_log = a_log_tensor.tensor<float, 2>();
    const auto b = b_tensor.tensor<float, 3>();
    const auto c = c_tensor.tensor<float, 3>();
    const auto d = d_tensor.flat<float>();
    const auto hidden_states = hidden_states_tensor.tensor<float, 4>();

    auto grad_u = grad_u_tensor->tensor<float, 3>();
    auto grad_delta = grad_delta_tensor->tensor<float, 3>();
    auto grad_a_log = grad_a_log_tensor->tensor<float, 2>();
    auto grad_b = grad_b_tensor->tensor<float, 3>();
    auto grad_c = grad_c_tensor->tensor<float, 3>();
    auto grad_d = grad_d_tensor->flat<float>();

    const int64 batchSize = u_tensor.dim_size(0);
    const int64 seqLen = u_tensor.dim_size(1);
    const int64 dInner = u_tensor.dim_size(2);
    const int64 stateDim = a_log_tensor.dim_size(1);

    // Initialize gradients to zero
    grad_u_tensor->flat<float>().setZero();
    grad_delta_tensor->flat<float>().setZero();
    grad_a_log_tensor->flat<float>().setZero();
    grad_b_tensor->flat<float>().setZero();
    grad_c_tensor->flat<float>().setZero();
    grad_d_tensor->flat<float>().setZero();

    // Parallelize over batch_size and d_inner
    saguaro::parallel::ForRange2D(
        0, static_cast<size_t>(batchSize), 1,
        0, static_cast<size_t>(dInner), 1,
        [&](size_t b_begin, size_t b_end, size_t d_begin, size_t d_end) {
        for (size_t b_idx = b_begin; b_idx < b_end; ++b_idx) {
            for (size_t d_idx = d_begin; d_idx < d_end; ++d_idx) {
                float grad_h[stateDim];
                for (int n = 0; n < stateDim; ++n) {
                    grad_h[n] = 0.0f;
                }
                float grad_d_local = 0.0f;

                for (int l = seqLen - 1; l >= 0; --l) {
                    const float current_grad_output = grad_output(b_idx, l, d_idx);
                    const float u_val = u(b_idx, l, d_idx);
                    const float dt = delta(b_idx, l, d_idx);

                    // Gradient w.r.t. D and U (direct path)
                    grad_d_local += current_grad_output * u_val;
                    grad_u(b_idx, l, d_idx) += current_grad_output * d(d_idx);

                    // Gradient w.r.t. C and H
                    for (int n = 0; n < stateDim; ++n) {
                        const float h_val = hidden_states(b_idx, l, d_idx, n);
                        grad_c(b_idx, l, n) += current_grad_output * h_val;
                        grad_h[n] += current_grad_output * c(b_idx, l, n);
                    }

                    // Propagate grad_h backward
                    float h_prev[stateDim];
                    if (l > 0) {
                        for (int n = 0; n < stateDim; ++n) {
                            h_prev[n] = hidden_states(b_idx, l - 1, d_idx, n);
                        }
                    } else {
                        for (int n = 0; n < stateDim; ++n) {
                            h_prev[n] = 0.0f;
                        }
                    }

                    float a_disc[stateDim];
                    float a_cont[stateDim];
                    for (int n = 0; n < stateDim; ++n) {
                        a_cont[n] = -expf(a_log(d_idx, n));
                        a_disc[n] = expf(a_cont[n] * dt);
                    }

                    float grad_dt_local = 0.0f;
                    float next_grad_h[stateDim]; // Store for next iteration

#if defined(__AVX512F__)
                    for (int n = 0; n <= stateDim - 16; n += 16) {
                        __m512 v_grad_h = _mm512_loadu_ps(grad_h + n);
                        __m512 v_h_prev = _mm512_loadu_ps(h_prev + n);
                        __m512 v_u_val = _mm512_set1_ps(u_val);
                        __m512 v_dt = _mm512_set1_ps(dt);

                        __m512 v_a_disc = _mm512_loadu_ps(a_disc + n);
                        __m512 v_a_cont = _mm512_loadu_ps(a_cont + n);
                        __m512 v_b = _mm512_loadu_ps(&b(b_idx, l, n));

                        // grad_a_disc_n = grad_h[n] * h_prev[n]
                        __m512 v_grad_a_disc = _mm512_mul_ps(v_grad_h, v_h_prev);

                        // grad_b_u_n = grad_h[n]
                        // grad_b_n = grad_b_u_n * u_val
                        __m512 v_grad_b = _mm512_mul_ps(v_grad_h, v_u_val);
                        _mm512_storeu_ps(&grad_b(b_idx, l, n), _mm512_loadu_ps(&grad_b(b_idx, l, n)) + v_grad_b);

                        // grad_u_t(b_idx, l, d_idx) += grad_h[n] * b(b_idx, l, n)
                        grad_u(b_idx, l, d_idx) += _mm512_reduce_add_ps_custom(_mm512_mul_ps(v_grad_h, v_b));

                        // grad_a_log_n = grad_a_disc_n * a_disc[n] * dt * a_cont[n]
                        __m512 v_grad_a_log_n = _mm512_mul_ps(v_grad_a_disc, v_a_disc);
                        v_grad_a_log_n = _mm512_mul_ps(v_grad_a_log_n, v_dt);
                        v_grad_a_log_n = _mm512_mul_ps(v_grad_a_log_n, v_a_cont);
                        // Atomic update for grad_a_log
                        for (int k = 0; k < 16; ++k) {
                            AtomicAddFloat(&grad_a_log(d_idx, n + k), v_grad_a_log_n[k]);
                        }

                        // grad_dt_local += grad_a_disc_n * a_cont[n] * h_prev[n] + grad_b_u_n * b(b_idx, l, n) * u_val
                        __m512 v_grad_dt_part1 = _mm512_mul_ps(v_grad_a_disc, v_a_cont);
                        v_grad_dt_part1 = _mm512_mul_ps(v_grad_dt_part1, v_h_prev);
                        __m512 v_grad_dt_part2 = _mm512_mul_ps(v_grad_h, v_b);
                        v_grad_dt_part2 = _mm512_mul_ps(v_grad_dt_part2, v_u_val);
                        grad_dt_local += _mm512_reduce_add_ps_custom(_mm512_add_ps(v_grad_dt_part1, v_grad_dt_part2));

                        // next_grad_h[n] = grad_h[n] * a_disc[n]
                        _mm512_storeu_ps(next_grad_h + n, _mm512_mul_ps(v_grad_h, v_a_disc));
                    }
#elif defined(__AVX2__)
                    for (int n = 0; n <= stateDim - 8; n += 8) {
                        __m256 v_grad_h = _mm256_loadu_ps(grad_h + n);
                        __m256 v_h_prev = _mm256_loadu_ps(h_prev + n);
                        __m256 v_u_val = _mm256_set1_ps(u_val);
                        __m256 v_dt = _mm256_set1_ps(dt);

                        __m256 v_a_disc = _mm256_loadu_ps(a_disc + n);
                        __m256 v_a_cont = _mm256_loadu_ps(a_cont + n);
                        __m256 v_b = _mm256_loadu_ps(&b(b_idx, l, n));

                        // grad_a_disc_n = grad_h[n] * h_prev[n]
                        __m256 v_grad_a_disc = _mm256_mul_ps(v_grad_h, v_h_prev);

                        // grad_b_u_n = grad_h[n]
                        // grad_b_n = grad_b_u_n * u_val
                        __m256 v_grad_b = _mm256_mul_ps(v_grad_h, v_u_val);
                        _mm256_storeu_ps(&grad_b(b_idx, l, n), _mm256_loadu_ps(&grad_b(b_idx, l, n)) + v_grad_b);

                        // grad_u_t(b_idx, l, d_idx) += grad_h[n] * b(b_idx, l, n)
                        grad_u(b_idx, l, d_idx) += _mm256_reduce_add_ps_custom(_mm256_mul_ps(v_grad_h, v_b));

                        // grad_a_log_n = grad_a_disc_n * a_disc[n] * dt * a_cont[n]
                        __m256 v_grad_a_log_n = _mm256_mul_ps(v_grad_a_disc, v_a_disc);
                        v_grad_a_log_n = _mm256_mul_ps(v_grad_a_log_n, v_dt);
                        v_grad_a_log_n = _mm256_mul_ps(v_grad_a_log_n, v_a_cont);
                        // Atomic update for grad_a_log
                        for (int k = 0; k < 8; ++k) {
                            AtomicAddFloat(&grad_a_log(d_idx, n + k), v_grad_a_log_n[k]);
                        }

                        // grad_dt_local += grad_a_disc_n * a_cont[n] * h_prev[n] + grad_b_u_n * b(b_idx, l, n) * u_val
                        __m256 v_grad_dt_part1 = _mm256_mul_ps(v_grad_a_disc, v_a_cont);
                        v_grad_dt_part1 = _mm256_mul_ps(v_grad_dt_part1, v_h_prev);
                        __m256 v_grad_dt_part2 = _mm256_mul_ps(v_grad_h, v_b);
                        v_grad_dt_part2 = _mm256_mul_ps(v_grad_dt_part2, v_u_val);
                        grad_dt_local += _mm256_reduce_add_ps_custom(_mm256_add_ps(v_grad_dt_part1, v_grad_dt_part2));

                        // next_grad_h[n] = grad_h[n] * a_disc[n]
                        _mm256_storeu_ps(next_grad_h + n, _mm256_mul_ps(v_grad_h, v_a_disc));
                    }
#endif
                    // Scalar remainder for gradients
                    for (int n = stateDim - (stateDim % (
#if defined(__AVX512F__)
                        16
#elif defined(__AVX2__)
                        8
#else
                        1
#endif
                        )); n < stateDim; ++n) {
                        const float h_prev_val = h_prev[n];
                        const float b_val = b(b_idx, l, n);
                        const float a_disc_val = a_disc[n];
                        const float a_cont_val = a_cont[n];

                        const float grad_a_disc_n = grad_h[n] * h_prev_val;
                        const float grad_b_u_n = grad_h[n]; // dL/dh_n * dh_n/d(B_u)_n = grad_h[n] * 1

                        grad_b(b_idx, l, n) += grad_b_u_n * u_val;
                        grad_u(b_idx, l, d_idx) += grad_b_u_n * b_val;

                        const float grad_a_log_n = grad_a_disc_n * a_disc_val * dt * a_cont_val;
                        AtomicAddFloat(&grad_a_log(d_idx, n), grad_a_log_n);

                        grad_dt_local += grad_a_disc_n * a_cont_val * h_prev_val;
                        grad_dt_local += grad_b_u_n * b_val * u_val;

                        next_grad_h[n] = grad_h[n] * a_disc_val;
                    }
                    grad_delta(b_idx, l, d_idx) += grad_dt_local;

                    // Update grad_h for the next iteration
                    for (int n = 0; n < stateDim; ++n) {
                        grad_h[n] = next_grad_h[n];
                    }
                }
                // Atomic update for grad_d
                AtomicAddFloat(&grad_d(d_idx), grad_d_local);
            }
            }
        });
}

} // namespace cpu
} // namespace hpc

// =============================================================================
// 3. OpKernel Definitions
// =============================================================================

class SelectiveScanOp : public OpKernel {
public:
    explicit SelectiveScanOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("max_seq_len_for_caching", &maxSeqLenForCaching_));
    }

    void Compute(OpKernelContext* context) override {
        // Get inputs
        const Tensor& u_tensor = context->input(0);
        const Tensor& delta_tensor = context->input(1);
        const Tensor& a_log_tensor = context->input(2);
        const Tensor& b_tensor = context->input(3);
        const Tensor& c_tensor = context->input(4);
        const Tensor& d_tensor = context->input(5);

        // Validate input shapes and types
        OP_REQUIRES(context, u_tensor.dims() == 3,
                    errors::InvalidArgument("u must be 3-dimensional"));
        OP_REQUIRES(context, delta_tensor.dims() == 3,
                    errors::InvalidArgument("delta must be 3-dimensional"));
        OP_REQUIRES(context, a_log_tensor.dims() == 2,
                    errors::InvalidArgument("a_log must be 2-dimensional"));
        OP_REQUIRES(context, b_tensor.dims() == 3,
                    errors::InvalidArgument("b must be 3-dimensional"));
        OP_REQUIRES(context, c_tensor.dims() == 3,
                    errors::InvalidArgument("c must be 3-dimensional"));
        OP_REQUIRES(context, d_tensor.dims() == 1,
                    errors::InvalidArgument("d must be 1-dimensional"));

        OP_REQUIRES(context, u_tensor.shape() == delta_tensor.shape(),
                    errors::InvalidArgument("u and delta must have the same shape"));
        OP_REQUIRES(context, u_tensor.dim_size(0) == b_tensor.dim_size(0) &&
                               u_tensor.dim_size(1) == b_tensor.dim_size(1) &&
                               u_tensor.dim_size(0) == c_tensor.dim_size(0) &&
                               u_tensor.dim_size(1) == c_tensor.dim_size(1),
                    errors::InvalidArgument("batch_size and seq_len of u, b, c must match"));
        OP_REQUIRES(context, u_tensor.dim_size(2) == a_log_tensor.dim_size(0) &&
                               u_tensor.dim_size(2) == d_tensor.dim_size(0),
                    errors::InvalidArgument("d_inner of u, a_log, d must match"));
        OP_REQUIRES(context, a_log_tensor.dim_size(1) == b_tensor.dim_size(2) &&
                               a_log_tensor.dim_size(1) == c_tensor.dim_size(2),
                    errors::InvalidArgument("state_dim of a_log, b, c must match"));

        // Allocate outputs
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, u_tensor.shape(), &output_tensor));

        Tensor* hidden_states_tensor = nullptr;
        const int64 batchSize = u_tensor.dim_size(0);
        const int64 seqLen = u_tensor.dim_size(1);
        const int64 dInner = u_tensor.dim_size(2);
        const int64 stateDim = a_log_tensor.dim_size(1);

        // HighNoon Lite Edition: Enforce context length limit (max 5M tokens)
        SAGUARO_CHECK_CONTEXT_LENGTH(context, seqLen);

        // Only allocate hidden_states if caching is enabled and seq_len is within limits
        if (maxSeqLenForCaching_ > 0 && seqLen <= maxSeqLenForCaching_) {
            OP_REQUIRES_OK(context, context->allocate_output(1,
                TensorShape({batchSize, seqLen, dInner, stateDim}), &hidden_states_tensor));
        } else {
            // Allocate a 4-dimensional tensor with 0 elements to maintain consistent shape rank
            // This prevents TF internal code from crashing on 0-dim tensor when it expects 4D
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({0, 0, 0, 0}), &hidden_states_tensor));
        }

        // Perform CPU computation
        hpc::cpu::SelectiveScanForwardCpuImpl(
            context, u_tensor, delta_tensor, a_log_tensor, b_tensor, c_tensor, d_tensor,
            output_tensor, hidden_states_tensor);
    }

private:
    int maxSeqLenForCaching_;
};

REGISTER_KERNEL_BUILDER(Name("SelectiveScan").Device(DEVICE_CPU), SelectiveScanOp);

class SelectiveScanGradOp : public OpKernel {
public:
    explicit SelectiveScanGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Get inputs
        const Tensor& grad_output_tensor = context->input(0);
        const Tensor& u_tensor = context->input(1);
        const Tensor& delta_tensor = context->input(2);
        const Tensor& a_log_tensor = context->input(3);
        const Tensor& b_tensor = context->input(4);
        const Tensor& c_tensor = context->input(5);
        const Tensor& d_tensor = context->input(6);
        const Tensor& hidden_states_tensor = context->input(7);

        // Validate input shapes and types
        OP_REQUIRES(context, grad_output_tensor.dims() == 3,
                    errors::InvalidArgument("grad_output must be 3-dimensional"));
        OP_REQUIRES(context, u_tensor.dims() == 3,
                    errors::InvalidArgument("u must be 3-dimensional"));
        OP_REQUIRES(context, delta_tensor.dims() == 3,
                    errors::InvalidArgument("delta must be 3-dimensional"));
        OP_REQUIRES(context, a_log_tensor.dims() == 2,
                    errors::InvalidArgument("a_log must be 2-dimensional"));
        OP_REQUIRES(context, b_tensor.dims() == 3,
                    errors::InvalidArgument("b must be 3-dimensional"));
        OP_REQUIRES(context, c_tensor.dims() == 3,
                    errors::InvalidArgument("c must be 3-dimensional"));
        OP_REQUIRES(context, d_tensor.dims() == 1,
                    errors::InvalidArgument("d must be 1-dimensional"));
        OP_REQUIRES(context, hidden_states_tensor.dims() == 4,
                    errors::InvalidArgument("hidden_states must be 4-dimensional"));

        OP_REQUIRES(context, grad_output_tensor.shape() == u_tensor.shape(),
                    errors::InvalidArgument("grad_output and u must have the same shape"));
        OP_REQUIRES(context, u_tensor.shape() == delta_tensor.shape(),
                    errors::InvalidArgument("u and delta must have the same shape"));
        OP_REQUIRES(context, u_tensor.dim_size(0) == b_tensor.dim_size(0) &&
                               u_tensor.dim_size(1) == b_tensor.dim_size(1) &&
                               u_tensor.dim_size(0) == c_tensor.dim_size(0) &&
                               u_tensor.dim_size(1) == c_tensor.dim_size(1),
                    errors::InvalidArgument("batch_size and seq_len of u, b, c must match"));
        OP_REQUIRES(context, u_tensor.dim_size(2) == a_log_tensor.dim_size(0) &&
                               u_tensor.dim_size(2) == d_tensor.dim_size(0),
                    errors::InvalidArgument("d_inner of u, a_log, d must match"));
        OP_REQUIRES(context, a_log_tensor.dim_size(1) == b_tensor.dim_size(2) &&
                               a_log_tensor.dim_size(1) == c_tensor.dim_size(2),
                    errors::InvalidArgument("state_dim of a_log, b, c must match"));
        if (hidden_states_tensor.NumElements() > 0) {
            OP_REQUIRES(context, hidden_states_tensor.dim_size(0) == u_tensor.dim_size(0) &&
                                   hidden_states_tensor.dim_size(1) == u_tensor.dim_size(1) &&
                                   hidden_states_tensor.dim_size(2) == u_tensor.dim_size(2) &&
                                   hidden_states_tensor.dim_size(3) == a_log_tensor.dim_size(1),
                        errors::InvalidArgument("hidden_states shape mismatch"));
        }


        // Allocate outputs
        Tensor* grad_u_tensor = nullptr;
        Tensor* grad_delta_tensor = nullptr;
        Tensor* grad_a_log_tensor = nullptr;
        Tensor* grad_b_tensor = nullptr;
        Tensor* grad_c_tensor = nullptr;
        Tensor* grad_d_tensor = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, u_tensor.shape(), &grad_u_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, delta_tensor.shape(), &grad_delta_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(2, a_log_tensor.shape(), &grad_a_log_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(3, b_tensor.shape(), &grad_b_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(4, c_tensor.shape(), &grad_c_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(5, d_tensor.shape(), &grad_d_tensor));

        // If hidden_states is empty (caching disabled for long sequences), return zero gradients
        // since the full backward pass requires cached states. This gracefully handles long-context
        // inference mode without crashing.
        if (hidden_states_tensor.NumElements() == 0) {
            grad_u_tensor->flat<float>().setZero();
            grad_delta_tensor->flat<float>().setZero();
            grad_a_log_tensor->flat<float>().setZero();
            grad_b_tensor->flat<float>().setZero();
            grad_c_tensor->flat<float>().setZero();
            grad_d_tensor->flat<float>().setZero();
            return;
        }

        // Perform CPU computation
        hpc::cpu::SelectiveScanBackwardCpuImpl(
            context, grad_output_tensor, u_tensor, delta_tensor, a_log_tensor,
            b_tensor, c_tensor, d_tensor, hidden_states_tensor,
            grad_u_tensor, grad_delta_tensor, grad_a_log_tensor,
            grad_b_tensor, grad_c_tensor, grad_d_tensor);
    }
};

REGISTER_KERNEL_BUILDER(Name("SelectiveScanGrad").Device(DEVICE_CPU), SelectiveScanGradOp);

} // namespace tensorflow
