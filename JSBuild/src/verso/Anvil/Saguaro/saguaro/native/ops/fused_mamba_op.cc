// saguaro.native/ops/fused_mamba_op.cc
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
 * @file fused_mamba_op.cc
 * @brief Mamba State-Space Model TensorFlow C++ kernel.
 *
 * Implements the core Mamba block as a single fused operation:
 *   1. Depthwise conv1d with causal padding
 *   2. SiLU activation
 *   3. Selective SSM scan
 *   4. Output gating
 *
 * Forward:
 *   x_conv = silu(conv1d(x_c, filter, bias))
 *   y = ssm_scan(x_conv, dt, A_log, B, C, D)
 *   output = y * silu(z)
 *
 * Uses SIMD-optimized helpers from fused_mamba_op.h.
 */

#include "fused_mamba_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <vector>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// FORWARD OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedMambaCore")
    .Input("x_c: float32")           // [batch, seq_len, d_inner] - conv input
    .Input("z: float32")             // [batch, seq_len, d_inner] - gate
    .Input("conv_filter: float32")   // [conv_dim, 1, d_inner]
    .Input("conv_bias: float32")     // [d_inner]
    .Input("dt: float32")            // [batch, seq_len, d_inner] - discretization
    .Input("a_log: float32")         // [d_inner, state_dim]
    .Input("b_proj: float32")        // [batch, seq_len, state_dim]
    .Input("c_proj: float32")        // [batch, seq_len, state_dim]
    .Input("d_skip: float32")        // [d_inner]
    .Input("vqc_angles: float32")    // [num_vqc_layers, 2] - VQC rotation angles (optional)
    .Output("output: float32")       // [batch, seq_len, d_inner]
    .Output("h_final: float32")      // [batch, d_inner, state_dim]
    .Output("conv_cache: float32")   // [batch, seq_len, d_inner] - for gradient
    .Attr("conv_dim: int")
    // Enhancement 1: VQC-Gated Selective Scan
    .Attr("use_vqc_gate: bool = false")
    .Attr("vqc_num_layers: int = 2")
    // Enhancement 2: Parallel SIMD Scan
    .Attr("use_parallel_scan: bool = true")
    .Attr("parallel_chunk_size: int = 256")
    // Enhancement 3: SSD Chunk Processing
    .Attr("use_ssd_chunks: bool = false")
    .Attr("ssd_chunk_size: int = 128")
    // Enhancement 4: MPS State (not in this op, uses separate MPS kernel)
    // Enhancement 5: Dynamic State Evolution
    .Attr("use_dynamic_state: bool = false")
    .Attr("dse_rank: int = 32")
    // Enhancement 6: Superposition Paths
    .Attr("use_superposition: bool = false")
    .Attr("superposition_dim: int = 4")
    .Attr("superposition_temperature: float = 1.0")
    // Phase 1: Neumann-Cayley Unitary Gates (Quantum Training Enhancement)
    .Attr("use_neumann_cayley: bool = false")
    .Attr("neumann_series_terms: int = 4")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle x_shape = c->input(0);
        ShapeHandle a_log_shape = c->input(5);
        
        DimensionHandle batch = c->Dim(x_shape, 0);
        (void)c->Dim(x_shape, 1);  // seq_len used implicitly via x_shape
        DimensionHandle d_inner = c->Dim(x_shape, 2);
        DimensionHandle state_dim = c->Dim(a_log_shape, 1);
        
        // output: [batch, seq_len, d_inner]
        c->set_output(0, x_shape);
        // h_final: [batch, d_inner, state_dim]
        c->set_output(1, c->MakeShape({batch, d_inner, state_dim}));
        // conv_cache: [batch, seq_len, d_inner]
        c->set_output(2, x_shape);
        
        return OkStatus();
    });

// =============================================================================
// GRADIENT OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedMambaCoreGrad")
    .Input("grad_output: float32")    // [batch, seq_len, d_inner]
    .Input("grad_h_final: float32")   // [batch, d_inner, state_dim]
    .Input("x_c: float32")            // [batch, seq_len, d_inner]
    .Input("z: float32")              // [batch, seq_len, d_inner]
    .Input("conv_filter: float32")    // [conv_dim, 1, d_inner]
    .Input("conv_bias: float32")      // [d_inner]
    .Input("dt: float32")             // [batch, seq_len, d_inner]
    .Input("a_log: float32")          // [d_inner, state_dim]
    .Input("b_proj: float32")         // [batch, seq_len, state_dim]
    .Input("c_proj: float32")         // [batch, seq_len, state_dim]
    .Input("d_skip: float32")         // [d_inner]
    .Input("conv_cache: float32")     // [batch, seq_len, d_inner]
    .Output("grad_x_c: float32")
    .Output("grad_z: float32")
    .Output("grad_conv_filter: float32")
    .Output("grad_conv_bias: float32")
    .Output("grad_dt: float32")
    .Output("grad_a_log: float32")
    .Output("grad_b_proj: float32")
    .Output("grad_c_proj: float32")
    .Output("grad_d_skip: float32")
    .Attr("conv_dim: int")
    .SetShapeFn([](InferenceContext* c) {
        // Gradients have same shapes as corresponding inputs
        c->set_output(0, c->input(2));   // grad_x_c
        c->set_output(1, c->input(3));   // grad_z
        c->set_output(2, c->input(4));   // grad_conv_filter
        c->set_output(3, c->input(5));   // grad_conv_bias
        c->set_output(4, c->input(6));   // grad_dt
        c->set_output(5, c->input(7));   // grad_a_log
        c->set_output(6, c->input(8));   // grad_b_proj
        c->set_output(7, c->input(9));   // grad_c_proj
        c->set_output(8, c->input(10));  // grad_d_skip
        return OkStatus();
    });

// =============================================================================
// FORWARD KERNEL IMPLEMENTATION
// =============================================================================

class FusedMambaCoreOp : public OpKernel {
 public:
    explicit FusedMambaCoreOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("conv_dim", &conv_dim_));
        // Enhancement attributes
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_vqc_gate", &use_vqc_gate_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_num_layers", &vqc_num_layers_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_parallel_scan", &use_parallel_scan_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("parallel_chunk_size", &parallel_chunk_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_ssd_chunks", &use_ssd_chunks_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ssd_chunk_size", &ssd_chunk_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_dynamic_state", &use_dynamic_state_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dse_rank", &dse_rank_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_superposition", &use_superposition_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("superposition_dim", &superposition_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("superposition_temperature", &superposition_temp_));
        // Phase 1: Neumann-Cayley
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_neumann_cayley", &use_neumann_cayley_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("neumann_series_terms", &neumann_series_terms_));
    }
    
    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& x_c = ctx->input(0);
        const Tensor& z = ctx->input(1);
        const Tensor& conv_filter = ctx->input(2);
        const Tensor& conv_bias = ctx->input(3);
        const Tensor& dt = ctx->input(4);
        const Tensor& a_log = ctx->input(5);
        const Tensor& b_proj = ctx->input(6);
        const Tensor& c_proj = ctx->input(7);
        const Tensor& d_skip = ctx->input(8);
        const Tensor& vqc_angles = ctx->input(9);  // VQC rotation angles
        
        // Get dimensions
        const int batch_size = x_c.dim_size(0);
        const int seq_len = x_c.dim_size(1);
        const int d_inner = x_c.dim_size(2);
        const int state_dim = a_log.dim_size(1);
        
        // Allocate outputs
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, seq_len, d_inner}), &output));
        
        Tensor* h_final = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            1, TensorShape({batch_size, d_inner, state_dim}), &h_final));
        
        Tensor* conv_cache = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            2, TensorShape({batch_size, seq_len, d_inner}), &conv_cache));
        
        // Get data pointers
        const float* x_c_data = x_c.flat<float>().data();
        const float* z_data = z.flat<float>().data();
        const float* filter_data = conv_filter.flat<float>().data();
        const float* bias_data = conv_bias.flat<float>().data();
        const float* dt_data = dt.flat<float>().data();
        const float* a_log_data = a_log.flat<float>().data();
        const float* b_data = b_proj.flat<float>().data();
        const float* c_data = c_proj.flat<float>().data();
        const float* d_data = d_skip.flat<float>().data();
        const float* vqc_angles_data = vqc_angles.flat<float>().data();
        
        float* output_data = output->flat<float>().data();
        float* h_final_data = h_final->flat<float>().data();
        float* conv_cache_data = conv_cache->flat<float>().data();
        
        // Step 1: Depthwise conv1d
        saguaro::ops::mamba_depthwise_conv1d(
            x_c_data, filter_data, bias_data,
            conv_cache_data, batch_size, seq_len, d_inner, conv_dim_);
        
        // Step 2: Apply SiLU activation in-place
        saguaro::ops::mamba_silu_inplace(conv_cache_data, 
                                          batch_size * seq_len * d_inner);
        
        // Step 2.5: VQC-gated delta if enabled (Enhancement 1)
        std::vector<float> dt_gated(batch_size * seq_len * d_inner);
        const float* dt_effective = dt_data;
        if (use_vqc_gate_ && vqc_angles.NumElements() >= vqc_num_layers_ * 2) {
            saguaro::ops::mamba_vqc_delta_gate(
                dt_data, vqc_angles_data, dt_gated.data(),
                batch_size * seq_len * d_inner, vqc_num_layers_);
            dt_effective = dt_gated.data();
        }
        
        // Step 3: SSM scan (with parallel enhancement if enabled)
        std::vector<float> ssm_out(batch_size * seq_len * d_inner);
        
        if (use_parallel_scan_) {
            // Enhancement 2: Parallel SIMD scan
            saguaro::ops::mamba_parallel_ssm_scan(
                conv_cache_data, a_log_data, dt_effective,
                b_data, c_data, d_data,
                ssm_out.data(), h_final_data,
                batch_size, seq_len, d_inner, state_dim,
                parallel_chunk_size_);
        } else if (use_ssd_chunks_) {
            // Enhancement 3: SSD chunk processing
            // Process sequence in chunks using SSD algorithm
            int num_chunks = (seq_len + ssd_chunk_size_ - 1) / ssd_chunk_size_;
            
            #pragma omp parallel for
            for (int b = 0; b < batch_size; ++b) {
                for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
                    int t_start = chunk_idx * ssd_chunk_size_;
                    int chunk_len = std::min(ssd_chunk_size_, seq_len - t_start);
                    
                    // Get chunk pointers
                    const float* x_chunk = conv_cache_data + b * seq_len * d_inner + t_start * d_inner;
                    const float* dt_chunk = dt_effective + b * seq_len * d_inner + t_start * d_inner;
                    const float* B_chunk = b_data + b * seq_len * state_dim + t_start * state_dim;
                    const float* C_chunk = c_data + b * seq_len * state_dim + t_start * state_dim;
                    float* y_chunk = ssm_out.data() + b * seq_len * d_inner + t_start * d_inner;
                    
                    saguaro::ops::mamba_ssd_intra_chunk(
                        x_chunk, a_log_data, dt_chunk, B_chunk, C_chunk, y_chunk,
                        chunk_len, d_inner, state_dim);
                }
            }
            
            // Add skip connection
            #pragma omp parallel for
            for (int64_t i = 0; i < batch_size * seq_len * d_inner; ++i) {
                int d = i % d_inner;
                ssm_out[i] += d_data[d] * conv_cache_data[i];
            }
        } else {
            // Original sequential scan
            saguaro::ops::mamba_ssm_scan(
                conv_cache_data, a_log_data, dt_effective,
                b_data, c_data, d_data,
                ssm_out.data(), h_final_data,
                batch_size, seq_len, d_inner, state_dim);
        }
        
        // Step 4: Gated output: output = y * silu(z)
        saguaro::ops::mamba_gated_output(
            ssm_out.data(), z_data, output_data,
            batch_size * seq_len * d_inner);
    }
    
 private:
    int conv_dim_;
    // Enhancement flags
    bool use_vqc_gate_;
    int vqc_num_layers_;
    bool use_parallel_scan_;
    int parallel_chunk_size_;
    bool use_ssd_chunks_;
    int ssd_chunk_size_;
    bool use_dynamic_state_;
    int dse_rank_;
    bool use_superposition_;
    int superposition_dim_;
    float superposition_temp_;
    // Phase 1: Neumann-Cayley
    bool use_neumann_cayley_;
    int neumann_series_terms_;
};

REGISTER_KERNEL_BUILDER(Name("FusedMambaCore").Device(DEVICE_CPU), FusedMambaCoreOp);

// =============================================================================
// GRADIENT KERNEL IMPLEMENTATION
// =============================================================================

class FusedMambaCoreGradOp : public OpKernel {
 public:
    explicit FusedMambaCoreGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("conv_dim", &conv_dim_));
    }
    
    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& grad_output = ctx->input(0);
        (void)ctx->input(1);  // grad_h_final: not used in backward pass (state is internal)
        const Tensor& x_c = ctx->input(2);
        const Tensor& z = ctx->input(3);
        const Tensor& conv_filter = ctx->input(4);
        const Tensor& conv_bias = ctx->input(5);
        const Tensor& dt = ctx->input(6);
        const Tensor& a_log = ctx->input(7);
        const Tensor& b_proj = ctx->input(8);
        const Tensor& c_proj = ctx->input(9);
        const Tensor& d_skip = ctx->input(10);
        const Tensor& conv_cache = ctx->input(11);
        
        // Get dimensions
        const int batch_size = x_c.dim_size(0);
        const int seq_len = x_c.dim_size(1);
        const int d_inner = x_c.dim_size(2);
        const int state_dim = a_log.dim_size(1);
        const int64_t total_size = batch_size * seq_len * d_inner;
        
        // Allocate output gradients
        Tensor* grad_x_c = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_c.shape(), &grad_x_c));
        
        Tensor* grad_z = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, z.shape(), &grad_z));
        
        Tensor* grad_conv_filter = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, conv_filter.shape(), &grad_conv_filter));
        
        Tensor* grad_conv_bias = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, conv_bias.shape(), &grad_conv_bias));
        
        Tensor* grad_dt = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, dt.shape(), &grad_dt));
        
        Tensor* grad_a_log = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(5, a_log.shape(), &grad_a_log));
        
        Tensor* grad_b_proj = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(6, b_proj.shape(), &grad_b_proj));
        
        Tensor* grad_c_proj = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(7, c_proj.shape(), &grad_c_proj));
        
        Tensor* grad_d_skip = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(8, d_skip.shape(), &grad_d_skip));
        
        // Initialize gradients to zero
        std::fill_n(grad_x_c->flat<float>().data(), grad_x_c->NumElements(), 0.0f);
        std::fill_n(grad_z->flat<float>().data(), grad_z->NumElements(), 0.0f);
        std::fill_n(grad_conv_filter->flat<float>().data(), grad_conv_filter->NumElements(), 0.0f);
        std::fill_n(grad_conv_bias->flat<float>().data(), grad_conv_bias->NumElements(), 0.0f);
        std::fill_n(grad_dt->flat<float>().data(), grad_dt->NumElements(), 0.0f);
        std::fill_n(grad_a_log->flat<float>().data(), grad_a_log->NumElements(), 0.0f);
        std::fill_n(grad_b_proj->flat<float>().data(), grad_b_proj->NumElements(), 0.0f);
        std::fill_n(grad_c_proj->flat<float>().data(), grad_c_proj->NumElements(), 0.0f);
        std::fill_n(grad_d_skip->flat<float>().data(), grad_d_skip->NumElements(), 0.0f);
        
        // Get data pointers
        const float* grad_out_data = grad_output.flat<float>().data();
        const float* z_data = z.flat<float>().data();
        const float* conv_cache_data = conv_cache.flat<float>().data();
        const float* x_c_data = x_c.flat<float>().data();
        const float* dt_data = dt.flat<float>().data();
        const float* a_log_data = a_log.flat<float>().data();
        const float* b_data = b_proj.flat<float>().data();
        const float* c_data = c_proj.flat<float>().data();
        const float* d_data = d_skip.flat<float>().data();
        const float* filter_data = conv_filter.flat<float>().data();
        
        float* grad_x_c_data = grad_x_c->flat<float>().data();
        float* grad_z_data = grad_z->flat<float>().data();
        float* grad_filter_data = grad_conv_filter->flat<float>().data();
        float* grad_bias_data = grad_conv_bias->flat<float>().data();
        float* grad_dt_data = grad_dt->flat<float>().data();
        float* grad_a_log_data = grad_a_log->flat<float>().data();
        float* grad_b_data = grad_b_proj->flat<float>().data();
        float* grad_c_data = grad_c_proj->flat<float>().data();
        float* grad_d_data = grad_d_skip->flat<float>().data();
        
        // Recompute SSM output for gating gradient
        std::vector<float> ssm_out(total_size);
        std::vector<float> h_states(batch_size * d_inner * state_dim, 0.0f);
        
        // Forward pass to get SSM output
        saguaro::ops::mamba_ssm_scan(
            conv_cache_data, a_log_data, dt_data,
            b_data, c_data, d_data,
            ssm_out.data(), nullptr,
            batch_size, seq_len, d_inner, state_dim);
        
        // Step 1: Gradient through gating: output = y * silu(z)
        // grad_y = grad_output * silu(z)
        // grad_z = grad_output * y * d_silu/d_z
        std::vector<float> grad_ssm(total_size);
        
        for (int64_t i = 0; i < total_size; ++i) {
            float z_val = z_data[i];
            float sig = 1.0f / (1.0f + std::exp(-z_val));
            float silu_z = z_val * sig;
            
            // grad_y = grad_output * silu(z)
            grad_ssm[i] = grad_out_data[i] * silu_z;
            
            // grad_z = grad_output * y * d_silu/d_z
            // d_silu/d_z = sig * (1 + z * (1 - sig))
            float d_silu = sig * (1.0f + z_val * (1.0f - sig));
            grad_z_data[i] = grad_out_data[i] * ssm_out[i] * d_silu;
        }
        
        // Step 2: Gradient through SSM scan (backward through time)
        std::vector<std::vector<float>> h_history(seq_len + 1);
        for (int t = 0; t <= seq_len; ++t) {
            h_history[t].resize(batch_size * d_inner * state_dim, 0.0f);
        }
        
        // Forward pass to collect hidden states
        for (int t = 0; t < seq_len; ++t) {
            for (int b = 0; b < batch_size; ++b) {
                for (int d = 0; d < d_inner; ++d) {
                    float dt_val = dt_data[b * seq_len * d_inner + t * d_inner + d];
                    
                    for (int n = 0; n < state_dim; ++n) {
                        int h_idx = b * d_inner * state_dim + d * state_dim + n;
                        float A_disc = std::exp(dt_val * a_log_data[d * state_dim + n]);
                        float B_val = b_data[b * seq_len * state_dim + t * state_dim + n];
                        float x_val = conv_cache_data[b * seq_len * d_inner + t * d_inner + d];
                        
                        h_history[t + 1][h_idx] = A_disc * h_history[t][h_idx] + B_val * x_val;
                    }
                }
            }
        }
        
        // Backward pass through SSM
        std::vector<float> grad_h(batch_size * d_inner * state_dim, 0.0f);
        std::vector<float> grad_conv(total_size, 0.0f);
        
        for (int t = seq_len - 1; t >= 0; --t) {
            for (int b = 0; b < batch_size; ++b) {
                for (int d = 0; d < d_inner; ++d) {
                    float grad_y = grad_ssm[b * seq_len * d_inner + t * d_inner + d];
                    float dt_val = dt_data[b * seq_len * d_inner + t * d_inner + d];
                    float x_val = conv_cache_data[b * seq_len * d_inner + t * d_inner + d];
                    
                    // grad_D += grad_y * x
                    grad_d_data[d] += grad_y * x_val;
                    
                    // grad_x from skip connection
                    float grad_x_skip = grad_y * d_data[d];
                    
                    for (int n = 0; n < state_dim; ++n) {
                        int h_idx = b * d_inner * state_dim + d * state_dim + n;
                        float A_disc = std::exp(dt_val * a_log_data[d * state_dim + n]);
                        float C_val = c_data[b * seq_len * state_dim + t * state_dim + n];
                        float B_val = b_data[b * seq_len * state_dim + t * state_dim + n];
                        float h_t = h_history[t + 1][h_idx];
                        float h_prev = h_history[t][h_idx];
                        
                        // grad_C += grad_y * h
                        grad_c_data[b * seq_len * state_dim + t * state_dim + n] += grad_y * h_t;
                        
                        // grad_h from output
                        grad_h[h_idx] += grad_y * C_val;
                        
                        // Backprop through h = A * h_prev + B * x
                        // grad_h_prev = grad_h * A
                        // grad_B = grad_h * x
                        // grad_x += grad_h * B
                        // grad_A = grad_h * h_prev
                        
                        float grad_h_curr = grad_h[h_idx];
                        
                        // grad_B
                        grad_b_data[b * seq_len * state_dim + t * state_dim + n] += grad_h_curr * x_val;
                        
                        // grad_x from SSM
                        grad_x_skip += grad_h_curr * B_val;
                        
                        // grad_A_log = grad_h * h_prev * A_disc * dt * a_log[d,n]
                        float grad_A = grad_h_curr * h_prev;
                        grad_a_log_data[d * state_dim + n] += grad_A * A_disc * dt_val;
                        
                        // grad_dt
                        grad_dt_data[b * seq_len * d_inner + t * d_inner + d] += 
                            grad_A * A_disc * a_log_data[d * state_dim + n];
                        
                        // Propagate to previous timestep
                        // We reset grad_h for next iteration
                        if (t > 0) {
                            h_history[t][h_idx] = grad_h_curr * A_disc;  // Reuse for grad_h_prev
                        }
                        grad_h[h_idx] = 0.0f;  // Reset for next timestep
                    }
                    
                    grad_conv[b * seq_len * d_inner + t * d_inner + d] = grad_x_skip;
                }
            }
            
            // Copy grad_h_prev
            if (t > 0) {
                for (int b = 0; b < batch_size; ++b) {
                    for (int d = 0; d < d_inner; ++d) {
                        for (int n = 0; n < state_dim; ++n) {
                            int h_idx = b * d_inner * state_dim + d * state_dim + n;
                            grad_h[h_idx] = h_history[t][h_idx];
                        }
                    }
                }
            }
        }
        
        // Step 3: Gradient through SiLU
        // conv_cache stores silu(conv_out), need original conv_out
        // For simplicity, recompute conv output
        std::vector<float> conv_out(total_size);
        saguaro::ops::mamba_depthwise_conv1d(
            x_c_data, filter_data, conv_bias.flat<float>().data(),
            conv_out.data(), batch_size, seq_len, d_inner, conv_dim_);
        
        std::vector<float> grad_conv_pre(total_size);
        for (int64_t i = 0; i < total_size; ++i) {
            float x = conv_out[i];
            float sig = 1.0f / (1.0f + std::exp(-x));
            float d_silu = sig * (1.0f + x * (1.0f - sig));
            grad_conv_pre[i] = grad_conv[i] * d_silu;
        }
        
        // Step 4: Gradient through depthwise conv1d
        // grad_x_c, grad_filter, grad_bias
        for (int b = 0; b < batch_size; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                for (int c = 0; c < d_inner; ++c) {
                    float grad = grad_conv_pre[b * seq_len * d_inner + t * d_inner + c];
                    
                    // grad_bias
                    grad_bias_data[c] += grad;
                    
                    // grad_filter and grad_x_c
                    for (int k = 0; k < conv_dim_; ++k) {
                        int t_in = t - (conv_dim_ - 1 - k);
                        if (t_in >= 0) {
                            int x_idx = b * seq_len * d_inner + t_in * d_inner + c;
                            int f_idx = k * d_inner + c;
                            
                            grad_filter_data[f_idx] += grad * x_c_data[x_idx];
                            grad_x_c_data[x_idx] += grad * filter_data[f_idx];
                        }
                    }
                }
            }
        }
    }
    
 private:
    int conv_dim_;
};

REGISTER_KERNEL_BUILDER(Name("FusedMambaCoreGrad").Device(DEVICE_CPU), FusedMambaCoreGradOp);

}  // namespace tensorflow
