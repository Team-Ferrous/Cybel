// saguaro.native/ops/fused_hd_hierarchical_block_op.cc
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
 * @file fused_hd_hierarchical_block_op.cc
 * @brief Phase 800+: Fused HD Hierarchical Block TensorFlow custom operations.
 *
 * Registers TensorFlow ops for single-kernel HD hierarchical processing.
 * Combines QHD Spatial Block with adaptive chunking, CTQW aggregation,
 * and Phase 900.2 quantum feature map cross-level attention.
 *
 * Phase 900.2: Removed EMA blending, upgraded to quantum cross-level attention.
 *
 * SUPERSEDES: Multiple Python->C++ round trips in QHDHierarchicalBlock.
 * Single kernel call eliminates all overhead.
 *
 * Complexity: O(K × L × D log D) - unchanged from QHDSpatialBlock
 * Memory: +15-20% for pooled levels
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "fused_hd_hierarchical_block_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: HDHierarchicalBlockForward
// =============================================================================

REGISTER_OP("HDHierarchicalBlockForward")
    .Input("hd_input: float")           // [batch, seq_len, hd_dim]
    .Input("a_log: float")              // [state_dim]
    .Input("b_proj: float")             // [hd_dim, state_dim]
    .Input("c_proj: float")             // [hd_dim, state_dim]
    .Input("dt: float")                 // [seq_len, hd_dim]
    .Input("skip_proj: float")          // [hd_dim, hd_dim]
    .Input("amplitudes_real: float")    // [num_paths]
    .Input("amplitudes_imag: float")    // [num_paths]
    .Input("rotation_angles: float")    // [entanglement_depth, num_paths]
    .Input("level_embeddings: float")   // [hierarchical_levels + 1, hd_dim]
    .Input("cross_q_proj: float")       // [hd_dim, hd_dim]
    .Input("cross_k_proj: float")       // [hd_dim, hd_dim]
    .Input("cross_v_proj: float")       // [hd_dim, hd_dim]
    .Input("cross_o_proj: float")       // [hd_dim, hd_dim]
    .Input("uncertainty_trace: float")  // [batch]
    .Input("prev_state: float")         // [batch, state_size]
    .Input("qfm_rotation: float")       // [qfm_depth, hd_dim] Phase 900.2
    .Input("qfm_bias: float")           // [qfm_depth, hd_dim] Phase 900.2
    .Output("hd_output: float")         // [batch, seq_len, hd_dim]
    .Output("h_final: float")           // [batch, num_paths, state_dim, hd_dim]
    .Output("coherence: float")         // [batch]
    .Output("next_state: float")        // [batch, state_size]
    .Attr("hd_dim: int = 4096")
    .Attr("hidden_dim: int = 512")
    .Attr("state_dim: int = 16")
    .Attr("num_paths: int = 2")
    .Attr("entanglement_depth: int = 2")
    .Attr("entanglement_strength: float = 0.3")
    .Attr("hierarchical_levels: int = 2")
    .Attr("pooling_ratio: int = 4")
    .Attr("use_ctqw: bool = true")
    .Attr("use_cross_attention: bool = true")
    .Attr("ctqw_time: float = 1.0")
    .Attr("use_quantum_cross_attention: bool = true")  // Phase 900.2
    .Attr("cross_attn_qfm_depth: int = 4")             // Phase 900.2
    .Attr("min_chunk_size: int = 2")
    .Attr("max_chunk_size: int = 16")
    .Attr("boundary_threshold: float = 0.5")
    .Attr("use_streaming: bool = true")
    .Attr("max_memory_slots: int = 128")
    .Attr("uncertainty_threshold: float = 0.5")
    .Attr("training: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape = c->input(0);

        if (c->RankKnown(input_shape) && c->Rank(input_shape) == 3) {
            auto batch = c->Dim(input_shape, 0);
            auto seq_len = c->Dim(input_shape, 1);
            auto hd_dim = c->Dim(input_shape, 2);

            int state_dim, num_paths;
            c->GetAttr("state_dim", &state_dim);
            c->GetAttr("num_paths", &num_paths);

            // Output: [batch, seq_len, hd_dim]
            c->set_output(0, c->MakeShape({batch, seq_len, hd_dim}));

            // h_final: [batch, num_paths, state_dim, hd_dim]
            c->set_output(1, c->MakeShape({batch, num_paths, state_dim, hd_dim}));

            // coherence: [batch]
            c->set_output(2, c->MakeShape({batch}));
            // next_state: [batch, state_size]
            c->set_output(3, c->input(15));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
            c->set_output(2, c->UnknownShape());
            c->set_output(3, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Fused HD Hierarchical Block Forward Pass.

Single-kernel execution of QHD Spatial Block with hierarchical multi-scale reasoning.
Combines FFT-domain SSM, VQC entanglement, Born rule collapse, adaptive chunking,
CTQW aggregation, and Phase 900.2 quantum feature map cross-level attention.
Time is O(K * L * D log D) where K is num_paths. Memory overhead is 15-20 percent.
)doc");

// =============================================================================
// KERNEL: HDHierarchicalBlockForward
// =============================================================================

class HDHierarchicalBlockForwardOp : public OpKernel {
 public:
  explicit HDHierarchicalBlockForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(context, context->GetAttr("state_dim", &config_.state_dim));
    OP_REQUIRES_OK(context, context->GetAttr("num_paths", &config_.num_paths));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_depth", &config_.entanglement_depth));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_strength", &config_.entanglement_strength));
    OP_REQUIRES_OK(context, context->GetAttr("hierarchical_levels", &config_.hierarchical_levels));
    OP_REQUIRES_OK(context, context->GetAttr("pooling_ratio", &config_.pooling_ratio));
    OP_REQUIRES_OK(context, context->GetAttr("use_ctqw", &config_.use_ctqw));
    OP_REQUIRES_OK(context, context->GetAttr("use_cross_attention", &config_.use_cross_attention));
    OP_REQUIRES_OK(context, context->GetAttr("ctqw_time", &config_.ctqw_time));
    OP_REQUIRES_OK(context, context->GetAttr("use_quantum_cross_attention", &config_.use_quantum_cross_attention));
    OP_REQUIRES_OK(context, context->GetAttr("cross_attn_qfm_depth", &config_.cross_attn_qfm_depth));
    OP_REQUIRES_OK(context, context->GetAttr("min_chunk_size", &config_.min_chunk_size));
    OP_REQUIRES_OK(context, context->GetAttr("max_chunk_size", &config_.max_chunk_size));
    OP_REQUIRES_OK(context, context->GetAttr("boundary_threshold", &config_.boundary_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("use_streaming", &config_.use_streaming));
    OP_REQUIRES_OK(context, context->GetAttr("max_memory_slots", &config_.max_memory_slots));
    OP_REQUIRES_OK(context, context->GetAttr("uncertainty_threshold", &config_.uncertainty_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& hd_input = context->input(0);
    const Tensor& a_log = context->input(1);
    const Tensor& b_proj = context->input(2);
    const Tensor& c_proj = context->input(3);
    const Tensor& dt = context->input(4);
    const Tensor& skip_proj = context->input(5);
    const Tensor& amplitudes_real = context->input(6);
    const Tensor& amplitudes_imag = context->input(7);
    const Tensor& rotation_angles = context->input(8);
    const Tensor& level_embeddings = context->input(9);
    const Tensor& cross_q_proj = context->input(10);
    const Tensor& cross_k_proj = context->input(11);
    const Tensor& cross_v_proj = context->input(12);
    const Tensor& cross_o_proj = context->input(13);
    const Tensor& uncertainty_trace = context->input(14);
    const Tensor& prev_state = context->input(15);
    const Tensor& qfm_rotation = context->input(16);  // Phase 900.2
    const Tensor& qfm_bias = context->input(17);      // Phase 900.2

    // Validate shapes
    OP_REQUIRES(context, hd_input.dims() == 3,
                errors::InvalidArgument("hd_input must be 3D [batch, seq, hd_dim]"));
    OP_REQUIRES(context, a_log.dims() == 1,
                errors::InvalidArgument("a_log must be 1D [state_dim]"));
    OP_REQUIRES(context, amplitudes_real.dims() == 1,
                errors::InvalidArgument("amplitudes_real must be 1D [num_paths]"));
    OP_REQUIRES(context, rotation_angles.dims() == 2,
                errors::InvalidArgument("rotation_angles must be 2D [ent_depth, num_paths]"));
    OP_REQUIRES(context, level_embeddings.dims() == 2,
                errors::InvalidArgument("level_embeddings must be 2D [levels+1, hd_dim]"));
    OP_REQUIRES(context, cross_q_proj.dims() == 2,
                errors::InvalidArgument("cross_q_proj must be 2D [hd_dim, hd_dim]"));

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);
    const int hd_dim = hd_input.dim_size(2);
    const int state_dim = a_log.dim_size(0);
    const int num_paths = amplitudes_real.dim_size(0);

    // Validate dimensions match config
    OP_REQUIRES(context, hd_dim == config_.hd_dim,
                errors::InvalidArgument("hd_dim mismatch: ", hd_dim, " vs ", config_.hd_dim));
    OP_REQUIRES(context, state_dim == config_.state_dim,
                errors::InvalidArgument("state_dim mismatch: ", state_dim, " vs ", config_.state_dim));
    OP_REQUIRES(context, num_paths == config_.num_paths,
                errors::InvalidArgument("num_paths mismatch: ", num_paths, " vs ", config_.num_paths));

    // Allocate output tensors
    Tensor* hd_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len, hd_dim}), &hd_output));

    Tensor* h_final = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, num_paths, state_dim, hd_dim}), &h_final));

    Tensor* coherence = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        2, TensorShape({batch_size}), &coherence));

    Tensor* next_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        3, prev_state.shape(), &next_state));

    const float* uncertainty_ptr = uncertainty_trace.flat<float>().data();
    const float* prev_state_ptr = prev_state.flat<float>().data();
    float* next_state_ptr = next_state->flat<float>().data();

    // Call fused hierarchical forward kernel
    // (In streaming mode, seq_len = 1 and we update state recurrently)
    saguaro::hd_hierarchical::HDHierarchicalForward(
        hd_input.flat<float>().data(),
        a_log.flat<float>().data(),
        b_proj.flat<float>().data(),
        c_proj.flat<float>().data(),
        dt.flat<float>().data(),
        skip_proj.flat<float>().data(),
        amplitudes_real.flat<float>().data(),
        amplitudes_imag.flat<float>().data(),
        rotation_angles.flat<float>().data(),
        nullptr,  // walk_hamiltonian - UQHA bypass generates internally
        level_embeddings.flat<float>().data(),
        cross_q_proj.flat<float>().data(),
        cross_k_proj.flat<float>().data(),
        cross_v_proj.flat<float>().data(),
        cross_o_proj.flat<float>().data(),
        uncertainty_ptr,
        prev_state_ptr,
        hd_output->flat<float>().data(),
        h_final->flat<float>().data(),
        coherence->flat<float>().data(),
        next_state_ptr,
        config_,
        batch_size,
        seq_len,
        training_,
        // Phase 900.2: QFM weights for quantum cross-level attention
        config_.use_quantum_cross_attention ? qfm_rotation.flat<float>().data() : nullptr,
        config_.use_quantum_cross_attention ? qfm_bias.flat<float>().data() : nullptr
    );


    // If streaming, copy prev_state to next_state for now (full logic in forward)
    if (config_.use_streaming) {
        std::memcpy(next_state_ptr, prev_state_ptr, prev_state.TotalBytes());
    }
  }

 private:
  saguaro::hd_hierarchical::HDHierarchicalConfig config_;
  bool training_ = false;
};

REGISTER_KERNEL_BUILDER(Name("HDHierarchicalBlockForward").Device(DEVICE_CPU),
                        HDHierarchicalBlockForwardOp);

// =============================================================================
// OP REGISTRATION: HDHierarchicalBlockBackward
// =============================================================================

REGISTER_OP("HDHierarchicalBlockBackward")
    .Input("grad_output: float")        // [batch, seq_len, hd_dim]
    .Input("hd_input: float")           // [batch, seq_len, hd_dim]
    .Input("a_log: float")              // [state_dim]
    .Input("b_proj: float")             // [hd_dim, state_dim]
    .Input("c_proj: float")             // [hd_dim, state_dim]
    .Input("dt: float")                 // [seq_len, hd_dim]
    .Input("skip_proj: float")          // [hd_dim, hd_dim]
    .Input("amplitudes_real: float")    // [num_paths]
    .Input("amplitudes_imag: float")    // [num_paths]
    .Input("rotation_angles: float")    // [entanglement_depth, num_paths]
    .Input("level_embeddings: float")   // [hierarchical_levels + 1, hd_dim]
    .Input("cross_q_proj: float")       // [hd_dim, hd_dim]
    .Input("cross_k_proj: float")       // [hd_dim, hd_dim]
    .Input("cross_v_proj: float")       // [hd_dim, hd_dim]
    .Input("cross_o_proj: float")       // [hd_dim, hd_dim]
    .Output("grad_input: float")        // [batch, seq_len, hd_dim]
    .Output("grad_a_log: float")        // [state_dim]
    .Output("grad_b_proj: float")       // [hd_dim, state_dim]
    .Output("grad_c_proj: float")       // [hd_dim, state_dim]
    .Output("grad_dt: float")           // [seq_len, hd_dim]
    .Output("grad_skip: float")         // [hd_dim, hd_dim]
    .Output("grad_amplitudes_real: float")   // [num_paths]
    .Output("grad_amplitudes_imag: float")   // [num_paths]
    .Output("grad_rotation_angles: float")   // [entanglement_depth, num_paths]
    .Output("grad_level_embeddings: float")  // [hierarchical_levels + 1, hd_dim]
    .Output("grad_cross_q_proj: float")      // [hd_dim, hd_dim]
    .Output("grad_cross_k_proj: float")      // [hd_dim, hd_dim]
    .Output("grad_cross_v_proj: float")      // [hd_dim, hd_dim]
    .Output("grad_cross_o_proj: float")      // [hd_dim, hd_dim]
    .Attr("hd_dim: int = 4096")
    .Attr("hidden_dim: int = 512")
    .Attr("state_dim: int = 16")
    .Attr("num_paths: int = 2")
    .Attr("entanglement_depth: int = 2")
    .Attr("entanglement_strength: float = 0.3")
    .Attr("hierarchical_levels: int = 2")
    .Attr("pooling_ratio: int = 4")
    .Attr("use_ctqw: bool = true")
    .Attr("use_cross_attention: bool = true")
    .Attr("ctqw_time: float = 1.0")
    .Attr("use_quantum_cross_attention: bool = true")  // Phase 900.2
    .Attr("cross_attn_qfm_depth: int = 4")             // Phase 900.2
    .Attr("min_chunk_size: int = 2")
    .Attr("max_chunk_size: int = 16")
    .Attr("boundary_threshold: float = 0.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));   // grad_input same as hd_input
        c->set_output(1, c->input(2));   // grad_a_log same as a_log
        c->set_output(2, c->input(3));   // grad_b_proj same as b_proj
        c->set_output(3, c->input(4));   // grad_c_proj same as c_proj
        c->set_output(4, c->input(5));   // grad_dt same as dt
        c->set_output(5, c->input(6));   // grad_skip same as skip_proj
        c->set_output(6, c->input(7));   // grad_amplitudes_real
        c->set_output(7, c->input(8));   // grad_amplitudes_imag
        c->set_output(8, c->input(9));   // grad_rotation_angles
        c->set_output(9, c->input(10));  // grad_level_embeddings
        c->set_output(10, c->input(11)); // grad_cross_q_proj
        c->set_output(11, c->input(12)); // grad_cross_k_proj
        c->set_output(12, c->input(13)); // grad_cross_v_proj
        c->set_output(13, c->input(14)); // grad_cross_o_proj
        return Status();
    })
    .Doc("Fused HD Hierarchical Block Backward Pass - Gradient computation for BPTT.");

// =============================================================================
// KERNEL: HDHierarchicalBlockBackward
// =============================================================================

class HDHierarchicalBlockBackwardOp : public OpKernel {
 public:
  explicit HDHierarchicalBlockBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(context, context->GetAttr("state_dim", &config_.state_dim));
    OP_REQUIRES_OK(context, context->GetAttr("num_paths", &config_.num_paths));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_depth", &config_.entanglement_depth));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_strength", &config_.entanglement_strength));
    OP_REQUIRES_OK(context, context->GetAttr("hierarchical_levels", &config_.hierarchical_levels));
    OP_REQUIRES_OK(context, context->GetAttr("pooling_ratio", &config_.pooling_ratio));
    OP_REQUIRES_OK(context, context->GetAttr("use_ctqw", &config_.use_ctqw));
    OP_REQUIRES_OK(context, context->GetAttr("use_cross_attention", &config_.use_cross_attention));
    OP_REQUIRES_OK(context, context->GetAttr("ctqw_time", &config_.ctqw_time));
    OP_REQUIRES_OK(context, context->GetAttr("use_quantum_cross_attention", &config_.use_quantum_cross_attention));
    OP_REQUIRES_OK(context, context->GetAttr("cross_attn_qfm_depth", &config_.cross_attn_qfm_depth));
    OP_REQUIRES_OK(context, context->GetAttr("min_chunk_size", &config_.min_chunk_size));
    OP_REQUIRES_OK(context, context->GetAttr("max_chunk_size", &config_.max_chunk_size));
    OP_REQUIRES_OK(context, context->GetAttr("boundary_threshold", &config_.boundary_threshold));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& grad_output = context->input(0);
    const Tensor& hd_input = context->input(1);
    const Tensor& a_log = context->input(2);
    const Tensor& b_proj = context->input(3);
    const Tensor& c_proj = context->input(4);
    const Tensor& dt = context->input(5);
    const Tensor& skip_proj = context->input(6);
    const Tensor& amplitudes_real = context->input(7);
    const Tensor& amplitudes_imag = context->input(8);
    const Tensor& rotation_angles = context->input(9);
    const Tensor& level_embeddings = context->input(10);
    const Tensor& cross_q_proj = context->input(11);
    const Tensor& cross_k_proj = context->input(12);
    const Tensor& cross_v_proj = context->input(13);
    const Tensor& cross_o_proj = context->input(14);

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);

    // Allocate gradient tensors
    Tensor* grad_input = nullptr;
    Tensor* grad_a_log = nullptr;
    Tensor* grad_b_proj = nullptr;
    Tensor* grad_c_proj = nullptr;
    Tensor* grad_dt = nullptr;
    Tensor* grad_skip = nullptr;
    Tensor* grad_amplitudes_real = nullptr;
    Tensor* grad_amplitudes_imag = nullptr;
    Tensor* grad_rotation_angles = nullptr;
    Tensor* grad_level_embeddings = nullptr;
    Tensor* grad_cross_q_proj = nullptr;
    Tensor* grad_cross_k_proj = nullptr;
    Tensor* grad_cross_v_proj = nullptr;
    Tensor* grad_cross_o_proj = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, hd_input.shape(), &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, a_log.shape(), &grad_a_log));
    OP_REQUIRES_OK(context, context->allocate_output(2, b_proj.shape(), &grad_b_proj));
    OP_REQUIRES_OK(context, context->allocate_output(3, c_proj.shape(), &grad_c_proj));
    OP_REQUIRES_OK(context, context->allocate_output(4, dt.shape(), &grad_dt));
    OP_REQUIRES_OK(context, context->allocate_output(5, skip_proj.shape(), &grad_skip));
    OP_REQUIRES_OK(context, context->allocate_output(6, amplitudes_real.shape(), &grad_amplitudes_real));
    OP_REQUIRES_OK(context, context->allocate_output(7, amplitudes_imag.shape(), &grad_amplitudes_imag));
    OP_REQUIRES_OK(context, context->allocate_output(8, rotation_angles.shape(), &grad_rotation_angles));
    OP_REQUIRES_OK(context, context->allocate_output(9, level_embeddings.shape(), &grad_level_embeddings));
    OP_REQUIRES_OK(context, context->allocate_output(10, cross_q_proj.shape(), &grad_cross_q_proj));
    OP_REQUIRES_OK(context, context->allocate_output(11, cross_k_proj.shape(), &grad_cross_k_proj));
    OP_REQUIRES_OK(context, context->allocate_output(12, cross_v_proj.shape(), &grad_cross_v_proj));
    OP_REQUIRES_OK(context, context->allocate_output(13, cross_o_proj.shape(), &grad_cross_o_proj));

    // Allocate local buffer for grad_walk_hamiltonian (not exposed through TF op)
    std::vector<float> local_grad_walk_hamiltonian(config_.num_paths * config_.num_paths, 0.0f);
    
    // Call fused hierarchical backward kernel
    saguaro::hd_hierarchical::HDHierarchicalBackward(
        grad_output.flat<float>().data(),
        hd_input.flat<float>().data(),
        a_log.flat<float>().data(),
        b_proj.flat<float>().data(),
        c_proj.flat<float>().data(),
        dt.flat<float>().data(),
        skip_proj.flat<float>().data(),
        amplitudes_real.flat<float>().data(),
        amplitudes_imag.flat<float>().data(),
        rotation_angles.flat<float>().data(),
        nullptr,  // walk_hamiltonian - not passed to this backward op
        level_embeddings.flat<float>().data(),
        cross_q_proj.flat<float>().data(),
        cross_k_proj.flat<float>().data(),
        cross_v_proj.flat<float>().data(),
        cross_o_proj.flat<float>().data(),
        grad_input->flat<float>().data(),
        grad_a_log->flat<float>().data(),
        grad_b_proj->flat<float>().data(),
        grad_c_proj->flat<float>().data(),
        grad_dt->flat<float>().data(),
        grad_skip->flat<float>().data(),
        grad_amplitudes_real->flat<float>().data(),
        grad_amplitudes_imag->flat<float>().data(),
        grad_rotation_angles->flat<float>().data(),
        local_grad_walk_hamiltonian.data(),  // UQHA: Local buffer for gradient
        grad_level_embeddings->flat<float>().data(),
        grad_cross_q_proj->flat<float>().data(),
        grad_cross_k_proj->flat<float>().data(),
        grad_cross_v_proj->flat<float>().data(),
        grad_cross_o_proj->flat<float>().data(),
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::hd_hierarchical::HDHierarchicalConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("HDHierarchicalBlockBackward").Device(DEVICE_CPU),
                        HDHierarchicalBlockBackwardOp);
