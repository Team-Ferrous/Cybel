// saguaro.native/ops/coconut_reservoir_op.cc
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
 * @file coconut_reservoir_op.cc
 * @brief Phase 39: Coconut Continuous Latent Reasoning TensorFlow ops.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "coconut_reservoir_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: CoconutThoughtRefinement
// =============================================================================

REGISTER_OP("CoconutThoughtRefinement")
    .Input("initial_hidden: float")       // [batch, seq, dim]
    .Input("w_in: float")                 // [reservoir_dim, dim]
    .Input("w_reservoir: float")          // [reservoir_dim, reservoir_dim]
    .Input("w_out: float")                // [dim, reservoir_dim]
    .Output("refined_output: float")      // [batch, seq, dim]
    .Output("reservoir_state: float")     // [batch*seq, reservoir_dim]
    .Attr("max_thought_steps: int = 8")
    .Attr("bfs_branches: int = 4")
    .Attr("halt_threshold: float = 0.9")
    .Attr("branch_alpha: float = 0.1")
    .Attr("reservoir_dim: int = 64")
    .Attr("dissipation_rate: float = 0.3")
    .Attr("use_echo_state: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle hidden = c->input(0);
        c->set_output(0, hidden);  // Same shape as input
        
        // reservoir_state shape
        if (c->RankKnown(hidden) && c->Rank(hidden) == 3) {
            int reservoir_dim;
            c->GetAttr("reservoir_dim", &reservoir_dim);
            auto batch = c->Dim(hidden, 0);
            auto seq = c->Dim(hidden, 1);
            // Flatten batch*seq
            c->set_output(1, c->MakeShape({c->UnknownDim(), reservoir_dim}));
        } else {
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 39: Coconut Continuous Latent Reasoning with Quantum Reservoir.

Implements BFS-style thought exploration with quantum reservoir context:
1. Initialize K parallel thought branches from hidden state
2. Iteratively refine branches using reservoir readout
3. Collapse to single output when halt confidence is reached

initial_hidden: Input hidden state [batch, seq, dim]
w_in: Reservoir input projection [reservoir_dim, dim]
w_reservoir: Reservoir recurrent weights [reservoir_dim, reservoir_dim]
w_out: Reservoir output projection [dim, reservoir_dim]

refined_output: Refined hidden state [batch, seq, dim]
reservoir_state: Final reservoir state [batch*seq, reservoir_dim]
)doc");

class CoconutThoughtRefinementOp : public OpKernel {
 public:
  explicit CoconutThoughtRefinementOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("max_thought_steps", &config_.max_thought_steps));
    OP_REQUIRES_OK(context, context->GetAttr("bfs_branches", &config_.bfs_branches));
    OP_REQUIRES_OK(context, context->GetAttr("halt_threshold", &config_.halt_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("branch_alpha", &config_.branch_alpha));
    OP_REQUIRES_OK(context, context->GetAttr("reservoir_dim", &config_.reservoir_dim));
    OP_REQUIRES_OK(context, context->GetAttr("dissipation_rate", &config_.dissipation_rate));
    OP_REQUIRES_OK(context, context->GetAttr("use_echo_state", &config_.use_echo_state));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& initial_hidden = context->input(0);
    const Tensor& w_in = context->input(1);
    const Tensor& w_reservoir = context->input(2);
    const Tensor& w_out = context->input(3);

    OP_REQUIRES(context, initial_hidden.dims() == 3,
                errors::InvalidArgument("initial_hidden must be 3D [batch, seq, dim]"));

    const int batch_size = initial_hidden.dim_size(0);
    const int seq_len = initial_hidden.dim_size(1);
    const int dim = initial_hidden.dim_size(2);
    const int total_batch = batch_size * seq_len;

    // Allocate outputs
    Tensor* refined_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, initial_hidden.shape(), &refined_output));
    
    Tensor* reservoir_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({total_batch, config_.reservoir_dim}), &reservoir_state));

    // Initialize reservoir to zero
    std::memset(reservoir_state->flat<float>().data(), 0,
                total_batch * config_.reservoir_dim * sizeof(float));

    // Run Coconut thought refinement
    saguaro::coconut::CoconutThoughtRefinement(
        initial_hidden.flat<float>().data(),
        refined_output->flat<float>().data(),
        reservoir_state->flat<float>().data(),
        w_in.flat<float>().data(),
        w_reservoir.flat<float>().data(),
        w_out.flat<float>().data(),
        config_,
        batch_size, seq_len, dim
    );
  }

 private:
  saguaro::coconut::CoconutConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("CoconutThoughtRefinement").Device(DEVICE_CPU),
                        CoconutThoughtRefinementOp);

// =============================================================================
// OP REGISTRATION: ReservoirUpdate
// =============================================================================

REGISTER_OP("QuantumReservoirUpdate")
    .Input("input: float")                // [batch, input_dim]
    .Input("reservoir: float")            // [batch, reservoir_dim]
    .Input("w_in: float")                 // [reservoir_dim, input_dim]
    .Input("w_reservoir: float")          // [reservoir_dim, reservoir_dim]
    .Output("new_reservoir: float")       // [batch, reservoir_dim]
    .Attr("dissipation_rate: float = 0.3")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status();
    })
    .Doc(R"doc(
Quantum reservoir state update with dissipation.

h_new = (1-γ) * W_res @ h + γ * W_in @ x

input: Input to reservoir [batch, input_dim]
reservoir: Current reservoir state [batch, reservoir_dim]
w_in: Input projection [reservoir_dim, input_dim]
w_reservoir: Recurrent weights [reservoir_dim, reservoir_dim]

new_reservoir: Updated reservoir state [batch, reservoir_dim]
)doc");

class QuantumReservoirUpdateOp : public OpKernel {
 public:
  explicit QuantumReservoirUpdateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dissipation_rate", &dissipation_rate_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& reservoir = context->input(1);
    const Tensor& w_in = context->input(2);
    const Tensor& w_reservoir = context->input(3);

    const int batch_size = input.dim_size(0);
    const int input_dim = input.dim_size(1);
    const int reservoir_dim = reservoir.dim_size(1);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, reservoir.shape(), &output));

    // Copy current reservoir to output
    std::memcpy(output->flat<float>().data(),
                reservoir.flat<float>().data(),
                reservoir.NumElements() * sizeof(float));

    saguaro::coconut::ReservoirUpdate(
        input.flat<float>().data(),
        output->flat<float>().data(),
        w_in.flat<float>().data(),
        w_reservoir.flat<float>().data(),
        dissipation_rate_,
        batch_size, input_dim, reservoir_dim
    );
  }

 private:
  float dissipation_rate_;
};

REGISTER_KERNEL_BUILDER(Name("QuantumReservoirUpdate").Device(DEVICE_CPU),
                        QuantumReservoirUpdateOp);

// =============================================================================
// OP REGISTRATION: ReservoirReadout
// =============================================================================

REGISTER_OP("QuantumReservoirReadout")
    .Input("reservoir: float")            // [batch, reservoir_dim]
    .Input("w_out: float")                // [output_dim, reservoir_dim]
    .Output("output: float")              // [batch, output_dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle reservoir = c->input(0);
        shape_inference::ShapeHandle w_out = c->input(1);
        if (c->RankKnown(reservoir) && c->RankKnown(w_out)) {
            auto batch = c->Dim(reservoir, 0);
            auto output_dim = c->Dim(w_out, 0);
            c->set_output(0, c->MakeShape({batch, output_dim}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Reservoir readout for context injection.

output = W_out @ reservoir

reservoir: Reservoir state [batch, reservoir_dim]
w_out: Output projection [output_dim, reservoir_dim]

output: Readout output [batch, output_dim]
)doc");

class QuantumReservoirReadoutOp : public OpKernel {
 public:
  explicit QuantumReservoirReadoutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& reservoir = context->input(0);
    const Tensor& w_out = context->input(1);

    const int batch_size = reservoir.dim_size(0);
    const int reservoir_dim = reservoir.dim_size(1);
    const int output_dim = w_out.dim_size(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, output_dim}), &output));

    saguaro::coconut::ReservoirReadout(
        reservoir.flat<float>().data(),
        w_out.flat<float>().data(),
        output->flat<float>().data(),
        batch_size, reservoir_dim, output_dim
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("QuantumReservoirReadout").Device(DEVICE_CPU),
                        QuantumReservoirReadoutOp);

// =============================================================================
// OP REGISTRATION: BranchCollapse
// =============================================================================

REGISTER_OP("CoconutBranchCollapse")
    .Input("thought_branches: float")     // [batch, num_branches, dim]
    .Output("collapsed: float")           // [batch, dim]
    .Output("confidence: float")          // [batch]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle branches = c->input(0);
        if (c->RankKnown(branches) && c->Rank(branches) == 3) {
            auto batch = c->Dim(branches, 0);
            auto dim = c->Dim(branches, 2);
            c->set_output(0, c->MakeShape({batch, dim}));
            c->set_output(1, c->MakeShape({batch}));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Collapse BFS thought branches to single output.

thought_branches: All branches [batch, num_branches, dim]

collapsed: Weighted collapsed output [batch, dim]
confidence: Halt confidence per batch [batch]
)doc");

class CoconutBranchCollapseOp : public OpKernel {
 public:
  explicit CoconutBranchCollapseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& branches = context->input(0);

    const int batch_size = branches.dim_size(0);
    const int num_branches = branches.dim_size(1);
    const int dim = branches.dim_size(2);

    Tensor* collapsed = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, dim}), &collapsed));
    
    Tensor* confidence = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size}), &confidence));

    // Collapse branches
    saguaro::coconut::CollapseBranchesToOutput(
        branches.flat<float>().data(),
        collapsed->flat<float>().data(),
        batch_size, num_branches, dim
    );

    // Compute confidence (needs per-batch, not global)
    float global_conf = saguaro::coconut::ComputeHaltConfidence(
        branches.flat<float>().data(),
        batch_size, num_branches, dim
    );
    
    // Fill with global confidence (simplified)
    float* conf_data = confidence->flat<float>().data();
    for (int b = 0; b < batch_size; ++b) {
      conf_data[b] = global_conf;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("CoconutBranchCollapse").Device(DEVICE_CPU),
                        CoconutBranchCollapseOp);
