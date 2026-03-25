// saguaro.native/ops/quantum_galore_op.cc
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
 * @file quantum_galore_op.cc
 * @brief Phase 91: TensorFlow custom ops for Quantum GaLore
 *
 * Registers TensorFlow ops:
 * - QuantumGaLoreProject: Entropy-based dynamic rank projection
 * - QuantumGaLoreDeproject: Inverse projection reconstruction
 * - ComputeEffectiveRank: Standalone effective rank computation
 * - ComputeBlockInfluence: Taylor expansion influence scoring
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "quantum_galore_op.h"

using namespace tensorflow;

// =============================================================================
// Op Registration: QuantumGaLoreProject
// =============================================================================

REGISTER_OP("QuantumGaLoreProject")
    .Input("gradient: float32")
    .Input("eigenvalues: float32")
    .Input("rotation_matrix: float32")
    .Input("bias: float32")
    .Output("compressed: float32")
    .Output("actual_rank: int32")
    .Attr("max_rank: int = 32")
    .Attr("min_rank: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // gradient: [rows, cols]
        // Output compressed: [actual_rank, cols] or [rows, actual_rank]
        // We can't know actual_rank statically, so use dynamic shape
        int max_rank;
        TF_RETURN_IF_ERROR(c->GetAttr("max_rank", &max_rank));
        
        shape_inference::ShapeHandle gradient_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &gradient_shape));
        
        auto rows = c->Dim(gradient_shape, 0);
        auto cols = c->Dim(gradient_shape, 1);
        
        // Output shape depends on whether rows >= cols
        // Use [max_rank, cols] as upper bound
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        c->set_output(1, c->Scalar());
        
        return Status();
    })
    .Doc(R"doc(
Phase 91: Quantum GaLore gradient projection with dynamic rank.

Projects gradient to low-rank space using entropy-based rank selection
and quantum random features for stable projection.

gradient: Input gradient tensor [rows, cols].
eigenvalues: Pre-computed singular values of gradient (descending order).
rotation_matrix: Quantum random rotation parameters.
bias: Quantum random bias values.
compressed: Compressed gradient in low-rank space.
actual_rank: The effective rank used for compression.
max_rank: Maximum allowable projection rank.
min_rank: Minimum projection rank.
)doc");

// =============================================================================
// Op Registration: QuantumGaLoreDeproject
// =============================================================================

REGISTER_OP("QuantumGaLoreDeproject")
    .Input("compressed: float32")
    .Input("rotation_matrix: float32")
    .Input("bias: float32")
    .Input("original_shape: int32")
    .Output("gradient: float32")
    .Attr("row_projection: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Reconstruct original gradient shape from original_shape input
        // Output: [rows, cols]
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        return Status();
    })
    .Doc(R"doc(
Phase 91: Quantum GaLore gradient decompression.

Reconstructs full gradient from low-rank compressed representation
using quantum random feature adjoint mapping.

compressed: Compressed gradient in low-rank space.
rotation_matrix: Same rotation parameters used in projection.
bias: Same bias values used in projection.
original_shape: Original gradient shape [rows, cols] as 1D tensor.
gradient: Reconstructed full gradient [rows, cols].
row_projection: Whether original projection was along rows.
)doc");

// =============================================================================
// Op Registration: ComputeEffectiveRank
// =============================================================================

REGISTER_OP("ComputeEffectiveRank")
    .Input("eigenvalues: float32")
    .Output("effective_rank: int32")
    .Attr("max_rank: int = 32")
    .Attr("min_rank: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Phase 91: Compute effective rank from eigenvalue spectrum.

Uses Shannon entropy of normalized eigenvalue distribution:
effective_rank = exp(-Σ p_i log(p_i)) where p_i = λ_i / Σλ_j

eigenvalues: Sorted eigenvalues (descending order).
effective_rank: Scalar effective rank value.
max_rank: Maximum rank cap.
min_rank: Minimum rank floor.
)doc");

// =============================================================================
// Op Registration: ComputeBlockInfluence
// =============================================================================

REGISTER_OP("ComputeBlockInfluence")
    .Input("gradient_norms: float32")
    .Input("weight_norms: float32")
    .Output("influence_scores: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output same shape as inputs (per-block scores)
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 91: Compute Taylor expansion influence scores for block-wise allocation.

Influence = ||∇W||² / ||W||² approximates Fisher information diagonal.

gradient_norms: L2 norms of gradients per block.
weight_norms: L2 norms of weights per block.
influence_scores: Normalized influence scores (sum to 1).
)doc");

// =============================================================================
// Op Registration: AllocateBlockRanks
// =============================================================================

REGISTER_OP("AllocateBlockRanks")
    .Input("influence_scores: float32")
    .Output("rank_allocations: int32")
    .Attr("total_rank_budget: int = 256")
    .Attr("min_rank_per_block: int = 4")
    .Attr("critical_block_ids: list(int) = []")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 91: Allocate rank budget across blocks based on influence.

Critical blocks (first/last layers) receive minimum 1.5x average allocation.

influence_scores: Normalized influence scores per block.
rank_allocations: Allocated rank per block.
total_rank_budget: Total rank budget to distribute.
min_rank_per_block: Minimum rank per block.
critical_block_ids: Indices of critical blocks (e.g., [0, N-1]).
)doc");

// =============================================================================
// Op Kernel: QuantumGaLoreProjectOp
// =============================================================================

class QuantumGaLoreProjectOp : public OpKernel {
 public:
  explicit QuantumGaLoreProjectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_rank", &max_rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_rank", &min_rank_));
    OP_REQUIRES(ctx, max_rank_ > 0, errors::InvalidArgument("max_rank must be positive"));
    OP_REQUIRES(ctx, min_rank_ > 0, errors::InvalidArgument("min_rank must be positive"));
    OP_REQUIRES(ctx, min_rank_ <= max_rank_, 
                errors::InvalidArgument("min_rank must be <= max_rank"));
  }

  void Compute(OpKernelContext* ctx) override {
    // Get inputs
    const Tensor& gradient = ctx->input(0);
    const Tensor& eigenvalues = ctx->input(1);
    const Tensor& rotation_matrix = ctx->input(2);
    const Tensor& bias = ctx->input(3);

    // Validate shapes
    OP_REQUIRES(ctx, gradient.dims() == 2,
                errors::InvalidArgument("gradient must be 2D, got ", gradient.dims()));
    
    int rows = gradient.dim_size(0);
    int cols = gradient.dim_size(1);
    int num_eigenvalues = eigenvalues.NumElements();
    
    // Compute effective rank
    const float* eigenvalues_ptr = eigenvalues.flat<float>().data();
    int effective_rank = saguaro::quantum_galore::ComputeEffectiveRank(
        eigenvalues_ptr, num_eigenvalues, max_rank_, min_rank_);
    
    // Determine projection direction
    bool project_rows = (rows >= cols);
    
    // Allocate output
    Tensor* compressed = nullptr;
    Tensor* actual_rank_tensor = nullptr;
    
    if (project_rows) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({effective_rank, cols}), &compressed));
    } else {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rows, effective_rank}), &compressed));
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &actual_rank_tensor));
    
    // Set actual rank output
    actual_rank_tensor->scalar<int32>()() = effective_rank;
    
    // Apply quantum random projection
    const float* gradient_ptr = gradient.flat<float>().data();
    const float* rotation_ptr = rotation_matrix.flat<float>().data();
    const float* bias_ptr = bias.flat<float>().data();
    float* output_ptr = compressed->flat<float>().data();
    
    saguaro::quantum_galore::QuantumRandomProjection(
        gradient_ptr,
        rotation_ptr,
        bias_ptr,
        output_ptr,
        rows,
        cols,
        effective_rank,
        project_rows
    );
  }

 private:
  int max_rank_;
  int min_rank_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumGaLoreProject").Device(DEVICE_CPU),
    QuantumGaLoreProjectOp);

// =============================================================================
// Op Kernel: QuantumGaLoreDeprojectOp
// =============================================================================

class QuantumGaLoreDeprojectOp : public OpKernel {
 public:
  explicit QuantumGaLoreDeprojectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("row_projection", &row_projection_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& compressed = ctx->input(0);
    const Tensor& rotation_matrix = ctx->input(1);
    const Tensor& bias = ctx->input(2);
    const Tensor& original_shape = ctx->input(3);

    OP_REQUIRES(ctx, original_shape.NumElements() == 2,
                errors::InvalidArgument("original_shape must have 2 elements"));
    
    auto shape_vec = original_shape.flat<int32>();
    int rows = shape_vec(0);
    int cols = shape_vec(1);
    
    int rank;
    if (row_projection_) {
      rank = compressed.dim_size(0);
    } else {
      rank = compressed.dim_size(1);
    }

    // Allocate output
    Tensor* gradient = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rows, cols}), &gradient));

    // Apply decompression
    const float* compressed_ptr = compressed.flat<float>().data();
    const float* rotation_ptr = rotation_matrix.flat<float>().data();
    const float* bias_ptr = bias.flat<float>().data();
    float* output_ptr = gradient->flat<float>().data();
    
    saguaro::quantum_galore::QuantumRandomDeproject(
        compressed_ptr,
        rotation_ptr,
        bias_ptr,
        output_ptr,
        rows,
        cols,
        rank,
        row_projection_
    );
  }

 private:
  bool row_projection_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumGaLoreDeproject").Device(DEVICE_CPU),
    QuantumGaLoreDeprojectOp);

// =============================================================================
// Op Kernel: ComputeEffectiveRankOp
// =============================================================================

class ComputeEffectiveRankOp : public OpKernel {
 public:
  explicit ComputeEffectiveRankOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("max_rank", &max_rank_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_rank", &min_rank_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& eigenvalues = ctx->input(0);
    
    const float* eigenvalues_ptr = eigenvalues.flat<float>().data();
    int num_eigenvalues = eigenvalues.NumElements();
    
    int effective_rank = saguaro::quantum_galore::ComputeEffectiveRank(
        eigenvalues_ptr, num_eigenvalues, max_rank_, min_rank_);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output));
    output->scalar<int32>()() = effective_rank;
  }

 private:
  int max_rank_;
  int min_rank_;
};

REGISTER_KERNEL_BUILDER(
    Name("ComputeEffectiveRank").Device(DEVICE_CPU),
    ComputeEffectiveRankOp);

// =============================================================================
// Op Kernel: ComputeBlockInfluenceOp
// =============================================================================

class ComputeBlockInfluenceOp : public OpKernel {
 public:
  explicit ComputeBlockInfluenceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& gradient_norms = ctx->input(0);
    const Tensor& weight_norms = ctx->input(1);

    OP_REQUIRES(ctx, gradient_norms.shape() == weight_norms.shape(),
                errors::InvalidArgument("gradient_norms and weight_norms must have same shape"));

    int num_blocks = gradient_norms.NumElements();
    
    Tensor* influence_scores = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, gradient_norms.shape(), &influence_scores));

    const float* grad_ptr = gradient_norms.flat<float>().data();
    const float* weight_ptr = weight_norms.flat<float>().data();
    float* output_ptr = influence_scores->flat<float>().data();
    
    saguaro::quantum_galore::ComputeBlockInfluenceScores(
        grad_ptr, weight_ptr, output_ptr, num_blocks);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ComputeBlockInfluence").Device(DEVICE_CPU),
    ComputeBlockInfluenceOp);

// =============================================================================
// Op Kernel: AllocateBlockRanksOp
// =============================================================================

class AllocateBlockRanksOp : public OpKernel {
 public:
  explicit AllocateBlockRanksOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("total_rank_budget", &total_rank_budget_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("min_rank_per_block", &min_rank_per_block_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("critical_block_ids", &critical_block_ids_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& influence_scores = ctx->input(0);
    int num_blocks = influence_scores.NumElements();

    Tensor* rank_allocations = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, influence_scores.shape(), &rank_allocations));

    const float* influence_ptr = influence_scores.flat<float>().data();
    
    // Convert to int output
    std::vector<int> allocations(num_blocks);
    
    const int* critical_ptr = critical_block_ids_.empty() ? nullptr : critical_block_ids_.data();
    int num_critical = static_cast<int>(critical_block_ids_.size());
    
    saguaro::quantum_galore::AllocateBlockRanks(
        influence_ptr,
        num_blocks,
        total_rank_budget_,
        allocations.data(),
        min_rank_per_block_,
        critical_ptr,
        num_critical
    );
    
    auto output_ptr = rank_allocations->flat<int32>();
    for (int i = 0; i < num_blocks; ++i) {
      output_ptr(i) = allocations[i];
    }
  }

 private:
  int total_rank_budget_;
  int min_rank_per_block_;
  std::vector<int> critical_block_ids_;
};

REGISTER_KERNEL_BUILDER(
    Name("AllocateBlockRanks").Device(DEVICE_CPU),
    AllocateBlockRanksOp);

// =============================================================================
// Gradient Registration for QuantumGaLoreProject
// =============================================================================

// Register gradient for backward pass
REGISTER_OP("QuantumGaLoreProjectGrad")
    .Input("grad_compressed: float32")
    .Input("rotation_matrix: float32")
    .Input("bias: float32")
    .Input("original_gradient_shape: int32")
    .Output("grad_gradient: float32")
    .Attr("row_projection: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->MakeShape({c->UnknownDim(), c->UnknownDim()}));
        return Status();
    })
    .Doc("Gradient of QuantumGaLoreProject. Equivalent to QuantumGaLoreDeproject.");

class QuantumGaLoreProjectGradOp : public OpKernel {
 public:
  explicit QuantumGaLoreProjectGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("row_projection", &row_projection_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& grad_compressed = ctx->input(0);
    const Tensor& rotation_matrix = ctx->input(1);
    const Tensor& bias = ctx->input(2);
    const Tensor& original_shape = ctx->input(3);

    auto shape_vec = original_shape.flat<int32>();
    int rows = shape_vec(0);
    int cols = shape_vec(1);
    
    int rank;
    if (row_projection_) {
      rank = grad_compressed.dim_size(0);
    } else {
      rank = grad_compressed.dim_size(1);
    }

    Tensor* grad_gradient = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({rows, cols}), &grad_gradient));

    const float* grad_ptr = grad_compressed.flat<float>().data();
    const float* rotation_ptr = rotation_matrix.flat<float>().data();
    const float* bias_ptr = bias.flat<float>().data();
    float* output_ptr = grad_gradient->flat<float>().data();
    
    // Gradient is the deproject operation (adjoint of project)
    saguaro::quantum_galore::QuantumRandomDeproject(
        grad_ptr,
        rotation_ptr,
        bias_ptr,
        output_ptr,
        rows,
        cols,
        rank,
        row_projection_
    );
  }

 private:
  bool row_projection_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumGaLoreProjectGrad").Device(DEVICE_CPU),
    QuantumGaLoreProjectGradOp);
