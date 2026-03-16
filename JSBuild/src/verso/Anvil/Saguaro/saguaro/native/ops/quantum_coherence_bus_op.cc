// saguaro.native/ops/quantum_coherence_bus_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "quantum_coherence_bus_op.h"
#include "common/edition_limits.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: QCBInitialize
// =============================================================================

REGISTER_OP("QCBInitialize")
    .Output("entangled_state: float")     // [num_blocks, entanglement_dim]
    .Output("initial_fidelity: float")    // []
    .Attr("num_blocks: int = 6")
    .Attr("entanglement_dim: int = 64")
    .Attr("bus_slots: int = 8")
    .Attr("bidirectional: bool = true")
    .Attr("coherence_threshold: float = 0.9")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int num_blocks, entanglement_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("num_blocks", &num_blocks));
        TF_RETURN_IF_ERROR(c->GetAttr("entanglement_dim", &entanglement_dim));
        c->set_output(0, c->MakeShape({num_blocks, entanglement_dim}));
        c->set_output(1, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Phase 76: Initialize Quantum Coherence Bus with GHZ-like entanglement.

Creates maximally entangled state spanning all blocks in the HSMN architecture.

entangled_state: GHZ-like entangled mesh [num_blocks, entanglement_dim]
initial_fidelity: Initial entanglement fidelity (scalar)
)doc");

class QCBInitializeOp : public OpKernel {
 public:
  explicit QCBInitializeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_blocks", &config_.num_blocks));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_dim", &config_.entanglement_dim));
    OP_REQUIRES_OK(context, context->GetAttr("bus_slots", &config_.bus_slots));
    OP_REQUIRES_OK(context, context->GetAttr("bidirectional", &config_.bidirectional));
    OP_REQUIRES_OK(context, context->GetAttr("coherence_threshold", &config_.coherence_threshold));
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
  }

  void Compute(OpKernelContext* context) override {
    // HighNoon Lite Edition: Enforce reasoning block limit (max 24)
    SAGUARO_CHECK_REASONING_BLOCKS(context, config_.num_blocks);
    
    Tensor* entangled_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({config_.num_blocks, config_.entanglement_dim}), &entangled_state));
    
    Tensor* initial_fidelity = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &initial_fidelity));

    saguaro::qcb::InitializeCoherenceMesh(
        entangled_state->flat<float>().data(),
        config_
    );

    float fidelity = saguaro::qcb::MeasureEntanglementFidelity(
        entangled_state->flat<float>().data(),
        config_
    );
    initial_fidelity->scalar<float>()() = fidelity;
  }

 private:
  saguaro::qcb::QCBConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QCBInitialize").Device(DEVICE_CPU), QCBInitializeOp);

// =============================================================================
// OP REGISTRATION: QCBCoherentTransfer
// =============================================================================

REGISTER_OP("QCBCoherentTransfer")
    .Input("source_state: float")         // [batch, dim]
    .Input("entangled_state: float")      // [num_blocks, entanglement_dim]
    .Output("teleported_state: float")    // [batch, dim]
    .Attr("source_block: int")
    .Attr("target_block: int")
    .Attr("num_blocks: int = 6")
    .Attr("entanglement_dim: int = 64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 76: Coherent state transfer between blocks via QCB entanglement.

source_state: State to transfer [batch, dim]
entangled_state: Global entanglement mesh [num_blocks, entanglement_dim]

teleported_state: State at target block [batch, dim]
)doc");

class QCBCoherentTransferOp : public OpKernel {
 public:
  explicit QCBCoherentTransferOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("source_block", &source_block_));
    OP_REQUIRES_OK(context, context->GetAttr("target_block", &target_block_));
    OP_REQUIRES_OK(context, context->GetAttr("num_blocks", &config_.num_blocks));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_dim", &config_.entanglement_dim));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& source_state = context->input(0);
    const Tensor& entangled_state = context->input(1);

    const int batch_size = source_state.dim_size(0);
    const int dim = source_state.dim_size(1);

    Tensor* teleported_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, source_state.shape(), &teleported_state));

    // Cast away const for internal mutation (entanglement preserved)
    float* ent_ptr = const_cast<float*>(entangled_state.flat<float>().data());

    saguaro::qcb::CoherentTransfer(
        source_state.flat<float>().data(),
        source_block_,
        target_block_,
        ent_ptr,
        teleported_state->flat<float>().data(),
        config_,
        batch_size, dim
    );
  }

 private:
  int source_block_;
  int target_block_;
  saguaro::qcb::QCBConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QCBCoherentTransfer").Device(DEVICE_CPU), QCBCoherentTransferOp);

// =============================================================================
// OP REGISTRATION: QCBTeleportGradient
// =============================================================================

REGISTER_OP("QCBTeleportGradient")
    .Input("block_gradients: float")      // [num_blocks, num_params]
    .Input("entangled_state: float")      // [num_blocks, entanglement_dim]
    .Output("aggregated_gradient: float") // [num_params]
    .Attr("num_blocks: int = 6")
    .Attr("entanglement_dim: int = 64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle grads = c->input(0);
        if (c->RankKnown(grads) && c->Rank(grads) == 2) {
            c->set_output(0, c->MakeShape({c->Dim(grads, 1)}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 76: Teleport and aggregate gradients from all blocks via QCB.

block_gradients: Gradients from each block [num_blocks, num_params]
entangled_state: Global entanglement mesh [num_blocks, entanglement_dim]

aggregated_gradient: Combined gradient for optimizer [num_params]
)doc");

class QCBTeleportGradientOp : public OpKernel {
 public:
  explicit QCBTeleportGradientOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_blocks", &config_.num_blocks));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_dim", &config_.entanglement_dim));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& block_gradients = context->input(0);
    const Tensor& entangled_state = context->input(1);

    const int num_params = block_gradients.dim_size(1);

    Tensor* aggregated_gradient = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({num_params}), &aggregated_gradient));

    saguaro::qcb::AggregateAllGradients(
        block_gradients.flat<float>().data(),
        aggregated_gradient->flat<float>().data(),
        entangled_state.flat<float>().data(),
        config_,
        num_params
    );
  }

 private:
  saguaro::qcb::QCBConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QCBTeleportGradient").Device(DEVICE_CPU), QCBTeleportGradientOp);

// =============================================================================
// OP REGISTRATION: QCBSynchronizePhase
// =============================================================================

REGISTER_OP("QCBSynchronizePhase")
    .Input("entangled_state: float")      // [num_blocks, entanglement_dim]
    .Output("synchronized_state: float")   // [num_blocks, entanglement_dim]
    .Output("fidelity: float")            // []
    .Attr("num_blocks: int = 6")
    .Attr("entanglement_dim: int = 64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Phase 76: Synchronize quantum phase across all blocks in QCB.

entangled_state: Input entanglement mesh [num_blocks, entanglement_dim]

synchronized_state: Phase-aligned mesh [num_blocks, entanglement_dim]
fidelity: Post-synchronization fidelity (scalar)
)doc");

class QCBSynchronizePhaseOp : public OpKernel {
 public:
  explicit QCBSynchronizePhaseOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_blocks", &config_.num_blocks));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_dim", &config_.entanglement_dim));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& entangled_state = context->input(0);

    Tensor* synchronized_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, entangled_state.shape(), &synchronized_state));
    
    Tensor* fidelity = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &fidelity));

    // Copy input to output
    std::copy_n(entangled_state.flat<float>().data(),
                config_.num_blocks * config_.entanglement_dim,
                synchronized_state->flat<float>().data());

    // Synchronize phase
    saguaro::qcb::SynchronizeGlobalPhase(
        synchronized_state->flat<float>().data(),
        config_
    );

    // Measure fidelity
    float f = saguaro::qcb::MeasureEntanglementFidelity(
        synchronized_state->flat<float>().data(),
        config_
    );
    fidelity->scalar<float>()() = f;
  }

 private:
  saguaro::qcb::QCBConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QCBSynchronizePhase").Device(DEVICE_CPU), QCBSynchronizePhaseOp);

// =============================================================================
// OP REGISTRATION: QCBUpdateMesh
// =============================================================================

REGISTER_OP("QCBUpdateMesh")
    .Input("entangled_state: float")      // [num_blocks, entanglement_dim]
    .Input("block_states: float")         // [num_blocks, state_dim]
    .Output("updated_state: float")       // [num_blocks, entanglement_dim]
    .Attr("num_blocks: int = 6")
    .Attr("entanglement_dim: int = 64")
    .Attr("learning_rate: float = 0.01")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 76: Update QCB mesh with new block states while preserving coherence.

entangled_state: Current mesh [num_blocks, entanglement_dim]
block_states: New block states [num_blocks, state_dim]

updated_state: Updated mesh [num_blocks, entanglement_dim]
)doc");

class QCBUpdateMeshOp : public OpKernel {
 public:
  explicit QCBUpdateMeshOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_blocks", &config_.num_blocks));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_dim", &config_.entanglement_dim));
    OP_REQUIRES_OK(context, context->GetAttr("learning_rate", &learning_rate_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& entangled_state = context->input(0);
    const Tensor& block_states = context->input(1);

    const int state_dim = block_states.dim_size(1);

    Tensor* updated_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, entangled_state.shape(), &updated_state));

    // Copy input to output
    std::copy_n(entangled_state.flat<float>().data(),
                config_.num_blocks * config_.entanglement_dim,
                updated_state->flat<float>().data());

    // Update mesh
    saguaro::qcb::UpdateCoherenceMesh(
        updated_state->flat<float>().data(),
        block_states.flat<float>().data(),
        config_,
        state_dim,
        learning_rate_
    );
  }

 private:
  saguaro::qcb::QCBConfig config_;
  float learning_rate_;
};

REGISTER_KERNEL_BUILDER(Name("QCBUpdateMesh").Device(DEVICE_CPU), QCBUpdateMeshOp);

// =============================================================================
// PHASE 127: UNIFIED QUANTUM ENTANGLEMENT BUS OPS
// =============================================================================

REGISTER_OP("UnifiedQuantumBusPropagateEntanglement")
    .Input("block_states: float")         // [batch, num_blocks, dim]
    .Input("entanglement_strength: float") // [num_blocks, num_blocks]
    .Output("entangled_states: float")    // [batch, num_blocks, dim]
    .Output("coherence: float")           // [num_blocks, num_blocks]
    .Attr("num_blocks: int = 6")
    .Attr("bus_dim: int = 64")
    .Attr("mps_bond_dim: int = 32")
    .Attr("coherence_threshold: float = 0.85")
    .Attr("propagation_rate: float = 0.1")
    .Attr("use_adaptive: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));  // Same shape as block_states
        int num_blocks;
        TF_RETURN_IF_ERROR(c->GetAttr("num_blocks", &num_blocks));
        c->set_output(1, c->MakeShape({num_blocks, num_blocks}));
        return Status();
    })
    .Doc(R"doc(
Phase 127: Unified Quantum Entanglement Bus - Propagate Entanglement.

Propagates quantum correlations across blocks with O(n·d) complexity.
Uses SIMD-optimized entanglement-weighted state mixing.

block_states: Input block states [batch, num_blocks, dim]
entanglement_strength: Learnable entanglement matrix [num_blocks, num_blocks]

entangled_states: Entanglement-propagated states [batch, num_blocks, dim]
coherence: Pairwise coherence matrix [num_blocks, num_blocks]
)doc");

class UnifiedQuantumBusPropagateEntanglementOp : public OpKernel {
 public:
  explicit UnifiedQuantumBusPropagateEntanglementOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_blocks", &config_.num_blocks));
    OP_REQUIRES_OK(context, context->GetAttr("bus_dim", &config_.bus_dim));
    OP_REQUIRES_OK(context, context->GetAttr("mps_bond_dim", &config_.mps_bond_dim));
    OP_REQUIRES_OK(context, context->GetAttr("coherence_threshold", &config_.coherence_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("propagation_rate", &config_.propagation_rate));
    OP_REQUIRES_OK(context, context->GetAttr("use_adaptive", &config_.use_adaptive));
  }

  void Compute(OpKernelContext* context) override {
    // HighNoon Lite Edition: Enforce reasoning block limit (max 24)
    SAGUARO_CHECK_REASONING_BLOCKS(context, config_.num_blocks);

    const Tensor& block_states = context->input(0);
    const Tensor& entanglement_strength = context->input(1);

    const int batch_size = block_states.dim_size(0);
    const int num_blocks = block_states.dim_size(1);
    const int dim = block_states.dim_size(2);

    OP_REQUIRES(context, num_blocks == config_.num_blocks,
        errors::InvalidArgument("num_blocks mismatch: expected ", config_.num_blocks,
                                " but got ", num_blocks));

    // Allocate outputs
    Tensor* entangled_states = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, block_states.shape(), &entangled_states));

    Tensor* coherence = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({num_blocks, num_blocks}), &coherence));

    // Compute coherence
    saguaro::qcb::ComputeCoherence(
        block_states.flat<float>().data(),
        coherence->flat<float>().data(),
        config_,
        batch_size, dim
    );

    // Propagate entanglement
    saguaro::qcb::PropagateEntanglement(
        block_states.flat<float>().data(),
        entanglement_strength.flat<float>().data(),
        coherence->flat<float>().data(),
        entangled_states->flat<float>().data(),
        config_,
        batch_size, dim
    );
  }

 private:
  saguaro::qcb::UnifiedBusConfig config_;
};

REGISTER_KERNEL_BUILDER(
    Name("UnifiedQuantumBusPropagateEntanglement").Device(DEVICE_CPU),
    UnifiedQuantumBusPropagateEntanglementOp);

// =============================================================================
// OP REGISTRATION: UnifiedQuantumBusUpdateStrength
// =============================================================================

REGISTER_OP("UnifiedQuantumBusUpdateStrength")
    .Input("entanglement_strength: float")  // [num_blocks, num_blocks]
    .Input("coherence: float")              // [num_blocks, num_blocks]
    .Output("updated_strength: float")      // [num_blocks, num_blocks]
    .Attr("num_blocks: int = 6")
    .Attr("coherence_threshold: float = 0.85")
    .Attr("propagation_rate: float = 0.1")
    .Attr("use_adaptive: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 127: Update entanglement strength based on coherence feedback.

entanglement_strength: Current entanglement matrix [num_blocks, num_blocks]
coherence: Measured coherence matrix [num_blocks, num_blocks]

updated_strength: Updated entanglement matrix [num_blocks, num_blocks]
)doc");

class UnifiedQuantumBusUpdateStrengthOp : public OpKernel {
 public:
  explicit UnifiedQuantumBusUpdateStrengthOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_blocks", &config_.num_blocks));
    OP_REQUIRES_OK(context, context->GetAttr("coherence_threshold", &config_.coherence_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("propagation_rate", &config_.propagation_rate));
    OP_REQUIRES_OK(context, context->GetAttr("use_adaptive", &config_.use_adaptive));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& entanglement_strength = context->input(0);
    const Tensor& coherence = context->input(1);

    Tensor* updated_strength = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, entanglement_strength.shape(), &updated_strength));

    // Copy input to output
    std::copy_n(entanglement_strength.flat<float>().data(),
                config_.num_blocks * config_.num_blocks,
                updated_strength->flat<float>().data());

    // Update strength
    saguaro::qcb::UpdateEntanglementStrength(
        updated_strength->flat<float>().data(),
        coherence.flat<float>().data(),
        config_
    );
  }

 private:
  saguaro::qcb::UnifiedBusConfig config_;
};

REGISTER_KERNEL_BUILDER(
    Name("UnifiedQuantumBusUpdateStrength").Device(DEVICE_CPU),
    UnifiedQuantumBusUpdateStrengthOp);
