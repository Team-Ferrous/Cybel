// saguaro.native/ops/intrinsic_plasticity_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "intrinsic_plasticity_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: CayleyParameterization
// =============================================================================

REGISTER_OP("CayleyParameterization")
    .Input("skew_params: float")          // [dim * (dim-1) / 2]
    .Output("unitary_weights: float")     // [dim, dim]
    .Attr("dim: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int dim;
        TF_RETURN_IF_ERROR(c->GetAttr("dim", &dim));
        c->set_output(0, c->MakeShape({dim, dim}));
        return Status();
    })
    .Doc(R"doc(
Phase 71: Cayley parameterization of unitary matrix.

Converts skew-symmetric parameters to orthogonal/unitary matrix:
  W = (I - A)(I + A)^{-1}

skew_params: Upper-triangular skew-symmetric params [dim * (dim-1) / 2]

unitary_weights: Orthogonal matrix [dim, dim]
)doc");

class CayleyParameterizationOp : public OpKernel {
 public:
  explicit CayleyParameterizationOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& skew_params = context->input(0);

    Tensor* unitary_weights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({dim_, dim_}), &unitary_weights));

    saguaro::iplast::CayleyParameterization(
        skew_params.flat<float>().data(),
        unitary_weights->flat<float>().data(),
        dim_
    );
  }

 private:
  int dim_;
};

REGISTER_KERNEL_BUILDER(Name("CayleyParameterization").Device(DEVICE_CPU),
                        CayleyParameterizationOp);

// =============================================================================
// OP REGISTRATION: EnforceUnitaryConstraint
// =============================================================================

REGISTER_OP("EnforceUnitaryConstraint")
    .Input("weights: float")              // [rows, cols]
    .Output("unitary_weights: float")     // [rows, cols]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 71: Enforce unitary constraint on weight matrix.

Projects weights to nearest orthonormal matrix via Gram-Schmidt.

weights: Input weight matrix [rows, cols]

unitary_weights: Orthonormalized matrix [rows, cols]
)doc");

class EnforceUnitaryConstraintOp : public OpKernel {
 public:
  explicit EnforceUnitaryConstraintOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& weights = context->input(0);

    const int rows = weights.dim_size(0);
    const int cols = weights.dim_size(1);

    Tensor* unitary_weights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, weights.shape(), &unitary_weights));

    // Copy weights
    std::copy_n(weights.flat<float>().data(), rows * cols,
                unitary_weights->flat<float>().data());

    // Enforce constraint
    saguaro::iplast::EnforceUnitaryConstraint(
        unitary_weights->flat<float>().data(),
        rows, cols
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("EnforceUnitaryConstraint").Device(DEVICE_CPU),
                        EnforceUnitaryConstraintOp);

// =============================================================================
// OP REGISTRATION: ProjectGradientTangent
// =============================================================================

REGISTER_OP("ProjectGradientTangent")
    .Input("gradient: float")             // [dim, dim]
    .Input("weights: float")              // [dim, dim]
    .Output("tangent_gradient: float")    // [dim, dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 71: Project gradient to tangent space of unitary manifold.

For W ∈ O(n): ∇_tang = ∇ - W * sym(W^T * ∇)

gradient: Euclidean gradient [dim, dim]
weights: Current unitary weights [dim, dim]

tangent_gradient: Projected gradient [dim, dim]
)doc");

class ProjectGradientTangentOp : public OpKernel {
 public:
  explicit ProjectGradientTangentOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& gradient = context->input(0);
    const Tensor& weights = context->input(1);

    const int dim = gradient.dim_size(0);

    Tensor* tangent_gradient = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, gradient.shape(), &tangent_gradient));

    // Copy gradient
    std::copy_n(gradient.flat<float>().data(), dim * dim,
                tangent_gradient->flat<float>().data());

    // Project to tangent space
    saguaro::iplast::ProjectGradientTangent(
        tangent_gradient->flat<float>().data(),
        weights.flat<float>().data(),
        dim
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("ProjectGradientTangent").Device(DEVICE_CPU),
                        ProjectGradientTangentOp);

// =============================================================================
// OP REGISTRATION: RetractToManifold
// =============================================================================

REGISTER_OP("RetractToManifold")
    .Input("weights: float")              // [dim, dim]
    .Input("direction: float")            // [dim, dim]
    .Output("updated_weights: float")     // [dim, dim]
    .Attr("step_size: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 71: Retract updated parameters back to unitary manifold.

Uses QR-based retraction: W_new = qr(W + step * direction).Q

weights: Current weights [dim, dim]
direction: Update direction [dim, dim]

updated_weights: Retracted weights on manifold [dim, dim]
)doc");

class RetractToManifoldOp : public OpKernel {
 public:
  explicit RetractToManifoldOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("step_size", &step_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& weights = context->input(0);
    const Tensor& direction = context->input(1);

    const int dim = weights.dim_size(0);

    Tensor* updated_weights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, weights.shape(), &updated_weights));

    // Copy weights
    std::copy_n(weights.flat<float>().data(), dim * dim,
                updated_weights->flat<float>().data());

    // Retract
    saguaro::iplast::RetractToManifold(
        updated_weights->flat<float>().data(),
        direction.flat<float>().data(),
        step_size_,
        dim
    );
  }

 private:
  float step_size_;
};

REGISTER_KERNEL_BUILDER(Name("RetractToManifold").Device(DEVICE_CPU),
                        RetractToManifoldOp);

// =============================================================================
// OP REGISTRATION: ComputePlasticityMetric
// =============================================================================

REGISTER_OP("ComputePlasticityMetric")
    .Input("weight_trajectory: float")    // [num_snapshots, num_params]
    .Output("plasticity: float")          // []
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Phase 71: Compute plasticity metric from weight trajectory.

Measures capacity to learn new information by tracking weight changes.

weight_trajectory: Weight snapshots [num_snapshots, num_params]

plasticity: Plasticity score in [0, 1] (scalar)
)doc");

class ComputePlasticityMetricOp : public OpKernel {
 public:
  explicit ComputePlasticityMetricOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& weight_trajectory = context->input(0);

    const int num_snapshots = weight_trajectory.dim_size(0);
    const int num_params = weight_trajectory.dim_size(1);

    Tensor* plasticity = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &plasticity));

    float metric = saguaro::iplast::ComputePlasticityMetric(
        weight_trajectory.flat<float>().data(),
        num_snapshots, num_params
    );
    plasticity->scalar<float>()() = metric;
  }
};

REGISTER_KERNEL_BUILDER(Name("ComputePlasticityMetric").Device(DEVICE_CPU),
                        ComputePlasticityMetricOp);

// =============================================================================
// OP REGISTRATION: MeasureLayerPlasticity
// =============================================================================

REGISTER_OP("MeasureLayerPlasticity")
    .Input("gradients: float")            // [num_params]
    .Input("weights: float")              // [num_params]
    .Output("plasticity: float")          // []
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Phase 71: Measure layer plasticity using relative gradient norm.

gradients: Current gradients [num_params]
weights: Current weights [num_params]

plasticity: Relative gradient norm (scalar)
)doc");

class MeasureLayerPlasticityOp : public OpKernel {
 public:
  explicit MeasureLayerPlasticityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& gradients = context->input(0);
    const Tensor& weights = context->input(1);

    const int num_params = gradients.NumElements();

    Tensor* plasticity = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &plasticity));

    float metric = saguaro::iplast::MeasureLayerPlasticity(
        gradients.flat<float>().data(),
        weights.flat<float>().data(),
        num_params
    );
    plasticity->scalar<float>()() = metric;
  }
};

REGISTER_KERNEL_BUILDER(Name("MeasureLayerPlasticity").Device(DEVICE_CPU),
                        MeasureLayerPlasticityOp);
