// src/ops/time_crystal_step_op.cc
// Copyright 2025 Verso Industries

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "ops/hnn_core_helpers.h"
#include "ops/time_crystal_step_op.h"

namespace tensorflow {

namespace {

struct TimeCrystalShapeConfig {
  int state_dim = 0;
  int input_dim = 0;
};

inline void RequireVector(OpKernelContext* context, const Tensor& tensor,
                          const char* name) {
  OP_REQUIRES(context, tensor.dims() == 1,
              errors::InvalidArgument(name, " must be 1-D, got ",
                                      tensor.shape().DebugString()));
}

inline void RequireMatrix(OpKernelContext* context, const Tensor& tensor,
                          const char* name) {
  OP_REQUIRES(context, tensor.dims() == 2,
              errors::InvalidArgument(name, " must be 2-D, got ",
                                      tensor.shape().DebugString()));
}

void ValidateAndExtractDims(OpKernelContext* context,
                            const Tensor& q_in,
                            const Tensor& p_in,
                            const Tensor& x_t,
                            const Tensor& W1,
                            const Tensor& b1,
                            const Tensor& W2,
                            const Tensor& b2,
                            const Tensor& W3,
                            const Tensor& b3,
                            const Tensor& W_out,
                            const Tensor& b_out,
                            TimeCrystalShapeConfig* config) {
  RequireVector(context, q_in, "q_in");
  RequireVector(context, p_in, "p_in");
  RequireVector(context, x_t, "x_t");

  const int state_dim = q_in.dim_size(0);
  const int input_dim = x_t.dim_size(0);

  OP_REQUIRES(context, state_dim > 0,
              errors::InvalidArgument("q_in must have positive length"));
  OP_REQUIRES(context, input_dim > 0,
              errors::InvalidArgument("x_t must have positive length"));
  OP_REQUIRES(context, p_in.dim_size(0) == state_dim,
              errors::InvalidArgument("p_in must match q_in length: got ",
                                      p_in.dim_size(0), " vs ", state_dim));

  const int h_input_dim = 2 * state_dim + input_dim;

  RequireMatrix(context, W1, "W1");
  const int h1_dim = W1.dim_size(1);
  OP_REQUIRES(context, W1.dim_size(0) == h_input_dim,
              errors::InvalidArgument("W1 rows must equal 2*state_dim + input_dim (",
                                      h_input_dim, "), got ", W1.dim_size(0)));

  RequireVector(context, b1, "b1");
  OP_REQUIRES(context, b1.dim_size(0) == h1_dim,
              errors::InvalidArgument("b1 must match W1 columns (",
                                      h1_dim, "), got ", b1.dim_size(0)));

  RequireMatrix(context, W2, "W2");
  const int h2_dim = W2.dim_size(1);
  OP_REQUIRES(context, W2.dim_size(0) == h1_dim,
              errors::InvalidArgument("W2 rows must equal W1 columns (",
                                      h1_dim, "), got ", W2.dim_size(0)));

  RequireVector(context, b2, "b2");
  OP_REQUIRES(context, b2.dim_size(0) == h2_dim,
              errors::InvalidArgument("b2 must match W2 columns (",
                                      h2_dim, "), got ", b2.dim_size(0)));

  RequireMatrix(context, W3, "W3");
  OP_REQUIRES(context, W3.dim_size(0) == h2_dim,
              errors::InvalidArgument("W3 rows must equal W2 columns (",
                                      h2_dim, "), got ", W3.dim_size(0)));
  OP_REQUIRES(context, W3.dim_size(1) == 1,
              errors::InvalidArgument("W3 must have output dimension 1, got ",
                                      W3.dim_size(1)));

  OP_REQUIRES(context, b3.dims() == 0,
              errors::InvalidArgument("b3 must be a scalar"));

  RequireMatrix(context, W_out, "W_out");
  OP_REQUIRES(context, W_out.dim_size(0) == 2 * state_dim,
              errors::InvalidArgument("W_out rows must equal 2*state_dim (",
                                      2 * state_dim, "), got ",
                                      W_out.dim_size(0)));
  OP_REQUIRES(context, W_out.dim_size(1) == input_dim,
              errors::InvalidArgument("W_out columns must equal input_dim (",
                                      input_dim, "), got ",
                                      W_out.dim_size(1)));

  RequireVector(context, b_out, "b_out");
  OP_REQUIRES(context, b_out.dim_size(0) == input_dim,
              errors::InvalidArgument("b_out must match input_dim (",
                                      input_dim, "), got ", b_out.dim_size(0)));

  config->state_dim = state_dim;
  config->input_dim = input_dim;
}

}  // namespace

REGISTER_OP("TimeCrystalStep")
    .Input("q_in: float")
    .Input("p_in: float")
    .Input("x_t: float")
    .Input("w1: float")
    .Input("b1: float")
    .Input("w2: float")
    .Input("b2: float")
    .Input("w3: float")
    .Input("b3: float")
    .Input("w_out: float")
    .Input("b_out: float")
    .Input("evolution_time: float")
    .Output("q_out: float")
    .Output("p_out: float")
    .Output("output_proj: float")
    // Phase 2: SPRK integrator order selection
    .Attr("sprk_order: int = 4")  // 4 or 6 for Yoshida order
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // q_in, p_in, x_t are vectors
        shape_inference::ShapeHandle q_in_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &q_in_shape));
        shape_inference::ShapeHandle p_in_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &p_in_shape));
        shape_inference::ShapeHandle x_t_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &x_t_shape));

        // All outputs have the same shape as their corresponding inputs
        c->set_output(0, c->input(0)); // q_out same shape as q_in
        c->set_output(1, c->input(1)); // p_out same shape as p_in
        c->set_output(2, c->input(2)); // output_proj same shape as x_t

        return OkStatus();
    });

TimeCrystalStepOp::TimeCrystalStepOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sprk_order", &sprk_order_));
    OP_REQUIRES(context, sprk_order_ == 4 || sprk_order_ == 6,
                errors::InvalidArgument("sprk_order must be 4 or 6, got ", sprk_order_));
}

void TimeCrystalStepOp::Compute(OpKernelContext* context) {
    // Grab the input tensors
    const Tensor& q_in_tensor = context->input(0);
    const Tensor& p_in_tensor = context->input(1);
    const Tensor& x_t_tensor = context->input(2);
    const Tensor& W1_tensor = context->input(3);
    const Tensor& b1_tensor = context->input(4);
    const Tensor& W2_tensor = context->input(5);
    const Tensor& b2_tensor = context->input(6);
    const Tensor& W3_tensor = context->input(7);
    const Tensor& b3_tensor = context->input(8);
    const Tensor& W_out_tensor = context->input(9);
    const Tensor& b_out_tensor = context->input(10);
    const Tensor& evolution_time_tensor = context->input(11);

    OP_REQUIRES(context, evolution_time_tensor.dims() == 0,
                errors::InvalidArgument("evolution_time must be a scalar"));

    TimeCrystalShapeConfig shape_config;
    ValidateAndExtractDims(
        context,
        q_in_tensor, p_in_tensor, x_t_tensor,
        W1_tensor, b1_tensor,
        W2_tensor, b2_tensor,
        W3_tensor, b3_tensor,
        W_out_tensor, b_out_tensor,
        &shape_config);
    const int state_dim = shape_config.state_dim;
    const int input_dim = shape_config.input_dim;

    // Allocate output tensors
    Tensor* q_out_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, q_in_tensor.shape(), &q_out_tensor));
    Tensor* p_out_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, p_in_tensor.shape(), &p_out_tensor));
    Tensor* output_proj_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, x_t_tensor.shape(), &output_proj_tensor));

    // Create local mutable copies for q and p to pass to the core function.
    Eigen::VectorXf q_vec = Eigen::Map<const Eigen::VectorXf>(q_in_tensor.flat<float>().data(), state_dim);
    Eigen::VectorXf p_vec = Eigen::Map<const Eigen::VectorXf>(p_in_tensor.flat<float>().data(), state_dim);
    const Eigen::Map<const Eigen::VectorXf> x_t_vec(x_t_tensor.flat<float>().data(), input_dim);

    Eigen::VectorXf output_proj(input_dim);
    output_proj.setZero();
    float h_initial = 0.f;
    float h_final = 0.f;

    const float evolution_time = evolution_time_tensor.scalar<float>()();

    // Call the core HNN step function with local copies
    hnn_core_step(q_vec, p_vec, x_t_vec,
                  W1_tensor, b1_tensor,
                  W2_tensor, b2_tensor,
                  W3_tensor, b3_tensor,
                  W_out_tensor, b_out_tensor,
                  evolution_time,
                  output_proj,
                  h_initial, h_final);

    // Copy results back to output tensors
    Eigen::Map<Eigen::VectorXf>(q_out_tensor->flat<float>().data(), state_dim) = q_vec;
    Eigen::Map<Eigen::VectorXf>(p_out_tensor->flat<float>().data(), state_dim) = p_vec;
    Eigen::Map<Eigen::VectorXf>(output_proj_tensor->flat<float>().data(), input_dim) = output_proj;
}

REGISTER_KERNEL_BUILDER(Name("TimeCrystalStep").Device(DEVICE_CPU), TimeCrystalStepOp);

}  // namespace tensorflow
