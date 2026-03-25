// saguaro.native/ops/tensor_network_kalman_op.cc
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
 * @file tensor_network_kalman_op.cc
 * @brief UQHA Phase 1001: TensorFlow custom ops for TensorNetworkKalmanFilter.
 * 
 * Exposes TT-compressed Kalman filter with O(n*r^2) time-complexity to Python.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <vector>
#include <memory>

#include "controllers/tensor_network_kalman.h"

using namespace tensorflow;
using namespace saguaro::controllers;

// =============================================================================
// Global TNKF instance cache for stateful operations
// =============================================================================
namespace {
    std::map<std::string, std::unique_ptr<TensorNetworkKalmanFilter>> g_tnkf_instances;
    std::mutex g_tnkf_mutex;
}

// =============================================================================
// OP REGISTRATION: TNKFInit
// =============================================================================

REGISTER_OP("TNKFInit")
    .Input("state_trans: float")     // [state_dim, state_dim] - A matrix
    .Input("control_mat: float")     // [state_dim, input_dim] - B matrix
    .Input("obs_mat: float")         // [output_dim, state_dim] - C matrix
    .Input("feedthrough: float")     // [output_dim, input_dim] - D matrix
    .Input("proc_noise: float")      // [state_dim, state_dim] - Q matrix
    .Input("meas_noise: float")      // [output_dim, output_dim] - R matrix
    .Output("success: bool")         // Scalar boolean
    .Attr("filter_id: string = 'default'")
    .Attr("max_rank: int = 8")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: Initialize TensorNetworkKalmanFilter.

Creates a new TNKF instance with the given state-space model.
Matrices A and P are TT-decomposed for O(n*r^2) time.

state_trans: State transition matrix A [state_dim, state_dim]
control_mat: Control input matrix B [state_dim, input_dim]
obs_mat: Observation matrix C [output_dim, state_dim]
feedthrough: Feedthrough matrix D [output_dim, input_dim]
proc_noise: Process noise covariance Q [state_dim, state_dim]
meas_noise: Measurement noise covariance R [output_dim, output_dim]
filter_id: Unique identifier for this filter instance
max_rank: Maximum TT rank for compression (4-16 typical)

success: True if initialization succeeded
)doc");

class TNKFInitOp : public OpKernel {
 public:
  explicit TNKFInitOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
    OP_REQUIRES_OK(context, context->GetAttr("max_rank", &max_rank_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& A = context->input(0);
    const Tensor& B = context->input(1);
    const Tensor& C = context->input(2);
    const Tensor& D = context->input(3);
    const Tensor& Q = context->input(4);
    const Tensor& R = context->input(5);

    const int state_dim = A.dim_size(0);
    const int input_dim = B.dim_size(1);
    const int output_dim = C.dim_size(0);

    // Convert TF tensors to Eigen matrices
    Eigen::MatrixXf A_mat = Eigen::Map<const Eigen::MatrixXf>(
        A.flat<float>().data(), state_dim, state_dim);
    Eigen::MatrixXf B_mat = Eigen::Map<const Eigen::MatrixXf>(
        B.flat<float>().data(), state_dim, input_dim);
    Eigen::MatrixXf C_mat = Eigen::Map<const Eigen::MatrixXf>(
        C.flat<float>().data(), output_dim, state_dim);
    Eigen::MatrixXf D_mat = Eigen::Map<const Eigen::MatrixXf>(
        D.flat<float>().data(), output_dim, input_dim);
    Eigen::MatrixXf Q_mat = Eigen::Map<const Eigen::MatrixXf>(
        Q.flat<float>().data(), state_dim, state_dim);
    Eigen::MatrixXf R_mat = Eigen::Map<const Eigen::MatrixXf>(
        R.flat<float>().data(), output_dim, output_dim);

    // Create or replace TNKF instance
    bool success = true;
    try {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto tnkf = std::make_unique<TensorNetworkKalmanFilter>();
      tnkf->init(A_mat, B_mat, C_mat, D_mat, Q_mat, R_mat, max_rank_);
      g_tnkf_instances[filter_id_] = std::move(tnkf);
    } catch (const std::exception& e) {
      success = false;
    }

    // Allocate output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));
    output->scalar<bool>()() = success;
  }

 private:
  std::string filter_id_;
  int max_rank_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFInit").Device(DEVICE_CPU), TNKFInitOp);

// =============================================================================
// OP REGISTRATION: TNKFPredict
// =============================================================================

REGISTER_OP("TNKFPredict")
    .Input("control_input: float")   // [input_dim] - control input u
    .Output("state_pred: float")     // [state_dim] - predicted state
    .Attr("filter_id: string = 'default'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // State dim is unknown at shape inference time
        c->set_output(0, c->UnknownShape());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: TNKF Predict step.

Performs state prediction using TT-accelerated matrix-vector product:
  x_pred = A @ x_hat + B @ u
  P_pred = A @ P @ A.T + Q (in TT format)

Runs in O(n*r^2) instead of O(n^2) for dense.

control_input: Control input vector [input_dim]
filter_id: Filter instance identifier

state_pred: Predicted state estimate [state_dim]
)doc");

class TNKFPredictOp : public OpKernel {
 public:
  explicit TNKFPredictOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& u = context->input(0);
    const int input_dim = u.dim_size(0);

    // Get TNKF instance
    TensorNetworkKalmanFilter* tnkf = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto it = g_tnkf_instances.find(filter_id_);
      OP_REQUIRES(context, it != g_tnkf_instances.end(),
                  errors::NotFound("TNKF instance not found: ", filter_id_));
      tnkf = it->second.get();
    }

    // Convert to std::vector
    std::vector<float> u_vec(input_dim);
    std::copy(u.flat<float>().data(),
              u.flat<float>().data() + input_dim,
              u_vec.begin());

    // Predict step
    tnkf->predict(u_vec);

    // Get state
    std::vector<float> state = tnkf->getState();
    const int state_dim = state.size();

    // Allocate output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({state_dim}), &output));
    std::copy(state.begin(), state.end(), output->flat<float>().data());
  }

 private:
  std::string filter_id_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFPredict").Device(DEVICE_CPU), TNKFPredictOp);

// =============================================================================
// OP REGISTRATION: TNKFUpdate
// =============================================================================

REGISTER_OP("TNKFUpdate")
    .Input("measurement: float")     // [output_dim] - measurement y
    .Output("state_post: float")     // [state_dim] - posterior state
    .Output("cov_diag: float")       // [state_dim] - covariance diagonal
    .Attr("filter_id: string = 'default'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->UnknownShape());
        c->set_output(1, c->UnknownShape());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: TNKF Update step.

Performs measurement update with Kalman gain:
  K = P @ C.T @ (C @ P @ C.T + R)^-1
  x_post = x_pred + K @ (y - C @ x_pred)
  P_post = (I - K @ C) @ P

Returns state and covariance diagonal (O(n) extraction from TT).

measurement: Observation vector [output_dim]
filter_id: Filter instance identifier

state_post: Posterior state estimate [state_dim]
cov_diag: Posterior covariance diagonal [state_dim]
)doc");

class TNKFUpdateOp : public OpKernel {
 public:
  explicit TNKFUpdateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& y = context->input(0);
    const int output_dim = y.dim_size(0);

    // Get TNKF instance
    TensorNetworkKalmanFilter* tnkf = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto it = g_tnkf_instances.find(filter_id_);
      OP_REQUIRES(context, it != g_tnkf_instances.end(),
                  errors::NotFound("TNKF instance not found: ", filter_id_));
      tnkf = it->second.get();
    }

    // Convert to std::vector
    std::vector<float> y_vec(output_dim);
    std::copy(y.flat<float>().data(),
              y.flat<float>().data() + output_dim,
              y_vec.begin());

    // Update step
    tnkf->update(y_vec);

    // Get results
    std::vector<float> state = tnkf->getState();
    std::vector<float> cov_diag = tnkf->getCovarianceDiagonal();
    const int state_dim = state.size();

    // Allocate outputs
    Tensor* state_out = nullptr;
    Tensor* cov_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({state_dim}), &state_out));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({state_dim}), &cov_out));

    std::copy(state.begin(), state.end(), state_out->flat<float>().data());
    std::copy(cov_diag.begin(), cov_diag.end(), cov_out->flat<float>().data());
  }

 private:
  std::string filter_id_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFUpdate").Device(DEVICE_CPU), TNKFUpdateOp);

// =============================================================================
// OP REGISTRATION: TNKFStep (Combined Predict + Update)
// =============================================================================

REGISTER_OP("TNKFStep")
    .Input("control_input: float")   // [input_dim] - control input u
    .Input("measurement: float")     // [output_dim] - measurement y
    .Output("state_post: float")     // [state_dim] - posterior state
    .Output("cov_diag: float")       // [state_dim] - covariance diagonal
    .Attr("filter_id: string = 'default'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->UnknownShape());
        c->set_output(1, c->UnknownShape());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: TNKF Combined Predict+Update step.

Combines predict and update in single call for efficiency.
Runs in O(n*r^2) for TT mode, O(n^3) for dense fallback.

control_input: Control input vector [input_dim]
measurement: Observation vector [output_dim]
filter_id: Filter instance identifier

state_post: Posterior state estimate [state_dim]
cov_diag: Posterior covariance diagonal [state_dim]
)doc");

class TNKFStepOp : public OpKernel {
 public:
  explicit TNKFStepOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& u = context->input(0);
    const Tensor& y = context->input(1);
    const int input_dim = u.dim_size(0);
    const int output_dim = y.dim_size(0);

    // Get TNKF instance
    TensorNetworkKalmanFilter* tnkf = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto it = g_tnkf_instances.find(filter_id_);
      OP_REQUIRES(context, it != g_tnkf_instances.end(),
                  errors::NotFound("TNKF instance not found: ", filter_id_));
      tnkf = it->second.get();
    }

    // Convert to std::vector
    std::vector<float> u_vec(input_dim);
    std::vector<float> y_vec(output_dim);
    std::copy(u.flat<float>().data(),
              u.flat<float>().data() + input_dim,
              u_vec.begin());
    std::copy(y.flat<float>().data(),
              y.flat<float>().data() + output_dim,
              y_vec.begin());

    // Predict + Update
    tnkf->predict(u_vec);
    tnkf->update(y_vec);

    // Get results
    std::vector<float> state = tnkf->getState();
    std::vector<float> cov_diag = tnkf->getCovarianceDiagonal();
    const int state_dim = state.size();

    // Allocate outputs
    Tensor* state_out = nullptr;
    Tensor* cov_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({state_dim}), &state_out));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({state_dim}), &cov_out));

    std::copy(state.begin(), state.end(), state_out->flat<float>().data());
    std::copy(cov_diag.begin(), cov_diag.end(), cov_out->flat<float>().data());
  }

 private:
  std::string filter_id_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFStep").Device(DEVICE_CPU), TNKFStepOp);

// =============================================================================
// OP REGISTRATION: TNKFGetState
// =============================================================================

REGISTER_OP("TNKFGetState")
    .Output("state: float")          // [state_dim]
    .Attr("filter_id: string = 'default'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->UnknownShape());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: Get current TNKF state estimate.

filter_id: Filter instance identifier
state: Current state estimate [state_dim]
)doc");

class TNKFGetStateOp : public OpKernel {
 public:
  explicit TNKFGetStateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
  }

  void Compute(OpKernelContext* context) override {
    // Get TNKF instance
    TensorNetworkKalmanFilter* tnkf = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto it = g_tnkf_instances.find(filter_id_);
      OP_REQUIRES(context, it != g_tnkf_instances.end(),
                  errors::NotFound("TNKF instance not found: ", filter_id_));
      tnkf = it->second.get();
    }

    std::vector<float> state = tnkf->getState();
    const int state_dim = state.size();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({state_dim}), &output));
    std::copy(state.begin(), state.end(), output->flat<float>().data());
  }

 private:
  std::string filter_id_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFGetState").Device(DEVICE_CPU), TNKFGetStateOp);

// =============================================================================
// OP REGISTRATION: TNKFGetMemoryUsage
// =============================================================================

REGISTER_OP("TNKFGetMemoryUsage")
    .Output("bytes: int64")          // Scalar
    .Output("is_tt_mode: bool")      // Scalar
    .Attr("filter_id: string = 'default'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: Get TNKF memory usage.

Reports memory usage and whether TT compression is active.

filter_id: Filter instance identifier
bytes: Memory usage in bytes
is_tt_mode: True if TT compression is active (state_dim >= 16)
)doc");

class TNKFGetMemoryUsageOp : public OpKernel {
 public:
  explicit TNKFGetMemoryUsageOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
  }

  void Compute(OpKernelContext* context) override {
    // Get TNKF instance
    TensorNetworkKalmanFilter* tnkf = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto it = g_tnkf_instances.find(filter_id_);
      OP_REQUIRES(context, it != g_tnkf_instances.end(),
                  errors::NotFound("TNKF instance not found: ", filter_id_));
      tnkf = it->second.get();
    }

    Tensor* bytes_out = nullptr;
    Tensor* mode_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &bytes_out));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &mode_out));

    bytes_out->scalar<int64_t>()() = static_cast<int64_t>(tnkf->getMemoryUsage());
    mode_out->scalar<bool>()() = tnkf->isTTModeActive();
  }

 private:
  std::string filter_id_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFGetMemoryUsage").Device(DEVICE_CPU), TNKFGetMemoryUsageOp);

// =============================================================================
// OP REGISTRATION: TNKFReset
// =============================================================================

REGISTER_OP("TNKFReset")
    .Output("success: bool")
    .Attr("filter_id: string = 'default'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: Reset TNKF state to initial (zero state, identity covariance).

filter_id: Filter instance identifier
success: True if reset succeeded
)doc");

class TNKFResetOp : public OpKernel {
 public:
  explicit TNKFResetOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
  }

  void Compute(OpKernelContext* context) override {
    // Get TNKF instance
    TensorNetworkKalmanFilter* tnkf = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto it = g_tnkf_instances.find(filter_id_);
      OP_REQUIRES(context, it != g_tnkf_instances.end(),
                  errors::NotFound("TNKF instance not found: ", filter_id_));
      tnkf = it->second.get();
    }

    tnkf->reset();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));
    output->scalar<bool>()() = true;
  }

 private:
  std::string filter_id_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFReset").Device(DEVICE_CPU), TNKFResetOp);

// =============================================================================
// OP REGISTRATION: TNKFUpdateA (Dynamic system identification)
// =============================================================================

REGISTER_OP("TNKFUpdateA")
    .Input("new_state_trans: float")  // [state_dim, state_dim]
    .Output("success: bool")
    .Attr("filter_id: string = 'default'")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
UQHA Phase 1001: Update TNKF state transition matrix.

Used for online system identification (e.g., from RLS).
Re-decomposes A into TT format.

new_state_trans: New state transition matrix [state_dim, state_dim]
filter_id: Filter instance identifier
success: True if update succeeded
)doc");

class TNKFUpdateAOp : public OpKernel {
 public:
  explicit TNKFUpdateAOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("filter_id", &filter_id_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& A = context->input(0);
    const int state_dim = A.dim_size(0);

    // Get TNKF instance
    TensorNetworkKalmanFilter* tnkf = nullptr;
    {
      std::lock_guard<std::mutex> lock(g_tnkf_mutex);
      auto it = g_tnkf_instances.find(filter_id_);
      OP_REQUIRES(context, it != g_tnkf_instances.end(),
                  errors::NotFound("TNKF instance not found: ", filter_id_));
      tnkf = it->second.get();
    }

    // Convert to Eigen matrix
    Eigen::MatrixXf A_mat = Eigen::Map<const Eigen::MatrixXf>(
        A.flat<float>().data(), state_dim, state_dim);

    tnkf->updateA(A_mat);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));
    output->scalar<bool>()() = true;
  }

 private:
  std::string filter_id_;
};

REGISTER_KERNEL_BUILDER(Name("TNKFUpdateA").Device(DEVICE_CPU), TNKFUpdateAOp);
