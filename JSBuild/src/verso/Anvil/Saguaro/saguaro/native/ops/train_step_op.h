#ifndef TENSORFLOW_CORE_KERNELS_TRAIN_STEP_OP_H_
#define TENSORFLOW_CORE_KERNELS_TRAIN_STEP_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "absl/synchronization/mutex.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor" // Include for Eigen::Tensor and Eigen::TensorMap
#include "ops/fused_reasoning_stack/fused_reasoning_stack_kernel.h"
#include "ops/n4sid_solver.h"

namespace tensorflow {

// Forward declarations for SophiaG optimizer helper functions
void SophiaG_ApplyGradients(OpKernelContext* context,
                            const Tensor& grad,
                            Tensor* var,
                            Tensor* m,
                            Tensor* h,
                            float lr,
                            float beta_1_t,
                            float rho_t,
                            float epsilon);

void SophiaG_UpdateHessian(OpKernelContext* context,
                           const Tensor& h_grad,
                           Tensor* h,
                           float beta_2_t);

// Full TrainStep operation with N4SID system identification, EWC, and reasoning stack
// This provides comprehensive training capabilities for the HSMN architecture.

class TrainStepOp : public OpKernel {
public:
    explicit TrainStepOp(OpKernelConstruction* context);

    void Compute(OpKernelContext* context) override;

private:
    int m_; 
    int o_; 
    int f_; 
    int n_;
    int p_;
    int num_n4sid_inputs_;
    int num_n4sid_outputs_;
    int num_n4sid_matrices_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRAIN_STEP_OP_H_
