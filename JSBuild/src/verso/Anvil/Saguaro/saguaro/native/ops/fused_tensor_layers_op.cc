// saguaro.native/ops/fused_tensor_layers_op.cc
#include "fused_tensor_layers_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("FusedTuckerForward")
    .Input("x: float32")
    .Input("u_in: float32")
    .Input("core: float32")
    .Input("u_out: float32")
    .Input("bias: float32")
    .Output("y: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto x_shape = c->input(0);
        auto u_out_shape = c->input(3);
        
        shape_inference::DimensionHandle batch = c->Dim(x_shape, 0);
        shape_inference::DimensionHandle d_out = c->Dim(u_out_shape, 0);
        
        c->set_output(0, c->MakeShape({batch, d_out}));
        return Status();
    })
    .Doc("Fused Tucker decomposition forward pass.");

REGISTER_OP("FusedTensorRingForward")
    .Input("inputs: N * float32")
    .Input("cores: N * float32")
    .Input("bias: float32")
    .Input("local_dims_in: int32")
    .Input("local_dims_out: int32")
    .Attr("N: int >= 2")
    .Attr("ring_rank: int")
    .Output("y: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto first_input = c->input(0);
        shape_inference::DimensionHandle batch = c->Dim(first_input, 0);
        
        // Sum of local_dims_out determines total_D_out
        c->set_output(0, c->MakeShape({batch, shape_inference::InferenceContext::kUnknownDim}));
        return Status();
    })
    .Doc("Fused Tensor-Ring decomposition forward pass with proper trace contraction.");

REGISTER_OP("FusedTuckerForwardGrad")
    .Input("grad_output: float32")
    .Input("x: float32")
    .Input("u_in: float32")
    .Input("core: float32")
    .Input("u_out: float32")
    .Input("bias: float32")
    .Output("grad_x: float32")
    .Output("grad_u_in: float32")
    .Output("grad_core: float32")
    .Output("grad_u_out: float32")
    .Output("grad_bias: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_x
        c->set_output(1, c->input(2));  // grad_u_in
        c->set_output(2, c->input(3));  // grad_core
        c->set_output(3, c->input(4));  // grad_u_out
        c->set_output(4, c->input(5));  // grad_bias
        return Status();
    })
    .Doc("Gradient for fused Tucker decomposition forward pass.");

REGISTER_OP("FusedTensorRingForwardGrad")
    .Input("grad_output: float32")
    .Input("inputs: N * float32")
    .Input("cores: N * float32")
    .Input("bias: float32")
    .Input("local_dims_in: int32")
    .Input("local_dims_out: int32")
    .Attr("N: int >= 2")
    .Attr("ring_rank: int")
    .Output("grad_inputs: N * float32")
    .Output("grad_cores: N * float32")
    .Output("grad_bias: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int num_cores = 0;
        TF_RETURN_IF_ERROR(c->GetAttr("N", &num_cores));
        for (int i = 0; i < num_cores; ++i) {
            c->set_output(i, c->input(1 + i));  // grad_inputs
        }
        for (int i = 0; i < num_cores; ++i) {
            c->set_output(num_cores + i, c->input(1 + num_cores + i));  // grad_cores
        }
        c->set_output(2 * num_cores, c->input(1 + 2 * num_cores));  // grad_bias
        return Status();
    })
    .Doc("Gradient for fused Tensor-Ring decomposition forward pass.");

REGISTER_OP("FusedTensorMPSContract")
    .Input("left: float32")
    .Input("right: float32")
    .Output("result: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto left = c->input(0);
        auto right = c->input(1);
        auto left_dim = c->Dim(left, 0);
        auto right_dim = c->Dim(right, 1);
        c->set_output(0, c->MakeShape({left_dim, right_dim}));
        return Status();
    })
    .Doc("MPS tensor contraction (Legacy).");

// =============================================================================
// Kernel Implementations
// =============================================================================

class FusedTuckerForwardOp : public OpKernel {
public:
    explicit FusedTuckerForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& x = ctx->input(0);
        const Tensor& u_in = ctx->input(1);
        const Tensor& core = ctx->input(2);
        const Tensor& u_out = ctx->input(3);
        const Tensor& bias_tensor = ctx->input(4);

        OP_REQUIRES(ctx, x.dims() == 2, errors::InvalidArgument("x must be 2D"));
        
        int64_t batch = x.dim_size(0);
        int64_t D_in = x.dim_size(1);
        int64_t D_out = u_out.dim_size(0);
        int64_t R_in = core.dim_size(1);
        int64_t R_out = core.dim_size(0);

        Tensor* y = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch, D_out}), &y));

        const float* bias_ptr = (bias_tensor.NumElements() > 0) ? bias_tensor.flat<float>().data() : nullptr;

        saguaro::ops::fused_tucker_forward(
            x.flat<float>().data(), u_in.flat<float>().data(),
            core.flat<float>().data(), u_out.flat<float>().data(), bias_ptr,
            y->flat<float>().data(), batch, D_in, D_out, R_in, R_out);
    }
};

class FusedTensorRingForwardOp : public OpKernel {
public:
    explicit FusedTensorRingForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_cores_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ring_rank", &ring_rank_));
    }

    void Compute(OpKernelContext* ctx) override {
        OpInputList inputs;
        OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));
        OpInputList cores;
        OP_REQUIRES_OK(ctx, ctx->input_list("cores", &cores));
        const Tensor& bias_tensor = ctx->input(2 * num_cores_);
        const Tensor& local_dims_in_t = ctx->input(2 * num_cores_ + 1);
        const Tensor& local_dims_out_t = ctx->input(2 * num_cores_ + 2);

        int64_t batch = inputs[0].dim_size(0);
        const int* local_in_ptr = local_dims_in_t.flat<int32>().data();
        const int* local_out_ptr = local_dims_out_t.flat<int32>().data();

        int64_t total_D_out = 0;
        for (int i = 0; i < num_cores_; ++i) total_D_out += local_out_ptr[i];

        Tensor* y = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch, total_D_out}), &y));

        std::vector<const float*> input_ptrs(num_cores_);
        std::vector<const float*> core_ptrs(num_cores_);
        for (int i = 0; i < num_cores_; ++i) {
            input_ptrs[i] = inputs[i].flat<float>().data();
            core_ptrs[i] = cores[i].flat<float>().data();
        }

        const float* bias_ptr = (bias_tensor.NumElements() > 0) ? bias_tensor.flat<float>().data() : nullptr;

        saguaro::ops::fused_tensor_ring_forward(
            input_ptrs.data(), core_ptrs.data(), bias_ptr, y->flat<float>().data(),
            batch, num_cores_, ring_rank_, local_in_ptr, local_out_ptr, total_D_out);
    }

private:
    int num_cores_;
    int ring_rank_;
};

class FusedTuckerForwardGradOp : public OpKernel {
public:
    explicit FusedTuckerForwardGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        const Tensor& x = ctx->input(1);
        const Tensor& u_in = ctx->input(2);
        const Tensor& core = ctx->input(3);
        const Tensor& u_out = ctx->input(4);
        const Tensor& bias_tensor = ctx->input(5);

        OP_REQUIRES(ctx, x.dims() == 2, errors::InvalidArgument("x must be 2D"));
        OP_REQUIRES(ctx, grad_output.dims() == 2, errors::InvalidArgument("grad_output must be 2D"));

        int64_t batch = x.dim_size(0);
        int64_t D_in = x.dim_size(1);
        int64_t D_out = u_out.dim_size(0);
        int64_t R_in = core.dim_size(1);
        int64_t R_out = core.dim_size(0);

        Tensor* grad_x = nullptr;
        Tensor* grad_u_in = nullptr;
        Tensor* grad_core = nullptr;
        Tensor* grad_u_out = nullptr;
        Tensor* grad_bias = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &grad_x));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, u_in.shape(), &grad_u_in));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, core.shape(), &grad_core));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, u_out.shape(), &grad_u_out));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, bias_tensor.shape(), &grad_bias));

        using Eigen::MatrixXf;
        using Eigen::Map;
        using Eigen::VectorXf;

        Map<const MatrixXf> m_x(x.flat<float>().data(), batch, D_in);
        Map<const MatrixXf> m_u_in(u_in.flat<float>().data(), D_in, R_in);
        Map<const MatrixXf> m_core(core.flat<float>().data(), R_out, R_in);
        Map<const MatrixXf> m_u_out(u_out.flat<float>().data(), D_out, R_out);
        Map<const MatrixXf> m_grad_out(grad_output.flat<float>().data(), batch, D_out);

        MatrixXf x_proj = m_x * m_u_in;  // [batch, R_in]
        MatrixXf z = x_proj * m_core.transpose();  // [batch, R_out]

        MatrixXf grad_u_out_m = m_grad_out.transpose() * z;  // [D_out, R_out]
        MatrixXf grad_z = m_grad_out * m_u_out;  // [batch, R_out]
        MatrixXf grad_core_m = grad_z.transpose() * x_proj;  // [R_out, R_in]
        MatrixXf grad_x_proj = grad_z * m_core;  // [batch, R_in]
        MatrixXf grad_u_in_m = m_x.transpose() * grad_x_proj;  // [D_in, R_in]
        MatrixXf grad_x_m = grad_x_proj * m_u_in.transpose();  // [batch, D_in]

        Map<MatrixXf> m_grad_x(grad_x->flat<float>().data(), batch, D_in);
        Map<MatrixXf> m_grad_u_in(grad_u_in->flat<float>().data(), D_in, R_in);
        Map<MatrixXf> m_grad_core(grad_core->flat<float>().data(), R_out, R_in);
        Map<MatrixXf> m_grad_u_out(grad_u_out->flat<float>().data(), D_out, R_out);

        m_grad_x = grad_x_m;
        m_grad_u_in = grad_u_in_m;
        m_grad_core = grad_core_m;
        m_grad_u_out = grad_u_out_m;

        if (bias_tensor.NumElements() > 0) {
            Map<VectorXf> m_grad_bias(grad_bias->flat<float>().data(), D_out);
            m_grad_bias = m_grad_out.colwise().sum();
        } else if (grad_bias->NumElements() > 0) {
            grad_bias->flat<float>().setZero();
        }
    }
};

class FusedTensorRingForwardGradOp : public OpKernel {
public:
    explicit FusedTensorRingForwardGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_cores_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ring_rank", &ring_rank_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        OpInputList inputs;
        OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));
        OpInputList cores;
        OP_REQUIRES_OK(ctx, ctx->input_list("cores", &cores));
        const Tensor& bias_tensor = ctx->input(1 + 2 * num_cores_);
        const Tensor& local_dims_in_t = ctx->input(1 + 2 * num_cores_ + 1);
        const Tensor& local_dims_out_t = ctx->input(1 + 2 * num_cores_ + 2);

        OP_REQUIRES(ctx, grad_output.dims() == 2,
                    errors::InvalidArgument("grad_output must be 2D"));

        const int* local_in_ptr = local_dims_in_t.flat<int32>().data();
        const int* local_out_ptr = local_dims_out_t.flat<int32>().data();

        int64_t batch = inputs[0].dim_size(0);
        int64_t total_D_out = 0;
        for (int i = 0; i < num_cores_; ++i) {
            total_D_out += local_out_ptr[i];
        }

        std::vector<Tensor*> grad_inputs(num_cores_, nullptr);
        std::vector<Tensor*> grad_cores(num_cores_, nullptr);
        for (int i = 0; i < num_cores_; ++i) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(i, inputs[i].shape(), &grad_inputs[i]));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(num_cores_ + i, cores[i].shape(), &grad_cores[i]));
            grad_inputs[i]->flat<float>().setZero();
            grad_cores[i]->flat<float>().setZero();
        }

        Tensor* grad_bias = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2 * num_cores_, bias_tensor.shape(), &grad_bias));
        if (grad_bias->NumElements() > 0) {
            grad_bias->flat<float>().setZero();
        }

        using Eigen::MatrixXf;

        std::vector<int> out_offsets(num_cores_);
        int current_offset = 0;
        for (int m = 0; m < num_cores_; ++m) {
            out_offsets[m] = current_offset;
            current_offset += local_out_ptr[m];
        }

        const float* grad_output_ptr = grad_output.flat<float>().data();

        for (int64_t b = 0; b < batch; ++b) {
            std::vector<MatrixXf> b_sum(num_cores_, MatrixXf::Zero(ring_rank_, ring_rank_));
            std::vector<std::vector<MatrixXf>> b_indexed(num_cores_);
            for (int m = 0; m < num_cores_; ++m) {
                b_indexed[m].resize(local_out_ptr[m], MatrixXf::Zero(ring_rank_, ring_rank_));
            }

            for (int m = 0; m < num_cores_; ++m) {
                int d_in = local_in_ptr[m];
                int d_out = local_out_ptr[m];
                const float* core_ptr = cores[m].flat<float>().data();
                const float* input_ptr = inputs[m].flat<float>().data() + b * d_in;

                for (int i = 0; i < d_out; ++i) {
                    for (int j = 0; j < d_in; ++j) {
                        float x_val = input_ptr[j];
                        if (std::abs(x_val) < 1e-12f) {
                            continue;
                        }
                        for (int r1 = 0; r1 < ring_rank_; ++r1) {
                            const float* r1_row = core_ptr + r1 * (d_out * d_in * ring_rank_) + (i * d_in + j) * ring_rank_;
                            for (int r2 = 0; r2 < ring_rank_; ++r2) {
                                b_indexed[m][i](r1, r2) += r1_row[r2] * x_val;
                            }
                        }
                    }
                    b_sum[m] += b_indexed[m][i];
                }
            }

            std::vector<MatrixXf> prefix(num_cores_ + 1, MatrixXf::Identity(ring_rank_, ring_rank_));
            std::vector<MatrixXf> suffix(num_cores_ + 1, MatrixXf::Identity(ring_rank_, ring_rank_));
            for (int m = 0; m < num_cores_; ++m) {
                prefix[m + 1] = prefix[m] * b_sum[m];
            }
            for (int m = num_cores_ - 1; m >= 0; --m) {
                suffix[m] = b_sum[m] * suffix[m + 1];
            }

            std::vector<MatrixXf> grad_prefix(num_cores_ + 1, MatrixXf::Zero(ring_rank_, ring_rank_));
            std::vector<MatrixXf> grad_suffix(num_cores_ + 1, MatrixXf::Zero(ring_rank_, ring_rank_));
            std::vector<MatrixXf> grad_b_sum(num_cores_, MatrixXf::Zero(ring_rank_, ring_rank_));
            std::vector<std::vector<MatrixXf>> grad_b_indexed(num_cores_);
            for (int m = 0; m < num_cores_; ++m) {
                grad_b_indexed[m].resize(local_out_ptr[m], MatrixXf::Zero(ring_rank_, ring_rank_));
            }

            const float* grad_batch = grad_output_ptr + b * total_D_out;
            for (int m = 0; m < num_cores_; ++m) {
                int d_out = local_out_ptr[m];
                int offset = out_offsets[m];
                const MatrixXf& L = prefix[m];
                const MatrixXf& R = suffix[m + 1];

                for (int i = 0; i < d_out; ++i) {
                    float g = grad_batch[offset + i];
                    if (g == 0.0f) {
                        continue;
                    }
                    const MatrixXf& Bi = b_indexed[m][i];
                    grad_b_indexed[m][i].noalias() += g * (L.transpose() * R.transpose());
                    grad_prefix[m].noalias() += g * (Bi * R).transpose();
                    grad_suffix[m + 1].noalias() += g * (L * Bi).transpose();
                }
            }

            for (int m = num_cores_ - 1; m >= 0; --m) {
                grad_b_sum[m].noalias() += prefix[m].transpose() * grad_prefix[m + 1];
                grad_prefix[m].noalias() += grad_prefix[m + 1] * b_sum[m].transpose();
            }

            for (int m = 0; m < num_cores_; ++m) {
                grad_b_sum[m].noalias() += grad_suffix[m] * suffix[m + 1].transpose();
                grad_suffix[m + 1].noalias() += b_sum[m].transpose() * grad_suffix[m];
            }

            for (int m = 0; m < num_cores_; ++m) {
                int d_out = local_out_ptr[m];
                for (int i = 0; i < d_out; ++i) {
                    grad_b_indexed[m][i].noalias() += grad_b_sum[m];
                }
            }

            for (int m = 0; m < num_cores_; ++m) {
                int d_in = local_in_ptr[m];
                int d_out = local_out_ptr[m];
                const float* input_ptr = inputs[m].flat<float>().data() + b * d_in;
                const float* core_ptr = cores[m].flat<float>().data();
                float* grad_input_ptr = grad_inputs[m]->flat<float>().data() + b * d_in;
                float* grad_core_ptr = grad_cores[m]->flat<float>().data();

                for (int j = 0; j < d_in; ++j) {
                    grad_input_ptr[j] = 0.0f;
                }

                for (int i = 0; i < d_out; ++i) {
                    const MatrixXf& grad_Bi = grad_b_indexed[m][i];
                    for (int j = 0; j < d_in; ++j) {
                        float x_val = input_ptr[j];
                        for (int r1 = 0; r1 < ring_rank_; ++r1) {
                            const float* r1_row = core_ptr + r1 * (d_out * d_in * ring_rank_) + (i * d_in + j) * ring_rank_;
                            float* grad_r1_row = grad_core_ptr + r1 * (d_out * d_in * ring_rank_) + (i * d_in + j) * ring_rank_;
                            for (int r2 = 0; r2 < ring_rank_; ++r2) {
                                float g = grad_Bi(r1, r2);
                                grad_input_ptr[j] += g * r1_row[r2];
                                grad_r1_row[r2] += g * x_val;
                            }
                        }
                    }
                }
            }

            if (grad_bias->NumElements() > 0) {
                float* grad_bias_ptr = grad_bias->flat<float>().data();
                for (int64_t d = 0; d < total_D_out; ++d) {
                    grad_bias_ptr[d] += grad_batch[d];
                }
            }
        }
    }

private:
    int num_cores_;
    int ring_rank_;
};

class FusedTensorMPSContractOp : public OpKernel {
 public:
    explicit FusedTensorMPSContractOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& left = ctx->input(0);
        const Tensor& right = ctx->input(1);
        
        const int64_t left_dim = left.dim_size(0);
        const int64_t bond_dim = left.dim_size(1);
        const int64_t right_dim = right.dim_size(1);
        
        Tensor* result = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({left_dim, right_dim}), &result));
        
        saguaro::ops::tensor_mps_contract(
            left.flat<float>().data(), right.flat<float>().data(),
            result->flat<float>().data(), left_dim, bond_dim, right_dim);
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedTuckerForward").Device(DEVICE_CPU), FusedTuckerForwardOp);
REGISTER_KERNEL_BUILDER(Name("FusedTensorRingForward").Device(DEVICE_CPU), FusedTensorRingForwardOp);
REGISTER_KERNEL_BUILDER(Name("FusedTuckerForwardGrad").Device(DEVICE_CPU), FusedTuckerForwardGradOp);
REGISTER_KERNEL_BUILDER(Name("FusedTensorRingForwardGrad").Device(DEVICE_CPU), FusedTensorRingForwardGradOp);
REGISTER_KERNEL_BUILDER(Name("FusedTensorMPSContract").Device(DEVICE_CPU), FusedTensorMPSContractOp);

} // namespace tensorflow
