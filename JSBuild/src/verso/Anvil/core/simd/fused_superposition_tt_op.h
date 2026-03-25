// highnoon/_native/ops/fused_superposition_tt_op.h
// Copyright 2025 Verso Industries
//
// Fused C++ kernel for SuperpositionTTLayer.

#ifndef HIGHNOON_NATIVE_OPS_FUSED_SUPERPOSITION_TT_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_SUPERPOSITION_TT_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include <vector>
#include <string>
#include <numeric>

namespace tensorflow {

// Helper to manage views into the flat core tensor
class SuperpositionTTCoresView {
public:
    SuperpositionTTCoresView(const float* data, const std::vector<int64>& ranks,
                             const std::vector<int64>& input_dims, const std::vector<int64>& output_dims,
                             int64 s_dim)
        : num_cores_(input_dims.size()) {
        
        core_pointers_.resize(num_cores_);
        core_shapes_.resize(num_cores_);
        int64 offset = 0;
        for (int i = 0; i < num_cores_; ++i) {
            core_shapes_[i] = {ranks[i], input_dims[i], output_dims[i], ranks[i + 1], s_dim};
            core_pointers_[i] = data + offset;
            offset += ranks[i] * input_dims[i] * output_dims[i] * ranks[i + 1] * s_dim;
        }
    }

    Eigen::TensorMap<Eigen::Tensor<const float, 5>> get_core(int i) const {
        return Eigen::TensorMap<Eigen::Tensor<const float, 5>>(
            core_pointers_[i], core_shapes_[i][0], core_shapes_[i][1],
            core_shapes_[i][2], core_shapes_[i][3], core_shapes_[i][4]);
    }
    
    // Writable version for gradients
    Eigen::TensorMap<Eigen::Tensor<float, 5>> get_core_writable(float* data, int i) const {
        return Eigen::TensorMap<Eigen::Tensor<float, 5>>(
            data, core_shapes_[i][0], core_shapes_[i][1],
            core_shapes_[i][2], core_shapes_[i][3], core_shapes_[i][4]);
    }
    
    int64 get_core_offset(int i) const {
        if (i == 0) return 0;
        int64 offset = 0;
        for (int j = 0; j < i; ++j) {
            offset += core_shapes_[j][0] * core_shapes_[j][1] * core_shapes_[j][2] * core_shapes_[j][3] * core_shapes_[j][4];
        }
        return offset;
    }

private:
    int num_cores_;
    std::vector<const float*> core_pointers_;
    std::vector<std::array<int64, 5>> core_shapes_;
};


class FusedSuperpositionTTOpCpu : public OpKernel {
public:
    explicit FusedSuperpositionTTOpCpu(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("input_dims", &input_dims_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dims", &output_dims_));
        OP_REQUIRES_OK(context, context->GetAttr("tt_ranks", &tt_ranks_));
        OP_REQUIRES_OK(context, context->GetAttr("superposition_dim", &superposition_dim_));
        num_cores_ = input_dims_.size();
        OP_REQUIRES(context, num_cores_ == 2, errors::Unimplemented("This FusedSuperpositionTT kernel only supports 2 cores."));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& tokens_tensor = context->input(0);
        const Tensor& cores_tensor = context->input(1);

        const int64 batch_size = tokens_tensor.dim_size(0);
        long output_dim = output_dims_[0] * output_dims_[1];

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
            TensorShape({batch_size, output_dim, superposition_dim_}), &output_tensor));

        const float* tokens_ptr = tokens_tensor.flat<float>().data();
        const float* cores_ptr = cores_tensor.flat<float>().data();
        float* output_ptr = output_tensor->flat<float>().data();

        #pragma omp parallel for
        for(int b = 0; b < batch_size; ++b) {
            Eigen::TensorMap<Eigen::Tensor<const float, 2>> token_mat(
                tokens_ptr + b * input_dims_[0] * input_dims_[1], input_dims_[0], input_dims_[1]);

            int64 core0_size = tt_ranks_[0] * input_dims_[0] * output_dims_[0] * tt_ranks_[1] * superposition_dim_;
            Eigen::TensorMap<Eigen::Tensor<const float, 5>> core0(
                cores_ptr, tt_ranks_[0], input_dims_[0], output_dims_[0], tt_ranks_[1], superposition_dim_);
            Eigen::TensorMap<Eigen::Tensor<const float, 5>> core1(
                cores_ptr + core0_size, tt_ranks_[1], input_dims_[1], output_dims_[1], tt_ranks_[2], superposition_dim_);

            Eigen::array<Eigen::IndexPair<int>, 1> contract1 = { Eigen::IndexPair<int>(0, 1) };
            auto temp = token_mat.contract(core0.chip(0, 0), contract1);

            Eigen::array<Eigen::IndexPair<int>, 2> contract2 = { Eigen::IndexPair<int>(0, 1), Eigen::IndexPair<int>(2, 0) };
            auto result = temp.contract(core1.chip(0, 4), contract2);
            
            auto result_shuffled = result.shuffle(Eigen::array<int, 3>{1, 2, 0});
            
            Eigen::TensorMap<Eigen::Tensor<float, 3>> out_slice(
                output_ptr + b * output_dim * superposition_dim_, output_dims_[0], output_dims_[1], superposition_dim_);
            out_slice = result_shuffled;
        }
    }
private:
    std::vector<int64> input_dims_, output_dims_, tt_ranks_;
    int64 superposition_dim_;
    int num_cores_;
};

class FusedSuperpositionTTGradOpCpu : public OpKernel {
public:
    explicit FusedSuperpositionTTGradOpCpu(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("input_dims", &input_dims_));
        OP_REQUIRES_OK(context, context->GetAttr("output_dims", &output_dims_));
        OP_REQUIRES_OK(context, context->GetAttr("tt_ranks", &tt_ranks_));
        OP_REQUIRES_OK(context, context->GetAttr("superposition_dim", &superposition_dim_));
        num_cores_ = input_dims_.size();
        OP_REQUIRES(context, num_cores_ == 2, errors::Unimplemented("This FusedSuperpositionTTGrad kernel only supports 2 cores."));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output_tensor = context->input(0);
        const Tensor& tokens_tensor = context->input(1);
        const Tensor& cores_tensor = context->input(2);
        
        const int64 batch_size = tokens_tensor.dim_size(0);
        long input_dim = input_dims_[0] * input_dims_[1];
        long output_dim = output_dims_[0] * output_dims_[1];

        Tensor* grad_tokens_tensor = nullptr;
        Tensor* grad_cores_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, tokens_tensor.shape(), &grad_tokens_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, cores_tensor.shape(), &grad_cores_tensor));

        grad_tokens_tensor->flat<float>().setZero();
        grad_cores_tensor->flat<float>().setZero();

        const float* grad_output_ptr = grad_output_tensor.flat<float>().data();
        const float* tokens_ptr = tokens_tensor.flat<float>().data();
        const float* cores_ptr = cores_tensor.flat<float>().data();
        float* grad_tokens_ptr = grad_tokens_tensor->flat<float>().data();
        float* grad_cores_ptr = grad_cores_tensor->flat<float>().data();
        
        int64 core0_size = tt_ranks_[0] * input_dims_[0] * output_dims_[0] * tt_ranks_[1] * superposition_dim_;
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            // Recompute forward intermediates for this token
            Eigen::TensorMap<Eigen::Tensor<const float, 2>> token_mat(
                tokens_ptr + b * input_dim, input_dims_[0], input_dims_[1]);
            Eigen::TensorMap<Eigen::Tensor<const float, 5>> core0(
                cores_ptr, tt_ranks_[0], input_dims_[0], output_dims_[0], tt_ranks_[1], superposition_dim_);
            Eigen::TensorMap<Eigen::Tensor<const float, 5>> core1(
                cores_ptr + core0_size, tt_ranks_[1], input_dims_[1], output_dims_[1], tt_ranks_[2], superposition_dim_);
            
            Eigen::array<Eigen::IndexPair<int>, 1> contract1 = { Eigen::IndexPair<int>(0, 1) };
            auto intermediate = token_mat.contract(core0.chip(0, 0), contract1);

            // Upstream gradient
            Eigen::TensorMap<Eigen::Tensor<const float, 3>> grad_output_mat(
                grad_output_ptr + b * output_dim * superposition_dim_, output_dims_[0], output_dims_[1], superposition_dim_);
            auto grad_output_shuffled = grad_output_mat.shuffle(Eigen::array<int, 3>{2, 0, 1});

            // Backprop through second contraction: result = intermediate.contract(core1)
            // Grad for core1
            Eigen::array<Eigen::IndexPair<int>, 2> g_core1_contract = { Eigen::IndexPair<int>(0, 1), Eigen::IndexPair<int>(1, 2) };
            auto grad_core1_shuffled = intermediate.contract(grad_output_shuffled, g_core1_contract);
            auto grad_core1 = grad_core1_shuffled.shuffle(Eigen::array<int, 4>{2, 0, 3, 1});

            // Grad for intermediate
            Eigen::array<Eigen::IndexPair<int>, 1> g_intermediate_contract = { Eigen::IndexPair<int>(1, 2) };
            auto grad_intermediate = grad_output_shuffled.contract(core1.chip(0, 4), g_intermediate_contract);

            // Backprop through first contraction: intermediate = token.contract(core0)
            // Grad for core0
            Eigen::array<Eigen::IndexPair<int>, 2> g_core0_contract = { Eigen::IndexPair<int>(0, 0), Eigen::IndexPair<int>(1, 2) };
            auto grad_core0_shuffled = token_mat.contract(grad_intermediate, g_core0_contract);
            auto grad_core0 = grad_core0_shuffled.shuffle(Eigen::array<int, 3>{1, 0, 2});
            
            // Grad for token
            Eigen::array<Eigen::IndexPair<int>, 2> g_token_contract = { Eigen::IndexPair<int>(1, 1), Eigen::IndexPair<int>(2, 3), Eigen::IndexPair<int>(3, 4) };
            auto grad_token = grad_intermediate.contract(core0.chip(0,0), g_token_contract);
            
            // Accumulate gradients
            Eigen::TensorMap<Eigen::Tensor<float, 2>> grad_token_slice(grad_tokens_ptr + b * input_dim, input_dims_[0], input_dims_[1]);
            grad_token_slice += grad_token;

            for(int i=0; i<grad_core0.size(); ++i) AtomicAdd(grad_cores_ptr + i, grad_core0.data()[i]);
            for(int i=0; i<grad_core1.size(); ++i) AtomicAdd(grad_cores_ptr + core0_size + i, grad_core1.data()[i]);
        }
    }
private:
    std::vector<int64> input_dims_, output_dims_, tt_ranks_;
    int64 superposition_dim_;
    int num_cores_;
};


} // namespace tensorflow

#endif // HIGHNOON_NATIVE_OPS_FUSED_SUPERPOSITION_TT_OP_H_
