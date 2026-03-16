// saguaro.native/ops/qbm_sample_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// QBM Sample operator - Quantum Boltzmann Machine sampling with annealing
// This operator performs simulated quantum annealing for expert selection
// in Mixture-of-Experts models.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/edition_limits.h"
#include <random>
#include <cmath>
#include <vector>

namespace tensorflow {

// ==================== Forward Op ====================
REGISTER_OP("QbmSample")
    .Input("energy_matrix: float32")        // [batch, num_experts]
    .Input("temperature_init: float32")     // scalar
    .Input("temperature_final: float32")    // scalar
    .Input("num_annealing_steps: int32")    // scalar
    .Input("seed: int32")                   // scalar
    .Output("expert_assignments: int32")    // [batch]
    .Output("sample_log_probs: float32")    // [batch]
    .Output("annealing_energies: float32")  // [batch, num_annealing_steps]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle energy_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &energy_shape));
        
        shape_inference::DimensionHandle batch = c->Dim(energy_shape, 0);
        c->set_output(0, c->Vector(batch));  // expert_assignments
        c->set_output(1, c->Vector(batch));  // sample_log_probs
        // annealing_energies: [batch, num_steps] - shape depends on runtime input
        c->set_output(2, c->UnknownShapeOfRank(2));
        return Status();
    })
    .Doc("Quantum Boltzmann Machine sampling with simulated annealing.");

class QbmSampleOp : public OpKernel {
public:
    explicit QbmSampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& energy_matrix = ctx->input(0);
        const Tensor& temp_init = ctx->input(1);
        const Tensor& temp_final = ctx->input(2);
        const Tensor& num_steps_tensor = ctx->input(3);
        const Tensor& seed_tensor = ctx->input(4);
        
        const int batch_size = energy_matrix.dim_size(0);
        const int num_experts = energy_matrix.dim_size(1);
        
        // HighNoon Lite Edition: Enforce MoE expert limit (max 12)
        SAGUARO_CHECK_MOE_EXPERTS(ctx, num_experts);
        
        const float t_init = temp_init.scalar<float>()();
        const float t_final = temp_final.scalar<float>()();
        const int num_steps = num_steps_tensor.scalar<int>()();
        const int seed = seed_tensor.scalar<int>()();
        
        // Allocate outputs
        Tensor* expert_assignments = nullptr;
        Tensor* sample_log_probs = nullptr;
        Tensor* annealing_energies = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &expert_assignments));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({batch_size}), &sample_log_probs));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({batch_size, num_steps}), &annealing_energies));
        
        auto energies_flat = energy_matrix.flat_inner_dims<float, 2>();
        auto assignments_out = expert_assignments->flat<int>();
        auto log_probs_out = sample_log_probs->flat<float>();
        auto anneal_out = annealing_energies->flat_inner_dims<float, 2>();
        
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        
        // Process each batch item
        for (int b = 0; b < batch_size; ++b) {
            // Initialize with random expert
            int current_expert = static_cast<int>(uniform(rng) * num_experts) % num_experts;
            float current_energy = energies_flat(b, current_expert);
            
            // Annealing loop
            for (int step = 0; step < num_steps; ++step) {
                float s = static_cast<float>(step) / static_cast<float>(num_steps - 1);
                float temperature = t_init * (1.0f - s) + t_final * s;
                
                // Propose new expert
                int proposed_expert = static_cast<int>(uniform(rng) * num_experts) % num_experts;
                float proposed_energy = energies_flat(b, proposed_expert);
                
                // Metropolis-Hastings acceptance
                float delta_e = proposed_energy - current_energy;
                float acceptance_prob = std::exp(-delta_e / temperature);
                
                if (uniform(rng) < acceptance_prob) {
                    current_expert = proposed_expert;
                    current_energy = proposed_energy;
                }
                
                anneal_out(b, step) = current_energy;
            }
            
            // Final assignment
            assignments_out(b) = current_expert;
            
            // Compute log probability using softmax at final temperature
            float log_sum_exp = 0.0f;
            float max_neg_energy = -energies_flat(b, 0);
            for (int e = 1; e < num_experts; ++e) {
                max_neg_energy = std::max(max_neg_energy, -energies_flat(b, e));
            }
            for (int e = 0; e < num_experts; ++e) {
                log_sum_exp += std::exp((-energies_flat(b, e) - max_neg_energy) / t_final);
            }
            log_probs_out(b) = (-energies_flat(b, current_expert) - max_neg_energy) / t_final 
                             - std::log(log_sum_exp);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("QbmSample").Device(DEVICE_CPU), QbmSampleOp);

// ==================== Gradient Op (for REINFORCE) ====================
REGISTER_OP("QbmSampleGrad")
    .Input("grad_sample_log_probs: float32")  // [batch]
    .Input("energy_matrix: float32")          // [batch, num_experts]
    .Input("expert_assignments: int32")       // [batch]
    .Input("sample_log_probs: float32")       // [batch]
    .Input("baseline: float32")               // scalar
    .Output("grad_energy_matrix: float32")    // [batch, num_experts]
    .Output("grad_temp_init: float32")        // scalar
    .Output("grad_temp_final: float32")       // scalar
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // same as energy_matrix
        c->set_output(1, c->Scalar());
        c->set_output(2, c->Scalar());
        return Status();
    })
    .Doc("Gradient for QBM sampling using REINFORCE.");

class QbmSampleGradOp : public OpKernel {
public:
    explicit QbmSampleGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_log_probs = ctx->input(0);
        const Tensor& energy_matrix = ctx->input(1);
        const Tensor& expert_assignments = ctx->input(2);
        const Tensor& sample_log_probs = ctx->input(3);
        const Tensor& baseline = ctx->input(4);
        
        const int batch_size = energy_matrix.dim_size(0);
        const int num_experts = energy_matrix.dim_size(1);
        
        // Allocate outputs
        Tensor* grad_energy = nullptr;
        Tensor* grad_t_init = nullptr;
        Tensor* grad_t_final = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, energy_matrix.shape(), &grad_energy));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &grad_t_init));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &grad_t_final));
        
        auto grad_e_out = grad_energy->flat_inner_dims<float, 2>();
        auto assignments = expert_assignments.flat<int>();
        auto grad_in = grad_log_probs.flat<float>();
        float base = baseline.scalar<float>()();
        
        // REINFORCE gradient: grad w.r.t. energy = -(reward - baseline) * grad_log_prob
        for (int b = 0; b < batch_size; ++b) {
            float advantage = grad_in(b) - base;
            for (int e = 0; e < num_experts; ++e) {
                if (e == assignments(b)) {
                    grad_e_out(b, e) = -advantage;
                } else {
                    grad_e_out(b, e) = 0.0f;
                }
            }
        }
        
        // Temperature gradients (zero for now - could be implemented)
        grad_t_init->scalar<float>()() = 0.0f;
        grad_t_final->scalar<float>()() = 0.0f;
    }
};

REGISTER_KERNEL_BUILDER(Name("QbmSampleGrad").Device(DEVICE_CPU), QbmSampleGradOp);

}  // namespace tensorflow
