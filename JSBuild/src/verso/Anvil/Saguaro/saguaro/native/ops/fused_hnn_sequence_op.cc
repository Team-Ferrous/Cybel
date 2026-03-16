// src/ops/fused_hnn_sequence_op.cc
// Copyright 2025 Verso Industries

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "common/parallel/parallel_backend.h"
#include "absl/synchronization/mutex.h"
#include "ops/fused_hnn_sequence/helpers.h" // Includes the inline definitions
#include <algorithm>
#include <atomic>

namespace tensorflow {

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;
using Eigen::RowMajor;
using shape_inference::InferenceContext;

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedHNNSequence")
    .Input("sequence_input: float")       // Input 0: [B, L, D_input]
    .Input("initial_q: float")            // Input 1: [B, D_state]
    .Input("initial_p: float")            // Input 2: [B, D_state]
    .Input("w1: float")
    .Input("b1: float")
    .Input("w2: float")
    .Input("b2: float")
    .Input("w3: float")
    .Input("b3: float")
    .Input("w_out: float")
    .Input("b_out: float")
    .Input("evolution_time_param: float")
    .Output("output_sequence: float")     // Output 0: [B, L, D_output]
    .Output("final_q: float")             // Output 1: [B, D_state]
    .Output("final_p: float")             // Output 2: [B, D_state]
    .Output("h_initial_seq: float")       // Output 3: [B, L, 1] - For energy analysis
    .Output("h_final_seq: float")         // Output 4: [B, L, 1] - For energy analysis
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(1, c->input(1));
        c->set_output(2, c->input(2));
        
        shape_inference::ShapeHandle seq_shape = c->input(0);
        shape_inference::ShapeHandle w_out_shape = c->input(9);
        
        shape_inference::DimensionHandle batch_size = c->Dim(seq_shape, 0);
        shape_inference::DimensionHandle seq_len = c->Dim(seq_shape, 1);
        shape_inference::DimensionHandle d_output = c->Dim(w_out_shape, 1);

        c->set_output(0, c->MakeShape({batch_size, seq_len, d_output}));

        // Set shapes for the new Hamiltonian outputs
        c->set_output(3, c->MakeShape({batch_size, seq_len, 1}));
        c->set_output(4, c->MakeShape({batch_size, seq_len, 1}));

        return OkStatus();
    });

REGISTER_OP("FusedHNNSequenceGrad")
    .Input("grad_output_sequence: float")
    .Input("grad_final_q: float")
    .Input("grad_final_p: float")
    .Input("sequence_input: float")
    .Input("initial_q: float")
    .Input("initial_p: float")
    .Input("w1: float")
    .Input("b1: float")
    .Input("w2: float")
    .Input("b2: float")
    .Input("w3: float")
    .Input("b3: float")
    .Input("w_out: float")
    .Input("b_out: float")
    .Input("evolution_time_param: float")
    .Output("grad_sequence_input: float")
    .Output("grad_initial_q: float")
    .Output("grad_initial_p: float")
    .Output("grad_w1: float")
    .Output("grad_b1: float")
    .Output("grad_w2: float")
    .Output("grad_b2: float")
    .Output("grad_w3: float")
    .Output("grad_b3: float")
    .Output("grad_w_out: float")
    .Output("grad_b_out: float")
    .Output("grad_evolution_time_param: float")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(3));  // grad_sequence_input matches sequence_input
        c->set_output(1, c->input(4));  // grad_initial_q matches initial_q
        c->set_output(2, c->input(5));  // grad_initial_p matches initial_p
        c->set_output(3, c->input(6));  // grad_w1
        c->set_output(4, c->input(7));  // grad_b1
        c->set_output(5, c->input(8));  // grad_w2
        c->set_output(6, c->input(9));  // grad_b2
        c->set_output(7, c->input(10)); // grad_w3
        c->set_output(8, c->input(11)); // grad_b3 (scalar)
        c->set_output(9, c->input(12)); // grad_w_out
        c->set_output(10, c->input(13)); // grad_b_out
        c->set_output(11, c->input(14)); // grad_evolution_time_param (scalar)
        return OkStatus();
    });

// =============================================================================
// FORWARD KERNEL
// =============================================================================
class FusedHNNSequenceOpCpu : public OpKernel {
public:
    explicit FusedHNNSequenceOpCpu(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // --- 1. Get Input Tensors ---
        // Retrieve all input tensors from the context.
        const Tensor& x_seq_tensor = context->input(0);
        const Tensor& q_init_tensor = context->input(1);
        const Tensor& p_init_tensor = context->input(2);
        const Tensor& W1_tensor = context->input(3);
        const Tensor& W_out_tensor = context->input(9);
        const Tensor& b_out_tensor = context->input(10);
        const Tensor& epsilon_param_tensor = context->input(11);
        
        // --- 2. Get Dimensions & Parameters ---
        // Extract dimensions for batch size, sequence length, and state sizes.
        const int64_t batch_size = x_seq_tensor.dim_size(0);
        const int64_t seq_len = x_seq_tensor.dim_size(1);
        const int64_t D_input = x_seq_tensor.dim_size(2);
        const int64_t D_state = q_init_tensor.dim_size(1);
        const int64_t D_in = 2 * D_state + D_input; // Combined input dimension for HNN
        const int64_t D_h = W1_tensor.dim_size(1); // Hidden dimension
        const int64_t D_output = b_out_tensor.dim_size(0);
        
        // --- 3. Extract Scalar Parameters ---
        // Phase 3.3: Removed hard tanh+min constraints - soft-potential regularization handles stability
        const float epsilon_param = epsilon_param_tensor.scalar<float>()();
        const float epsilon = epsilon_param; // Use raw parameter directly
        const float b3_scalar = context->input(8).scalar<float>()(); // b3 is a scalar bias

        // --- 4. Create Eigen Maps for Weights ---
        // Map the flat TensorFlow tensors to Eigen matrices and vectors.
        // This allows using Eigen's high-level API without copying data.
        // These are constant across all time steps in the sequence.
        Map<const MatrixXf> W1_map(context->input(3).flat<float>().data(), D_in, D_h);
        Map<const VectorXf> b1_map(context->input(4).flat<float>().data(), D_h);
        Map<const MatrixXf> W2_map(context->input(5).flat<float>().data(), D_h, D_h);
        Map<const VectorXf> b2_map(context->input(6).flat<float>().data(), D_h);
        Map<const MatrixXf> W3_map(context->input(7).flat<float>().data(), D_h, 1);
        Map<const MatrixXf> W_out_map(W_out_tensor.flat<float>().data(), 2 * D_state, D_output);
        Map<const VectorXf> b_out_map(b_out_tensor.flat<float>().data(), D_output);

        // --- 5. Allocate Output Tensors ---
        // Allocate memory for all output tensors.
        Tensor* output_seq_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size, seq_len, D_output}), &output_seq_tensor));
        Tensor* q_final_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, q_init_tensor.shape(), &q_final_tensor));
        Tensor* p_final_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, p_init_tensor.shape(), &p_final_tensor));
        Tensor* h_initial_seq_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({batch_size, seq_len, 1}), &h_initial_seq_tensor));
        Tensor* h_final_seq_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({batch_size, seq_len, 1}), &h_final_seq_tensor));
        
        // --- 6. Main Computation Loop (Parallelized over Batch) ---
        // The `work` lambda contains the logic for a single batch item.
        // `saguaro::parallel::ForShard` handles parallel execution.
        auto work = [&](int64_t start, int64_t end) {
            for (int b = start; b < end; ++b) {
                // Initialize state for the current batch item.
                VectorXf q_t = Map<const VectorXf>(q_init_tensor.flat<float>().data() + b * D_state, D_state);
                VectorXf p_t = Map<const VectorXf>(p_init_tensor.flat<float>().data() + b * D_state, D_state);
                
                // Iterate over the sequence length.
                for (int l = 0; l < seq_len; ++l) {
                    const float* x_l_ptr = x_seq_tensor.flat<float>().data() + b * seq_len * D_input + l * D_input;
                    Map<const VectorXf> x_l(x_l_ptr, D_input);

                    // --- 6.1. Yoshida 4th-Order Symplectic Integration (Phase 3.1) ---
                    // Calculate initial Hamiltonian for energy monitoring
                    VectorXf z(D_in); z << q_t, p_t, x_l;
                    auto intermediates1 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);

                    // Compose three 2nd-order leapfrog steps with Yoshida coefficients
                    const float dt1 = static_cast<float>(YoshidaCoefficients::w1 * epsilon);
                    const float dt0 = static_cast<float>(YoshidaCoefficients::w0 * epsilon);

                    // First Yoshida sub-step (w1 * dt)
                    leapfrog_substep(q_t, p_t, x_l, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar, dt1, D_state, D_in);

                    // Second Yoshida sub-step (w0 * dt)
                    leapfrog_substep(q_t, p_t, x_l, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar, dt0, D_state, D_in);

                    // Third Yoshida sub-step (w1 * dt)
                    leapfrog_substep(q_t, p_t, x_l, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar, dt1, D_state, D_in);

                    // Final state after Yoshida integration
                    VectorXf q_next = q_t;
                    VectorXf p_next = p_t;
                    
                    // --- 6.2. Output Projection ---
                    // Project the final state (q_next, p_next) to the output dimension.
                    VectorXf final_state(2 * D_state);
                    final_state << q_next, p_next;
                    VectorXf output_l = (final_state.transpose() * W_out_map).transpose() + b_out_map;

                    // --- 6.3. Update State & Write Output ---
                    // Update the state for the next time step.
                    q_t = q_next;
                    p_t = p_next;
                    
                    // Write the output for the current time step.
                    float* output_l_ptr = output_seq_tensor->flat<float>().data() + b * seq_len * D_output + l * D_output;
                    Map<VectorXf>(output_l_ptr, D_output) = output_l;

                    // --- 6.4. Write Hamiltonian Values for Analysis ---
                    // These are auxiliary outputs for monitoring energy conservation.
                    VectorXf final_state_for_H(D_in); final_state_for_H << q_next, p_next, x_l;
                    auto intermediates_final = compute_H_and_intermediates(final_state_for_H, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    h_initial_seq_tensor->tensor<float, 3>()(b, l, 0) = intermediates1.H;
                    h_final_seq_tensor->tensor<float, 3>()(b, l, 0) = intermediates_final.H;
                }
                
                // --- 7. Write Final State of the Sequence ---
                // After the loop, write the final q and p to the output tensors.
                float* q_final_ptr = q_final_tensor->flat<float>().data() + b * D_state;
                Map<VectorXf>(q_final_ptr, D_state) = q_t;
                float* p_final_ptr = p_final_tensor->flat<float>().data() + b * D_state;
                Map<VectorXf>(p_final_ptr, D_state) = p_t;
            }
        };

        // --- 8. Execute Parallel Computation ---
        // Define the cost per unit for the parallel scheduler and run the work.
        const std::size_t cost_per_unit =
            static_cast<std::size_t>(seq_len * 1000 * D_in * D_h);
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(batch_size),
            cost_per_unit,
            work);
    }
};
REGISTER_KERNEL_BUILDER(Name("FusedHNNSequence").Device(DEVICE_CPU), FusedHNNSequenceOpCpu);

// =============================================================================
// BACKWARD KERNEL
// =============================================================================

struct HNNForwardState {
    VectorXf q_t; VectorXf p_t; VectorXf x_t;
    VectorXf p_half;
    VectorXf q_next; VectorXf p_next;
    HNNIntermediate int1, int2, int3;
    VectorXf dH_dz1, dH_dz2, dH_dz3;
};

class FusedHNNSequenceGradOpCpu : public OpKernel {
public:
    explicit FusedHNNSequenceGradOpCpu(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // --- 1. Get Input Tensors ---
        // Gradients from the upstream operations and original inputs from the forward pass.
        const Tensor& grad_y_seq_tensor = context->input(0);
        const Tensor& grad_q_final_tensor = context->input(1);
        const Tensor& grad_p_final_tensor = context->input(2);
        const Tensor& x_seq_tensor = context->input(3);
        const Tensor& q_init_tensor = context->input(4);
        const Tensor& p_init_tensor = context->input(5);
        
        // --- 2. Get Dimensions & Parameters ---
        const int64_t batch_size = x_seq_tensor.dim_size(0);
        const int64_t seq_len = x_seq_tensor.dim_size(1);
        const int64_t D_input = x_seq_tensor.dim_size(2);
        const int64_t D_state = q_init_tensor.dim_size(1);
        const int64_t D_in = 2 * D_state + D_input;
        const int64_t D_h = context->input(6).dim_size(1);
        const int64_t D_output = context->input(13).dim_size(0);
        
        const float epsilon_param = context->input(14).scalar<float>()();
        const float epsilon = epsilon_param; // Phase 3.3: Use raw parameter directly
        const float b3_scalar = context->input(11).scalar<float>()();
        
        // --- 3. Allocate Output Tensors (Gradients for Inputs) ---
        Tensor* grad_x_seq_tensor; OP_REQUIRES_OK(context, context->allocate_output(0, x_seq_tensor.shape(), &grad_x_seq_tensor));
        Tensor* grad_q_init_tensor; OP_REQUIRES_OK(context, context->allocate_output(1, q_init_tensor.shape(), &grad_q_init_tensor));
        Tensor* grad_p_init_tensor; OP_REQUIRES_OK(context, context->allocate_output(2, p_init_tensor.shape(), &grad_p_init_tensor));
        Tensor* grad_W1_tensor; OP_REQUIRES_OK(context, context->allocate_output(3, context->input(6).shape(), &grad_W1_tensor));
        Tensor* grad_b1_tensor; OP_REQUIRES_OK(context, context->allocate_output(4, context->input(7).shape(), &grad_b1_tensor));
        Tensor* grad_W2_tensor; OP_REQUIRES_OK(context, context->allocate_output(5, context->input(8).shape(), &grad_W2_tensor));
        Tensor* grad_b2_tensor; OP_REQUIRES_OK(context, context->allocate_output(6, context->input(9).shape(), &grad_b2_tensor));
        Tensor* grad_W3_tensor; OP_REQUIRES_OK(context, context->allocate_output(7, context->input(10).shape(), &grad_W3_tensor));
        Tensor* grad_b3_tensor; OP_REQUIRES_OK(context, context->allocate_output(8, context->input(11).shape(), &grad_b3_tensor));
        Tensor* grad_W_out_tensor; OP_REQUIRES_OK(context, context->allocate_output(9, context->input(12).shape(), &grad_W_out_tensor));
        Tensor* grad_b_out_tensor; OP_REQUIRES_OK(context, context->allocate_output(10, context->input(13).shape(), &grad_b_out_tensor));
        Tensor* grad_epsilon_param_tensor; OP_REQUIRES_OK(context, context->allocate_output(11, context->input(14).shape(), &grad_epsilon_param_tensor));

        // --- 4. Zero Gradients & Initialize Accumulators ---
        // It's crucial to zero out all gradient tensors before accumulation.
        grad_x_seq_tensor->flat<float>().setZero(); grad_q_init_tensor->flat<float>().setZero(); grad_p_init_tensor->flat<float>().setZero();
        grad_W1_tensor->flat<float>().setZero(); grad_b1_tensor->flat<float>().setZero();
        grad_W2_tensor->flat<float>().setZero(); grad_b2_tensor->flat<float>().setZero();
        grad_W3_tensor->flat<float>().setZero(); grad_b3_tensor->flat<float>().setZero();
        grad_W_out_tensor->flat<float>().setZero(); grad_b_out_tensor->flat<float>().setZero();
        grad_epsilon_param_tensor->flat<float>().setZero();

        // Per-thread accumulators for weight gradients to avoid race conditions.
        MatrixXf grad_W1_acc = MatrixXf::Zero(D_in, D_h);
        VectorXf grad_b1_acc = VectorXf::Zero(D_h);
        MatrixXf grad_W2_acc = MatrixXf::Zero(D_h, D_h);
        VectorXf grad_b2_acc = VectorXf::Zero(D_h);
        MatrixXf grad_W3_acc = MatrixXf::Zero(D_h, 1);
        float grad_b3_acc = 0.0f; 
        MatrixXf grad_W_out_acc = MatrixXf::Zero(2 * D_state, D_output);
        VectorXf grad_b_out_acc = VectorXf::Zero(D_output);
        std::atomic<float> grad_epsilon_param_acc(0.0f);
        
        // --- 5. Create Eigen Maps for Weights ---
        Map<const MatrixXf> W1_map(context->input(6).flat<float>().data(), D_in, D_h);
        Map<const VectorXf> b1_map(context->input(7).flat<float>().data(), D_h);
        Map<const MatrixXf> W2_map(context->input(8).flat<float>().data(), D_h, D_h);
        Map<const VectorXf> b2_map(context->input(9).flat<float>().data(), D_h);
        Map<const MatrixXf> W3_map(context->input(10).flat<float>().data(), D_h, 1);
        Map<const MatrixXf> W_out_map(context->input(12).flat<float>().data(), 2 * D_state, D_output);

        // --- 6. Recompute Forward Pass to Get Intermediate Activations ---
        // This is a common strategy in custom gradients (the "re-materialization" trick)
        // to save memory, as storing all intermediate states from the forward pass
        // would be too memory-intensive.
        std::vector<std::vector<HNNForwardState>> forward_states(batch_size, std::vector<HNNForwardState>(seq_len));

        auto recompute_work = [&](int64_t start, int64_t end) {
            for (int b = start; b < end; ++b) {
                VectorXf q_t = Map<const VectorXf>(q_init_tensor.flat<float>().data() + b * D_state, D_state);
                VectorXf p_t = Map<const VectorXf>(p_init_tensor.flat<float>().data() + b * D_state, D_state);

                for (int l = 0; l < seq_len; ++l) {
                    const float* x_l_ptr = x_seq_tensor.flat<float>().data() + b * seq_len * D_input + l * D_input;
                    Map<const VectorXf> x_l(x_l_ptr, D_input);

                    // Store all necessary states for the backward pass.
                    HNNForwardState& state = forward_states[b][l];
                    state.q_t = q_t; state.p_t = p_t; state.x_t = x_l;

                    VectorXf z1(D_in); z1 << q_t, p_t, x_l;
                    state.int1 = compute_H_and_intermediates(z1, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    state.dH_dz1 = compute_dH_dz(state.int1, W1_map, W2_map, W3_map);
                    state.p_half = p_t - (epsilon / 2.0f) * state.dH_dz1.head(D_state);

                    VectorXf z2(D_in); z2 << q_t, state.p_half, x_l;
                    state.int2 = compute_H_and_intermediates(z2, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    state.dH_dz2 = compute_dH_dz(state.int2, W1_map, W2_map, W3_map);
                    VectorXf q_next = q_t + epsilon * state.dH_dz2.segment(D_state, D_state);

                    VectorXf z3(D_in); z3 << q_next, state.p_half, x_l;
                    state.int3 = compute_H_and_intermediates(z3, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    state.dH_dz3 = compute_dH_dz(state.int3, W1_map, W2_map, W3_map);
                    VectorXf p_next = state.p_half - (epsilon / 2.0f) * state.dH_dz3.head(D_state);

                    state.q_next = q_next; state.p_next = p_next;
                    q_t = q_next; p_t = p_next;
                }
            }
        };
        const std::size_t recompute_cost =
            static_cast<std::size_t>(seq_len * 1000 * D_in * D_h);
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(batch_size),
            recompute_cost,
            recompute_work);
        
        // --- 7. Backward Pass (Adjoint Sensitivity Analysis) ---
        // The backward pass is unrolled in reverse time order (from l=seq_len-1 to 0).
        // It's parallelized over the batch dimension.
        auto backward_work = [&](int64_t start, int64_t end) {
            // Each thread gets its own local accumulator for weight gradients.
            MatrixXf local_grad_W1 = MatrixXf::Zero(D_in, D_h);
            VectorXf local_grad_b1 = VectorXf::Zero(D_h);
            MatrixXf local_grad_W2 = MatrixXf::Zero(D_h, D_h);
            VectorXf local_grad_b2 = VectorXf::Zero(D_h);
            MatrixXf local_grad_W3 = MatrixXf::Zero(D_h, 1);
            float local_grad_b3 = 0.0f;
            MatrixXf local_grad_W_out = MatrixXf::Zero(2 * D_state, D_output);
            VectorXf local_grad_b_out = VectorXf::Zero(D_output);
            float local_grad_epsilon_param = 0.0f;

            for (int b = start; b < end; ++b) {
                // Initialize state gradients with the final upstream gradients.
                VectorXf grad_q = Map<const VectorXf>(grad_q_final_tensor.flat<float>().data() + b * D_state, D_state);
                VectorXf grad_p = Map<const VectorXf>(grad_p_final_tensor.flat<float>().data() + b * D_state, D_state);
                
                for (int l = seq_len - 1; l >= 0; --l) {
                    const HNNForwardState& state = forward_states[b][l];
                    
                    // --- 7.1. Backprop through Output Projection ---
                    const float* grad_y_l_ptr = grad_y_seq_tensor.flat<float>().data() + b * seq_len * D_output + l * D_output;
                    Map<const VectorXf> grad_output_l(grad_y_l_ptr, D_output);

                    VectorXf current_final_state(2 * D_state);
                    current_final_state << state.q_next, state.p_next;
                    
                    local_grad_W_out += current_final_state * grad_output_l.transpose();
                    local_grad_b_out += grad_output_l;
                    
                    VectorXf grad_final_state = W_out_map * grad_output_l;
                    grad_q += grad_final_state.head(D_state);
                    grad_p += grad_final_state.tail(D_state);

                    // --- 7.2. Backprop through Leapfrog Step (in reverse) ---
                    // This follows the chain rule back through the three stages of the leapfrog integrator.
                    
                    // Backprop through p_next = p_half - (epsilon / 2.0f) * dH_dz3.head(D_state)
                    VectorXf grad_p_half = grad_p;
                    float grad_H3_scalar = grad_p.dot(-0.5f * epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int3, W1_map, W2_map, W3_map, grad_H3_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                    VectorXf grad_z3 = state.dH_dz3 * grad_H3_scalar;
                    
                    grad_q += grad_z3.head(D_state);
                    grad_p_half += grad_z3.segment(D_state, D_state);
                    Map<VectorXf>(grad_x_seq_tensor->flat<float>().data() + b * seq_len * D_input + l * D_input, D_input) += grad_z3.tail(D_input);

                    // Backprop through q_next = q_t + epsilon * dH_dz2.segment(D_state, D_state)
                    float grad_H2_scalar = grad_q.dot(epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int2, W1_map, W2_map, W3_map, grad_H2_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                    VectorXf grad_z2 = state.dH_dz2 * grad_H2_scalar;
                    grad_q += grad_z2.head(D_state);
                    grad_p_half += grad_z2.segment(D_state, D_state);
                    Map<VectorXf>(grad_x_seq_tensor->flat<float>().data() + b * seq_len * D_input + l * D_input, D_input) += grad_z2.tail(D_input);
                
                    // Backprop through p_half = p_t - (epsilon / 2.0f) * dH_dz1.head(D_state)
                    VectorXf grad_p_t = grad_p_half;
                    float grad_H1_scalar = grad_p_half.dot(-0.5f * epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int1, W1_map, W2_map, W3_map, grad_H1_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                    VectorXf grad_z1 = state.dH_dz1 * grad_H1_scalar;

                    // Update gradients for the inputs of this time step (q_t, p_t, x_t)
                    grad_q = grad_q + grad_z1.head(D_state);
                    grad_p = grad_p_t + grad_z1.segment(D_state, D_state);
                    Map<VectorXf>(grad_x_seq_tensor->flat<float>().data() + b * seq_len * D_input + l * D_input, D_input) += grad_z1.tail(D_input);
                    
                    // --- 7.3. Backprop through epsilon parameter ---
                    float grad_eps = 0;
                    grad_eps -= (grad_p.transpose() * state.dH_dz3.head(D_state) / 2.0f)(0,0);
                    grad_eps += (grad_q.transpose() * state.dH_dz2.segment(D_state, D_state))(0,0);
                    grad_eps -= (grad_p_half.transpose() * state.dH_dz1.head(D_state) / 2.0f)(0,0);
                    // Phase 3.3: Direct gradient since epsilon = epsilon_param
                    local_grad_epsilon_param += grad_eps;
                }
                
                // Write the final gradients for the initial state of this batch item.
                Map<VectorXf>(grad_q_init_tensor->flat<float>().data() + b * D_state, D_state) = grad_q;
                Map<VectorXf>(grad_p_init_tensor->flat<float>().data() + b * D_state, D_state) = grad_p;
            }

            // Phase 1.2: Floquet-Guided Gradient Clipping (HNN_TIMECRYSTAL_ENHANCEMENT_ROADMAP)
            // Enhanced gradient stability with evolution-time-aware rescaling.
            // Larger evolution_time → more aggressive damping to prevent drift explosion.
            // Uses implicit regularization: scale = 1 / (1 + adaptive_factor * ||grad||²)
            float grad_norm_sq = 0.0f;
            for (int i = 0; i < D_in * D_h; ++i) {
                grad_norm_sq += local_grad_W1.data()[i] * local_grad_W1.data()[i];
            }
            
            // Phase 1.2: Evolution-time-aware stability factor
            // Base factor: 1e-6, boosted by evolution time magnitude
            // sin(epsilon * phase_scale) provides periodic modulation aligned with Floquet dynamics
            constexpr float BASE_STABILITY = 1e-6f;
            constexpr float PHASE_SCALE = 100.0f;
            float phase_factor = 1.0f + 0.5f * std::abs(std::sin(epsilon * PHASE_SCALE));
            float adaptive_stability = BASE_STABILITY * phase_factor * (1.0f + 10.0f * epsilon);
            float stability_factor = 1.0f / (1.0f + adaptive_stability * grad_norm_sq);
            
            local_grad_W1 *= stability_factor;
            local_grad_b1 *= stability_factor;
            local_grad_W2 *= stability_factor;
            local_grad_b2 *= stability_factor;
            local_grad_W3 *= stability_factor;
            local_grad_b3 *= stability_factor;
            local_grad_W_out *= stability_factor;
            local_grad_b_out *= stability_factor;

            // --- 8. Atomically Update Global Gradient Accumulators ---
            // Lock to prevent race conditions when updating the shared accumulators.
            absl::MutexLock l(&mu_);
            grad_W1_acc += local_grad_W1;
            grad_b1_acc += local_grad_b1;
            grad_W2_acc += local_grad_W2;
            grad_b2_acc += local_grad_b2;
            grad_W3_acc += local_grad_W3;
            grad_b3_acc += local_grad_b3;
            grad_W_out_acc += local_grad_W_out;
            grad_b_out_acc += local_grad_b_out;
            // Use atomic exchange for the scalar epsilon gradient.
            auto current_eps = grad_epsilon_param_acc.load();
            while (!grad_epsilon_param_acc.compare_exchange_weak(current_eps, current_eps + local_grad_epsilon_param));
        };
        
        // --- 9. Execute Parallel Backward Pass ---
        const std::size_t backward_cost =
            static_cast<std::size_t>(seq_len * 3000 * D_in * D_h);
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(batch_size),
            backward_cost,
            backward_work);

        // --- 10. Write Final Accumulated Gradients to Output Tensors ---
        Map<MatrixXf>(grad_W1_tensor->flat<float>().data(), D_in, D_h) = grad_W1_acc;
        Map<VectorXf>(grad_b1_tensor->flat<float>().data(), D_h) = grad_b1_acc;
        Map<MatrixXf>(grad_W2_tensor->flat<float>().data(), D_h, D_h) = grad_W2_acc;
        Map<VectorXf>(grad_b2_tensor->flat<float>().data(), D_h) = grad_b2_acc;
        Map<MatrixXf>(grad_W3_tensor->flat<float>().data(), D_h, 1) = grad_W3_acc;
        grad_b3_tensor->scalar<float>()() = grad_b3_acc;
        Map<MatrixXf>(grad_W_out_tensor->flat<float>().data(), 2 * D_state, D_output) = grad_W_out_acc;
        Map<VectorXf>(grad_b_out_tensor->flat<float>().data(), D_output) = grad_b_out_acc;
        grad_epsilon_param_tensor->scalar<float>()() = grad_epsilon_param_acc.load();
    }
private:
    absl::Mutex mu_;
};
REGISTER_KERNEL_BUILDER(Name("FusedHNNSequenceGrad").Device(DEVICE_CPU), FusedHNNSequenceGradOpCpu);

} // namespace tensorflow
