// saguaro.native/ops/vqc_expectation_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// VQC (Variational Quantum Circuit) Expectation operator
// Computes expectation values for quantum circuit simulations
// Phase 1 Enhancement: Real gradient computation via parameter-shift rule and SPSA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cmath>
#include <complex>
#include <vector>
#include <random>
#include <algorithm>

namespace tensorflow {

// =============================================================================
// Constants and Configuration
// =============================================================================

constexpr float PI_HALF = 1.5707963267948966f;
constexpr float SPSA_PERTURBATION = 0.1f;
constexpr int PARAM_THRESHOLD_GUIDED_SPSA = 50;
constexpr int PARAM_THRESHOLD_PURE_SPSA = 500;
constexpr float GUIDED_SPSA_SAMPLE_RATIO = 0.2f;

// Gradient computation modes
enum class VQCGradientMode {
    FULL_PARAMETER_SHIFT = 0,    // Exact gradients, O(2p) evaluations
    GUIDED_SPSA = 1,             // Hybrid, O(2 + αp) evaluations
    PURE_SPSA = 2,               // Approximate, O(2) evaluations
    AUTO = 3                     // Auto-select based on param count
};

// =============================================================================
// Helper Functions
// =============================================================================

// Thread-local RNG for SPSA
thread_local std::mt19937 tl_rng(std::random_device{}());

std::vector<float> generate_bernoulli_perturbation(int size) {
    std::vector<float> delta(size);
    std::uniform_int_distribution<int> dist(0, 1);
    for (int i = 0; i < size; ++i) {
        delta[i] = (dist(tl_rng) == 0) ? -1.0f : 1.0f;
    }
    return delta;
}

// Apply single-qubit RZ rotation to statevector
// Phase 1.7: Uses double precision for quantum computations
void apply_rz_gate(std::vector<std::complex<double>>& state, int qubit, double angle, int num_qubits) {
    const int state_dim = 1 << num_qubits;
    std::complex<double> phase_pos = {std::cos(angle / 2), -std::sin(angle / 2)};
    std::complex<double> phase_neg = std::conj(phase_pos);

    for (int i = 0; i < state_dim; ++i) {
        if ((i >> qubit) & 1) {
            state[i] *= phase_neg;
        } else {
            state[i] *= phase_pos;
        }
    }
}

// Apply RY rotation
// Phase 1.7: Uses double precision for quantum computations
void apply_ry_gate(std::vector<std::complex<double>>& state, int qubit, double angle, int num_qubits) {
    const int state_dim = 1 << num_qubits;
    double c = std::cos(angle / 2);
    double s = std::sin(angle / 2);

    for (int i = 0; i < state_dim; ++i) {
        int partner = i ^ (1 << qubit);  // Flip qubit
        if (i < partner) {  // Process each pair once
            std::complex<double> a0 = state[i];
            std::complex<double> a1 = state[partner];

            if ((i >> qubit) & 1) {
                // i has qubit=1, partner has qubit=0
                state[i] = c * a0 + s * a1;
                state[partner] = -s * a0 + c * a1;
            } else {
                // i has qubit=0, partner has qubit=1
                state[i] = c * a0 - s * a1;
                state[partner] = s * a0 + c * a1;
            }
        }
    }
}

// Apply CNOT gate
// Phase 1.7: Uses double precision for quantum computations
void apply_cnot_gate(std::vector<std::complex<double>>& state, int control, int target, int num_qubits) {
    const int state_dim = 1 << num_qubits;

    for (int i = 0; i < state_dim; ++i) {
        // If control qubit is 1, swap amplitudes based on target qubit
        if ((i >> control) & 1) {
            int partner = i ^ (1 << target);
            if (i < partner) {
                std::swap(state[i], state[partner]);
            }
        }
    }
}

// Compute Z expectation for a Pauli string
// Phase 1.7: Uses double precision for quantum computations
double compute_pauli_z_expectation(
    const std::vector<std::complex<double>>& state,
    const int* pauli_indices,
    int num_qubits
) {
    double expectation = 0.0;
    const int state_dim = static_cast<int>(state.size());

    for (int i = 0; i < state_dim; ++i) {
        int parity = 0;
        for (int q = 0; q < num_qubits; ++q) {
            if (pauli_indices[q] == 3 && ((i >> q) & 1)) {  // Z Pauli
                parity ^= 1;
            }
        }
        double prob = std::norm(state[i]);
        expectation += (parity ? -1.0 : 1.0) * prob;
    }

    return expectation;
}

// =============================================================================
// Core VQC Simulation
// =============================================================================

// Phase 1.7: Uses double precision internally, returns float for TF compatibility
float run_vqc_circuit(
    const float* data_angles,
    const float* circuit_params,
    const int* entangler_pairs,
    const int* measurement_paulis,
    const float* measurement_coeffs,
    int num_qubits,
    int num_layers,
    int num_params_per_gate,
    int num_entanglers,
    int num_measurements,
    int batch_idx
) {
    const int state_dim = 1 << num_qubits;
    std::vector<std::complex<double>> state(state_dim, {0.0, 0.0});
    state[0] = {1.0, 0.0};  // |0...0⟩

    // 1. Data encoding layer (RZ gates) - cast to double for internal precision
    for (int q = 0; q < num_qubits; ++q) {
        double angle = static_cast<double>(data_angles[batch_idx * num_qubits + q]);
        apply_rz_gate(state, q, angle, num_qubits);
    }

    // 2. Variational layers
    for (int layer = 0; layer < num_layers; ++layer) {
        // Single-qubit rotations (RY, RZ)
        for (int q = 0; q < num_qubits; ++q) {
            int param_base = layer * num_qubits * num_params_per_gate + q * num_params_per_gate;

            // RY rotation
            if (num_params_per_gate >= 1) {
                apply_ry_gate(state, q, static_cast<double>(circuit_params[param_base]), num_qubits);
            }
            // RZ rotation
            if (num_params_per_gate >= 2) {
                apply_rz_gate(state, q, static_cast<double>(circuit_params[param_base + 1]), num_qubits);
            }
            // Second RY (if 3 params per gate)
            if (num_params_per_gate >= 3) {
                apply_ry_gate(state, q, static_cast<double>(circuit_params[param_base + 2]), num_qubits);
            }
        }

        // Entangling layer (CNOT gates)
        for (int e = 0; e < num_entanglers; ++e) {
            int control = entangler_pairs[e * 2];
            int target = entangler_pairs[e * 2 + 1];
            apply_cnot_gate(state, control, target, num_qubits);
        }
    }

    // 3. Compute expectation values
    double total_expectation = 0.0;
    for (int m = 0; m < num_measurements; ++m) {
        double coeff = static_cast<double>(measurement_coeffs[m]);
        double exp_val = compute_pauli_z_expectation(
            state,
            measurement_paulis + m * num_qubits,
            num_qubits
        );
        total_expectation += coeff * exp_val;
    }

    // Cast back to float for TensorFlow interface
    return static_cast<float>(total_expectation);
}

// =============================================================================
// Forward Op (unchanged interface, improved implementation)
// =============================================================================

REGISTER_OP("RunVqcExpectation")
    .Input("data_angles: float32")        // [batch, num_qubits]
    .Input("circuit_params: float32")     // [num_layers, num_qubits, num_params_per_gate]
    .Input("entangler_pairs: int32")      // [num_entanglers, 2]
    .Input("measurement_paulis: int32")   // [num_measurements, num_qubits]
    .Input("measurement_coeffs: float32") // [num_measurements]
    .Output("expectations: float32")       // [batch]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle data_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &data_shape));
        c->set_output(0, c->Vector(c->Dim(data_shape, 0)));
        return Status();
    })
    .Doc("Compute VQC expectation values for quantum circuit.");

class RunVqcExpectationOp : public OpKernel {
public:
    explicit RunVqcExpectationOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& data_angles = ctx->input(0);
        const Tensor& circuit_params = ctx->input(1);
        const Tensor& entangler_pairs = ctx->input(2);
        const Tensor& measurement_paulis = ctx->input(3);
        const Tensor& measurement_coeffs = ctx->input(4);

        const int batch_size = data_angles.dim_size(0);
        const int num_qubits = data_angles.dim_size(1);
        const int num_layers = circuit_params.dim_size(0);
        const int num_params_per_gate = circuit_params.dim_size(2);
        const int num_entanglers = entangler_pairs.dim_size(0);
        const int num_measurements = measurement_coeffs.dim_size(0);

        // Allocate output
        Tensor* expectations = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &expectations));

        auto data_ptr = data_angles.flat<float>().data();
        auto params_ptr = circuit_params.flat<float>().data();
        auto entanglers_ptr = entangler_pairs.flat<int>().data();
        auto paulis_ptr = measurement_paulis.flat<int>().data();
        auto coeffs_ptr = measurement_coeffs.flat<float>().data();
        auto out_flat = expectations->flat<float>();

        // Parallel batch processing
        #pragma omp parallel for schedule(dynamic)
        for (int b = 0; b < batch_size; ++b) {
            out_flat(b) = run_vqc_circuit(
                data_ptr, params_ptr, entanglers_ptr, paulis_ptr, coeffs_ptr,
                num_qubits, num_layers, num_params_per_gate,
                num_entanglers, num_measurements, b
            );
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("RunVqcExpectation").Device(DEVICE_CPU), RunVqcExpectationOp);

// =============================================================================
// Gradient Op - REAL IMPLEMENTATION with Parameter-Shift Rule and SPSA
// =============================================================================

REGISTER_OP("RunVqcExpectationGrad")
    .Input("grad_expectations: float32")   // [batch]
    .Input("data_angles: float32")         // [batch, num_qubits]
    .Input("circuit_params: float32")      // [num_layers, num_qubits, num_params_per_gate]
    .Input("entangler_pairs: int32")       // [num_entanglers, 2]
    .Input("measurement_paulis: int32")    // [num_measurements, num_qubits]
    .Input("measurement_coeffs: float32")  // [num_measurements]
    .Attr("gradient_mode: int = 3")        // 0=FULL_PSR, 1=GUIDED_SPSA, 2=PURE_SPSA, 3=AUTO
    .Attr("spsa_sample_ratio: float = 0.2") // Ratio for Guided-SPSA exact params
    .Output("grad_data_angles: float32")   // [batch, num_qubits]
    .Output("grad_circuit_params: float32") // same as circuit_params
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // same as data_angles
        c->set_output(1, c->input(2));  // same as circuit_params
        return Status();
    })
    .Doc("Gradient for VQC expectation using parameter-shift rule or SPSA.");

class RunVqcExpectationGradOp : public OpKernel {
private:
    VQCGradientMode gradient_mode_;
    float spsa_sample_ratio_;

public:
    explicit RunVqcExpectationGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        int mode;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gradient_mode", &mode));
        gradient_mode_ = static_cast<VQCGradientMode>(mode);
        OP_REQUIRES_OK(ctx, ctx->GetAttr("spsa_sample_ratio", &spsa_sample_ratio_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_exp = ctx->input(0);
        const Tensor& data_angles = ctx->input(1);
        const Tensor& circuit_params = ctx->input(2);
        const Tensor& entangler_pairs = ctx->input(3);
        const Tensor& measurement_paulis = ctx->input(4);
        const Tensor& measurement_coeffs = ctx->input(5);

        const int batch_size = data_angles.dim_size(0);
        const int num_qubits = data_angles.dim_size(1);
        const int num_layers = circuit_params.dim_size(0);
        const int num_params_per_gate = circuit_params.dim_size(2);
        const int num_entanglers = entangler_pairs.dim_size(0);
        const int num_measurements = measurement_coeffs.dim_size(0);
        const int total_params = num_layers * num_qubits * num_params_per_gate;

        // Allocate outputs
        Tensor* grad_data = nullptr;
        Tensor* grad_params = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, data_angles.shape(), &grad_data));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, circuit_params.shape(), &grad_params));

        auto grad_data_flat = grad_data->flat<float>();
        auto grad_params_flat = grad_params->flat<float>();

        // Get pointers
        auto upstream_grad = grad_exp.flat<float>();
        auto data_ptr = data_angles.flat<float>().data();
        auto params_ptr = circuit_params.flat<float>().data();
        auto entanglers_ptr = entangler_pairs.flat<int>().data();
        auto paulis_ptr = measurement_paulis.flat<int>().data();
        auto coeffs_ptr = measurement_coeffs.flat<float>().data();

        // Auto-select gradient mode based on parameter count
        VQCGradientMode mode = gradient_mode_;
        if (mode == VQCGradientMode::AUTO) {
            if (total_params < PARAM_THRESHOLD_GUIDED_SPSA) {
                mode = VQCGradientMode::FULL_PARAMETER_SHIFT;
            } else if (total_params < PARAM_THRESHOLD_PURE_SPSA) {
                mode = VQCGradientMode::GUIDED_SPSA;
            } else {
                mode = VQCGradientMode::PURE_SPSA;
            }
        }

        // Create mutable params buffer
        std::vector<float> params_buffer(params_ptr, params_ptr + total_params);

        // Initialize gradients
        std::fill_n(grad_data_flat.data(), batch_size * num_qubits, 0.0f);
        std::fill_n(grad_params_flat.data(), total_params, 0.0f);

        // ===================================================================
        // Compute gradients based on selected mode
        // ===================================================================

        if (mode == VQCGradientMode::FULL_PARAMETER_SHIFT) {
            // Full parameter-shift rule: exact gradients
            // df/dθ = [f(θ+π/2) - f(θ-π/2)] / 2

            #pragma omp parallel for schedule(dynamic)
            for (int p = 0; p < total_params; ++p) {
                std::vector<float> params_plus = params_buffer;
                std::vector<float> params_minus = params_buffer;
                params_plus[p] += PI_HALF;
                params_minus[p] -= PI_HALF;

                float grad_sum = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    float exp_plus = run_vqc_circuit(
                        data_ptr, params_plus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                        num_qubits, num_layers, num_params_per_gate,
                        num_entanglers, num_measurements, b
                    );
                    float exp_minus = run_vqc_circuit(
                        data_ptr, params_minus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                        num_qubits, num_layers, num_params_per_gate,
                        num_entanglers, num_measurements, b
                    );

                    float param_grad = (exp_plus - exp_minus) / 2.0f;
                    grad_sum += upstream_grad(b) * param_grad;
                }

                grad_params_flat(p) = grad_sum;
            }

        } else if (mode == VQCGradientMode::GUIDED_SPSA) {
            // Guided-SPSA: exact PSR for high-variance params, SPSA for rest
            // Reference: arXiv:2404.15751

            int num_exact = static_cast<int>(total_params * spsa_sample_ratio_);
            num_exact = std::max(1, std::min(num_exact, total_params));

            // Generate SPSA perturbation
            std::vector<float> delta = generate_bernoulli_perturbation(total_params);

            // Compute SPSA gradient estimate for all parameters
            std::vector<float> params_plus = params_buffer;
            std::vector<float> params_minus = params_buffer;
            for (int p = 0; p < total_params; ++p) {
                params_plus[p] += SPSA_PERTURBATION * delta[p];
                params_minus[p] -= SPSA_PERTURBATION * delta[p];
            }

            // Batch-summed SPSA gradient
            float spsa_grad_numerator = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                float exp_plus = run_vqc_circuit(
                    data_ptr, params_plus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                    num_qubits, num_layers, num_params_per_gate,
                    num_entanglers, num_measurements, b
                );
                float exp_minus = run_vqc_circuit(
                    data_ptr, params_minus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                    num_qubits, num_layers, num_params_per_gate,
                    num_entanglers, num_measurements, b
                );
                spsa_grad_numerator += upstream_grad(b) * (exp_plus - exp_minus);
            }

            // Apply SPSA gradients to all params
            for (int p = 0; p < total_params; ++p) {
                grad_params_flat(p) = spsa_grad_numerator / (2.0f * SPSA_PERTURBATION * delta[p]);
            }

            // Override with exact PSR for a subset of params (round-robin selection)
            // In production, would select based on gradient variance history
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < num_exact; ++i) {
                int p = (i * (total_params / num_exact)) % total_params;

                std::vector<float> p_plus = params_buffer;
                std::vector<float> p_minus = params_buffer;
                p_plus[p] += PI_HALF;
                p_minus[p] -= PI_HALF;

                float grad_sum = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    float exp_plus = run_vqc_circuit(
                        data_ptr, p_plus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                        num_qubits, num_layers, num_params_per_gate,
                        num_entanglers, num_measurements, b
                    );
                    float exp_minus = run_vqc_circuit(
                        data_ptr, p_minus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                        num_qubits, num_layers, num_params_per_gate,
                        num_entanglers, num_measurements, b
                    );
                    grad_sum += upstream_grad(b) * (exp_plus - exp_minus) / 2.0f;
                }
                grad_params_flat(p) = grad_sum;
            }

        } else {  // PURE_SPSA
            // Pure SPSA: O(2) circuit evaluations total (fastest, approximate)

            std::vector<float> delta = generate_bernoulli_perturbation(total_params);

            std::vector<float> params_plus = params_buffer;
            std::vector<float> params_minus = params_buffer;
            for (int p = 0; p < total_params; ++p) {
                params_plus[p] += SPSA_PERTURBATION * delta[p];
                params_minus[p] -= SPSA_PERTURBATION * delta[p];
            }

            for (int b = 0; b < batch_size; ++b) {
                float exp_plus = run_vqc_circuit(
                    data_ptr, params_plus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                    num_qubits, num_layers, num_params_per_gate,
                    num_entanglers, num_measurements, b
                );
                float exp_minus = run_vqc_circuit(
                    data_ptr, params_minus.data(), entanglers_ptr, paulis_ptr, coeffs_ptr,
                    num_qubits, num_layers, num_params_per_gate,
                    num_entanglers, num_measurements, b
                );

                float grad_factor = upstream_grad(b) * (exp_plus - exp_minus) / (2.0f * SPSA_PERTURBATION);

                for (int p = 0; p < total_params; ++p) {
                    grad_params_flat(p) += grad_factor / delta[p];
                }
            }
        }

        // ===================================================================
        // Gradient w.r.t. data angles (always use PSR - usually few params)
        // ===================================================================

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int b = 0; b < batch_size; ++b) {
            for (int q = 0; q < num_qubits; ++q) {
                // Create shifted data angles
                std::vector<float> data_plus(data_ptr, data_ptr + batch_size * num_qubits);
                std::vector<float> data_minus(data_ptr, data_ptr + batch_size * num_qubits);

                data_plus[b * num_qubits + q] += PI_HALF;
                data_minus[b * num_qubits + q] -= PI_HALF;

                float exp_plus = run_vqc_circuit(
                    data_plus.data(), params_ptr, entanglers_ptr, paulis_ptr, coeffs_ptr,
                    num_qubits, num_layers, num_params_per_gate,
                    num_entanglers, num_measurements, b
                );
                float exp_minus = run_vqc_circuit(
                    data_minus.data(), params_ptr, entanglers_ptr, paulis_ptr, coeffs_ptr,
                    num_qubits, num_layers, num_params_per_gate,
                    num_entanglers, num_measurements, b
                );

                float data_grad = upstream_grad(b) * (exp_plus - exp_minus) / 2.0f;
                grad_data_flat(b * num_qubits + q) = data_grad;
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("RunVqcExpectationGrad").Device(DEVICE_CPU), RunVqcExpectationGradOp);

}  // namespace tensorflow
