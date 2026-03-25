// saguaro.native/ops/mps_evolution_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <cmath>
#include <algorithm>

#include "mps_evolution_op.h"

namespace tensorflow {

REGISTER_OP("MPSTrotterStep")
    .Input("mps_tensors: N * float32")
    .Input("gate_sites: int32")
    .Input("gates_real: M * float32")
    .Input("gates_imag: M * float32")
    .Input("max_bond_dim: int32")
    .Input("truncation_threshold: float32")
    .Attr("N: int >= 1")
    .Attr("M: int >= 1")
    .Output("out_mps_tensors: N * float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int N;
        TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
        for (int i = 0; i < N; ++i) {
            c->set_output(i, c->input(i));
        }
        return absl::OkStatus();
    });

namespace {

// Helper: SVD truncation (copied/adapted from mps_contract_op.cc for consistency)
void TruncateSVD(Eigen::MatrixXf& U, Eigen::VectorXf& S, Eigen::MatrixXf& Vt,
                 int max_bond_dim, float truncation_threshold) {
    const int full_rank = S.size();
    float total_weight = 0.0f;
    for (int i = 0; i < full_rank; ++i) {
        total_weight += S(i) * S(i);
    }

    int trunc_rank = max_bond_dim;
    float cumulative_weight = 0.0f;
    for (int i = 0; i < full_rank && i < max_bond_dim; ++i) {
        cumulative_weight += S(i) * S(i);
        if ((total_weight - cumulative_weight) / total_weight < truncation_threshold) {
            trunc_rank = i + 1;
            break;
        }
    }
    trunc_rank = std::min(trunc_rank, static_cast<int>(std::min(U.cols(), Vt.rows())));

    U.conservativeResize(Eigen::NoChange, trunc_rank);
    S.conservativeResize(trunc_rank);
    Vt.conservativeResize(trunc_rank, Eigen::NoChange);
}

} // anonymous namespace

class MPSTrotterStepOpCpu : public OpKernel {
public:
    explicit MPSTrotterStepOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_sites_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &num_gates_));
    }

    void Compute(OpKernelContext* ctx) override {
        OpInputList mps_tensors;
        OP_REQUIRES_OK(ctx, ctx->input_list("mps_tensors", &mps_tensors));
        
        const Tensor& gate_sites_tensor = ctx->input(num_sites_);
        auto gate_sites = gate_sites_tensor.flat<int32>();
        
        OpInputList gates_real;
        OP_REQUIRES_OK(ctx, ctx->input_list("gates_real", &gates_real));
        OpInputList gates_imag;
        OP_REQUIRES_OK(ctx, ctx->input_list("gates_imag", &gates_imag));
        
        const int max_bond_dim = ctx->input(num_sites_ + 1 + 2 * num_gates_).scalar<int32>()();
        const float threshold = ctx->input(num_sites_ + 1 + 2 * num_gates_ + 1).scalar<float>()();

        // Detect batching
        bool is_batched = mps_tensors[0].dims() == 4;
        const int batch_size = is_batched ? mps_tensors[0].dim_size(0) : 1;

        // Allocate output tensors
        OpOutputList out_mps_tensors;
        OP_REQUIRES_OK(ctx, ctx->output_list("out_mps_tensors", &out_mps_tensors));

        for (int i = 0; i < num_sites_; ++i) {
            Tensor* out_tensor = nullptr;
            OP_REQUIRES_OK(ctx, out_mps_tensors.allocate(i, mps_tensors[i].shape(), &out_tensor));
        }

        // Process each batch element
        for (int b = 0; b < batch_size; ++b) {
            std::vector<Eigen::Tensor<float, 3, Eigen::RowMajor>> current_cores;
            for (int i = 0; i < num_sites_; ++i) {
                const Tensor& core_tensor = mps_tensors[i];
                if (is_batched) {
                    const int chi_L = core_tensor.dim_size(1);
                    const int d = core_tensor.dim_size(2);
                    const int chi_R = core_tensor.dim_size(3);
                    // Create a 3D view of the batch element
                    current_cores.push_back(Eigen::TensorMap<const Eigen::Tensor<float, 3, Eigen::RowMajor>>(
                        core_tensor.flat<float>().data() + b * chi_L * d * chi_R, chi_L, d, chi_R));
                } else {
                    current_cores.push_back(core_tensor.tensor<float, 3>());
                }
            }

            // Apply gates
            for (int g = 0; g < num_gates_; ++g) {
                int site_i = gate_sites(g);
                OP_REQUIRES(ctx, site_i >= 0 && site_i < num_sites_ - 1,
                            errors::InvalidArgument("Gate site index out of bounds"));

                const auto& g_real_tensor = gates_real[g].tensor<float, 2>();
                Eigen::MatrixXf U_real = Eigen::Map<const Eigen::MatrixXf>(
                    g_real_tensor.data(), g_real_tensor.dimension(0), g_real_tensor.dimension(1));

                ApplyTwoSiteGate(current_cores[site_i], current_cores[site_i + 1], U_real, 
                                 max_bond_dim, threshold,
                                 current_cores[site_i], current_cores[site_i + 1]);
            }

            // Copy results back to output tensors
            for (int i = 0; i < num_sites_; ++i) {
                Tensor* out_tensor = out_mps_tensors[i];
                const auto& core = current_cores[i];
                const int core_size = core.size();
                float* out_data = out_tensor->flat<float>().data() + b * core_size;
                std::copy(core.data(), core.data() + core_size, out_data);
            }
        }
    }

private:
    int num_sites_;
    int num_gates_;

    void ApplyTwoSiteGate(
        Eigen::Tensor<float, 3, Eigen::RowMajor>& core_i,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& core_next,
        const Eigen::MatrixXf& gate_real,
        int max_bond_dim,
        float threshold,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& out_core_i,
        Eigen::Tensor<float, 3, Eigen::RowMajor>& out_core_next) {
        
        const int chi_L = core_i.dimension(0);
        const int d1 = core_i.dimension(1);
        const int chi_M = core_i.dimension(2);
        const int d2 = core_next.dimension(1);
        const int chi_R = core_next.dimension(2);

        // Contract core_i and core_next over chi_M
        // core_i: [chi_L, d1, chi_M]
        // core_next: [chi_M, d2, chi_R]
        // result: [chi_L, d1, d2, chi_R]
        
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(2, 0) };
        Eigen::Tensor<float, 4, Eigen::RowMajor> theta = core_i.contract(core_next, product_dims);
        
        // Reshape theta to [chi_L, d1*d2, chi_R]
        // Wait, indices are (chi_L, d1, d2, chi_R).
        // Gate acts on (d1, d2).
        
        // Reshape to matrix for gate application: [chi_L * chi_R, d1 * d2]
        // Actually, gate applies to d1, d2.
        // Let's reshape to [chi_L, d1*d2, chi_R]
        Eigen::Tensor<float, 3, Eigen::RowMajor> theta_reshaped = theta.reshape(Eigen::array<int, 3>{chi_L, d1 * d2, chi_R});
        
        // Apply gate: theta' = door_operator * theta
        // Matrix multiplication over the d1*d2 dimension.
        // result_i,j,k = sum_m Gate_j,m * theta_i,m,k
        
        Eigen::Tensor<float, 3, Eigen::RowMajor> theta_prime(chi_L, d1 * d2, chi_R);
        for(int l=0; l<chi_L; ++l) {
            for(int r=0; r<chi_R; ++r) {
                Eigen::VectorXf vec(d1*d2);
                for(int m=0; m<d1*d2; ++m) vec(m) = theta_reshaped(l, m, r);
                Eigen::VectorXf vec_prime = gate_real * vec;
                for(int m=0; m<d1*d2; ++m) theta_prime(l, m, r) = vec_prime(m);
            }
        }
        
        // Reshape theta_prime for SVD: [chi_L * d1, d2 * chi_R]
        // Wait, indices of theta_prime are (chi_L, d1*d2, chi_R).
        // We need to split it as (chi_L * d1, d2 * chi_R).
        // Original indices: (chi_L, d1, d2, chi_R)
        Eigen::Tensor<float, 4, Eigen::RowMajor> theta_prime_full = theta_prime.reshape(Eigen::array<int, 4>{chi_L, d1, d2, chi_R});
        
        // Permute to (chi_L, d1, d2, chi_R) -> (chi_L, d1, d2, chi_R) - no change needed, just reshape.
        // Matrix for SVD: [chi_L * d1, d2 * chi_R]
        Eigen::MatrixXf mat_svd(chi_L * d1, d2 * chi_R);
        for(int l=0; l<chi_L; ++l) {
            for(int p1=0; p1<d1; ++p1) {
                for(int p2=0; p2<d2; ++p2) {
                    for(int r=0; r<chi_R; ++r) {
                        mat_svd(l * d1 + p1, p2 * chi_R + r) = theta_prime_full(l, p1, p2, r);
                    }
                }
            }
        }
        
        // Perform SVD
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(mat_svd, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::VectorXf S = svd.singularValues();
        Eigen::MatrixXf Vt = svd.matrixV().transpose();
        
        // Truncate
        TruncateSVD(U, S, Vt, max_bond_dim, threshold);
        
        const int new_chi_M = S.size();
        
        // Reconstruct cores
        // out_core_i: [chi_L, d1, new_chi_M]
        out_core_i = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>(U.data(), chi_L, d1, new_chi_M);
        
        // out_core_next: [new_chi_M, d2, chi_R]
        // Combine S and Vt
        Eigen::MatrixXf SVt = S.asDiagonal() * Vt;
        out_core_next = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>(SVt.data(), new_chi_M, d2, chi_R);
    }
};

REGISTER_KERNEL_BUILDER(Name("MPSTrotterStep").Device(DEVICE_CPU), MPSTrotterStepOpCpu);

} // namespace tensorflow
