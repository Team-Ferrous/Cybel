// src/ops/mps_contract_op.cc
// Matrix Product State (MPS) contraction operator for efficient quantum state representation
//
// Implements tensor network contractions with SVD truncation to maintain bond dimension,
// supporting both wavefunction contraction and expectation value computation.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <cmath>

// Conditional SIMD includes (Phase 11 compliance)
#if defined(__AVX512F__)
#include <immintrin.h>  // AVX512 intrinsics
#elif defined(__AVX2__)
#include <immintrin.h>  // AVX2 intrinsics
#elif defined(__ARM_NEON)
#include <arm_neon.h>   // ARM NEON intrinsics
#endif

#include "common/parallel/parallel_backend.h"

namespace tensorflow {
namespace {

// Helper: Compute entanglement entropy from Schmidt singular values
// Phase 11: Added explicit SIMD guards for AVX512/AVX2/NEON
float ComputeEntanglementEntropy(const std::vector<float>& singular_values) {
    const int64_t size = singular_values.size();
    const float* sv_data = singular_values.data();
    float entropy = 0.0f;
    float sum_sq = 0.0f;
    int64_t i = 0;

    // Compute sum of squares (normalize)
#if defined(__AVX512F__)
    __m512 v_sum_sq = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v_sv = _mm512_loadu_ps(sv_data + i);
        v_sum_sq = _mm512_fmadd_ps(v_sv, v_sv, v_sum_sq);
    }
    sum_sq = _mm512_reduce_add_ps(v_sum_sq);
#elif defined(__AVX2__)
    __m256 v_sum_sq = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v_sv = _mm256_loadu_ps(sv_data + i);
        v_sum_sq = _mm256_fmadd_ps(v_sv, v_sv, v_sum_sq);
    }
    // Horizontal sum reduction for AVX2
    __m128 vlow = _mm256_castps256_ps128(v_sum_sq);
    __m128 vhigh = _mm256_extractf128_ps(v_sum_sq, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    sum_sq = _mm_cvtss_f32(sums);
#elif defined(__ARM_NEON)
    float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v_sv = vld1q_f32(sv_data + i);
        v_sum_sq = vmlaq_f32(v_sum_sq, v_sv, v_sv);
    }
    // Horizontal sum reduction for NEON
    float32x2_t sum_pair = vadd_f32(vget_low_f32(v_sum_sq), vget_high_f32(v_sum_sq));
    sum_pair = vpadd_f32(sum_pair, sum_pair);
    sum_sq = vget_lane_f32(sum_pair, 0);
#endif

    // Scalar remainder for sum_sq
    for (; i < size; ++i) {
        sum_sq += sv_data[i] * sv_data[i];
    }

    if (sum_sq < 1e-12f) return 0.0f;

    // Compute entropy: -sum(p * log(p)) where p = sv^2 / sum_sq
    // Note: log(p) requires scalar fallback (no fast vectorized log in intrinsics)
    for (i = 0; i < size; ++i) {
        float prob = (sv_data[i] * sv_data[i]) / sum_sq;
        if (prob > 1e-12f) {
            entropy -= prob * std::log(prob);
        }
    }

    return entropy;
}

// Helper: SVD truncation to maintain bond dimension
// Phase 11: Added explicit SIMD guards for cumulative weight computation
void TruncateSVD(Eigen::MatrixXf& U, Eigen::VectorXf& S, Eigen::MatrixXf& Vt,
                 int max_bond_dim, float truncation_threshold,
                 float* entropy_out = nullptr) {
    const int full_rank = S.size();
    const float* s_data = S.data();

    // Compute total weight (sum of squared singular values) with SIMD
    float total_weight = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 v_total = _mm512_setzero_ps();
    for (; i + 16 <= full_rank; i += 16) {
        __m512 v_s = _mm512_loadu_ps(s_data + i);
        v_total = _mm512_fmadd_ps(v_s, v_s, v_total);
    }
    total_weight = _mm512_reduce_add_ps(v_total);
#elif defined(__AVX2__)
    __m256 v_total = _mm256_setzero_ps();
    for (; i + 8 <= full_rank; i += 8) {
        __m256 v_s = _mm256_loadu_ps(s_data + i);
        v_total = _mm256_fmadd_ps(v_s, v_s, v_total);
    }
    __m128 vlow = _mm256_castps256_ps128(v_total);
    __m128 vhigh = _mm256_extractf128_ps(v_total, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    total_weight = _mm_cvtss_f32(sums);
#elif defined(__ARM_NEON)
    float32x4_t v_total = vdupq_n_f32(0.0f);
    for (; i + 4 <= full_rank; i += 4) {
        float32x4_t v_s = vld1q_f32(s_data + i);
        v_total = vmlaq_f32(v_total, v_s, v_s);
    }
    float32x2_t sum_pair = vadd_f32(vget_low_f32(v_total), vget_high_f32(v_total));
    sum_pair = vpadd_f32(sum_pair, sum_pair);
    total_weight = vget_lane_f32(sum_pair, 0);
#endif

    // Scalar remainder
    for (; i < full_rank; ++i) {
        total_weight += s_data[i] * s_data[i];
    }

    // Determine truncation rank (sequential, data-dependent)
    int trunc_rank = max_bond_dim;
    float cumulative_weight = 0.0f;

    for (i = 0; i < full_rank && i < max_bond_dim; ++i) {
        cumulative_weight += s_data[i] * s_data[i];
        if ((total_weight - cumulative_weight) / total_weight < truncation_threshold) {
            trunc_rank = i + 1;
            break;
        }
    }

    trunc_rank = std::min(trunc_rank, static_cast<int>(std::min(U.cols(), Vt.rows())));

    // Compute entanglement entropy before truncation
    if (entropy_out != nullptr) {
        std::vector<float> sv_vec(S.data(), S.data() + trunc_rank);
        *entropy_out = ComputeEntanglementEntropy(sv_vec);
    }

    // Truncate matrices
    U.conservativeResize(Eigen::NoChange, trunc_rank);
    S.conservativeResize(trunc_rank);
    Vt.conservativeResize(trunc_rank, Eigen::NoChange);
}

}  // anonymous namespace

// ============================================================================
// Op Registration
// ============================================================================

REGISTER_OP("MPSContract")
    .Input("mps_tensors: N * float32")
    .Input("physical_dims: int32")
    .Input("bond_dims: int32")
    .Input("max_bond_dim: int32")
    .Attr("N: int >= 1")
    .Attr("compute_entropy: bool = false")
    .Attr("truncation_threshold: float = 1e-10")
    .Attr("uniform: bool = false")
    .Output("contracted_state: float32")
    .Output("entanglement_entropies: float32")
    .Output("log_norm: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // contracted_state: [2^N] for N sites
        c->set_output(0, c->UnknownShapeOfRank(1));
        // entanglement_entropies: [N-1] Schmidt cuts
        c->set_output(1, c->UnknownShapeOfRank(1));
        // log_norm: scalar
        c->set_output(2, c->Scalar());
        return absl::OkStatus();
    });

REGISTER_OP("MPSCanonicalize")
    .Input("mps_tensors: N * float32")
    .Input("physical_dims: int32")
    .Input("bond_dims: int32")
    .Input("center_site: int32")
    .Attr("N: int >= 1")
    .Output("canonical_tensors: N * float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int N;
        TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
        for (int i = 0; i < N; ++i) {
            c->set_output(i, c->UnknownShapeOfRank(3));
        }
        return absl::OkStatus();
    });

REGISTER_OP("MPSContractGrad")
    .Input("grad_contracted: float32")
    .Input("mps_tensors: N * float32")
    .Input("physical_dims: int32")
    .Input("bond_dims: int32")
    .Input("max_bond_dim: int32")
    .Attr("N: int >= 1")
    .Attr("uniform: bool = false")
    .Attr("use_tdvp: bool = false")
    .Output("grad_mps_tensors: N * float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int N;
        TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
        for (int i = 0; i < N; ++i) {
            c->set_output(i, c->input(i + 1));
        }
        return absl::OkStatus();
    });

REGISTER_OP("MPSExpect")
    .Input("mps_tensors: N * float32")
    .Input("operator: float32")
    .Input("physical_dims: int32")
    .Input("bond_dims: int32")
    .Input("max_bond_dim: int32")
    .Attr("N: int >= 1")
    .Output("expectation: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return absl::OkStatus();
    });

REGISTER_OP("MPSExpectPauli")
    .Input("mps_tensors: N * float32")
    .Input("pauli_indices: int32")
    .Input("coefficients: float32")
    .Attr("N: int >= 1")
    .Output("expectation: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output is [batch_size] if batched, or scalar if not.
        // Take batch from first MPS tensor if rank is 4
        shape_inference::ShapeHandle mps0 = c->input(0);
        if (c->Rank(mps0) == 4) {
            c->set_output(0, c->Vector(c->Dim(mps0, 0)));
        } else {
            c->set_output(0, c->Scalar());
        }
        return absl::OkStatus();
    });

REGISTER_OP("MPSFeatureImportance")
    .Input("entanglement_entropies: float32")
    .Output("feature_importance: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input = c->input(0);
        if (c->Rank(input) == 2) {
            // [batch, num_bonds] -> [batch, num_bonds + 1]
            shape_inference::DimensionHandle num_sites = c->DebugString(c->Dim(input, 1)) == "unknown" ? 
                c->UnknownDim() : c->MakeDim(c->Value(c->Dim(input, 1)) + 1);
            c->set_output(0, c->Matrix(c->Dim(input, 0), num_sites));
        } else {
            // [num_bonds] -> [num_bonds + 1]
            shape_inference::DimensionHandle num_sites = c->DebugString(c->Dim(input, 0)) == "unknown" ? 
                c->UnknownDim() : c->MakeDim(c->Value(c->Dim(input, 0)) + 1);
            c->set_output(0, c->Vector(num_sites));
        }
        return absl::OkStatus();
    });

// ============================================================================
// CPU Kernel
// ============================================================================

class MPSContractOpCpu : public OpKernel {
public:
    explicit MPSContractOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_sites_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("compute_entropy", &compute_entropy_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("truncation_threshold", &truncation_threshold_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("uniform", &uniform_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Input validation
        OpInputList mps_tensors;
        OP_REQUIRES_OK(ctx, ctx->input_list("mps_tensors", &mps_tensors));
        OP_REQUIRES(ctx, mps_tensors.size() == num_sites_,
                    errors::InvalidArgument("Expected ", num_sites_, " MPS tensors, got ",
                                           mps_tensors.size()));

        const Tensor& physical_dims_tensor = ctx->input(num_sites_);
        const Tensor& bond_dims_tensor = ctx->input(num_sites_ + 1);
        const Tensor& max_bond_dim_tensor = ctx->input(num_sites_ + 2);

        OP_REQUIRES(ctx, physical_dims_tensor.NumElements() == num_sites_,
                    errors::InvalidArgument("physical_dims must have ", num_sites_, " elements"));
        OP_REQUIRES(ctx, bond_dims_tensor.NumElements() == num_sites_ + 1,
                    errors::InvalidArgument("bond_dims must have ", num_sites_ + 1, " elements"));

        auto physical_dims = physical_dims_tensor.flat<int32>();
        auto bond_dims = bond_dims_tensor.flat<int32>();
        int max_bond_dim = max_bond_dim_tensor.scalar<int32>()();

        // Validate MPS tensor shapes: [bond_left, physical, bond_right]
        for (int i = 0; i < num_sites_; ++i) {
            const auto& tensor = mps_tensors[i];
            OP_REQUIRES(ctx, tensor.dims() == 3,
                        errors::InvalidArgument("MPS tensor ", i, " must be rank-3, got rank ",
                                               tensor.dims()));
            OP_REQUIRES(ctx, tensor.dim_size(0) == bond_dims(i) &&
                            tensor.dim_size(1) == physical_dims(i) &&
                            tensor.dim_size(2) == bond_dims(i + 1),
                        errors::InvalidArgument(
                            "MPS tensor ", i, " shape mismatch: expected [",
                            bond_dims(i), ",", physical_dims(i), ",", bond_dims(i + 1),
                            "], got [", tensor.dim_size(0), ",", tensor.dim_size(1), ",",
                            tensor.dim_size(2), "]"));
        }

        // Perform sequential contraction
        std::vector<float> entropies;
        float log_norm = 0.0f;
        ContractMPS(ctx, mps_tensors, physical_dims, bond_dims, max_bond_dim, entropies, log_norm);

        // Output log_norm
        Tensor* log_norm_output;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &log_norm_output));
        log_norm_output->scalar<float>()() = log_norm;
    }

private:
    void ContractMPS(OpKernelContext* ctx, const OpInputList& mps_tensors,
                     const TTypes<int32>::ConstFlat& physical_dims,
                     const TTypes<int32>::ConstFlat& bond_dims,
                     int max_bond_dim,
                     std::vector<float>& entropies,
                     float& log_norm) {
        // Start with the first tensor
        Eigen::Tensor<float, 3, Eigen::RowMajor> state =
            mps_tensors[0].tensor<float, 3>();

        // Contract sequentially: state * mps[i] for i = 1..N-1
        for (int site = 1; site < num_sites_; ++site) {
            const int source_site = uniform_ ? 0 : site;
            Eigen::Tensor<float, 3, Eigen::RowMajor> next_tensor =
                mps_tensors[source_site].tensor<float, 3>();

            // Contract over shared bond dimension
            // state: [bond_left, phys_accumulated, bond_middle]
            // next: [bond_middle, phys_site, bond_right]
            // result: [bond_left, phys_accumulated * phys_site, bond_right]

            const int bond_left = state.dimension(0);
            const int phys_acc = state.dimension(1);
            const int bond_middle = state.dimension(2);
            const int phys_site = next_tensor.dimension(1);
            const int bond_right = next_tensor.dimension(2);

            // Reshape and perform matrix multiplication
            // CRITICAL: Use RowMajor matrices to match tensor memory layout
            using MatrixRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            MatrixRowMajor state_mat = Eigen::Map<MatrixRowMajor>(
                state.data(), bond_left * phys_acc, bond_middle);
            MatrixRowMajor next_mat = Eigen::Map<MatrixRowMajor>(
                next_tensor.data(), bond_middle, phys_site * bond_right);

            MatrixRowMajor product = state_mat * next_mat;

            // Normalize to prevent overflow and accumulate log-norm
            float norm = product.norm();
            if (norm > 1e-12f) {
                product /= norm;
                log_norm += std::log(norm);
            }

            // Compute entropy and/or truncate bond dimension
            bool needs_truncation = (bond_left * phys_acc * phys_site * bond_right >
                                     max_bond_dim * max_bond_dim);
            bool needs_svd = compute_entropy_ || needs_truncation;

            if (needs_svd) {
                Eigen::JacobiSVD<Eigen::MatrixXf> svd(
                    product, Eigen::ComputeThinU | Eigen::ComputeThinV);

                Eigen::MatrixXf U = svd.matrixU();
                Eigen::VectorXf S = svd.singularValues();
                Eigen::MatrixXf Vt = svd.matrixV().transpose();

                // Always compute entropy when requested, even without truncation
                if (compute_entropy_) {
                    std::vector<float> sv_vec(S.data(), S.data() + S.size());
                    float entropy = ComputeEntanglementEntropy(sv_vec);
                    entropies.push_back(entropy);
                }

                // Only truncate and reconstruct if needed
                if (needs_truncation) {
                    TruncateSVD(U, S, Vt, max_bond_dim, truncation_threshold_, nullptr);

                    // Reconstruct truncated product
                    Eigen::MatrixXf S_sqrt_diag = S.asDiagonal();
                    product = U * S_sqrt_diag * Vt;
                }
            }

            // Reshape back to tensor format (product is already RowMajor)
            state = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>(
                product.data(), bond_left, phys_acc * phys_site, bond_right);
        }

        // Allocate output
        const int total_phys_dim = state.dimension(1);
        Tensor* output;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({total_phys_dim}), &output));

        // Squeeze bond dimensions (should be 1 at boundaries)
        auto output_flat = output->flat<float>();
        for (int i = 0; i < total_phys_dim; ++i) {
            output_flat(i) = state(0, i, 0);
        }

        // Output entanglement entropies
        Tensor* entropy_output;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            1, TensorShape({static_cast<int64_t>(entropies.size())}), &entropy_output));
        auto entropy_flat = entropy_output->flat<float>();
        for (size_t i = 0; i < entropies.size(); ++i) {
            entropy_flat(i) = entropies[i];
        }
    }

    int num_sites_;
    bool compute_entropy_;
    float truncation_threshold_;
    bool uniform_;
};

REGISTER_KERNEL_BUILDER(Name("MPSContract").Device(DEVICE_CPU), MPSContractOpCpu);

// ============================================================================
// Gradient Kernel - DMRG Adjoint Algorithm
// ============================================================================

class MPSContractGradOpCpu : public OpKernel {
public:
    explicit MPSContractGradOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_sites_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("uniform", &uniform_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_tdvp", &use_tdvp_));
    }
private:
    int num_sites_;
    bool uniform_;
    bool use_tdvp_;
public:

    void Compute(OpKernelContext* ctx) override {
        // DMRG adjoint gradient implementation
        // Implements backward pass through MPS contraction using environment tensors
        // Reference: White & Chan (2004), Vidal (2004) TEBD

        const Tensor& grad_output = ctx->input(0);  // Gradient w.r.t. contracted state

        OpInputList mps_tensors;
        OP_REQUIRES_OK(ctx, ctx->input_list("mps_tensors", &mps_tensors));

        const Tensor& physical_dims_tensor = ctx->input(num_sites_ + 1);
        const Tensor& bond_dims_tensor = ctx->input(num_sites_ + 2);
        const Tensor& max_bond_dim_tensor = ctx->input(num_sites_ + 3);  // Read max_bond_dim

        auto physical_dims = physical_dims_tensor.flat<int32>();
        auto bond_dims = bond_dims_tensor.flat<int32>();
        // max_bond_dim used to be here but was unused.
        auto grad_flat = grad_output.flat<float>();

        // Compute environment tensors for gradient propagation
        std::vector<Eigen::MatrixXf> left_envs;
        std::vector<Eigen::MatrixXf> right_envs;

        ComputeEnvironments(ctx, mps_tensors, physical_dims, bond_dims,
                          left_envs, right_envs);

        // Compute gradients for each MPS tensor using environments
        std::vector<Tensor*> grad_tensors(num_sites_);
        for (int site = 0; site < num_sites_; ++site) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(site, mps_tensors[site].shape(), &grad_tensors[site]));
            grad_tensors[site]->tensor<float, 3>().setZero();
        }

        if (uniform_) {
            // In uniform mode, we accumulate gradients from all sites into the first output
            // and then replicate it to all N outputs (since they refer to the same parameters)
            Tensor* shared_grad = grad_tensors[0];
            for (int site = 0; site < num_sites_; ++site) {
                const int phys = physical_dims(site);
                const int bond_left = bond_dims(site);
                const int bond_right = bond_dims(site + 1);

                // Create a temporary tensor to hold the site gradient
                Tensor site_grad_tensor(DT_FLOAT, mps_tensors[site].shape());
                site_grad_tensor.tensor<float, 3>().setZero();

                ComputeSiteGradient(site, grad_flat, mps_tensors,
                                  left_envs, right_envs,
                                  bond_left, phys, bond_right,
                                  physical_dims, bond_dims,
                                  &site_grad_tensor);
                
                // Add to shared_grad
                auto shared_flat = shared_grad->flat<float>();
                auto site_flat = site_grad_tensor.flat<float>();
                for (int i = 0; i < shared_flat.size(); ++i) {
                    shared_flat(i) += site_flat(i);
                }
            }

            // Copy shared_grad to all other outputs
            for (int site = 1; site < num_sites_; ++site) {
                auto shared_flat = shared_grad->flat<float>();
                auto target_flat = grad_tensors[site]->flat<float>();
                std::copy(shared_flat.data(), shared_flat.data() + shared_flat.size(), target_flat.data());
            }
        } else {
            // Standard mode: one gradient per site
            for (int site = 0; site < num_sites_; ++site) {
                const int bond_left = bond_dims(site);
                const int phys = physical_dims(site);
                const int bond_right = bond_dims(site + 1);

                ComputeSiteGradient(site, grad_flat, mps_tensors,
                                  left_envs, right_envs,
                                  bond_left, phys, bond_right,
                                  physical_dims, bond_dims,
                                  grad_tensors[site]);
            }
        }

        // Apply TDVP projection if requested
        if (use_tdvp_) {
            ApplyTDVPProjection(ctx, grad_tensors, mps_tensors, physical_dims, bond_dims);
        }
    }

private:
    void ComputeEnvironments(OpKernelContext* ctx,
                            const OpInputList& mps_tensors,
                            const TTypes<int32>::ConstFlat& physical_dims,
                            const TTypes<int32>::ConstFlat& bond_dims,
                            std::vector<Eigen::MatrixXf>& left_envs,
                            std::vector<Eigen::MatrixXf>& right_envs) {
        // Build left environments: L[i] represents accumulated contraction from left boundary to site i-1
        // L[0] = identity (left boundary)
        // L[i] = L[i-1] contracted with MPS[i-1]
        left_envs.resize(num_sites_ + 1);

        // Left boundary: [1, bond_dims[0]]
        left_envs[0] = Eigen::MatrixXf::Zero(1, bond_dims(0));
        if (bond_dims(0) > 0) {
            left_envs[0](0, 0) = 1.0f;  // Initialize left boundary
        }

        // Forward sweep: accumulate left environments
        int phys_acc = 1;  // Accumulated physical dimension
        for (int i = 0; i < num_sites_; ++i) {
            const auto& mps_tensor = mps_tensors[i].tensor<float, 3>();
            const int bond_left = bond_dims(i);
            const int phys = physical_dims(i);
            const int bond_right = bond_dims(i + 1);

            // Next environment: [phys_acc * phys, bond_right]
            Eigen::MatrixXf next_left = Eigen::MatrixXf::Zero(phys_acc * phys, bond_right);

            // Contract: L[i] (shape [phys_acc, bond_left]) with MPS[i] (shape [bond_left, phys, bond_right])
            for (int pa = 0; pa < phys_acc; ++pa) {
                for (int p = 0; p < phys; ++p) {
                    for (int bl = 0; bl < bond_left; ++bl) {
                        for (int br = 0; br < bond_right; ++br) {
                            next_left(pa * phys + p, br) += left_envs[i](pa, bl) * mps_tensor(bl, p, br);
                        }
                    }
                }
            }

            left_envs[i + 1] = next_left;
            phys_acc *= phys;
        }

        // Build right environments: R[i] represents accumulated contraction from right boundary to site i+1
        // R[N] = identity (right boundary)
        // R[i] = MPS[i] contracted with R[i+1]
        right_envs.resize(num_sites_ + 1);

        // Right boundary: [bond_dims[N], 1]
        right_envs[num_sites_] = Eigen::MatrixXf::Zero(bond_dims(num_sites_), 1);
        if (bond_dims(num_sites_) > 0) {
            right_envs[num_sites_](0, 0) = 1.0f;  // Initialize right boundary
        }

        // Backward sweep: accumulate right environments
        phys_acc = 1;
        for (int i = num_sites_ - 1; i >= 0; --i) {
            const auto& mps_tensor = mps_tensors[i].tensor<float, 3>();
            const int bond_left = bond_dims(i);
            const int phys = physical_dims(i);
            const int bond_right = bond_dims(i + 1);

            // Next environment: [bond_left, phys * phys_acc]
            Eigen::MatrixXf next_right = Eigen::MatrixXf::Zero(bond_left, phys * phys_acc);

            // Contract: MPS[i] (shape [bond_left, phys, bond_right]) with R[i+1] (shape [bond_right, phys_acc])
            for (int bl = 0; bl < bond_left; ++bl) {
                for (int p = 0; p < phys; ++p) {
                    for (int br = 0; br < bond_right; ++br) {
                        for (int pa = 0; pa < phys_acc; ++pa) {
                            next_right(bl, p * phys_acc + pa) += mps_tensor(bl, p, br) * right_envs[i + 1](br, pa);
                        }
                    }
                }
            }

            right_envs[i] = next_right;
            phys_acc *= phys;
        }
    }

    void ComputeSiteGradient(int site,
                            const TTypes<float>::ConstFlat& grad_output,
                            const OpInputList& mps_tensors,
                            const std::vector<Eigen::MatrixXf>& left_envs,
                            const std::vector<Eigen::MatrixXf>& right_envs,
                            int bond_left, int phys, int bond_right,
                            const TTypes<int32>::ConstFlat& physical_dims,
                            const TTypes<int32>::ConstFlat& bond_dims,
                            Tensor* grad_tensor) {
        // Gradient for MPS tensor at site i:
        // ∂L/∂A[i]_{α,σ,β} = Σ_{left,right} L[i]_{left,α} * grad_ψ_{left,σ,right} * R[i+1]_{β,right}
        //
        // Key insight: Left environment accumulates physical indices to the LEFT of site i
        //              Right environment accumulates physical indices to the RIGHT of site i
        //              grad_output is indexed as: left_physical_configs, site_physical, right_physical_configs

        auto grad_3d = grad_tensor->tensor<float, 3>();
        grad_3d.setZero();

        // Get environment dimensions
        const int left_phys_acc = left_envs[site].rows();       // Number of left physical configurations
        const int right_phys_acc = right_envs[site + 1].cols(); // Number of right physical configurations

        // Compute gradient by contracting grad_output with environments
        // Phase 11: Added TBB parallelism for bond_left dimension
        // For each MPS element A[i]_{bl, p, br}, accumulate contributions from all configurations
        const int64_t total_work = static_cast<int64_t>(bond_left);
        const int64_t cost_per_unit = phys * bond_right * left_phys_acc * right_phys_acc;

        saguaro::parallel::ForShard(
            total_work, cost_per_unit,
            [&](int64_t bl_start, int64_t bl_end) {
                for (int bl = bl_start; bl < bl_end; ++bl) {
                    for (int p = 0; p < phys; ++p) {
                        for (int br = 0; br < bond_right; ++br) {
                            float grad_sum = 0.0f;

                            // Sum over all left and right physical configurations
                            for (int left_idx = 0; left_idx < left_phys_acc; ++left_idx) {
                                for (int right_idx = 0; right_idx < right_phys_acc; ++right_idx) {
                                    // Compute flat index in grad_output
                                    // Row-major order: left configs are slow, right configs are fast
                                    int flat_idx = left_idx * phys * right_phys_acc +
                                                  p * right_phys_acc + right_idx;

                                    if (flat_idx >= grad_output.size()) {
                                        continue;  // Skip if out of bounds
                                    }

                                    float grad_val = grad_output(flat_idx);

                                    // Environment contributions
                                    // left_envs[site]: shape [left_phys_acc, bond_left]
                                    // right_envs[site+1]: shape [bond_right, right_phys_acc]
                                    float left_val = left_envs[site](left_idx, bl);
                                    float right_val = right_envs[site + 1](br, right_idx);

                                    grad_sum += grad_val * left_val * right_val;
                                }
                            }

                            grad_3d(bl, p, br) = grad_sum;
                        }
                    }
                }
            }
        );
    }

    void ApplyTDVPProjection(OpKernelContext* ctx, 
                             std::vector<Tensor*>& grad_tensors,
                             const OpInputList& mps_tensors,
                             const TTypes<int32>::ConstFlat& physical_dims,
                             const TTypes<int32>::ConstFlat& bond_dims) {
        // Placeholder for TDVP tangent space projection.
        // In a true TDVP implementation, we would project each grad_tensor[i] 
        // onto the subspace orthogonal to the current A[i] to ensure 
        // the update stays on the MPS manifold.
        // This requires canonical form and specific gauge choices.
        // For now, we perform local normalization to stabilize gradients.
        for (int i = 0; i < num_sites_; ++i) {
            auto g_flat = grad_tensors[i]->flat<float>();
            float g_norm = 0.0f;
            for(int j=0; j<g_flat.size(); ++j) g_norm += g_flat(j) * g_flat(j);
            g_norm = std::sqrt(g_norm);
            if (g_norm > 1.0f) {
                for(int j=0; j<g_flat.size(); ++j) g_flat(j) /= g_norm;
            }
        }
    }

};

REGISTER_KERNEL_BUILDER(Name("MPSContractGrad").Device(DEVICE_CPU), MPSContractGradOpCpu);

// ============================================================================
// MPSCanonicalize Kernel
// ============================================================================

class MPSCanonicalizeOpCpu : public OpKernel {
public:
    explicit MPSCanonicalizeOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_sites_));
    }

    void Compute(OpKernelContext* ctx) override {
        OpInputList mps_tensors;
        OP_REQUIRES_OK(ctx, ctx->input_list("mps_tensors", &mps_tensors));
        OP_REQUIRES(ctx, mps_tensors.size() == num_sites_,
                    errors::InvalidArgument("Expected ", num_sites_, " MPS tensors, got ",
                                           mps_tensors.size()));

        const Tensor& physical_dims_tensor = ctx->input(num_sites_);
        const Tensor& bond_dims_tensor = ctx->input(num_sites_ + 1);
        const Tensor& center_site_tensor = ctx->input(num_sites_ + 2);

        OP_REQUIRES(ctx, physical_dims_tensor.NumElements() == num_sites_,
                    errors::InvalidArgument("physical_dims must have ", num_sites_, " elements"));
        OP_REQUIRES(ctx, bond_dims_tensor.NumElements() == num_sites_ + 1,
                    errors::InvalidArgument("bond_dims must have ", num_sites_ + 1, " elements"));

        const int center_site = center_site_tensor.scalar<int32>()();
        OP_REQUIRES(ctx, center_site >= 0 && center_site < num_sites_,
                    errors::InvalidArgument("center_site must be in [0, ", num_sites_ - 1,
                                           "], got ", center_site));

        auto physical_dims = physical_dims_tensor.flat<int32>();
        auto bond_dims = bond_dims_tensor.flat<int32>();

        // Validate MPS tensor shapes: [bond_left, physical, bond_right]
        for (int i = 0; i < num_sites_; ++i) {
            const auto& tensor = mps_tensors[i];
            OP_REQUIRES(ctx, tensor.dims() == 3,
                        errors::InvalidArgument("MPS tensor ", i, " must be rank-3, got rank ",
                                               tensor.dims()));
            OP_REQUIRES(ctx, tensor.dim_size(0) == bond_dims(i) &&
                            tensor.dim_size(1) == physical_dims(i) &&
                            tensor.dim_size(2) == bond_dims(i + 1),
                        errors::InvalidArgument(
                            "MPS tensor ", i, " shape mismatch: expected [",
                            bond_dims(i), ",", physical_dims(i), ",", bond_dims(i + 1),
                            "], got [", tensor.dim_size(0), ",", tensor.dim_size(1), ",",
                            tensor.dim_size(2), "]"));
        }

        using MatrixRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        std::vector<Eigen::Tensor<float, 3, Eigen::RowMajor>> working;
        working.reserve(num_sites_);
        for (int i = 0; i < num_sites_; ++i) {
            const int bond_left = mps_tensors[i].dim_size(0);
            const int phys = mps_tensors[i].dim_size(1);
            const int bond_right = mps_tensors[i].dim_size(2);
            Eigen::Tensor<float, 3, Eigen::RowMajor> core(bond_left, phys, bond_right);
            core = mps_tensors[i].tensor<float, 3>();
            working.push_back(core);
        }

        std::vector<Eigen::Tensor<float, 3, Eigen::RowMajor>> canonical;
        canonical.resize(num_sites_);

        // Left-orthogonalization sweep
        for (int i = 0; i < center_site; ++i) {
            auto& core = working[i];
            const int bond_left = core.dimension(0);
            const int phys = core.dimension(1);
            const int bond_right = core.dimension(2);
            const int m = bond_left * phys;
            const int n = bond_right;

            MatrixRowMajor core_mat_rm = Eigen::Map<MatrixRowMajor>(core.data(), m, n);
            Eigen::MatrixXf core_mat = core_mat_rm;
            const int k = std::min(m, n);

            Eigen::HouseholderQR<Eigen::MatrixXf> qr(core_mat);
            Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(m, k);
            Eigen::MatrixXf R = qr.matrixQR().topLeftCorner(k, n).template triangularView<Eigen::Upper>();

            Eigen::Tensor<float, 3, Eigen::RowMajor> q_tensor(bond_left, phys, k);
            Eigen::Map<MatrixRowMajor>(q_tensor.data(), m, k) = Q;
            canonical[i] = q_tensor;

            auto& next_core = working[i + 1];
            const int next_phys = next_core.dimension(1);
            const int next_bond_right = next_core.dimension(2);
            MatrixRowMajor next_mat = Eigen::Map<MatrixRowMajor>(
                next_core.data(), n, next_phys * next_bond_right);
            MatrixRowMajor r_rm = R;
            MatrixRowMajor result = r_rm * next_mat;

            Eigen::Tensor<float, 3, Eigen::RowMajor> updated_next(k, next_phys, next_bond_right);
            Eigen::Map<MatrixRowMajor>(updated_next.data(), k, next_phys * next_bond_right) = result;
            next_core = updated_next;
        }

        // Right-orthogonalization sweep
        for (int i = num_sites_ - 1; i > center_site; --i) {
            auto& core = working[i];
            const int bond_left = core.dimension(0);
            const int phys = core.dimension(1);
            const int bond_right = core.dimension(2);
            const int m = bond_left;
            const int n = phys * bond_right;

            MatrixRowMajor core_mat_rm = Eigen::Map<MatrixRowMajor>(core.data(), m, n);
            Eigen::MatrixXf core_mat_t = core_mat_rm.transpose();
            const int k = std::min(n, m);

            Eigen::HouseholderQR<Eigen::MatrixXf> qr(core_mat_t);
            Eigen::MatrixXf Q = qr.householderQ() * Eigen::MatrixXf::Identity(n, k);
            Eigen::MatrixXf R = qr.matrixQR().topLeftCorner(k, m).template triangularView<Eigen::Upper>();

            OP_REQUIRES(ctx, k == bond_left,
                        errors::InvalidArgument("Right canonicalization requires bond_left <= phys * bond_right; "
                                               "got bond_left=", bond_left, ", phys*bond_right=", n));

            Eigen::MatrixXf q_t = Q.transpose();
            Eigen::Tensor<float, 3, Eigen::RowMajor> q_tensor(bond_left, phys, bond_right);
            Eigen::Map<MatrixRowMajor>(q_tensor.data(), bond_left, n) = q_t;
            canonical[i] = q_tensor;

            if (i > 0) {
                auto& prev_core = working[i - 1];
                const int prev_bond_left = prev_core.dimension(0);
                const int prev_phys = prev_core.dimension(1);
                MatrixRowMajor prev_mat = Eigen::Map<MatrixRowMajor>(
                    prev_core.data(), prev_bond_left * prev_phys, bond_left);
                MatrixRowMajor r_rm = R;
                MatrixRowMajor result = prev_mat * r_rm;
                Eigen::Tensor<float, 3, Eigen::RowMajor> updated_prev(prev_bond_left, prev_phys, bond_left);
                Eigen::Map<MatrixRowMajor>(updated_prev.data(), prev_bond_left * prev_phys, bond_left) = result;
                prev_core = updated_prev;
            }
        }

        canonical[center_site] = working[center_site];

        OpOutputList out_tensors;
        OP_REQUIRES_OK(ctx, ctx->output_list("canonical_tensors", &out_tensors));
        for (int i = 0; i < num_sites_; ++i) {
            const auto& core = canonical[i];
            Tensor* out_tensor = nullptr;
            TensorShape shape({core.dimension(0), core.dimension(1), core.dimension(2)});
            OP_REQUIRES_OK(ctx, out_tensors.allocate(i, shape, &out_tensor));
            out_tensor->tensor<float, 3>() = core;
        }
    }

private:
    int num_sites_;
};

REGISTER_KERNEL_BUILDER(Name("MPSCanonicalize").Device(DEVICE_CPU), MPSCanonicalizeOpCpu);

// ============================================================================
// MPSExpect Kernel
// ============================================================================

class MPSExpectOpCpu : public OpKernel {
public:
    explicit MPSExpectOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_sites_));
    }

    void Compute(OpKernelContext* ctx) override {
        OpInputList mps_tensors;
        OP_REQUIRES_OK(ctx, ctx->input_list("mps_tensors", &mps_tensors));
        OP_REQUIRES(ctx, mps_tensors.size() == num_sites_,
                    errors::InvalidArgument("Expected ", num_sites_, " MPS tensors, got ",
                                           mps_tensors.size()));

        const Tensor& operator_tensor = ctx->input(num_sites_);
        const Tensor& physical_dims_tensor = ctx->input(num_sites_ + 1);
        const Tensor& bond_dims_tensor = ctx->input(num_sites_ + 2);
        const Tensor& max_bond_dim_tensor = ctx->input(num_sites_ + 3);

        OP_REQUIRES(ctx, operator_tensor.dims() == 2,
                    errors::InvalidArgument("operator must be rank-2, got rank ",
                                           operator_tensor.dims()));
        OP_REQUIRES(ctx, physical_dims_tensor.NumElements() == num_sites_,
                    errors::InvalidArgument("physical_dims must have ", num_sites_, " elements"));
        OP_REQUIRES(ctx, bond_dims_tensor.NumElements() == num_sites_ + 1,
                    errors::InvalidArgument("bond_dims must have ", num_sites_ + 1, " elements"));

        auto physical_dims = physical_dims_tensor.flat<int32>();
        auto bond_dims = bond_dims_tensor.flat<int32>();
        const int max_bond_dim = max_bond_dim_tensor.scalar<int32>()();

        // Validate MPS tensor shapes: [bond_left, physical, bond_right]
        for (int i = 0; i < num_sites_; ++i) {
            const auto& tensor = mps_tensors[i];
            OP_REQUIRES(ctx, tensor.dims() == 3,
                        errors::InvalidArgument("MPS tensor ", i, " must be rank-3, got rank ",
                                               tensor.dims()));
            OP_REQUIRES(ctx, tensor.dim_size(0) == bond_dims(i) &&
                            tensor.dim_size(1) == physical_dims(i) &&
                            tensor.dim_size(2) == bond_dims(i + 1),
                        errors::InvalidArgument(
                            "MPS tensor ", i, " shape mismatch: expected [",
                            bond_dims(i), ",", physical_dims(i), ",", bond_dims(i + 1),
                            "], got [", tensor.dim_size(0), ",", tensor.dim_size(1), ",",
                            tensor.dim_size(2), "]"));
        }

        const float truncation_threshold = 1e-10f;

        // Start with the first tensor
        Eigen::Tensor<float, 3, Eigen::RowMajor> state =
            mps_tensors[0].tensor<float, 3>();

        // Contract sequentially: state * mps[i] for i = 1..N-1
        for (int site = 1; site < num_sites_; ++site) {
            Eigen::Tensor<float, 3, Eigen::RowMajor> next_tensor =
                mps_tensors[site].tensor<float, 3>();

            const int bond_left = state.dimension(0);
            const int phys_acc = state.dimension(1);
            const int bond_middle = state.dimension(2);
            const int phys_site = next_tensor.dimension(1);
            const int bond_right = next_tensor.dimension(2);

            using MatrixRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

            MatrixRowMajor state_mat = Eigen::Map<MatrixRowMajor>(
                state.data(), bond_left * phys_acc, bond_middle);
            MatrixRowMajor next_mat = Eigen::Map<MatrixRowMajor>(
                next_tensor.data(), bond_middle, phys_site * bond_right);

            MatrixRowMajor product = state_mat * next_mat;

            float norm = product.norm();
            if (norm > 1e-12f) {
                product /= norm;
            }

            bool needs_truncation = (bond_left * phys_acc * phys_site * bond_right >
                                     max_bond_dim * max_bond_dim);
            if (needs_truncation) {
                Eigen::JacobiSVD<Eigen::MatrixXf> svd(
                    product, Eigen::ComputeThinU | Eigen::ComputeThinV);

                Eigen::MatrixXf U = svd.matrixU();
                Eigen::VectorXf S = svd.singularValues();
                Eigen::MatrixXf Vt = svd.matrixV().transpose();

                TruncateSVD(U, S, Vt, max_bond_dim, truncation_threshold, nullptr);
                Eigen::MatrixXf S_sqrt_diag = S.asDiagonal();
                product = U * S_sqrt_diag * Vt;
            }

            state = Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>>(
                product.data(), bond_left, phys_acc * phys_site, bond_right);
        }

        const int total_phys_dim = state.dimension(1);
        OP_REQUIRES(ctx, operator_tensor.dim_size(0) == total_phys_dim &&
                            operator_tensor.dim_size(1) == total_phys_dim,
                    errors::InvalidArgument(
                        "operator shape mismatch: expected [", total_phys_dim, ",",
                        total_phys_dim, "], got [", operator_tensor.dim_size(0), ",",
                        operator_tensor.dim_size(1), "]"));

        Eigen::VectorXf state_vec(total_phys_dim);
        for (int i = 0; i < total_phys_dim; ++i) {
            state_vec(i) = state(0, i, 0);
        }

        using MatrixRowMajor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const MatrixRowMajor> op_mat(
            operator_tensor.flat<float>().data(), total_phys_dim, total_phys_dim);

        Eigen::VectorXf op_state = op_mat * state_vec;
        const float expectation_unnorm = state_vec.dot(op_state);
        const float norm_squared = state_vec.dot(state_vec);
        const float expectation = expectation_unnorm / (norm_squared + 1e-10f);

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<float>()() = expectation;
    }

private:
    int num_sites_;
};

REGISTER_KERNEL_BUILDER(Name("MPSExpect").Device(DEVICE_CPU), MPSExpectOpCpu);

// ============================================================================
// MPSExpectPauli Kernel
// ============================================================================

class MPSExpectPauliOpCpu : public OpKernel {
public:
    explicit MPSExpectPauliOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &num_sites_));
    }

    void Compute(OpKernelContext* ctx) override {
        OpInputList mps_tensors;
        OP_REQUIRES_OK(ctx, ctx->input_list("mps_tensors", &mps_tensors));

        const Tensor& pauli_indices_tensor = ctx->input(num_sites_);
        const Tensor& coefficients_tensor = ctx->input(num_sites_ + 1);

        auto pauli_indices = pauli_indices_tensor.matrix<int32>();
        auto coefficients = coefficients_tensor.flat<float>();

        const int num_strings = pauli_indices.dimension(0);
        
        // Detect batching
        bool is_batched = mps_tensors[0].dims() == 4;
        const int batch_size = is_batched ? mps_tensors[0].dim_size(0) : 1;

        Tensor* output_tensor = nullptr;
        if (is_batched) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &output_tensor));
        } else {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &output_tensor));
        }
        auto out_flat = output_tensor->flat<float>();
        out_flat.setZero();

        for (int b = 0; b < batch_size; ++b) {
            float total_expectation = 0.0f;
            // For each Pauli string, compute expectation site-by-site
            for (int m = 0; m < num_strings; ++m) {
                Eigen::MatrixXf env = Eigen::MatrixXf::Identity(1, 1);

                for (int i = 0; i < num_sites_; ++i) {
                    const Tensor& core_tensor = mps_tensors[i];
                    int chi_L, d, chi_R;
                    const float* data_ptr = nullptr;

                    if (is_batched) {
                        chi_L = core_tensor.dim_size(1);
                        d = core_tensor.dim_size(2);
                        chi_R = core_tensor.dim_size(3);
                        data_ptr = core_tensor.flat<float>().data() + b * chi_L * d * chi_R;
                    } else {
                        chi_L = core_tensor.dim_size(0);
                        d = core_tensor.dim_size(1);
                        chi_R = core_tensor.dim_size(2);
                        data_ptr = core_tensor.flat<float>().data();
                    }
                    
                    const int p_idx = pauli_indices(m, i);
                    Eigen::MatrixXf next_env = Eigen::MatrixXf::Zero(chi_R, chi_R);

                    for (int p_out = 0; p_out < d; ++p_out) {
                        for (int p_in = 0; p_in < d; ++p_in) {
                            float sigma = 0.0f;
                            if (p_idx == 0) sigma = (p_out == p_in) ? 1.0f : 0.0f;
                            else if (p_idx == 1) sigma = (p_out != p_in) ? 1.0f : 0.0f;
                            else if (p_idx == 3) sigma = (p_out == p_in) ? (p_out == 0 ? 1.0f : -1.0f) : 0.0f;
                            
                            if (std::abs(sigma) < 1e-9f) continue;

                            // Map Eigen matrices from raw data pointer
                            // core_i: [chi_L, d, chi_R] -> offset for p is p * chi_R
                            Eigen::Map<const Eigen::MatrixXf> Ai_pin(
                                data_ptr + p_in * chi_R, chi_L, chi_R);
                            Eigen::Map<const Eigen::MatrixXf> Ai_pout(
                                data_ptr + p_out * chi_R, chi_L, chi_R);

                            next_env += sigma * (Ai_pout.transpose() * env * Ai_pin);
                        }
                    }
                    env = next_env;
                }
                total_expectation += coefficients(m) * env(0, 0);
            }
            out_flat(b) = total_expectation;
        }
    }

private:
    int num_sites_;
};

REGISTER_KERNEL_BUILDER(Name("MPSExpectPauli").Device(DEVICE_CPU), MPSExpectPauliOpCpu);

// ============================================================================
// MPSFeatureImportance Kernel
// ============================================================================

class MPSFeatureImportanceOpCpu : public OpKernel {
public:
    explicit MPSFeatureImportanceOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& entropies_tensor = ctx->input(0);
        const bool is_batched = entropies_tensor.dims() == 2;
        const int64_t batch_size = is_batched ? entropies_tensor.dim_size(0) : 1;
        const int64_t num_bonds = is_batched ? entropies_tensor.dim_size(1) : entropies_tensor.NumElements();
        const int64_t num_sites = num_bonds + 1;
        
        auto entropies = entropies_tensor.flat<float>();

        Tensor* output_tensor = nullptr;
        if (is_batched) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size, num_sites}), &output_tensor));
        } else {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({num_sites}), &output_tensor));
        }
        auto importance = output_tensor->flat<float>();

        for (int b = 0; b < batch_size; ++b) {
            const int64_t batch_offset_in = b * num_bonds;
            const int64_t batch_offset_out = b * num_sites;

            if (num_sites == 1) {
                importance(batch_offset_out) = 0.0f;
                continue;
            }

            // Site i importance is average of neighboring bond entropies
            importance(batch_offset_out) = entropies(batch_offset_in);
            for (int i = 1; i < num_sites - 1; ++i) {
                importance(batch_offset_out + i) = (entropies(batch_offset_in + i - 1) + entropies(batch_offset_in + i)) / 2.0f;
            }
            importance(batch_offset_out + num_sites - 1) = entropies(batch_offset_in + num_bonds - 1);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("MPSFeatureImportance").Device(DEVICE_CPU), MPSFeatureImportanceOpCpu);

}  // namespace tensorflow
