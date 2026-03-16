// src/ops/fused_qwt_tokenizer_op.cc
// Copyright 2025 Verso Industries
//
// Enterprise-grade fused Quantum Wavelet Tokenizer operator.
// This kernel integrates three stages that previously required
// separate TensorFlow ops:
//   1. Learnable 1D DWT chunking (from fused_wavelet_encoder_op)
//   2. Wavelet-informed sparse Hamiltonian construction
//   3. Cayley-integrated Continuous-Time Quantum Walk evolution
// The op outputs the raw approximation/detail coefficients as well as
// the evolved, globally contextualized embeddings. A paired gradient
// kernel back-propagates through both the wavelet filters and the
// quantum evolution, enabling end-to-end training of the QWT front-end.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"
#include "common/parallel/parallel_backend.h"
#include "common/edition_limits.h"
#include "Eigen/Core"
#include "Eigen/Sparse"
#include "Eigen/IterativeLinearSolvers" // Added for Eigen::BiCGSTAB
#include "absl/synchronization/mutex.h"
#include "fused_qwt_tokenizer_op.h"  // Phase 17: Enhanced QWT helpers

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

// Phase 11 SIMD include guards (AVX512/AVX2/NEON)
#if defined(__AVX512F__)
#include <immintrin.h>  // AVX512 intrinsics
#elif defined(__AVX2__)
#include <immintrin.h>  // AVX2 intrinsics
#elif defined(__ARM_NEON)
#include <arm_neon.h>   // NEON intrinsics
#endif

namespace tensorflow {
namespace {

using shape_inference::InferenceContext;
using Eigen::MatrixXf;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using Eigen::VectorXf;
using RowMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr float kBaseStepCap = 0.35f;
constexpr float kAdaptiveNumerator = 1.6f;
constexpr float kMinStep = 1e-4f;
constexpr int kMaxRefinement = 4;
constexpr int kDefaultWaveletLevels = 1;  // Phase 10.3: Default for backward compat
constexpr int kMaxWaveletLevels = 5;       // Phase 10.3: Max supported levels

// Phase 10.3: Compute total output nodes for multi-scale wavelet decomposition.
// For num_levels=1: seq_len/2
// For num_levels=2: seq_len/2 + seq_len/4 = 3*seq_len/4
// For num_levels=3: seq_len/2 + seq_len/4 + seq_len/8 = 7*seq_len/8
// General formula: sum of seq_len/(2^i) for i in [1, num_levels]
int64 ComputeMultiScaleOutputNodes(int64 seq_len, int num_levels) {
    int64 total = 0;
    int64 current_len = seq_len;
    for (int level = 0; level < num_levels; ++level) {
        current_len /= 2;
        if (current_len < 1) break;
        total += current_len;
    }
    return std::max(total, (int64)1);
}

struct IntegratorContext {
  int num_steps = 1;
  float local_step = 0.0f;
  float alpha = 0.0f;
  bool converged = false;
  SparseMatrix<float> system_matrix;
};

// Thread-local scratch space for the QWT op to avoid reallocations in loops.
struct QwtScratch {
  SparseMatrix<float> h_sparse;
  MatrixXf evolved_features;
  IntegratorContext integrator_context;
  std::vector<MatrixXf> step_features; // For backward pass

  // Phase 17: Padé approximation cached H powers (H², H³, H⁴)
  std::vector<SparseMatrix<float>> h_powers;

  // Phase 17: Jacobi preconditioner diagonal inverse
  VectorXf jacobi_diag_inv;

  // Phase 17: Skip-connection triplets
  std::vector<Triplet<float>> skip_triplets;

  // Phase 17: Lifting scheme buffers
  std::vector<float> lifting_even_buffer;
  std::vector<float> lifting_odd_buffer;

  void resize(int num_nodes, int hidden_dim) {
    h_sparse.resize(num_nodes, num_nodes);
    evolved_features.resize(num_nodes, hidden_dim);
    jacobi_diag_inv.resize(num_nodes);
  }

  void resize_lifting(int64_t half_seq, int64_t embed_dim) {
    lifting_even_buffer.resize(half_seq * embed_dim);
    lifting_odd_buffer.resize(half_seq * embed_dim);
  }
};

// Phase 11 FULL: Vectorized energy computation with SIMD guards (AVX512/AVX2/NEON/Scalar)
float SafeEnergy(const float* detail_row, int64 hidden_dim, float epsilon) {
  if (hidden_dim <= 0) return epsilon;
  float sum_sq = 0.0f;
  int64 d = 0;

#if defined(__AVX512F__)
  // AVX512: 16-wide SIMD
  __m512 sum_vec_512 = _mm512_setzero_ps();
  for (; d + 16 <= hidden_dim; d += 16) {
    __m512 vals = _mm512_loadu_ps(detail_row + d);
    sum_vec_512 = _mm512_fmadd_ps(vals, vals, sum_vec_512);
  }
  sum_sq += _mm512_reduce_add_ps(sum_vec_512);
#elif defined(__AVX2__)
  // AVX2: 8-wide SIMD
  __m256 sum_vec_256 = _mm256_setzero_ps();
  for (; d + 8 <= hidden_dim; d += 8) {
    __m256 vals = _mm256_loadu_ps(detail_row + d);
    sum_vec_256 = _mm256_fmadd_ps(vals, vals, sum_vec_256);
  }
  // Horizontal sum for AVX2
  __m128 vlow  = _mm256_castps256_ps128(sum_vec_256);
  __m128 vhigh = _mm256_extractf128_ps(sum_vec_256, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  __m128 shuf = _mm_movehdup_ps(vlow);
  __m128 sums = _mm_add_ps(vlow, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  sum_sq += _mm_cvtss_f32(sums);
#elif defined(__ARM_NEON)
  // NEON: 4-wide SIMD
  float32x4_t sum_vec_neon = vdupq_n_f32(0.0f);
  for (; d + 4 <= hidden_dim; d += 4) {
    float32x4_t vals = vld1q_f32(detail_row + d);
    sum_vec_neon = vmlaq_f32(sum_vec_neon, vals, vals);  // FMA: sum_vec_neon + vals * vals
  }
  // Horizontal sum for NEON
  float32x2_t sum_pair = vadd_f32(vget_low_f32(sum_vec_neon), vget_high_f32(sum_vec_neon));
  sum_sq += vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
#endif

  // Scalar fallback for remainder
  for (; d < hidden_dim; ++d) {
    sum_sq += detail_row[d] * detail_row[d];
  }

  return std::sqrt(sum_sq / static_cast<float>(hidden_dim) + epsilon);
}

float EstimateSpectralRadius(const SparseMatrix<float>& h_sparse) {
  VectorXf row_abs = VectorXf::Zero(h_sparse.rows());
  for (int k = 0; k < h_sparse.outerSize(); ++k) {
    for (SparseMatrix<float>::InnerIterator it(h_sparse, k); it; ++it) {
      row_abs(it.row()) += std::abs(it.value());
    }
  }
  return row_abs.maxCoeff();
}

SparseMatrix<float> BuildCayleySystem(const SparseMatrix<float>& h_sparse,
                                      float alpha) {
  std::vector<Triplet<float>> triplets;
  triplets.reserve(h_sparse.nonZeros() + h_sparse.rows());
  for (int k = 0; k < h_sparse.outerSize(); ++k) {
    for (SparseMatrix<float>::InnerIterator it(h_sparse, k); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), -alpha * it.value());
    }
  }
  for (int i = 0; i < h_sparse.rows(); ++i) {
    triplets.emplace_back(i, i, 1.0f);
  }
  SparseMatrix<float> system(h_sparse.rows(), h_sparse.cols());
  system.setFromTriplets(triplets.begin(), triplets.end());
  system.makeCompressed();
  return system;
}

float ResolveLocalStep(float evolution_time, float spectral_upper) {
  spectral_upper = std::max(spectral_upper, 1e-6f);
  const float adaptive_cap = kAdaptiveNumerator / spectral_upper;
  const float step_cap = std::max(kMinStep, std::min(kBaseStepCap, adaptive_cap));
  const float t_abs = std::abs(evolution_time);
  if (t_abs == 0.0f) {
    return 0.0f;
  }
  const float steps = std::ceil(t_abs / step_cap);
  return t_abs / std::max(1.0f, steps);
}

// Phase 17: Multi-level lifting scheme cascade
void RunLiftingCascadeForward(
    const float* input,
    const float* low_pass,  // predict
    const float* high_pass, // update
    float* approx_out,
    float* detail_out,
    int64 batch,
    int64 seq_len,
    int64 hidden,
    int num_levels) {
    
    int64 output_offset = 0;
    std::vector<float> cascade_buffer;
    
    for (int level = 0; level < num_levels; ++level) {
        int64 l_input_len = seq_len >> level;
        int64 l_output_len = l_input_len / 2;
        if (l_output_len < 1) break;

        auto level_work = [&](int64 b_start, int64 b_end) {
            for (int64 b = b_start; b < b_end; ++b) {
                const float* src = (level == 0) ? input + b * seq_len * hidden : 
                                   cascade_buffer.data() + b * l_input_len * hidden;
                float* l_approx = approx_out + (b * ComputeMultiScaleOutputNodes(seq_len, num_levels) + output_offset) * hidden;
                float* l_detail = detail_out + (b * ComputeMultiScaleOutputNodes(seq_len, num_levels) + output_offset) * hidden;

                saguaro::ops::qwt::qwt_lifting_forward(
                    src, low_pass, high_pass, l_approx, l_detail,
                    l_input_len, hidden, 1); // Lifting kernels are size 1 for Haar/simple
            }
        };

        saguaro::parallel::ForShard(static_cast<std::size_t>(batch), 5000, level_work);

        if (level + 1 < num_levels) {
            cascade_buffer.resize(batch * l_output_len * hidden);
            for (int64 b = 0; b < batch; ++b) {
                std::memcpy(cascade_buffer.data() + b * l_output_len * hidden,
                            approx_out + (b * ComputeMultiScaleOutputNodes(seq_len, num_levels) + output_offset) * hidden,
                            l_output_len * hidden * sizeof(float));
            }
        }
        output_offset += l_output_len;
    }
}

// Phase 17: Multi-level lifting scheme backward (gradients)
void RunLiftingCascadeBackward(
    const float* input,
    const float* low_pass,  // predict
    const float* high_pass, // update
    const float* grad_approx,
    const float* grad_detail,
    float* grad_input_out,
    float* grad_low_out,
    float* grad_high_out,
    int64 batch,
    int64 seq_len,
    int hidden,
    int num_levels) {
    
    // Gradient flow through levels is the inverse of the forward pass
    // We go from level num_levels-1 down to 0
    // Each level requires:
    // 1. Inputs to forward (approx from level-1 or input)
    // 2. Output of forward (approx, detail)
    // 3. Upstream gradients (grad_approx, grad_detail)
    
    // This requires caching or re-running forward. For simplicity here,
    // we assume we have approx/detail from the forward pass.
    
    // [Implementation of cascaded lifting gradients goes here]
    // For now, let's use a similar ForShard approach as forward.
    // Each level's gradient is computed and then propagated down.
}

void BuildHamiltonianFromDetail(const Eigen::Ref<const RowMatrixXf>& detail,
                                float epsilon,
                                SparseMatrix<float>* h_sparse_out) {
  const int num_nodes = static_cast<int>(detail.rows());
  const int hidden_dim = static_cast<int>(detail.cols());
  std::vector<float> energies(num_nodes, epsilon);

  // Phase 17: Use optimized energy computation
  saguaro::ops::qwt::qwt_compute_node_energies(detail.data(), energies.data(), 
                                               num_nodes, hidden_dim, epsilon);

  std::vector<Triplet<float>> adjacency_triplets;
  adjacency_triplets.reserve(num_nodes * 3);
  VectorXf degrees = VectorXf::Zero(num_nodes);

  for (int i = 0; i < num_nodes; ++i) {
    if (energies[i] > 0.0f) {
      adjacency_triplets.emplace_back(i, i, energies[i]);
      degrees(i) += energies[i];
    }
  }
  for (int i = 0; i + 1 < num_nodes; ++i) {
    const float weight = 0.5f * (energies[i] + energies[i + 1]);
    if (weight <= 0.0f) continue;
    adjacency_triplets.emplace_back(i, i + 1, weight);
    adjacency_triplets.emplace_back(i + 1, i, weight);
    degrees(i) += weight;
    degrees(i + 1) += weight;
  }

  SparseMatrix<float> adjacency(num_nodes, num_nodes);
  adjacency.setFromTriplets(adjacency_triplets.begin(), adjacency_triplets.end());
  adjacency.makeCompressed();

  SparseMatrix<float> degree(num_nodes, num_nodes);
  degree.reserve(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    if (degrees(i) != 0.0f) {
      degree.insert(i, i) = degrees(i);
    }
  }
  degree.makeCompressed();

  *h_sparse_out = degree - adjacency;
  h_sparse_out->makeCompressed();
}

/**
 * @brief Runs the Cayley evolution for a given Hamiltonian and initial state.
 *
 * This function simulates the continuous-time quantum walk (CTQW) evolution
 * using the Cayley approximation of the matrix exponential. It iteratively
 * solves a linear system to propagate the state forward in time.
 *
 * The Cayley approximation for `exp(-iHt)` is given by `(I + iHt/2)^-1 (I - iHt/2)`.
 * In this implementation, we solve `(I + alpha * H) X_{n+1} = (I - alpha * H) X_n`
 * where `alpha = 0.5 * local_step`.
 *
 * @param h_sparse The sparse Hamiltonian matrix.
 * @param initial The initial feature matrix (state).
 * @param evolution_time The total time for the evolution.
 * @param evolved Output parameter: the evolved feature matrix after `evolution_time`.
 * @param ctx Output parameter: IntegratorContext containing details of the integration.
 * @param step_features Output parameter (optional): stores intermediate feature matrices at each step.
 * @param override_steps If > 0, forces a specific number of steps for the evolution.
 * @return True if the evolution converged successfully, false otherwise.
 */
bool RunCayleyEvolution(const SparseMatrix<float>& h_sparse,
                        const Eigen::Ref<const RowMatrixXf>& initial,
                        float evolution_time,
                        MatrixXf* evolved,
                        IntegratorContext* ctx,
                        std::vector<MatrixXf>* step_features,
                        int override_steps = 0,
                        int pade_order = 1,
                        const VectorXf* jacobi_diag_inv = nullptr) {
  const int num_nodes = h_sparse.rows();
  if (num_nodes == 0) {
    *evolved = initial;
    if (ctx) {
      ctx->num_steps = 1;
      ctx->local_step = 0.0f;
      ctx->alpha = 0.0f;
      ctx->converged = true;
      ctx->system_matrix.resize(0, 0);
    }
    if (step_features) {
      step_features->clear();
      step_features->push_back(initial);
    }
    return true;
  }

  float t_abs = std::abs(evolution_time);
  float spectral_radius_est = EstimateSpectralRadius(h_sparse);
  float local_step = ResolveLocalStep(evolution_time, spectral_radius_est);
  
  // Phase 17: Padé expansion allows for larger steps
  if (pade_order > 1) {
    local_step *= (1.0f + 0.5f * (pade_order - 1));
  }

  int num_steps = (t_abs <= 0.0f || local_step <= 0.0f)
                       ? 1
                       : std::max(1, static_cast<int>(std::ceil(t_abs / local_step)));
  if (override_steps > 0) {
    num_steps = std::max(1, override_steps);
  }

  Eigen::BiCGSTAB<SparseMatrix<float>> solver;
  SparseMatrix<float> system_matrix;
  MatrixXf aggregate = initial;
  bool ok = (t_abs == 0.0f);
  int refinement = 0;
  const bool forced_steps = override_steps > 0;

  // Cache powers if using Padé[m/m] where m > 1
  std::vector<SparseMatrix<float>> h_powers;
  if (pade_order > 1) {
    saguaro::ops::qwt::qwt_precompute_h_powers(h_sparse, pade_order, h_powers);
  }

  while (!ok && refinement <= kMaxRefinement) {
    local_step = (num_steps > 0) ? t_abs / static_cast<float>(num_steps) : 0.0f;
    const float alpha = 0.5f * local_step;
    
    if (pade_order <= 1) {
      system_matrix = BuildCayleySystem(h_sparse, alpha);
    } else {
      const auto& pade = saguaro::ops::qwt::GetPadeCoefficients(pade_order);
      system_matrix = saguaro::ops::qwt::qwt_build_pade_denominator(h_sparse, alpha, pade, &h_powers);
    }

    solver.compute(system_matrix);
    if (solver.info() != Eigen::Success) {
      num_steps *= 2;
      ++refinement;
      continue;
    }

    if (step_features) {
      step_features->assign(num_steps + 1, MatrixXf(num_nodes, initial.cols()));
      (*step_features)[0] = initial;
    }

    MatrixXf current = initial;
    bool failed = false;
    for (int step = 0; step < num_steps; ++step) {
      MatrixXf rhs;
      if (pade_order <= 1) {
        MatrixXf h_x = h_sparse * current;
        rhs = current + alpha * h_x;
      } else {
        const auto& pade = saguaro::ops::qwt::GetPadeCoefficients(pade_order);
        SparseMatrix<float> num_mat = saguaro::ops::qwt::qwt_build_pade_numerator(h_sparse, alpha, pade, &h_powers);
        rhs = num_mat * current;
      }

      // Phase 17: Apply Jacobi Preconditioning if available
      if (jacobi_diag_inv) {
        MatrixXf precond_rhs;
        saguaro::ops::qwt::qwt_apply_jacobi_preconditioner_matrix(*jacobi_diag_inv, rhs, precond_rhs);
        MatrixXf next = solver.solve(precond_rhs);
        if (solver.info() != Eigen::Success || !next.allFinite()) {
          failed = true;
          break;
        }
        current = std::move(next);
      } else {
        MatrixXf next = solver.solve(rhs);
        if (solver.info() != Eigen::Success || !next.allFinite()) {
          failed = true;
          break;
        }
        current = std::move(next);
      }

      if (step_features) {
        (*step_features)[step + 1] = current;
      }
    }

    if (!failed) {
      aggregate = current;
      ok = true;
      if (ctx) {
        ctx->num_steps = num_steps;
        ctx->local_step = local_step;
        ctx->alpha = alpha;
        ctx->system_matrix = system_matrix;
        ctx->converged = true;
      }
    } else {
      if (forced_steps) break;
      num_steps *= 2;
      ++refinement;
    }
  }

  if (!ok) {
    aggregate = initial;
    if (ctx) {
      ctx->num_steps = 1;
      ctx->local_step = 0.0f;
      ctx->alpha = 0.0f;
      ctx->system_matrix.resize(num_nodes, num_nodes);
      ctx->system_matrix.setIdentity();
      ctx->converged = false;
    }
  }

  *evolved = aggregate;
  return ok;
}

Status ValidateEvolutionTensor(const Tensor& evolution_time, int64 batch) {
  if (evolution_time.dims() == 0) {
    return OkStatus();
  }
  if (evolution_time.dims() != 1) {
    return errors::InvalidArgument(
        "evolution_time must be scalar or 1-D, received rank ",
        evolution_time.dims());
  }
  if (evolution_time.dim_size(0) != batch) {
    return errors::InvalidArgument(
        "evolution_time shape mismatch: expected length ", batch,
        ", got ", evolution_time.dim_size(0));
  }
  return OkStatus();
}

std::vector<float> ResolveEvolutionTimes(const Tensor& evolution_time,
                                         int64 batch) {
  std::vector<float> values(batch, 0.0f);
  if (evolution_time.dims() == 0) {
    const float t = evolution_time.scalar<float>()();
    std::fill(values.begin(), values.end(), t);
  } else {
    auto vec = evolution_time.flat<float>();
    for (int64 i = 0; i < batch; ++i) {
      values[i] = vec(i);
    }
  }
  return values;
}

}  // namespace

// ----------------------------------------------------------------------------
// OP REGISTRATION
// ----------------------------------------------------------------------------

REGISTER_OP("FusedQwtTokenizer")
    .Input("input_data: float")           // [B, S, D]
    .Input("low_pass_filter: float")      // [2, 1, D]
    .Input("high_pass_filter: float")     // [2, 1, D]
    .Input("mask: bool")                  // [B, S]
    .Input("evolution_time: float")       // scalar or [B]
    .Input("ctqw_steps: int32")           // scalar or [B]
    .Attr("epsilon: float = 1e-5")
    .Attr("num_wavelet_levels: int = 1")  // Phase 10.3: Multi-scale DWT levels
    // Phase 17: QWT Enhancement Attributes
    .Attr("use_lifting_scheme: bool = true")    // Enhancement 1: Lifting DWT
    .Attr("pade_order: int = 4")                // Enhancement 2: Padé order (1-4)
    .Attr("use_jacobi_preconditioner: bool = true")  // Enhancement 3: Jacobi precond
    .Attr("skip_stride: int = 0")               // Enhancement 4: Skip connections
    .Attr("max_skips_per_node: int = 2")        // Enhancement 4: Max skips
    .Output("approx_coeffs: float")       // [B, multi_out, D]
    .Output("detail_coeffs: float")       // [B, multi_out, D]
    .Output("qwt_embeddings: float")      // [B, multi_out, D]
    .SetShapeFn([](InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
      shape_inference::DimensionHandle batch = c->Dim(input_shape, 0);
      shape_inference::DimensionHandle seq_len = c->Dim(input_shape, 1);
      shape_inference::DimensionHandle hidden = c->Dim(input_shape, 2);
      
      // Phase 10.3: Compute output size based on num_wavelet_levels
      int num_levels = 1;
      TF_RETURN_IF_ERROR(c->GetAttr("num_wavelet_levels", &num_levels));
      num_levels = std::max(1, std::min(num_levels, kMaxWaveletLevels));
      
      // For static shape inference, we compute based on known dimensions
      // or use unknown dimension if seq_len not known at graph construction
      shape_inference::DimensionHandle output_nodes;
      if (c->ValueKnown(seq_len)) {
          int64_t s = c->Value(seq_len);
          int64_t out_nodes = ComputeMultiScaleOutputNodes(s, num_levels);
          output_nodes = c->MakeDim(out_nodes);
      } else {
          // Fallback: unknown output dimension
          output_nodes = c->UnknownDim();
      }
      
      c->set_output(0, c->MakeShape({batch, output_nodes, hidden}));
      c->set_output(1, c->MakeShape({batch, output_nodes, hidden}));
      c->set_output(2, c->MakeShape({batch, output_nodes, hidden}));
      return OkStatus();
    });

REGISTER_OP("FusedQwtTokenizerGrad")
    .Input("grad_approx: float")          // [B, multi_out, D]
    .Input("grad_detail: float")          // [B, multi_out, D]
    .Input("grad_qwt: float")             // [B, multi_out, D]
    .Input("input_data: float")
    .Input("low_pass_filter: float")
    .Input("high_pass_filter: float")
    .Input("mask: bool")
    .Input("evolution_time: float")
    .Input("ctqw_steps: int32")
    .Input("approx_coeffs: float")
    .Input("detail_coeffs: float")
    .Attr("epsilon: float = 1e-5")
    .Attr("num_wavelet_levels: int = 1")  // Phase 10.3: Multi-scale DWT levels
    // Phase 17: QWT Enhancement Attributes (must match forward op)
    .Attr("use_lifting_scheme: bool = true")
    .Attr("pade_order: int = 4")
    .Attr("use_jacobi_preconditioner: bool = true")
    .Attr("skip_stride: int = 0")
    .Attr("max_skips_per_node: int = 2")
    .Output("grad_input_data: float")
    .Output("grad_low_pass_filter: float")
    .Output("grad_high_pass_filter: float")
    .Output("grad_evolution_time: float")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(3));
      c->set_output(1, c->input(4));
      c->set_output(2, c->input(5));
      // grad evolution time matches evolution_time input
      c->set_output(3, c->input(7));
      return OkStatus();
    });

// ----------------------------------------------------------------------------
// FORWARD KERNEL
// ----------------------------------------------------------------------------

class FusedQwtTokenizerOp : public OpKernel {
 public:
  explicit FusedQwtTokenizerOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    epsilon_ = std::max(1e-8f, epsilon_);
    
    // Phase 10.3: Read num_wavelet_levels attribute
    OP_REQUIRES_OK(context, context->GetAttr("num_wavelet_levels", &num_wavelet_levels_));
    num_wavelet_levels_ = std::max(1, std::min(num_wavelet_levels_, kMaxWaveletLevels));

    // Phase 17: Read enhancement attributes
    OP_REQUIRES_OK(context, context->GetAttr("use_lifting_scheme", &use_lifting_scheme_));
    OP_REQUIRES_OK(context, context->GetAttr("pade_order", &pade_order_));
    pade_order_ = std::max(1, std::min(pade_order_, saguaro::ops::qwt::QWT_MAX_PADE_ORDER));
    OP_REQUIRES_OK(context, context->GetAttr("use_jacobi_preconditioner", &use_jacobi_preconditioner_));
    OP_REQUIRES_OK(context, context->GetAttr("skip_stride", &skip_stride_));
    OP_REQUIRES_OK(context, context->GetAttr("max_skips_per_node", &max_skips_per_node_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& low_filter_tensor = context->input(1);
    const Tensor& high_filter_tensor = context->input(2);
    const Tensor& mask_tensor = context->input(3);
    const Tensor& evolution_time_tensor = context->input(4);
    const Tensor& ctqw_steps_tensor = context->input(5);

    const int64 batch = input_tensor.dim_size(0);
    const int64 seq_len = input_tensor.dim_size(1);
    const int64 hidden = input_tensor.dim_size(2);
    const int64 filter_width = low_filter_tensor.dim_size(0);
    
    // Phase 10.3: Calculate multi-scale output dimensions
    const int64 total_output_nodes = ComputeMultiScaleOutputNodes(seq_len, num_wavelet_levels_);
    const int eigen_output_nodes = static_cast<int>(total_output_nodes);
    const int eigen_hidden = static_cast<int>(hidden);

    OP_REQUIRES(context, seq_len > 0 && seq_len % 2 == 0,
                errors::InvalidArgument("Sequence length must be positive and even."));
    
    // HighNoon Lite Edition: Enforce context length limit (max 5M tokens)
    SAGUARO_CHECK_CONTEXT_LENGTH(context, seq_len);
    
    OP_REQUIRES(context, filter_width >= 2,
                errors::InvalidArgument("Filter width must be >= 2 for DWT."));
    OP_REQUIRES(context, high_filter_tensor.dim_size(0) == filter_width,
                errors::InvalidArgument("Low/high pass filters must share kernel length."));
    OP_REQUIRES(context, mask_tensor.dims() == 2,
                errors::InvalidArgument("mask must be rank-2 [batch, seq_len], received rank ",
                                        mask_tensor.dims()));
    OP_REQUIRES(context, mask_tensor.dim_size(0) == batch,
                errors::InvalidArgument("mask batch dimension mismatch: expected ",
                                        batch, ", got ", mask_tensor.dim_size(0)));
    OP_REQUIRES(context, mask_tensor.dim_size(1) == seq_len,
                errors::InvalidArgument("mask seq_len mismatch: expected ",
                                        seq_len, ", got ", mask_tensor.dim_size(1)));
    OP_REQUIRES_OK(context, ValidateEvolutionTensor(evolution_time_tensor, batch));

    // Phase 10.3: Allocate outputs for multi-scale coefficients
    Tensor* approx_tensor = nullptr;
    Tensor* detail_tensor = nullptr;
    Tensor* qwt_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch, total_output_nodes, hidden}), &approx_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({batch, total_output_nodes, hidden}), &detail_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({batch, total_output_nodes, hidden}), &qwt_tensor));

    approx_tensor->flat<float>().setZero();
    detail_tensor->flat<float>().setZero();

    auto input = input_tensor.tensor<float, 3>();
    auto low_filter = low_filter_tensor.tensor<float, 3>();
    auto high_filter = high_filter_tensor.tensor<float, 3>();
    auto approx = approx_tensor->tensor<float, 3>();
    auto detail = detail_tensor->tensor<float, 3>();
    float* approx_data = approx_tensor->flat<float>().data();
    float* detail_data = detail_tensor->flat<float>().data();
    float* qwt_data = qwt_tensor->flat<float>().data();
    auto mask = mask_tensor.matrix<bool>();

    // Phase 10.3: For multi-scale DWT, we compute level 1 first
    // The first level always has seq_len/2 nodes
    const int64 level1_nodes = seq_len / 2;

    auto node_offset = [&](int64 b_idx, int64 node_idx) -> int64 {
      return ((b_idx * total_output_nodes) + node_idx) * hidden;
    };

    // =========================================================================
    // Phase 17: Multi-Scale DWT Implementation (Lifting + Parallel)
    // =========================================================================
    // Phase 17: Multi-Scale DWT Implementation (Lifting + Parallel)
    // =========================================================================
    if (use_lifting_scheme_) {
      RunLiftingCascadeForward(
          input.data(), low_filter.data(), high_filter.data(),
          approx_data, detail_data, 
          batch, seq_len, hidden, num_wavelet_levels_);
    } else {
      // Phase 10.3 / 17: Parallel FIR Cascade
      // We need temporary storage for cascaded processing
      std::vector<float> cascade_buffer(batch * seq_len * hidden / 2);
      float* cascade_data = cascade_buffer.data();

      int64 output_offset = 0;

      for (int level = 0; level < num_wavelet_levels_; ++level) {
          const int64 level_input_len = seq_len >> level;
          const int64 level_output_len = level_input_len / 2;
          if (level_output_len < 1) break;

          auto level_work = [&](int64_t b_start, int64_t b_end) {
              for (int64 b = b_start; b < b_end; ++b) {
                  for (int64 m = 0; m < level_output_len; ++m) {
                      const int64 base_index = m * 2;
                      for (int64 k = 0; k < filter_width; ++k) {
                          const int64 src = base_index + k;
                          if (src >= level_input_len) break;

                          const float* src_ptr = (level == 0) ? &input(b, src, 0) : 
                                               cascade_data + (b * level_input_len + src) * hidden;
                          
                          if (level == 0 && !mask(b, src)) continue;

                          float* approx_ptr = &approx(b, output_offset + m, 0);
                          float* detail_ptr = &detail(b, output_offset + m, 0);

                          for (int64 d = 0; d < hidden; ++d) {
                              approx_ptr[d] += src_ptr[d] * low_filter(k, 0, d);
                              detail_ptr[d] += src_ptr[d] * high_filter(k, 0, d);
                          }
                      }
                  }
              }
          };

          saguaro::parallel::ForShard(static_cast<std::size_t>(batch), 1000, level_work);

          if (level + 1 < num_wavelet_levels_) {
              for (int64 b = 0; b < batch; ++b) {
                  std::memcpy(cascade_data + b * level_output_len * hidden,
                              &approx(b, output_offset, 0),
                              level_output_len * hidden * sizeof(float));
              }
          }
          output_offset += level_output_len;
      }
    }

    // =========================================================================
    // End of Phase 10.3 Cascaded DWT
    // =========================================================================

    // For Hamiltonian/CTQW, we use the finest resolution (level 1 nodes)
    const int64 nodes = level1_nodes;

    std::vector<float> time_values = ResolveEvolutionTimes(evolution_time_tensor, batch);
    std::vector<int> ctqw_overrides(batch, 0);
    if (ctqw_steps_tensor.dims() == 0) {
        int override = std::max(0, ctqw_steps_tensor.scalar<int32>()());
        std::fill(ctqw_overrides.begin(), ctqw_overrides.end(), override);
    } else {
        OP_REQUIRES(context, ctqw_steps_tensor.dims() == 1,
                    errors::InvalidArgument("ctqw_steps must be scalar or 1-D."));
        OP_REQUIRES(context, ctqw_steps_tensor.dim_size(0) == batch,
                    errors::InvalidArgument("ctqw_steps vector length (%ld) must match batch (%ld).",
                                            ctqw_steps_tensor.dim_size(0), batch));
        auto steps_vec = ctqw_steps_tensor.vec<int32>();
        for (int64 i = 0; i < batch; ++i) {
            ctqw_overrides[i] = std::max(0, steps_vec(i));
        }
    }

    auto work = [&](int64_t start, int64_t end) {
        // Phase 10.3: Hamiltonian building uses first-level (finest) nodes
        const int eigen_nodes_local = static_cast<int>(nodes);
        
        for (int64 b = start; b < end; ++b) {
            static thread_local QwtScratch scratch;
            scratch.resize(eigen_nodes_local, eigen_hidden);

            float* approx_batch = approx_data + b * total_output_nodes * hidden;
            float* detail_batch = detail_data + b * total_output_nodes * hidden;

            Eigen::Map<const RowMatrixXf> detail_map(detail_batch, eigen_nodes_local, eigen_hidden);
            BuildHamiltonianFromDetail(detail_map, epsilon_, &scratch.h_sparse);
            
            // Phase 17.4: Add skip connections if enabled
            if (skip_stride_ > 0) {
              // We need the energies for skip connection weights
              std::vector<float> node_energies(eigen_nodes_local);
              saguaro::ops::qwt::qwt_compute_node_energies(detail_batch, node_energies.data(), 
                                                           eigen_nodes_local, eigen_hidden, epsilon_);
              saguaro::ops::qwt::qwt_compute_skip_connections(node_energies, eigen_nodes_local,
                                                              skip_stride_, max_skips_per_node_, scratch.skip_triplets);
              scratch.h_sparse.setFromTriplets(scratch.skip_triplets.begin(), scratch.skip_triplets.end());
              scratch.h_sparse.makeCompressed();
            }

            // Phase 17.3: Extract Jacobi preconditioner if enabled
            if (use_jacobi_preconditioner_) {
              saguaro::ops::qwt::qwt_extract_diagonal_inverse(scratch.h_sparse, scratch.jacobi_diag_inv);
            }

            Eigen::Map<const RowMatrixXf> approx_map(approx_batch, eigen_nodes_local, eigen_hidden);
            Eigen::Map<RowMatrixXf> qwt_map(qwt_data + node_offset(b, 0), eigen_nodes_local, eigen_hidden);

            // Phase 17.2: Use Padé approximation if requested
            RunCayleyEvolution(scratch.h_sparse,
                               approx_map,
                               time_values[b],
                               &scratch.evolved_features,
                               &scratch.integrator_context,
                               /*step_features=*/nullptr,
                               ctqw_overrides[b],
                               pade_order_,
                               use_jacobi_preconditioner_ ? &scratch.jacobi_diag_inv : nullptr);
            qwt_map = scratch.evolved_features;
        }
    };
    const std::size_t cost_per_unit = static_cast<std::size_t>(nodes * hidden * 1000);
    saguaro::parallel::ForShard(
        static_cast<std::size_t>(batch),
        cost_per_unit,
        work);
  }

 private:
  float epsilon_;
  int num_wavelet_levels_;  // Phase 10.3: Number of cascaded DWT levels
  // Phase 17: Enhancement attributes
  bool use_lifting_scheme_;
  int pade_order_;
  bool use_jacobi_preconditioner_;
  int skip_stride_;
  int max_skips_per_node_;
};

// ----------------------------------------------------------------------------
// BACKWARD KERNEL
// ----------------------------------------------------------------------------

class FusedQwtTokenizerGradOp : public OpKernel {
 public:
  explicit FusedQwtTokenizerGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    epsilon_ = std::max(1e-8f, epsilon_);
    
    // Phase 10.3: Read num_wavelet_levels attribute
    OP_REQUIRES_OK(context, context->GetAttr("num_wavelet_levels", &num_wavelet_levels_));
    num_wavelet_levels_ = std::max(1, std::min(num_wavelet_levels_, kMaxWaveletLevels));

    // Phase 17: Read enhancement attributes
    OP_REQUIRES_OK(context, context->GetAttr("use_lifting_scheme", &use_lifting_scheme_));
    OP_REQUIRES_OK(context, context->GetAttr("pade_order", &pade_order_));
    pade_order_ = std::max(1, std::min(pade_order_, saguaro::ops::qwt::QWT_MAX_PADE_ORDER));
    OP_REQUIRES_OK(context, context->GetAttr("use_jacobi_preconditioner", &use_jacobi_preconditioner_));
    OP_REQUIRES_OK(context, context->GetAttr("skip_stride", &skip_stride_));
    OP_REQUIRES_OK(context, context->GetAttr("max_skips_per_node", &max_skips_per_node_));
  }

  /**
   * @brief Computes the gradients for the FusedQwtTokenizer operation.
   *
   * This kernel implements the backward pass for the fused Quantum Wavelet Tokenizer.
   * It calculates gradients with respect to the input data, low-pass filter,
   * high-pass filter, and evolution time.
   *
   * The gradient computation involves:
   * 1. Reconstructing the Hamiltonian from the cached detail coefficients.
   * 2. Re-running the forward Cayley evolution to capture intermediate states (`step_features`).
   * 3. Performing a backward pass through the Cayley evolution using an adjoint method,
   *    solving linear systems involving the transpose of the system matrix.
   * 4. Accumulating gradients for the evolution time based on the sensitivity of the evolution.
   * 5. Propagating gradients back through the wavelet transform stage to compute
   *    gradients for the input data and the low/high-pass filters.
   *
   * @param context The OpKernelContext containing inputs and outputs for the gradient computation.
   *
   * Inputs:
   *   - grad_approx: Gradient of the loss with respect to approximation coefficients.
   *   - grad_detail: Gradient of the loss with respect to detail coefficients.
   *   - grad_qwt: Gradient of the loss with respect to QWT embeddings.
   *   - input_data: Original input data from the forward pass.
   *   - low_pass_filter: Original low-pass filter from the forward pass.
   *   - high_pass_filter: Original high-pass filter from the forward pass.
   *   - mask: Original mask from the forward pass.
   *   - evolution_time: Original evolution time from the forward pass.
   *   - ctqw_steps: Original CTQW steps from the forward pass.
   *   - approx_coeffs: Cached approximation coefficients from the forward pass.
   *   - detail_coeffs: Cached detail coefficients from the forward pass.
   *
   * Outputs:
   *   - grad_input_data: Gradient with respect to the input data.
   *   - grad_low_pass_filter: Gradient with respect to the low-pass filter.
   *   - grad_high_pass_filter: Gradient with respect to the high-pass filter.
   *   - grad_evolution_time: Gradient with respect to the evolution time.
   */
  void Compute(OpKernelContext* context) override {
    // Upstream gradients
    const Tensor& grad_approx_tensor = context->input(0);
    const Tensor& grad_detail_tensor = context->input(1);
    const Tensor& grad_qwt_tensor = context->input(2);

    // Forward inputs
    const Tensor& input_tensor = context->input(3);
    const Tensor& low_filter_tensor = context->input(4);
    const Tensor& high_filter_tensor = context->input(5);
    const Tensor& mask_tensor = context->input(6);
    const Tensor& evolution_time_tensor = context->input(7);
    const Tensor& ctqw_steps_tensor = context->input(8);

    // Forward outputs (used as caches)
    const Tensor& approx_tensor = context->input(9);
    const Tensor& detail_tensor = context->input(10);

    const int64 batch = input_tensor.dim_size(0);
    const int64 seq_len = input_tensor.dim_size(1);
    const int64 hidden = input_tensor.dim_size(2);
    
    // Phase 10.3: Calculate multi-scale output dimensions
    // The CTQW operates on level 1 nodes (finest resolution)
    // but gradients may flow through all levels
    const int64 total_output_nodes = ComputeMultiScaleOutputNodes(seq_len, num_wavelet_levels_);
    const int64 nodes = seq_len / 2;  // Level 1 nodes for CTQW
    const int eigen_nodes = static_cast<int>(nodes);
    const int eigen_hidden = static_cast<int>(hidden);
    const int64 filter_width = low_filter_tensor.dim_size(0);
    OP_REQUIRES(context, high_filter_tensor.dim_size(0) == filter_width,
                errors::InvalidArgument("Low/high pass filters must share kernel length."));

    OP_REQUIRES(context, mask_tensor.dims() == 2,
                errors::InvalidArgument("mask must be rank-2 [batch, seq_len], received rank ",
                                        mask_tensor.dims()));
    OP_REQUIRES(context, mask_tensor.dim_size(0) == batch,
                errors::InvalidArgument("mask batch dimension mismatch: expected ",
                                        batch, ", got ", mask_tensor.dim_size(0)));
    OP_REQUIRES(context, mask_tensor.dim_size(1) == seq_len,
                errors::InvalidArgument("mask seq_len mismatch: expected ",
                                        seq_len, ", got ", mask_tensor.dim_size(1)));

    OP_REQUIRES_OK(context, ValidateEvolutionTensor(evolution_time_tensor, batch));

    Tensor* grad_input_tensor = nullptr;
    Tensor* grad_low_tensor = nullptr;
    Tensor* grad_high_tensor = nullptr;
    Tensor* grad_time_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &grad_input_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, low_filter_tensor.shape(), &grad_low_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, high_filter_tensor.shape(), &grad_high_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(3, evolution_time_tensor.shape(), &grad_time_tensor));

    grad_input_tensor->flat<float>().setZero();
    grad_low_tensor->flat<float>().setZero();
    grad_high_tensor->flat<float>().setZero();

    std::vector<float> time_values = ResolveEvolutionTimes(evolution_time_tensor, batch);
    std::vector<float> grad_time(batch, 0.0f);
    std::vector<int32> ctqw_overrides(batch, 0);
    if (ctqw_steps_tensor.dims() == 0) {
        int32 override_val = std::max(0, ctqw_steps_tensor.scalar<int32>()());
        std::fill(ctqw_overrides.begin(), ctqw_overrides.end(), override_val);
    } else {
        OP_REQUIRES(context, ctqw_steps_tensor.dims() == 1,
                    errors::InvalidArgument("ctqw_steps must be scalar or 1-D."));
        OP_REQUIRES(context, ctqw_steps_tensor.dim_size(0) == batch,
                    errors::InvalidArgument("ctqw_steps vector length (%ld) must match batch (%ld).",
                                            ctqw_steps_tensor.dim_size(0), batch));
        auto steps_vec = ctqw_steps_tensor.vec<int32>();
        for (int64 i = 0; i < batch; ++i) {
            ctqw_overrides[i] = std::max(0, steps_vec(i));
        }
    }

    auto grad_input = grad_input_tensor->tensor<float, 3>();
    const auto input = input_tensor.tensor<float, 3>();
    const auto low_filter = low_filter_tensor.tensor<float, 3>();
    const auto high_filter = high_filter_tensor.tensor<float, 3>();
    const auto mask = mask_tensor.matrix<bool>();

    const float* approx_data = approx_tensor.flat<float>().data();
    const float* detail_data = detail_tensor.flat<float>().data();
    const float* grad_qwt_data = grad_qwt_tensor.flat<float>().data();
    const float* grad_approx_data = grad_approx_tensor.flat<float>().data();
    const float* grad_detail_data = grad_detail_tensor.flat<float>().data();

    auto work = [&](int64_t start, int64_t end) {
        // Thread-local storage for accumulating gradients to filters.
        // These will be aggregated into the global gradient tensors at the end of the work unit.
        Tensor local_grad_low_tensor(DT_FLOAT, low_filter_tensor.shape());
        auto local_grad_low = local_grad_low_tensor.tensor<float, 3>();
        local_grad_low.setZero();
        Tensor local_grad_high_tensor(DT_FLOAT, high_filter_tensor.shape());
        auto local_grad_high = local_grad_high_tensor.tensor<float, 3>();
        local_grad_high.setZero();

        for (int64 b = start; b < end; ++b) {
            const int64 base_offset = b * total_output_nodes * hidden;
            const int64 node_offset_val = b * nodes * hidden; // finest level nodes

            Eigen::Map<const RowMatrixXf> approx_map(approx_data + node_offset_val, eigen_nodes, eigen_hidden);
            Eigen::Map<const RowMatrixXf> detail_map(detail_data + node_offset_val, eigen_nodes, eigen_hidden);
            Eigen::Map<const RowMatrixXf> grad_qwt(grad_qwt_data + node_offset_val, eigen_nodes, eigen_hidden);
            Eigen::Map<const RowMatrixXf> grad_approx_in(grad_approx_data + base_offset, eigen_nodes, eigen_hidden);
            Eigen::Map<const RowMatrixXf> grad_detail_in(grad_detail_data + base_offset, eigen_nodes, eigen_hidden);

            static thread_local QwtScratch scratch;
            scratch.resize(eigen_nodes, eigen_hidden);

            // Reconstruct Hamiltonian from detail coefficients for this batch item.
            SparseMatrix<float> h_sparse(eigen_nodes, eigen_nodes);
            BuildHamiltonianFromDetail(detail_map, epsilon_, &h_sparse);

            // Phase 17: Jacobi Preconditioner and Skip Connections in Gradient Pass
            if (use_jacobi_preconditioner_) {
              saguaro::ops::qwt::qwt_extract_diagonal_inverse(h_sparse, scratch.jacobi_diag_inv);
            }
            if (skip_stride_ > 0) {
              std::vector<float> node_energies(eigen_nodes);
              saguaro::ops::qwt::qwt_compute_node_energies(detail_map.data(), node_energies.data(), 
                                                           eigen_nodes, eigen_hidden, epsilon_);
              saguaro::ops::qwt::qwt_compute_skip_connections(node_energies, eigen_nodes,
                                                              skip_stride_, max_skips_per_node_, scratch.skip_triplets);
              h_sparse.setFromTriplets(scratch.skip_triplets.begin(), scratch.skip_triplets.end());
            }

            // Re-run forward pass to capture intermediate step features for backward pass.
            IntegratorContext ctx_fwd;
            std::vector<MatrixXf> step_features;
            MatrixXf forward_out(eigen_nodes, eigen_hidden);
            RunCayleyEvolution(h_sparse,
                               approx_map,
                               time_values[b],
                               &forward_out,
                               &ctx_fwd,
                               &step_features,
                               ctqw_overrides[b],
                               pade_order_,
                               use_jacobi_preconditioner_ ? &scratch.jacobi_diag_inv : nullptr);

            // Solvers for the backward pass through Cayley evolution.
            Eigen::BiCGSTAB<SparseMatrix<float>> solver;
            Eigen::BiCGSTAB<SparseMatrix<float>> solver_transpose;
            solver.compute(ctx_fwd.system_matrix);
            SparseMatrix<float> system_matrix_transpose = ctx_fwd.system_matrix.transpose();
            system_matrix_transpose.makeCompressed();
            solver_transpose.compute(system_matrix_transpose);

            MatrixXf grad_current = grad_qwt;
            const float inv_steps = (ctx_fwd.num_steps > 0) ? 1.0f / static_cast<float>(ctx_fwd.num_steps) : 1.0f;

            // Phase 17: Generalized Adjoint for Padé[m/m]
            std::vector<SparseMatrix<float>> h_powers_t;
            if (pade_order_ > 1) {
              saguaro::ops::qwt::qwt_precompute_h_powers(h_sparse.transpose(), pade_order_, h_powers_t);
            }
            const auto& pade_coeffs = saguaro::ops::qwt::GetPadeCoefficients(pade_order_);

            for (int step = ctx_fwd.num_steps - 1; step >= 0; --step) {
                const MatrixXf& before = step_features[step];
                const MatrixXf& after = step_features[step + 1];
                MatrixXf grad_output = grad_current;

                // Step A: Solve for grad_rhs: q(H)^T * grad_rhs = grad_output
                MatrixXf precond_grad_output;
                if (use_jacobi_preconditioner_) {
                  saguaro::ops::qwt::qwt_apply_jacobi_preconditioner_matrix(scratch.jacobi_diag_inv, grad_output, precond_grad_output);
                } else {
                  precond_grad_output = grad_output;
                }

                MatrixXf grad_rhs = solver_transpose.solve(precond_grad_output);
                if (solver_transpose.info() != Eigen::Success) {
                  grad_rhs.setZero();
                }

                // Step B: grad_input_step = p(H)^T * grad_rhs
                if (pade_order_ <= 1) {
                  grad_current = grad_rhs + ctx_fwd.alpha * (h_sparse.transpose() * grad_rhs);
                } else {
                  SparseMatrix<float> num_mat_t = saguaro::ops::qwt::qwt_build_pade_numerator(h_sparse.transpose(), ctx_fwd.alpha, pade_coeffs, &h_powers_t);
                  grad_current = num_mat_t * grad_rhs;
                }

                // Step C: Accumulate gradient w.r.t evolution time
                if (ctx_fwd.local_step > 0.0f) {
                  // Sensitivity for Padé is more complex; here we use an approximation
                  // based on the first-order term which dominates for small alpha
                  MatrixXf h_x = h_sparse * before;
                  MatrixXf h_y = h_sparse * after;
                  MatrixXf sensitivity_rhs = h_x + h_y;
                  MatrixXf sensitivity = solver.solve(sensitivity_rhs);
                  if (solver.info() == Eigen::Success) {
                    const float grad_step = 0.5f * (grad_output.cwiseProduct(sensitivity)).sum();
                    grad_time[b] += grad_step * inv_steps;
                  }
                }
            }

            // Combine gradients from QWT embeddings and direct approx/detail gradients.
            RowMatrixXf grad_current_row = grad_current;
            RowMatrixXf grad_approx_in_row = grad_approx_in;
            RowMatrixXf grad_detail_in_row = grad_detail_in;
            RowMatrixXf grad_approx_total = grad_current_row + grad_approx_in_row;

            // Backward pass through the wavelet transform.
            // Distributes gradients from approx/detail coefficients back to input_data and filters.
            for (int64 m = 0; m < nodes; ++m) {
                const int64 base_index = m * 2;
                for (int64 d = 0; d < hidden; ++d) {
                    const float gA = grad_approx_total(m, d);
                    const float gD = grad_detail_in_row(m, d);
                    if (gA == 0.0f && gD == 0.0f) {
                        continue;
                    }
                    for (int64 k = 0; k < filter_width; ++k) {
                        const int64 src = base_index + k;
                        if (src >= seq_len) {
                            break;
                        }
                        if (!mask(b, src)) {
                            continue;
                        }
                        // Gradient to input_data
                        grad_input(b, src, d) += gA * low_filter(k, 0, d) + gD * high_filter(k, 0, d);
                        const float sample = input(b, src, d);
                        // Gradient to low/high pass filters
                        local_grad_low(k, 0, d) += gA * sample;
                        local_grad_high(k, 0, d) += gD * sample;
                    }
                }
            }
        }

        // Aggregate thread-local filter gradients into global tensors.
        {
            absl::MutexLock l(&mu_);
            auto local_low_flat = local_grad_low_tensor.flat<float>();
            auto local_high_flat = local_grad_high_tensor.flat<float>();
            auto global_low_flat = grad_low_tensor->flat<float>();
            auto global_high_flat = grad_high_tensor->flat<float>();
            for (int64 i = 0; i < local_grad_low_tensor.NumElements(); ++i) {
                global_low_flat(i) += local_low_flat(i);
                global_high_flat(i) += local_high_flat(i);
            }
        }
    };

    const std::size_t cost_per_unit = static_cast<std::size_t>(nodes * hidden * 1000);
    saguaro::parallel::ForShard(
        static_cast<std::size_t>(batch),
        cost_per_unit,
        work);

    // Finalize gradient with respect to evolution_time.
    if (evolution_time_tensor.dims() == 0) {
      float total = 0.0f;
      for (float v : grad_time) total += v * ((evolution_time_tensor.scalar<float>()() >= 0.0f) ? 1.0f : -1.0f);
      grad_time_tensor->scalar<float>()() = total;
    } else {
      auto grad_vec = grad_time_tensor->flat<float>();
      for (int64 b = 0; b < batch; ++b) {
        const float sign = (time_values[b] >= 0.0f) ? 1.0f : -1.0f;
        grad_vec(b) = grad_time[b] * sign;
      }
    }
  }

 private:
  float epsilon_;
  int num_wavelet_levels_;  // Phase 10.3: Number of cascaded DWT levels
  // Phase 17: Enhancement attributes
  bool use_lifting_scheme_;
  int pade_order_;
  bool use_jacobi_preconditioner_;
  int skip_stride_;
  int max_skips_per_node_;
  absl::Mutex mu_;
};

REGISTER_KERNEL_BUILDER(Name("FusedQwtTokenizer").Device(DEVICE_CPU), FusedQwtTokenizerOp);
REGISTER_KERNEL_BUILDER(Name("FusedQwtTokenizerGrad").Device(DEVICE_CPU), FusedQwtTokenizerGradOp);

}  // namespace tensorflow
