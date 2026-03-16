#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/parallel/parallel_backend.h"
#include "common/perf_utils.h"
#include "absl/synchronization/mutex.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor" // Include for Eigen::Tensor and Eigen::TensorMap
#include "ops/fused_reasoning_stack/fused_reasoning_stack_kernel.h"
#include "ops/n4sid_solver.h"
#include "ops/train_step_op.h"
#include "ops/quantum_training_config.h"  // Quantum training enhancements
#include "tensorflow/core/platform/logging.h"
#include <cstring>
#include <cmath>

// Phase 11 SIMD compliance: Explicit SIMD guards
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace tensorflow {

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::Map;
using Eigen::RowMajor;

// Phase 11 SIMD compliance: Vectorized SophiaG optimizer update kernel
// Implements: m_new = beta_1 * m + (1 - beta_1) * grad
//             update = m_new / max(h, epsilon)
//             clipped_update = clamp(update, -rho, rho)
//             var_new = var - lr * clipped_update
inline void VectorizedSophiaGUpdate(const float* grad, float* var, float* m, const float* h,
                                    int64_t size, float lr, float beta_1, float rho, float epsilon) {
    int64_t i = 0;
    const float one_minus_beta_1 = 1.0f - beta_1;

#if defined(__AVX512F__)
    const __m512 beta_1_vec = _mm512_set1_ps(beta_1);
    const __m512 one_minus_beta_1_vec = _mm512_set1_ps(one_minus_beta_1);
    const __m512 epsilon_vec = _mm512_set1_ps(epsilon);
    const __m512 lr_vec = _mm512_set1_ps(lr);
    const __m512 rho_vec = _mm512_set1_ps(rho);
    const __m512 neg_rho_vec = _mm512_set1_ps(-rho);

    for (; i + 16 <= size; i += 16) {
        __m512 grad_vec = _mm512_loadu_ps(&grad[i]);
        __m512 m_vec = _mm512_loadu_ps(&m[i]);
        __m512 h_vec = _mm512_loadu_ps(&h[i]);
        __m512 var_vec = _mm512_loadu_ps(&var[i]);

        // m_new = beta_1 * m + (1 - beta_1) * grad (FMA)
        __m512 m_new = _mm512_fmadd_ps(beta_1_vec, m_vec, _mm512_mul_ps(one_minus_beta_1_vec, grad_vec));

        // update = m_new / max(h, epsilon)
        __m512 h_max = _mm512_max_ps(h_vec, epsilon_vec);
        __m512 update = _mm512_div_ps(m_new, h_max);

        // clipped_update = clamp(update, -rho, rho)
        __m512 clipped = _mm512_min_ps(_mm512_max_ps(update, neg_rho_vec), rho_vec);

        // var_new = var - lr * clipped_update (FNMADD: var - lr * clipped)
        __m512 var_new = _mm512_fnmadd_ps(lr_vec, clipped, var_vec);

        _mm512_storeu_ps(&m[i], m_new);
        _mm512_storeu_ps(&var[i], var_new);
    }
#elif defined(__AVX2__)
    const __m256 beta_1_vec = _mm256_set1_ps(beta_1);
    const __m256 one_minus_beta_1_vec = _mm256_set1_ps(one_minus_beta_1);
    const __m256 epsilon_vec = _mm256_set1_ps(epsilon);
    const __m256 lr_vec = _mm256_set1_ps(lr);
    const __m256 rho_vec = _mm256_set1_ps(rho);
    const __m256 neg_rho_vec = _mm256_set1_ps(-rho);

    for (; i + 8 <= size; i += 8) {
        __m256 grad_vec = _mm256_loadu_ps(&grad[i]);
        __m256 m_vec = _mm256_loadu_ps(&m[i]);
        __m256 h_vec = _mm256_loadu_ps(&h[i]);
        __m256 var_vec = _mm256_loadu_ps(&var[i]);

        // m_new = beta_1 * m + (1 - beta_1) * grad (FMA)
        __m256 m_new = _mm256_fmadd_ps(beta_1_vec, m_vec, _mm256_mul_ps(one_minus_beta_1_vec, grad_vec));

        // update = m_new / max(h, epsilon)
        __m256 h_max = _mm256_max_ps(h_vec, epsilon_vec);
        __m256 update = _mm256_div_ps(m_new, h_max);

        // clipped_update = clamp(update, -rho, rho)
        __m256 clipped = _mm256_min_ps(_mm256_max_ps(update, neg_rho_vec), rho_vec);

        // var_new = var - lr * clipped_update (FNMADD: var - lr * clipped)
        __m256 var_new = _mm256_fnmadd_ps(lr_vec, clipped, var_vec);

        _mm256_storeu_ps(&m[i], m_new);
        _mm256_storeu_ps(&var[i], var_new);
    }
#elif defined(__ARM_NEON)
    const float32x4_t beta_1_vec = vdupq_n_f32(beta_1);
    const float32x4_t one_minus_beta_1_vec = vdupq_n_f32(one_minus_beta_1);
    const float32x4_t epsilon_vec = vdupq_n_f32(epsilon);
    const float32x4_t lr_vec = vdupq_n_f32(lr);
    const float32x4_t rho_vec = vdupq_n_f32(rho);
    const float32x4_t neg_rho_vec = vdupq_n_f32(-rho);

    for (; i + 4 <= size; i += 4) {
        float32x4_t grad_vec = vld1q_f32(&grad[i]);
        float32x4_t m_vec = vld1q_f32(&m[i]);
        float32x4_t h_vec = vld1q_f32(&h[i]);
        float32x4_t var_vec = vld1q_f32(&var[i]);

        // m_new = beta_1 * m + (1 - beta_1) * grad (FMA)
        float32x4_t m_new = vfmaq_f32(vmulq_f32(one_minus_beta_1_vec, grad_vec), beta_1_vec, m_vec);

        // update = m_new / max(h, epsilon)
        float32x4_t h_max = vmaxq_f32(h_vec, epsilon_vec);
        float32x4_t update = vdivq_f32(m_new, h_max);

        // clipped_update = clamp(update, -rho, rho)
        float32x4_t clipped = vminq_f32(vmaxq_f32(update, neg_rho_vec), rho_vec);

        // var_new = var - lr * clipped_update (FNMSUB: var - lr * clipped)
        float32x4_t var_new = vfmsq_f32(var_vec, lr_vec, clipped);

        vst1q_f32(&m[i], m_new);
        vst1q_f32(&var[i], var_new);
    }
#endif

    // Scalar fallback
    for (; i < size; ++i) {
        m[i] = beta_1 * m[i] + one_minus_beta_1 * grad[i];
        float update = m[i] / std::max(h[i], epsilon);
        float clipped_update = std::min(std::max(update, -rho), rho);
        var[i] -= lr * clipped_update;
    }
}

// SophiaG optimizer helper functions definitions (Full version)
void SophiaG_ApplyGradients(OpKernelContext* context,
                                const Tensor& grad,
                                Tensor* var,
                                Tensor* m,
                                Tensor* h,
                                float lr,
                                float beta_1_t,
                                float rho_t,
                                float epsilon) {
    auto grad_flat = grad.flat<float>();
    auto var_flat = var->flat<float>();
    auto m_flat = m->flat<float>();
    auto h_flat = h->flat<float>();

    const int64 num_elements = grad_flat.size();
    saguaro::parallel::ForShard(
        static_cast<std::size_t>(num_elements),
        static_cast<std::size_t>(1000),
        [&](int64 start, int64 end) {
        VectorizedSophiaGUpdate(&grad_flat(start), &var_flat(start), &m_flat(start),
                                &h_flat(start), end - start, lr, beta_1_t, rho_t, epsilon);
    });
}

// Phase 11 SIMD compliance: Vectorized Hessian update kernel
// Implements: h_new = beta_2 * h + (1 - beta_2) * grad^2
inline void VectorizedHessianUpdate(const float* h_grad, float* h, int64_t size,
                                    float beta_2) {
    int64_t i = 0;
    const float one_minus_beta_2 = 1.0f - beta_2;

#if defined(__AVX512F__)
    const __m512 beta_2_vec = _mm512_set1_ps(beta_2);
    const __m512 one_minus_beta_2_vec = _mm512_set1_ps(one_minus_beta_2);

    for (; i + 16 <= size; i += 16) {
        __m512 h_grad_vec = _mm512_loadu_ps(&h_grad[i]);
        __m512 h_vec = _mm512_loadu_ps(&h[i]);

        // h_hat = h_grad * h_grad
        __m512 h_hat = _mm512_mul_ps(h_grad_vec, h_grad_vec);

        // h_new = beta_2 * h + (1 - beta_2) * h_hat (FMA)
        __m512 h_new = _mm512_fmadd_ps(beta_2_vec, h_vec, _mm512_mul_ps(one_minus_beta_2_vec, h_hat));

        _mm512_storeu_ps(&h[i], h_new);
    }
#elif defined(__AVX2__)
    const __m256 beta_2_vec = _mm256_set1_ps(beta_2);
    const __m256 one_minus_beta_2_vec = _mm256_set1_ps(one_minus_beta_2);

    for (; i + 8 <= size; i += 8) {
        __m256 h_grad_vec = _mm256_loadu_ps(&h_grad[i]);
        __m256 h_vec = _mm256_loadu_ps(&h[i]);

        // h_hat = h_grad * h_grad
        __m256 h_hat = _mm256_mul_ps(h_grad_vec, h_grad_vec);

        // h_new = beta_2 * h + (1 - beta_2) * h_hat (FMA)
        __m256 h_new = _mm256_fmadd_ps(beta_2_vec, h_vec, _mm256_mul_ps(one_minus_beta_2_vec, h_hat));

        _mm256_storeu_ps(&h[i], h_new);
    }
#elif defined(__ARM_NEON)
    const float32x4_t beta_2_vec = vdupq_n_f32(beta_2);
    const float32x4_t one_minus_beta_2_vec = vdupq_n_f32(one_minus_beta_2);

    for (; i + 4 <= size; i += 4) {
        float32x4_t h_grad_vec = vld1q_f32(&h_grad[i]);
        float32x4_t h_vec = vld1q_f32(&h[i]);

        // h_hat = h_grad * h_grad
        float32x4_t h_hat = vmulq_f32(h_grad_vec, h_grad_vec);

        // h_new = beta_2 * h + (1 - beta_2) * h_hat (FMA)
        float32x4_t h_new = vfmaq_f32(vmulq_f32(one_minus_beta_2_vec, h_hat), beta_2_vec, h_vec);

        vst1q_f32(&h[i], h_new);
    }
#endif

    // Scalar fallback
    for (; i < size; ++i) {
        float h_hat = h_grad[i] * h_grad[i];
        h[i] = beta_2 * h[i] + one_minus_beta_2 * h_hat;
    }
}

void SophiaG_UpdateHessian(OpKernelContext* context,
                               const Tensor& h_grad,
                               Tensor* h,
                               float beta_2_t) {
    auto h_grad_flat = h_grad.flat<float>();
    auto h_flat = h->flat<float>();
    const int64 num_elements = h_grad_flat.size();

    saguaro::parallel::ForShard(
        static_cast<std::size_t>(num_elements),
        static_cast<std::size_t>(1000),
        [&](int64 start, int64 end) {
        VectorizedHessianUpdate(&h_grad_flat(start), &h_flat(start), end - start, beta_2_t);
    });
}

REGISTER_OP("TrainStep")
    .Input("context_tokens: int32")           // Token IDs for the batch (batch_size x seq_len)
    .Input("target_tokens: int32")            // Target token IDs (batch_size x seq_len)
    .Input("batch_size_val: int64")           // Actual batch size for this step
    .Input("seq_len_val: int64")              // Sequence length for this batch
    .Input("input_dim_val: int64")            // Input dimension for sequence_input
    .Input("label_dim_val: int64")            // Label dimension (e.g., 1)
    .Input("model_weights: M * float")
    .Input("optimizer_state: O * float")
    .Input("ewc_fisher: F * float")
    .Input("ewc_optimal_weights: F * float")
    .Input("ewc_lambda: float")
    .Input("k_val: int32")
    .Input("global_step: int64")
    .Input("n4sid_order: int32")
    .Input("block_types: string")
    .Input("block_weight_counts: int32")
    .Input("block_descriptors: string")
    .Input("initial_float_states: N * float")
    .Input("initial_int_states: P * int32")
    .Input("n4sid_u: float") // Input data for N4SID
    .Input("n4sid_y: float") // Output data for N4SID
    .Input("lr: float")
    .Input("beta_1: float")
    .Input("rho: float")
    .Input("epsilon: float")
    .Input("beta_2: float") // New input for beta_2
    .Input("loss_mask: float") // Optional per-token loss mask
    // Quantum Training Config (Phase T1-T6)
    .Input("enable_qng: bool")              // Enable Quantum Natural Gradient
    .Input("qng_damping: float")            // QFIM regularization damping
    .Input("qng_ema_decay: float")          // EMA decay for QFIM update
    .Input("enable_barren_plateau: bool")   // Enable barren plateau monitor
    .Input("barren_plateau_threshold: float") // Detection threshold
    .Input("barren_plateau_lr_scale: float") // LR scaling factor for mitigation
    .Attr("M: int >= 1")
    .Attr("O: int >= 0")
    .Attr("F: int >= 0")
    .Attr("N: int >= 0")
    .Attr("P: int >= 0")
    .Attr("num_n4sid_inputs: int >= 0")
    .Attr("num_n4sid_outputs: int >= 0")
    .Attr("NumN4SIDMatrices: int >= 1")
    .Output("new_model_weights: M * float")
    .Output("new_optimizer_state: O * float")
    .Output("total_loss: float")
    .Output("std_loss: float")
    .Output("ewc_loss: float")
    .Output("gradient_norm: float")
    .Output("logits: float")
    .Output("n4sid_matrices: NumN4SIDMatrices * float")
    .Output("final_float_states_out: N * float")
    .Output("final_int_states_out: P * int32")
    .Output("grad_initial_float_states_out: N * float")
    .Output("grad_initial_int_states_out: P * int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int m, o, f, n, p, num_n4sid_inputs, num_n4sid_outputs, num_n4sid_matrices;
        TF_RETURN_IF_ERROR(c->GetAttr("M", &m));
        TF_RETURN_IF_ERROR(c->GetAttr("O", &o));
        TF_RETURN_IF_ERROR(c->GetAttr("F", &f));
        TF_RETURN_IF_ERROR(c->GetAttr("N", &n));
        TF_RETURN_IF_ERROR(c->GetAttr("P", &p));
        TF_RETURN_IF_ERROR(c->GetAttr("num_n4sid_inputs", &num_n4sid_inputs));
        TF_RETURN_IF_ERROR(c->GetAttr("num_n4sid_outputs", &num_n4sid_outputs));
        TF_RETURN_IF_ERROR(c->GetAttr("NumN4SIDMatrices", &num_n4sid_matrices));

        for (int i = 0; i < m; ++i) {
            c->set_output(i, c->input(i + 6)); 
        }
        for (int i = 0; i < o; ++i) {
            c->set_output(i + m, c->input(i + 6 + m));
        }
        c->set_output(m + o, c->Scalar()); 
        c->set_output(m + o + 1, c->Scalar()); 
        c->set_output(m + o + 2, c->Scalar()); 
        c->set_output(m + o + 3, c->Scalar()); 
        c->set_output(m + o + 4, c->UnknownShapeOfRank(3)); // logits

        // n4sid_matrices (num_n4sid_matrices outputs)
        for (int i = 0; i < num_n4sid_matrices; ++i) {
            c->set_output(m + o + 5 + i, c->UnknownShape()); 
        }

        for (int i = 0; i < n; ++i) {
            c->set_output(m + o + 5 + num_n4sid_matrices + i, c->UnknownShape()); // final_float_states_out
        }
        for (int i = 0; i < p; ++i) {
            c->set_output(m + o + 5 + num_n4sid_matrices + n + i, c->UnknownShape()); // final_int_states_out
        }
        for (int i = 0; i < n; ++i) {
            c->set_output(m + o + 5 + num_n4sid_matrices + n + p + i, c->UnknownShape()); // grad_initial_float_states_out
        }
        for (int i = 0; i < p; ++i) {
            c->set_output(m + o + 5 + num_n4sid_matrices + n + p + n + i, c->UnknownShape()); // grad_initial_int_states_out
        }
        return OkStatus();
    });

// TrainStepOp class definition
TrainStepOp::TrainStepOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("M", &m_));
    OP_REQUIRES_OK(context, context->GetAttr("O", &o_));
    OP_REQUIRES_OK(context, context->GetAttr("F", &f_));
    OP_REQUIRES_OK(context, context->GetAttr("N", &n_));
    OP_REQUIRES_OK(context, context->GetAttr("P", &p_));
    OP_REQUIRES(context, o_ == 2 * m_,
                errors::InvalidArgument(
                    "TrainStepOp expects optimizer_state to contain exactly two "
                    "slots (momentum, hessian) per model weight. Received ",
                    o_, " optimizer_state tensors for ", m_,
                    " weights. Ensure _prepare_trainable_lists installs missing "
                    "SophiaG slots before invoking the custom op."));
    OP_REQUIRES_OK(context, context->GetAttr("num_n4sid_inputs", &num_n4sid_inputs_));
    OP_REQUIRES_OK(context, context->GetAttr("num_n4sid_outputs", &num_n4sid_outputs_));
    OP_REQUIRES_OK(context, context->GetAttr("NumN4SIDMatrices", &num_n4sid_matrices_));
}

void TrainStepOp::Compute(OpKernelContext* context) {
    const Tensor& context_tokens_tensor = context->input(0);
    const Tensor& target_tokens_tensor = context->input(1);
    const int64_t batch_size = context->input(2).scalar<int64>()();
    const int64_t seq_len = context->input(3).scalar<int64>()();
    const int64_t input_dim = context->input(4).scalar<int64>()();
    const int64_t label_dim = context->input(5).scalar<int64>()();

    OP_REQUIRES(context, context_tokens_tensor.dtype() == DT_INT32,
                errors::InvalidArgument("context_tokens must be int32."));
    OP_REQUIRES(context, target_tokens_tensor.dtype() == DT_INT32,
                errors::InvalidArgument("target_tokens must be int32."));
    OP_REQUIRES(context, context_tokens_tensor.dims() == 2,
                errors::InvalidArgument("context_tokens must be rank-2 [batch, seq]."));
    OP_REQUIRES(context, target_tokens_tensor.dims() == 2,
                errors::InvalidArgument("target_tokens must be rank-2 [batch, seq]."));
    OP_REQUIRES(context, context_tokens_tensor.dim_size(0) == batch_size,
                errors::InvalidArgument("context_tokens batch dimension mismatch."));
    OP_REQUIRES(context, target_tokens_tensor.dim_size(0) == batch_size,
                errors::InvalidArgument("target_tokens batch dimension mismatch."));
    OP_REQUIRES(context, context_tokens_tensor.dim_size(1) == seq_len,
                errors::InvalidArgument("context_tokens sequence length mismatch."));
    OP_REQUIRES(context, target_tokens_tensor.dim_size(1) == seq_len,
                errors::InvalidArgument("target_tokens sequence length mismatch."));

    OpInputList model_weights;
    OP_REQUIRES_OK(context, context->input_list("model_weights", &model_weights));
    for (int i = 0; i < model_weights.size(); ++i) {
        OP_REQUIRES(context, model_weights[i].dtype() == DT_FLOAT, errors::InvalidArgument("model_weights must be float."));
    }

    OpInputList optimizer_state;
    OP_REQUIRES_OK(context, context->input_list("optimizer_state", &optimizer_state));
    for (int i = 0; i < optimizer_state.size(); ++i) {
        OP_REQUIRES(context, optimizer_state[i].dtype() == DT_FLOAT, errors::InvalidArgument("optimizer_state must be float."));
    }

    OpInputList ewc_fisher_list;
    OP_REQUIRES_OK(context, context->input_list("ewc_fisher", &ewc_fisher_list));
    for (int i = 0; i < ewc_fisher_list.size(); ++i) {
        OP_REQUIRES(context, ewc_fisher_list[i].dtype() == DT_FLOAT, errors::InvalidArgument("ewc_fisher must be float."));
    }

    OpInputList ewc_optimal_weights_list;
    OP_REQUIRES_OK(context, context->input_list("ewc_optimal_weights", &ewc_optimal_weights_list));
    for (int i = 0; i < ewc_optimal_weights_list.size(); ++i) {
        OP_REQUIRES(context, ewc_optimal_weights_list[i].dtype() == DT_FLOAT, errors::InvalidArgument("ewc_optimal_weights must be float."));
    }
    const int32* sequence_input_data = context_tokens_tensor.flat<int32>().data();
    const int32* labels_data = target_tokens_tensor.flat<int32>().data();

    Eigen::Map<const Eigen::Matrix<int32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> sequence_input_map(
        sequence_input_data, batch_size, seq_len);

    Eigen::Map<const Eigen::Matrix<int32, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> labels_map(
        labels_data, batch_size, seq_len);
    const auto& labels_tensor = labels_map;

    OP_REQUIRES(context, m_ > 0, errors::InvalidArgument("Model weights list is empty."));
    const Tensor& embedding_weights_tensor = model_weights[0];
    OP_REQUIRES(context, embedding_weights_tensor.dims() == 2,
                errors::InvalidArgument("Embedding weights tensor must be 2-dimensional."));

    const int64 vocab_size = embedding_weights_tensor.dim_size(0);
    const int64 embedding_dim = embedding_weights_tensor.dim_size(1);
    OP_REQUIRES(context, embedding_dim == input_dim,
                errors::InvalidArgument("Embedding dimension mismatch."));

    LOG(INFO) << "[TrainStepOp] Starting compute. batch_size=" << batch_size
              << " seq_len=" << seq_len << " embedding_dim=" << embedding_dim;

    Tensor embedded_sequence_input_tensor(DT_FLOAT, TensorShape({batch_size, seq_len, embedding_dim}));
    float* embedded_sequence_input_base = embedded_sequence_input_tensor.flat<float>().data();
    const float* embedding_weights_base = embedding_weights_tensor.flat<float>().data();

    const int64 total_positions = batch_size * seq_len;
    OP_REQUIRES(context, batch_size == 0 || total_positions / seq_len == batch_size,
                errors::InvalidArgument("batch_size * seq_len overflow."));

    const std::size_t copy_elems = static_cast<std::size_t>(embedding_dim);
    for (int64 pos = 0; pos < total_positions; ++pos) {
        const int64 batch_idx = pos / seq_len;
        const int64 seq_idx = pos % seq_len;
        int32 token_id = sequence_input_map(batch_idx, seq_idx);
        OP_REQUIRES(context, token_id >= 0 && token_id < vocab_size,
                    errors::InvalidArgument("Token ID out of bounds."));

        const float* src = embedding_weights_base + static_cast<int64>(token_id) * embedding_dim;
        float* dst = embedded_sequence_input_base + static_cast<int64>(pos) * embedding_dim;
        saguaro::ops::PrefetchL1(src + embedding_dim);
        saguaro::ops::CopySpan(dst, src, copy_elems);
    }
    const auto& sequence_input_tensor = embedded_sequence_input_tensor;
    LOG(INFO) << "[TrainStepOp] Embedded tokens prepared.";

    int current_input_idx = 6 + m_ + o_ + f_ + f_;

    const Tensor& ewc_lambda_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, ewc_lambda_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("ewc_lambda must be float."));
    const float ewc_lambda = ewc_lambda_tensor.scalar<float>()();

    const Tensor& k_val_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, k_val_tensor.dtype() == DT_INT32, errors::InvalidArgument("k_val must be int32."));
    const int k_val = k_val_tensor.scalar<int32>()();

    const Tensor& global_step_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, global_step_tensor.dtype() == DT_INT64, errors::InvalidArgument("global_step must be int64."));
    const int64 global_step = global_step_tensor.scalar<int64>()();

    const Tensor& n4sid_order_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, n4sid_order_tensor.dtype() == DT_INT32, errors::InvalidArgument("n4sid_order must be int32."));
    const int n4sid_order = n4sid_order_tensor.scalar<int32>()();

    const Tensor& block_types_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, block_types_tensor.dtype() == DT_STRING, errors::InvalidArgument("block_types must be string."));
    const Tensor& block_weight_counts_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, block_weight_counts_tensor.dtype() == DT_INT32, errors::InvalidArgument("block_weight_counts must be int32."));
    const Tensor& block_descriptors_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, block_descriptors_tensor.dtype() == DT_STRING, errors::InvalidArgument("block_descriptors must be string."));

    OpInputList initial_float_states;
    OP_REQUIRES_OK(context, context->input_list("initial_float_states", &initial_float_states));
    for (int i = 0; i < initial_float_states.size(); ++i) {
        OP_REQUIRES(context, initial_float_states[i].dtype() == DT_FLOAT, errors::InvalidArgument("initial_float_states must be float."));
    }

    OpInputList initial_int_states;
    OP_REQUIRES_OK(context, context->input_list("initial_int_states", &initial_int_states));
    for (int i = 0; i < initial_int_states.size(); ++i) {
        OP_REQUIRES(context, initial_int_states[i].dtype() == DT_INT32, errors::InvalidArgument("initial_int_states must be int32."));
    }

    // Advance the positional index past the initial state inputs that were materialized above.
    current_input_idx += initial_float_states.size();
    current_input_idx += initial_int_states.size();

    const Tensor& n4sid_u_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, n4sid_u_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("n4sid_u must be float."));
    const Tensor& n4sid_y_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, n4sid_y_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("n4sid_y must be float."));

    const Tensor& lr_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, lr_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("lr must be float."));
    const float lr = lr_tensor.scalar<float>()();

    const Tensor& beta_1_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, beta_1_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("beta_1 must be float."));
    const float beta_1_t = beta_1_tensor.scalar<float>()();

    const Tensor& rho_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, rho_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("rho must be float."));
    const float rho_t = rho_tensor.scalar<float>()();

    const Tensor& epsilon_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, epsilon_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("epsilon must be float."));
    const float epsilon = epsilon_tensor.scalar<float>()();

    const Tensor& beta_2_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, beta_2_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("beta_2 must be float."));
    const float beta_2_t = beta_2_tensor.scalar<float>()();

    const Tensor& loss_mask_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, loss_mask_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("loss_mask must be float."));
    OP_REQUIRES(context, loss_mask_tensor.dims() == 2,
                errors::InvalidArgument("loss_mask must be rank-2 [batch, seq]."));
    OP_REQUIRES(context, loss_mask_tensor.dim_size(0) == batch_size,
                errors::InvalidArgument("loss_mask batch dimension mismatch."));
    OP_REQUIRES(context, loss_mask_tensor.dim_size(1) == seq_len,
                errors::InvalidArgument("loss_mask sequence length mismatch."));
    const float* loss_mask_data = loss_mask_tensor.flat<float>().data();

    // =========================================================================
    // Quantum Training Config (Phase T1-T6)
    // =========================================================================
    const Tensor& enable_qng_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, enable_qng_tensor.dtype() == DT_BOOL, errors::InvalidArgument("enable_qng must be bool."));
    const bool enable_qng = enable_qng_tensor.scalar<bool>()();

    const Tensor& qng_damping_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, qng_damping_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("qng_damping must be float."));
    const float qng_damping = qng_damping_tensor.scalar<float>()();

    const Tensor& qng_ema_decay_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, qng_ema_decay_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("qng_ema_decay must be float."));
    const float qng_ema_decay = qng_ema_decay_tensor.scalar<float>()();

    const Tensor& enable_barren_plateau_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, enable_barren_plateau_tensor.dtype() == DT_BOOL, errors::InvalidArgument("enable_barren_plateau must be bool."));
    const bool enable_barren_plateau = enable_barren_plateau_tensor.scalar<bool>()();

    const Tensor& barren_plateau_threshold_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, barren_plateau_threshold_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("barren_plateau_threshold must be float."));
    const float barren_plateau_threshold = barren_plateau_threshold_tensor.scalar<float>()();

    const Tensor& barren_plateau_lr_scale_tensor = context->input(current_input_idx++);
    OP_REQUIRES(context, barren_plateau_lr_scale_tensor.dtype() == DT_FLOAT, errors::InvalidArgument("barren_plateau_lr_scale must be float."));
    const float barren_plateau_lr_scale = barren_plateau_lr_scale_tensor.scalar<float>()();

    // Log quantum training config if enabled
    if (enable_qng || enable_barren_plateau) {
        LOG(INFO) << "[TrainStepOp] Quantum Training: QNG=" << (enable_qng ? "ON" : "OFF")
                  << ", BarrenPlateau=" << (enable_barren_plateau ? "ON" : "OFF");
    }

    // Phase 11 SIMD compliance: Vectorized EWC penalty computation
    // Implements: penalty = sum(fisher * (var - optimal)^2)
    auto VectorizedEWCPenalty = [](const float* var, const float* fisher,
                                   const float* optimal, int64_t size) -> float {
        int64_t i = 0;
        float penalty = 0.0f;

#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        for (; i + 16 <= size; i += 16) {
            __m512 var_vec = _mm512_loadu_ps(&var[i]);
            __m512 optimal_vec = _mm512_loadu_ps(&optimal[i]);
            __m512 fisher_vec = _mm512_loadu_ps(&fisher[i]);

            // diff = var - optimal
            __m512 diff = _mm512_sub_ps(var_vec, optimal_vec);

            // fisher * diff * diff (FMA: fisher * diff^2)
            __m512 contrib = _mm512_mul_ps(fisher_vec, _mm512_mul_ps(diff, diff));
            acc = _mm512_add_ps(acc, contrib);
        }
        // Horizontal sum for AVX512
        penalty = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8) {
            __m256 var_vec = _mm256_loadu_ps(&var[i]);
            __m256 optimal_vec = _mm256_loadu_ps(&optimal[i]);
            __m256 fisher_vec = _mm256_loadu_ps(&fisher[i]);

            // diff = var - optimal
            __m256 diff = _mm256_sub_ps(var_vec, optimal_vec);

            // fisher * diff * diff (FMA: fisher * diff^2)
            __m256 contrib = _mm256_mul_ps(fisher_vec, _mm256_mul_ps(diff, diff));
            acc = _mm256_add_ps(acc, contrib);
        }
        // Horizontal sum for AVX2
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        penalty = _mm_cvtss_f32(sum);
#elif defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; i + 4 <= size; i += 4) {
            float32x4_t var_vec = vld1q_f32(&var[i]);
            float32x4_t optimal_vec = vld1q_f32(&optimal[i]);
            float32x4_t fisher_vec = vld1q_f32(&fisher[i]);

            // diff = var - optimal
            float32x4_t diff = vsubq_f32(var_vec, optimal_vec);

            // fisher * diff * diff
            float32x4_t contrib = vmulq_f32(fisher_vec, vmulq_f32(diff, diff));
            acc = vaddq_f32(acc, contrib);
        }
        // Horizontal sum for NEON
        float32x2_t sum = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        sum = vpadd_f32(sum, sum);
        penalty = vget_lane_f32(sum, 0);
#endif

        // Scalar fallback
        for (; i < size; ++i) {
            float diff = var[i] - optimal[i];
            penalty += fisher[i] * diff * diff;
        }
        return penalty;
    };

    float ewc_penalty = 0.0f;
    if (ewc_lambda > 0.0f && f_ > 0) {
        for (int i = 0; i < m_; ++i) {
            const Tensor& var = model_weights[i];
            const Tensor& fisher_info = ewc_fisher_list[i];
            const Tensor& optimal_weight = ewc_optimal_weights_list[i];

            OP_REQUIRES(context, var.shape() == fisher_info.shape(),
                        errors::InvalidArgument("Shape mismatch for EWC fisher_info and model_weight."));
            OP_REQUIRES(context, var.shape() == optimal_weight.shape(),
                        errors::InvalidArgument("Shape mismatch for EWC optimal_weight and model_weight."));

            auto var_flat = var.flat<float>();
            auto fisher_flat = fisher_info.flat<float>();
            auto optimal_flat = optimal_weight.flat<float>();

            float layer_penalty = VectorizedEWCPenalty(
                var_flat.data(), fisher_flat.data(), optimal_flat.data(), var_flat.size());
            ewc_penalty += layer_penalty;
        }
        ewc_penalty *= (ewc_lambda / 2.0f);
    }

    if (n4sid_order > 0) {
        N4SID_Solver<float> n4sid_solver;
        const auto& u_tensor = n4sid_u_tensor;
        const auto& y_tensor = n4sid_y_tensor;

        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u_matrix(
            u_tensor.flat<float>().data(), u_tensor.dim_size(0), u_tensor.dim_size(1));
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> y_matrix(
            y_tensor.flat<float>().data(), y_tensor.dim_size(0), y_tensor.dim_size(1));

        auto [A_mat, B_mat, C_mat, D_mat] = n4sid_solver.compute(y_matrix, u_matrix, n4sid_order);

        OpOutputList n4sid_matrices_output;
        OP_REQUIRES_OK(context, context->output_list("n4sid_matrices", &n4sid_matrices_output));

        Tensor* n4sid_A_output = nullptr;
        OP_REQUIRES_OK(context, n4sid_matrices_output.allocate(0,
                                                        TensorShape({A_mat.rows(), A_mat.cols()}),
                                                        &n4sid_A_output));
        Eigen::Map<Eigen::MatrixXf>(n4sid_A_output->flat<float>().data(), A_mat.rows(), A_mat.cols()) = A_mat;

        Tensor* n4sid_B_output = nullptr;
        OP_REQUIRES_OK(context, n4sid_matrices_output.allocate(1,
                                                        TensorShape({B_mat.rows(), B_mat.cols()}),
                                                        &n4sid_B_output));
        Eigen::Map<Eigen::MatrixXf>(n4sid_B_output->flat<float>().data(), B_mat.rows(), B_mat.cols()) = B_mat;

        Tensor* n4sid_C_output = nullptr;
        OP_REQUIRES_OK(context, n4sid_matrices_output.allocate(2,
                                                        TensorShape({C_mat.rows(), C_mat.cols()}),
                                                        &n4sid_C_output));
        Eigen::Map<Eigen::MatrixXf>(n4sid_C_output->flat<float>().data(), C_mat.rows(), C_mat.cols()) = C_mat;

        Tensor* n4sid_D_output = nullptr;
        OP_REQUIRES_OK(context, n4sid_matrices_output.allocate(3,
                                                        TensorShape({D_mat.rows(), D_mat.cols()}),
                                                        &n4sid_D_output));
        Eigen::Map<Eigen::MatrixXf>(n4sid_D_output->flat<float>().data(), D_mat.rows(), D_mat.cols()) = D_mat;
    }

    Tensor* logits_tensor_ptr = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(m_ + o_ + 4,
                                                    TensorShape({batch_size, seq_len, label_dim}),
                                                    &logits_tensor_ptr));
    CHECK_EQ(logits_tensor_ptr->dtype(), DT_FLOAT);


    OpOutputList final_float_states_out;
    OP_REQUIRES_OK(context, context->output_list("final_float_states_out", &final_float_states_out));
    OpOutputList final_int_states_out;
    OP_REQUIRES_OK(context, context->output_list("final_int_states_out", &final_int_states_out));

    ComputeFusedReasoningStackForward(context, sequence_input_tensor, block_types_tensor,
                                      block_weight_counts_tensor, block_descriptors_tensor,
                                      initial_float_states,
                                      initial_int_states, model_weights, logits_tensor_ptr,
                                      &final_float_states_out, &final_int_states_out);
    LOG(INFO) << "[TrainStepOp] Reasoning stack forward complete.";

    float standard_loss = 0.0f;
    auto logits_tensor = *logits_tensor_ptr;
    auto logits_flat = logits_tensor.flat<float>();
    const int32* labels_flat = labels_data;

    int num_elements_in_batch_seq = batch_size * seq_len;
    float total_mask_sum = 0.0f;

    for (int i = 0; i < num_elements_in_batch_seq; ++i) {
        const float mask_val = loss_mask_data[i];
        if (mask_val <= 0.0f) {
            continue;
        }
        int true_label = labels_flat[i]; // Declare true_label here
        int logit_vector_offset = i * label_dim; // Declare logit_vector_offset here

        Eigen::Map<const Eigen::VectorXf> logit_vector( // Declare logit_vector here
            &logits_flat(logit_vector_offset), label_dim);

        Eigen::VectorXf exp_logits = logit_vector.array().exp();
        float sum_exp_logits = exp_logits.sum();
        Eigen::VectorXf probabilities = exp_logits / sum_exp_logits;

        float true_label_prob = probabilities(true_label); // Declare and assign here

        if (true_label >= 0 && true_label < label_dim) {
            standard_loss += -std::log(true_label_prob) * mask_val;
            total_mask_sum += mask_val;
        }
    }
    if (total_mask_sum > 0) {
        standard_loss /= total_mask_sum;
    } else {
        standard_loss = 0.0f;
    }

    float total_loss = standard_loss + ewc_penalty;
    LOG(INFO) << "[TrainStepOp] Loss computed. standard=" << standard_loss
              << " ewc=" << ewc_penalty;

    Tensor grad_logits_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, logits_tensor_ptr->shape(), &grad_logits_tensor));
    auto grad_logits_flat = grad_logits_tensor.flat<float>();
    grad_logits_flat.setZero();

    auto logits_tensor_ref = *logits_tensor_ptr;
    auto logits_flat_ref = logits_tensor_ref.flat<float>();
    const int32* labels_flat_ref = labels_data;

    int num_elements_in_batch_seq_grad = batch_size * seq_len;
    float total_mask_sum_grad = 0.0f;

    for (int i = 0; i < num_elements_in_batch_seq_grad; ++i) {
        const float mask_val = loss_mask_data[i];
        if (mask_val <= 0.0f) {
            continue;
        }
        int true_label = labels_flat_ref[i];
        int logit_vector_offset = i * label_dim;

        Eigen::Map<const Eigen::VectorXf> logit_vector(
            &logits_flat_ref(logit_vector_offset), label_dim);

        Eigen::VectorXf exp_logits = logit_vector.array().exp();
        float sum_exp_logits = exp_logits.sum();
        Eigen::VectorXf probabilities = exp_logits / sum_exp_logits;

        if (true_label >= 0 && true_label < label_dim) {
            total_mask_sum_grad += mask_val;
            for (int k = 0; k < label_dim; ++k) {
                const float contrib = (k == true_label) ? (probabilities(k) - 1.0f) : probabilities(k);
                grad_logits_flat(logit_vector_offset + k) += contrib * mask_val;
            }
        }
    }

    if (total_mask_sum_grad > 0) {
        for (int i = 0; i < grad_logits_flat.size(); ++i) {
            grad_logits_flat(i) /= total_mask_sum_grad;
        }
    }
    CHECK_EQ(grad_logits_tensor.dtype(), DT_FLOAT);

    Tensor grad_seq_in_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, sequence_input_tensor.shape(), &grad_seq_in_tensor));
    CHECK_EQ(grad_seq_in_tensor.dtype(), DT_FLOAT);

    OpOutputList grad_initial_float_states_out;
    OP_REQUIRES_OK(context, context->output_list("grad_initial_float_states_out", &grad_initial_float_states_out));
    OpOutputList grad_initial_int_states_out;
    OP_REQUIRES_OK(context, context->output_list("grad_initial_int_states_out", &grad_initial_int_states_out));

    std::vector<Tensor> grads(m_);
    for (int i = 0; i < m_; ++i) {
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, model_weights[i].shape(), &grads[i]));
        grads[i].flat<float>().setZero();
        CHECK_EQ(grads[i].dtype(), DT_FLOAT);
    }
    
    std::vector<Tensor*> grad_weight_ptrs;
    grad_weight_ptrs.reserve(grads.size());
    for (int i = 0; i < grads.size(); ++i) {
        grad_weight_ptrs.push_back(&grads[i]);
    }
    
    std::vector<Tensor> grad_final_float_states_vec(n_);
    std::vector<const Tensor*> grad_final_states_ptrs;
    grad_final_states_ptrs.reserve(n_);
    for (int i = 0; i < n_; ++i) {
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, initial_float_states[i].shape(), &grad_final_float_states_vec[i]));
        grad_final_float_states_vec[i].flat<float>().setZero();
        grad_final_states_ptrs.push_back(&grad_final_float_states_vec[i]);
    }

    ComputeFusedReasoningStackBackward(
        context,
        grad_logits_tensor,
        grad_final_states_ptrs,
        sequence_input_tensor,
        block_types_tensor,
        block_weight_counts_tensor,
        block_descriptors_tensor,
        initial_float_states,
        model_weights,
        &grad_seq_in_tensor,
        &grad_initial_float_states_out,
        nullptr,
        &grad_weight_ptrs
    );

    // Phase 11 SIMD compliance: Vectorized EWC gradient addition
    // Implements: grad += ewc_lambda * fisher * (var - optimal)
    auto VectorizedEWCGradAdd = [](float* grad, const float* var, const float* fisher,
                                   const float* optimal, int64_t size, float ewc_lambda) {
        int64_t i = 0;

#if defined(__AVX512F__)
        const __m512 ewc_lambda_vec = _mm512_set1_ps(ewc_lambda);
        for (; i + 16 <= size; i += 16) {
            __m512 grad_vec = _mm512_loadu_ps(&grad[i]);
            __m512 var_vec = _mm512_loadu_ps(&var[i]);
            __m512 optimal_vec = _mm512_loadu_ps(&optimal[i]);
            __m512 fisher_vec = _mm512_loadu_ps(&fisher[i]);

            // diff = var - optimal
            __m512 diff = _mm512_sub_ps(var_vec, optimal_vec);

            // grad += ewc_lambda * fisher * diff (FMA)
            __m512 grad_new = _mm512_fmadd_ps(ewc_lambda_vec, _mm512_mul_ps(fisher_vec, diff), grad_vec);

            _mm512_storeu_ps(&grad[i], grad_new);
        }
#elif defined(__AVX2__)
        const __m256 ewc_lambda_vec = _mm256_set1_ps(ewc_lambda);
        for (; i + 8 <= size; i += 8) {
            __m256 grad_vec = _mm256_loadu_ps(&grad[i]);
            __m256 var_vec = _mm256_loadu_ps(&var[i]);
            __m256 optimal_vec = _mm256_loadu_ps(&optimal[i]);
            __m256 fisher_vec = _mm256_loadu_ps(&fisher[i]);

            // diff = var - optimal
            __m256 diff = _mm256_sub_ps(var_vec, optimal_vec);

            // grad += ewc_lambda * fisher * diff (FMA)
            __m256 grad_new = _mm256_fmadd_ps(ewc_lambda_vec, _mm256_mul_ps(fisher_vec, diff), grad_vec);

            _mm256_storeu_ps(&grad[i], grad_new);
        }
#elif defined(__ARM_NEON)
        const float32x4_t ewc_lambda_vec = vdupq_n_f32(ewc_lambda);
        for (; i + 4 <= size; i += 4) {
            float32x4_t grad_vec = vld1q_f32(&grad[i]);
            float32x4_t var_vec = vld1q_f32(&var[i]);
            float32x4_t optimal_vec = vld1q_f32(&optimal[i]);
            float32x4_t fisher_vec = vld1q_f32(&fisher[i]);

            // diff = var - optimal
            float32x4_t diff = vsubq_f32(var_vec, optimal_vec);

            // grad += ewc_lambda * fisher * diff (FMA)
            float32x4_t grad_new = vfmaq_f32(grad_vec, ewc_lambda_vec, vmulq_f32(fisher_vec, diff));

            vst1q_f32(&grad[i], grad_new);
        }
#endif

        // Scalar fallback
        for (; i < size; ++i) {
            float diff = var[i] - optimal[i];
            grad[i] += ewc_lambda * fisher[i] * diff;
        }
    };

    if (ewc_lambda > 0.0f && f_ > 0) {
        for (int i = 0; i < m_; ++i) {
            const Tensor& var = model_weights[i];
            const Tensor& fisher_info = ewc_fisher_list[i];
            const Tensor& optimal_weight = ewc_optimal_weights_list[i];

            auto var_flat = var.flat<float>();
            auto fisher_flat = fisher_info.flat<float>();
            auto optimal_flat = optimal_weight.flat<float>();
            auto grad_flat = grads[i].flat<float>();

            VectorizedEWCGradAdd(grad_flat.data(), var_flat.data(), fisher_flat.data(),
                                optimal_flat.data(), var_flat.size(), ewc_lambda);
        }
    }

    OpOutputList new_model_weights_list;
    OP_REQUIRES_OK(context, context->output_list("new_model_weights", &new_model_weights_list));

    OpOutputList new_optimizer_state_list;
    OP_REQUIRES_OK(context, context->output_list("new_optimizer_state", &new_optimizer_state_list));

    std::vector<Tensor*> new_model_weights_ptrs(m_);
    for (int i = 0; i < m_; ++i) {
        OP_REQUIRES_OK(context, new_model_weights_list.allocate(i, model_weights[i].shape(), &new_model_weights_ptrs[i]));
        new_model_weights_ptrs[i]->flat<float>() = model_weights[i].flat<float>();
    }

    std::vector<Tensor*> new_optimizer_state_ptrs(o_);
    for (int i = 0; i < o_; ++i) {
        OP_REQUIRES_OK(context, new_optimizer_state_list.allocate(i, optimizer_state[i].shape(), &new_optimizer_state_ptrs[i]));
        new_optimizer_state_ptrs[i]->flat<float>() = optimizer_state[i].flat<float>();
    }

    if (global_step % k_val == 0) {
        for (int i = 0; i < m_; ++i) {
            Tensor* h_tensor = new_optimizer_state_ptrs[i * 2 + 1];
            const Tensor& h_grad = grads[i];
            auto h_grad_flat = h_grad.flat<float>();
            auto h_flat = h_tensor->flat<float>();
            const int64 num_elements = h_grad_flat.size();

            // Use vectorized Hessian update for better performance
            saguaro::parallel::ForShard(
                static_cast<std::size_t>(num_elements),
                static_cast<std::size_t>(1000),
                [&](int64 start, int64 end) {
                VectorizedHessianUpdate(&h_grad_flat(start), &h_flat(start), end - start, beta_2_t);
            });
        }
    }
    // =========================================================================
    // Quantum Training: Barren Plateau Detection and QNG Preconditioning
    // =========================================================================
    float effective_lr = lr;
    
    // Barren Plateau Detection: Compute aggregate gradient norm and adjust LR
    if (enable_barren_plateau) {
        float total_grad_norm_sq = 0.0f;
        for (int i = 0; i < m_; ++i) {
            auto grad_flat = grads[i].flat<float>();
            float norm = saguaro::quantum_training::VectorizedGradientNorm(
                grad_flat.data(), grad_flat.size());
            total_grad_norm_sq += norm * norm;
        }
        float aggregate_grad_norm = std::sqrt(total_grad_norm_sq);
        
        // Apply LR scaling if in barren plateau
        if (aggregate_grad_norm < barren_plateau_threshold) {
            effective_lr = lr * barren_plateau_lr_scale;
            LOG(INFO) << "[TrainStepOp] Barren plateau detected (grad_norm="
                      << aggregate_grad_norm << " < " << barren_plateau_threshold
                      << "), LR scaled: " << lr << " -> " << effective_lr;
        }
    }
    
    // QNG Preconditioning: Apply QFIM-based gradient preconditioning
    // Note: QFIM is stored as part of the Hessian diagonal (h_tensor)
    // We use the Hessian as a curvature estimate for natural gradient
    for (int i = 0; i < m_; ++i) {
        Tensor* var_tensor = new_model_weights_ptrs[i];
        Tensor* m_tensor = new_optimizer_state_ptrs[i * 2];
        Tensor* h_tensor = new_optimizer_state_ptrs[i * 2 + 1];

        if (enable_qng) {
            // Update QFIM diagonal estimate and apply preconditioning
            auto grad_flat = grads[i].flat<float>();
            auto h_flat = h_tensor->flat<float>();
            const int64_t num_elements = grad_flat.size();
            
            // Create a copy for in-place QNG preconditioning
            Tensor precond_grad;
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, grads[i].shape(), &precond_grad));
            auto precond_flat = precond_grad.flat<float>();
            
            // Copy gradient for modification
            std::memcpy(precond_flat.data(), grad_flat.data(), num_elements * sizeof(float));
            
            // Update QFIM (EMA of squared gradients) and apply preconditioning
            saguaro::parallel::ForShard(
                static_cast<std::size_t>(num_elements),
                static_cast<std::size_t>(1000),
                [&](int64 start, int64 end) {
                    saguaro::quantum_training::VectorizedQFIMUpdate(
                        &grad_flat(start), &h_flat(start), end - start, qng_ema_decay);
                    saguaro::quantum_training::VectorizedQNGPrecondition(
                        &precond_flat(start), &h_flat(start), end - start, qng_damping);
                });
            
            // Apply SophiaG with preconditioned gradient
            SophiaG_ApplyGradients(context, precond_grad, var_tensor, m_tensor, h_tensor, 
                                   effective_lr, beta_1_t, rho_t, epsilon);
        } else {
            // Standard SophiaG without QNG
            SophiaG_ApplyGradients(context, grads[i], var_tensor, m_tensor, h_tensor, 
                                   effective_lr, beta_1_t, rho_t, epsilon);
        }
    }

    Tensor* total_loss_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(m_ + o_, TensorShape({}), &total_loss_output));
    total_loss_output->scalar<float>()() = total_loss;

    Tensor* std_loss_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(m_ + o_ + 1, TensorShape({}), &std_loss_output));
    std_loss_output->scalar<float>()() = standard_loss;

    Tensor* ewc_loss_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(m_ + o_ + 2, TensorShape({}), &ewc_loss_output));
    ewc_loss_output->scalar<float>()() = ewc_penalty;

    // Phase 11 SIMD compliance: Vectorized gradient norm computation
    // Implements: norm = sqrt(sum(grad^2))
    auto VectorizedGradNorm = [](const float* grad, int64_t size) -> float {
        int64_t i = 0;
        float sum_sq = 0.0f;

#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        for (; i + 16 <= size; i += 16) {
            __m512 grad_vec = _mm512_loadu_ps(&grad[i]);
            acc = _mm512_fmadd_ps(grad_vec, grad_vec, acc);
        }
        sum_sq = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8) {
            __m256 grad_vec = _mm256_loadu_ps(&grad[i]);
            acc = _mm256_fmadd_ps(grad_vec, grad_vec, acc);
        }
        // Horizontal sum for AVX2
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        sum_sq = _mm_cvtss_f32(sum);
#elif defined(__ARM_NEON)
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (; i + 4 <= size; i += 4) {
            float32x4_t grad_vec = vld1q_f32(&grad[i]);
            acc = vfmaq_f32(acc, grad_vec, grad_vec);
        }
        // Horizontal sum for NEON
        float32x2_t sum = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        sum = vpadd_f32(sum, sum);
        sum_sq = vget_lane_f32(sum, 0);
#endif

        // Scalar fallback
        for (; i < size; ++i) {
            sum_sq += grad[i] * grad[i];
        }
        return sum_sq;
    };

    float gradient_norm_val = 0.0f;
    for (int i = 0; i < m_; ++i) {
        auto grad_flat = grads[i].flat<float>();
        gradient_norm_val += VectorizedGradNorm(grad_flat.data(), grad_flat.size());
    }
    gradient_norm_val = std::sqrt(gradient_norm_val);

    Tensor* gradient_norm_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(m_ + o_ + 3, TensorShape({}), &gradient_norm_output));
    gradient_norm_output->scalar<float>()() = gradient_norm_val;
} // End of TrainStepOp class

REGISTER_KERNEL_BUILDER(Name("TrainStep").Device(DEVICE_CPU), TrainStepOp);

}  // namespace tensorflow
