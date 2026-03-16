// src/ops/fused_reasoning_stack/helpers.h
// Copyright 2025 Verso Industries
//
// This header file declares the helper functions and data structures used by the
// FusedReasoningStack custom operator kernels. It defines a clear interface
// for the forward and backward computation of each distinct reasoning block.

#ifndef TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_HELPERS_H_
#define TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_HELPERS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/errors.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/strings/numbers.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <cmath>
#include <cstdint>
#include <limits>
#include "ops/hnn_core_helpers.h" // For HNN/TimeCrystal logic and state structs
#include <vector>
#include <string>

namespace tensorflow {

using Eigen::MatrixXf;
using Eigen::VectorXf;

/**
 * @brief Context structure to pass common parameters to block helpers, reducing
 * redundant arguments and improving code readability.
 */
struct BlockContext {
    OpKernelContext* op_context;
    int64_t batch_size;
    int64_t seq_len_combined;
    int64_t d_embed;
};

struct BlockDescriptorInfo {
    tstring raw_json;
    std::string type;
    int weight_count;
    bool stateful;
    absl::flat_hash_map<std::string, std::string> metadata;

    bool TryGetMetadataString(absl::string_view key,
                              std::string* value_out = nullptr) const {
        auto it = metadata.find(std::string(key));
        if (it == metadata.end()) {
            return false;
        }
        if (value_out != nullptr) {
            *value_out = it->second;
        }
        return true;
    }

    bool TryGetMetadataInt(absl::string_view key,
                           int64_t* value_out) const {
        if (value_out == nullptr) {
            return false;
        }
        auto it = metadata.find(std::string(key));
        if (it == metadata.end()) {
            return false;
        }
        int64_t parsed_value = 0;
        if (absl::SimpleAtoi(it->second, &parsed_value)) {
            *value_out = parsed_value;
            return true;
        }
        double float_value = 0.0;
        if (absl::SimpleAtod(it->second, &float_value)) {
            *value_out = static_cast<int64_t>(float_value);
            return true;
        }
        return false;
    }

    bool TryGetMetadataFloat(absl::string_view key,
                             double* value_out) const {
        if (value_out == nullptr) {
            return false;
        }
        auto it = metadata.find(std::string(key));
        if (it == metadata.end()) {
            return false;
        }
        double parsed_value = 0.0;
        if (absl::SimpleAtod(it->second, &parsed_value)) {
            *value_out = parsed_value;
            return true;
        }
        int64_t int_value = 0;
        if (absl::SimpleAtoi(it->second, &int_value)) {
            *value_out = static_cast<double>(int_value);
            return true;
        }
        return false;
    }
};

Status ParseBlockDescriptor(const tstring& json_descriptor,
                           BlockDescriptorInfo* descriptor_out);

namespace stateless_internal {

struct FlattenedWeightShape {
    int64_t input_dim;
    int64_t output_dim;
    bool reshaped;
};

struct BiasVectorShape {
    int64_t length;
    bool reshaped;
};

inline bool ResolveWeightShape(OpKernelContext* context,
                               const Tensor& weight,
                               int layer_index,
                               FlattenedWeightShape* shape) {
    if (shape == nullptr) {
        context->CtxFailure(errors::Internal(
            "Stateless block weight shape resolver requires a valid output struct."));
        return false;
    }
    if (weight.dims() < 2) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block layer ", layer_index,
            " weight must have rank >= 2, got rank ", weight.dims(),
            " for shape ", weight.shape().DebugString()));
        return false;
    }
    const int64_t out_dim = weight.dim_size(weight.dims() - 1);
    if (out_dim <= 0) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block layer ", layer_index,
            " weight must have a positive last dimension, got ", out_dim,
            " for shape ", weight.shape().DebugString()));
        return false;
    }
    const int64_t total_elems = weight.NumElements();
    if (total_elems % out_dim != 0) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block layer ", layer_index,
            " weight shape ", weight.shape().DebugString(),
            " cannot be flattened into a 2D matrix with output dimension ", out_dim));
        return false;
    }
    shape->input_dim = total_elems / out_dim;
    shape->output_dim = out_dim;
    shape->reshaped = (weight.dims() != 2);
    return true;
}

inline bool ResolveBiasVectorShape(OpKernelContext* context,
                                   const Tensor& bias,
                                   int layer_index,
                                   int64_t expected_length,
                                   BiasVectorShape* shape = nullptr) {
    if (bias.dims() < 1) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block layer ", layer_index,
            " bias must have rank >= 1, got rank ", bias.dims(),
            " for shape ", bias.shape().DebugString()));
        return false;
    }
    const int64_t num_elems = bias.NumElements();
    if (num_elems != expected_length) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block layer ", layer_index,
            " bias element count mismatch. Expected ", expected_length,
            ", got ", num_elems,
            " for shape ", bias.shape().DebugString()));
        return false;
    }
    if (shape != nullptr) {
        shape->length = num_elems;
        shape->reshaped = (bias.dims() != 1 || bias.dim_size(0) != num_elems);
    }
    return true;
}

// ============================================================================
// SIMD-Accelerated GELU Implementation (Phase 11 GROUP_4_MEDIUM_RISK)
// ============================================================================
// Precision: float32
// Target: AVX512 (16-wide), AVX2 (8-wide), NEON (4-wide)
// Hot paths: Dense layer activation in stateless reasoning blocks
// ============================================================================

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Scalar fallback versions
inline float gelu_scalar(float x) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;      // 1 / sqrt(2)
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

inline float gelu_grad_scalar(float x) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;    // 1 / sqrt(2 * pi)
    const float erf_term = std::erf(x * kInvSqrt2);
    const float exp_term = std::exp(-0.5f * x * x);
    return 0.5f * (1.0f + erf_term) + 0.5f * x * kInvSqrt2Pi * exp_term;
}

// Vectorized GELU activation (forward)
inline void gelu_vectorized(const float* in, float* out, int64_t size) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 inv_sqrt2 = _mm512_set1_ps(kInvSqrt2);

    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&in[i]);
        __m512 x_scaled = _mm512_mul_ps(x, inv_sqrt2);

        // Approximate erf using AVX512 exp
        // For GELU, use tanh approximation: erf(x) ≈ tanh(√(2/π) * (x + 0.044715 * x^3))
        __m512 x_sq = _mm512_mul_ps(x_scaled, x_scaled);
        __m512 x_cu = _mm512_mul_ps(x_sq, x_scaled);
        __m512 tanh_arg = _mm512_fmadd_ps(_mm512_set1_ps(0.044715f), x_cu, x_scaled);
        tanh_arg = _mm512_mul_ps(_mm512_set1_ps(0.7978845608f), tanh_arg);  // sqrt(2/pi)

        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        // For simplicity, use libm fallback for each element (AVX512 doesn't have tanh)
        float tmp[16];
        _mm512_storeu_ps(tmp, tanh_arg);
        for (int j = 0; j < 16; ++j) {
            tmp[j] = std::tanh(tmp[j]);
        }
        __m512 erf_approx = _mm512_loadu_ps(tmp);

        // gelu(x) = 0.5 * x * (1 + erf(x/sqrt(2)))
        __m512 one_plus_erf = _mm512_add_ps(_mm512_set1_ps(1.0f), erf_approx);
        __m512 result = _mm512_mul_ps(half, _mm512_mul_ps(x, one_plus_erf));
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 inv_sqrt2 = _mm256_set1_ps(kInvSqrt2);

    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        __m256 x_scaled = _mm256_mul_ps(x, inv_sqrt2);

        // Tanh approximation for erf
        __m256 x_sq = _mm256_mul_ps(x_scaled, x_scaled);
        __m256 x_cu = _mm256_mul_ps(x_sq, x_scaled);
        __m256 tanh_arg = _mm256_fmadd_ps(_mm256_set1_ps(0.044715f), x_cu, x_scaled);
        tanh_arg = _mm256_mul_ps(_mm256_set1_ps(0.7978845608f), tanh_arg);

        // Fallback to scalar tanh
        float tmp[8];
        _mm256_storeu_ps(tmp, tanh_arg);
        for (int j = 0; j < 8; ++j) {
            tmp[j] = std::tanh(tmp[j]);
        }
        __m256 erf_approx = _mm256_loadu_ps(tmp);

        __m256 one_plus_erf = _mm256_add_ps(_mm256_set1_ps(1.0f), erf_approx);
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_erf));
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t inv_sqrt2 = vdupq_n_f32(kInvSqrt2);

    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        float32x4_t x_scaled = vmulq_f32(x, inv_sqrt2);

        // Tanh approximation
        float32x4_t x_sq = vmulq_f32(x_scaled, x_scaled);
        float32x4_t x_cu = vmulq_f32(x_sq, x_scaled);
        float32x4_t tanh_arg = vmlaq_f32(x_scaled, vdupq_n_f32(0.044715f), x_cu);
        tanh_arg = vmulq_f32(vdupq_n_f32(0.7978845608f), tanh_arg);

        // Scalar tanh fallback
        float tmp[4];
        vst1q_f32(tmp, tanh_arg);
        for (int j = 0; j < 4; ++j) {
            tmp[j] = std::tanh(tmp[j]);
        }
        float32x4_t erf_approx = vld1q_f32(tmp);

        float32x4_t one_plus_erf = vaddq_f32(vdupq_n_f32(1.0f), erf_approx);
        float32x4_t result = vmulq_f32(half, vmulq_f32(x, one_plus_erf));
        vst1q_f32(&out[i], result);
    }
#endif

    // Scalar remainder
    for (; i < size; ++i) {
        out[i] = gelu_scalar(in[i]);
    }
}

// Vectorized GELU gradient (backward)
inline void gelu_grad_vectorized(const float* in, float* out, int64_t size) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    constexpr float kInvSqrt2Pi = 0.39894228040143267794f;
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 inv_sqrt2 = _mm512_set1_ps(kInvSqrt2);
    const __m512 inv_sqrt2_pi = _mm512_set1_ps(kInvSqrt2Pi);

    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&in[i]);

        // erf term
        __m512 x_scaled = _mm512_mul_ps(x, inv_sqrt2);
        float tmp_erf[16];
        _mm512_storeu_ps(tmp_erf, x_scaled);
        for (int j = 0; j < 16; ++j) {
            tmp_erf[j] = std::erf(tmp_erf[j]);
        }
        __m512 erf_term = _mm512_loadu_ps(tmp_erf);

        // exp term: exp(-0.5 * x^2)
        __m512 x_sq = _mm512_mul_ps(x, x);
        __m512 neg_half_x_sq = _mm512_mul_ps(_mm512_set1_ps(-0.5f), x_sq);
        float tmp_exp[16];
        _mm512_storeu_ps(tmp_exp, neg_half_x_sq);
        for (int j = 0; j < 16; ++j) {
            tmp_exp[j] = std::exp(tmp_exp[j]);
        }
        __m512 exp_term = _mm512_loadu_ps(tmp_exp);

        // grad = 0.5 * (1 + erf_term) + 0.5 * x * inv_sqrt2_pi * exp_term
        __m512 term1 = _mm512_mul_ps(half, _mm512_add_ps(_mm512_set1_ps(1.0f), erf_term));
        __m512 term2 = _mm512_mul_ps(half, _mm512_mul_ps(x, _mm512_mul_ps(inv_sqrt2_pi, exp_term)));
        __m512 result = _mm512_add_ps(term1, term2);
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 inv_sqrt2 = _mm256_set1_ps(kInvSqrt2);
    const __m256 inv_sqrt2_pi = _mm256_set1_ps(kInvSqrt2Pi);

    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);

        __m256 x_scaled = _mm256_mul_ps(x, inv_sqrt2);
        float tmp_erf[8];
        _mm256_storeu_ps(tmp_erf, x_scaled);
        for (int j = 0; j < 8; ++j) {
            tmp_erf[j] = std::erf(tmp_erf[j]);
        }
        __m256 erf_term = _mm256_loadu_ps(tmp_erf);

        __m256 x_sq = _mm256_mul_ps(x, x);
        __m256 neg_half_x_sq = _mm256_mul_ps(_mm256_set1_ps(-0.5f), x_sq);
        float tmp_exp[8];
        _mm256_storeu_ps(tmp_exp, neg_half_x_sq);
        for (int j = 0; j < 8; ++j) {
            tmp_exp[j] = std::exp(tmp_exp[j]);
        }
        __m256 exp_term = _mm256_loadu_ps(tmp_exp);

        __m256 term1 = _mm256_mul_ps(half, _mm256_add_ps(_mm256_set1_ps(1.0f), erf_term));
        __m256 term2 = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_mul_ps(inv_sqrt2_pi, exp_term)));
        __m256 result = _mm256_add_ps(term1, term2);
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t inv_sqrt2 = vdupq_n_f32(kInvSqrt2);
    const float32x4_t inv_sqrt2_pi = vdupq_n_f32(kInvSqrt2Pi);

    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);

        float32x4_t x_scaled = vmulq_f32(x, inv_sqrt2);
        float tmp_erf[4];
        vst1q_f32(tmp_erf, x_scaled);
        for (int j = 0; j < 4; ++j) {
            tmp_erf[j] = std::erf(tmp_erf[j]);
        }
        float32x4_t erf_term = vld1q_f32(tmp_erf);

        float32x4_t x_sq = vmulq_f32(x, x);
        float32x4_t neg_half_x_sq = vmulq_f32(vdupq_n_f32(-0.5f), x_sq);
        float tmp_exp[4];
        vst1q_f32(tmp_exp, neg_half_x_sq);
        for (int j = 0; j < 4; ++j) {
            tmp_exp[j] = std::exp(tmp_exp[j]);
        }
        float32x4_t exp_term = vld1q_f32(tmp_exp);

        float32x4_t term1 = vmulq_f32(half, vaddq_f32(vdupq_n_f32(1.0f), erf_term));
        float32x4_t term2 = vmulq_f32(half, vmulq_f32(x, vmulq_f32(inv_sqrt2_pi, exp_term)));
        float32x4_t result = vaddq_f32(term1, term2);
        vst1q_f32(&out[i], result);
    }
#endif

    // Scalar remainder
    for (; i < size; ++i) {
        out[i] = gelu_grad_scalar(in[i]);
    }
}

// Legacy scalar interface for compatibility
inline float gelu(float x) {
    return gelu_scalar(x);
}

inline float gelu_grad(float x) {
    return gelu_grad_scalar(x);
}

inline bool ValidateWeightPair(OpKernelContext* context,
                               const Tensor& weight,
                               const Tensor& bias,
                               int64_t expected_in_dim,
                               int layer_index,
                               FlattenedWeightShape* resolved_shape = nullptr) {
    FlattenedWeightShape local_shape;
    if (!ResolveWeightShape(context, weight, layer_index, &local_shape)) {
        return false;
    }
    if (local_shape.input_dim != expected_in_dim) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block layer ", layer_index,
            " weight input dimension mismatch. Expected ", expected_in_dim,
            ", got ", local_shape.input_dim,
            " after flattening shape ", weight.shape().DebugString()));
        return false;
    }
    if (!ResolveBiasVectorShape(context, bias, layer_index, local_shape.output_dim)) {
        return false;
    }
    if (resolved_shape != nullptr) {
        *resolved_shape = local_shape;
    }
    return true;
}

/**
 * Lightweight shape validation used to decide whether a stateless block can
 * run through the Dense/GELU helper or should fall back to a no-op residual.
 */
bool CanRunDenseStatelessBlock(const MatrixXf& input_seq,
                               const OpInputList& weights,
                               int weight_idx,
                               int num_weights_to_consume,
                               std::string* failure_reason = nullptr);

inline MatrixXf RunStatelessDenseForward(
    OpKernelContext* context,
    const MatrixXf& input_seq,
    const OpInputList& weights,
    int weight_idx,
    int num_weights_to_consume,
    std::vector<MatrixXf>* layer_inputs = nullptr,
    std::vector<MatrixXf>* pre_activations = nullptr) {

    if (num_weights_to_consume <= 0) {
        return MatrixXf::Zero(input_seq.rows(), input_seq.cols());
    }

    if (weight_idx < 0 || weight_idx + num_weights_to_consume > weights.size()) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block requested ", num_weights_to_consume,
            " weights starting at index ", weight_idx,
            ", but only ", weights.size(), " weights are available."));
        return MatrixXf();
    }

    if (num_weights_to_consume % 2 != 0) {
        context->CtxFailure(errors::InvalidArgument(
            "Stateless block expects weight/bias pairs; received ",
            num_weights_to_consume, " tensors."));
        return MatrixXf();
    }

    const int pair_count = num_weights_to_consume / 2;
    if (layer_inputs) {
        layer_inputs->clear();
        layer_inputs->reserve(pair_count + 1);
        layer_inputs->push_back(input_seq);
    }
    if (pre_activations) {
        pre_activations->clear();
        pre_activations->reserve(pair_count);
    }

    MatrixXf hidden = input_seq;
    for (int layer = 0; layer < pair_count; ++layer) {
        const Tensor& weight_tensor = weights[weight_idx + 2 * layer];
        const Tensor& bias_tensor = weights[weight_idx + 2 * layer + 1];

        FlattenedWeightShape weight_shape;
        if (!ValidateWeightPair(context, weight_tensor, bias_tensor,
                                static_cast<int64_t>(hidden.cols()), layer,
                                &weight_shape)) {
            return MatrixXf();
        }

        const int64_t in_dim = weight_shape.input_dim;
        const int64_t out_dim = weight_shape.output_dim;

        Eigen::Map<const MatrixXf> weight_map(weight_tensor.flat<float>().data(),
                                              in_dim, out_dim);
        Eigen::Map<const VectorXf> bias_map(bias_tensor.flat<float>().data(),
                                            out_dim);

        MatrixXf pre_activation = hidden * weight_map;
        pre_activation.rowwise() += bias_map.transpose();

        if (pre_activations) {
            pre_activations->push_back(pre_activation);
        }

        if (layer < pair_count - 1) {
            // SIMD-optimized GELU activation (Phase 11)
            hidden.resize(pre_activation.rows(), pre_activation.cols());
            gelu_vectorized(pre_activation.data(), hidden.data(),
                           pre_activation.rows() * pre_activation.cols());
        } else {
            hidden = std::move(pre_activation);
        }

        if (layer_inputs) {
            layer_inputs->push_back(hidden);
        }
    }

    return hidden;
}

} // namespace stateless_internal

/**
 * @brief Forward pass for stateless blocks (e.g., MoE, Mamba placeholder).
 * @param ctx The block context containing shared dimensions.
 * @param input_seq The input sequence for this block.
 * @param weights The list of all weights for the entire stack.
 * @param weight_idx The current index into the weights list (passed by reference and advanced).
 * @param num_weights_to_consume The number of weight tensors this block uses.
 * @return The output sequence after applying the block's transformation.
 */
MatrixXf StatelessBlockForward(
    const BlockContext& ctx,
    const BlockDescriptorInfo& descriptor,
    const MatrixXf& input_seq,
    const OpInputList& weights,
    int& weight_idx,
    int num_weights_to_consume);

/**
 * @brief Backward pass for stateless blocks.
 * @param ctx The block context.
 * @param input_seq The input sequence to this block (recomputed from the forward pass).
 * @param weights The list of all weights.
 * @param weight_idx_start The starting index for this block's weights.
 * @param num_weights_to_consume The number of weight tensors this block uses.
 * @param grad_adj_state The adjoint state, updated in place to become the gradient w.r.t the block's input.
 * @param grad_weights_tensors A vector of pointers to the output gradient tensors for all weights.
 */
void StatelessBlockBackward(
    const BlockContext& ctx,
    const BlockDescriptorInfo& descriptor,
    MatrixXf& grad_adj_state,
    const MatrixXf& input_seq,
    const OpInputList& weights,
    int weight_idx_start,
    int num_weights_to_consume,
    std::vector<Tensor*>& grad_weights_tensors);

/**
 * @brief Forward pass for the stateful TimeCrystalSequenceBlock.
 * @param ctx The block context.
 * @param current_sequence The input sequence, which is updated in place.
 * @param initial_states List of all initial states for the stack.
 * @param weights List of all weights for the stack.
 * @param state_idx Current state index (passed by reference and advanced).
 * @param weight_idx Current weight index (passed by reference and advanced).
 * @param final_state_tensors Vector of output tensors for the final states.
 * @param hnn_forward_states A data structure to store all intermediate values needed for the backward pass.
 */
void TimeCrystalSequenceForward(
    const BlockContext& ctx,
    MatrixXf& current_sequence,
    const OpInputList& initial_states,
    const OpInputList& weights,
    int& state_idx,
    int& weight_idx,
    std::vector<Tensor>& final_state_tensors,
    std::vector<std::vector<HNNForwardState>>& hnn_forward_states);

/**
 * @brief Backward pass for the TimeCrystalSequenceBlock using the adjoint sensitivity method.
 * @param ctx The block context.
 * @param grad_adj_state The adjoint state, updated in place.
 * @param hnn_forward_states Intermediate values from the forward pass.
 * @param weights The list of all weights.
 * @param grad_final_states The upstream gradients for the final states of the block.
 * @param state_idx_bwd The starting index for this block's states (from reverse iteration).
 * @param weight_idx_bwd The starting index for this block's weights (from reverse iteration).
 * @param grad_initial_states_tensors Vector of output gradient tensors for initial states.
 * @param grad_weights_tensors Vector of output gradient tensors for weights.
 */
void TimeCrystalSequenceBackward(
    const BlockContext& ctx,
    MatrixXf& grad_adj_state,
    const std::vector<std::vector<HNNForwardState>>& hnn_forward_states,
    const OpInputList& weights,
    const OpInputList& grad_final_states,
    int state_idx_bwd,
    int weight_idx_bwd,
    std::vector<Tensor*>& grad_initial_states_tensors,
    std::vector<Tensor*>& grad_weights_tensors);

} // namespace tensorflow

#endif // TENSORFLOW_CORE_USER_OPS_FUSED_REASONING_STACK_HELPERS_H_
