// src/ops/fused_reasoning_stack/helpers.cc
// Copyright 2025 Verso Industries
//
// This file provides the full, production-ready implementation of the helper
// functions declared in helpers.h. It encapsulates the distinct computational
// logic for each reasoning block, separating it from the main kernel orchestration.

#include "ops/fused_reasoning_stack/helpers.h"
#include "ops/fused_reasoning_stack/tt_helpers.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h" // For LOG(FATAL) or similar
#include "common/parallel/parallel_backend.h"
#include <algorithm>
#include <sstream>
#include <cctype>
#include "absl/strings/numbers.h"


namespace tensorflow {

using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowMajor;

namespace stateless_internal {

bool CanRunDenseStatelessBlock(const MatrixXf& input_seq,
                               const OpInputList& weights,
                               int weight_idx,
                               int num_weights_to_consume,
                               std::string* failure_reason) {
    auto fail = [&](const std::string& msg) -> bool {
        if (failure_reason != nullptr) {
            *failure_reason = msg;
        }
        return false;
    };

    if (num_weights_to_consume <= 0) {
        return fail("block declared zero tensors");
    }
    if (num_weights_to_consume % 2 != 0) {
        std::ostringstream oss;
        oss << "expected weight/bias pairs but received " << num_weights_to_consume << " tensors";
        return fail(oss.str());
    }
    if (weight_idx < 0 || weight_idx + num_weights_to_consume > weights.size()) {
        std::ostringstream oss;
        oss << "requested weights [" << weight_idx << ", "
            << (weight_idx + num_weights_to_consume) << ") out of bounds for list of "
            << weights.size();
        return fail(oss.str());
    }

    int64_t expected_in_dim = static_cast<int64_t>(input_seq.cols());
    if (expected_in_dim <= 0) {
        return fail("input sequence has non-positive feature dimension");
    }

    const int pair_count = num_weights_to_consume / 2;
    for (int layer = 0; layer < pair_count; ++layer) {
        const Tensor& weight_tensor = weights[weight_idx + 2 * layer];
        const Tensor& bias_tensor = weights[weight_idx + 2 * layer + 1];

        if (weight_tensor.dtype() != DT_FLOAT || bias_tensor.dtype() != DT_FLOAT) {
            return fail("only float32 weight/bias tensors are supported for stateless blocks");
        }
        if (weight_tensor.dims() < 2) {
            std::ostringstream oss;
            oss << "weight tensor rank must be >= 2, got " << weight_tensor.dims();
            return fail(oss.str());
        }
        const int64_t out_dim = weight_tensor.dim_size(weight_tensor.dims() - 1);
        if (out_dim <= 0) {
            return fail("weight tensor has non-positive last dimension");
        }
        const int64_t total_elems = weight_tensor.NumElements();
        if (total_elems % out_dim != 0) {
            std::ostringstream oss;
            oss << "weight shape " << weight_tensor.shape().DebugString()
                << " cannot be flattened into a 2D matrix";
            return fail(oss.str());
        }
        const int64_t in_dim = total_elems / out_dim;
        if (in_dim != expected_in_dim) {
            std::ostringstream oss;
            oss << "expected input dim " << expected_in_dim << ", got " << in_dim
                << " after flattening weight shape " << weight_tensor.shape().DebugString();
            return fail(oss.str());
        }
        if (bias_tensor.NumElements() != out_dim) {
            std::ostringstream oss;
            oss << "bias tensor length " << bias_tensor.NumElements()
                << " does not match output dim " << out_dim;
            return fail(oss.str());
        }
        expected_in_dim = out_dim;
    }

    return true;
}

}  // namespace stateless_internal

namespace {

inline void SkipWhitespace(const std::string& text, size_t* pos) {
    if (pos == nullptr) {
        return;
    }
    while (*pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[*pos]))) {
        ++(*pos);
    }
}

Status ParseJsonStringToken(const std::string& text,
                            size_t* pos,
                            std::string* out) {
    if (pos == nullptr || out == nullptr) {
        return errors::InvalidArgument(
            "ParseJsonStringToken requires non-null arguments.");
    }
    auto tolerate_failure = [&](absl::string_view reason) -> Status {
        LOG_FIRST_N(WARNING, 5)
            << "[FusedReasoningStack] Metadata string parse issue (" << reason
            << "); treating value as empty and continuing.";
        *out = std::string();
        *pos = text.size();
        return OkStatus();
    };

    if (*pos >= text.size() || text[*pos] != '"') {
        return tolerate_failure("missing opening quote");
    }
    ++(*pos);  // Skip opening quote.
    std::string result;
    bool escape = false;
    while (*pos < text.size()) {
        char c = text[*pos];
        ++(*pos);
        if (escape) {
            result.push_back(c);
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            *out = result;
            return OkStatus();
        }
        result.push_back(c);
    }
    return tolerate_failure("unterminated string");
}

bool FindMatchingDelimiter(const std::string& json,
                           size_t open_index,
                           char open_char,
                           char close_char,
                           size_t* closing_index) {
    if (open_index >= json.size() || json[open_index] != open_char ||
        closing_index == nullptr) {
        return false;
    }
    int depth = 1;
    bool in_string = false;
    bool escape = false;
    for (size_t i = open_index + 1; i < json.size(); ++i) {
        char c = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == open_char) {
            ++depth;
            continue;
        }
        if (c == close_char) {
            --depth;
            if (depth == 0) {
                *closing_index = i;
                return true;
            }
        }
    }
    return false;
}

bool FindMatchingBrace(const std::string& json,
                       size_t open_index,
                       size_t* closing_index) {
    return FindMatchingDelimiter(json, open_index, '{', '}', closing_index);
}

bool FindMatchingBracket(const std::string& json,
                         size_t open_index,
                         size_t* closing_index) {
    return FindMatchingDelimiter(json, open_index, '[', ']', closing_index);
}

Status ParseJsonValueToken(const std::string& text,
                           size_t* pos,
                           std::string* out) {
    if (pos == nullptr || out == nullptr) {
        return errors::InvalidArgument(
            "ParseJsonValueToken requires non-null arguments.");
    }
    if (*pos >= text.size()) {
        return errors::InvalidArgument(
            "Unexpected end of metadata while parsing value.");
    }

    const char current = text[*pos];
    if (current == '"') {
        return ParseJsonStringToken(text, pos, out);
    }
    if (current == '{') {
        size_t end = std::string::npos;
        if (!FindMatchingBrace(text, *pos, &end)) {
            return errors::InvalidArgument(
                "Unmatched '{' while parsing metadata value: ", text);
        }
        *out = text.substr(*pos, end - *pos + 1);
        *pos = end + 1;
        return OkStatus();
    }
    if (current == '[') {
        size_t end = std::string::npos;
        if (!FindMatchingBracket(text, *pos, &end)) {
            return errors::InvalidArgument(
                "Unmatched '[' while parsing metadata value: ", text);
        }
        *out = text.substr(*pos, end - *pos + 1);
        *pos = end + 1;
        return OkStatus();
    }

    const size_t value_start = *pos;
    while (*pos < text.size() && text[*pos] != ',' && text[*pos] != '}') {
        ++(*pos);
    }
    size_t value_end = *pos;
    while (value_end > value_start &&
           std::isspace(static_cast<unsigned char>(text[value_end - 1]))) {
        --value_end;
    }
    if (value_end < value_start) {
        return errors::InvalidArgument(
            "Malformed metadata value in descriptor: ", text);
    }
    *out = text.substr(value_start, value_end - value_start);
    return OkStatus();
}

Status ParseMetadataObject(const std::string& json,
                           BlockDescriptorInfo* descriptor) {
    if (descriptor == nullptr) {
        return errors::InvalidArgument(
            "descriptor must not be null when parsing metadata");
    }
    descriptor->metadata.clear();
    const std::string key_literal = "\"metadata\"";
    size_t meta_pos = json.find(key_literal);
    if (meta_pos == std::string::npos) {
        return OkStatus();
    }
    size_t brace_start = json.find('{', meta_pos + key_literal.size());
    if (brace_start == std::string::npos) {
        return errors::InvalidArgument(
            "metadata object missing opening brace inside descriptor: ", json);
    }
    size_t brace_end = std::string::npos;
    if (!FindMatchingBrace(json, brace_start, &brace_end)) {
        return errors::InvalidArgument(
            "metadata object missing closing brace inside descriptor: ", json);
    }
    if (brace_end <= brace_start + 1) {
        return OkStatus();
    }
    const std::string body =
        json.substr(brace_start + 1, brace_end - brace_start - 1);
    size_t pos = 0;
    while (pos < body.size()) {
        SkipWhitespace(body, &pos);
        if (pos >= body.size()) {
            break;
        }
        if (body[pos] == ',') {
            ++pos;
            continue;
        }
        if (body[pos] == '}') {
            break;
        }
        if (body[pos] != '"') {
            LOG_FIRST_N(WARNING, 5)
                << "[FusedReasoningStack] Metadata key parse failed; "
                << "treating metadata as empty. Descriptor fragment: "
                << json;
            descriptor->metadata.clear();
            return OkStatus();
        }
        std::string key;
        Status key_status = ParseJsonStringToken(body, &pos, &key);
        if (!key_status.ok()) {
            LOG_FIRST_N(WARNING, 5)
                << "[FusedReasoningStack] Metadata key parse failed; "
                << "treating metadata as empty. Descriptor fragment: "
                << json << " error=" << key_status.message();
            descriptor->metadata.clear();
            return OkStatus();
        }
        SkipWhitespace(body, &pos);
        if (pos >= body.size() || body[pos] != ':') {
            return errors::InvalidArgument(
                "Malformed metadata entry (missing colon) in descriptor: ",
                json);
        }
        ++pos;
        SkipWhitespace(body, &pos);
        std::string value;
        Status value_status = ParseJsonValueToken(body, &pos, &value);
        if (!value_status.ok()) {
            LOG_FIRST_N(WARNING, 5)
                << "[FusedReasoningStack] Metadata value parse failed; "
                << "treating metadata as empty. Descriptor fragment: "
                << json << " error=" << value_status.message();
            descriptor->metadata.clear();
            return OkStatus();
        }
        // Preserve nested JSON blobs while trimming outer whitespace.
        size_t trimmed_start = 0;
        while (trimmed_start < value.size() &&
               std::isspace(static_cast<unsigned char>(value[trimmed_start]))) {
            ++trimmed_start;
        }
        size_t trimmed_end = value.size();
        while (trimmed_end > trimmed_start &&
               std::isspace(static_cast<unsigned char>(value[trimmed_end - 1]))) {
            --trimmed_end;
        }
        descriptor->metadata[key] = value.substr(trimmed_start, trimmed_end - trimmed_start);
        SkipWhitespace(body, &pos);
        if (pos < body.size() && body[pos] == ',') {
            ++pos;
        }
    }
    return OkStatus();
}

std::string StripWhitespace(const std::string& value) {
    std::string cleaned;
    cleaned.reserve(value.size());
    for (char c : value) {
        if (!std::isspace(static_cast<unsigned char>(c))) {
            cleaned.push_back(c);
        }
    }
    return cleaned;
}

bool ExtractQuotedField(const std::string& json,
                        const std::string& field,
                        std::string* out) {
    const std::string key = "\"" + field + "\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + key.size());
    if (pos == std::string::npos) return false;
    pos = json.find('"', pos);
    if (pos == std::string::npos) return false;
    size_t end = json.find('"', pos + 1);
    while (end != std::string::npos && end > 0 && json[end - 1] == '\\') {
        end = json.find('"', end + 1);
    }
    if (end == std::string::npos) return false;
    *out = json.substr(pos + 1, end - pos - 1);
    return true;
}

bool ExtractNumericField(const std::string& json,
                         const std::string& field,
                         int* out) {
    const std::string key = "\"" + field + "\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + key.size());
    if (pos == std::string::npos) return false;
    size_t start = json.find_first_of("-0123456789", pos + 1);
    if (start == std::string::npos) return false;
    size_t end = json.find_first_of(",}\n\r", start);
    std::string token = json.substr(start, end == std::string::npos ? std::string::npos : end - start);
    token = StripWhitespace(token);
    return absl::SimpleAtoi(token, out);
}

bool ExtractBoolField(const std::string& json,
                      const std::string& field,
                      bool* out) {
    const std::string key = "\"" + field + "\"";
    size_t pos = json.find(key);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + key.size());
    if (pos == std::string::npos) return false;
    size_t start = json.find_first_not_of(" \t\n\r", pos + 1);
    if (start == std::string::npos) return false;
    if (json.compare(start, 4, "true") == 0) {
        *out = true;
        return true;
    }
    if (json.compare(start, 5, "false") == 0) {
        *out = false;
        return true;
    }
    return false;
}

}  // namespace

Status ParseBlockDescriptor(const tstring& json_descriptor,
                            BlockDescriptorInfo* descriptor_out) {
    if (descriptor_out == nullptr) {
        return errors::InvalidArgument(
            "descriptor_out must not be null when parsing block descriptor");
    }
    const std::string json = std::string(json_descriptor.data(), json_descriptor.size());
    descriptor_out->raw_json = json_descriptor;
    std::string type_field;
    if (!ExtractQuotedField(json, "type", &type_field)) {
        return errors::InvalidArgument(
            "Block descriptor is missing required 'type' field: ", json);
    }
    int weight_count = 0;
    if (!ExtractNumericField(json, "weight_count", &weight_count)) {
        return errors::InvalidArgument(
            "Block descriptor is missing required 'weight_count' field: ", json);
    }
    bool stateful = false;
    if (!ExtractBoolField(json, "stateful", &stateful)) {
        return errors::InvalidArgument(
            "Block descriptor is missing required 'stateful' field: ", json);
    }
    descriptor_out->type = type_field;
    descriptor_out->weight_count = weight_count;
    descriptor_out->stateful = stateful;
    TF_RETURN_IF_ERROR(ParseMetadataObject(json, descriptor_out));
    return OkStatus();
}

/**
 * @brief Forward pass for stateless blocks using the shared dense-stack helper or TT helper.
 */
MatrixXf StatelessBlockForward(
    const BlockContext& ctx,
    const BlockDescriptorInfo& descriptor,
    const MatrixXf& input_seq,
    const OpInputList& weights,
    int& weight_idx,
    int num_weights_to_consume) {

    // First check if this block has TT layers
    tt_helpers::TTBlockInfo tt_info;
    std::string json_str(descriptor.raw_json.data(), descriptor.raw_json.size());
    LOG(WARNING) << "@@@@@ [TT DEBUG] Checking TT layers for block type="
                 << descriptor.type << " has tt_layers in JSON: "
                 << (json_str.find("tt_layers") != std::string::npos);
    Status tt_parse_status = tt_helpers::ParseTTBlockInfo(descriptor.raw_json, &tt_info);

    if (!tt_parse_status.ok()) {
        LOG(WARNING) << "@@@@@ [TT DEBUG] TT parse failed: " << tt_parse_status.message();
    } else if (!tt_info.has_tt_layers) {
        LOG(WARNING) << "@@@@@ [TT DEBUG] Block " << descriptor.type << " has no TT layers";
    }

    if (tt_parse_status.ok() && tt_info.has_tt_layers) {
        // This block uses TT decomposition - use TT kernel
        LOG_FIRST_N(INFO, 5) << "[FusedReasoningStack] Using TT kernel for block type="
                             << descriptor.type << " with " << tt_info.tt_layers.size()
                             << " TT layer(s)";

        // Validate TT weights
        bool all_valid = true;
        for (const auto& layer : tt_info.tt_layers) {
            std::string failure_reason;
            if (!tt_helpers::ValidateTTLayerWeights(weights, layer, &failure_reason, weight_idx)) {
                LOG(WARNING) << "[FusedReasoningStack] TT validation failed for layer "
                             << layer.name << ": " << failure_reason;
                all_valid = false;
                break;
            }
        }

        if (all_valid) {
            MatrixXf result = tt_helpers::RunTTBlockForward(
                ctx.op_context, input_seq, weights, tt_info, weight_idx);

            if (!ctx.op_context->status().ok()) {
                return MatrixXf();
            }
            weight_idx += num_weights_to_consume;
            return result;
        } else {
            // Validation failed, fall through to identity
            LOG_FIRST_N(WARNING, 10) << "[FusedReasoningStack] TT validation failed, using identity";
            weight_idx += std::max(0, num_weights_to_consume);
            return input_seq;
        }
    }

    // Not a TT block - use dense kernel
    std::string failure_reason;
    if (!stateless_internal::CanRunDenseStatelessBlock(input_seq, weights,
                                                       weight_idx,
                                                       num_weights_to_consume,
                                                       &failure_reason)) {
        LOG_FIRST_N(WARNING, 10) << "[FusedReasoningStack] Stateless block fallback to identity: "
                                 << failure_reason << " metadata=" << descriptor.raw_json;
        weight_idx += std::max(0, num_weights_to_consume);
        return input_seq;
    }

    MatrixXf transformed = stateless_internal::RunStatelessDenseForward(
        ctx.op_context, input_seq, weights, weight_idx, num_weights_to_consume);

    if (!ctx.op_context->status().ok()) {
        return MatrixXf();
    }
    weight_idx += num_weights_to_consume;
    return transformed;
}

/**
 * @brief Backward pass for stateless blocks using dense-stack recomputation.
 */
void StatelessBlockBackward(
    const BlockContext& ctx,
    const BlockDescriptorInfo& descriptor,
    MatrixXf& grad_adj_state,
    const MatrixXf& input_seq,
    const OpInputList& weights,
    int weight_idx_start,
    int num_weights_to_consume,
    std::vector<Tensor*>& grad_weights_tensors) {

    if (num_weights_to_consume <= 0) {
        return;
    }

    OpKernelContext* context = ctx.op_context;

    std::vector<MatrixXf> layer_inputs;
    std::vector<MatrixXf> pre_activations;
    MatrixXf block_output = stateless_internal::RunStatelessDenseForward(
        context, input_seq, weights, weight_idx_start, num_weights_to_consume,
        &layer_inputs, &pre_activations);

    if (!context->status().ok()) {
        return;
    }

    const int pair_count = num_weights_to_consume / 2;
    if (layer_inputs.size() != pair_count + 1 ||
        pre_activations.size() != pair_count) {
        context->CtxFailure(errors::Internal(
            "Stateless block cache invariant violated during backward pass."));
        return;
    }

    MatrixXf residual_grad = grad_adj_state;
    MatrixXf grad_current = residual_grad;

    static absl::Mutex grad_mutex;

    for (int layer = pair_count - 1; layer >= 0; --layer) {
        const int weight_offset = weight_idx_start + 2 * layer;
        if (weight_offset + 1 >= grad_weights_tensors.size()) {
            context->CtxFailure(errors::InvalidArgument(
                "Gradient tensor list is too small for stateless block layer ",
                layer));
            return;
        }

        const Tensor& weight_tensor = weights[weight_offset];
        const Tensor& bias_tensor = weights[weight_offset + 1];

        stateless_internal::FlattenedWeightShape weight_shape;
        if (!stateless_internal::ResolveWeightShape(context, weight_tensor, layer,
                                                    &weight_shape)) {
            return;
        }

        Map<const MatrixXf> weight_map(weight_tensor.flat<float>().data(),
                                       weight_shape.input_dim,
                                       weight_shape.output_dim);
        Map<MatrixXf> grad_weight_map(
            grad_weights_tensors[weight_offset]->flat<float>().data(),
            weight_shape.input_dim,
            weight_shape.output_dim);
        Map<VectorXf> grad_bias_map(
            grad_weights_tensors[weight_offset + 1]->flat<float>().data(),
            weight_shape.output_dim);

        const MatrixXf& layer_input = layer_inputs[layer];
        MatrixXf grad_weight_contrib = layer_input.transpose() * grad_current;
        VectorXf grad_bias_contrib = grad_current.colwise().sum();

        {
            absl::MutexLock lock(&grad_mutex);
            grad_weight_map += grad_weight_contrib;
            grad_bias_map += grad_bias_contrib;
        }

        MatrixXf grad_prev = grad_current * weight_map.transpose();

        if (layer > 0) {
            const MatrixXf& pre_act_prev = pre_activations[layer - 1];
            // SIMD-optimized GELU gradient (Phase 11)
            MatrixXf activation_grad(pre_act_prev.rows(), pre_act_prev.cols());
            stateless_internal::gelu_grad_vectorized(pre_act_prev.data(),
                                                     activation_grad.data(),
                                                     pre_act_prev.rows() * pre_act_prev.cols());
            grad_current = grad_prev.cwiseProduct(activation_grad);
        } else {
            grad_adj_state = grad_prev + residual_grad;
        }
    }
}

/**
 * @brief Forward pass for the stateful TimeCrystalSequenceBlock.
 * This function unrolls the HNN dynamics over the entire sequence for each batch item.
 */
void TimeCrystalSequenceForward(
    const BlockContext& ctx,
    MatrixXf& current_sequence,
    const OpInputList& initial_states,
    const OpInputList& weights,
    int& state_idx,
    int& weight_idx,
    std::vector<Tensor>& final_state_tensors,
    std::vector<std::vector<HNNForwardState>>& hnn_forward_states) {

    const int64 total_rows = ctx.batch_size * ctx.seq_len_combined;
    const Tensor& h_padded_tensor = initial_states[state_idx];

    // Unpack HNN state (q and p) from the padded state tensor.
    const int64 D_state_padded_len = h_padded_tensor.dim_size(1);
    const int64 D_mamba_state = h_padded_tensor.dim_size(2);
    const int64 D_state_total = D_state_padded_len * D_mamba_state;
    const int64 D_state = D_state_total / 2;

    Tensor q_init_tensor(DT_FLOAT, TensorShape({ctx.batch_size, D_state}));
    Tensor p_init_tensor(DT_FLOAT, TensorShape({ctx.batch_size, D_state}));

    const Map<const MatrixXf, RowMajor> h_padded_flat(
        h_padded_tensor.flat<float>().data(), ctx.batch_size, D_state_total);

    for(int b=0; b < ctx.batch_size; ++b) {
        Map<VectorXf>(q_init_tensor.flat<float>().data() + b * D_state, D_state) = h_padded_flat.row(b).head(D_state).transpose();
        Map<VectorXf>(p_init_tensor.flat<float>().data() + b * D_state, D_state) = h_padded_flat.row(b).segment(D_state, D_state).transpose();
    }

    // Consume 9 weights for the HNN.
    const Tensor& W1 = weights[weight_idx];
    const float b3_scalar = weights[weight_idx+5].scalar<float>()();
    const float epsilon_param_val = weights[weight_idx+6].scalar<float>()();
    const float epsilon = std::min(1.0f, 0.01f + 0.99f * std::tanh(epsilon_param_val));

    // Get dimensions.
    const int64_t D_in = 2 * D_state + ctx.d_embed;
    const int64_t D_h = W1.dim_size(1);
    const int64_t D_output = weights[weight_idx+8].dim_size(0);

    // Map weight tensors to Eigen for efficient computation.
    // Note: The `helpers.h` file contains the declarations for `compute_H_and_intermediates`
    // and `compute_dH_dz`, which are used here. This file provides their implementation
    // details for the forward pass of the TimeCrystal block.

    Map<const MatrixXf> W1_map(weights[weight_idx+0].flat<float>().data(), D_in, D_h);
    Map<const VectorXf> b1_map(weights[weight_idx+1].flat<float>().data(), D_h);
    Map<const MatrixXf> W2_map(weights[weight_idx+2].flat<float>().data(), D_h, D_h);
    Map<const VectorXf> b2_map(weights[weight_idx+3].flat<float>().data(), D_h);
    Map<const MatrixXf> W3_map(weights[weight_idx+4].flat<float>().data(), D_h, 1);
    Map<const MatrixXf> W_out_map(weights[weight_idx+7].flat<float>().data(), 2 * D_state, D_output);
    Map<const VectorXf> b_out_map(weights[weight_idx+8].flat<float>().data(), D_output);

    MatrixXf temp_output_seq_eigen = MatrixXf::Zero(total_rows, D_output);
    Tensor final_q_tensor(DT_FLOAT, TensorShape({ctx.batch_size, D_state}));
    // --- START: DEFINITIVE FIX ---
    // Allocate tensors for the Hamiltonian values, which were missing.
    Tensor h_initial_seq_tensor(DT_FLOAT, TensorShape({ctx.batch_size, ctx.seq_len_combined}));
    Tensor h_final_seq_tensor(DT_FLOAT, TensorShape({ctx.batch_size, ctx.seq_len_combined}));
    Tensor drift_seq_tensor(DT_FLOAT, TensorShape({ctx.batch_size, ctx.seq_len_combined}));
    // --- END: DEFINITIVE FIX ---
    Tensor final_p_tensor(DT_FLOAT, TensorShape({ctx.batch_size, D_state}));

    // Main sequence unrolling loop, parallelized over the batch.
    saguaro::parallel::ForRange(
        0, static_cast<std::size_t>(ctx.batch_size), 1,
        [&](std::size_t range_begin, std::size_t range_end) {
            for (std::size_t idx = range_begin; idx < range_end; ++idx) {
                const int b = static_cast<int>(idx);
                VectorXf q_t = Map<const VectorXf>(q_init_tensor.flat<float>().data() + b * D_state, D_state);
                VectorXf p_t = Map<const VectorXf>(p_init_tensor.flat<float>().data() + b * D_state, D_state);

                for (int l = 0; l < ctx.seq_len_combined; ++l) {
                    const int64_t row_idx = b * ctx.seq_len_combined + l;
                    VectorXf x_l = current_sequence.row(row_idx).transpose();

                    // Store all intermediates for this step for the backward pass.
                    HNNForwardState& state = hnn_forward_states[b][l];
                    state.q_t = q_t; state.p_t = p_t; state.x_t = x_l;

                    // --- HNN Leapfrog Step (Time Crystal Dynamics) ---
                    // --- START: DEFINITIVE FIX ---
                    // Compute and store the initial Hamiltonian value for this step.
                    VectorXf z(D_in); z << q_t, p_t, x_l;
                    state.int1 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    h_initial_seq_tensor.matrix<float>()(b, l) = state.int1.H;
                    // --- END: DEFINITIVE FIX ---

                    state.dH_dz1 = compute_dH_dz(state.int1, W1_map, W2_map, W3_map);
                    state.p_half = p_t - (epsilon / 2.0f) * state.dH_dz1.head(D_state);

                    z.segment(D_state, D_state) = state.p_half;
                    state.int2 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    state.dH_dz2 = compute_dH_dz(state.int2, W1_map, W2_map, W3_map);
                    VectorXf q_next = q_t + epsilon * state.dH_dz2.segment(D_state, D_state);

                    z.head(D_state) = q_next;
                    state.int3 = compute_H_and_intermediates(z, W1_map, b1_map, W2_map, b2_map, W3_map, b3_scalar);
                    // --- START: DEFINITIVE FIX ---
                    // Compute and store the final Hamiltonian value for this step.
                    h_final_seq_tensor.matrix<float>()(b, l) = state.int3.H;
                    // --- END: DEFINITIVE FIX ---

                    state.dH_dz3 = compute_dH_dz(state.int3, W1_map, W2_map, W3_map);
                    VectorXf p_next = state.p_half - (epsilon / 2.0f) * state.dH_dz3.head(D_state);

                    // --- Output Projection ---
                    VectorXf final_state_vec(2 * D_state);
                    final_state_vec << q_next, p_next;
                    VectorXf output_l = (final_state_vec.transpose() * W_out_map).transpose() + b_out_map;

                    // --- Update State, Store Intermediates, and Write Output ---
                    state.q_next = q_next;
                    state.p_next = p_next;
                    q_t = q_next;
                    p_t = p_next;
                    temp_output_seq_eigen.row(row_idx) = output_l.transpose();
                }

                // Store the final state of the sequence for this batch item.
                Map<VectorXf>(final_q_tensor.flat<float>().data() + b * D_state, D_state) = q_t;
                Map<VectorXf>(final_p_tensor.flat<float>().data() + b * D_state, D_state) = p_t;
            }
        });

    current_sequence = temp_output_seq_eigen;

    const auto h_initial_seq = h_initial_seq_tensor.matrix<float>();
    const auto h_final_seq = h_final_seq_tensor.matrix<float>();
    auto drift_seq = drift_seq_tensor.matrix<float>();
    constexpr float kDriftDenomEpsilon = 1e-6f;
    for (int b = 0; b < ctx.batch_size; ++b) {
        for (int l = 0; l < ctx.seq_len_combined; ++l) {
            const float h_initial = h_initial_seq(b, l);
            const float h_final = h_final_seq(b, l);
            drift_seq(b, l) =
                std::abs(h_final - h_initial) /
                (std::abs(h_initial) + kDriftDenomEpsilon);
        }
    }

    // --- START: DEFINITIVE FIX ---
    // The final state tensors now include the Hamiltonian sequences.
    // These are packed into the `final_state_tensors` vector which is passed by reference.
    final_state_tensors.push_back(h_initial_seq_tensor);
    final_state_tensors.push_back(h_final_seq_tensor);
    final_state_tensors.push_back(drift_seq_tensor);
    // --- END: DEFINITIVE FIX ---
    // Repack the final (q, p) states into the padded output state tensor.
    Tensor& final_h_padded_tensor = final_state_tensors[state_idx];
    Map<MatrixXf, RowMajor> final_h_padded_flat(
        final_h_padded_tensor.flat<float>().data(), ctx.batch_size, D_state_total);

    for (int b=0; b < ctx.batch_size; ++b) {
        final_h_padded_flat.row(b).head(D_state) = Map<const VectorXf>(final_q_tensor.flat<float>().data() + b * D_state, D_state).transpose();
        final_h_padded_flat.row(b).segment(D_state, D_state) = Map<const VectorXf>(final_p_tensor.flat<float>().data() + b * D_state, D_state).transpose();
    }

    // Advance state and weight indices for the main loop.
    state_idx += 2;
    weight_idx += 9;
}


/**
 * @brief Backward pass for the TimeCrystalSequenceBlock.
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
    std::vector<Tensor*>& grad_weights_tensors) {

    // Get dimensions and parameters.
    const Tensor& W1 = weights[weight_idx_bwd];
    const int64_t D_state = (W1.dim_size(0) - ctx.d_embed) / 2;
    const int64_t D_in_hnn = W1.dim_size(0);
    const int64_t D_h = W1.dim_size(1);
    const int64_t D_output = weights[weight_idx_bwd+8].dim_size(0);

    const float b3_scalar = weights[weight_idx_bwd+5].scalar<float>()();
    const float epsilon_param_val = weights[weight_idx_bwd+6].scalar<float>()();
    const float epsilon = std::min(1.0f, 0.01f + 0.99f * std::tanh(epsilon_param_val));

    // Map weight tensors.
    Map<const MatrixXf> W1_map(W1.flat<float>().data(), D_in_hnn, D_h);
    Map<const VectorXf> b1_map(weights[weight_idx_bwd+1].flat<float>().data(), D_h);
    Map<const MatrixXf> W2_map(weights[weight_idx_bwd+2].flat<float>().data(), D_h, D_h);
    Map<const VectorXf> b2_map(weights[weight_idx_bwd+3].flat<float>().data(), D_h);
    Map<const MatrixXf> W3_map(weights[weight_idx_bwd+4].flat<float>().data(), D_h, 1);
    Map<const MatrixXf> W_out_map(weights[weight_idx_bwd+7].flat<float>().data(), 2 * D_state, D_output);

    // Global gradient accumulators for weights.
    MatrixXf grad_W1_acc = MatrixXf::Zero(D_in_hnn, D_h);
    VectorXf grad_b1_acc = VectorXf::Zero(D_h);
    MatrixXf grad_W2_acc = MatrixXf::Zero(D_h, D_h);
    VectorXf grad_b2_acc = VectorXf::Zero(D_h);
    MatrixXf grad_W3_acc = MatrixXf::Zero(D_h, 1);
    float grad_b3_acc = 0.0f;
    MatrixXf grad_W_out_acc = MatrixXf::Zero(2 * D_state, D_output);
    VectorXf grad_b_out_acc = VectorXf::Zero(D_output);
    float grad_epsilon_param_acc = 0.0f;

    MatrixXf grad_input_from_hnn = MatrixXf::Zero(ctx.batch_size * ctx.seq_len_combined, ctx.d_embed);

    // --- Main Backward Loop (BPTT via Adjoint Sensitivity) ---
    saguaro::parallel::SpinMutex accumulator_mutex;
    saguaro::parallel::ForRange(
        0, static_cast<std::size_t>(ctx.batch_size), 1,
        [&](std::size_t range_begin, std::size_t range_end) {
            for (std::size_t idx = range_begin; idx < range_end; ++idx) {
                const int b = static_cast<int>(idx);
                // Initialize adjoint state from upstream gradients.
                VectorXf grad_q = Map<const VectorXf>(grad_final_states[state_idx_bwd].flat<float>().data() + b * D_state, D_state);
                VectorXf grad_p = Map<const VectorXf>(grad_final_states[state_idx_bwd+1].flat<float>().data() + b * D_state, D_state);

                // Thread-local accumulators for weight gradients.
                MatrixXf local_grad_W1 = MatrixXf::Zero(D_in_hnn, D_h);
                VectorXf local_grad_b1 = VectorXf::Zero(D_h);
                MatrixXf local_grad_W2 = MatrixXf::Zero(D_h, D_h);
                VectorXf local_grad_b2 = VectorXf::Zero(D_h);
                MatrixXf local_grad_W3 = MatrixXf::Zero(D_h, 1);
                float local_grad_b3 = 0.0f;
                MatrixXf local_grad_W_out = MatrixXf::Zero(2 * D_state, D_output);
                VectorXf local_grad_b_out = VectorXf::Zero(D_output);
                float local_grad_epsilon_param = 0.0f;

                // Iterate backwards through the sequence.
                for (int l = ctx.seq_len_combined - 1; l >= 0; --l) {
                    const HNNForwardState& state = hnn_forward_states[b][l];
                    Map<const VectorXf> grad_output_l(grad_adj_state.data() + (b * ctx.seq_len_combined + l) * ctx.d_embed, ctx.d_embed);

                    // --- Backprop through Output Projection ---
                    VectorXf current_final_state(2 * D_state);
                    current_final_state << state.q_next, state.p_next;
                    local_grad_W_out += current_final_state * grad_output_l.transpose();
                    local_grad_b_out += grad_output_l;
                    VectorXf grad_final_state = W_out_map * grad_output_l;
                    grad_q += grad_final_state.head(D_state);
                    grad_p += grad_final_state.tail(D_state);

                    // --- Backprop through Leapfrog Integrator (in reverse) ---
                    VectorXf grad_p_half = grad_p;
                    float grad_H3_scalar = grad_p.dot(-0.5f * epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int3, W1_map, W2_map, W3_map, grad_H3_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                    VectorXf grad_z3 = state.dH_dz3 * grad_H3_scalar;

                    grad_q += grad_z3.head(D_state);
                    grad_p_half += grad_z3.segment(D_state, D_state);
                    grad_input_from_hnn.row(b * ctx.seq_len_combined + l) += grad_z3.tail(ctx.d_embed).transpose();

                    float grad_H2_scalar = grad_q.dot(epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int2, W1_map, W2_map, W3_map, grad_H2_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                    VectorXf grad_z2 = state.dH_dz2 * grad_H2_scalar;
                    grad_q += grad_z2.head(D_state);
                    grad_p_half += grad_z2.segment(D_state, D_state);
                    grad_input_from_hnn.row(b * ctx.seq_len_combined + l) += grad_z2.tail(ctx.d_embed).transpose();

                    VectorXf grad_p_t = grad_p_half;
                    float grad_H1_scalar = grad_p_half.dot(-0.5f * epsilon * VectorXf::Ones(D_state));
                    backprop_dH_dweights(state.int1, W1_map, W2_map, W3_map, grad_H1_scalar, local_grad_W1, local_grad_b1, local_grad_W2, local_grad_b2, local_grad_W3, local_grad_b3);
                    VectorXf grad_z1 = state.dH_dz1 * grad_H1_scalar;

                    // Propagate adjoint state to the previous time step (q_t, p_t).
                    grad_q = grad_q + grad_z1.head(D_state);
                    grad_p = grad_p_t + grad_z1.segment(D_state, D_state);
                    grad_input_from_hnn.row(b * ctx.seq_len_combined + l) += grad_z1.tail(ctx.d_embed).transpose();

                    // --- Adjoint of evolution_time (epsilon) ---
                    float grad_eps = 0;
                    grad_eps -= (grad_p.transpose() * state.dH_dz3.head(D_state) / 2.0f)(0, 0);
                    grad_eps += (grad_q.transpose() * state.dH_dz2.segment(D_state, D_state))(0, 0);
                    grad_eps -= (grad_p_half.transpose() * state.dH_dz1.head(D_state) / 2.0f)(0, 0);
                    local_grad_epsilon_param += grad_eps * (0.99f * (1.0f - std::pow(std::tanh(epsilon_param_val), 2)));
                }

                // Write initial state gradients for this batch item.
                Map<VectorXf>(grad_initial_states_tensors[state_idx_bwd]->flat<float>().data() + b * D_state, D_state) = grad_q;
                Map<VectorXf>(grad_initial_states_tensors[state_idx_bwd+1]->flat<float>().data() + b * D_state, D_state) = grad_p;

                // Accumulate thread-local gradients into global accumulators.
                {
                    saguaro::parallel::SpinLockGuard lock(accumulator_mutex);
                    grad_W1_acc += local_grad_W1;
                    grad_b1_acc += local_grad_b1;
                    grad_W2_acc += local_grad_W2;
                    grad_b2_acc += local_grad_b2;
                    grad_W3_acc += local_grad_W3;
                    grad_b3_acc += local_grad_b3;
                    grad_W_out_acc += local_grad_W_out;
                    grad_b_out_acc += local_grad_b_out;
                    grad_epsilon_param_acc += local_grad_epsilon_param;
                }
            }
        });

    // Update the adjoint state for the next block in the backward pass.
    grad_adj_state = grad_input_from_hnn;

    // Write final accumulated gradients to output tensors.
    Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+0]->flat<float>().data(), D_in_hnn, D_h) = grad_W1_acc;
    Map<VectorXf>(grad_weights_tensors[weight_idx_bwd+1]->flat<float>().data(), D_h) = grad_b1_acc;
    Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+2]->flat<float>().data(), D_h, D_h) = grad_W2_acc;
    Map<VectorXf>(grad_weights_tensors[weight_idx_bwd+3]->flat<float>().data(), D_h) = grad_b2_acc;
    Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+4]->flat<float>().data(), D_h, 1) = grad_W3_acc;
    grad_weights_tensors[weight_idx_bwd+5]->scalar<float>()() = grad_b3_acc;
    grad_weights_tensors[weight_idx_bwd+6]->scalar<float>()() = grad_epsilon_param_acc;
    Map<MatrixXf>(grad_weights_tensors[weight_idx_bwd+7]->flat<float>().data(), 2 * D_state, D_output) = grad_W_out_acc;
    Map<VectorXf>(grad_weights_tensors[weight_idx_bwd+8]->flat<float>().data(), D_output) = grad_b_out_acc;
}

} // namespace tensorflow
