// saguaro.native/ops/fused_speculative_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file fused_speculative_op.cc
 * @brief Speculative Decoding TensorFlow Op.
 *
 * Accelerated speculative decoding for inference:
 *   - Parallel verification of draft tokens
 *   - Rejection sampling with target/draft probability ratios
 *   - Efficient token acceptance counting
 */

#include "fused_speculative_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <random>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedSpeculativeVerify")
    .Input("target_logits: float32")  // [batch, num_spec, vocab]
    .Input("draft_probs: float32")    // [batch, num_spec, vocab]
    .Input("draft_tokens: int32")     // [batch, num_spec]
    .Output("num_accepted: int32")    // [batch]
    .Output("bonus_token: int32")     // [batch]
    .Attr("temperature: float = 1.0")
    .Attr("top_k: int = 50")
    .Attr("use_sampling: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle target_logits = c->input(0);
        auto batch = c->Dim(target_logits, 0);
        c->set_output(0, c->MakeShape({batch}));
        c->set_output(1, c->MakeShape({batch}));
        return Status();
    })
    .Doc(R"doc(
Fused speculative decoding verification.
Verifies draft tokens against target distribution using rejection sampling.
)doc");

REGISTER_OP("FusedSpeculativeSample")
    .Input("logits: float32")  // [batch, vocab]
    .Output("tokens: int32")   // [batch]
    .Output("probs: float32")  // [batch, vocab]
    .Attr("temperature: float = 1.0")
    .Attr("top_k: int = 50")
    .Attr("use_sampling: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle logits = c->input(0);
        auto batch = c->Dim(logits, 0);
        c->set_output(0, c->MakeShape({batch}));
        c->set_output(1, logits);
        return Status();
    })
    .Doc(R"doc(
Sample tokens from logits with temperature scaling and top-k filtering.
)doc");

// =============================================================================
// SPECULATIVE VERIFY KERNEL
// =============================================================================

class FusedSpeculativeVerifyOp : public OpKernel {
 public:
    explicit FusedSpeculativeVerifyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("top_k", &top_k_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_sampling", &use_sampling_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& target_logits = ctx->input(0);
        const Tensor& draft_probs = ctx->input(1);
        const Tensor& draft_tokens = ctx->input(2);

        const int64_t batch_size = target_logits.dim_size(0);
        const int64_t num_speculative = target_logits.dim_size(1);
        const int64_t vocab_size = target_logits.dim_size(2);

        // Allocate outputs
        Tensor* num_accepted = nullptr;
        Tensor* bonus_token = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &num_accepted));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({batch_size}), &bonus_token));

        const float* target_data = target_logits.flat<float>().data();
        const float* draft_data = draft_probs.flat<float>().data();
        const int32_t* token_data = draft_tokens.flat<int32_t>().data();
        int32_t* accepted_data = num_accepted->flat<int32_t>().data();
        int32_t* bonus_data = bonus_token->flat<int32_t>().data();

        // Random number generator (thread-local for parallel processing)
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        #pragma omp parallel
        {
            std::mt19937 rng(seed + omp_get_thread_num());
            std::vector<float> target_probs(vocab_size);
            std::vector<float> logits_copy(vocab_size);

            #pragma omp for
            for (int64_t b = 0; b < batch_size; ++b) {
                int num_acc = 0;

                for (int64_t s = 0; s < num_speculative; ++s) {
                    // Get target logits for this position
                    const float* tgt = target_data + (b * num_speculative + s) * vocab_size;
                    const float* drf = draft_data + (b * num_speculative + s) * vocab_size;
                    int32_t token = token_data[b * num_speculative + s];

                    // Apply temperature and compute target probs
                    std::copy(tgt, tgt + vocab_size, logits_copy.data());
                    saguaro::ops::speculative_temperature_scale(
                        logits_copy.data(), vocab_size, temperature_);
                    
                    if (top_k_ > 0) {
                        saguaro::ops::speculative_top_k_filter(
                            logits_copy.data(), vocab_size, top_k_);
                    }
                    
                    saguaro::ops::speculative_softmax(
                        logits_copy.data(), target_probs.data(), vocab_size);

                    // Get probabilities for drafted token
                    float p_target = (token >= 0 && token < vocab_size) 
                        ? target_probs[token] : 0.0f;
                    float p_draft = (token >= 0 && token < vocab_size) 
                        ? drf[token] : 0.0f;

                    // Acceptance criterion
                    float accept_ratio = std::min(1.0f, p_target / (p_draft + 1e-8f));

                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    float u = dist(rng);

                    if (u >= accept_ratio) {
                        // Reject - sample bonus token from target
                        if (use_sampling_) {
                            bonus_data[b] = saguaro::ops::speculative_sample_token(
                                target_probs.data(), vocab_size, rng);
                        } else {
                            bonus_data[b] = saguaro::ops::speculative_argmax(
                                target_probs.data(), vocab_size);
                        }
                        break;
                    }
                    num_acc++;
                }

                accepted_data[b] = num_acc;

                // If all accepted, sample bonus from last position + 1
                if (num_acc == static_cast<int>(num_speculative)) {
                    // Use last target probs computed
                    if (use_sampling_) {
                        std::vector<float> last_probs(vocab_size);
                        const float* last_tgt = target_data + 
                            (b * num_speculative + num_speculative - 1) * vocab_size;
                        std::copy(last_tgt, last_tgt + vocab_size, logits_copy.data());
                        saguaro::ops::speculative_temperature_scale(
                            logits_copy.data(), vocab_size, temperature_);
                        saguaro::ops::speculative_softmax(
                            logits_copy.data(), last_probs.data(), vocab_size);
                        bonus_data[b] = saguaro::ops::speculative_sample_token(
                            last_probs.data(), vocab_size, rng);
                    } else {
                        bonus_data[b] = saguaro::ops::speculative_argmax(
                            target_data + (b * num_speculative + num_speculative - 1) * vocab_size,
                            vocab_size);
                    }
                }
            }
        }
    }

 private:
    float temperature_;
    int top_k_;
    bool use_sampling_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedSpeculativeVerify").Device(DEVICE_CPU),
    FusedSpeculativeVerifyOp);

// =============================================================================
// SPECULATIVE SAMPLE KERNEL
// =============================================================================

class FusedSpeculativeSampleOp : public OpKernel {
 public:
    explicit FusedSpeculativeSampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("top_k", &top_k_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_sampling", &use_sampling_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& logits = ctx->input(0);

        const int64_t batch_size = logits.dim_size(0);
        const int64_t vocab_size = logits.dim_size(1);

        // Allocate outputs
        Tensor* tokens = nullptr;
        Tensor* probs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({batch_size}), &tokens));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, logits.shape(), &probs));

        const float* logits_data = logits.flat<float>().data();
        int32_t* tokens_data = tokens->flat<int32_t>().data();
        float* probs_data = probs->flat<float>().data();

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        #pragma omp parallel
        {
            std::mt19937 rng(seed + omp_get_thread_num());
            std::vector<float> logits_copy(vocab_size);

            #pragma omp for
            for (int64_t b = 0; b < batch_size; ++b) {
                const float* src = logits_data + b * vocab_size;
                float* dst = probs_data + b * vocab_size;

                std::copy(src, src + vocab_size, logits_copy.data());

                saguaro::ops::speculative_temperature_scale(
                    logits_copy.data(), vocab_size, temperature_);

                if (top_k_ > 0) {
                    saguaro::ops::speculative_top_k_filter(
                        logits_copy.data(), vocab_size, top_k_);
                }

                saguaro::ops::speculative_softmax(logits_copy.data(), dst, vocab_size);

                if (use_sampling_) {
                    tokens_data[b] = saguaro::ops::speculative_sample_token(
                        dst, vocab_size, rng);
                } else {
                    tokens_data[b] = saguaro::ops::speculative_argmax(dst, vocab_size);
                }
            }
        }
    }

 private:
    float temperature_;
    int top_k_;
    bool use_sampling_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedSpeculativeSample").Device(DEVICE_CPU),
    FusedSpeculativeSampleOp);

}  // namespace tensorflow
