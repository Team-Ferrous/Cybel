// saguaro/native/ops/fused_text_tokenizer_op.cc
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
 * @file fused_text_tokenizer_op.cc
 * @brief TensorFlow op registration for fused text tokenization.
 *
 * Registers the following ops:
 * - FusedTextTokenize: SIMD UTF-8 byte tokenization + optional trie merge
 * - FusedTextTokenizeBatch: Batched parallel tokenization
 * - SuperwordTrieBuild: Build trie from n-gram/superword mappings
 * - StreamingNgramCount: Count n-grams from token sequences
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_op_kernel.h"

#include "fused_text_tokenizer_op.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using namespace saguaro::ops::text_tokenizer;

// =============================================================================
// SUPERWORD TRIE RESOURCE
// =============================================================================

/**
 * @brief ResourceBase wrapper for SuperwordTrie to enable stateful ops.
 *
 * MEMORY SAFETY:
 * - Uses unique_ptr to ensure proper destruction order
 * - Virtual destructor for proper polymorphic cleanup
 * - Thread-safe for TensorFlow's resource manager
 */
class SuperwordTrieResource : public ResourceBase {
public:
    SuperwordTrieResource() : trie_(std::make_unique<SuperwordTrie>()) {}
    
    // Virtual destructor required for ResourceBase inheritance
    ~SuperwordTrieResource() override {
        // Explicitly clear before destruction to ensure proper cleanup order
        if (trie_) {
            trie_->clear();
            trie_.reset();
        }
    }
    
    // Disable copy
    SuperwordTrieResource(const SuperwordTrieResource&) = delete;
    SuperwordTrieResource& operator=(const SuperwordTrieResource&) = delete;

    SuperwordTrie& trie() { return *trie_; }
    const SuperwordTrie& trie() const { return *trie_; }
    
    // Check if trie is valid
    bool valid() const { return trie_ != nullptr; }

    string DebugString() const override {
        if (trie_) {
            return strings::StrCat("SuperwordTrieResource with ", trie_->node_count(), " nodes");
        }
        return "SuperwordTrieResource (uninitialized)";
    }

private:
    std::unique_ptr<SuperwordTrie> trie_;
};

// =============================================================================
// OP REGISTRATION: FusedTextTokenize
// =============================================================================

REGISTER_OP("SAGUAROTextTokenize")
    .Input("input_text: string")
    .Input("trie_handle: resource")
    .Output("output_tokens: int32")
    .Output("output_length: int32")
    .Attr("byte_offset: int = 32")
    .Attr("add_special_tokens: bool = true")
    .Attr("inject_thinking: bool = false")
    .Attr("max_length: int = 131072")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int max_length;
        TF_RETURN_IF_ERROR(c->GetAttr("max_length", &max_length));
        c->set_output(0, c->MakeShape({max_length}));
        c->set_output(1, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
SIMD-optimized UTF-8 text tokenization with optional superword merging.

Converts UTF-8 text to token IDs using vectorized byte encoding, then
optionally applies superword merges using a pre-built trie.

input_text: Scalar string tensor containing UTF-8 text.
trie_handle: Handle to a SuperwordTrieResource (or empty resource for no merging).
output_tokens: Int32 tensor of token IDs, padded to max_length.
output_length: Scalar int32 indicating actual number of tokens.
byte_offset: Offset added to each byte value (default: 32).
add_special_tokens: Whether to add CLS (start) and EOS (end) tokens.
max_length: Maximum output length, sequences are truncated to this.
)doc");

// =============================================================================
// OP KERNEL: FusedTextTokenize
// =============================================================================

class FusedTextTokenizeOp : public OpKernel {
public:
    explicit FusedTextTokenizeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("byte_offset", &byte_offset_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("add_special_tokens", &add_special_tokens_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("inject_thinking", &inject_thinking_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_length", &max_length_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get input text
        const Tensor& input_tensor = ctx->input(0);
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(input_tensor.shape()),
                    errors::InvalidArgument("Input must be a scalar string tensor"));

        const tstring& input_text = input_tensor.scalar<tstring>()();
        const uint8_t* utf8_data = reinterpret_cast<const uint8_t*>(input_text.data());
        int utf8_len = static_cast<int>(input_text.size());

        // Get trie resource (may be empty)
        SuperwordTrieResource* trie_resource = nullptr;
        if (ctx->input(1).NumElements() > 0) {
            auto status = LookupResource(ctx, HandleFromInput(ctx, 1), &trie_resource);
            // OK if not found - just means no merging
        }
        core::ScopedUnref unref_trie(trie_resource);

        // Allocate outputs
        Tensor* output_tokens = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({max_length_}), &output_tokens));
        Tensor* output_length = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &output_length));

        int32_t* out_ptr = output_tokens->flat<int32_t>().data();

        // Allocate scratch buffer for intermediate tokens
        std::vector<int32_t> scratch(utf8_len + 2);

        int final_len;
        if (trie_resource != nullptr) {
            // Tokenize with trie merging
            final_len = fused_tokenize_and_merge(
                utf8_data,
                utf8_len,
                trie_resource->trie(),
                out_ptr,
                max_length_,
                byte_offset_,
                add_special_tokens_,
                scratch.data(),
                inject_thinking_
            );
        } else {
            // Just tokenize without merging
            final_len = text_tokenize_utf8_complex(
                utf8_data,
                utf8_len,
                out_ptr,
                byte_offset_,
                add_special_tokens_,
                inject_thinking_
            );
        }

        // Truncate if needed
        if (final_len > max_length_) {
            final_len = max_length_;
            if (add_special_tokens_) {
                out_ptr[max_length_ - 1] = TEXT_TOK_EOS_ID;
            }
        }

        // Pad remaining with PAD tokens
        for (int i = final_len; i < max_length_; ++i) {
            out_ptr[i] = TEXT_TOK_PAD_ID;
        }

        output_length->scalar<int32_t>()() = final_len;
    }

private:
    int byte_offset_;
    bool add_special_tokens_;
    bool inject_thinking_;
    int max_length_;
};

REGISTER_KERNEL_BUILDER(Name("SAGUAROTextTokenize").Device(DEVICE_CPU), FusedTextTokenizeOp);

// =============================================================================
// OP REGISTRATION: FusedTextTokenizeBatch
// =============================================================================

REGISTER_OP("SAGUAROTextTokenizeBatch")
    .Input("input_texts: string")
    .Input("trie_handle: resource")
    .Output("output_tokens: int32")
    .Output("output_lengths: int32")
    .Attr("byte_offset: int = 32")
    .Attr("add_special_tokens: bool = true")
    .Attr("inject_thinking: bool = false")
    .Attr("max_length: int = 131072")
    .Attr("num_threads: int = 0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch_dim = c->Dim(c->input(0), 0);
        int max_length;
        TF_RETURN_IF_ERROR(c->GetAttr("max_length", &max_length));
        c->set_output(0, c->MakeShape({batch_dim, max_length}));
        c->set_output(1, c->MakeShape({batch_dim}));
        return Status();
    })
    .Doc(R"doc(
Batched parallel SIMD text tokenization.

Processes multiple texts in parallel using multi-threading for maximum throughput.

input_texts: 1D string tensor of UTF-8 texts.
trie_handle: Handle to SuperwordTrieResource (or empty for no merging).
output_tokens: Int32 tensor [batch_size, max_length] of token IDs.
output_lengths: Int32 tensor [batch_size] of actual lengths.
num_threads: Number of parallel threads (0 = auto-detect).
)doc");

// =============================================================================
// OP KERNEL: FusedTextTokenizeBatch
// =============================================================================

class FusedTextTokenizeBatchOp : public OpKernel {
public:
    explicit FusedTextTokenizeBatchOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("byte_offset", &byte_offset_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("add_special_tokens", &add_special_tokens_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("inject_thinking", &inject_thinking_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_length", &max_length_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_threads", &num_threads_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input_tensor = ctx->input(0);
        OP_REQUIRES(ctx, input_tensor.dims() == 1,
                    errors::InvalidArgument("Input must be a 1D string tensor"));

        int batch_size = input_tensor.dim_size(0);
        auto input_flat = input_tensor.flat<tstring>();

        // Get trie resource (optional)
        SuperwordTrieResource* trie_resource = nullptr;
        if (ctx->input(1).NumElements() > 0) {
            LookupResource(ctx, HandleFromInput(ctx, 1), &trie_resource);
        }
        core::ScopedUnref unref_trie(trie_resource);

        // Allocate outputs
        Tensor* output_tokens = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            TensorShape({batch_size, max_length_}), &output_tokens));
        Tensor* output_lengths = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, 
            TensorShape({batch_size}), &output_lengths));

        int32_t* out_tokens = output_tokens->flat<int32_t>().data();
        int32_t* out_lengths = output_lengths->flat<int32_t>().data();

        // Prepare pointers
        std::vector<const uint8_t*> text_ptrs(batch_size);
        std::vector<int> text_lens(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            text_ptrs[i] = reinterpret_cast<const uint8_t*>(input_flat(i).data());
            text_lens[i] = static_cast<int>(input_flat(i).size());
        }

        // If we have a trie, process with merging
        if (trie_resource != nullptr && !trie_resource->trie().empty()) {
            // Process each text with trie merging
            int num_threads = num_threads_ > 0 ? num_threads_ : 
                std::max(1, static_cast<int>(std::thread::hardware_concurrency()));

            auto process_range = [&](int start, int end) {
                std::vector<int32_t> scratch;
                for (int i = start; i < end; ++i) {
                    // resize scratch if needed to hold all tokens from this text
                    // text_tokenize_utf8_simd produces at most text_lens[i] + 2 tokens
                    size_t needed = static_cast<size_t>(text_lens[i]) + 2;
                    if (scratch.size() < needed) {
                        scratch.resize(needed);
                    }

                    int32_t* out_ptr = out_tokens + i * max_length_;
                    int final_len = fused_tokenize_and_merge(
                        text_ptrs[i],
                        text_lens[i],
                        trie_resource->trie(),
                        out_ptr,
                        max_length_,
                        byte_offset_,
                        add_special_tokens_,
                        scratch.data(),
                        inject_thinking_
                    );

                    if (final_len > max_length_) {
                        final_len = max_length_;
                        if (add_special_tokens_) {
                            out_ptr[max_length_ - 1] = TEXT_TOK_EOS_ID;
                        }
                    }
                    out_lengths[i] = final_len;

                    for (int j = final_len; j < max_length_; ++j) {
                        out_ptr[j] = TEXT_TOK_PAD_ID;
                    }
                }
            };

            if (batch_size <= num_threads * 2) {
                process_range(0, batch_size);
            } else {
                std::vector<std::thread> threads;
                int chunk = (batch_size + num_threads - 1) / num_threads;
                for (int t = 0; t < num_threads; ++t) {
                    int start = t * chunk;
                    int end = std::min(start + chunk, batch_size);
                    if (start < end) {
                        threads.emplace_back(process_range, start, end);
                    }
                }
                for (auto& th : threads) th.join();
            }
        } else {
            // No trie - use optimized batch function
            std::vector<int> int_lengths(batch_size);
            text_tokenize_batch_parallel(
                text_ptrs.data(),
                text_lens.data(),
                batch_size,
                byte_offset_,
                add_special_tokens_,
                max_length_,
                out_tokens,
                int_lengths.data(),
                num_threads_,
                inject_thinking_
            );
            for (int i = 0; i < batch_size; ++i) {
                out_lengths[i] = int_lengths[i];
            }
        }
    }

private:
    int byte_offset_;
    bool add_special_tokens_;
    bool inject_thinking_;
    int max_length_;
    int num_threads_;
};

REGISTER_KERNEL_BUILDER(Name("SAGUAROTextTokenizeBatch").Device(DEVICE_CPU), 
                        FusedTextTokenizeBatchOp);

// =============================================================================
// OP REGISTRATION: SuperwordTrieCreate
// =============================================================================

REGISTER_OP("SAGUAROTrieCreate")
    .Output("handle: resource")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc("Create an empty SuperwordTrie resource.");

class SuperwordTrieCreateOp : public ResourceOpKernel<SuperwordTrieResource> {
public:
    explicit SuperwordTrieCreateOp(OpKernelConstruction* ctx)
        : ResourceOpKernel<SuperwordTrieResource>(ctx) {}

private:
    Status CreateResource(SuperwordTrieResource** resource) override {
        *resource = new SuperwordTrieResource();
        return Status();
    }
};

REGISTER_KERNEL_BUILDER(Name("SAGUAROTrieCreate").Device(DEVICE_CPU), SuperwordTrieCreateOp);

// =============================================================================
// OP REGISTRATION: SuperwordTrieInsert
// =============================================================================

REGISTER_OP("SuperwordTrieInsert")
    .Input("handle: resource")
    .Input("ngram: int32")
    .Input("superword_id: int32")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Insert an n-gram to superword_id mapping into the trie.

handle: Handle to SuperwordTrieResource.
ngram: 1D int32 tensor of token IDs forming the n-gram.
superword_id: Scalar int32 superword ID to map to.
)doc");

class SuperwordTrieInsertOp : public OpKernel {
public:
    explicit SuperwordTrieInsertOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        SuperwordTrieResource* resource = nullptr;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
        core::ScopedUnref unref(resource);

        const Tensor& ngram_tensor = ctx->input(1);
        OP_REQUIRES(ctx, ngram_tensor.dims() == 1,
                    errors::InvalidArgument("ngram must be a 1D tensor"));

        const Tensor& id_tensor = ctx->input(2);
        OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(id_tensor.shape()),
                    errors::InvalidArgument("superword_id must be a scalar"));

        auto ngram = ngram_tensor.flat<int32_t>();
        int ngram_len = ngram_tensor.dim_size(0);
        int32_t superword_id = id_tensor.scalar<int32_t>()();

        resource->trie().insert(ngram.data(), ngram_len, superword_id);
    }
};

REGISTER_KERNEL_BUILDER(Name("SuperwordTrieInsert").Device(DEVICE_CPU),
                        SuperwordTrieInsertOp);

// =============================================================================
// OP REGISTRATION: SuperwordTrieBuildFromTable
// =============================================================================

REGISTER_OP("SuperwordTrieBuildFromTable")
    .Input("handle: resource")
    .Input("ngram_offsets: int32")
    .Input("ngram_tokens: int32")
    .Input("superword_ids: int32")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Build trie from a table of n-grams and superword IDs.

handle: Handle to SuperwordTrieResource (will be cleared and rebuilt).
ngram_offsets: 1D int32 tensor [num_ngrams + 1] with start offsets into ngram_tokens.
ngram_tokens: 1D int32 tensor concatenation of all n-gram token sequences.
superword_ids: 1D int32 tensor [num_ngrams] of corresponding superword IDs.
)doc");

class SuperwordTrieBuildFromTableOp : public OpKernel {
public:
    explicit SuperwordTrieBuildFromTableOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        SuperwordTrieResource* resource = nullptr;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
        core::ScopedUnref unref(resource);

        const Tensor& offsets_tensor = ctx->input(1);
        const Tensor& tokens_tensor = ctx->input(2);
        const Tensor& ids_tensor = ctx->input(3);

        auto offsets = offsets_tensor.flat<int32_t>();
        auto tokens = tokens_tensor.flat<int32_t>();
        auto ids = ids_tensor.flat<int32_t>();

        int num_ngrams = offsets_tensor.dim_size(0) - 1;

        // Clear and rebuild trie
        resource->trie().clear();

        for (int i = 0; i < num_ngrams; ++i) {
            int start = offsets(i);
            int end = offsets(i + 1);
            int len = end - start;
            resource->trie().insert(tokens.data() + start, len, ids(i));
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("SuperwordTrieBuildFromTable").Device(DEVICE_CPU),
                        SuperwordTrieBuildFromTableOp);


// =============================================================================
// STREAMING N-GRAM COUNT RESOURCE
// =============================================================================

class StreamingNgramCountResource : public ResourceBase {
public:
    StreamingNgramCountResource(int min_ngram, int max_ngram)
        : counter_(std::make_unique<StreamingNgramCounter>(min_ngram, max_ngram)) {}

    ~StreamingNgramCountResource() override {
        // Explicitly clear
        if (counter_) {
            counter_->clear();
            counter_.reset();
        }
    }

    StreamingNgramCounter& counter() { return *counter_; }

    string DebugString() const override {
        if (counter_) {
            return strings::StrCat("StreamingNgramCountResource with ", counter_->unique_count(), " unique n-grams");
        }
        return "StreamingNgramCountResource (uninitialized)";
    }

private:
    std::unique_ptr<StreamingNgramCounter> counter_;
};

// =============================================================================
// OP REGISTRATION: StreamingNgramCountCreate
// =============================================================================

REGISTER_OP("StreamingNgramCountCreate")
    .Output("handle: resource")
    .Attr("min_ngram: int = 2")
    .Attr("max_ngram: int = 5")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .SetShapeFn(shape_inference::ScalarShape)
    .Doc("Create a StreamingNgramCount resource.");

class StreamingNgramCountCreateOp : public ResourceOpKernel<StreamingNgramCountResource> {
public:
    explicit StreamingNgramCountCreateOp(OpKernelConstruction* ctx)
        : ResourceOpKernel<StreamingNgramCountResource>(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("min_ngram", &min_ngram_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_ngram", &max_ngram_));
    }

private:
    Status CreateResource(StreamingNgramCountResource** resource) override {
        *resource = new StreamingNgramCountResource(min_ngram_, max_ngram_);
        return Status();
    }
    
    int min_ngram_;
    int max_ngram_;
};

REGISTER_KERNEL_BUILDER(Name("StreamingNgramCountCreate").Device(DEVICE_CPU), StreamingNgramCountCreateOp);

// =============================================================================
// OP REGISTRATION: StreamingNgramCount
// =============================================================================

REGISTER_OP("StreamingNgramCount")
    .Input("handle: resource")
    .Input("tokens: int32")
    .Input("lengths: int32")
    .SetShapeFn(shape_inference::NoOutputs)
    .Doc(R"doc(
Count n-grams in a batch of token sequences.

handle: Handle to StreamingNgramCountResource.
tokens: 2D int32 tensor [batch_size, max_length] of token IDs.
lengths: 1D int32 tensor [batch_size] of valid lengths.
)doc");

class StreamingNgramCountOp : public OpKernel {
public:
    explicit StreamingNgramCountOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        StreamingNgramCountResource* resource = nullptr;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
        core::ScopedUnref unref(resource);

        const Tensor& tokens = ctx->input(1);
        const Tensor& lengths = ctx->input(2);

        OP_REQUIRES(ctx, tokens.dims() == 2, errors::InvalidArgument("tokens must be 2D"));
        OP_REQUIRES(ctx, lengths.dims() == 1, errors::InvalidArgument("lengths must be 1D"));
        OP_REQUIRES(ctx, tokens.dim_size(0) == lengths.dim_size(0), 
                    errors::InvalidArgument("Batch dimension mismatch"));

        int batch_size = tokens.dim_size(0);
        int max_len = tokens.dim_size(1);
        auto tokens_flat = tokens.flat<int32_t>();
        auto lengths_flat = lengths.flat<int32_t>();
        
        auto process_range = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                int len = lengths_flat(i);
                if (len > 0) {
                    const int32_t* seq = tokens_flat.data() + (i * max_len);
                    resource->counter().count_sequence(seq, len);
                }
            }
        };

        // Determine thread count
        int num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        if (batch_size < 100) num_threads = 1; // Avoid overhead for small batches

        if (num_threads == 1) {
            process_range(0, batch_size);
        } else {
            std::vector<std::thread> threads;
            int chunk = (batch_size + num_threads - 1) / num_threads;
            for (int t = 0; t < num_threads; ++t) {
                int start = t * chunk;
                int end = std::min(start + chunk, batch_size);
                if (start < end) {
                    threads.emplace_back(process_range, start, end);
                }
            }
            for (auto& th : threads) th.join();
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("StreamingNgramCount").Device(DEVICE_CPU), StreamingNgramCountOp);

// =============================================================================
// OP REGISTRATION: StreamingNgramCountExport
// =============================================================================

REGISTER_OP("StreamingNgramCountExport")
    .Input("handle: resource")
    .Output("ngrams: int32")
    .Output("counts: int32")
    .Output("splits: int32")
    .Attr("min_frequency: int = 5")
    .Attr("max_count: int = 10000")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Vector(shape_inference::InferenceContext::kUnknownDim));
        c->set_output(1, c->Vector(shape_inference::InferenceContext::kUnknownDim));
        c->set_output(2, c->Vector(shape_inference::InferenceContext::kUnknownDim));
        return Status();
    })
    .Doc(R"doc(
Export top frequent n-grams from the counter.

handle: Handle to StreamingNgramCountResource.
min_frequency: Minimum occurrence count.
max_count: Maximum number of n-grams to return.

ngrams: Flat tensor of all n-gram tokens concatenated.
counts: Frequency count for each n-gram.
splits: Row splits for the ngrams tensor (RaggedTensor format).
)doc");

class StreamingNgramCountExportOp : public OpKernel {
public:
    explicit StreamingNgramCountExportOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("min_frequency", &min_freq_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_count", &max_count_));
    }

    void Compute(OpKernelContext* ctx) override {
        StreamingNgramCountResource* resource = nullptr;
        OP_REQUIRES_OK(ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &resource));
        core::ScopedUnref unref(resource);

        std::vector<std::pair<std::vector<int32_t>, int>> top_ngrams;
        resource->counter().get_top_ngrams(min_freq_, max_count_, top_ngrams);

        int num_ngrams = top_ngrams.size();
        int total_tokens = 0;
        for (const auto& pair : top_ngrams) {
            total_tokens += pair.first.size();
        }

        // Allocate outputs
        Tensor* ngrams_out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({total_tokens}), &ngrams_out));
        
        Tensor* counts_out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({num_ngrams}), &counts_out));

        Tensor* splits_out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({num_ngrams + 1}), &splits_out));

        auto ngrams_flat = ngrams_out->flat<int32_t>();
        auto counts_flat = counts_out->flat<int32_t>();
        auto splits_flat = splits_out->flat<int32_t>();

        int token_idx = 0;
        splits_flat(0) = 0;

        for (int i = 0; i < num_ngrams; ++i) {
            const auto& ngram = top_ngrams[i].first;
            int count = top_ngrams[i].second;

            counts_flat(i) = count;
            
            for (int32_t token : ngram) {
                ngrams_flat(token_idx++) = token;
            }
            splits_flat(i + 1) = token_idx;
        }
    }

private:
    int min_freq_;
    int max_count_;
};

REGISTER_KERNEL_BUILDER(Name("StreamingNgramCountExport").Device(DEVICE_CPU), StreamingNgramCountExportOp);


}  // namespace tensorflow
