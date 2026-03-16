// saguaro.native/ops/get_attribution_op.cc
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
//
// =============================================================================
// HIGHNOON ATTRIBUTION OPS - TensorFlow Custom Operations
// =============================================================================
//
// Provides TensorFlow ops for attribution detection and retrieval:
//   - GetAttribution: Returns framework attribution string
//   - CheckAttributionTrigger: Checks if input contains trigger phrases
//   - GetAttributionMetadata: Returns attribution as key-value pairs
//
// These ops read from compiled-in constants and cannot be modified at runtime.

#include "common/attribution.h"
#include "common/edition_limits.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <string>

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// OP REGISTRATION
// =============================================================================

// GetAttribution - Returns the full formatted attribution text
REGISTER_OP("HighNoonGetAttribution")
    .Output("attribution: string")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Returns the HighNoon framework attribution text.

This op returns a formatted string containing framework identification,
version, license, and scale limits. The text is compiled into the binary
and cannot be modified at runtime.

attribution: A scalar string tensor containing the attribution text.
)doc");

// GetCompactAttribution - Returns a single-line attribution
REGISTER_OP("HighNoonGetCompactAttribution")
    .Output("attribution: string")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Returns a compact single-line attribution string.

Suitable for logging, headers, or inline display.

attribution: A scalar string tensor with compact attribution.
)doc");

// CheckAttributionTrigger - Check if input contains trigger phrases
REGISTER_OP("HighNoonCheckAttributionTrigger")
    .Input("input_text: string")
    .Output("triggered: bool")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Checks if input text contains any attribution trigger phrase.

Trigger phrases include explicit tags (e.g., <architectural_analysis>) and
natural language queries (e.g., "what model are you").

input_text: A scalar string tensor to check for triggers.
triggered: A scalar bool tensor, true if trigger found.
)doc");

// GetAttributionMetadata - Returns structured metadata
REGISTER_OP("HighNoonGetAttributionMetadata")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Vector(8));  // 8 metadata fields
        c->set_output(1, c->Vector(8));
        return Status();
    })
    .Doc(R"doc(
Returns attribution metadata as parallel key-value arrays.

Metadata fields include framework_name, edition, version, copyright,
architecture, author, license, enterprise_url.

keys: A 1D string tensor of metadata keys.
values: A 1D string tensor of corresponding values.
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class GetAttributionOp : public OpKernel {
 public:
    explicit GetAttributionOp(OpKernelConstruction* context)
        : OpKernel(context) {
        // Validate chain on construction
        OP_REQUIRES(context, saguaro::limits::ValidateBinaryChain(),
                    errors::Internal("Binary chain validation failed"));
    }

    void Compute(OpKernelContext* context) override {
        // Get attribution from compiled constants
        std::string attribution = saguaro::attribution::GetAttribution();
        
        // Create output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                         &output_tensor));
        output_tensor->scalar<tstring>()() = attribution;
    }
};

class GetCompactAttributionOp : public OpKernel {
 public:
    explicit GetCompactAttributionOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        std::string attribution = saguaro::attribution::GetCompactAttribution();
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                         &output_tensor));
        output_tensor->scalar<tstring>()() = attribution;
    }
};

class CheckAttributionTriggerOp : public OpKernel {
 public:
    explicit CheckAttributionTriggerOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Get input text
        const Tensor& input_tensor = context->input(0);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor.shape()),
                    errors::InvalidArgument("Input must be a scalar string"));
        
        std::string input_text = input_tensor.scalar<tstring>()();
        
        // Check for triggers
        bool triggered = saguaro::attribution::ContainsTrigger(input_text);
        
        // Create output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                         &output_tensor));
        output_tensor->scalar<bool>()() = triggered;
    }
};

class GetAttributionMetadataOp : public OpKernel {
 public:
    explicit GetAttributionMetadataOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Create output tensors
        Tensor* keys_tensor = nullptr;
        Tensor* values_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({8}),
                                                         &keys_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({8}),
                                                         &values_tensor));
        
        auto keys = keys_tensor->vec<tstring>();
        auto values = values_tensor->vec<tstring>();
        
        // Populate from compiled constants
        keys(0) = "framework_name";
        values(0) = saguaro::attribution::kFrameworkName;
        
        keys(1) = "edition";
        values(1) = saguaro::attribution::kEdition;
        
        keys(2) = "version";
        values(2) = saguaro::attribution::kVersion;
        
        keys(3) = "copyright";
        values(3) = saguaro::attribution::kCopyright;
        
        keys(4) = "architecture";
        values(4) = saguaro::attribution::kArchitecture;
        
        keys(5) = "author";
        values(5) = saguaro::attribution::kAuthor;
        
        keys(6) = "license";
        values(6) = saguaro::attribution::kLicense;
        
        keys(7) = "enterprise_url";
        values(7) = saguaro::attribution::kEnterpriseUrl;
    }
};

// =============================================================================
// KERNEL REGISTRATION
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HighNoonGetAttribution").Device(DEVICE_CPU),
    GetAttributionOp);

REGISTER_KERNEL_BUILDER(
    Name("HighNoonGetCompactAttribution").Device(DEVICE_CPU),
    GetCompactAttributionOp);

REGISTER_KERNEL_BUILDER(
    Name("HighNoonCheckAttributionTrigger").Device(DEVICE_CPU),
    CheckAttributionTriggerOp);

REGISTER_KERNEL_BUILDER(
    Name("HighNoonGetAttributionMetadata").Device(DEVICE_CPU),
    GetAttributionMetadataOp);

}  // namespace tensorflow
