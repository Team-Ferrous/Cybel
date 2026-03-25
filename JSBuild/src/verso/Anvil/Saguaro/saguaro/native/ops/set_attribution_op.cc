// saguaro.native/ops/set_attribution_op.cc
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
// SET ATTRIBUTION OP - Custom Attribution for Pro/Enterprise Editions
// =============================================================================
//
// TensorFlow Custom Ops for managing custom attribution:
//   - HighNoonSetCustomAttribution: Set custom attribution values
//   - HighNoonClearCustomAttribution: Reset to default values
//   - HighNoonGetEditionCode: Get current edition (0=Lite, 1=Pro, 2=Enterprise)
//   - HighNoonIsCustomAttributionAllowed: Check if customization is enabled
//   - HighNoonGetCurrentAttribution: Get current attribution values as metadata
//
// For Lite edition, SetCustomAttribution is a no-op returning false.
// For Pro/Enterprise editions, it updates the global custom attribution store.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "common/attribution.h"
#include "common/edition_limits.h"

using namespace tensorflow;

namespace saguaro {
namespace ops {

// =============================================================================
// OP REGISTRATION
// =============================================================================

// SetCustomAttribution - Set custom attribution values (Pro/Enterprise only)
REGISTER_OP("HighNoonSetCustomAttribution")
    .Input("framework_name: string")
    .Input("author: string")
    .Input("copyright_notice: string")
    .Input("version: string")
    .Input("support_url: string")
    .Output("success: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Set custom attribution values for trained models.

Only functional in Pro and Enterprise editions. Lite edition returns false.

framework_name: The custom framework name to display
author: The author/company name
copyright_notice: The copyright notice string
version: The version string
support_url: The support URL
success: True if attribution was set, false if not allowed (Lite edition)
)doc");

// ClearCustomAttribution - Reset to default Verso Industries attribution
REGISTER_OP("HighNoonClearCustomAttribution")
    .Output("success: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Clear custom attribution and revert to default values.

Returns true on success. For Lite edition, this is a no-op returning true.
)doc");

// GetEditionCode - Get current edition as integer
REGISTER_OP("HighNoonGetEditionCode")
    .Output("edition_code: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Get the current edition code.

Output edition_code:
  0 = Lite Edition (scale limits enforced)
  1 = Pro Edition (unlimited scale)
  2 = Enterprise Edition (unlimited + source access)
)doc");

// IsCustomAttributionAllowed - Check if custom attribution is enabled
REGISTER_OP("HighNoonIsCustomAttributionAllowed")
    .Output("allowed: bool")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Check if custom attribution is allowed for this edition.

Returns true for Pro and Enterprise editions, false for Lite.
)doc");

// GetCurrentAttribution - Get current attribution values as metadata
REGISTER_OP("HighNoonGetCurrentAttribution")
    .Output("keys: string")
    .Output("values: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Vector(6));
        c->set_output(1, c->Vector(6));
        return Status();
    })
    .Doc(R"doc(
Get current attribution values as key-value pairs.

Returns the effective attribution values (custom if set for Pro+, else defaults).
Output keys: framework_name, author, copyright, version, support_url, is_custom
)doc");

// =============================================================================
// OP IMPLEMENTATIONS
// =============================================================================

class SetCustomAttributionOp : public OpKernel {
public:
    explicit SetCustomAttributionOp(OpKernelConstruction* context)
        : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        // Get inputs
        const Tensor& framework_name_tensor = context->input(0);
        const Tensor& author_tensor = context->input(1);
        const Tensor& copyright_tensor = context->input(2);
        const Tensor& version_tensor = context->input(3);
        const Tensor& support_url_tensor = context->input(4);
        
        std::string framework_name = framework_name_tensor.scalar<tstring>()();
        std::string author = author_tensor.scalar<tstring>()();
        std::string copyright_notice = copyright_tensor.scalar<tstring>()();
        std::string version = version_tensor.scalar<tstring>()();
        std::string support_url = support_url_tensor.scalar<tstring>()();
        
        // Call the attribution function (no-op for Lite, functional for Pro+)
        bool success = attribution::SetCustomAttribution(
            framework_name, author, copyright_notice, version, support_url
        );
        
        // Create output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<bool>()() = success;
    }
};

class ClearCustomAttributionOp : public OpKernel {
public:
    explicit ClearCustomAttributionOp(OpKernelConstruction* context)
        : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        // Clear custom attribution (no-op for Lite)
        attribution::ClearCustomAttribution();
        
        // Create output tensor
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<bool>()() = true;
    }
};

class GetEditionCodeOp : public OpKernel {
public:
    explicit GetEditionCodeOp(OpKernelConstruction* context)
        : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<int32>()() = limits::GetEditionCode();
    }
};

class IsCustomAttributionAllowedOp : public OpKernel {
public:
    explicit IsCustomAttributionAllowedOp(OpKernelConstruction* context)
        : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<bool>()() = attribution::IsCustomAttributionAllowed();
    }
};

class GetCurrentAttributionOp : public OpKernel {
public:
    explicit GetCurrentAttributionOp(OpKernelConstruction* context)
        : OpKernel(context) {}
    
    void Compute(OpKernelContext* context) override {
        // Create output tensors for keys and values
        Tensor* keys_tensor = nullptr;
        Tensor* values_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({6}), &keys_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({6}), &values_tensor));
        
        auto keys = keys_tensor->flat<tstring>();
        auto values = values_tensor->flat<tstring>();
        
        // Populate keys
        keys(0) = "framework_name";
        keys(1) = "author";
        keys(2) = "copyright";
        keys(3) = "version";
        keys(4) = "support_url";
        keys(5) = "is_custom";
        
        // Populate values with effective attribution
        values(0) = attribution::GetEffectiveFrameworkName();
        values(1) = attribution::GetEffectiveAuthor();
        values(2) = attribution::GetEffectiveCopyright();
        values(3) = attribution::GetEffectiveVersion();
        values(4) = attribution::GetEffectiveSupportUrl();
        values(5) = attribution::IsCustomAttributionActive() ? "true" : "false";
    }
};

// =============================================================================
// KERNEL REGISTRATION
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HighNoonSetCustomAttribution").Device(DEVICE_CPU),
    SetCustomAttributionOp);

REGISTER_KERNEL_BUILDER(
    Name("HighNoonClearCustomAttribution").Device(DEVICE_CPU),
    ClearCustomAttributionOp);

REGISTER_KERNEL_BUILDER(
    Name("HighNoonGetEditionCode").Device(DEVICE_CPU),
    GetEditionCodeOp);

REGISTER_KERNEL_BUILDER(
    Name("HighNoonIsCustomAttributionAllowed").Device(DEVICE_CPU),
    IsCustomAttributionAllowedOp);

REGISTER_KERNEL_BUILDER(
    Name("HighNoonGetCurrentAttribution").Device(DEVICE_CPU),
    GetCurrentAttributionOp);

}  // namespace ops
}  // namespace saguaro
