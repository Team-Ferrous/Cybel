// saguaro.native/ops/quantum_curriculum_op.cc
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
 * @file quantum_curriculum_op.cc
 * @brief Phase 200+: SAQC TensorFlow custom operations.
 *
 * SAGUARO_UPGRADE_ROADMAP.md complementary implementation.
 * Synergizes with QULS telemetry and QULSFeedbackCallback for curriculum control.
 *
 * Registers TensorFlow ops for spectral complexity analysis used by
 * QuantumSynergyCurriculum to drive data progression based on model
 * quantum state (entropy, fidelity, coherence).
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "quantum_curriculum_op.h"

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace tensorflow;

namespace saguaro {
namespace saqc {

// =============================================================================
// Implementation: FFT Helper Functions
// =============================================================================

void ComputePowerSpectrum(
    const float* input,
    float* power_spectrum,
    int dim
) {
    // Simple DFT-based power spectrum for small dimensions
    // For production with large dims, use FFTW or Eigen FFT
    const int spectrum_size = dim / 2 + 1;
    
    for (int k = 0; k < spectrum_size; ++k) {
        float real_sum = 0.0f;
        float imag_sum = 0.0f;
        const float freq = 2.0f * M_PI * k / dim;
        
        for (int n = 0; n < dim; ++n) {
            real_sum += input[n] * std::cos(freq * n);
            imag_sum -= input[n] * std::sin(freq * n);
        }
        
        // Power = |X_k|^2 = real^2 + imag^2
        power_spectrum[k] = real_sum * real_sum + imag_sum * imag_sum;
    }
}

float ComputeNormalizedEntropy(
    const float* power_spectrum,
    int spectrum_size
) {
    // Compute normalized spectral entropy: H = -Σ p_i log(p_i) / log(N)
    const float epsilon = 1e-10f;
    
    // Sum for normalization
    float total_power = 0.0f;
    for (int i = 0; i < spectrum_size; ++i) {
        total_power += power_spectrum[i];
    }
    
    if (total_power < epsilon) {
        return 0.0f;  // Degenerate case
    }
    
    // Compute entropy
    float entropy = 0.0f;
    for (int i = 0; i < spectrum_size; ++i) {
        float p = power_spectrum[i] / total_power;
        if (p > epsilon) {
            entropy -= p * std::log(p);
        }
    }
    
    // Normalize by log(N) to get [0, 1] range
    float max_entropy = std::log(static_cast<float>(spectrum_size));
    return (max_entropy > epsilon) ? entropy / max_entropy : 0.0f;
}

// =============================================================================
// Implementation: Core Kernels
// =============================================================================

void ComputeSpectralComplexity(
    const float* representations,
    float* complexity_scores,
    float* spectral_entropy,
    const SAQCConfig& config,
    int batch_size
) {
    const int spectrum_size = config.fft_dim / 2 + 1;
    std::vector<float> power_spectrum(spectrum_size);
    std::vector<float> projected(config.fft_dim);
    
    for (int b = 0; b < batch_size; ++b) {
        const float* repr = representations + b * config.hidden_dim;
        
        // Project to FFT dimension via strided sampling + interpolation
        const float stride = static_cast<float>(config.hidden_dim) / config.fft_dim;
        for (int i = 0; i < config.fft_dim; ++i) {
            int src_idx = static_cast<int>(i * stride);
            src_idx = std::min(src_idx, config.hidden_dim - 1);
            projected[i] = repr[src_idx];
        }
        
        // Compute power spectrum
        ComputePowerSpectrum(projected.data(), power_spectrum.data(), config.fft_dim);
        
        // Compute normalized entropy
        float entropy = ComputeNormalizedEntropy(power_spectrum.data(), spectrum_size);
        spectral_entropy[b] = entropy;
        
        // Complexity score: combine entropy with power distribution statistics
        // Higher entropy = more complex (more frequency content)
        // Also consider spectral centroid for additional complexity signal
        float centroid = 0.0f;
        float total_power = 0.0f;
        for (int k = 0; k < spectrum_size; ++k) {
            centroid += k * power_spectrum[k];
            total_power += power_spectrum[k];
        }
        if (total_power > 1e-10f) {
            centroid /= total_power;
            centroid /= spectrum_size;  // Normalize to [0, 1]
        }
        
        // Complexity = 0.7 * entropy + 0.3 * centroid
        complexity_scores[b] = 0.7f * entropy + 0.3f * centroid;
    }
}

void ComputeSpectralComplexityGrad(
    const float* grad_complexity,
    const float* representations,
    float* grad_representations,
    const SAQCConfig& config,
    int batch_size
) {
    // Gradient computation via finite differences approximation
    // For true analytical gradients, would need complex chain rule through FFT
    const float epsilon = 1e-4f;
    const float inv_2eps = 1.0f / (2.0f * epsilon);
    
    std::vector<float> repr_plus(config.hidden_dim);
    std::vector<float> repr_minus(config.hidden_dim);
    std::vector<float> complexity_plus(1);
    std::vector<float> complexity_minus(1);
    std::vector<float> entropy_dummy(1);
    
    for (int b = 0; b < batch_size; ++b) {
        const float* repr = representations + b * config.hidden_dim;
        float* grad = grad_representations + b * config.hidden_dim;
        float upstream_grad = grad_complexity[b];
        
        for (int d = 0; d < config.hidden_dim; ++d) {
            // Copy and perturb
            std::copy(repr, repr + config.hidden_dim, repr_plus.data());
            std::copy(repr, repr + config.hidden_dim, repr_minus.data());
            repr_plus[d] += epsilon;
            repr_minus[d] -= epsilon;
            
            // Forward pass for perturbed inputs
            ComputeSpectralComplexity(repr_plus.data(), complexity_plus.data(),
                                      entropy_dummy.data(), config, 1);
            ComputeSpectralComplexity(repr_minus.data(), complexity_minus.data(),
                                      entropy_dummy.data(), config, 1);
            
            // Central difference gradient
            float local_grad = (complexity_plus[0] - complexity_minus[0]) * inv_2eps;
            grad[d] = upstream_grad * local_grad;
        }
    }
}

CurriculumMode DetermineCurriculumMode(
    float spectral_entropy,
    float fidelity_loss,
    float coherence,
    bool barren_plateau_detected,
    const SAQCConfig& config
) {
    // Priority: TUNNELING > RETREAT > ACCELERATE > NORMAL
    
    if (barren_plateau_detected) {
        return CurriculumMode::TUNNELING;
    }
    
    if (spectral_entropy < config.entropy_threshold) {
        return CurriculumMode::RETREAT;
    }
    
    // High fidelity = low fidelity_loss (loss is 1 - fidelity)
    float fidelity = 1.0f - fidelity_loss;
    if (fidelity > config.fidelity_threshold && coherence > config.coherence_threshold) {
        return CurriculumMode::ACCELERATE;
    }
    
    return CurriculumMode::NORMAL;
}

}  // namespace saqc
}  // namespace saguaro

// =============================================================================
// OP REGISTRATION: QuantumCurriculumScoreForward
// =============================================================================

REGISTER_OP("QuantumCurriculumScoreForward")
    .Input("representations: float")      // [batch, hidden_dim]
    .Output("complexity_scores: float")   // [batch]
    .Output("spectral_entropy: float")    // [batch]
    .Attr("fft_dim: int = 64")
    .Attr("hidden_dim: int = 512")
    .Attr("entropy_threshold: float = 0.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape = c->input(0);
        
        if (c->RankKnown(input_shape) && c->Rank(input_shape) == 2) {
            auto batch = c->Dim(input_shape, 0);
            c->set_output(0, c->MakeShape({batch}));  // complexity_scores
            c->set_output(1, c->MakeShape({batch}));  // spectral_entropy
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
SAQC Spectral Complexity Forward - FFT-domain curriculum scoring.

Phase 200+: Synergizes with QULS telemetry for quantum-adaptive curriculum.
Analyzes representation spectral properties to determine data complexity.

representations: Hidden state representations [batch, hidden_dim]
complexity_scores: Spectral complexity scores in [0, 1] [batch]
spectral_entropy: Normalized spectral entropy [batch]
)doc");

// =============================================================================
// KERNEL: QuantumCurriculumScoreForward
// =============================================================================

class QuantumCurriculumScoreForwardOp : public OpKernel {
 public:
  explicit QuantumCurriculumScoreForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fft_dim", &config_.fft_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(context, context->GetAttr("entropy_threshold", &config_.entropy_threshold));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& representations = context->input(0);
    
    OP_REQUIRES(context, representations.dims() == 2,
                errors::InvalidArgument("representations must be 2D [batch, hidden_dim]"));
    
    const int batch_size = representations.dim_size(0);
    const int hidden_dim = representations.dim_size(1);
    
    OP_REQUIRES(context, hidden_dim == config_.hidden_dim,
                errors::InvalidArgument("hidden_dim mismatch: ", hidden_dim,
                                        " vs config ", config_.hidden_dim));
    
    // Allocate outputs
    Tensor* complexity_scores = nullptr;
    Tensor* spectral_entropy = nullptr;
    
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size}), &complexity_scores));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size}), &spectral_entropy));
    
    // Compute spectral complexity
    saguaro::saqc::ComputeSpectralComplexity(
        representations.flat<float>().data(),
        complexity_scores->flat<float>().data(),
        spectral_entropy->flat<float>().data(),
        config_,
        batch_size
    );
  }

 private:
  saguaro::saqc::SAQCConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QuantumCurriculumScoreForward").Device(DEVICE_CPU),
                        QuantumCurriculumScoreForwardOp);

// =============================================================================
// OP REGISTRATION: QuantumCurriculumScoreBackward
// =============================================================================

REGISTER_OP("QuantumCurriculumScoreBackward")
    .Input("grad_complexity: float")      // [batch]
    .Input("representations: float")      // [batch, hidden_dim]
    .Output("grad_representations: float") // [batch, hidden_dim]
    .Attr("fft_dim: int = 64")
    .Attr("hidden_dim: int = 512")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // Same shape as representations
        return Status();
    })
    .Doc("SAQC Spectral Complexity Backward - Gradient computation.");

// =============================================================================
// KERNEL: QuantumCurriculumScoreBackward
// =============================================================================

class QuantumCurriculumScoreBackwardOp : public OpKernel {
 public:
  explicit QuantumCurriculumScoreBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fft_dim", &config_.fft_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_complexity = context->input(0);
    const Tensor& representations = context->input(1);
    
    const int batch_size = representations.dim_size(0);
    
    // Allocate output
    Tensor* grad_representations = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, representations.shape(), &grad_representations));
    
    // Compute gradient
    saguaro::saqc::ComputeSpectralComplexityGrad(
        grad_complexity.flat<float>().data(),
        representations.flat<float>().data(),
        grad_representations->flat<float>().data(),
        config_,
        batch_size
    );
  }

 private:
  saguaro::saqc::SAQCConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QuantumCurriculumScoreBackward").Device(DEVICE_CPU),
                        QuantumCurriculumScoreBackwardOp);

// =============================================================================
// OP REGISTRATION: DetermineCurriculumMode
// =============================================================================

REGISTER_OP("DetermineCurriculumMode")
    .Input("spectral_entropy: float")      // scalar
    .Input("fidelity_loss: float")         // scalar
    .Input("coherence: float")             // scalar
    .Input("barren_plateau: bool")         // scalar
    .Output("mode: int32")                 // scalar: 0=NORMAL, 1=RETREAT, 2=TUNNELING, 3=ACCELERATE
    .Attr("entropy_threshold: float = 0.5")
    .Attr("fidelity_threshold: float = 0.85")
    .Attr("coherence_threshold: float = 0.9")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
SAQC Curriculum Mode Determination.

Implements SAQC decision logic based on QULS telemetry:
  - barren_plateau → TUNNELING (2)
  - low entropy → RETREAT (1)
  - high fidelity + coherence → ACCELERATE (3)
  - otherwise → NORMAL (0)
)doc");

// =============================================================================
// KERNEL: DetermineCurriculumMode
// =============================================================================

class DetermineCurriculumModeOp : public OpKernel {
 public:
  explicit DetermineCurriculumModeOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("entropy_threshold", &config_.entropy_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("fidelity_threshold", &config_.fidelity_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("coherence_threshold", &config_.coherence_threshold));
  }

  void Compute(OpKernelContext* context) override {
    float spectral_entropy = context->input(0).scalar<float>()();
    float fidelity_loss = context->input(1).scalar<float>()();
    float coherence = context->input(2).scalar<float>()();
    bool barren_plateau = context->input(3).scalar<bool>()();
    
    // Allocate output
    Tensor* mode_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &mode_tensor));
    
    // Determine mode
    saguaro::saqc::CurriculumMode mode = saguaro::saqc::DetermineCurriculumMode(
        spectral_entropy, fidelity_loss, coherence, barren_plateau, config_
    );
    
    mode_tensor->scalar<int32>()() = static_cast<int32>(mode);
  }

 private:
  saguaro::saqc::SAQCConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("DetermineCurriculumMode").Device(DEVICE_CPU),
                        DetermineCurriculumModeOp);
