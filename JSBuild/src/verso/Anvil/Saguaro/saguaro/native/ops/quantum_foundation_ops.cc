// saguaro.native/ops/quantum_foundation_ops.cc
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file quantum_foundation_ops.cc
 * @brief Implementation of Unified Quantum Foundation Operations
 *
 * Phase 3 of V2 Performance Optimization.
 * Consolidates 15 quantum mechanisms into unified kernels.
 */

#include "quantum_foundation_ops.h"

#include <cstring>
#include <random>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// TensorFlow op registration
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

namespace saguaro {
namespace quantum {

// =============================================================================
// QUANTUM EMBEDDING KERNEL
// =============================================================================

void QuantumEmbeddingForward(
    const int32_t* token_ids,
    const float* holographic_store,
    const float* token_keys,
    float* output,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int dim = config.embedding_dim;
    int num_bundles = config.num_bundles;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int token_id = token_ids[b * seq_len + s];
            
            // Clamp token ID to valid range
            token_id = std::max(0, std::min(token_id, config.vocab_size - 1));
            
            // Get token key
            const float* key = token_keys + token_id * dim;
            
            // Unbind from holographic store
            float* out = output + (b * seq_len + s) * dim;
            
            // Accumulate across bundles
            std::fill(out, out + dim, 0.0f);
            
            for (int bund = 0; bund < num_bundles; ++bund) {
                const float* bundle = holographic_store + bund * dim;
                std::vector<float> unbound(dim);
                
                fft::holographic_unbind(bundle, key, unbound.data(), dim);
                
                for (int d = 0; d < dim; ++d) {
                    out[d] += unbound[d];
                }
            }
            
            // Normalize
            float inv_bundles = 1.0f / num_bundles;
            for (int d = 0; d < dim; ++d) {
                out[d] *= inv_bundles;
            }
        }
    }
}

// =============================================================================
// QUANTUM POSITION ENCODING KERNEL
// =============================================================================

void QuantumPositionEncodingForward(
    const float* input,
    float* output,
    int position_offset,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int dim = config.d_model;
    float omega = config.floquet_omega;
    float amplitude = config.floquet_amplitude;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int pos = s + position_offset;
            
            const float* in = input + (b * seq_len + s) * dim;
            float* out = output + (b * seq_len + s) * dim;
            
            // Apply Floquet-style position encoding
            // Uses SU(2) rotation pattern: R_y(θ)R_z(φ) where θ,φ depend on position
            for (int d = 0; d < dim; d += 2) {
                // Position-dependent angles
                float theta = omega * pos * (d + 1) / static_cast<float>(dim);
                float phi = amplitude * std::sin(theta);
                
                // Apply SU(2) rotation to pair of dimensions
                float x0 = (d < dim) ? in[d] : 0.0f;
                float x1 = (d + 1 < dim) ? in[d + 1] : 0.0f;
                
                // RY(theta)
                float cos_half = std::cos(theta * 0.5f);
                float sin_half = std::sin(theta * 0.5f);
                float y0 = cos_half * x0 - sin_half * x1;
                float y1 = sin_half * x0 + cos_half * x1;
                
                // RZ(phi) - phase shift
                float cos_phi = std::cos(phi * 0.5f);
                out[d] = cos_phi * y0;
                if (d + 1 < dim) {
                    out[d + 1] = cos_phi * y1;
                }
            }
        }
    }
}

// =============================================================================
// QUANTUM LM HEAD KERNEL
// =============================================================================

void QuantumLMHeadForward(
    const float* hidden,
    const float* vqc_params,
    const float* output_weights,
    float* logits,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int d_model = config.d_model;
    int vocab_size = config.vocab_size;
    int num_qubits = config.num_qubits;
    int vqc_layers = config.vqc_layers;
    
    int params_per_layer = 2 * num_qubits;  // RY + RZ for each qubit
    int total_params = vqc_layers * params_per_layer;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const float* h = hidden + (b * seq_len + s) * d_model;
            float* log_out = logits + (b * seq_len + s) * vocab_size;
            
            // Initialize quantum state in |+⟩ superposition
            std::vector<float> state(2 * num_qubits);
            for (int q = 0; q < num_qubits; ++q) {
                vqc::init_plus(state.data() + 2 * q);
            }
            
            // Create data-dependent VQC parameters
            std::vector<float> layer_params(params_per_layer);
            
            for (int l = 0; l < vqc_layers; ++l) {
                // Combine hidden state with learned parameters
                for (int p = 0; p < params_per_layer; ++p) {
                    int h_idx = (l * params_per_layer + p) % d_model;
                    layer_params[p] = vqc_params[l * params_per_layer + p] + h[h_idx];
                }
                
                // Apply VQC layer
                vqc::apply_vqc_layer(state.data(), layer_params.data(), num_qubits);
            }
            
            // Extract probabilities via Born rule
            measurement::born_rule(
                state.data(),
                log_out,
                num_qubits,
                vocab_size,
                output_weights
            );
            
            // Convert to log probabilities
            for (int v = 0; v < vocab_size; ++v) {
                log_out[v] = std::log(log_out[v] + config.epsilon);
            }
        }
    }
}

// =============================================================================
// QUANTUM EXPERT KERNEL (UNITARY NETWORKS)
// =============================================================================

void QuantumExpertForward(
    const float* input,
    const float* U_skew,
    float* output,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int d_model = config.d_model;
    int d_ff = config.d_ff;
    float activation_angle = config.activation_angle;
    int neumann_terms = config.neumann_terms;
    
    // Compute unitary matrices via Cayley transform
    std::vector<float> U1(d_ff * d_model);
    std::vector<float> U2(d_model * d_ff);
    
    // For simplicity, assume U_skew contains both U1_skew and U2_skew
    // U1_skew: [d_ff, d_model], U2_skew: [d_model, d_ff]
    int u1_size = d_ff * d_model;
    int u2_size = d_model * d_ff;
    
    // Note: Full Cayley transform here is expensive. 
    // For production, use pre-computed or approximate methods.
    // Here we use the smaller dimension for square assumption.
    
    int num_tokens = batch_size * seq_len;
    
    #pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        const float* x = input + t * d_model;
        float* out = output + t * d_model;
        
        // Project to hidden: h = U1 @ x
        std::vector<float> hidden(d_ff, 0.0f);
        for (int i = 0; i < d_ff; ++i) {
            for (int j = 0; j < d_model; ++j) {
                hidden[i] += U_skew[i * d_model + j] * x[j];
            }
        }
        
        // Apply quantum activation (rotation)
        for (int d = 0; d + 1 < d_ff; d += 2) {
            float h0 = hidden[d], h1 = hidden[d + 1];
            float c = std::cos(activation_angle), s = std::sin(activation_angle);
            hidden[d] = c * h0 - s * h1;
            hidden[d + 1] = s * h0 + c * h1;
        }
        
        // Project back: out = U2 @ hidden
        std::fill(out, out + d_model, 0.0f);
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_ff; ++j) {
                out[i] += U_skew[u1_size + i * d_ff + j] * hidden[j];
            }
        }
    }
}

// =============================================================================
// QUANTUM NORM KERNEL (UNITARY/STIEFEL NORMALIZATION)
// =============================================================================

void QuantumNormForward(
    const float* input,
    const float* scale,
    const float* bias,
    float* output,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int dim = config.d_model;
    float eps = config.epsilon;
    bool use_bias = config.use_bias;
    
    int num_tokens = batch_size * seq_len;
    
    #pragma omp parallel for
    for (int t = 0; t < num_tokens; ++t) {
        const float* x = input + t * dim;
        float* out = output + t * dim;
        
        // Compute L2 norm
        float norm_sq = 0.0f;
        int d = 0;
        
#if defined(QF_AVX2)
        __m256 acc = _mm256_setzero_ps();
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&x[d]);
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int i = 0; i < 8; ++i) norm_sq += tmp[i];
#endif
        for (; d < dim; ++d) {
            norm_sq += x[d] * x[d];
        }
        
        float norm = std::sqrt(norm_sq + eps);
        float inv_norm = 1.0f / norm;
        
        // Normalize and scale
        d = 0;
#if defined(QF_AVX2)
        __m256 inv_norm_vec = _mm256_set1_ps(inv_norm);
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(&x[d]);
            __m256 s = _mm256_loadu_ps(&scale[d]);
            __m256 normed = _mm256_mul_ps(v, inv_norm_vec);
            __m256 scaled = _mm256_mul_ps(normed, s);
            if (use_bias) {
                __m256 b = _mm256_loadu_ps(&bias[d]);
                scaled = _mm256_add_ps(scaled, b);
            }
            _mm256_storeu_ps(&out[d], scaled);
        }
#endif
        for (; d < dim; ++d) {
            float normed = x[d] * inv_norm;
            out[d] = normed * scale[d];
            if (use_bias) {
                out[d] += bias[d];
            }
        }
    }
}

// =============================================================================
// QUANTUM RESIDUAL KERNEL
// =============================================================================

void QuantumResidualForward(
    const float* x,
    const float* fx,
    const float* alpha,
    float* output,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int dim = config.d_model;
    
    int num_elements = batch_size * seq_len * dim;
    
    // Unitary residual: output = x + alpha * fx
    // where alpha is learnable to maintain unitarity-like properties
    int i = 0;
    
#if defined(QF_AVX2)
    __m256 alpha_vec = _mm256_set1_ps(*alpha);
    for (; i + 8 <= num_elements; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        __m256 fxv = _mm256_loadu_ps(&fx[i]);
        __m256 result = _mm256_fmadd_ps(alpha_vec, fxv, xv);
        _mm256_storeu_ps(&output[i], result);
    }
#endif
    float a = *alpha;
    for (; i < num_elements; ++i) {
        output[i] = x[i] + a * fx[i];
    }
}

// =============================================================================
// QUANTUM COHERENCE BUS KERNEL
// =============================================================================

void QuantumCoherenceBusForward(
    const float* state,
    const float* phase_keys,
    float* transported,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int dim = config.d_model;
    int num_channels = config.num_channels;
    float threshold = config.coherence_threshold;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const float* src = state + (b * seq_len + s) * dim;
            float* dst = transported + (b * seq_len + s) * dim;
            
            // Phase-coherent transport via FFT modulation
            std::vector<float> real(dim), imag(dim, 0.0f);
            std::copy(src, src + dim, real.data());
            
            // FFT to frequency domain
            fft::radix2(real.data(), imag.data(), dim, false);
            
            // Apply phase modulation based on keys
            for (int ch = 0; ch < std::min(num_channels, dim); ++ch) {
                float phase = phase_keys[ch];
                float c = std::cos(phase), s = std::sin(phase);
                float r = real[ch], i = imag[ch];
                real[ch] = c * r - s * i;
                imag[ch] = s * r + c * i;
            }
            
            // IFFT back to time domain
            fft::radix2(real.data(), imag.data(), dim, true);
            
            std::copy(real.begin(), real.end(), dst);
        }
    }
}

// =============================================================================
// QUANTUM TELEPORT BUS KERNEL
// =============================================================================

void QuantumTeleportForward(
    const float* source,
    const float* bell_state,
    float* destination,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int dim = config.d_model;
    
    // Simulate quantum teleportation:
    // 1. Alice combines source with her half of Bell pair
    // 2. Bob applies corrections based on measurement
    // For classical simulation, this becomes state transfer with phase correction
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const float* src = source + b * dim;
        float* dst = destination + b * dim;
        
        // Apply Bell-state-based transformation
        int d = 0;
#if defined(QF_AVX2)
        for (; d + 8 <= dim; d += 8) {
            __m256 s = _mm256_loadu_ps(&src[d]);
            __m256 bell = _mm256_loadu_ps(&bell_state[d]);
            // Teleport: dst = src * bell_correction
            __m256 result = _mm256_mul_ps(s, bell);
            _mm256_storeu_ps(&dst[d], result);
        }
#endif
        for (; d < dim; ++d) {
            dst[d] = src[d] * bell_state[d];
        }
    }
}

// =============================================================================
// VQC KERNEL
// =============================================================================

void VQCForward(
    const float* input,
    const float* params,
    float* output,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int dim = config.d_model;
    int num_qubits = config.num_qubits;
    int vqc_layers = config.vqc_layers;
    
    int params_per_layer = 2 * num_qubits;
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const float* x = input + b * dim;
        float* out = output + b * dim;
        
        // Initialize quantum state
        std::vector<float> state(2 * num_qubits);
        for (int q = 0; q < num_qubits; ++q) {
            vqc::init_plus(state.data() + 2 * q);
        }
        
        // Apply VQC layers
        for (int l = 0; l < vqc_layers; ++l) {
            const float* layer_params = params + l * params_per_layer;
            
            // Data encoding: modulate angles by input
            std::vector<float> modulated(params_per_layer);
            for (int p = 0; p < params_per_layer; ++p) {
                modulated[p] = layer_params[p] + x[p % dim] * 0.1f;
            }
            
            vqc::apply_vqc_layer(state.data(), modulated.data(), num_qubits);
        }
        
        // Extract features from quantum state
        for (int d = 0; d < dim; ++d) {
            int q_idx = d % num_qubits;
            out[d] = state[2 * q_idx] * state[2 * q_idx];  // Born rule
        }
    }
}

// =============================================================================
// TENSOR RING VQC KERNEL
// =============================================================================

void TensorRingVQCForward(
    const float* input,
    const float* core_params,
    float* output,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int dim = config.d_model;
    int tr_rank = config.tr_rank;
    int tr_cores = config.tr_cores;
    float bp_mitigation = config.bp_mitigation_strength;
    
    int core_size = tr_rank * tr_rank;
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const float* x = input + b * dim;
        float* out = output + b * dim;
        
        // Contract tensor ring cores
        std::vector<float> result(tr_rank * tr_rank, 0.0f);
        
        // Initialize with identity-like
        for (int r = 0; r < tr_rank; ++r) {
            result[r * tr_rank + r] = 1.0f;
        }
        
        // Contract each core
        for (int c = 0; c < tr_cores; ++c) {
            const float* core = core_params + c * core_size;
            std::vector<float> temp(tr_rank * tr_rank, 0.0f);
            
            // Matrix multiply with data modulation
            for (int i = 0; i < tr_rank; ++i) {
                for (int j = 0; j < tr_rank; ++j) {
                    for (int k = 0; k < tr_rank; ++k) {
                        float mod = 1.0f + x[c % dim] * 0.01f;
                        temp[i * tr_rank + j] += 
                            result[i * tr_rank + k] * core[k * tr_rank + j] * mod;
                    }
                }
            }
            std::copy(temp.begin(), temp.end(), result.begin());
            
            // Neural BP mitigation: add small noise
            if (bp_mitigation > 0.0f) {
                for (int r = 0; r < tr_rank * tr_rank; ++r) {
                    result[r] += bp_mitigation * (0.5f - 0.5f * std::cos(result[r]));
                }
            }
        }
        
        // Extract output from contraction
        for (int d = 0; d < dim; ++d) {
            out[d] = result[(d * d) % (tr_rank * tr_rank)];
        }
    }
}

// =============================================================================
// QUANTUM MEASUREMENT KERNEL
// =============================================================================

void QuantumMeasurementForward(
    const float* state,
    float* probabilities,
    float* collapsed,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int num_qubits = config.num_qubits;
    int num_amps = 2 * num_qubits;
    
    for (int b = 0; b < batch_size; ++b) {
        const float* s = state + b * num_amps;
        float* probs = probabilities + b * num_amps;
        
        // Born rule: P = |amplitude|²
        for (int a = 0; a < num_amps; ++a) {
            probs[a] = s[a] * s[a];
        }
        
        // Optional collapse
        if (collapsed != nullptr) {
            float* c = collapsed + b * num_amps;
            std::copy(s, s + num_amps, c);
            
            // Collapse each qubit probabilistically
            std::mt19937 rng(b);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            for (int q = 0; q < num_qubits; ++q) {
                float p0 = c[2*q] * c[2*q];
                float p1 = c[2*q+1] * c[2*q+1];
                float total = p0 + p1 + 1e-10f;
                
                if (dist(rng) < p0 / total) {
                    c[2*q] = 1.0f;
                    c[2*q+1] = 0.0f;
                } else {
                    c[2*q] = 0.0f;
                    c[2*q+1] = 1.0f;
                }
            }
        }
    }
}

// =============================================================================
// QUANTUM CRYSTALLIZATION KERNEL
// =============================================================================

void QuantumCrystallizationForward(
    const float* state,
    float* crystallized,
    float* memory,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int dim = config.d_model;
    float rate = config.crystallization_rate;
    int slots = config.memory_slots;
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const float* s = state + b * dim;
        float* cry = crystallized + b * dim;
        
        // Crystallization: stabilize important patterns
        // Use exponential moving average to "crystallize" recurring patterns
        
        for (int d = 0; d < dim; ++d) {
            float current = s[d];
            float memorized = (memory != nullptr) ? memory[d] : 0.0f;
            
            // Crystallize based on difference from memory
            float diff = std::abs(current - memorized);
            float crystal_factor = std::exp(-diff / (rate + 1e-6f));
            
            cry[d] = crystal_factor * memorized + (1.0f - crystal_factor) * current;
            
            // Update memory
            if (memory != nullptr) {
                memory[d] = rate * memory[d] + (1.0f - rate) * current;
            }
        }
    }
}

// =============================================================================
// QUANTUM FIDELITY LOSS KERNEL
// =============================================================================

void QuantumFidelityLoss(
    const float* state,
    const float* target,
    float* loss,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int dim = config.d_model;
    
    *loss = 0.0f;
    
    for (int b = 0; b < batch_size; ++b) {
        const float* s = state + b * dim;
        const float* t = target + b * dim;
        
        // Fidelity = |<state|target>|²
        float overlap = 0.0f;
        float norm_s = 0.0f, norm_t = 0.0f;
        
        for (int d = 0; d < dim; ++d) {
            overlap += s[d] * t[d];
            norm_s += s[d] * s[d];
            norm_t += t[d] * t[d];
        }
        
        float fidelity = overlap * overlap / (norm_s * norm_t + config.epsilon);
        
        // Loss = 1 - fidelity (we want to maximize fidelity)
        *loss += (1.0f - fidelity);
    }
    
    *loss /= batch_size;
}

// =============================================================================
// QUANTUM DROPOUT KERNEL
// =============================================================================

void QuantumDropoutForward(
    const float* state,
    float* output,
    bool training,
    uint64_t seed,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int num_qubits = config.num_qubits;
    int num_amps = 2 * num_qubits;
    float dropout_rate = config.dropout_rate;
    
    for (int b = 0; b < batch_size; ++b) {
        const float* s = state + b * num_amps;
        float* out = output + b * num_amps;
        
        std::copy(s, s + num_amps, out);
        
        if (training) {
            measurement::measurement_dropout(out, num_qubits, dropout_rate, seed + b);
        }
    }
}

// =============================================================================
// QUANTUM CURRICULUM SCORE KERNEL
// =============================================================================

void QuantumCurriculumScore(
    const float* input,
    float* score,
    const QuantumConfig& config) {
    
    int batch_size = config.batch_size;
    int seq_len = config.seq_len;
    int dim = config.d_model;
    bool use_fft = config.use_fft_analysis;
    
    for (int b = 0; b < batch_size; ++b) {
        float complexity = 0.0f;
        
        if (use_fft) {
            // FFT-based spectral complexity
            std::vector<float> real(dim), imag(dim, 0.0f);
            
            // Average across sequence
            for (int s = 0; s < seq_len; ++s) {
                const float* x = input + (b * seq_len + s) * dim;
                for (int d = 0; d < dim; ++d) {
                    real[d] += x[d] / seq_len;
                }
            }
            
            // FFT
            fft::radix2(real.data(), imag.data(), dim, false);
            
            // Compute spectral entropy
            float total_power = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float power = real[d] * real[d] + imag[d] * imag[d];
                total_power += power;
            }
            
            float entropy = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float power = real[d] * real[d] + imag[d] * imag[d];
                float p = power / (total_power + config.epsilon);
                if (p > 1e-10f) {
                    entropy -= p * std::log(p);
                }
            }
            
            complexity = entropy / std::log(static_cast<float>(dim));  // Normalize [0, 1]
        } else {
            // Simple variance-based complexity
            for (int s = 0; s < seq_len; ++s) {
                const float* x = input + (b * seq_len + s) * dim;
                float mean = 0.0f, var = 0.0f;
                
                for (int d = 0; d < dim; ++d) {
                    mean += x[d];
                }
                mean /= dim;
                
                for (int d = 0; d < dim; ++d) {
                    float diff = x[d] - mean;
                    var += diff * diff;
                }
                
                complexity += std::sqrt(var / dim);
            }
            complexity /= seq_len;
        }
        
        score[b] = complexity;
    }
}

}  // namespace quantum
}  // namespace saguaro

// =============================================================================
// TENSORFLOW OP REGISTRATION
// =============================================================================

using namespace tensorflow;

REGISTER_OP("UnifiedQuantumOp")
    .Input("input: float")
    .Input("params: float")
    .Input("aux_input: float")
    .Output("output: float")
    .Attr("op_type: int = 0")
    .Attr("batch_size: int = 1")
    .Attr("seq_len: int = 512")
    .Attr("d_model: int = 512")
    .Attr("vocab_size: int = 32000")
    .Attr("num_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("d_ff: int = 2048")
    .Attr("num_bundles: int = 4")
    .Attr("tr_rank: int = 8")
    .Attr("tr_cores: int = 4")
    .Attr("dropout_rate: float = 0.1")
    .Attr("epsilon: float = 1e-6")
    .Attr("training: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Unified Quantum Foundation Operation.

Consolidates 15 quantum mechanisms into a single dispatched op.
)doc");

class UnifiedQuantumOp : public OpKernel {
public:
    explicit UnifiedQuantumOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        int op_type_int;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("op_type", &op_type_int));
        config_.op_type = static_cast<saguaro::quantum::QuantumOpType>(op_type_int);
        
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &config_.batch_size));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seq_len", &config_.seq_len));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("d_model", &config_.d_model));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vocab_size", &config_.vocab_size));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_qubits", &config_.num_qubits));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_layers", &config_.vqc_layers));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("d_ff", &config_.d_ff));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_bundles", &config_.num_bundles));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("tr_rank", &config_.tr_rank));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("tr_cores", &config_.tr_cores));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout_rate", &config_.dropout_rate));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &config_.epsilon));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training_));
        
        config_.embedding_dim = config_.d_model;
    }
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& input = ctx->input(0);
        const Tensor& params = ctx->input(1);
        const Tensor& aux_input = ctx->input(2);
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
        
        const float* input_data = input.flat<float>().data();
        const float* params_data = params.flat<float>().data();
        const float* aux_data = aux_input.flat<float>().data();
        float* output_data = output->flat<float>().data();
        
        // Create a runtime config with actual tensor dimensions
        // This is critical for EXPERT ops where batch_size determines num_tokens
        saguaro::quantum::QuantumConfig runtime_config = config_;
        
        // Infer batch_size from input tensor shape
        // For 2D input [num_tokens, d_model]: batch_size = num_tokens, seq_len = 1
        // For 3D input [batch, seq, d_model]: batch_size = batch, seq_len = seq
        auto input_shape = input.shape();
        if (input_shape.dims() == 2) {
            runtime_config.batch_size = static_cast<int>(input_shape.dim_size(0));
            runtime_config.seq_len = 1;
            runtime_config.d_model = static_cast<int>(input_shape.dim_size(1));
        } else if (input_shape.dims() == 3) {
            runtime_config.batch_size = static_cast<int>(input_shape.dim_size(0));
            runtime_config.seq_len = static_cast<int>(input_shape.dim_size(1));
            runtime_config.d_model = static_cast<int>(input_shape.dim_size(2));
        }
        
        saguaro::quantum::UnifiedQuantumForward(
            input_data, params_data, output_data, runtime_config, aux_data, training_
        );
    }

private:
    saguaro::quantum::QuantumConfig config_;
    bool training_ = false;
};

REGISTER_KERNEL_BUILDER(Name("UnifiedQuantumOp").Device(DEVICE_CPU), UnifiedQuantumOp);
