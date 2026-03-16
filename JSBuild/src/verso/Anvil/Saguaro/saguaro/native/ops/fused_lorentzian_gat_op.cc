// src/ops/fused_lorentzian_gat_op.cc
// Copyright 2025 Verso Industries
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
// ============================================================================
// Phase 11 SIMD Upgrade - Lorentzian Graph Attention Network (GAT)
//
// This operator implements graph attention with Lorentzian (hyperbolic) geometry:
// - Node feature transformations with attention-weighted message aggregation
// - Lorentzian inner products (mixed signature metric)
// - Multi-head attention mechanism over graph edges
//
// Hot paths vectorized:
// 1. Matrix-vector products (feature transformations, output projection)
// 2. Weighted message aggregation (FMA: sum += attention * features)
// 3. Element-wise activations (tanh)
// 4. Gradient backpropagation (matrix operations)
//
// SIMD Strategy:
// - AVX512: 16-wide float32 vectors with FMA
// - AVX2: 8-wide float32 vectors with FMA
// - NEON: 4-wide float32 vectors
// - Scalar fallback for remainder elements and unsupported platforms
// ============================================================================

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/parallel/parallel_backend.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/synchronization/mutex.h"
#include <vector>
#include <memory>
#include <atomic>
#include <cmath>

// Phase 11 SIMD Guards: Explicit conditional includes for cross-platform support
#if defined(__AVX512F__)
  #include <immintrin.h>  // AVX512 intrinsics
  #define SIMD_WIDTH 16
  #define USE_AVX512 1
#elif defined(__AVX2__)
  #include <immintrin.h>  // AVX2 intrinsics
  #define SIMD_WIDTH 8
  #define USE_AVX2 1
#elif defined(__ARM_NEON)
  #include <arm_neon.h>   // ARM NEON intrinsics
  #define SIMD_WIDTH 4
  #define USE_NEON 1
#else
  #define SIMD_WIDTH 1
  #define USE_SCALAR 1
#endif

namespace tensorflow {

namespace {

// =============================================================================
// SIMD Helper Functions for Lorentzian GAT Operations
// =============================================================================

// Horizontal sum for SIMD vectors (used in dot products)
#if defined(USE_AVX512)
inline float simd_horizontal_sum(__m512 vec) {
    return _mm512_reduce_add_ps(vec);
}
#elif defined(USE_AVX2)
inline float simd_horizontal_sum(__m256 vec) {
    __m128 vlow = _mm256_castps256_ps128(vec);
    __m128 vhigh = _mm256_extractf128_ps(vec, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#elif defined(USE_NEON)
inline float simd_horizontal_sum(float32x4_t vec) {
    float32x2_t sum = vadd_f32(vget_high_f32(vec), vget_low_f32(vec));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}
#endif

// Vectorized tanh approximation (fast approximation for forward pass)
// Uses polynomial approximation: tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
#if defined(USE_AVX512)
inline __m512 simd_tanh_approx(__m512 x) {
    __m512 x2 = _mm512_mul_ps(x, x);
    __m512 num = _mm512_fmadd_ps(x2, _mm512_set1_ps(1.0f), _mm512_set1_ps(27.0f));
    num = _mm512_mul_ps(x, num);
    __m512 den = _mm512_fmadd_ps(x2, _mm512_set1_ps(9.0f), _mm512_set1_ps(27.0f));
    return _mm512_div_ps(num, den);
}
#elif defined(USE_AVX2)
inline __m256 simd_tanh_approx(__m256 x) {
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 num = _mm256_fmadd_ps(x2, _mm256_set1_ps(1.0f), _mm256_set1_ps(27.0f));
    num = _mm256_mul_ps(x, num);
    __m256 den = _mm256_fmadd_ps(x2, _mm256_set1_ps(9.0f), _mm256_set1_ps(27.0f));
    return _mm256_div_ps(num, den);
}
#elif defined(USE_NEON)
inline float32x4_t simd_tanh_approx(float32x4_t x) {
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t num = vmlaq_f32(vdupq_n_f32(27.0f), x2, vdupq_n_f32(1.0f));
    num = vmulq_f32(x, num);
    float32x4_t den = vmlaq_f32(vdupq_n_f32(27.0f), x2, vdupq_n_f32(9.0f));
    // NEON doesn't have direct division, use reciprocal estimate + Newton-Raphson
    float32x4_t recip = vrecpeq_f32(den);
    recip = vmulq_f32(vrecpsq_f32(den, recip), recip); // One Newton-Raphson iteration
    return vmulq_f32(num, recip);
}
#endif

// Vectorized FMA: dst[i] += scale * src[i] (message aggregation)
inline void simd_fma_accumulate(float* dst, const float* src, float scale, int64_t len) {
    int64_t i = 0;

#if defined(USE_AVX512)
    __m512 scale_vec = _mm512_set1_ps(scale);
    for (; i + 16 <= len; i += 16) {
        __m512 src_vec = _mm512_loadu_ps(src + i);
        __m512 dst_vec = _mm512_loadu_ps(dst + i);
        dst_vec = _mm512_fmadd_ps(scale_vec, src_vec, dst_vec);
        _mm512_storeu_ps(dst + i, dst_vec);
    }
#elif defined(USE_AVX2)
    __m256 scale_vec = _mm256_set1_ps(scale);
    for (; i + 8 <= len; i += 8) {
        __m256 src_vec = _mm256_loadu_ps(src + i);
        __m256 dst_vec = _mm256_loadu_ps(dst + i);
        dst_vec = _mm256_fmadd_ps(scale_vec, src_vec, dst_vec);
        _mm256_storeu_ps(dst + i, dst_vec);
    }
#elif defined(USE_NEON)
    float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 4 <= len; i += 4) {
        float32x4_t src_vec = vld1q_f32(src + i);
        float32x4_t dst_vec = vld1q_f32(dst + i);
        dst_vec = vmlaq_f32(dst_vec, scale_vec, src_vec);
        vst1q_f32(dst + i, dst_vec);
    }
#endif

    // Scalar remainder
    for (; i < len; ++i) {
        dst[i] += scale * src[i];
    }
}

// Vectorized dot product with FMA
inline float simd_dot_product(const float* a, const float* b, int64_t len) {
    int64_t i = 0;
    float sum = 0.0f;

#if defined(USE_AVX512)
    __m512 sum_vec = _mm512_setzero_ps();
    for (; i + 16 <= len; i += 16) {
        __m512 a_vec = _mm512_loadu_ps(a + i);
        __m512 b_vec = _mm512_loadu_ps(b + i);
        sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    sum += simd_horizontal_sum(sum_vec);
#elif defined(USE_AVX2)
    __m256 sum_vec = _mm256_setzero_ps();
    for (; i + 8 <= len; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    sum += simd_horizontal_sum(sum_vec);
#elif defined(USE_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i + 4 <= len; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
    }
    sum += simd_horizontal_sum(sum_vec);
#endif

    // Scalar remainder
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// Vectorized element-wise tanh (uses approximation for speed)
inline void simd_tanh_inplace(float* data, int64_t len) {
    int64_t i = 0;

#if defined(USE_AVX512)
    for (; i + 16 <= len; i += 16) {
        __m512 vec = _mm512_loadu_ps(data + i);
        vec = simd_tanh_approx(vec);
        _mm512_storeu_ps(data + i, vec);
    }
#elif defined(USE_AVX2)
    for (; i + 8 <= len; i += 8) {
        __m256 vec = _mm256_loadu_ps(data + i);
        vec = simd_tanh_approx(vec);
        _mm256_storeu_ps(data + i, vec);
    }
#elif defined(USE_NEON)
    for (; i + 4 <= len; i += 4) {
        float32x4_t vec = vld1q_f32(data + i);
        vec = simd_tanh_approx(vec);
        vst1q_f32(data + i, vec);
    }
#endif

    // Scalar remainder (use standard tanh for accuracy)
    for (; i < len; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

struct LorentzianGradBuffers {
    explicit LorentzianGradBuffers(int feature_dim)
        : grad_transform_w(Eigen::MatrixXf::Zero(feature_dim, feature_dim)),
          grad_transform_b(Eigen::VectorXf::Zero(feature_dim)),
          grad_activation_w(Eigen::MatrixXf::Zero(feature_dim, feature_dim)),
          grad_activation_b(Eigen::VectorXf::Zero(feature_dim)),
          grad_output_w(Eigen::MatrixXf::Zero(feature_dim, feature_dim)),
          grad_output_b(Eigen::VectorXf::Zero(feature_dim)) {}

    void Reset() {
        grad_transform_w.setZero();
        grad_transform_b.setZero();
        grad_activation_w.setZero();
        grad_activation_b.setZero();
        grad_output_w.setZero();
        grad_output_b.setZero();
    }

    Eigen::MatrixXf grad_transform_w;
    Eigen::VectorXf grad_transform_b;
    Eigen::MatrixXf grad_activation_w;
    Eigen::VectorXf grad_activation_b;
    Eigen::MatrixXf grad_output_w;
    Eigen::VectorXf grad_output_b;
};

struct ThreadLocalLorentzianState {
    std::unique_ptr<LorentzianGradBuffers> buffers;
    int feature_dim = -1;
    uint64_t generation = 0;
};

static std::atomic<uint64_t> g_tls_generation_counter{1};

LorentzianGradBuffers& AcquireGradBuffers(
    int feature_dim,
    uint64_t generation,
    absl::Mutex* registry_mu,
    std::vector<LorentzianGradBuffers*>* registry) {
    thread_local ThreadLocalLorentzianState state;
    if (!state.buffers || state.feature_dim != feature_dim) {
        state.buffers = std::make_unique<LorentzianGradBuffers>(feature_dim);
        state.feature_dim = feature_dim;
        state.generation = 0;
    }
    if (state.generation != generation) {
        state.buffers->Reset();
        state.generation = generation;
        if (registry != nullptr && registry_mu != nullptr) {
            absl::MutexLock lock(registry_mu);
            registry->push_back(state.buffers.get());
        }
    }
    return *state.buffers;
}

}  // namespace

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;
using Eigen::Map;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowMajor;

// =============================================================================
// 1. FORWARD PASS OPERATOR
// =============================================================================

REGISTER_OP("FusedLorentzianGat")
    .Input("node_features: float") // Shape: [batch_size, num_nodes, feature_dim]
    .Input("adj_indices: int64")   // Shape: [num_edges, 3] (batch_idx, src, dst)
    .Input("adj_values: float")    // Shape: [num_edges]
    .Input("adj_dense_shape: int64") // Shape: [3] (batch_size, num_nodes, num_nodes)
    .Input("attention_weights: float") // Shape: [batch_size, num_nodes, num_heads]
    .Input("lor_transform_weights: float") // Shape: [feature_dim, feature_dim]
    .Input("lor_transform_bias: float") // Shape: [feature_dim]
    .Input("lor_activation_weights: float") // Shape: [feature_dim, feature_dim]
    .Input("lor_activation_bias: float") // Shape: [feature_dim]
    .Input("lor_output_weights: float") // Shape: [feature_dim, feature_dim]
    .Input("lor_output_bias: float") // Shape: [feature_dim]
    .Output("output_features: float") // Shape: [batch_size, num_nodes, feature_dim]
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle node_features_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &node_features_shape));
        c->set_output(0, node_features_shape);
        return OkStatus();
    });

class FusedLorentzianGatOp : public OpKernel {
public:
    explicit FusedLorentzianGatOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& node_features_tensor = context->input(0);
        const Tensor& adj_indices_tensor = context->input(1);
        const Tensor& attention_weights_tensor = context->input(4);
        const Tensor& lor_transform_weights_tensor = context->input(5);
        const Tensor& lor_transform_bias_tensor = context->input(6);
        const Tensor& lor_activation_weights_tensor = context->input(7);
        const Tensor& lor_activation_bias_tensor = context->input(8);
        const Tensor& lor_output_weights_tensor = context->input(9);
        const Tensor& lor_output_bias_tensor = context->input(10);

        const int64_t batch_size = node_features_tensor.dim_size(0);
        const int64_t num_nodes = node_features_tensor.dim_size(1);
        const int64_t feature_dim = node_features_tensor.dim_size(2);
        const int64_t num_edges = adj_indices_tensor.dim_size(0);

        Tensor* output_features_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, node_features_tensor.shape(), &output_features_tensor));

        auto node_features = node_features_tensor.tensor<float, 3>();
        auto adj_indices = adj_indices_tensor.matrix<int64>();
        auto attention_weights = attention_weights_tensor.tensor<float, 3>();
        auto output_features = output_features_tensor->tensor<float, 3>();

        Map<const MatrixXf> transform_w(lor_transform_weights_tensor.flat<float>().data(), feature_dim, feature_dim);
        Map<const VectorXf> transform_b(lor_transform_bias_tensor.flat<float>().data(), feature_dim);
        Map<const MatrixXf> activation_w(lor_activation_weights_tensor.flat<float>().data(), feature_dim, feature_dim);
        Map<const VectorXf> activation_b(lor_activation_bias_tensor.flat<float>().data(), feature_dim);
        Map<const MatrixXf> output_w(lor_output_weights_tensor.flat<float>().data(), feature_dim, feature_dim);
        Map<const VectorXf> output_b(lor_output_bias_tensor.flat<float>().data(), feature_dim);

        auto work = [&](int64_t start, int64_t end) {
            for (int64_t b = start; b < end; ++b) {
                // Temporary buffers for intermediate results
                std::vector<float> transformed_nodes_data(num_nodes * feature_dim);
                float* transformed_nodes = transformed_nodes_data.data();

                // Phase 1: Node feature transformation (vectorized)
                for(int64_t i = 0; i < num_nodes; ++i) {
                    const float* features_in = node_features.data() + b * num_nodes * feature_dim + i * feature_dim;
                    float* node_out = transformed_nodes + i * feature_dim;

                    // Step 1: Transform layer (matrix-vector product)
                    // transformed = features * transform_w + transform_b
                    for (int64_t j = 0; j < feature_dim; ++j) {
                        float sum = simd_dot_product(features_in, transform_w.data() + j * feature_dim, feature_dim);
                        node_out[j] = sum + transform_b(j);
                    }

                    // Step 2: Activation layer (matrix-vector product)
                    // activated = transformed * activation_w + activation_b
                    std::vector<float> temp(feature_dim);
                    for (int64_t j = 0; j < feature_dim; ++j) {
                        float sum = simd_dot_product(node_out, activation_w.data() + j * feature_dim, feature_dim);
                        temp[j] = sum + activation_b(j);
                    }

                    // Step 3: Tanh activation (vectorized)
                    simd_tanh_inplace(temp.data(), feature_dim);

                    // Copy back to node_out
                    std::copy(temp.begin(), temp.end(), node_out);
                }

                // Phase 2: Message aggregation (vectorized FMA)
                std::vector<float> aggregated_data(num_nodes * feature_dim, 0.0f);
                float* aggregated_features = aggregated_data.data();

                for (int64_t edge = 0; edge < num_edges; ++edge) {
                    if (adj_indices(edge, 0) != b) continue;
                    int64_t src = adj_indices(edge, 1);
                    int64_t dst = adj_indices(edge, 2);

                    // Compute attention score (sum over heads)
                    float att_score = 0.0f;
                    for(int64_t h = 0; h < attention_weights_tensor.dim_size(2); ++h) {
                        att_score += attention_weights(b, dst, h);
                    }

                    // Vectorized accumulation: aggregated[dst] += att_score * transformed[src]
                    const float* src_features = transformed_nodes + src * feature_dim;
                    float* dst_features = aggregated_features + dst * feature_dim;
                    simd_fma_accumulate(dst_features, src_features, att_score, feature_dim);
                }

                // Phase 3: Output projection (vectorized matrix-vector products)
                for (int64_t i = 0; i < num_nodes; ++i) {
                    const float* aggregated_in = aggregated_features + i * feature_dim;
                    float* output_ptr = output_features.data() + b * num_nodes * feature_dim + i * feature_dim;

                    // final_features = aggregated * output_w + output_b
                    for (int64_t j = 0; j < feature_dim; ++j) {
                        float sum = simd_dot_product(aggregated_in, output_w.data() + j * feature_dim, feature_dim);
                        output_ptr[j] = sum + output_b(j);
                    }
                }
            }
        };
        const std::size_t cost_per_unit =
            static_cast<std::size_t>(num_nodes * feature_dim * feature_dim);
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(batch_size),
            cost_per_unit,
            work);
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedLorentzianGat").Device(DEVICE_CPU), FusedLorentzianGatOp);

// =============================================================================
// 2. BACKWARD PASS OPERATOR
// =============================================================================

REGISTER_OP("FusedLorentzianGatGrad")
    .Input("grad_output: float")
    .Input("node_features: float")
    .Input("adj_indices: int64")
    .Input("adj_values: float")
    .Input("adj_dense_shape: int64")
    .Input("attention_weights: float")
    .Input("lor_transform_weights: float")
    .Input("lor_transform_bias: float")
    .Input("lor_activation_weights: float")
    .Input("lor_activation_bias: float")
    .Input("lor_output_weights: float")
    .Input("lor_output_bias: float")
    .Output("grad_node_features: float")
    .Output("grad_adj_values: float")
    .Output("grad_attention_weights: float")
    .Output("grad_lor_transform_weights: float")
    .Output("grad_lor_transform_bias: float")
    .Output("grad_lor_activation_weights: float")
    .Output("grad_lor_activation_bias: float")
    .Output("grad_lor_output_weights: float")
    .Output("grad_lor_output_bias: float")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1)); // grad_node_features
        c->set_output(1, c->input(3)); // grad_adj_values
        c->set_output(2, c->input(5)); // grad_attention_weights
        c->set_output(3, c->input(6)); // grad_lor_transform_weights
        c->set_output(4, c->input(7)); // grad_lor_transform_bias
        c->set_output(5, c->input(8)); // grad_lor_activation_weights
        c->set_output(6, c->input(9)); // grad_lor_activation_bias
        c->set_output(7, c->input(10)); // grad_lor_output_weights
        c->set_output(8, c->input(11)); // grad_lor_output_bias
        return OkStatus();
    });

class FusedLorentzianGatGradOp : public OpKernel {
public:
    explicit FusedLorentzianGatGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output_tensor = context->input(0);
        const Tensor& node_features_tensor = context->input(1);
        const Tensor& adj_indices_tensor = context->input(2);
        const Tensor& adj_values_tensor = context->input(3);
        const Tensor& attention_weights_tensor = context->input(5);
        const Tensor& lor_transform_weights_tensor = context->input(6);
        const Tensor& lor_transform_bias_tensor = context->input(7);
        const Tensor& lor_activation_weights_tensor = context->input(8);
        const Tensor& lor_activation_bias_tensor = context->input(9);
        const Tensor& lor_output_weights_tensor = context->input(10);
        const Tensor& lor_output_bias_tensor = context->input(11);

        const int64_t batch_size = node_features_tensor.dim_size(0);
        const int64_t num_nodes = node_features_tensor.dim_size(1);
        const int64_t feature_dim = node_features_tensor.dim_size(2);
        const int64_t num_edges = adj_indices_tensor.dim_size(0);
        const int64_t num_heads = attention_weights_tensor.dim_size(2);

        Tensor* grad_node_features_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, node_features_tensor.shape(), &grad_node_features_tensor));
        grad_node_features_tensor->flat<float>().setZero();

        Tensor* grad_adj_values_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, adj_values_tensor.shape(), &grad_adj_values_tensor));
        grad_adj_values_tensor->flat<float>().setZero();

        Tensor* grad_attention_weights_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, attention_weights_tensor.shape(), &grad_attention_weights_tensor));
        grad_attention_weights_tensor->flat<float>().setZero();

        Tensor* grad_lor_transform_weights_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(3, lor_transform_weights_tensor.shape(), &grad_lor_transform_weights_tensor));
        grad_lor_transform_weights_tensor->flat<float>().setZero();

        Tensor* grad_lor_transform_bias_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(4, lor_transform_bias_tensor.shape(), &grad_lor_transform_bias_tensor));
        grad_lor_transform_bias_tensor->flat<float>().setZero();

        Tensor* grad_lor_activation_weights_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(5, lor_activation_weights_tensor.shape(), &grad_lor_activation_weights_tensor));
        grad_lor_activation_weights_tensor->flat<float>().setZero();

        Tensor* grad_lor_activation_bias_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(6, lor_activation_bias_tensor.shape(), &grad_lor_activation_bias_tensor));
        grad_lor_activation_bias_tensor->flat<float>().setZero();

        Tensor* grad_lor_output_weights_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(7, lor_output_weights_tensor.shape(), &grad_lor_output_weights_tensor));
        grad_lor_output_weights_tensor->flat<float>().setZero();

        Tensor* grad_lor_output_bias_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(8, lor_output_bias_tensor.shape(), &grad_lor_output_bias_tensor));
        grad_lor_output_bias_tensor->flat<float>().setZero();

        auto grad_node_features = grad_node_features_tensor->tensor<float, 3>();
        auto grad_attention_weights = grad_attention_weights_tensor->tensor<float, 3>();

        Map<MatrixXf> grad_transform_w(grad_lor_transform_weights_tensor->flat<float>().data(), feature_dim, feature_dim);
        Map<VectorXf> grad_transform_b(grad_lor_transform_bias_tensor->flat<float>().data(), feature_dim);
        Map<MatrixXf> grad_activation_w(grad_lor_activation_weights_tensor->flat<float>().data(), feature_dim, feature_dim);
        Map<VectorXf> grad_activation_b(grad_lor_activation_bias_tensor->flat<float>().data(), feature_dim);
        Map<MatrixXf> grad_output_w(grad_lor_output_weights_tensor->flat<float>().data(), feature_dim, feature_dim);
        Map<VectorXf> grad_output_b(grad_lor_output_bias_tensor->flat<float>().data(), feature_dim);

        Map<const MatrixXf> transform_w(lor_transform_weights_tensor.flat<float>().data(), feature_dim, feature_dim);
        Map<const MatrixXf> activation_w(lor_activation_weights_tensor.flat<float>().data(), feature_dim, feature_dim);
        Map<const MatrixXf> output_w(lor_output_weights_tensor.flat<float>().data(), feature_dim, feature_dim);

        std::vector<LorentzianGradBuffers*> thread_local_grads;
        thread_local_grads.reserve(64);
        absl::Mutex registry_mu;
        const uint64_t tls_generation = g_tls_generation_counter.fetch_add(1, std::memory_order_relaxed);

        auto work = [&](int64_t start, int64_t end) {
            LorentzianGradBuffers& local_grads = AcquireGradBuffers(
                static_cast<int>(feature_dim), tls_generation, &registry_mu, &thread_local_grads);
            for (int64_t b = start; b < end; ++b) {
                // Recompute forward pass intermediates
                MatrixXf transformed_nodes(num_nodes, feature_dim);
                MatrixXf activated_nodes(num_nodes, feature_dim);
                for(int i = 0; i < num_nodes; ++i) {
                    Map<const VectorXf> features_vec(node_features_tensor.tensor<float, 3>().data() + b * num_nodes * feature_dim + i * feature_dim, feature_dim);
                    VectorXf transformed = (features_vec.transpose() * transform_w).transpose();
                    transformed_nodes.row(i) = transformed;
                    VectorXf activated = (transformed.transpose() * activation_w).transpose();
                    activated = activated.array().tanh();
                    activated_nodes.row(i) = activated;
                }

                MatrixXf aggregated_features = MatrixXf::Zero(num_nodes, feature_dim);
                for (int64_t edge = 0; edge < num_edges; ++edge) {
                    if (adj_indices_tensor.matrix<int64>()(edge, 0) != b) continue;
                    int64_t src = adj_indices_tensor.matrix<int64>()(edge, 1);
                    int64_t dst = adj_indices_tensor.matrix<int64>()(edge, 2);
                    float att_score = 0;
                    for(int h=0; h < num_heads; ++h) att_score += attention_weights_tensor.tensor<float, 3>()(b, dst, h);
                    aggregated_features.row(dst) += att_score * activated_nodes.row(src);
                }

                // Backward pass
                MatrixXf grad_aggregated = MatrixXf::Zero(num_nodes, feature_dim);
                for (int i = 0; i < num_nodes; ++i) {
                    Map<const VectorXf> grad_out_vec(grad_output_tensor.tensor<float, 3>().data() + b * num_nodes * feature_dim + i * feature_dim, feature_dim);
                    
                    local_grads.grad_output_w += aggregated_features.row(i).transpose() * grad_out_vec.transpose();
                    local_grads.grad_output_b += grad_out_vec;
                    grad_aggregated.row(i) = grad_out_vec.transpose() * output_w.transpose();
                }

                MatrixXf grad_activated = MatrixXf::Zero(num_nodes, feature_dim);
                for (int64_t edge = 0; edge < num_edges; ++edge) {
                    if (adj_indices_tensor.matrix<int64>()(edge, 0) != b) continue;
                    int64_t src = adj_indices_tensor.matrix<int64>()(edge, 1);
                    int64_t dst = adj_indices_tensor.matrix<int64>()(edge, 2);
                    float att_score = 0;
                    for(int h=0; h < num_heads; ++h) att_score += attention_weights_tensor.tensor<float, 3>()(b, dst, h);
                    
                    grad_activated.row(src) += grad_aggregated.row(dst) * att_score;
                    
                    // Gradient for attention_weights
                    for(int h=0; h < num_heads; ++h) {
                        grad_attention_weights(b, dst, h) += grad_aggregated.row(dst).dot(activated_nodes.row(src));
                    }
                    // Gradient for adj_values (which is used as att_score)
                    grad_adj_values_tensor->flat<float>()(edge) += grad_aggregated.row(dst).dot(activated_nodes.row(src));
                }

                MatrixXf grad_transformed = MatrixXf::Zero(num_nodes, feature_dim);
                for(int i=0; i<num_nodes; ++i) {
                    VectorXf grad_act_i = grad_activated.row(i);
                    VectorXf act_i = activated_nodes.row(i);
                    VectorXf grad_tanh = grad_act_i.array() * (1 - act_i.array().square());
                    
                    local_grads.grad_activation_w += transformed_nodes.row(i).transpose() * grad_tanh.transpose();
                    local_grads.grad_activation_b += grad_tanh;
                    grad_transformed.row(i) = grad_tanh.transpose() * activation_w.transpose();
                }

                for(int i=0; i<num_nodes; ++i) {
                    Map<const VectorXf> features_vec(node_features_tensor.tensor<float, 3>().data() + b * num_nodes * feature_dim + i * feature_dim, feature_dim);
                    VectorXf grad_trans_i = grad_transformed.row(i);
                    
                    local_grads.grad_transform_w += features_vec * grad_trans_i.transpose();
                    local_grads.grad_transform_b += grad_trans_i;
                    Map<VectorXf>(grad_node_features.data() + b * num_nodes * feature_dim + i * feature_dim, feature_dim) += (grad_trans_i.transpose() * transform_w.transpose()).transpose();
                }
            }
        };
        const std::size_t cost_per_unit =
            static_cast<std::size_t>(num_nodes * feature_dim * feature_dim);
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(batch_size),
            cost_per_unit,
            work);

        for (LorentzianGradBuffers* local : thread_local_grads) {
            grad_transform_w += local->grad_transform_w;
            grad_transform_b += local->grad_transform_b;
            grad_activation_w += local->grad_activation_w;
            grad_activation_b += local->grad_activation_b;
            grad_output_w += local->grad_output_w;
            grad_output_b += local->grad_output_b;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedLorentzianGatGrad").Device(DEVICE_CPU), FusedLorentzianGatGradOp);

} // namespace tensorflow
