// src/ops/structured_sparse_matmul_op.cc
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
// --- ACTION ITEM 3.3.2 COMPLETE ---
// This file implements a custom C++ operator for structured sparse matrix-vector
// multiplication, replacing the previous placeholder.
//
// Key Features:
// 1.  **Band-Diagonal Specialization:** The kernel is optimized for band-diagonal
//     matrices.
// 2.  **CPU Performance Optimization:** Utilizes TBB and AVX intrinsics.
// 3.  **Production Ready:** Includes a corresponding gradient operator.
// 4.  **FIX (2025-10-05):** Converted static Attrs (structure, bands) to dynamic
//     Inputs to ensure compatibility with tf.while_loop and graph-mode gradients.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/parallel/parallel_backend.h"
#include "absl/synchronization/mutex.h"
#include <vector>

// SIMD intrinsics (Phase 11 FULL compliance)
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
#endif // GOOGLE_CUDA


using namespace tensorflow;

// =============================================================================
// Forward Pass Operator
// =============================================================================

REGISTER_OP("StructuredSparseMatmul")
    .Input("matrix_diagonals: float")
    .Input("vector: float")
    // --- START: FIX ---
    // Converted Attrs to Inputs for graph-mode compatibility.
    .Input("structure: string")
    .Input("lower_bands: int32")
    .Input("upper_bands: int32")
    // --- END: FIX ---
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle matrix_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &matrix_shape));
        shape_inference::ShapeHandle vector_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &vector_shape));

        shape_inference::DimensionHandle batch_size = c->Dim(vector_shape, 0);
        shape_inference::DimensionHandle num_rows = c->Dim(matrix_shape, 0);

        c->set_output(0, c->MakeShape({batch_size, num_rows}));
        return OkStatus();
    });


// --- CPU Kernel Implementation ---
namespace { // Anonymous namespace for CPU helpers

// SIMD horizontal sum helpers (Phase 11 FULL compliance)
#if defined(__AVX512F__)
inline float HorizontalSum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#elif defined(__AVX2__)
inline float HorizontalSum(__m256 x) {
    __m128 vlow = _mm256_castps256_ps128(x);
    __m128 vhigh = _mm256_extractf128_ps(x, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}
#elif defined(__ARM_NEON)
inline float HorizontalSum(float32x4_t x) {
    float32x2_t sum_pair = vadd_f32(vget_low_f32(x), vget_high_f32(x));
    return vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
}
#endif

void BandDiagonalMatmulCPU(
    const float* diagonals, const float* vector, float* output,
    int kl, int ku, int num_rows, int num_cols, OpKernelContext* context) {

    const int num_diagonals = kl + ku + 1;

    auto work = [&](int64_t start, int64_t end) {
        for (int i = start; i < end; ++i) {
            float sum = 0.0f;
            int start_col = std::max(0, i - kl);
            int end_col = std::min(num_cols - 1, i + ku);
            const int band_width = end_col - start_col + 1;

            // SIMD vectorization over columns within the band
            int j = start_col;

#if defined(__AVX512F__)
            // AVX512: 16-wide SIMD
            __m512 vsum = _mm512_setzero_ps();
            for (; j + 16 <= end_col + 1 && j + 16 <= num_cols; j += 16) {
                // Load 16 diagonal elements and 16 vector elements
                float diag_buf[16];
                for (int k = 0; k < 16; ++k) {
                    int col_idx = j + k;
                    if (col_idx <= end_col) {
                        int diag_idx = col_idx - i + kl;
                        diag_buf[k] = diagonals[i * num_diagonals + diag_idx];
                    } else {
                        diag_buf[k] = 0.0f;
                    }
                }
                __m512 vdiag = _mm512_loadu_ps(diag_buf);
                __m512 vvec = _mm512_loadu_ps(&vector[j]);
                vsum = _mm512_fmadd_ps(vdiag, vvec, vsum);
            }
            sum += HorizontalSum(vsum);
#elif defined(__AVX2__)
            // AVX2: 8-wide SIMD
            __m256 vsum = _mm256_setzero_ps();
            for (; j + 8 <= end_col + 1 && j + 8 <= num_cols; j += 8) {
                // Load 8 diagonal elements and 8 vector elements
                float diag_buf[8];
                for (int k = 0; k < 8; ++k) {
                    int col_idx = j + k;
                    if (col_idx <= end_col) {
                        int diag_idx = col_idx - i + kl;
                        diag_buf[k] = diagonals[i * num_diagonals + diag_idx];
                    } else {
                        diag_buf[k] = 0.0f;
                    }
                }
                __m256 vdiag = _mm256_loadu_ps(diag_buf);
                __m256 vvec = _mm256_loadu_ps(&vector[j]);
                vsum = _mm256_fmadd_ps(vdiag, vvec, vsum);
            }
            sum += HorizontalSum(vsum);
#elif defined(__ARM_NEON)
            // NEON: 4-wide SIMD
            float32x4_t vsum = vdupq_n_f32(0.0f);
            for (; j + 4 <= end_col + 1 && j + 4 <= num_cols; j += 4) {
                // Load 4 diagonal elements and 4 vector elements
                float diag_buf[4];
                for (int k = 0; k < 4; ++k) {
                    int col_idx = j + k;
                    if (col_idx <= end_col) {
                        int diag_idx = col_idx - i + kl;
                        diag_buf[k] = diagonals[i * num_diagonals + diag_idx];
                    } else {
                        diag_buf[k] = 0.0f;
                    }
                }
                float32x4_t vdiag = vld1q_f32(diag_buf);
                float32x4_t vvec = vld1q_f32(&vector[j]);
                vsum = vfmaq_f32(vsum, vdiag, vvec);
            }
            sum += HorizontalSum(vsum);
#endif

            // Scalar fallback for remainder
            for (; j <= end_col; ++j) {
                int diag_idx = j - i + kl;
                sum += diagonals[i * num_diagonals + diag_idx] * vector[j];
            }
            output[i] = sum;
        }
    };
    const std::size_t cost_per_unit = static_cast<std::size_t>(num_diagonals);
    saguaro::parallel::ForShard(
        static_cast<std::size_t>(num_rows),
        cost_per_unit,
        work);
}
} // anonymous namespace

class StructuredSparseMatmulOpCpu : public OpKernel {
public:
    explicit StructuredSparseMatmulOpCpu(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& matrix_diagonals_tensor = context->input(0);
        const Tensor& vector_tensor = context->input(1);
        // --- START: FIX ---
        const Tensor& structure_tensor = context->input(2);
        const Tensor& lower_bands_tensor = context->input(3);
        const Tensor& upper_bands_tensor = context->input(4);

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(structure_tensor.shape()), errors::InvalidArgument("structure must be a scalar string."));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(lower_bands_tensor.shape()), errors::InvalidArgument("lower_bands must be a scalar int32."));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(upper_bands_tensor.shape()), errors::InvalidArgument("upper_bands must be a scalar int32."));

        const std::string structure = structure_tensor.scalar<tstring>()();
        const int kl = lower_bands_tensor.scalar<int32>()();
        const int ku = upper_bands_tensor.scalar<int32>()();
        // --- END: FIX ---

        const int64 num_rows = matrix_diagonals_tensor.dim_size(0);
        const int64 num_diagonals = matrix_diagonals_tensor.dim_size(1);
        const int64 batch_size = vector_tensor.dim_size(0);
        const int64 num_cols = vector_tensor.dim_size(1);

        OP_REQUIRES(context, kl + ku + 1 == num_diagonals,
            errors::InvalidArgument("Number of diagonals does not match lower_bands + upper_bands + 1"));

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({batch_size, num_rows}), &output_tensor));

        auto diagonals_ptr = matrix_diagonals_tensor.flat<float>().data();
        auto vector_ptr = vector_tensor.flat<float>().data();
        auto output_ptr = output_tensor->flat<float>().data();

        for (int b = 0; b < batch_size; ++b) {
            const float* current_vector = vector_ptr + b * num_cols;
            float* current_output = output_ptr + b * num_rows;

            if (structure == "band_diagonal") {
                BandDiagonalMatmulCPU(diagonals_ptr, current_vector, current_output, kl, ku, num_rows, num_cols, context);
            } else {
                OP_REQUIRES(context, false, errors::InvalidArgument("Unsupported sparse structure: ", structure));
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("StructuredSparseMatmul").Device(DEVICE_CPU), StructuredSparseMatmulOpCpu);


// =============================================================================
// Gradient Pass Operator
// =============================================================================

REGISTER_OP("StructuredSparseMatmulGrad")
    .Input("grad_y: float")
    .Input("matrix_diagonals: float")
    .Input("vector: float")
    // --- START: FIX ---
    .Input("structure: string")
    .Input("lower_bands: int32")
    .Input("upper_bands: int32")
    // --- END: FIX ---
    .Output("grad_matrix_diagonals: float")
    .Output("grad_vector: float")
    // --- START: FIX ---
    .Output("grad_structure: string")
    .Output("grad_lower_bands: int32")
    .Output("grad_upper_bands: int32");
    // --- END: FIX ---


class StructuredSparseMatmulGradOpCpu : public OpKernel {
public:
    explicit StructuredSparseMatmulGradOpCpu(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_y_tensor = context->input(0);
        const Tensor& matrix_diagonals_tensor = context->input(1);
        const Tensor& vector_tensor = context->input(2);
        const Tensor& structure_tensor = context->input(3);
        const Tensor& lower_bands_tensor = context->input(4);
        const Tensor& upper_bands_tensor = context->input(5);

        const std::string structure = structure_tensor.scalar<tstring>()();
        const int kl = lower_bands_tensor.scalar<int32>()();
        const int ku = upper_bands_tensor.scalar<int32>()();

        const int64 num_rows = matrix_diagonals_tensor.dim_size(0);
        const int64 num_diagonals = matrix_diagonals_tensor.dim_size(1);
        const int64 batch_size = vector_tensor.dim_size(0);
        const int64 num_cols = vector_tensor.dim_size(1);

        Tensor* grad_matrix_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, matrix_diagonals_tensor.shape(), &grad_matrix_tensor));
        grad_matrix_tensor->flat<float>().setZero();

        Tensor* grad_vector_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, vector_tensor.shape(), &grad_vector_tensor));
        grad_vector_tensor->flat<float>().setZero();

        // --- START: FIX ---
        // Allocate null outputs for non-differentiable inputs
        Tensor* grad_structure_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}), &grad_structure_tensor));
        Tensor* grad_lower_bands_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({}), &grad_lower_bands_tensor));
        Tensor* grad_upper_bands_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape({}), &grad_upper_bands_tensor));
        // --- END: FIX ---
        
        if (structure != "band_diagonal") {
            OP_REQUIRES(context, false, errors::InvalidArgument("Unsupported sparse structure for gradient: ", structure));
        }

        auto grad_y_ptr = grad_y_tensor.flat<float>().data();
        auto diagonals_ptr = matrix_diagonals_tensor.flat<float>().data();
        auto vector_ptr = vector_tensor.flat<float>().data();
        auto grad_matrix_ptr = grad_matrix_tensor->flat<float>().data();
        auto grad_vector_ptr = grad_vector_tensor->flat<float>().data();
        
        absl::Mutex mu;
        for (int b = 0; b < batch_size; ++b) {
            const float* current_grad_y = grad_y_ptr + b * num_rows;
            const float* current_vector = vector_ptr + b * num_cols;

            auto work_grad_matrix = [&](int64_t start, int64_t end) {
                for (int i = start; i < end; ++i) {
                    int start_col = std::max(0, i - kl);
                    int end_col = std::min(static_cast<int>(num_cols - 1), i + ku);
                    for (int j = start_col; j <= end_col; ++j) {
                        int diag_idx = j - i + kl;
                        absl::MutexLock l(&mu);
                        grad_matrix_ptr[i * num_diagonals + diag_idx] += current_grad_y[i] * current_vector[j];
                    }
                }
            };
            const std::size_t cost_per_row_matrix = static_cast<std::size_t>(num_diagonals);
            saguaro::parallel::ForShard(
                static_cast<std::size_t>(num_rows),
                cost_per_row_matrix,
                work_grad_matrix);
        }

        for (int b = 0; b < batch_size; ++b) {
            const float* current_grad_y = grad_y_ptr + b * num_rows;
            float* current_grad_vector = grad_vector_ptr + b * num_cols;

            auto work_grad_vector = [&](int64_t start, int64_t end) {
                for (int j = start; j < end; ++j) {
                    float sum = 0.0f;
                    int start_row = std::max(0, j - ku);
                    int end_row = std::min(static_cast<int>(num_rows - 1), j + kl);

                    // SIMD vectorization over rows
                    int i = start_row;

#if defined(__AVX512F__)
                    // AVX512: 16-wide SIMD
                    __m512 vsum = _mm512_setzero_ps();
                    for (; i + 16 <= end_row + 1 && i + 16 <= num_rows; i += 16) {
                        // Load 16 diagonal elements and 16 grad_y elements
                        float diag_buf[16];
                        for (int k = 0; k < 16; ++k) {
                            int row_idx = i + k;
                            if (row_idx <= end_row) {
                                int diag_idx = j - row_idx + kl;
                                diag_buf[k] = diagonals_ptr[row_idx * num_diagonals + diag_idx];
                            } else {
                                diag_buf[k] = 0.0f;
                            }
                        }
                        __m512 vdiag = _mm512_loadu_ps(diag_buf);
                        __m512 vgrad = _mm512_loadu_ps(&current_grad_y[i]);
                        vsum = _mm512_fmadd_ps(vdiag, vgrad, vsum);
                    }
                    sum += HorizontalSum(vsum);
#elif defined(__AVX2__)
                    // AVX2: 8-wide SIMD
                    __m256 vsum = _mm256_setzero_ps();
                    for (; i + 8 <= end_row + 1 && i + 8 <= num_rows; i += 8) {
                        // Load 8 diagonal elements and 8 grad_y elements
                        float diag_buf[8];
                        for (int k = 0; k < 8; ++k) {
                            int row_idx = i + k;
                            if (row_idx <= end_row) {
                                int diag_idx = j - row_idx + kl;
                                diag_buf[k] = diagonals_ptr[row_idx * num_diagonals + diag_idx];
                            } else {
                                diag_buf[k] = 0.0f;
                            }
                        }
                        __m256 vdiag = _mm256_loadu_ps(diag_buf);
                        __m256 vgrad = _mm256_loadu_ps(&current_grad_y[i]);
                        vsum = _mm256_fmadd_ps(vdiag, vgrad, vsum);
                    }
                    sum += HorizontalSum(vsum);
#elif defined(__ARM_NEON)
                    // NEON: 4-wide SIMD
                    float32x4_t vsum = vdupq_n_f32(0.0f);
                    for (; i + 4 <= end_row + 1 && i + 4 <= num_rows; i += 4) {
                        // Load 4 diagonal elements and 4 grad_y elements
                        float diag_buf[4];
                        for (int k = 0; k < 4; ++k) {
                            int row_idx = i + k;
                            if (row_idx <= end_row) {
                                int diag_idx = j - row_idx + kl;
                                diag_buf[k] = diagonals_ptr[row_idx * num_diagonals + diag_idx];
                            } else {
                                diag_buf[k] = 0.0f;
                            }
                        }
                        float32x4_t vdiag = vld1q_f32(diag_buf);
                        float32x4_t vgrad = vld1q_f32(&current_grad_y[i]);
                        vsum = vfmaq_f32(vsum, vdiag, vgrad);
                    }
                    sum += HorizontalSum(vsum);
#endif

                    // Scalar fallback for remainder
                    for (; i <= end_row; ++i) {
                        int diag_idx = j - i + kl;
                        sum += diagonals_ptr[i * num_diagonals + diag_idx] * current_grad_y[i];
                    }
                    current_grad_vector[j] = sum;
                }
            };
            const std::size_t cost_per_col_vector = static_cast<std::size_t>(kl + ku + 1);
            saguaro::parallel::ForShard(
                static_cast<std::size_t>(num_cols),
                cost_per_col_vector,
                work_grad_vector);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("StructuredSparseMatmulGrad").Device(DEVICE_CPU), StructuredSparseMatmulGradOpCpu);
