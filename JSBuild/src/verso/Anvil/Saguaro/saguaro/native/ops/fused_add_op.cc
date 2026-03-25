#include "ops/fused_add_op.h"

#include <cstdint>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/bcast.h"

// SIMD intrinsics
#if defined(__AVX512F__)
  #include <immintrin.h>
#elif defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

// Parallel backend for TBB/OpenMP
#include "common/parallel/parallel_backend.h"

namespace tensorflow {

REGISTER_OP("VersoFusedAdd")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float32, float64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

// Vectorized element-wise addition kernel for contiguous arrays
// This function provides SIMD-optimized addition for the common case where
// both inputs have the same shape and can be processed as flat arrays.
inline void VectorizedAdd(const float* a, const float* b, float* c, int64_t size) {
  int64_t i = 0;

#if defined(__AVX512F__)
  // AVX-512: Process 16 floats at a time
  for (; i + 16 <= size; i += 16) {
    __m512 va = _mm512_loadu_ps(&a[i]);
    __m512 vb = _mm512_loadu_ps(&b[i]);
    __m512 vc = _mm512_add_ps(va, vb);
    _mm512_storeu_ps(&c[i], vc);
  }
#elif defined(__AVX2__)
  // AVX2: Process 8 floats at a time
  for (; i + 8 <= size; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);
    __m256 vb = _mm256_loadu_ps(&b[i]);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(&c[i], vc);
  }
#elif defined(__ARM_NEON)
  // NEON: Process 4 floats at a time
  for (; i + 4 <= size; i += 4) {
    float32x4_t va = vld1q_f32(&a[i]);
    float32x4_t vb = vld1q_f32(&b[i]);
    float32x4_t vc = vaddq_f32(va, vb);
    vst1q_f32(&c[i], vc);
  }
#endif

  // Scalar fallback for remaining elements
  for (; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

// Overload for double precision (no SIMD optimization, used only for compatibility)
inline void VectorizedAdd(const double* a, const double* b, double* c, int64_t size) {
  for (int64_t i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

template <typename T>
FusedAddOp<T>::FusedAddOp(OpKernelConstruction* context) : OpKernel(context) {}

template <typename T>
void FusedAddOp<T>::Compute(OpKernelContext* context) {
  const Tensor& x_tensor = context->input(0);
  const Tensor& y_tensor = context->input(1);

  BCast bcast(BCast::FromShape(x_tensor.shape()), BCast::FromShape(y_tensor.shape()));
  OP_REQUIRES(context, bcast.IsValid(),
              errors::InvalidArgument("Incompatible shapes for broadcasting: ",
                                      x_tensor.shape().DebugString(), " and ",
                                      y_tensor.shape().DebugString()));

  TensorShape output_shape;
  for (int64_t dim : bcast.output_shape()) {
    output_shape.AddDim(dim);
  }
  Tensor* z_tensor = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &z_tensor));

  auto z_flat = z_tensor->flat<T>();
  auto x_flat = x_tensor.flat<T>();
  auto y_flat = y_tensor.flat<T>();

  const T* x_data = x_flat.data();
  const T* y_data = y_flat.data();
  T* z_data = z_flat.data();
  const int64_t total = z_flat.size();

  // Fast path: Same shape, use vectorized addition with TBB parallelism
  if (x_tensor.shape() == y_tensor.shape()) {
    // Estimate cost per element (memory load + add + store ≈ 10 cycles)
    constexpr int64_t kCostPerElement = 10;

    // Parallelize over chunks of the flat array
    saguaro::parallel::ForShard(
        static_cast<size_t>(total),
        static_cast<size_t>(kCostPerElement),
        [x_data, y_data, z_data](int64_t start, int64_t end) {
          VectorizedAdd(x_data + start, y_data + start, z_data + start, end - start);
        });
    return;
  }

  // Broadcasting path: Use stride-based indexing with TBB parallelism
  const auto& x_reshape = bcast.x_reshape();
  const auto& y_reshape = bcast.y_reshape();
  const auto& z_reshape = bcast.output_shape();

  auto make_strides = [](const BCast::Vec& reshape) {
    std::vector<int64_t> strides(reshape.size(), 0);
    int64_t stride = 1;
    for (int64_t d = reshape.size() - 1; d >= 0; --d) {
      if (reshape[d] == 1) {
        strides[d] = 0;
      } else {
        strides[d] = stride;
        stride *= reshape[d];
      }
    }
    return strides;
  };

  const auto x_strides = make_strides(x_reshape);
  const auto y_strides = make_strides(y_reshape);
  const int64_t ndims = static_cast<int64_t>(z_reshape.size());

  // Estimate cost per element (index computation + loads + add + store ≈ 50 cycles)
  constexpr int64_t kCostPerElementBroadcast = 50;

  // Parallelize over output elements
  saguaro::parallel::ForShard(
      static_cast<size_t>(total),
      static_cast<size_t>(kCostPerElementBroadcast),
      [x_data, y_data, z_data, &x_reshape, &y_reshape, &z_reshape,
       &x_strides, &y_strides, ndims](int64_t start, int64_t end) {
        for (int64_t index = start; index < end; ++index) {
          int64_t x_index = 0;
          int64_t y_index = 0;
          int64_t remainder = index;

          // Compute strided indices for broadcasting
          for (int64_t dim = ndims - 1; dim >= 0; --dim) {
            const int64_t dim_size = z_reshape[dim];
            const int64_t coord = remainder % dim_size;
            remainder /= dim_size;
            if (x_reshape[dim] == dim_size) {
              x_index += coord * x_strides[dim];
            }
            if (y_reshape[dim] == dim_size) {
              y_index += coord * y_strides[dim];
            }
          }

          z_data[index] = x_data[x_index] + y_data[y_index];
        }
      });
}

template class FusedAddOp<float>;
template class FusedAddOp<double>;

REGISTER_KERNEL_BUILDER(Name("VersoFusedAdd").Device(DEVICE_CPU).TypeConstraint<float>("T"),
                        FusedAddOp<float>);
REGISTER_KERNEL_BUILDER(Name("VersoFusedAdd").Device(DEVICE_CPU).TypeConstraint<double>("T"),
                        FusedAddOp<double>);

}  // namespace tensorflow
