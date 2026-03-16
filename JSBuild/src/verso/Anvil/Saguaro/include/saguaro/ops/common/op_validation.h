#ifndef OPS_COMMON_OP_VALIDATION_H_
#define OPS_COMMON_OP_VALIDATION_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

// Helper function to require a specific rank for a tensor.
inline Status RequireRank(OpKernelContext* context, const Tensor& tensor, int expected_rank,
                          const string& tensor_name) {
  if (tensor.dims() != expected_rank) {
    return errors::InvalidArgument(tensor_name, " must have rank ", expected_rank,
                                   ", but has rank ", tensor.dims());
  }
  return OkStatus();
}

// Helper function to require a specific rank and dimension sizes for a tensor.
inline Status RequireRankAndDims(OpKernelContext* context, const Tensor& tensor,
                                 int expected_rank,
                                 const std::vector<int64>& expected_dims,
                                 const string& tensor_name) {
  if (tensor.dims() != expected_rank) {
    return errors::InvalidArgument(tensor_name, " must have rank ", expected_rank,
                                   ", but has rank ", tensor.dims());
  }
  if (expected_dims.size() != expected_rank) {
    return errors::Internal("Internal error: expected_dims size mismatch for ", tensor_name);
  }
  for (int i = 0; i < expected_rank; ++i) {
    if (expected_dims[i] != -1 && tensor.dim_size(i) != expected_dims[i]) {
      return errors::InvalidArgument(tensor_name, " dimension ", i, " must be ",
                                     expected_dims[i], ", but is ", tensor.dim_size(i));
    }
  }
  return OkStatus();
}

// Helper function to require a specific rank and a specific dimension size for a tensor.
inline Status RequireRankAndDimSize(OpKernelContext* context, const Tensor& tensor,
                                    int expected_rank, int dim_idx, int64 expected_dim_size,
                                    const string& tensor_name) {
  if (tensor.dims() != expected_rank) {
    return errors::InvalidArgument(tensor_name, " must have rank ", expected_rank,
                                   ", but has rank ", tensor.dims());
  }
  if (dim_idx < 0 || dim_idx >= expected_rank) {
    return errors::Internal("Internal error: dim_idx out of bounds for ", tensor_name);
  }
  if (tensor.dim_size(dim_idx) != expected_dim_size) {
    return errors::InvalidArgument(tensor_name, " dimension ", dim_idx, " must be ",
                                   expected_dim_size, ", but is ", tensor.dim_size(dim_idx));
  }
  return OkStatus();
}

}  // namespace tensorflow

#endif  // OPS_COMMON_OP_VALIDATION_H_
