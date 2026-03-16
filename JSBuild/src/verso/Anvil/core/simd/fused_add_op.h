#ifndef SRC_OPS_FUSED_ADD_OP_H_
#define SRC_OPS_FUSED_ADD_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

template <typename T>
class FusedAddOp : public OpKernel {
 public:
  explicit FusedAddOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* context) override;
};

}  // namespace tensorflow

#endif  // SRC_OPS_FUSED_ADD_OP_H_
