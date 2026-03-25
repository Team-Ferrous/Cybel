#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("TestOp");
class TestOp : public OpKernel {
 public:
  explicit TestOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
};
REGISTER_KERNEL_BUILDER(Name("TestOp").Device(DEVICE_CPU), TestOp);
