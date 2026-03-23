import numpy as np
import _kernel_lib
from   futhark_ffi import Futhark

kernels = Futhark(_kernel_lib)

x = np.random.rand(10, 3).astype(np.float32)
y = np.random.rand(8, 3).astype(np.float32)

# Call RBF (kind=0) || (kind=LAPLACE)
res_dev = kernels.dispatch_kernel(0, x, y, 1.0, 0.0, 0)
res = kernels.from_futhark(res_dev)

print(res.shape)
print(res)