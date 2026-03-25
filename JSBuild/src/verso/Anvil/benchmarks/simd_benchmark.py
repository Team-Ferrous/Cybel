import time
import numpy as np
import sys
import os

# Check if SIMD lib is built, handled by imports hopefully
sys.path.append(os.getcwd())

from core.simd.simd_ops import SIMDOps


def benchmark_simd():
    """Compare SIMD ops vs NumPy."""
    print("Initializing SIMD Benchmark...")

    try:
        simd = SIMDOps()
        simd_avail = True
    except Exception as e:
        print(f"SIMD lib load failed: {e}")
        simd_avail = False

    size = 10_000_000
    data = np.random.randn(size).astype(np.float32)

    print(f"Data size: {size} floats")

    # 1. NumPy EXP
    start = time.time()
    np.exp(data)
    np_time = time.time() - start
    print(f"NumPy Exp Time: {np_time:.4f}s")

    if simd_avail:
        # Copy data as SIMD runs inplace
        simd_data = data.copy()
        start = time.time()
        simd.exp_inplace(simd_data)
        simd_time = time.time() - start
        print(f"SIMD Exp Time:  {simd_time:.4f}s")
        print(f"Speedup: {np_time / simd_time:.2f}x")
    else:
        print("Skipping SIMD test.")


if __name__ == "__main__":
    benchmark_simd()
