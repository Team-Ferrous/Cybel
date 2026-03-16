import numpy as np
import time
from core.simd.simd_ops import SIMDOps
from core.memory.scratch_pool import ScratchPool


def test_simd_vs_numpy():
    ops = SIMDOps()
    N = 1000000
    data = np.random.randn(N).astype(np.float32)

    # EXP
    print(f"Testing EXP on {N} elements...")
    d1 = data.copy()
    start = time.time()
    ops.exp_inplace(d1)
    simd_time = time.time() - start

    d2 = data.copy()
    start = time.time()
    np.exp(d2, out=d2)
    np_time = time.time() - start

    diff = np.max(np.abs(d1 - d2))
    print(
        f"EXP: SIMD={simd_time*1000:.2f}ms, Numpy={np_time*1000:.2f}ms. Max Diff={diff:.6f}"
    )
    assert diff < 1e-3, "EXP mismatch too high"

    # LOG
    print(f"Testing LOG on {N} elements...")
    data_pos = np.abs(data) + 1.0  # Ensure positive
    d1 = data_pos.copy()
    start = time.time()
    ops.log_inplace(d1)
    simd_time = time.time() - start

    d2 = data_pos.copy()
    start = time.time()
    np.log(d2, out=d2)
    np_time = time.time() - start

    diff = np.max(np.abs(d1 - d2))
    print(
        f"LOG: SIMD={simd_time*1000:.2f}ms, Numpy={np_time*1000:.2f}ms. Max Diff={diff:.6f}"
    )
    assert diff < 0.1, "LOG mismatch too high"


def test_scratch_pool():
    pool = ScratchPool()
    print("Testing ScratchPool...")

    # Allocate
    buf = pool.get(1024)
    buf[:] = 1.0
    assert buf.shape == (1024,)
    assert buf[0] == 1.0

    # Re-get same size (should be same pointer usually, but wrappers might hide it)
    buf2 = pool.get(1024)
    assert np.allclose(buf2, 1.0)  # Content persists if not realloc

    pool.clear()
    print("ScratchPool test passed.")


if __name__ == "__main__":
    test_simd_vs_numpy()
    test_scratch_pool()
