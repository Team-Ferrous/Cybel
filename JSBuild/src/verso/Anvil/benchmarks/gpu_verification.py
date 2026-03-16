import time
import numpy as np

from core.native.coconut_bridge import CoconutNativeBridge
from config.settings import AGENTIC_THINKING


def run_benchmark(use_gpu=False):
    print(
        f"\n--- Benchmarking COCONUT (GPU={'ENABLED' if use_gpu else 'DISABLED'}) ---"
    )

    # Force GPU mode in settings for this run
    AGENTIC_THINKING["coconut_use_gpu"] = use_gpu

    # Initialize bridge
    bridge = CoconutNativeBridge()

    # Set up data
    batch_size = 4
    num_paths = 8
    dim = 4096
    hidden_state = np.random.randn(batch_size, dim).astype(np.float32)

    # Warm up
    print("Warming up...")
    warm_paths = bridge.expand_paths(hidden_state, num_paths)
    print(f"DEBUG: warm_paths type = {type(warm_paths)}")

    # Time it
    print("Running iterations...")
    start_time = time.time()
    iterations = 10

    for i in range(iterations):
        # 1. Expand
        paths = bridge.expand_paths(hidden_state, num_paths)

        # 2. Evolve (MLP reasoning)
        # Note: We'll use random weights for benchmarking speed
        w1 = np.random.randn(dim, dim * 4).astype(np.float32) * 0.02
        b1 = np.zeros(dim * 4).astype(np.float32)
        w2 = np.random.randn(dim * 4, dim).astype(np.float32) * 0.02
        b2 = np.zeros(dim).astype(np.float32)
        gamma = np.ones(dim).astype(np.float32)
        beta = np.zeros(dim).astype(np.float32)

        bridge.evolve_paths(paths, gamma, beta, w1, b1, w2, b2, dim * 4)

        # 3. Score & Aggregate
        amplitudes = bridge.score_paths(paths, hidden_state)
        bridge.aggregate_paths(paths, amplitudes)

    end_time = time.time()
    avg_time = (end_time - start_time) / iterations
    print(f"AVERAGE ITERATION TIME: {avg_time:.4f}s")

    if use_gpu:
        try:
            device_info = bridge.backend.get_device_info()
            print(f"Device Info: {device_info}")
        except Exception:
            pass


if __name__ == "__main__":
    # Test CPU
    run_benchmark(use_gpu=False)

    # Test GPU
    run_benchmark(use_gpu=True)
