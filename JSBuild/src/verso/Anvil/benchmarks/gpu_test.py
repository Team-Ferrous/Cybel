import sys
import os
import time

# Add root to sys.path
sys.path.append(os.getcwd())

from core.native.engine import NativeInferenceEngine, detect_gpu
import numpy as np


def test_inference_engine():
    print("--- Testing Inference Engine ---")
    gpu_info = detect_gpu()
    print(f"Detected GPU: {gpu_info}")

    model_path = "/usr/share/ollama/.ollama/models/blobs/sha256-491ba81786c46a345a5da9a60cdb9f9a3056960c8411dd857153c194b1f91313"

    print(f"Model path: {model_path}")
    if not os.path.exists(model_path):
        print(f"Skipping Inference Engine test: Model not found at {model_path}")
        return

    try:
        engine = NativeInferenceEngine(model_path, context_length=2048)
        print("Engine initialized successfully.")

        prompt = "Hello, how are you?"
        tokens = engine.tokenize(prompt)

        start_time = time.time()
        output = engine.generate(tokens, max_new_tokens=20)
        end_time = time.time()

        print(f"Generated text: {engine.detokenize(output[len(tokens):])}")
        print(f"Time taken: {end_time - start_time:.2f}s")
    except Exception as e:
        print(f"Inference Engine test failed: {e}")


def test_coconut_bridge():
    print("\n--- Testing Coconut Bridge ---")
    try:
        from core.native.coconut_bridge import CoconutNativeBridge

        bridge = CoconutNativeBridge()
        print("Coconut Native Bridge initialized.")

        # Benchmarking
        dim = 4096
        num_paths = 4
        batch_size = 1
        hidden_state = np.random.randn(batch_size, dim).astype(np.float32)

        start_time = time.time()
        for _ in range(100):
            bridge.expand_paths(hidden_state, num_paths)
            # w1, b1, w2, b2 etc. need to be provided for evolve_paths
            # But we just want to see if expansion works for now
        end_time = time.time()
        print(f"100 expand_paths calls: {end_time - start_time:.4f}s")

    except Exception as e:
        print(f"Coconut Bridge test failed: {e}")


if __name__ == "__main__":
    test_inference_engine()
    test_coconut_bridge()
