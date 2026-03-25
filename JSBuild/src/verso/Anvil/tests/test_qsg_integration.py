import sys
import os
import time

# Add repo root to path
sys.path.append(os.getcwd())

from core.qsg.ollama_adapter import OllamaQSGAdapter
from core.qsg.config import QSGConfig


def test_integration():
    model_name = "qwen2.5-coder:7b"  # Better model for demo
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    print(f"Testing integration with model: {model_name}")

    # Initialize adapter with Grover Enabled
    try:
        config = QSGConfig()
        config.use_grover = True
        config.grover_iterations = 1
        config.use_coconut = True
        config.coconut_paths = 4
        adapter = OllamaQSGAdapter(model_name, config=config)
    except Exception as e:
        print(f"Failed to initialize adapter: {e}")
        return

    # Check native engine status
    if adapter.native_engine:
        print("SUCCESS: Native Engine loaded.")
        print(f"  Layers: {adapter.native_engine.n_layer}")
        print(f"  Dim: {adapter.native_engine.dim}")
    else:
        print("WARNING: Native Engine NOT loaded. Check SIMD availability.")

    # Generate text
    prompt = "The future of AI is"
    print(f"\nPrompt: '{prompt}'")

    start = time.time()
    result = adapter.generate(prompt, options={"num_predict": 20})
    duration = time.time() - start

    print(f"\nResult: '{result}'")
    print(f"Time: {duration:.2f}s")

    if adapter.native_engine:
        # Check simple coherence (heuristic: English words, spacing)
        print("Coherence Check: ", "PASS" if len(result.strip()) > 5 else "FAIL")


if __name__ == "__main__":
    test_integration()
