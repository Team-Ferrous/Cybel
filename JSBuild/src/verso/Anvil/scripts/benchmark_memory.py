import time
import os
import psutil
from core.ollama_client import DeterministicOllama
from config.settings import MASTER_MODEL


def get_vram_usage():
    try:
        # Try vulkaninfo or similar if possible, but psutil/system tools are more reliable for general RAM
        # For GPU OOM debugging, we'll monitor system RAM as a proxy for double-loading
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # MB
    except Exception:
        return 0


def run_benchmark():
    print(f"--- Starting Memory Benchmark for {MASTER_MODEL} ---")

    start_mem = get_vram_usage()
    print(f"Initial Process RAM: {start_mem:.2f} MB")

    start_time = time.time()
    brain = DeterministicOllama(MASTER_MODEL)
    load_time = time.time() - start_time

    after_load_mem = get_vram_usage()
    print(
        f"RAM after model init: {after_load_mem:.2f} MB (+{after_load_mem - start_mem:.2f} MB)"
    )
    print(f"Model Load Time: {load_time:.2f} seconds")

    # Trigger lazy loading of embeddings
    print("\nTriggering lazy-loading of QSG components...")
    if hasattr(brain.loader, "vocab_embeddings"):
        _ = brain.loader.vocab_embeddings
        after_lazy_mem = get_vram_usage()
        print(
            f"RAM after lazy-loading embeddings: {after_lazy_mem:.2f} MB (+{after_lazy_mem - after_load_mem:.2f} MB)"
        )

    print(
        "\nBenchmark Complete. Native Engine OOM issues should be resolved by Flash Attention and lazy loading."
    )


if __name__ == "__main__":
    run_benchmark()
