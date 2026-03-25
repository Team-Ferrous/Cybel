import time
import numpy as np
import sys
import os

# Add root to pythonpath
sys.path.append(os.getcwd())

from core.qsg.config import QSGConfig
from core.qsg.generator import QSGGenerator


def benchmark_qsg():
    """Evaluate QSG performance."""
    print("Initializing QSG Benchmark...")

    vocab_dim = 128
    vocab_size = 1000
    vocab_emb = np.random.randn(vocab_size, vocab_dim).astype(np.float32)

    config = QSGConfig(
        jacobi_iterations=3,
        grover_iterations=2,
        hopfield_beta=1.0,
        speculative_drafts=4,
        use_coconut_reasoning=False,  # Baseline
    )

    generator = QSGGenerator(config, vocab_emb)
    context = np.random.randn(1, vocab_dim).astype(np.float32)

    num_runs = 50
    seq_len = 8

    start_time = time.time()
    for _ in range(num_runs):
        tokens, probs = generator.generate_draft(context, seq_len=seq_len)

    end_time = time.time()
    total_time = end_time - start_time
    total_tokens = num_runs * seq_len

    tps = total_tokens / total_time
    print(f"QSG Performance: {tps:.2f} tokens/sec")
    print(f"Total time for {total_tokens} tokens: {total_time:.4f}s")


if __name__ == "__main__":
    benchmark_qsg()
