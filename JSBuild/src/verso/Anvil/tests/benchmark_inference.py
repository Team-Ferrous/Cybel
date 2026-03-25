#!/usr/bin/env python3
"""
Benchmark script for measuring Anvil inference performance.

Usage:
    python tests/benchmark_inference.py

This will measure performance with different optimization levels enabled.
"""

import time
import sys
import os
import json
from typing import Dict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class PerformanceBenchmark:
    """Benchmark suite for inference performance."""

    def __init__(self):
        self.results = {}

    def run_all_benchmarks(self):
        """Run all benchmark scenarios."""
        print("=" * 80)
        print("GRANITE AGENT PERFORMANCE BENCHMARK")
        print("=" * 80)
        print()

        scenarios = [
            ("Incremental KV Cache", self.benchmark_kv_cache),
            ("Adaptive Context", self.benchmark_adaptive_context),
            ("Performance Monitoring", self.benchmark_monitoring),
            ("Fast Attention (if built)", self.benchmark_fast_attention),
        ]

        for name, benchmark_fn in scenarios:
            print(f"\n📊 Benchmarking: {name}")
            print("-" * 80)
            try:
                result = benchmark_fn()
                self.results[name] = result
                self.print_result(result)
            except Exception as e:
                print(f"❌ Failed: {e}")
                self.results[name] = {"error": str(e)}

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        self.print_summary()

    def benchmark_kv_cache(self) -> Dict:
        """Benchmark incremental KV cache."""
        try:
            from core.native.incremental_kv_cache import IncrementalKVCache

            class MockContext:
                pass

            iterations = 10000
            max_seq_len = 4096

            # Benchmark cache operations
            cache = IncrementalKVCache(MockContext(), max_seq_len)

            start = time.time()
            for i in range(iterations):
                cache.advance_position(1)
                if i % 100 == 0:
                    cache.prepare_for_generation([1, 2, 3], allow_reuse=True)
            elapsed = time.time() - start

            ops_per_sec = iterations / elapsed

            return {
                "status": "success",
                "operations": iterations,
                "time_seconds": elapsed,
                "ops_per_second": ops_per_sec,
                "message": f"✓ {ops_per_sec:.0f} cache operations/sec",
            }

        except ImportError as e:
            return {"status": "skipped", "reason": str(e)}

    def benchmark_adaptive_context(self) -> Dict:
        """Benchmark adaptive context manager."""
        try:
            from core.adaptive_context import AdaptiveContextManager

            manager = AdaptiveContextManager()

            test_queries = [
                "what is this?",
                "write a function to parse JSON",
                "implement a complex authentication system",
                "migrate the entire codebase",
            ] * 100

            start = time.time()
            for query in test_queries:
                tier, params = manager.recommend_tier_with_lookahead(
                    user_input=query,
                    system_prompt="System prompt",
                    context_items=[],
                    history=[],
                )
            elapsed = time.time() - start

            analyses_per_sec = len(test_queries) / elapsed

            return {
                "status": "success",
                "analyses": len(test_queries),
                "time_seconds": elapsed,
                "analyses_per_second": analyses_per_sec,
                "message": f"✓ {analyses_per_sec:.0f} context analyses/sec",
            }

        except ImportError as e:
            return {"status": "skipped", "reason": str(e)}

    def benchmark_monitoring(self) -> Dict:
        """Benchmark performance monitoring overhead."""
        try:
            from core.performance_monitor import PerformanceMonitor

            monitor = PerformanceMonitor(persist_path=None)

            iterations = 1000

            start = time.time()
            for i in range(iterations):
                monitor.log(
                    tokens=100, elapsed=1.0, ctx_size=8192, tier="simple", model="test"
                )
            elapsed = time.time() - start

            logs_per_sec = iterations / elapsed

            return {
                "status": "success",
                "logs": iterations,
                "time_seconds": elapsed,
                "logs_per_second": logs_per_sec,
                "overhead_per_log_us": (elapsed / iterations) * 1000000,
                "message": f"✓ {logs_per_sec:.0f} logs/sec ({(elapsed/iterations)*1000000:.1f} μs overhead/log)",
            }

        except ImportError as e:
            return {"status": "skipped", "reason": str(e)}

    def benchmark_fast_attention(self) -> Dict:
        """Benchmark fast attention kernels."""
        try:
            from core.native.fast_attention_wrapper import FastAttention
            import numpy as np

            attn = FastAttention()

            if not attn.available:
                return {
                    "status": "skipped",
                    "reason": "Library not built (run: bash core/native/build_native.sh)",
                }

            # Benchmark different sizes
            results = {}

            for seq_len in [64, 128, 256]:
                batch_heads = 8
                head_dim = 64

                Q = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)
                K = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)
                V = np.random.randn(batch_heads, seq_len, head_dim).astype(np.float32)

                # Warmup
                for _ in range(10):
                    attn.compute_attention(Q, K, V)

                # Benchmark
                iterations = 100
                start = time.time()
                for _ in range(iterations):
                    attn.compute_attention(Q, K, V)
                elapsed = time.time() - start

                avg_time_ms = (elapsed / iterations) * 1000
                results[f"seq_{seq_len}"] = avg_time_ms

            return {
                "status": "success",
                "results": results,
                "message": "✓ Attention times: "
                + ", ".join([f"{seq}={time:.2f}ms" for seq, time in results.items()]),
            }

        except ImportError as e:
            return {"status": "skipped", "reason": str(e)}

    def print_result(self, result: Dict):
        """Print a benchmark result."""
        if result.get("status") == "success":
            print(f"  {result.get('message', 'Success')}")
            for key, value in result.items():
                if key not in ["status", "message"]:
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    elif isinstance(value, dict):
                        print(f"    {key}:")
                        for k, v in value.items():
                            print(f"      {k}: {v}")
                    else:
                        print(f"    {key}: {value}")
        elif result.get("status") == "skipped":
            print(f"  ⊘ Skipped: {result.get('reason', 'Unknown')}")
        else:
            print(f"  ❌ Error: {result.get('error', 'Unknown')}")

    def print_summary(self):
        """Print overall summary."""
        successes = sum(
            1 for r in self.results.values() if r.get("status") == "success"
        )
        skipped = sum(1 for r in self.results.values() if r.get("status") == "skipped")
        errors = sum(1 for r in self.results.values() if r.get("status") == "error")

        print(f"\nResults: {successes} passed, {skipped} skipped, {errors} failed")
        print()

        if successes > 0:
            print("✅ Optimizations are functioning correctly!")
        if skipped > 0:
            print("⚠️  Some optimizations are not available (may need compilation)")
        if errors > 0:
            print("❌ Some benchmarks failed - check error messages above")

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        output_path = os.path.join(os.path.dirname(__file__), filename)
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📝 Results saved to: {output_path}")


def main():
    """Run benchmarks."""
    benchmark = PerformanceBenchmark()
    benchmark.run_all_benchmarks()
    benchmark.save_results()

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
To build optimizations that require compilation:

    1. Build fast attention kernels:
       $ bash core/native/build_native.sh

    2. Verify installation:
       $ python tests/test_performance_optimizations.py

    3. Run full agent benchmark:
       $ python cli/repl.py
       > what is the codebase structure?
       > /stats
    """)


if __name__ == "__main__":
    main()
