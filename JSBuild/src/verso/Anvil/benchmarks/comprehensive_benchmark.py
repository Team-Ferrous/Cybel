import time
import numpy as np
from typing import Dict
from core.agent import BaseAgent
from rich.console import Console
from rich.panel import Panel


class AnvilBenchmark:
    """Comprehensive check performance benchmarking."""

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.console = Console()

    def run_all_benchmarks(self):
        """Run all benchmarks and generate report."""
        self.console.print(
            Panel("Running Comprehensive Benchmarks...", style="bold magenta")
        )

        results = {
            "simple_chat": self.benchmark_simple_chat(),
            "file_operations": self.benchmark_file_ops(),
            # "semantic_search": self.benchmark_semantic_search(),
        }

        self.generate_report(results)

    def benchmark_simple_chat(self) -> Dict:
        """Benchmark simple chat loop performance."""
        tasks = [
            "What files are in this directory?",
            "Read the README.md file (or create one if missing)",
            "Explain how the agent works",
            "Find all TODO comments in the codebase",
        ]

        latencies = []
        self.console.print("[cyan]Benchmarking Simple Chat...[/cyan]")

        for i, task in enumerate(tasks):
            start = time.time()
            # We use simple_chat directly
            # Note: This might make actual LLM calls unless mocked
            # For a real benchmark, we want actual calls to test end-to-end
            try:
                self.agent.simple_chat(task)
            except Exception as e:
                self.console.print(f"[red]Error in task {i}: {e}[/red]")

            latency = time.time() - start
            latencies.append(latency)
            self.console.print(f"  Task {i+1}: {latency:.2f}s")

        if not latencies:
            return {}

        return {
            "avg_latency": float(np.mean(latencies)),
            "p50_latency": float(np.percentile(latencies, 50)),
            "p95_latency": float(np.percentile(latencies, 95)),
            "max": float(np.max(latencies)),
        }

    def benchmark_file_ops(self) -> Dict:
        """Benchmark raw file operations."""
        from tools.file_ops import FileOps

        ops = FileOps(".")

        self.console.print("[cyan]Benchmarking File Ops...[/cyan]")

        # Write
        start = time.time()
        for i in range(10):
            ops.write_file(f"bench_temp_{i}.txt", f"Content {i}")
        write_time = (time.time() - start) / 10

        # Read
        start = time.time()
        for i in range(10):
            ops.read_file(f"bench_temp_{i}.txt")
        read_time = (time.time() - start) / 10

        # Cleanup
        for i in range(10):
            ops.delete_file(f"bench_temp_{i}.txt")

        self.console.print(f"  Avg Write: {write_time*1000:.2f}ms")
        self.console.print(f"  Avg Read: {read_time*1000:.2f}ms")

        return {"avg_write_ms": write_time * 1000, "avg_read_ms": read_time * 1000}

    def generate_report(self, results: Dict):
        """Print markdown report."""
        self.console.print("\n[bold green]BENCHMARK REPORT[/bold green]")

        if "simple_chat" in results and results["simple_chat"]:
            sc = results["simple_chat"]
            self.console.print("\n[bold]Simple Chat[/bold]")
            self.console.print(f"  Avg Latency: {sc['avg_latency']:.2f}s")
            self.console.print(f"  P95 Latency: {sc['p95_latency']:.2f}s")

        if "file_operations" in results:
            fo = results["file_operations"]
            self.console.print("\n[bold]File Operations[/bold]")
            self.console.print(f"  Read: {fo['avg_read_ms']:.2f}ms")
            self.console.print(f"  Write: {fo['avg_write_ms']:.2f}ms")


if __name__ == "__main__":
    # Test run
    from core.agent import BaseAgent

    agent = BaseAgent(name="Benchmarker")
    bench = AnvilBenchmark(agent)
    bench.run_all_benchmarks()
