"""
Performance Monitoring for Anvil
Tracks inference metrics to measure optimization impact.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class InferenceMetrics:
    """Metrics for a single inference request."""

    tokens_generated: int
    time_elapsed: float
    context_size: int
    tier: str = "unknown"
    throughput: float = field(init=False)
    gpu_memory_mb: Optional[float] = None
    coconut_paths: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.throughput = (
            self.tokens_generated / self.time_elapsed if self.time_elapsed > 0 else 0
        )

    def to_dict(self) -> Dict:
        return {
            "tokens": self.tokens_generated,
            "time": self.time_elapsed,
            "context_size": self.context_size,
            "tier": self.tier,
            "model": self.model,
            "throughput": self.throughput,
            "gpu_memory": self.gpu_memory_mb,
            "coconut_paths": self.coconut_paths,
            "timestamp": self.timestamp,
        }


class PerformanceMonitor:
    """
    Tracks and reports performance metrics for inference operations.

    Usage:
        monitor = PerformanceMonitor()

        # During generation
        start = time.time()
        tokens = generate(...)
        elapsed = time.time() - start

        monitor.log(len(tokens), elapsed, ctx_size=32768, tier="medium")

        # Get stats
        print(monitor.get_stats())
        print(monitor.get_report())
    """

    def __init__(self, persist_path: Optional[str] = ".anvil/perf_metrics.json"):
        self.metrics: List[InferenceMetrics] = []
        self.persist_path = persist_path

        # Load existing metrics if available
        if persist_path and Path(persist_path).exists():
            self._load_metrics()

    def log(
        self,
        tokens: int,
        elapsed: float,
        ctx_size: int,
        tier: str = "unknown",
        model: str = "unknown",
        gpu_memory: Optional[float] = None,
        coconut_paths: Optional[int] = None,
    ):
        """
        Log a new inference metric.

        Args:
            tokens: Number of tokens generated
            elapsed: Time elapsed in seconds
            ctx_size: Context window size used
            tier: Complexity tier (simple/medium/complex/extreme)
            model: Model name
        """
        metric = InferenceMetrics(
            tokens_generated=tokens,
            time_elapsed=elapsed,
            context_size=ctx_size,
            tier=tier,
            model=model,
            gpu_memory_mb=gpu_memory,
            coconut_paths=coconut_paths,
        )
        self.metrics.append(metric)

        # Persist to disk
        if self.persist_path:
            self._save_metrics()

    def get_stats(self) -> Dict[str, float]:
        """
        Get aggregate statistics.

        Returns:
            stats: Dictionary with aggregate metrics
        """
        if not self.metrics:
            return {
                "total_requests": 0,
                "avg_throughput": 0.0,
                "total_tokens": 0,
                "total_time": 0.0,
            }

        return {
            "total_requests": len(self.metrics),
            "avg_throughput": sum(m.throughput for m in self.metrics)
            / len(self.metrics),
            "max_throughput": max(m.throughput for m in self.metrics),
            "min_throughput": min(m.throughput for m in self.metrics),
            "total_tokens": sum(m.tokens_generated for m in self.metrics),
            "total_time": sum(m.time_elapsed for m in self.metrics),
            "avg_tokens_per_request": sum(m.tokens_generated for m in self.metrics)
            / len(self.metrics),
            "avg_time_per_request": sum(m.time_elapsed for m in self.metrics)
            / len(self.metrics),
        }

    def get_stats_by_tier(self) -> Dict[str, Dict[str, float]]:
        """Get statistics broken down by complexity tier."""
        tier_metrics = {}

        for metric in self.metrics:
            if metric.tier not in tier_metrics:
                tier_metrics[metric.tier] = []
            tier_metrics[metric.tier].append(metric)

        tier_stats = {}
        for tier, metrics_list in tier_metrics.items():
            tier_stats[tier] = {
                "count": len(metrics_list),
                "avg_throughput": sum(m.throughput for m in metrics_list)
                / len(metrics_list),
                "total_tokens": sum(m.tokens_generated for m in metrics_list),
                "total_time": sum(m.time_elapsed for m in metrics_list),
            }

        return tier_stats

    def get_report(self) -> str:
        """Generate a human-readable performance report."""
        if not self.metrics:
            return "No performance metrics recorded."

        stats = self.get_stats()
        tier_stats = self.get_stats_by_tier()

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          GRANITE AGENT PERFORMANCE REPORT                    ║
╚══════════════════════════════════════════════════════════════╝

Overall Statistics:
  Total Requests:        {stats['total_requests']}
  Total Tokens:          {stats['total_tokens']:,}
  Total Time:            {stats['total_time']:.1f}s

  Avg Throughput:        {stats['avg_throughput']:.2f} tokens/sec
  Max Throughput:        {stats['max_throughput']:.2f} tokens/sec
  Min Throughput:        {stats['min_throughput']:.2f} tokens/sec

  Avg Tokens/Request:    {stats['avg_tokens_per_request']:.0f}
  Avg Time/Request:      {stats['avg_time_per_request']:.2f}s

Performance by Tier:
"""

        for tier, tier_stat in sorted(tier_stats.items()):
            report += f"""
  {tier.upper()}:
    Requests:            {tier_stat['count']}
    Avg Throughput:      {tier_stat['avg_throughput']:.2f} tokens/sec
    Total Tokens:        {tier_stat['total_tokens']:,}
    Total Time:          {tier_stat['total_time']:.1f}s
"""

        # Recent performance trend
        if len(self.metrics) >= 5:
            recent_5 = self.metrics[-5:]
            recent_throughput = sum(m.throughput for m in recent_5) / 5
            report += f"""
Recent Performance (last 5 requests):
  Avg Throughput:        {recent_throughput:.2f} tokens/sec
"""

        return report

    def get_speedup_estimate(self, baseline_tps: float = 3.5) -> str:
        """
        Estimate speedup compared to baseline.

        Args:
            baseline_tps: Baseline tokens per second (default: 3.5 for FP32)

        Returns:
            Speedup report
        """
        stats = self.get_stats()
        if stats["total_requests"] == 0:
            return "No metrics to compare."

        current_tps = stats["avg_throughput"]
        speedup = current_tps / baseline_tps

        return f"""
Speedup Analysis:
  Baseline:              {baseline_tps:.2f} tokens/sec
  Current:               {current_tps:.2f} tokens/sec
  Speedup:               {speedup:.2f}x faster
"""

    def track_gpu_usage(self) -> Dict[str, float]:
        """CPU-only mode: GPU telemetry is always zeroed."""
        return {"memory_mb": 0.0, "utilization": 0.0}

    def clear(self):
        """Clear all metrics."""
        self.metrics = []
        if self.persist_path and Path(self.persist_path).exists():
            Path(self.persist_path).unlink()

    def _save_metrics(self):
        """Persist metrics to disk."""
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump([m.to_dict() for m in self.metrics], f, indent=2)
        except Exception:
            # Silent fail - don't break agent if persistence fails
            pass

    def _load_metrics(self):
        """Load metrics from disk."""
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
                for item in data:
                    metric = InferenceMetrics(
                        tokens_generated=item["tokens"],
                        time_elapsed=item["time"],
                        context_size=item["context_size"],
                        tier=item.get("tier", "unknown"),
                        model=item.get("model", "unknown"),
                    )
                    metric.timestamp = item.get("timestamp", time.time())
                    metric.throughput = item["throughput"]
                    self.metrics.append(metric)
        except Exception:
            # Silent fail - start fresh if load fails
            pass


# Global performance monitor (singleton)
_global_monitor = None


def get_global_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor
