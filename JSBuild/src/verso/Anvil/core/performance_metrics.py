"""
Performance Metrics System for Anvil

Phase 4.1: Comprehensive performance tracking for loops, subagents, and workflows.
Monitors token usage, throughput, latency, and efficiency metrics.
"""

import hashlib
import json
import os
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich import box

from shared_kernel.event_store import get_event_store


class MetricType(Enum):
    """Types of metrics tracked."""

    TOKEN_USAGE = "token_usage"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    TOOL_CALLS = "tool_calls"
    SUCCESS_RATE = "success_rate"
    CONTEXT_EFFICIENCY = "context_efficiency"


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""

    timestamp: float
    component_name: str  # Loop, agent, or workflow name
    component_type: str  # "loop", "agent", "workflow"
    span_id: str = ""
    signal_ids: List[str] = field(default_factory=list)
    run_id: Optional[str] = None

    # Token metrics
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    thinking_tokens: int = 0  # COCONUT usage
    gpu_memory_mb: float = 0.0

    # Timing metrics
    latency_ms: float = 0.0
    ttft_ms: float = 0.0  # Time to first token

    # Throughput metrics
    tokens_per_second: float = 0.0

    # Context metrics
    context_window_used: int = 0
    context_window_total: int = 400000
    context_efficiency: float = 0.0  # % of context window utilized

    # Tool metrics
    tool_calls: Dict[str, int] = field(default_factory=dict)
    total_tool_calls: int = 0

    # Quality metrics
    success: bool = True
    error_message: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics over multiple runs."""

    component_name: str
    component_type: str
    total_runs: int

    # Aggregated token stats
    avg_tokens_input: float = 0.0
    avg_tokens_output: float = 0.0
    avg_tokens_total: float = 0.0
    total_tokens: int = 0

    # Aggregated timing stats
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Aggregated throughput
    avg_throughput_tps: float = 0.0
    peak_throughput_tps: float = 0.0

    # Context efficiency
    avg_context_efficiency: float = 0.0

    # Tool usage
    tool_usage_breakdown: Dict[str, int] = field(default_factory=dict)
    avg_tool_calls: float = 0.0

    # Success metrics
    success_rate: float = 0.0
    failure_count: int = 0


class PerformanceMonitor:
    """
    Real-time performance monitoring for Anvil components.

    Phase 4.1: Tracks and analyzes performance metrics across:
    - Execution loops (simple chat, run_loop, enhanced loop)
    - Subagents (all 6 types)
    - Workflows (multi-agent orchestration)
    """

    def __init__(self, storage_path: str = ".anvil/metrics"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.event_store = get_event_store()

        # In-memory metrics store
        self.snapshots: List[PerformanceSnapshot] = []
        self.aggregated: Dict[str, AggregatedMetrics] = {}

        # Current tracking context
        self.current_component: Optional[str] = None
        self.current_start_time: Optional[float] = None
        self.current_tokens_input: int = 0

        self.console = Console()

    def start_tracking(
        self,
        component_name: str,
        component_type: str,
        *,
        run_id: Optional[str] = None,
    ):
        """
        Start tracking a new component execution.

        Args:
            component_name: Name of loop/agent/workflow
            component_type: Type ("loop", "agent", "workflow")
        """
        self.current_component = component_name
        self.current_component_type = component_type
        self.current_start_time = time.time()
        self.current_run_id = run_id
        self.current_tokens_input = 0
        self.current_thinking_tokens = 0
        self.current_tool_calls = defaultdict(int)
        self.current_span_id = hashlib.sha1(
            f"{component_type}:{component_name}:{self.current_start_time}".encode("utf-8")
        ).hexdigest()[:12]
        self.current_signal_ids = [f"sig_{self.current_span_id}_start"]
        self.event_store.emit(
            "performance_span_started",
            {
                "component_name": component_name,
                "component_type": component_type,
                "span_id": self.current_span_id,
            },
            source="performance_monitor",
            metadata={"run_id": run_id, "signal_ids": list(self.current_signal_ids)},
            run_id=run_id,
        )

    def record_tokens(
        self, tokens_input: int = 0, tokens_output: int = 0, thinking_tokens: int = 0
    ):
        """Record token usage during execution."""
        if tokens_input > 0:
            self.current_tokens_input += tokens_input

        if hasattr(self, "current_tokens_output"):
            self.current_tokens_output += tokens_output
        else:
            self.current_tokens_output = tokens_output

        if hasattr(self, "current_thinking_tokens"):
            self.current_thinking_tokens += thinking_tokens
        else:
            self.current_thinking_tokens = thinking_tokens

    def record_tool_call(self, tool_name: str):
        """Record a tool invocation."""
        if not hasattr(self, "current_tool_calls"):
            self.current_tool_calls = defaultdict(int)
        self.current_tool_calls[tool_name] += 1
        if hasattr(self, "current_signal_ids") and getattr(self, "current_span_id", ""):
            self.current_signal_ids.append(
                f"sig_{self.current_span_id}_{tool_name}_{self.current_tool_calls[tool_name]}"
            )

    def end_tracking(
        self,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> PerformanceSnapshot:
        """
        End tracking and create performance snapshot.

        Args:
            success: Whether execution succeeded
            error_message: Error message if failed
            metadata: Additional metadata

        Returns:
            PerformanceSnapshot with all metrics
        """
        if not self.current_component or not self.current_start_time:
            raise ValueError("No active tracking session")

        end_time = time.time()
        latency_sec = end_time - self.current_start_time
        latency_ms = latency_sec * 1000

        # Calculate metrics
        tokens_input = getattr(self, "current_tokens_input", 0)
        tokens_output = getattr(self, "current_tokens_output", 0)
        thinking_tokens = getattr(self, "current_thinking_tokens", 0)
        tokens_total = tokens_input + tokens_output

        throughput = tokens_output / latency_sec if latency_sec > 0 else 0

        tool_calls = dict(getattr(self, "current_tool_calls", {}))
        total_tool_calls = sum(tool_calls.values())

        context_efficiency = (tokens_total / 400000) * 100 if tokens_total > 0 else 0

        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=end_time,
            component_name=self.current_component,
            component_type=self.current_component_type,
            span_id=getattr(self, "current_span_id", ""),
            signal_ids=list(getattr(self, "current_signal_ids", [])),
            run_id=getattr(self, "current_run_id", None),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_total,
            thinking_tokens=thinking_tokens,
            latency_ms=latency_ms,
            tokens_per_second=throughput,
            context_window_used=tokens_total,
            context_efficiency=context_efficiency,
            tool_calls=tool_calls,
            total_tool_calls=total_tool_calls,
            gpu_memory_mb=self._get_gpu_memory(),
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )

        # Store snapshot
        self.snapshots.append(snapshot)
        self.event_store.emit(
            "performance_span_completed",
            {
                "component_name": snapshot.component_name,
                "component_type": snapshot.component_type,
                "span_id": snapshot.span_id,
                "latency_ms": snapshot.latency_ms,
                "tokens_total": snapshot.tokens_total,
                "tool_calls": snapshot.tool_calls,
            },
            source="performance_monitor",
            metadata={
                "run_id": snapshot.run_id,
                "signal_ids": list(snapshot.signal_ids),
                "success": snapshot.success,
            },
            run_id=snapshot.run_id,
        )

        # Update aggregated metrics
        self._update_aggregated_metrics(snapshot)

        # Reset tracking state
        self.current_component = None
        self.current_start_time = None
        self.current_tokens_input = 0
        self.current_tokens_output = 0
        self.current_thinking_tokens = 0
        self.current_tool_calls = defaultdict(int)
        self.current_span_id = ""
        self.current_signal_ids = []
        self.current_run_id = None

        return snapshot

    def _get_gpu_memory(self) -> float:
        """CPU-only mode: GPU memory telemetry disabled."""
        return 0.0

    def _update_aggregated_metrics(self, snapshot: PerformanceSnapshot):
        """Update aggregated metrics with new snapshot."""
        key = f"{snapshot.component_type}:{snapshot.component_name}"

        if key not in self.aggregated:
            # Initialize new aggregated metrics
            self.aggregated[key] = AggregatedMetrics(
                component_name=snapshot.component_name,
                component_type=snapshot.component_type,
                total_runs=0,
            )

        agg = self.aggregated[key]
        agg.total_runs += 1

        # Update running averages
        n = agg.total_runs

        agg.avg_tokens_input = (
            (agg.avg_tokens_input * (n - 1)) + snapshot.tokens_input
        ) / n
        agg.avg_tokens_output = (
            (agg.avg_tokens_output * (n - 1)) + snapshot.tokens_output
        ) / n
        agg.avg_tokens_total = (
            (agg.avg_tokens_total * (n - 1)) + snapshot.tokens_total
        ) / n
        agg.total_tokens += snapshot.tokens_total

        agg.avg_latency_ms = ((agg.avg_latency_ms * (n - 1)) + snapshot.latency_ms) / n

        # Update min/max
        if n == 1:
            agg.min_latency_ms = snapshot.latency_ms
            agg.max_latency_ms = snapshot.latency_ms
        else:
            agg.min_latency_ms = min(agg.min_latency_ms, snapshot.latency_ms)
            agg.max_latency_ms = max(agg.max_latency_ms, snapshot.latency_ms)

        agg.avg_throughput_tps = (
            (agg.avg_throughput_tps * (n - 1)) + snapshot.tokens_per_second
        ) / n
        agg.peak_throughput_tps = max(
            agg.peak_throughput_tps, snapshot.tokens_per_second
        )

        agg.avg_context_efficiency = (
            (agg.avg_context_efficiency * (n - 1)) + snapshot.context_efficiency
        ) / n

        # Tool usage
        for tool, count in snapshot.tool_calls.items():
            agg.tool_usage_breakdown[tool] = (
                agg.tool_usage_breakdown.get(tool, 0) + count
            )

        agg.avg_tool_calls = (
            (agg.avg_tool_calls * (n - 1)) + snapshot.total_tool_calls
        ) / n

        # Success rate
        if not snapshot.success:
            agg.failure_count += 1
        agg.success_rate = ((n - agg.failure_count) / n) * 100

    def get_metrics(self, component_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.

        Args:
            component_name: If provided, get metrics for specific component

        Returns:
            Dictionary of metrics
        """
        if component_name:
            # Find aggregated metrics for component
            for key, agg in self.aggregated.items():
                if component_name in key:
                    return asdict(agg)
            return {}
        else:
            # Return all aggregated metrics
            return {key: asdict(agg) for key, agg in self.aggregated.items()}

    def print_metrics(self, component_name: Optional[str] = None):
        """
        Print formatted metrics table.

        Args:
            component_name: If provided, show metrics for specific component
        """
        metrics = self.get_metrics(component_name)

        if not metrics:
            self.console.print("[yellow]No metrics available[/yellow]")
            return

        self.console.print("\n[bold magenta]📊 Performance Metrics[/bold magenta]\n")

        # Create summary table
        table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        table.add_column("Component", style="green")
        table.add_column("Runs", style="cyan", justify="right")
        table.add_column("Avg Tokens", style="yellow", justify="right")
        table.add_column("Avg Latency", style="magenta", justify="right")
        table.add_column("Throughput", style="blue", justify="right")
        table.add_column("Success Rate", style="green", justify="right")

        if component_name:
            # Single component
            agg = AggregatedMetrics(**metrics)
            table.add_row(
                agg.component_name,
                str(agg.total_runs),
                f"{int(agg.avg_tokens_total):,}",
                f"{agg.avg_latency_ms:.0f}ms",
                f"{agg.avg_throughput_tps:.1f} t/s",
                f"{agg.success_rate:.1f}%",
            )
        else:
            # Multiple components
            for key, agg_dict in metrics.items():
                agg = AggregatedMetrics(**agg_dict)
                table.add_row(
                    agg.component_name,
                    str(agg.total_runs),
                    f"{int(agg.avg_tokens_total):,}",
                    f"{agg.avg_latency_ms:.0f}ms",
                    f"{agg.avg_throughput_tps:.1f} t/s",
                    f"{agg.success_rate:.1f}%",
                )

        self.console.print(table)

        # Print detailed metrics if single component
        if component_name and metrics:
            agg = AggregatedMetrics(**metrics)

            self.console.print(
                f"\n[bold]Detailed Metrics: {agg.component_name}[/bold]\n"
            )

            details = Table(show_header=False, box=box.SIMPLE)
            details.add_column("Metric", style="cyan")
            details.add_column("Value", style="white")

            details.add_row("Total Runs", str(agg.total_runs))
            details.add_row("Total Tokens", f"{agg.total_tokens:,}")
            details.add_row("Avg Input Tokens", f"{int(agg.avg_tokens_input):,}")
            details.add_row("Avg Output Tokens", f"{int(agg.avg_tokens_output):,}")
            details.add_row("Min Latency", f"{agg.min_latency_ms:.0f}ms")
            details.add_row("Max Latency", f"{agg.max_latency_ms:.0f}ms")
            details.add_row("Avg Latency", f"{agg.avg_latency_ms:.0f}ms")
            details.add_row("Peak Throughput", f"{agg.peak_throughput_tps:.1f} t/s")
            details.add_row(
                "Avg Context Efficiency", f"{agg.avg_context_efficiency:.1f}%"
            )
            details.add_row("Avg Tool Calls", f"{agg.avg_tool_calls:.1f}")
            details.add_row("Success Rate", f"{agg.success_rate:.1f}%")

            self.console.print(details)

            # Tool usage breakdown
            if agg.tool_usage_breakdown:
                self.console.print("\n[bold]Tool Usage Breakdown:[/bold]\n")

                tool_table = Table(
                    show_header=True, header_style="bold cyan", box=box.SIMPLE
                )
                tool_table.add_column("Tool", style="green")
                tool_table.add_column("Count", style="yellow", justify="right")
                tool_table.add_column("%", style="dim", justify="right")

                total_calls = sum(agg.tool_usage_breakdown.values())

                for tool, count in sorted(
                    agg.tool_usage_breakdown.items(), key=lambda x: x[1], reverse=True
                ):
                    percentage = (count / total_calls) * 100 if total_calls > 0 else 0
                    tool_table.add_row(tool, str(count), f"{percentage:.1f}%")

                self.console.print(tool_table)

    def export_metrics(self, output_path: Optional[str] = None) -> str:
        """
        Export all metrics to JSON file.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = int(time.time())
            output_path = os.path.join(self.storage_path, f"metrics_{timestamp}.json")

        export_data = {
            "exported_at": time.time(),
            "total_snapshots": len(self.snapshots),
            "aggregated_metrics": {
                key: asdict(agg) for key, agg in self.aggregated.items()
            },
            "recent_snapshots": [
                asdict(snapshot)
                for snapshot in self.snapshots[-100:]  # Last 100 snapshots
            ],
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        self.console.print(f"[green]✔ Metrics exported to {output_path}[/green]")
        return output_path

    def clear_metrics(self):
        """Clear all stored metrics."""
        self.snapshots = []
        self.aggregated = {}
        self.console.print("[yellow]Metrics cleared[/yellow]")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance."""
    global _global_monitor

    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()

    return _global_monitor
