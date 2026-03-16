"""Telemetry contracts for shared experiment lanes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


CPU_FIRST_REQUIRED_METRICS = [
    "wall_time_seconds",
    "command_count",
    "success_count",
    "failure_count",
    "correctness_pass",
    "determinism_pass",
]

CPU_FIRST_OPTIONAL_METRICS = [
    "p50_latency",
    "p95_latency",
    "throughput",
    "peak_memory",
    "cpu_utilization",
    "context_switches",
    "page_faults",
    "cache_misses",
    "branch_mispredicts",
    "instructions_retired",
    "ipc",
]


class TelemetryContractRegistry:
    """Builds and validates telemetry contracts across execution modes."""

    def build(
        self,
        caller_mode: str,
        *,
        extras: Iterable[str] | None = None,
        minimum_success_count: int = 1,
    ) -> Dict[str, Any]:
        required = list(CPU_FIRST_REQUIRED_METRICS)
        optional = list(CPU_FIRST_OPTIONAL_METRICS)
        extra_metrics = [str(metric) for metric in extras or [] if str(metric).strip()]
        for metric in extra_metrics:
            if metric not in required and metric not in optional:
                optional.append(metric)
        return {
            "schema_version": "lane.telemetry.v1",
            "caller_mode": caller_mode,
            "required_metrics": required,
            "optional_metrics": optional,
            "minimum_success_count": max(0, int(minimum_success_count)),
        }

    def evaluate(
        self,
        summary_metrics: Dict[str, Any],
        contract: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        contract = dict(contract or {})
        aggregate = dict(summary_metrics.get("aggregate_metrics") or {})
        available = {
            **aggregate,
            "wall_time_seconds": summary_metrics.get("wall_time_seconds"),
            "command_count": summary_metrics.get("command_count"),
            "success_count": summary_metrics.get("success_count"),
            "failure_count": summary_metrics.get("failure_count"),
            "correctness_pass": 1.0
            if int(summary_metrics.get("failure_count") or 0) == 0
            else 0.0,
            "determinism_pass": float(
                aggregate.get(
                    "determinism_pass",
                    aggregate.get("determinism", aggregate.get("replayability", 1.0)),
                )
            ),
        }
        missing = [
            metric
            for metric in contract.get("required_metrics") or []
            if available.get(metric) is None
        ]
        minimum_success_count = int(contract.get("minimum_success_count", 1))
        success_ok = int(summary_metrics.get("success_count") or 0) >= minimum_success_count
        return {
            "schema_version": contract.get("schema_version", "lane.telemetry.v1"),
            "required_metrics": list(contract.get("required_metrics") or []),
            "optional_metrics": list(contract.get("optional_metrics") or []),
            "missing_metrics": missing,
            "success_count_ok": success_ok,
            "contract_satisfied": not missing and success_ok,
            "captured_metrics": available,
        }
