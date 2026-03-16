"""CPU-first scorecard used by the shared experiment lane."""

from __future__ import annotations

from typing import Any, Dict


class CPUScorecard:
    """Evaluates experiment telemetry against a baseline."""

    LOWER_IS_BETTER = {
        "wall_time_seconds",
        "p50_latency",
        "p95_latency",
        "peak_memory",
        "context_switches",
        "page_faults",
        "cache_misses",
        "branch_mispredicts",
    }
    HIGHER_IS_BETTER = {
        "throughput",
        "cpu_utilization",
        "correctness_pass",
        "determinism_pass",
        "replayability",
        "telemetry_contract_satisfied",
    }

    def score(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
        *,
        complexity_penalty: float = 0.0,
        instability_penalty: float = 0.0,
    ) -> Dict[str, Any]:
        baseline_metrics = dict(baseline or {})
        current_metrics = dict(current or {})
        metric_deltas: Dict[str, float] = {}
        objective_gain = 0.0
        regression_penalty = 0.0

        metric_names = set(baseline_metrics) | set(current_metrics)
        for metric in metric_names:
            current_value = self._coerce(current_metrics.get(metric))
            baseline_value = self._coerce(baseline_metrics.get(metric))
            if current_value is None:
                continue
            if baseline_value is None:
                metric_deltas[metric] = round(current_value, 6)
                continue
            delta = current_value - baseline_value
            metric_deltas[metric] = round(delta, 6)
            if metric in self.LOWER_IS_BETTER:
                objective_gain += max(0.0, -delta)
                regression_penalty += max(0.0, delta)
            elif metric in self.HIGHER_IS_BETTER:
                objective_gain += max(0.0, delta)
                regression_penalty += max(0.0, -delta)

        determinism_penalty = 0.0
        determinism_value = self._coerce(current_metrics.get("determinism_pass"))
        if determinism_value is not None and determinism_value < 1.0:
            determinism_penalty = 1.0 - determinism_value

        score = (
            objective_gain
            - regression_penalty
            - float(complexity_penalty)
            - float(instability_penalty)
            - determinism_penalty
        )
        return {
            "objective_gain": round(objective_gain, 6),
            "regression_penalty": round(regression_penalty, 6),
            "complexity_penalty": round(float(complexity_penalty), 6),
            "instability_penalty": round(float(instability_penalty), 6),
            "determinism_penalty": round(determinism_penalty, 6),
            "metric_deltas": metric_deltas,
            "score": round(score, 6),
        }

    @staticmethod
    def _coerce(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
