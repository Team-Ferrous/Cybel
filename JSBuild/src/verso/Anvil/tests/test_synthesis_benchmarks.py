from __future__ import annotations

from audit.runner.benchmark_suite import build_synthesis_governance_summary
from benchmarks.synthesis_suite import SynthesisBenchmarkSuite


def test_synthesis_benchmark_suite_provides_ten_repo_shaped_tasks() -> None:
    summary = SynthesisBenchmarkSuite().run()

    assert summary["task_count"] == 10
    assert summary["proof_pass_rate"] == 1.0


def test_audit_runner_exposes_synthesis_governance_summary() -> None:
    summary = build_synthesis_governance_summary()

    assert summary["task_count"] == 10
    assert "median_synthesis_latency_ms" in summary

