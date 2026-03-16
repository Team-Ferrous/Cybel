from __future__ import annotations

from saguaro.indexing import native_coordinator
from saguaro.indexing.auto_scaler import plan_index_concurrency


def test_budgeted_plan_uses_all_cpu_threads_when_budget_allows() -> None:
    plan = plan_index_concurrency(
        candidate_files=3000,
        batch_count=128,
        batch_size=8,
        active_dim=1024,
        total_dim=2048,
        vocab_size=4096,
        cpu_threads=12,
        rss_budget_mb=16384.0,
        threads_per_worker_hint=1,
    )

    assert plan["workers"] == 12
    assert plan["threads_per_worker"] == 1
    assert plan["total_threads"] == 12


def test_parallel_indexing_defaults_to_process_pool_mode(monkeypatch) -> None:
    monkeypatch.delenv("SAGUARO_INDEX_SINGLE_RUNTIME", raising=False)
    assert native_coordinator._single_runtime_mode() is False


def test_parallel_indexing_allows_explicit_single_runtime(monkeypatch) -> None:
    monkeypatch.setenv("SAGUARO_INDEX_SINGLE_RUNTIME", "1")
    assert native_coordinator._single_runtime_mode() is True
