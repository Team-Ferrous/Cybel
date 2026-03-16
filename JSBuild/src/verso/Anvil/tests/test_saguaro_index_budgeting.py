from __future__ import annotations

from saguaro.indexing.auto_scaler import plan_index_concurrency


def test_budgeted_plan_throttles_workers_under_low_rss_budget() -> None:
    plan = plan_index_concurrency(
        candidate_files=1200,
        batch_count=80,
        batch_size=32,
        active_dim=4096,
        total_dim=8192,
        vocab_size=16384,
        cpu_threads=32,
        rss_budget_mb=768.0,
    )

    assert plan["workers"] >= 1
    assert plan["workers"] <= 2
    assert plan["total_threads"] <= plan["cpu_threads"]
    assert plan["batch_capacity"] > 0
    assert plan["quota_mode"] == "budgeted"
