from __future__ import annotations

from saguaro.indexing.auto_scaler import index_rss_budget_mb, plan_index_concurrency


def test_explicit_rss_budget_is_respected() -> None:
    assert index_rss_budget_mb(640.0) == 640.0

    plan = plan_index_concurrency(
        candidate_files=4000,
        batch_count=200,
        batch_size=32,
        active_dim=4096,
        total_dim=8192,
        vocab_size=16384,
        cpu_threads=24,
        rss_budget_mb=640.0,
    )

    assert plan["rss_budget_mb"] == 640.0
    assert plan["workers"] == 1


def test_shared_projection_reduces_large_dim_budget_pressure() -> None:
    private_plan = plan_index_concurrency(
        candidate_files=20000,
        batch_count=4000,
        batch_size=8,
        active_dim=16384,
        total_dim=32768,
        vocab_size=16384,
        cpu_threads=16,
        projection_shared=False,
    )
    shared_plan = plan_index_concurrency(
        candidate_files=20000,
        batch_count=4000,
        batch_size=8,
        active_dim=16384,
        total_dim=32768,
        vocab_size=16384,
        cpu_threads=16,
        projection_shared=True,
    )

    assert private_plan["projection_shared"] is False
    assert shared_plan["projection_shared"] is True
    assert float(shared_plan["worker_rss_mb"]) < float(private_plan["worker_rss_mb"])
    assert float(shared_plan["predicted_rss_mb"]) < float(private_plan["predicted_rss_mb"])
    assert int(shared_plan["workers"]) >= int(private_plan["workers"])
