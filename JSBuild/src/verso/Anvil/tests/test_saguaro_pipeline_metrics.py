from __future__ import annotations

from pathlib import Path

from saguaro.api import SaguaroAPI


def test_index_reports_stage_and_budget_metrics(tmp_path: Path) -> None:
    src = tmp_path / "pkg" / "metrics_demo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "def metrics_demo(x: int) -> int:\n    return x + 1\n",
        encoding="utf-8",
    )

    result = SaguaroAPI(str(tmp_path)).index(
        path=".",
        force=False,
        incremental=True,
        prune_deleted=True,
    )
    execution = dict(result.get("execution") or {})

    assert result["status"] == "ok"
    assert execution.get("projection_bytes", 0) > 0
    assert "predicted_rss_mb" in execution
    assert "observed_rss_mb" in execution
    assert "worker_emitted_bytes" in execution
    assert "queue_wait_seconds" in execution
    assert "memory" in execution
