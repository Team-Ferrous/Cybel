from __future__ import annotations

from pathlib import Path

from benchmarks import saguaro_index_audit


def test_index_audit_uses_cached_stats_and_reports_index_result(tmp_path: Path) -> None:
    payload = saguaro_index_audit._run_audit(
        tmp_path,
        force_refresh=False,
        warm_cache=True,
    )

    assert payload["stats"]["loc"] == 0
    assert payload["cached_stats_match"] is True
    assert "index_seconds" in payload["timings"]
    assert payload["index"]["status"] == "ok"
    assert "native_index_seconds" in payload["index"]["execution"]["timings"]
