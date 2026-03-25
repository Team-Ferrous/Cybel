from __future__ import annotations

from pathlib import Path

import saguaro.api as api_module
from saguaro.api import SaguaroAPI


def test_health_reports_storage_budget(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api_module, "_SAGUARO_STORAGE_WARN_BYTES", 32)
    monkeypatch.setattr(api_module, "_SAGUARO_STORAGE_HARD_BYTES", 64)

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()

    graph_dir = tmp_path / ".saguaro" / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "graph.json").write_text(
        '{"nodes": {}, "edges": {}, "files": {}}', encoding="utf-8"
    )
    (tmp_path / ".saguaro" / "vectors" / "vectors.bin").write_bytes(b"x" * 80)

    health = api.health()
    budget = health["storage_budget"]

    assert budget["status"] == "fail"
    assert budget["index_disk_bytes"] >= 80
    assert budget["vector_bytes"] >= 80


def test_doctor_propagates_storage_budget(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(api_module, "_SAGUARO_STORAGE_WARN_BYTES", 16)
    monkeypatch.setattr(api_module, "_SAGUARO_STORAGE_HARD_BYTES", 32)

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    (tmp_path / ".saguaro" / "vectors" / "vectors.bin").write_bytes(b"x" * 40)

    report = api.doctor()

    assert report["storage_budget"]["status"] == "fail"
    assert report["status"] == "warning"
