from __future__ import annotations

from pathlib import Path

import saguaro.analysis.liveness as liveness_module
import saguaro.api as api_module
from saguaro.analysis.liveness import LivenessAnalyzer
from saguaro.api import SaguaroAPI
from saguaro.indexing.backends import NumPyBackend


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_duplicates_reports_structural_mirrors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write(
        tmp_path / "core" / "native" / "mirror.cpp",
        "int helper_value() {\n"
        "    int local = 7;\n"
        "    return local;\n"
        "}\n",
    )
    _write(
        tmp_path / "saguaro" / "native" / "mirror.cpp",
        "int renamed_value() {\n"
        "    int local = 9;\n"
        "    return local;\n"
        "}\n",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    report = api.duplicates()

    assert report["status"] == "ok"
    assert report["count"] >= 1


def test_liveness_marks_unreachable_symbol(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write(
        tmp_path / "main.py",
        "from app.main import run\n\nif __name__ == '__main__':\n    run()\n",
    )
    _write(tmp_path / "app" / "main.py", "def run():\n    return 1\n")
    _write(tmp_path / "app" / "unused.py", "def dead_symbol():\n    return 2\n")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    api.graph_build(path=".", incremental=True)
    report = api.liveness(threshold=0.0)

    target = next(
        item
        for item in report["candidates"]
        if item["name"] == "dead_symbol"
    )
    assert target["classification"] in {"dead_confident", "dead_probable"}


def test_deadcode_rebuilds_graph_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write(
        tmp_path / "main.py",
        "from app.main import run\n\nif __name__ == '__main__':\n    run()\n",
    )
    _write(tmp_path / "app" / "main.py", "def run():\n    return 1\n")
    _write(tmp_path / "app" / "unused.py", "def dead_symbol():\n    return 2\n")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()

    report = api.deadcode(threshold=0.0)

    assert report["status"] == "ok"
    assert report["graph_path"]
    assert any(item["symbol"] == "dead_symbol" for item in report["candidates"])


def test_liveness_reports_low_usage_with_reference_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        liveness_module.DuplicateAnalyzer,
        "file_cluster_map",
        lambda self, path=".": {},
    )
    monkeypatch.setattr(
        liveness_module.EntryPointDetector,
        "detect",
        lambda self: [{"file": "main.py"}],
    )
    monkeypatch.setattr(
        liveness_module.DeadCodeAnalyzer,
        "analyze",
        lambda self: [],
    )
    monkeypatch.setattr(
        liveness_module.UnwiredAnalyzer,
        "analyze",
        lambda self, **kwargs: {"clusters": [], "summary": {"cluster_count": 0}},
    )

    analyzer = LivenessAnalyzer(str(tmp_path))
    graph = {
        "nodes": {
            "n_main": {
                "id": "n_main",
                "name": "main",
                "qualified_name": "main",
                "type": "function",
                "file": "main.py",
                "line": 1,
            },
            "n_run": {
                "id": "n_run",
                "name": "run",
                "qualified_name": "app.run",
                "type": "function",
                "file": "app.py",
                "line": 3,
            },
            "n_heavy": {
                "id": "n_heavy",
                "name": "heavy",
                "qualified_name": "app.heavy",
                "type": "function",
                "file": "app.py",
                "line": 9,
            },
        },
        "edges": {
            "e1": {"from": "n_main", "to": "n_run", "relation": "calls"},
            "e2": {"from": "n_run", "to": "n_heavy", "relation": "calls"},
            "e3": {"from": "n_main", "to": "n_heavy", "relation": "calls"},
        },
        "files": {
            "main.py": {"nodes": ["n_main"]},
            "app.py": {"nodes": ["n_run", "n_heavy"]},
        },
    }
    monkeypatch.setattr(
        analyzer,
        "_load_graph",
        lambda: {"graph_path": "synthetic", "graph": graph},
    )

    report = analyzer.analyze(threshold=0.0, max_low_usage_refs=1)
    low_usage = report["low_usage"]

    assert low_usage["max_refs"] == 1
    assert low_usage["count"] >= 1
    assert all(item["classification"] == "live" for item in low_usage["candidates"])
    assert all(item["evidence"]["usage_count"] <= 1 for item in low_usage["candidates"])

    run_item = next(item for item in low_usage["candidates"] if item["name"] == "run")
    assert run_item["evidence"]["usage_count"] == 1
    assert run_item["evidence"]["referencing_files"] == ["main.py"]

    names = {item["name"] for item in low_usage["candidates"]}
    assert "heavy" not in names


def test_liveness_low_usage_surfaces_dry_candidates_and_path_filter(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        liveness_module.DuplicateAnalyzer,
        "file_cluster_map",
        lambda self, path=".": {
            "pkg/feature.py": [{"id": "dup::structural::feature"}],
        },
    )
    monkeypatch.setattr(
        liveness_module.EntryPointDetector,
        "detect",
        lambda self: [{"file": "main.py"}],
    )
    monkeypatch.setattr(
        liveness_module.DeadCodeAnalyzer,
        "analyze",
        lambda self: [],
    )
    monkeypatch.setattr(
        liveness_module.UnwiredAnalyzer,
        "analyze",
        lambda self, **kwargs: {"clusters": [], "summary": {"cluster_count": 0}},
    )

    analyzer = LivenessAnalyzer(str(tmp_path))
    graph = {
        "nodes": {
            "n_main": {
                "id": "n_main",
                "name": "main",
                "qualified_name": "main",
                "type": "function",
                "file": "main.py",
                "line": 1,
            },
            "n_feature": {
                "id": "n_feature",
                "name": "run_feature",
                "qualified_name": "pkg.run_feature",
                "type": "function",
                "file": "pkg/feature.py",
                "line": 3,
            },
            "n_local": {
                "id": "n_local",
                "name": "normalize",
                "qualified_name": "pkg.normalize",
                "type": "function",
                "file": "pkg/feature.py",
                "line": 9,
            },
            "n_other": {
                "id": "n_other",
                "name": "normalize",
                "qualified_name": "pkg.reuse.normalize",
                "type": "function",
                "file": "pkg/reuse.py",
                "line": 5,
            },
        },
        "edges": {
            "e1": {"from": "n_main", "to": "n_feature", "relation": "calls"},
            "e2": {"from": "n_feature", "to": "n_local", "relation": "calls"},
        },
        "files": {
            "main.py": {"nodes": ["n_main"]},
            "pkg/feature.py": {"nodes": ["n_feature", "n_local"]},
            "pkg/reuse.py": {"nodes": ["n_other"]},
        },
    }
    monkeypatch.setattr(
        analyzer,
        "_load_graph",
        lambda: {"graph_path": "synthetic", "graph": graph},
    )

    report = analyzer.analyze(
        threshold=0.0,
        max_low_usage_refs=1,
        path_prefix="pkg",
        limit=1,
    )
    low_usage = report["low_usage"]

    assert low_usage["path_filter"] == "pkg"
    assert low_usage["count"] >= 2
    assert low_usage["returned_count"] == 1
    assert low_usage["dry_count"] >= 1
    assert low_usage["dry_candidates"][0]["name"] == "normalize"
    assert low_usage["dry_candidates"][0]["reuse_candidate"] is True
    assert "same_file_only" in low_usage["dry_candidates"][0]["dry_signals"]
    assert low_usage["areas"][0]["path"] == "pkg/feature.py"
