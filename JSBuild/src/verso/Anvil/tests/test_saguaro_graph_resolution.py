from __future__ import annotations

import json
from pathlib import Path

from saguaro.analysis.dead_code import DeadCodeAnalyzer
from saguaro.analysis.impact import ImpactAnalyzer
from saguaro.analysis.unwired import UnwiredAnalyzer
from saguaro.health import HealthDashboard
from saguaro.parsing.runtime_symbols import RuntimeSymbolResolver


def _write_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-03-10T00:00:00Z",
        "files": {"pkg/demo.py": {"nodes": ["file::pkg/demo.py"]}},
        "nodes": {
            "file::pkg/demo.py": {
                "id": "file::pkg/demo.py",
                "type": "file",
                "file": "pkg/demo.py",
                "name": "demo.py",
            }
        },
        "edges": {},
        "stats": {
            "files": 1,
            "nodes": 1,
            "edges": 0,
            "graph_coverage_percent": 100.0,
            "ffi_patterns": 0,
            "bridge_edges": 0,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_graph_resolution_prefers_canonical_code_graph_path(tmp_path: Path) -> None:
    graph_path = tmp_path / ".saguaro" / "graph" / "code_graph.json"
    _write_graph(graph_path)

    dead = DeadCodeAnalyzer(str(tmp_path))
    impact = ImpactAnalyzer(str(tmp_path))
    unwired = UnwiredAnalyzer(str(tmp_path))
    health = HealthDashboard(str(tmp_path / ".saguaro"), repo_path=str(tmp_path))

    assert dead._load_code_graph()["graph_path"] == str(graph_path)  # noqa: SLF001
    assert impact._load_code_graph()["graph_path"] == str(graph_path)  # noqa: SLF001
    assert (
        unwired._resolve_graph_payload(None)["graph_path"] == str(graph_path)  # noqa: SLF001
    )

    graph_report = health._graph_report()  # noqa: SLF001
    assert graph_report["status"] == "ready"
    assert graph_report["path"] == str(graph_path)
    assert graph_report["files"] == 1


def test_health_reports_graph_confidence_and_symbol_truth(tmp_path: Path) -> None:
    _write_graph(tmp_path / ".saguaro" / "graph" / "code_graph.json")
    (tmp_path / "saguaro").mkdir()
    (tmp_path / "Saguaro" / "saguaro").mkdir(parents=True)
    (tmp_path / "saguaro" / "demo.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (tmp_path / "Saguaro" / "saguaro" / "demo.py").write_text(
        "def run():\n    return 2\n",
        encoding="utf-8",
    )

    health = HealthDashboard(str(tmp_path / ".saguaro"), repo_path=str(tmp_path))
    report = health.generate_report()

    assert "graph_confidence" in report
    assert "ffi_bridge" in report["graph_confidence"]["deficits"]
    assert report["symbol_truth"]["shadowed_module_count"] >= 1
    assert Path(report["symbol_truth"]["artifact_path"]).exists()
    symbol_truth = json.loads(
        Path(report["symbol_truth"]["artifact_path"]).read_text(encoding="utf-8")
    )
    assert symbol_truth["modules"]["saguaro.demo"]["canonical_path"] == "Saguaro/saguaro/demo.py"


def test_health_derives_edge_class_counts_from_graph_payload(tmp_path: Path) -> None:
    graph_path = tmp_path / ".saguaro" / "graph" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-03-10T00:00:00Z",
        "files": {
            "pkg/a.py": {"nodes": ["pkg/a.py::a::function::1", "pkg/a.py::a.py::file::1"]},
            "pkg/b.py": {"nodes": ["pkg/b.py::b::function::1", "pkg/b.py::b.py::file::1"]},
        },
        "nodes": {
            "pkg/a.py::a::function::1": {
                "id": "pkg/a.py::a::function::1",
                "type": "function",
                "file": "pkg/a.py",
                "name": "a",
            },
            "pkg/b.py::b::function::1": {
                "id": "pkg/b.py::b::function::1",
                "type": "function",
                "file": "pkg/b.py",
                "name": "b",
            },
        },
        "edges": {
            "call": {
                "id": "call",
                "from": "pkg/a.py::a::function::1",
                "to": "pkg/b.py::b::function::1",
                "relation": "calls",
                "file": "pkg/a.py",
                "line": 1,
            }
        },
        "stats": {
            "files": 2,
            "nodes": 2,
            "edges": 1,
            "graph_coverage_percent": 100.0,
        },
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    health = HealthDashboard(str(tmp_path / ".saguaro"), repo_path=str(tmp_path))
    report = health.generate_report()

    assert report["graph_confidence"]["edge_classes"]["call"] == 1
    assert "call" not in report["graph_confidence"]["deficits"]


def test_runtime_symbol_manifest_and_health_coverage(tmp_path: Path) -> None:
    native_dir = tmp_path / "core" / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    (native_dir / "demo.cpp").write_text(
        'extern "C" int anvil_demo_symbol() { return 1; }\n',
        encoding="utf-8",
    )
    (native_dir / "demo_wrapper.py").write_text(
        "def bind(lib):\n"
        '    fn = getattr(lib, "anvil_demo_symbol", None)\n'
        "    return fn\n",
        encoding="utf-8",
    )

    manifest = RuntimeSymbolResolver(tmp_path).build_symbol_manifest(persist=True)

    assert manifest["referenced_symbol_count"] == 1
    assert manifest["matched_symbol_count"] == 1
    assert manifest["coverage_percent"] == 100.0

    health = HealthDashboard(str(tmp_path / ".saguaro"), repo_path=str(tmp_path))
    report = health.generate_report()

    assert report["runtime_symbols"]["matched_symbol_count"] == 1
    assert report["coverage_vector"]["runtime_symbol_coverage_pct"] == 100.0


def test_runtime_symbol_resolver_returns_qualified_ids(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    demo = pkg / "demo.py"
    demo.write_text(
        "def crash_here():\n    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )

    resolver = RuntimeSymbolResolver(tmp_path)
    matches = resolver.resolve_output(
        f'File "{demo}", line 1, in crash_here\n'
    )

    assert len(matches) == 1
    assert matches[0]["corpus_id"] == "primary"
    assert matches[0]["qualified_symbol_id"] == "primary:pkg/demo.py:runtime_symbol:crash_here"
