from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import saguaro.api as api_module
from saguaro.api import SaguaroAPI
from saguaro.indexing.backends import NumPyBackend
from saguaro.omnigraph.store import OmniGraphStore
from saguaro.services.platform import GraphService, ParseService


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_relation_repo(tmp_path: Path) -> None:
    _write(
        tmp_path / "core" / "qsg" / "continuous_engine.py",
        "class QSGInferenceEngine:\n"
        "    def evaluation_pipeline(self) -> int:\n"
        "        return 1\n",
    )
    _write(
        tmp_path / "Saguaro" / "saguaro" / "query" / "pipeline.py",
        "class QueryPipeline:\n"
        "    def evaluation_pipeline(self) -> int:\n"
        "        return 2\n",
    )
    _write(
        tmp_path / "core" / "runtime" / "registry.py",
        "class TargetRegistry:\n"
        "    def register_target(self, name: str) -> str:\n"
        "        return name.lower()\n",
    )


def test_graph_build_persists_native_disparate_relations(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write_relation_repo(tmp_path)

    api = SaguaroAPI(repo_path=str(tmp_path))
    built = api.graph_build(path=".", incremental=False)
    assert built["status"] == "ok"
    assert int(built.get("edges") or 0) > 0

    payload = api.disparate_relations(limit=20)
    assert payload["status"] == "ok"
    assert payload["count"] >= 1
    assert any(
        {
            row.get("source_path"),
            row.get("target_path"),
        }
        == {
            "core/qsg/continuous_engine.py",
            "Saguaro/saguaro/query/pipeline.py",
        }
        for row in payload["relations"]
    )
    relation_row = next(
        row
        for row in payload["relations"]
        if {
            row.get("source_path"),
            row.get("target_path"),
        }
        == {
            "core/qsg/continuous_engine.py",
            "Saguaro/saguaro/query/pipeline.py",
        }
    )
    assert relation_row["relation"] in {
        "evaluation_analogue",
        "subsystem_analogue",
        "port_program_candidate",
    }
    assert relation_row["evidence_spans"]
    assert relation_row["evidence_mix"]
    assert relation_row["confidence_components"]

    relation_filtered = api.graph_query(
        file="core/qsg/continuous_engine.py",
        relation=str(relation_row["relation"]),
        depth=1,
        limit=20,
    )
    assert any(
        str(node.get("file") or "") == "Saguaro/saguaro/query/pipeline.py"
        for node in relation_filtered.get("nodes", [])
    )


def test_omnigraph_imports_native_disparate_relations(tmp_path: Path) -> None:
    _write_relation_repo(tmp_path)
    graph_service = GraphService(str(tmp_path), ParseService(str(tmp_path)))
    built = graph_service.build(path=".", incremental=False)
    assert built["status"] == "ok"

    payload = OmniGraphStore(str(tmp_path), graph_service=graph_service).build(
        traceability_payload={"requirements": [], "records": []}
    )
    assert payload["summary"]["disparate_relation_count"] >= 1
    assert any(
        relation.get("relation_type") in {
            "analogous_to",
            "subsystem_analogue",
            "evaluation_analogue",
            "adaptation_candidate",
            "native_upgrade_path",
            "port_program_candidate",
        }
        for relation in payload["relations"].values()
    )


def test_cli_exposes_disparate_tool(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write_relation_repo(tmp_path)

    api = SaguaroAPI(repo_path=str(tmp_path))
    assert api.graph_build(path=".", incremental=False)["status"] == "ok"

    help_result = subprocess.run(
        [sys.executable, "-m", "saguaro.cli", "disparate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0
    assert "--relation" in help_result.stdout
    assert "--refresh" in help_result.stdout

    run_result = subprocess.run(
        [sys.executable, "-m", "saguaro.cli", "--repo", str(tmp_path), "disparate", "--limit", "5"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0
    assert "relations" in run_result.stdout.lower()
