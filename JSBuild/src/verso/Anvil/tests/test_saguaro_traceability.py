from __future__ import annotations

import json
from pathlib import Path

from saguaro.requirements.model import RequirementStrength
from saguaro.requirements.traceability import TraceabilityService


def _write_graph(repo_root: Path, *files: str) -> None:
    graph_path = repo_root / ".saguaro" / "graph" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps({"files": {path: {"nodes": [], "edges": []} for path in files}}),
        encoding="utf-8",
    )


def test_traceability_service_persists_v2_ledger_and_cache(tmp_path: Path) -> None:
    (tmp_path / "specs").mkdir()
    (tmp_path / "saguaro" / "requirements").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "saguaro" / "requirements" / "traceability.py").write_text(
        "SERVICE = True\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_traceability.py").write_text(
        "def test_ok():\n    assert True\n",
        encoding="utf-8",
    )
    _write_graph(
        tmp_path,
        "saguaro/requirements/traceability.py",
        "tests/test_traceability.py",
    )

    spec = tmp_path / "specs" / "traceability.md"
    spec.write_text(
        """
# Traceability

## Requirements

- The traceability service MUST persist `saguaro/requirements/traceability.py`.
- Verification SHOULD run `pytest tests/test_traceability.py -q`.
""".strip(),
        encoding="utf-8",
    )

    service = TraceabilityService(repo_root=tmp_path)
    first = service.build_from_markdown_paths(
        [spec], owner="trace-test", run_id="run-1"
    )
    second = service.build_from_markdown_paths(
        [spec], owner="trace-test", run_id="run-2"
    )

    ledger_path = tmp_path / "standards" / "traceability" / "TRACEABILITY.v2.jsonl"
    cache_path = tmp_path / ".saguaro" / "traceability" / "cache.json"

    assert first["requirement_count"] == 2
    assert first["appended_count"] == 2
    assert second["appended_count"] == 0
    assert second["skipped_count"] == 2
    assert ledger_path.exists()
    assert cache_path.exists()

    records = service.load_records()
    assert len(records) == 2
    assert records[0]["version"] == "2"
    assert records[0]["requirement_id"].startswith("REQ-")
    assert records[0]["owner"] == "trace-test"
    assert records[0]["design_refs"] == [
        "specs/traceability.md#traceability/requirements"
    ]
    assert records[0]["code_refs"] == ["saguaro/requirements/traceability.py"]

    verification_record = next(
        record for record in records if record["verification_refs"]
    )
    assert verification_record["verification_refs"] == [
        "pytest tests/test_traceability.py -q"
    ]

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["version"] == 2
    assert len(cache["requirements"]) == 2


def test_traceability_service_maps_recommended_cli_surface_to_cli_refs(
    tmp_path: Path,
) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "saguaro").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "saguaro" / "cli.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )
    (tmp_path / "saguaro" / "api.py").write_text(
        "class API:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_saguaro_interface.py").write_text(
        "def test_cli_surface():\n    assert True\n",
        encoding="utf-8",
    )
    _write_graph(
        tmp_path,
        "saguaro/cli.py",
        "saguaro/api.py",
        "tests/test_saguaro_interface.py",
    )

    roadmap = tmp_path / "docs" / "ROADMAP.md"
    roadmap.write_text(
        """
# Roadmap

## 24. Recommended CLI Surface

- The new roadmap should reuse the current CLI style.
""".strip(),
        encoding="utf-8",
    )

    service = TraceabilityService(repo_root=tmp_path)
    result = service.build_from_markdown_paths(
        [roadmap], owner="trace-test", run_id="run-1"
    )

    assert result["requirement_count"] == 1
    records = service.load_records()
    assert len(records) == 1
    assert records[0]["strength"] == RequirementStrength.RECOMMENDED.value
    assert records[0]["code_refs"] == ["saguaro/cli.py", "saguaro/api.py"]
    assert records[0]["test_refs"] == ["tests/test_saguaro_interface.py"]


def test_traceability_service_maps_benchmark_roadmap_to_control_plane_refs(
    tmp_path: Path,
) -> None:
    (tmp_path / "audit" / "control_plane").mkdir(parents=True)
    (tmp_path / "audit" / "runner").mkdir(parents=True)
    (tmp_path / "tests" / "audit").mkdir(parents=True)
    for rel_path in (
        "audit/control_plane/nodes.py",
        "audit/control_plane/capsules.py",
        "audit/control_plane/traceability.py",
        "audit/control_plane/compiler.py",
        "audit/control_plane/ledger.py",
        "audit/control_plane/topology.py",
        "audit/control_plane/comparators.py",
        "audit/runner/benchmark_suite.py",
        "audit/runner/suite_preflight.py",
        "audit/runner/assurance_control_plane.py",
        "tests/audit/test_benchmark_suite.py",
        "tests/test_saguaro_traceability.py",
        "tests/test_saguaro_roadmap_validator.py",
        "saguaro/health.py",
        "saguaro/coverage.py",
        "saguaro/cli.py",
    ):
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("VALUE = 1\n", encoding="utf-8")
    _write_graph(
        tmp_path,
        "audit/control_plane/nodes.py",
        "audit/control_plane/capsules.py",
        "audit/control_plane/traceability.py",
        "audit/control_plane/compiler.py",
        "audit/control_plane/ledger.py",
        "audit/control_plane/topology.py",
        "audit/control_plane/comparators.py",
        "audit/runner/benchmark_suite.py",
        "audit/runner/suite_preflight.py",
        "audit/runner/assurance_control_plane.py",
        "tests/audit/test_benchmark_suite.py",
        "tests/test_saguaro_traceability.py",
        "tests/test_saguaro_roadmap_validator.py",
        "saguaro/health.py",
        "saguaro/coverage.py",
        "saguaro/cli.py",
    )

    roadmap = tmp_path / "benchmark_audit_roadmap.md"
    roadmap.write_text(
        """
# Benchmark Audit Upgrade Roadmap

## 3. Repo Grounding Summary

### 3.2 Health and verification results

- `Repo-grounded observation`

## 8. Target Architecture

### 8.6 Data model principle

- What was intended?

## 21. Appendix E - Exit Criteria for the Roadmap Itself

- topology is normalized,
""".strip(),
        encoding="utf-8",
    )

    service = TraceabilityService(repo_root=tmp_path)
    service.build_from_markdown_paths([roadmap], owner="trace-test", run_id="run-1")
    records = service.load_records()

    data_model_record = next(
        record for record in records if record["statement"] == "What was intended?"
    )
    assert "audit/control_plane/nodes.py" in data_model_record["code_refs"]
    assert "tests/audit/test_benchmark_suite.py" in data_model_record["test_refs"]

    health_record = next(
        record
        for record in records
        if record["statement"] == "`Repo-grounded observation`"
    )
    assert "saguaro/health.py" in health_record["code_refs"]

    exit_record = next(
        record for record in records if record["statement"] == "topology is normalized,"
    )
    assert "audit/control_plane/topology.py" in exit_record["code_refs"]
