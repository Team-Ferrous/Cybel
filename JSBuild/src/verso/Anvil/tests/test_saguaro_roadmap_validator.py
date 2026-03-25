import json
from pathlib import Path

from saguaro.cpu import CPUScanner
from saguaro.requirements.extractor import RequirementExtractor
from saguaro.roadmap.validator import RoadmapValidator
from saguaro.validation.engine import ValidationEngine
import pytest


def _write_graph_snapshot(repo_root: Path, *files: str) -> None:
    graph_path = repo_root / ".saguaro" / "graph" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps({"files": {path: {"nodes": [], "edges": []} for path in files}}),
        encoding="utf-8",
    )


def test_requirement_extractor_discovers_single_markdown_file(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "# Roadmap\n\nThe system shall expose `src/feature.py`.\n", encoding="utf-8"
    )

    docs = RequirementExtractor(tmp_path).discover_docs("ROADMAP.md")

    assert docs == [roadmap]


def test_validation_engine_supports_single_markdown_file_path(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    source = tmp_path / "src" / "feature.py"
    tests = tmp_path / "tests" / "test_feature.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("def feature():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Acceptance Criteria\n\n"
        "The system shall implement `src/feature.py` and be tested by `tests/test_feature.py`.\n",
        encoding="utf-8",
    )

    report = ValidationEngine(str(tmp_path)).validate_docs("ROADMAP.md")

    assert report["summary"]["count"] == 1
    assert report["requirements"][0]["state"] == "implemented_witnessed"


def test_roadmap_validator_emits_completion_graph_and_worklist(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    source = tmp_path / "src" / "feature.py"
    tests = tmp_path / "tests" / "test_feature.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("def feature():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Acceptance Criteria\n\n"
        "The system shall implement `src/feature.py` and be tested by `tests/test_feature.py`.\n\n"
        "The system shall emit a verified graph for roadmap completion.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    assert payload["summary"]["count"] == 1
    assert payload["summary"]["completed"] == 1
    assert payload["summary"]["partial"] == 0
    assert payload["summary"]["missing"] == 0
    assert payload["summary"]["blockers"] == 0
    assert {item["completion_state"] for item in payload["requirements"]} == {
        "completed"
    }
    assert payload["graph"]["summary"]["requirement_count"] == 1
    assert payload["graph"]["summary"]["gap_count"] == 0
    assert payload["worklist"] == []


def test_roadmap_validator_prefers_implementation_contract_section(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    source = tmp_path / "src" / "feature.py"
    tests = tmp_path / "tests" / "test_feature.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("def feature():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Roadmap\n\n"
        "This research roadmap should not become a hard requirement.\n\n"
        "## Implementation Contract\n\n"
        "The system shall implement `src/feature.py` and be tested by `tests/test_feature.py`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    assert payload["summary"]["count"] == 1
    assert payload["summary"]["completed"] == 1
    assert payload["worklist"] == []


def test_roadmap_extractor_filters_narrative_sections(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "# Roadmap\n\n"
        "## Executive Summary\n\n"
        "This is background prose and should not become a contract item.\n\n"
        "## 24. Recommended CLI Surface\n\n"
        "### 24.1 Docs and requirements\n\n"
        "- `saguaro docs parse --path .`\n\n"
        "## 38. Minimal Viable Product Definition\n\n"
        "- weak-model packet interface\n",
        encoding="utf-8",
    )

    result = RequirementExtractor(tmp_path).extract_file(roadmap)

    assert [item.statement for item in result.requirements] == [
        "`saguaro docs parse --path .`",
        "weak-model packet interface",
    ]


def test_inventive_research_roadmap_prefers_actionable_sections_over_narrative(
    tmp_path,
):
    roadmap = tmp_path / "ROADMAP_INVENTIVE_RESEARCH.md"
    roadmap.write_text(
        "# Inventive Research Roadmap\n\n"
        "## 1. First-Principles Framing\n\n"
        "The live path touches `core/unified_chat_loop.py` but this section is analysis.\n\n"
        "## 5. Proposed Capabilities\n\n"
        "### P1. Trace Spine\n\n"
        "- Core insight: unify `shared_kernel/event_store.py` with replay evidence.\n"
        "- Smallest credible first experiment: verify `tests/test_event_store.py`.\n\n"
        "## 8. Implementation Program\n\n"
        "### Workstream A: Control-Plane Truth\n\n"
        "- Deliverables:\n"
        "  - `symbol_truth.json` generated during health/index refresh\n\n"
        "## 9. References\n\n"
        "- [R1] https://example.com/paper\n",
        encoding="utf-8",
    )

    result = RequirementExtractor(tmp_path).extract_file(roadmap)

    assert [item.statement for item in result.requirements] == [
        "Implement roadmap idea 'P1. Trace Spine'. First experiment: verify `tests/test_event_store.py`.",
        "`symbol_truth.json` generated during health/index refresh",
    ]


def test_qsg_roadmap_mapping_can_complete_requirement(tmp_path):
    roadmap = tmp_path / "qsg_inference_roadmap.md"
    for rel_path in (
        "core/qsg/runtime_contracts.py",
        "core/native/native_qsg_engine.py",
        "core/qsg/continuous_engine.py",
        "core/qsg/latent_bridge.py",
        "core/memory/latent_memory.py",
        "core/memory/fabric/policies.py",
        "saguaro/state/ledger.py",
        "benchmarks/native_qsg_benchmark.py",
        "cli/commands/features.py",
        "tests/test_qsg_continuous_engine.py",
        "tests/test_latent_package_capture.py",
        "tests/test_development_replay.py",
        "tests/test_runtime_telemetry.py",
        "tests/test_state_ledger.py",
        "tests/test_qsg_runtime_contracts.py",
    ):
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n", encoding="utf-8")

    roadmap.write_text(
        "# QSG Inference Pipeline Upgrade Roadmap\n\n"
        "## 20. Synthesis\n\n"
        "### 20.1 What the system should become\n\n"
        "- `[Inference]` QSG should become a CPU-first delta-aware cognitive runtime.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("qsg_inference_roadmap.md")

    assert payload["summary"]["count"] == 1
    assert payload["summary"]["completed"] == 1
    assert payload["worklist"] == []


def test_roadmap_validator_tracks_inventive_idea_sections(tmp_path):
    roadmap = tmp_path / "ROADMAP_INVENTIVE_RESEARCH.md"
    source = tmp_path / "src" / "delta_capsule.py"
    tests = tmp_path / "tests" / "test_delta_capsule.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("class DeltaCapsule:\n    pass\n", encoding="utf-8")
    tests.write_text("def test_delta_capsule():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Inventive Research Roadmap\n\n"
        "## 5. Proposed Capabilities\n\n"
        "### 5.1 Delta Capsule Replay\n\n"
        "- Core insight: unify `src/delta_capsule.py` with replay validation.\n"
        "- Smallest credible first experiment: verify `tests/test_delta_capsule.py`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP_INVENTIVE_RESEARCH.md")

    assert payload["summary"]["count"] == 1
    assert payload["summary"]["completed"] == 0
    assert payload["summary"]["missing"] == 1
    assert "non_authoritative_roadmap" in payload["requirements"][0]["blockers"]


def test_qsg_roadmap_idea_requires_section_specific_evidence(tmp_path):
    roadmap = tmp_path / "qsg_inference_roadmap.md"
    for rel_path in (
        "core/native/fast_attention.cpp",
        "core/native/fast_attention_wrapper.py",
        "core/native/qsg_forward.py",
    ):
        target = tmp_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n", encoding="utf-8")

    roadmap.write_text(
        "# QSG Inference Pipeline Upgrade Roadmap\n\n"
        "## 10. Proposed Capabilities\n\n"
        "### Idea 03. CPU FlashAttention Rebuild Around Cache Tiles\n\n"
        "- First experiment: tile only the MQA path for common head dimensions and compare TTFT/TPOT against the existing kernel.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("qsg_inference_roadmap.md")

    assert payload["summary"]["count"] == 1
    assert payload["summary"]["completed"] == 0
    assert payload["summary"]["missing"] == 1
    assert "missing_test_refs" in payload["requirements"][0]["blockers"]


def test_roadmap_validator_downgrades_completed_items_when_blockers_remain():
    validator = RoadmapValidator(".")

    assert (
        validator._completion_state("implemented_witnessed", ["missing_test_refs"])
        == "partial"
    )
    assert (
        validator._completion_state(
            "implemented_witnessed", ["missing_passing_witness"]
        )
        == "missing"
    )


def test_roadmap_validator_flags_declared_missing_test_artifacts(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    source = tmp_path / "src" / "feature.py"
    tests = tmp_path / "tests" / "test_feature.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("def feature():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Implementation Contract\n\n"
        "The system shall implement `src/feature.py` and be tested by "
        "`tests/test_feature.py` and `tests/test_feature_perf.py`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    assert payload["summary"]["completed"] == 0
    assert payload["summary"]["partial"] == 1
    assert "missing_declared_test_artifacts" in payload["requirements"][0]["blockers"]


def test_roadmap_validator_scopes_contract_graph_refs_to_explicit_statement_artifacts(
    tmp_path,
):
    roadmap = tmp_path / "roadmap.md"
    feature_a = tmp_path / "src" / "feature_a.py"
    feature_b = tmp_path / "src" / "feature_b.py"
    test_a = tmp_path / "tests" / "test_feature_a.py"
    test_b = tmp_path / "tests" / "test_feature_b.py"
    feature_a.parent.mkdir()
    test_a.parent.mkdir()
    feature_a.write_text("def feature_a():\n    return 'a'\n", encoding="utf-8")
    feature_b.write_text("def feature_b():\n    return 'b'\n", encoding="utf-8")
    test_a.write_text("def test_feature_a():\n    assert True\n", encoding="utf-8")
    test_b.write_text("def test_feature_b():\n    assert True\n", encoding="utf-8")
    _write_graph_snapshot(tmp_path, "src/feature_a.py", "tests/test_feature_a.py")
    roadmap.write_text(
        "# Roadmap\n\n"
        "## 9. Implementation Contract\n\n"
        "- The system shall implement `src/feature_a.py`, be tested by "
        "`tests/test_feature_a.py`, and be verified with "
        "`pytest tests/test_feature_a.py -q`.\n"
        "- The system shall implement `src/feature_b.py`, be tested by "
        "`tests/test_feature_b.py`, and be verified with "
        "`pytest tests/test_feature_b.py -q`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("roadmap.md")

    requirement = next(
        item for item in payload["requirements"] if "feature_b.py" in item["statement"]
    )

    assert requirement["code_refs"] == ["src/feature_b.py"]
    assert requirement["test_refs"] == ["tests/test_feature_b.py"]
    assert requirement["graph_refs"] == ["src/feature_b.py", "tests/test_feature_b.py"]


def test_roadmap_validator_reports_candidate_phase_and_contract_coverage(tmp_path):
    roadmap = tmp_path / "roadmap.md"
    source = tmp_path / "src" / "feature.py"
    tests = tmp_path / "tests" / "test_feature.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("def feature():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Roadmap\n\n"
        "## 5. Candidate Implementation Phases\n\n"
        "### C01. Feature Candidate\n\n"
        "- Suggested `phase_id`: `research`\n"
        "- Core insight: unify `src/feature.py` with evidence.\n"
        "- Smallest credible first experiment: verify `tests/test_feature.py`.\n\n"
        "## 8. Implementation Program\n\n"
        "### Phase 1\n\n"
        "- `phase_id`: `research`\n"
        "- Phase title: Feature Truth Pack\n"
        "- Objective: Prove `src/feature.py` works end to end.\n"
        "- Exact wiring points: `src/feature.py`, `tests/test_feature.py`\n"
        "- Tests: `tests/test_feature.py`\n"
        "- Verification commands: `pytest tests/test_feature.py -q`\n\n"
        "## 9. Implementation Contract\n\n"
        "- The system shall implement `src/feature.py`, be tested by `tests/test_feature.py`, "
        "and be verified with `pytest tests/test_feature.py -q`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("roadmap.md")

    assert payload["summary"]["count"] == 3
    assert payload["summary"]["completed"] == 3
    assert payload["coverage"]["kinds"]["roadmap_idea"]["completed"] == 1
    assert payload["coverage"]["kinds"]["roadmap_phase"]["completed"] == 1
    assert payload["coverage"]["kinds"]["roadmap_contract"]["completed"] == 1
    assert payload["coverage"]["phase_ids"] == ["research"]


def test_roadmap_validator_does_not_inherit_nested_candidate_refs(tmp_path):
    roadmap = tmp_path / "roadmap.md"
    source = tmp_path / "src" / "feature.py"
    tests = tmp_path / "tests" / "test_feature.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("def feature():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Roadmap\n\n"
        "## 5. Candidate Implementation Phases\n\n"
        "### C01. Feature Candidate\n\n"
        "- Core insight: improve feature ergonomics.\n"
        "- Smallest credible first experiment: prove the feature with explicit evidence.\n\n"
        "## 9. Implementation Contract\n\n"
        "- The system shall implement `src/feature.py`, be tested by `tests/test_feature.py`, "
        "and be verified with `pytest tests/test_feature.py -q`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("roadmap.md")
    requirement = next(
        item for item in payload["requirements"] if item["concept_kind"] == "roadmap_idea"
    )

    assert requirement["code_refs"] == []
    assert requirement["test_refs"] == []
    assert "missing_code_refs" in requirement["blockers"]


def test_roadmap_validator_does_not_promote_research_roadmap_contracts(tmp_path):
    roadmap = tmp_path / "inventive_research_roadmap.md"
    source = tmp_path / "core" / "campaign" / "control_plane.py"
    tests = tmp_path / "tests" / "test_campaign_control_kernel.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    tests.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("def score_frontier():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_score_frontier():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Engineering Inventive Designer (R&D) Loop: Inventive Research Roadmap\n\n"
        "## 9. Implementation Contract\n\n"
        "- The system shall implement research frontier utility scoring through "
        "`core/campaign/control_plane.py`, tested by `tests/test_campaign_control_kernel.py`, "
        "and verified with `pytest tests/test_campaign_control_kernel.py`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("inventive_research_roadmap.md")

    assert payload["summary"]["count"] == 1
    assert payload["summary"]["completed"] == 0
    assert payload["summary"]["missing"] == 1
    assert payload["requirements"][0]["gate_context"]["authoritative_contract"] is False
    assert "non_authoritative_roadmap" in payload["requirements"][0]["blockers"]


def test_roadmap_validator_ignores_pytest_commands_as_code_artifacts(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    source = tmp_path / "src" / "feature.py"
    tests = tmp_path / "tests" / "test_feature.py"
    source.parent.mkdir()
    tests.parent.mkdir()
    source.write_text("def feature():\n    return 1\n", encoding="utf-8")
    tests.write_text("def test_feature():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Implementation Contract\n\n"
        "The system shall implement `src/feature.py`, be tested by `tests/test_feature.py`, "
        "and be verified with `pytest tests/test_feature.py`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    assert payload["summary"]["completed"] == 1
    assert payload["summary"]["partial"] == 0
    assert payload["requirements"][0]["blockers"] == []


def test_roadmap_validator_raises_for_missing_markdown_file(tmp_path):
    validator = RoadmapValidator(str(tmp_path))

    with pytest.raises(FileNotFoundError):
        validator.validate("ROADMAP_INVENTIVE_RESEARCH.md")


def test_roadmap_validator_writes_cpu_math_gate_report(tmp_path):
    kernel = tmp_path / "core" / "native" / "kernel.cpp"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests" / "test_kernel.py"
    tests.parent.mkdir(parents=True, exist_ok=True)
    tests.write_text("def test_kernel():\n    assert True\n", encoding="utf-8")
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "# Implementation Contract\n\n"
        "The system shall emit proof-carrying hotspot capsules for `core/native/kernel.cpp` "
        "and be tested by `tests/test_kernel.py`.\n",
        encoding="utf-8",
    )

    CPUScanner(str(tmp_path)).scan(path="core/native", arch="x86_64-avx2")
    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    gate_path = tmp_path / payload["gate_report"]["path"]
    assert gate_path.exists()
    assert payload["gate_report"]["summary"]["hotspot_capsule_count"] >= 1


def test_roadmap_validator_uses_capsule_source_freshness_for_manifest_fallback(
    tmp_path,
):
    kernel = tmp_path / "core" / "native" / "kernel.cpp"
    validator_file = tmp_path / "saguaro" / "roadmap" / "validator.py"
    traceability_file = (
        tmp_path / "standards" / "traceability" / "TRACEABILITY.jsonl"
    )
    tests = tmp_path / "tests" / "test_capsules.py"
    roadmap = tmp_path / "ROADMAP.md"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    validator_file.parent.mkdir(parents=True, exist_ok=True)
    traceability_file.parent.mkdir(parents=True, exist_ok=True)
    tests.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    validator_file.write_text("VALIDATOR = True\n", encoding="utf-8")
    traceability_file.write_text("{}\n", encoding="utf-8")
    tests.write_text("def test_capsules():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Implementation Contract\n\n"
        "The system shall emit proof-carrying hotspot capsules through "
        "`saguaro/roadmap/validator.py` and "
        "`standards/traceability/TRACEABILITY.jsonl`, tested by "
        "`tests/test_capsules.py`.\n",
        encoding="utf-8",
    )

    CPUScanner(str(tmp_path)).scan(path="core/native", arch="x86_64-avx2")
    validator_file.write_text("VALIDATOR = 'updated'\n", encoding="utf-8")

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    assert payload["requirements"][0]["completion_state"] == "completed"
    assert payload["requirements"][0]["gate_context"]["capsule_scope"] == (
        "manifest_fallback"
    )
    assert "stale_hotspot_evidence" not in payload["requirements"][0]["blockers"]


def test_roadmap_validator_flags_stale_hotspot_evidence(tmp_path):
    kernel = tmp_path / "core" / "native" / "kernel.cpp"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests" / "test_kernel.py"
    tests.parent.mkdir(parents=True, exist_ok=True)
    tests.write_text("def test_kernel():\n    assert True\n", encoding="utf-8")
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "# Implementation Contract\n\n"
        "The system shall emit proof-carrying hotspot capsules for `core/native/kernel.cpp` "
        "and be tested by `tests/test_kernel.py`.\n",
        encoding="utf-8",
    )

    CPUScanner(str(tmp_path)).scan(path="core/native", arch="x86_64-avx2")
    kernel.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i] + 1.0f;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    assert "stale_hotspot_evidence" in payload["requirements"][0]["blockers"]
    assert payload["gate_report"]["summary"]["stale_evidence_count"] >= 1


def test_roadmap_validator_flags_runtime_symbol_gate_for_ffi_requirements(tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    source = tmp_path / "core" / "native" / "wrapper.py"
    tests = tmp_path / "tests" / "test_wrapper.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    tests.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "def bind(lib):\n    return getattr(lib, 'missing_symbol', None)\n",
        encoding="utf-8",
    )
    tests.write_text("def test_wrapper():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Implementation Contract\n\n"
        "The system shall close runtime symbol coverage for FFI bindings in `core/native/wrapper.py` "
        "and be tested by `tests/test_wrapper.py`.\n",
        encoding="utf-8",
    )

    payload = RoadmapValidator(str(tmp_path)).validate("ROADMAP.md")

    assert "runtime_symbol_coverage_below_gate" in payload["requirements"][0]["blockers"]
