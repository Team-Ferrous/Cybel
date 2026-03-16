from __future__ import annotations

import json
from pathlib import Path

from saguaro.requirements.extractor import RequirementExtractor


def _write_graph(repo_root: Path, *files: str) -> None:
    graph_path = repo_root / ".saguaro" / "graph" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(
        json.dumps({"files": {path: {"nodes": [], "edges": []} for path in files}}),
        encoding="utf-8",
    )


def test_requirement_extractor_tags_modality_and_stable_ids(tmp_path: Path) -> None:
    (tmp_path / "specs").mkdir()
    (tmp_path / "saguaro" / "requirements").mkdir(parents=True)
    (tmp_path / "tests").mkdir()

    (tmp_path / "saguaro" / "requirements" / "extractor.py").write_text(
        "class Placeholder:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_saguaro_requirements.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )
    _write_graph(
        tmp_path,
        "saguaro/requirements/extractor.py",
        "tests/test_saguaro_requirements.py",
    )

    spec = tmp_path / "specs" / "requirements.md"
    spec.write_text(
        """
# Requirement Roadmap

## Acceptance Criteria

- The extractor MUST reuse `saguaro/requirements/extractor.py`.
- The service SHOULD validate `tests/test_saguaro_requirements.py`.
- A build artifact is required for release readiness.
""".strip(),
        encoding="utf-8",
    )

    extractor = RequirementExtractor(repo_root=tmp_path)
    first = extractor.extract_file(spec)
    second = extractor.extract_file(spec)

    assert len(first.requirements) == 3
    assert first.requirement_ids() == second.requirement_ids()

    modalities = [item.classification.modality.value for item in first.requirements]
    strengths = [item.classification.strength.value for item in first.requirements]
    assert modalities == ["must", "should", "required"]
    assert strengths == ["mandatory", "recommended", "mandatory"]

    assert first.requirements[0].code_refs == ("saguaro/requirements/extractor.py",)
    assert first.requirements[1].test_refs == ("tests/test_saguaro_requirements.py",)
    assert first.requirements[1].verification_refs == (
        "pytest tests/test_saguaro_requirements.py",
    )


def test_requirement_extractor_uses_normative_sections_for_implicit_items(
    tmp_path: Path,
) -> None:
    (tmp_path / "specs").mkdir()
    spec = tmp_path / "specs" / "implicit.md"
    spec.write_text(
        """
# Plan

## Requirements

- Stable IDs across reruns
""".strip(),
        encoding="utf-8",
    )

    extractor = RequirementExtractor(repo_root=tmp_path)
    result = extractor.extract_file(spec)

    assert len(result.requirements) == 1
    assert result.requirements[0].classification.modality.value == "implicit"
    assert result.requirements[0].classification.strength.value == "unspecified"


def test_requirement_extractor_emits_synthetic_records_for_roadmap_ideas(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "delta_capsule.py").write_text(
        "class DeltaCapsule:\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_delta_capsule.py").write_text(
        "def test_delta_capsule():\n    assert True\n",
        encoding="utf-8",
    )
    _write_graph(
        tmp_path,
        "src/delta_capsule.py",
        "tests/test_delta_capsule.py",
    )

    roadmap = tmp_path / "ROADMAP_INVENTIVE_RESEARCH.md"
    roadmap.write_text(
        """
# Inventive Research Roadmap

## 5. Proposed Capabilities

### 5.1 Delta Capsule Replay

- Core insight: unify `src/delta_capsule.py` with replay validation.
- Smallest credible first experiment: verify `tests/test_delta_capsule.py`.
""".strip(),
        encoding="utf-8",
    )

    result = RequirementExtractor(tmp_path).extract_file(roadmap)

    assert [item.statement for item in result.requirements] == [
        "Implement roadmap idea 'Delta Capsule Replay'. First experiment: verify `tests/test_delta_capsule.py`."
    ]
    assert result.requirements[0].classification.modality.value == "recommended"
    assert result.requirements[0].classification.strength.value == "recommended"
    assert result.requirements[0].code_refs == ("src/delta_capsule.py",)
    assert result.requirements[0].test_refs == ("tests/test_delta_capsule.py",)


def test_requirement_extractor_keeps_roadmap_candidates_and_phases_with_contract(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "feature.py").write_text(
        "def feature():\n    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_feature.py").write_text(
        "def test_feature():\n    assert True\n",
        encoding="utf-8",
    )
    _write_graph(
        tmp_path,
        "src/feature.py",
        "tests/test_feature.py",
    )

    roadmap = tmp_path / "roadmap.md"
    roadmap.write_text(
        """
# Roadmap

## 5. Candidate Implementation Phases

### C01. Feature Candidate

- Suggested `phase_id`: `research`
- Core insight: unify `src/feature.py` with evidence.
- Smallest credible first experiment: verify `tests/test_feature.py`.

## 8. Implementation Program

### Phase 1

- `phase_id`: `research`
- Phase title: Feature Truth Pack
- Objective: Prove `src/feature.py` works end to end.
- Exact wiring points: `src/feature.py`, `tests/test_feature.py`
- Tests: `tests/test_feature.py`
- Verification commands: `pytest tests/test_feature.py -q`

## 9. Implementation Contract

- The system shall implement `src/feature.py`, be tested by `tests/test_feature.py`, and be verified with `pytest tests/test_feature.py -q`.
""".strip(),
        encoding="utf-8",
    )

    result = RequirementExtractor(tmp_path).extract_file(roadmap)

    assert len(result.requirements) == 3
    assert [item.metadata.get("concept_kind") for item in result.requirements] == [
        "roadmap_idea",
        "roadmap_phase",
        "roadmap_contract",
    ]
    assert [
        item.metadata.get("phase_id")
        for item in result.requirements
        if item.metadata.get("phase_id")
    ] == ["research", "research"]
