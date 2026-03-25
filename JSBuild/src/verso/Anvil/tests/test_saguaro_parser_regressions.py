from __future__ import annotations

import warnings
from pathlib import Path

from saguaro.parsing.parser import SAGUAROParser


def test_parse_file_suppresses_invalid_escape_syntaxwarning(tmp_path: Path) -> None:
    file_path = tmp_path / "pkg" / "regexy.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        'PATTERN = "\\d+\\s+token"\n'
        "def run_pipeline() -> str:\n"
        "    return PATTERN\n",
        encoding="utf-8",
    )

    parser = SAGUAROParser()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entities = parser.parse_file(str(file_path))

    assert any(entity.type == "function" and entity.name == "run_pipeline" for entity in entities)
    assert not any(issubclass(item.category, SyntaxWarning) for item in caught)


def test_entity_metadata_emits_relation_ready_facets(tmp_path: Path) -> None:
    file_path = tmp_path / "core" / "qsg" / "continuous_engine.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "class QSGInferenceEngine:\n"
        "    def evaluation_pipeline(self) -> int:\n"
        "        return 1\n",
        encoding="utf-8",
    )

    parser = SAGUAROParser()
    entities = parser.parse_file(str(file_path))
    file_entity = next(entity for entity in entities if entity.type == "file")

    assert "role_tags" in file_entity.metadata
    assert "feature_families" in file_entity.metadata
    assert "signature_fingerprint" in file_entity.metadata
    assert "structural_fingerprint" in file_entity.metadata
    assert "boundary_markers" in file_entity.metadata
    assert "qsg_surface" in set(file_entity.metadata["role_tags"])
    assert "evaluation_pipeline" in set(file_entity.metadata["feature_families"])
