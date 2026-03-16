from __future__ import annotations

from pathlib import Path

from saguaro.synthesis.spec import SpecLowerer


def test_spec_lowerer_supports_plain_command_objectives() -> None:
    spec = SpecLowerer().lower_objective(
        "Implement bounded normalize helper in generated/normalize.py"
    )

    assert spec.language == "python"
    assert spec.target_files == ["generated/normalize.py"]
    assert spec.verification.commands


def test_spec_lowerer_supports_markdown_roadmaps(tmp_path: Path) -> None:
    roadmap = tmp_path / "ROADMAP.md"
    roadmap.write_text(
        "# Normalize Kernel\n\n"
        "## Goal\n"
        "Implement a deterministic normalize helper.\n\n"
        "## Proposed Changes\n"
        "#### Normalize\n"
        "- `generated/normalize.py` - Add normalize helper\n"
        "- `tests/test_normalize.py` - Verify normalize helper\n\n"
        "## Verification Plan\n"
        "1. `pytest tests/test_normalize.py`\n"
        "2. `./venv/bin/saguaro verify . --engines native,ruff,semantic --format json`\n",
        encoding="utf-8",
    )

    spec = SpecLowerer().lower_objective(str(roadmap))

    assert spec.stage == "roadmap_lowering"
    assert spec.title == "Normalize Kernel"
    assert spec.target_files == ["generated/normalize.py"]
    assert "pytest tests/test_normalize.py" in spec.verification.commands

