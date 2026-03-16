import json
from pathlib import Path
from typing import Any

from core.aes.lint import run_aes_lint
from core.aes.checks import ruff_checks
from tools.verify import verify_aes


def _seed_rules(tmp_path: Path, rules: list[dict]) -> None:
    (tmp_path / "standards").mkdir(parents=True, exist_ok=True)
    (tmp_path / "standards" / "AES_RULES.json").write_text(
        json.dumps(rules),
        encoding="utf-8",
    )


def _static_rule() -> dict:
    return {
        "id": "AES-CR-2",
        "section": "3.3",
        "text": "No bare except",
        "severity": "AAL-0",
        "engine": "agent",
        "auto_fixable": False,
        "domain": ["universal"],
        "language": ["python"],
        "check_function": "core.aes.checks.universal_checks.check_no_bare_except",
        "source_version": "v2",
        "source_refs": ["aes_analysis/v2/deep-research-report v2.md#error-handling"],
        "precedence": 100,
        "selectors": {},
        "execution_mode": "static",
        "parameters": {},
        "required_artifacts": [],
        "waiverable": False,
        "rollout_stage": "ratchet",
        "status": "blocking",
        "title": "No Bare Except",
    }


def _artifact_rule() -> dict:
    return {
        "id": "AES-VIS-1",
        "section": "2",
        "text": "Visual governance packs must include v1 and v2 directives",
        "severity": "AAL-0",
        "engine": "agent",
        "auto_fixable": False,
        "domain": ["universal"],
        "language": ["json"],
        "check_function": "core.aes.checks.universal_checks.check_aes_visuals_pack_presence",
        "source_version": "v2",
        "source_refs": ["aes_analysis/v2/deep-research-report v2.md#visual-governance"],
        "precedence": 100,
        "selectors": {},
        "execution_mode": "artifact",
        "parameters": {},
        "required_artifacts": [
            "aes_visuals/v1/directives.json",
            "aes_visuals/v2/directives.json",
        ],
        "waiverable": False,
        "rollout_stage": "ratchet",
        "status": "blocking",
        "title": "Visual Governance Packs Present",
    }


def _ruff_rule() -> dict:
    return {
        "id": "AES-PY-2",
        "section": "7",
        "text": "Imports SHOULD be deterministic and ordered.",
        "severity": "AAL-2",
        "engine": "ruff",
        "auto_fixable": False,
        "domain": ["universal"],
        "language": ["python"],
        "check_function": "core.aes.checks.ruff_checks.check_ruff_import_order",
        "source_version": "v2",
        "source_refs": ["aes_analysis/v2/deep-research-report v2.md#error-handling"],
        "precedence": 100,
        "selectors": {},
        "execution_mode": "static",
        "parameters": {},
        "required_artifacts": [],
        "waiverable": False,
        "rollout_stage": "ratchet",
        "status": "guarded",
        "title": "Ruff Baseline",
    }


def test_aes_lint_emits_ruff_style_static_findings(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [_static_rule()])
    target = tmp_path / "module.py"
    target.write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")

    violations = run_aes_lint([str(target)], repo_root=str(tmp_path))

    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CR-2"


def test_aes_lint_emits_ruff_backed_findings(tmp_path: Path, monkeypatch) -> None:
    _seed_rules(tmp_path, [_ruff_rule()])
    target = tmp_path / "module.py"
    target.write_text("from math import cos, sin\nimport os\n", encoding="utf-8")

    def fake_ruff_command(path: str, select: tuple[str, ...], timeout_seconds: int = 15) -> list[dict[str, Any]]:
        return [
            {
                "code": "I001",
                "filename": path,
                "line": 1,
                "message": "Imports are unsorted",
            }
        ]

    monkeypatch.setattr(ruff_checks, "_run_ruff_command", fake_ruff_command)

    violations = run_aes_lint([str(target)], repo_root=str(tmp_path))

    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-PY-2"
    assert "Ruff I001" in violations[0]["message"]


def test_aes_lint_catches_dynamic_execution_and_wildcard_imports(
    tmp_path: Path,
) -> None:
    (tmp_path / "standards").mkdir()
    (tmp_path / "standards" / "AES_RULES.json").write_text(
        """
[
  {
    "id": "AES-PY-4",
    "section": "7",
    "text": "No dynamic execution",
    "severity": "AAL-1",
    "engine": "agent",
    "auto_fixable": false,
    "domain": ["universal"],
    "language": ["python"],
    "check_function": "core.aes.checks.universal_checks.check_no_dynamic_execution",
    "source_version": "v2",
    "source_refs": ["aes_analysis/v1/aes4.md#AES-PY5"],
    "precedence": 100,
    "selectors": {},
    "execution_mode": "static",
    "parameters": {},
    "required_artifacts": [],
    "waiverable": false,
    "rollout_stage": "ratchet",
    "status": "blocking",
    "title": "No Dynamic Execution"
  },
  {
    "id": "AES-PY-5",
    "section": "7",
    "text": "No wildcard imports",
    "severity": "AAL-2",
    "engine": "agent",
    "auto_fixable": false,
    "domain": ["universal"],
    "language": ["python"],
    "check_function": "core.aes.checks.universal_checks.check_no_wildcard_imports",
    "source_version": "v2",
    "source_refs": ["aes_analysis/v1/aes4.md#AES-PY3"],
    "precedence": 100,
    "selectors": {},
    "execution_mode": "static",
    "parameters": {},
    "required_artifacts": [],
    "waiverable": false,
    "rollout_stage": "ratchet",
    "status": "guarded",
    "title": "No Wildcard Imports"
  }
]
        """.strip(),
        encoding="utf-8",
    )
    target = tmp_path / "module.py"
    target.write_text(
        "from helpers import *\n\nvalue = eval('40 + 2')\n",
        encoding="utf-8",
    )

    violations = run_aes_lint([str(target)], repo_root=str(tmp_path))
    rule_ids = {item["rule_id"] for item in violations}

    assert "AES-PY-4" in rule_ids
    assert "AES-PY-5" in rule_ids


def test_aes_lint_deduplicates_findings_across_overlapping_targets(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [_static_rule()])
    target = tmp_path / "module.py"
    target.write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")

    violations = run_aes_lint(
        [str(tmp_path), str(target)],
        repo_root=str(tmp_path),
    )

    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CR-2"


def test_aes_lint_filters_to_static_rules_by_default(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [_artifact_rule()])

    violations = run_aes_lint([str(tmp_path)], repo_root=str(tmp_path))

    assert violations == []


def test_aes_lint_can_include_artifact_rules_explicitly(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [_artifact_rule()])

    violations = run_aes_lint(
        [str(tmp_path)],
        repo_root=str(tmp_path),
        include_modes=["artifact"],
    )

    assert len(violations) == 2
    assert {item["rule_id"] for item in violations} == {"AES-VIS-1"}


def test_verify_aes_reports_static_findings_when_catalog_exists(
    tmp_path: Path, monkeypatch
) -> None:
    _seed_rules(tmp_path, [_static_rule()])
    target = tmp_path / "module.py"
    target.write_text("try:\n    pass\nexcept:\n    pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = verify_aes(str(target))

    assert result.passed is False
    assert result.tool == "aes"
    assert any("AES-CR-2" in line for line in result.details)
