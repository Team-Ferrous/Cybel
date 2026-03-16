import json
from pathlib import Path

from core.aes.checks.universal_checks import check_aes_visuals_pack_presence
from saguaro.sentinel.engines.aes import AESEngine

VISUALS_PRESENCE_RULE = {
    "id": "AES-VIS-1",
    "section": "2",
    "text": "Visual governance packs must include v1 and v2 directives",
    "severity": "AAL-0",
    "engine": "agent",
    "auto_fixable": False,
    "domain": ["universal"],
    "language": ["json"],
    "check_function": "core.aes.checks.universal_checks.check_aes_visuals_pack_presence",
}

VISUALS_SHAPE_RULE = {
    "id": "AES-VIS-2",
    "section": "2",
    "text": "Visual governance packs must remain model-readable",
    "severity": "AAL-0",
    "engine": "agent",
    "auto_fixable": False,
    "domain": ["universal"],
    "language": ["json"],
    "check_function": "core.aes.checks.universal_checks.check_aes_visuals_pack_shape",
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_rules(repo_root: Path) -> None:
    _write(
        repo_root / "standards" / "AES_RULES.json",
        json.dumps([VISUALS_PRESENCE_RULE, VISUALS_SHAPE_RULE]),
    )


def _write_valid_pack(path: Path, version: str) -> None:
    _write(
        path,
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact": "anvil_aes_visual_directives",
                "profile": version,
                "generated_on": "2026-03-03",
                "owner": "Anvil",
                "upstream_context": {
                    "aes_baseline_file": "standards/AES_SOURCE_BASELINES.md",
                    "intent": "deterministic test pack",
                },
                "directives": [
                    {
                        "directive_id": f"{version.upper()}-001",
                        "title": "Normative Keyword Discipline",
                        "rationale": "Keep requirement language deterministic.",
                        "enforcement_targets": ["prompt_generation"],
                        "implementation_patterns": ["Classify MUST/SHOULD before planning."],
                        "verification_checks": ["Normative tokens preserved."],
                        "source_refs": ["S01"],
                    }
                ],
            }
        ),
    )


def test_visuals_shape_check_reports_schema_violations(tmp_path: Path) -> None:
    _seed_rules(tmp_path)
    _write(
        tmp_path / "aes_visuals" / "v1" / "directives.json",
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact": "anvil_aes_visual_directives",
                "profile": "v2",
                "generated_on": "2026-03-03",
                "owner": "Anvil",
                "upstream_context": {
                    "aes_baseline_file": "standards/AES_SOURCE_BASELINES.md",
                    "intent": "deterministic test pack",
                },
                "directives": [
                    {
                        "directive_id": "AV1-001",
                        "title": "Bad list field",
                        "rationale": "Missing required list should fail.",
                        "enforcement_targets": [],
                        "implementation_patterns": ["Pattern one"],
                        "verification_checks": ["Check one"],
                        "source_refs": ["S01"],
                    }
                ],
            }
        ),
    )
    _write_valid_pack(tmp_path / "aes_visuals" / "v2" / "directives.json", "v2")

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path))

    presence_ids = {item["rule_id"] for item in violations if item["rule_id"] == "AES-VIS-1"}
    assert not presence_ids
    shape_messages = [item["message"] for item in violations if item["rule_id"] == "AES-VIS-2"]
    assert any("Field 'profile' must equal 'v1' (found 'v2')." in msg for msg in shape_messages)
    assert any(
        "Directive #0 is missing non-empty list field 'enforcement_targets'." in msg
        for msg in shape_messages
    )


def test_visuals_shape_check_reports_invalid_json(tmp_path: Path) -> None:
    _seed_rules(tmp_path)
    _write(tmp_path / "aes_visuals" / "v1" / "directives.json", "{bad json")
    _write_valid_pack(tmp_path / "aes_visuals" / "v2" / "directives.json", "v2")

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path))

    shape_messages = [item["message"] for item in violations if item["rule_id"] == "AES-VIS-2"]
    assert any("Invalid JSON for visuals governance pack" in msg for msg in shape_messages)


def test_visuals_presence_check_is_noop_outside_anchor() -> None:
    violations = check_aes_visuals_pack_presence("{}", "module.py")
    assert violations == []
