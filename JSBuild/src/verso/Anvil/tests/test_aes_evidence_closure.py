import json
from pathlib import Path

import jsonschema
import pytest

from saguaro.sentinel.engines.aes import AESEngine


def _load_schema() -> dict:
    return json.loads(
        Path("standards/schemas/evidence_bundle.schema.json").read_text(
            encoding="utf-8"
        )
    )


def _valid_bundle() -> dict:
    return {
        "bundle_id": "bundle-1",
        "change_id": "change-1",
        "trace_id": "trace-1",
        "changed_files": ["core/example.py"],
        "aal": "AAL-1",
        "chronicle_snapshot": "snap-1",
        "chronicle_diff": "diff-1",
        "verification_report_path": "reports/verify.json",
        "red_team_report_path": "reports/red-team.json",
        "review_signoffs": [
            {
                "reviewer": "reviewer-a",
                "timestamp": "2026-03-03T00:00:00Z",
                "decision": "approved",
            }
        ],
        "waivers": [],
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_evidence_bundle_schema_accepts_valid_payload() -> None:
    jsonschema.validate(instance=_valid_bundle(), schema=_load_schema())


def test_evidence_bundle_schema_rejects_missing_review_signoffs() -> None:
    invalid = _valid_bundle()
    invalid.pop("review_signoffs")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=_load_schema())


def test_missing_evidence_bundle_blocks_high_aal_with_aes_tr_2(tmp_path: Path) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": False,
            "verify_context": {"require_evidence": True},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    tr2 = [item for item in violations if item["rule_id"] == "AES-TR-2"]
    assert tr2
    assert tr2[0]["closure_level"] == "blocking"


def test_valid_evidence_bundle_satisfies_evidence_requirement(tmp_path: Path) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")
    _write(
        tmp_path / "standards" / "evidence" / "bundle.json", json.dumps(_valid_bundle())
    )

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": False,
            "verify_context": {"require_evidence": True},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    assert not [item for item in violations if item["rule_id"] == "AES-TR-2"]
