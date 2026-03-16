import json
from datetime import date, timedelta
from pathlib import Path

import jsonschema
import pytest

from saguaro.sentinel.engines.aes import AESEngine


def _load_schema() -> dict:
    return json.loads(
        Path("standards/schemas/waiver.schema.json").read_text(encoding="utf-8")
    )


def _valid_waiver(expiry: str) -> dict:
    return {
        "waiver_id": "waiver-1",
        "rule_id": "AES-CR-2",
        "change_scope": "core/high_assurance.py",
        "compensating_control": "paired review and runtime guard",
        "risk_owner": "safety-team",
        "expiry": expiry,
        "remediation_ticket": "ENG-1234",
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_waiver_schema_accepts_valid_payload() -> None:
    expiry = (date.today() + timedelta(days=7)).isoformat()
    jsonschema.validate(instance=_valid_waiver(expiry), schema=_load_schema())


def test_waiver_schema_rejects_missing_compensating_control() -> None:
    payload = _valid_waiver((date.today() + timedelta(days=7)).isoformat())
    payload.pop("compensating_control")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=payload, schema=_load_schema())


def test_expired_waiver_blocks_with_aes_tr_3(tmp_path: Path) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")
    expired = _valid_waiver((date.today() - timedelta(days=1)).isoformat())
    _write(tmp_path / "standards" / "waivers" / "expired.json", json.dumps(expired))

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": False,
            "verify_context": {"require_valid_waivers": True},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    tr3 = [item for item in violations if item["rule_id"] == "AES-TR-3"]
    assert tr3
    assert "expired" in tr3[0]["message"].lower()


def test_waiver_with_invalid_expiry_format_blocks_with_reason(tmp_path: Path) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")
    malformed = _valid_waiver("03/03/2026")
    _write(tmp_path / "standards" / "waivers" / "malformed.json", json.dumps(malformed))

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": False,
            "verify_context": {"require_valid_waivers": True},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    tr3 = [item for item in violations if item["rule_id"] == "AES-TR-3"]
    assert tr3
    assert "iso date" in tr3[0]["message"].lower()
