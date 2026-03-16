import json
from pathlib import Path

import jsonschema
import pytest

from saguaro.sentinel.engines.aes import AESEngine


def _load_schema() -> dict:
    return json.loads(
        Path("standards/schemas/traceability.schema.json").read_text(encoding="utf-8")
    )


def _valid_record() -> dict:
    return {
        "trace_id": "trace::phase7::0001",
        "requirement_id": "AES-R7-5",
        "design_ref": "specs/phase7_aes_testing_infrastructure.md",
        "code_refs": ["core/aes/rule_registry.py"],
        "test_refs": ["tests/test_aes_traceability.py"],
        "verification_refs": ["pytest tests/test_aes_traceability.py -q"],
        "aal": "AAL-1",
        "owner": "phase7",
        "timestamp": "2026-03-03T00:00:00Z",
    }


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_traceability_schema_accepts_valid_record() -> None:
    jsonschema.validate(instance=_valid_record(), schema=_load_schema())


def test_traceability_schema_rejects_missing_required_field() -> None:
    invalid = _valid_record()
    invalid.pop("trace_id")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid, schema=_load_schema())


def test_existing_traceability_ledger_records_validate_against_schema() -> None:
    schema = _load_schema()
    ledger = Path("standards/traceability/TRACEABILITY.jsonl")
    lines = [
        line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    assert lines
    for line in lines:
        record = json.loads(line)
        jsonschema.validate(instance=record, schema=schema)
        for ref in record["code_refs"] + record["test_refs"]:
            assert Path(ref).exists(), f"Missing referenced artifact: {ref}"


def test_missing_traceability_record_fails_closed_with_aes_tr_1(tmp_path: Path) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": True,
            "verify_context": {"require_trace": True},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    tr1 = [item for item in violations if item["rule_id"] == "AES-TR-1"]
    assert tr1
    assert tr1[0]["closure_level"] == "blocking"
