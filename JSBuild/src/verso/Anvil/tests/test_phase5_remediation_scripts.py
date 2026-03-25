"""Tests for Phase 5 remediation automation scripts."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_script_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_trace_record(path: Path, *, code_ref: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_id": "trace::existing",
        "requirement_id": "REQ-1",
        "design_ref": "specs/existing.md",
        "code_refs": [code_ref],
        "test_refs": ["tests/test_existing.py"],
        "verification_refs": ["verify-existing"],
        "aal": "AAL-1",
        "owner": "owner",
        "timestamp": "2026-03-03T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_backfill_traceability_creates_missing_records(tmp_path: Path) -> None:
    module = _load_script_module(
        REPO_ROOT / "scripts" / "aes_backfill_traceability.py",
        "aes_backfill_traceability",
    )

    (tmp_path / "core").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "core" / "agent.py").write_text("print('x')\n", encoding="utf-8")
    (tmp_path / "core" / "legacy_path.py").write_text("print('y')\n", encoding="utf-8")
    (tmp_path / "tests" / "test_legacy_path.py").write_text(
        "def test_ok():\n    pass\n", encoding="utf-8"
    )

    trace_path = tmp_path / "standards" / "traceability" / "TRACEABILITY.jsonl"
    _write_trace_record(trace_path, code_ref="core/agent.py")

    result = module.run(
        tmp_path,
        traceability_jsonl=Path("standards/traceability/TRACEABILITY.jsonl"),
        high_aal_globs=["core/*.py"],
        report_path=Path(".anvil/artifacts/phase5/report.json"),
        dry_run=False,
        owner="phase5-test",
    )

    assert result["created_count"] == 1
    assert result["missing_targets"] == ["core/legacy_path.py"]

    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    appended = json.loads(lines[1])
    assert appended["code_refs"] == ["core/legacy_path.py"]
    assert appended["owner"] == "phase5-test"


def test_backfill_traceability_dry_run_does_not_modify_jsonl(tmp_path: Path) -> None:
    module = _load_script_module(
        REPO_ROOT / "scripts" / "aes_backfill_traceability.py",
        "aes_backfill_traceability_dry_run",
    )

    (tmp_path / "core").mkdir(parents=True)
    (tmp_path / "core" / "legacy_path.py").write_text("print('y')\n", encoding="utf-8")

    trace_path = tmp_path / "standards" / "traceability" / "TRACEABILITY.jsonl"
    _write_trace_record(trace_path, code_ref="core/agent.py")
    before = trace_path.read_text(encoding="utf-8")

    result = module.run(
        tmp_path,
        traceability_jsonl=Path("standards/traceability/TRACEABILITY.jsonl"),
        high_aal_globs=["core/*.py"],
        report_path=Path(".anvil/artifacts/phase5/report.json"),
        dry_run=True,
    )

    after = trace_path.read_text(encoding="utf-8")
    assert before == after
    assert result["created_count"] == 1


def test_waiver_expiry_classification(tmp_path: Path) -> None:
    module = _load_script_module(
        REPO_ROOT / "scripts" / "aes_waiver_expiry_check.py",
        "aes_waiver_expiry_check",
    )

    waiver_path = tmp_path / "standards" / "waivers.jsonl"
    waiver_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "waiver_id": "w-expired",
            "rule_id": "AES-1",
            "change_scope": "legacy",
            "compensating_control": "monitor",
            "risk_owner": "ops",
            "expiry": "2026-03-01",
            "remediation_ticket": "ANVIL-1",
        },
        {
            "waiver_id": "w-soon",
            "rule_id": "AES-2",
            "change_scope": "legacy",
            "compensating_control": "monitor",
            "risk_owner": "ops",
            "expiry": "2026-03-07",
            "remediation_ticket": "ANVIL-2",
        },
        {
            "waiver_id": "w-active",
            "rule_id": "AES-3",
            "change_scope": "legacy",
            "compensating_control": "monitor",
            "risk_owner": "ops",
            "expiry": "2026-04-07",
            "remediation_ticket": "ANVIL-3",
        },
        {
            "waiver_id": "w-invalid",
            "rule_id": "AES-4",
            "change_scope": "legacy",
            "compensating_control": "monitor",
            "risk_owner": "ops",
            "expiry": "not-a-date",
            "remediation_ticket": "ANVIL-4",
        },
    ]
    waiver_path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )

    result = module.run(
        tmp_path,
        waiver_paths=["standards/waivers.jsonl"],
        due_soon_days=7,
        today=dt.date(2026, 3, 3),
        report_path=Path(".anvil/artifacts/phase5/waiver_report.json"),
        ticket_log=Path("standards/remediation/tickets.md"),
        write_ticket_log=True,
    )

    assert result["expired_count"] == 1
    assert result["due_soon_count"] == 1
    assert result["active_count"] == 1
    assert result["invalid_count"] == 1
    assert result["ticket_recommendations"] == [
        {"waiver_id": "w-expired", "ticket": "ANVIL-1", "priority": "P0"}
    ]

    ticket_log = tmp_path / "standards" / "remediation" / "tickets.md"
    assert ticket_log.exists()
    assert "w-expired" in ticket_log.read_text(encoding="utf-8")
