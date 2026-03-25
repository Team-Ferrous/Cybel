from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
from saguaro.cli import (
    _build_aes_compliance_report,
    _categorize_aes_violation,
    _format_aes_compliance_report,
)


def test_categorize_aes_violation_maps_traceability() -> None:
    violation = {
        "rule_id": "AES-TR-1",
        "message": "Missing traceability artifact",
    }
    assert _categorize_aes_violation(violation) == "Traceability"


def test_build_aes_compliance_report_aggregates_categories() -> None:
    verify_result = {
        "status": "fail",
        "count": 3,
        "violations": [
            {
                "rule_id": "AES-TR-1",
                "message": "Missing trace",
                "file": "core/a.py",
            },
            {
                "rule_id": "BLE001",
                "message": "Blind exception",
                "file": "core/b.py",
            },
            {
                "rule_id": "D100",
                "message": "Missing docstring",
                "file": "core/c.py",
            },
        ],
    }

    report = _build_aes_compliance_report(verify_result, total_files_scanned=10)

    assert report["verification_status"] == "fail"
    assert report["verification_violation_count"] == 3
    categories = {row["category"]: row for row in report["categories"]}
    assert categories["Traceability"]["violations"] == 1
    assert categories["Error Handling"]["violations"] == 1
    assert categories["Documentation"]["violations"] == 1


def test_format_aes_compliance_report_contains_table() -> None:
    report = {
        "generated_at": "2026-03-03T00:00:00+00:00",
        "overall_compliance_percent": 90.0,
        "target_percent": 95.0,
        "categories": [
            {
                "category": "Traceability",
                "compliant_files": 9,
                "total_files": 10,
                "violations": 1,
                "waivers": 0,
            }
        ],
    }
    output = _format_aes_compliance_report(report)

    assert "AES Compliance Report" in output
    assert "Category" in output
    assert "Traceability" in output


def test_substrate_writes_chronicle_delta_log(tmp_path: Path) -> None:
    substrate = SaguaroSubstrate(root_dir=str(tmp_path))
    substrate._api = SimpleNamespace(
        chronicle_diff=lambda: {"status": "ok", "drift": 0.0}
    )

    diff_payload = substrate.create_chronicle_diff()
    log_path = substrate.write_chronicle_delta_log(
        diff_payload=diff_payload,
        trace_id="trace::phase6",
        task="phase6 drift test",
    )

    payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
    assert payload["trace_id"] == "trace::phase6"
    assert payload["chronicle_diff"]["status"] == "ok"
