from __future__ import annotations

from infrastructure.ci.phase6_hard_gate import (
    evaluate_execution_errors,
    evaluate_policy_errors,
    extract_trailing_json,
)


def test_extract_trailing_json_ignores_debug_preamble() -> None:
    raw = (
        "DEBUG: startup\n"
        "DEBUG: loaded ops [foo,bar]\n"
        '{"status":"fail","count":3,"violations":[]}\n'
    )
    payload = extract_trailing_json(raw)
    assert payload["status"] == "fail"
    assert payload["count"] == 3


def test_evaluate_policy_errors_reports_expected_blockers() -> None:
    verify = {
        "status": "fail",
        "count": 2,
        "violations": [
            {"rule_id": "AES-TR-1", "message": "Missing traceability artifact"},
            {"rule_id": "AES-REV-1", "message": "Missing independent review proof"},
        ],
    }
    deadcode = {"count": 1}
    audit = {"status": "fail"}

    errors = evaluate_policy_errors(verify, deadcode, audit)
    assert any("Missing traceability closure" in error for error in errors)
    assert any("Missing independent-review proof" in error for error in errors)
    assert any("Dead code candidates present" in error for error in errors)
    assert any("Governance audit failed" in error for error in errors)


def test_evaluate_execution_errors_flags_nonstandard_exit_codes() -> None:
    status = {
        "verify": {"exit_code": 2},
        "aes_report": {"exit_code": 0},
        "deadcode": {"exit_code": 0},
        "audit": {"exit_code": 1},
        "impact": {"attempted": 3, "failed": 1},
    }
    errors = evaluate_execution_errors(status)
    assert any("Verify command execution failed" in error for error in errors)
    assert any("Impact stage had 1 failed target(s)" in error for error in errors)

