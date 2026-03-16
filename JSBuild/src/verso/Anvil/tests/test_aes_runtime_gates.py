import json
from pathlib import Path

from core.aes import ComplianceContext, ObligationEngine, RuntimeGateRunner


def test_obligation_engine_matches_high_assurance_hot_path() -> None:
    engine = ObligationEngine("standards/AES_OBLIGATIONS.json")
    context = ComplianceContext(
        run_id="run-1",
        aal="AAL-1",
        domains=["ml"],
        changed_files=["benchmarks/kernel.py"],
        hot_paths=["benchmarks/kernel.py"],
        dependency_changes=["requirements.txt"],
        trace_id="trace-1",
        evidence_bundle_id="bundle-1",
    )

    result = engine.evaluate(context)

    assert "traceability_gate" in result.required_runtime_gates
    assert "chronicle_gate" in result.required_runtime_gates
    assert "domain_report_gate" in result.required_runtime_gates
    assert "AES-TR-1" in result.required_rule_ids


def test_runtime_gate_runner_fails_when_required_artifacts_are_missing(tmp_path: Path) -> None:
    runner = RuntimeGateRunner(str(tmp_path))
    context = ComplianceContext(
        run_id="run-2",
        aal="AAL-1",
        hot_paths=["benchmarks/kernel.py"],
        trace_id="trace-2",
        evidence_bundle_id="bundle-2",
    )

    result = runner.evaluate(context, ["traceability_gate", "chronicle_gate"])

    assert result.passed is False
    assert "traceability.json" in result.missing_artifacts
    assert "chronicle.json" in result.missing_artifacts


def test_runtime_gate_runner_passes_when_expected_artifacts_exist(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".anvil" / "compliance" / "run-3"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "traceability.json").write_text(
        json.dumps(
            {
                "trace_id": "trace-3",
                "run_id": "run-3",
                "requirement_id": "AES-R3",
                "design_ref": "specs/example.md",
                "code_refs": ["core/aes/runtime_gate_runner.py"],
                "test_refs": ["tests/test_aes_runtime_gates.py"],
                "verification_refs": ["pytest tests/test_aes_runtime_gates.py -q"],
                "aal": "AAL-1",
                "owner": "tests",
                "timestamp": "2026-03-05T00:00:00Z",
                "changed_files": ["core/aes/runtime_gate_runner.py"],
                "evidence_bundle_id": "bundle-3",
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "evidence_bundle.json").write_text(
        json.dumps(
            {
                "bundle_id": "bundle-3",
                "change_id": "run-3",
                "trace_id": "trace-3",
                "changed_files": ["core/aes/runtime_gate_runner.py"],
                "aal": "AAL-1",
                "chronicle_snapshot": "snap",
                "chronicle_diff": "diff",
                "verification_report_path": "reports/verify.json",
                "red_team_report_path": "reports/red-team.json",
                "review_signoffs": [
                    {
                        "reviewer": "reviewer-a",
                        "timestamp": "2026-03-05T00:00:00Z",
                        "decision": "approved",
                    }
                ],
                "waivers": [],
            }
        ),
        encoding="utf-8",
    )

    runner = RuntimeGateRunner(str(tmp_path))
    context = ComplianceContext(
        run_id="run-3",
        aal="AAL-1",
        changed_files=["core/aes/runtime_gate_runner.py"],
        trace_id="trace-3",
        evidence_bundle_id="bundle-3",
    )

    result = runner.evaluate(context, ["traceability_gate", "evidence_closure_gate"])

    assert result.passed is True
    assert result.missing_artifacts == []


def test_runtime_gate_runner_rejects_invalid_schema_payloads(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".anvil" / "compliance" / "run-4"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "traceability.json").write_text(
        json.dumps({"trace_id": "trace-4"}),
        encoding="utf-8",
    )

    runner = RuntimeGateRunner(str(tmp_path))
    context = ComplianceContext(run_id="run-4", aal="AAL-1", trace_id="trace-4")

    result = runner.evaluate(context, ["traceability_gate"])

    assert result.passed is False
    assert "traceability.json" in result.missing_artifacts
    assert "schema validation failed" in result.results[0].message


def test_runtime_gate_runner_rejects_stale_traceability_context(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".anvil" / "compliance" / "run-6"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "traceability.json").write_text(
        json.dumps(
            {
                "trace_id": "trace-other",
                "run_id": "run-6",
                "requirement_id": "AES-R6",
                "design_ref": "specs/example.md",
                "code_refs": ["module.py"],
                "test_refs": ["tests/test_aes_runtime_gates.py"],
                "verification_refs": ["pytest -q"],
                "aal": "AAL-1",
                "owner": "tests",
                "timestamp": "2026-03-05T00:00:00Z",
                "changed_files": ["module.py"],
                "evidence_bundle_id": "bundle-6",
            }
        ),
        encoding="utf-8",
    )

    runner = RuntimeGateRunner(str(tmp_path))
    context = ComplianceContext(
        run_id="run-6",
        aal="AAL-1",
        changed_files=["module.py"],
        trace_id="trace-6",
        evidence_bundle_id="bundle-6",
    )

    result = runner.evaluate(context, ["traceability_gate"])

    assert result.passed is False
    assert "traceability.json" in result.missing_artifacts
    assert "does not match compliance context" in result.results[0].message


def test_runtime_gate_runner_rejects_stale_evidence_bundle_context(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".anvil" / "compliance" / "run-7"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "evidence_bundle.json").write_text(
        json.dumps(
            {
                "bundle_id": "bundle-other",
                "change_id": "run-7",
                "trace_id": "trace-7",
                "changed_files": ["module.py"],
                "aal": "AAL-1",
                "chronicle_snapshot": "snap",
                "chronicle_diff": "diff",
                "verification_report_path": "reports/verify.json",
                "red_team_report_path": "reports/red-team.json",
                "review_signoffs": [
                    {
                        "reviewer": "reviewer-a",
                        "timestamp": "2026-03-05T00:00:00Z",
                        "decision": "approved",
                    }
                ],
                "waivers": [],
            }
        ),
        encoding="utf-8",
    )

    runner = RuntimeGateRunner(str(tmp_path))
    context = ComplianceContext(
        run_id="run-7",
        aal="AAL-1",
        changed_files=["module.py"],
        trace_id="trace-7",
        evidence_bundle_id="bundle-7",
    )

    result = runner.evaluate(context, ["evidence_closure_gate"])

    assert result.passed is False
    assert "evidence_bundle.json" in result.missing_artifacts
    assert "bundle_id does not match compliance context" in result.results[0].message


def test_runtime_gate_runner_rejects_domain_reports_missing_threshold_keys(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / ".anvil" / "compliance" / "run-5" / "domain_reports"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "ml_run_manifest.json").write_text(
        json.dumps({"run_manifest": "present"}),
        encoding="utf-8",
    )

    runner = RuntimeGateRunner(str(tmp_path))
    context = ComplianceContext(run_id="run-5", aal="AAL-1", domains=["ml"])
    thresholds = {
        "domain_reports": {
            "ml": {
                "required": [
                    "run_manifest",
                    "deterministic_eval",
                    "non_finite_check",
                    "drift_report",
                ]
            }
        }
    }

    result = runner.evaluate(context, ["domain_report_gate"], thresholds=thresholds)

    assert result.passed is False
    assert "domain_reports/ml_run_manifest.json" in result.missing_artifacts
    assert "missing required keys" in result.results[0].message
