from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit.runner import suite_certification


def test_campaign_stage_sequence_for_gold() -> None:
    assert suite_certification.campaign_stage_sequence("gold") == [
        "bronze",
        "silver",
        "gold-fast",
        "gold",
    ]


def test_campaign_stage_sequence_for_non_gold() -> None:
    assert suite_certification.campaign_stage_sequence("silver") == ["silver"]
    assert suite_certification.campaign_stage_sequence("calibrate") == ["calibrate"]
    assert suite_certification.campaign_stage_sequence("platinum") == [
        "bronze",
        "silver",
        "gold-fast",
        "gold",
        "platinum",
    ]
    assert suite_certification.campaign_stage_sequence("smoke") == ["bronze"]


def test_ensure_host_contract_auto_generates_expected_visible_threads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        suite_certification, "expected_visible_threads", lambda fallback=None: 16
    )

    payload, path, created = suite_certification.ensure_host_contract(
        tmp_path,
        contract_id="auto",
        launch_runtime={
            "captured_at": "2026-03-09T00:00:00Z",
            "host": {
                "host_fingerprint": "host1234",
                "logical_cpus": 16,
                "visible_threads": 2,
                "cpu_model": "Unit Test CPU",
                "platform": "Linux",
                "machine": "x86_64",
                "hostname": "unit-host",
            },
        },
        allow_create=True,
    )

    assert created is True
    assert path == tmp_path / "audit/contracts/hosts/host1234.yaml"
    assert payload["contract_id"] == "host1234"
    assert payload["required_visible_threads"] == 16
    assert payload["logical_cpus"] == 16


def test_assess_runtime_tuning_marks_stale_contracts(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        suite_certification,
        "capture_runtime_provenance",
        lambda _repo_root: {
            "host": {
                "host_fingerprint": "host1234",
                "logical_cpus": 8,
                "visible_threads": 8,
            }
        },
    )
    monkeypatch.setattr(
        suite_certification,
        "capture_model_provenance",
        lambda model: {
            "digest": f"digest:{model}",
            "digest_validated": True,
            "strict_native_supported": True,
        },
    )
    monkeypatch.setattr(
        suite_certification,
        "ensure_host_contract",
        lambda *_args, **_kwargs: (
            {
                "contract_id": "host1234",
                "host_fingerprint": "host1234",
                "required_visible_threads": 8,
            },
            tmp_path / "audit/contracts/hosts/host1234.yaml",
            False,
        ),
    )
    host_contract_path = tmp_path / "audit/contracts/hosts/host1234.yaml"
    host_contract_path.parent.mkdir(parents=True, exist_ok=True)
    host_contract_path.write_text("contract_id: host1234\n", encoding="utf-8")
    monkeypatch.setattr(
        suite_certification,
        "load_tuning_contract",
        lambda _repo_root, _fingerprint, model: (
            {
                "thread_config": {
                    "decode_threads": 8,
                    "batch_threads": 8,
                    "ubatch": 32,
                },
                "benchmark_harness_hash": "sha256:stale",
                "model_digest": f"digest:{model}",
            },
            tmp_path / f"{model}.json",
        ),
    )
    monkeypatch.setattr(
        suite_certification,
        "validate_tuning_contract",
        lambda *_args, **_kwargs: ["benchmark_harness_hash_mismatch:sha256:new"],
    )

    report = suite_certification.assess_runtime_tuning(
        Path("."),
        profile_name="gold",
        models=["granite4:tiny-h"],
    )

    assert report["ready"] is False
    assert report["status"] == "calibration_required"
    assert report["tuning_state"] == "stale"
    assert report["stale_models"] == ["granite4:tiny-h"]
    assert report["refresh_models"] == ["granite4:tiny-h"]
    assert report["admission_decision"] == "revoke"
    assert report["budget_tier"] == "search"
    assert report["tuning_contracts"]["granite4:tiny-h"]["status"] == "stale"
    assert report["tuning_contracts"]["granite4:tiny-h"]["readiness_state"] == "stale"


def test_bootstrap_runtime_tuning_runs_calibration_for_requested_models(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reports = iter(
        [
            {
                "status": "calibration_required",
                "models": ["granite4:tiny-h"],
                "refresh_models": ["granite4:tiny-h"],
                "host_fingerprint": "host1234",
                "required_visible_threads": 8,
                "ready": False,
                "host_contract_created": False,
            },
            {
                "status": "ready",
                "models": ["granite4:tiny-h"],
                "refresh_models": [],
                "host_fingerprint": "host1234",
                "required_visible_threads": 8,
                "ready": True,
                "host_contract_created": False,
            },
        ]
    )
    monkeypatch.setattr(
        suite_certification,
        "assess_runtime_tuning",
        lambda *_args, **_kwargs: next(reports),
    )

    seen: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _run(cmd, **kwargs):  # noqa: ANN001
        seen["cmd"] = cmd
        seen["cwd"] = kwargs.get("cwd")
        return _Completed()

    monkeypatch.setattr(suite_certification.subprocess, "run", _run)

    report = suite_certification.bootstrap_runtime_tuning(
        tmp_path,
        profile_name="gold",
        models=["granite4:tiny-h"],
    )

    assert report["status"] == "bootstrapped"
    assert seen["cmd"] == [
        suite_certification.sys.executable,
        "-m",
        "audit.runner.benchmark_suite",
        "--profile",
        "calibrate",
        "--calibration-mode",
        "search",
        "--calibration-source",
        "campaign",
        "--calibration-target-profile",
        "gold",
        "--models",
        "granite4:tiny-h",
    ]
    assert seen["cwd"] == str(tmp_path)


def test_bootstrap_runtime_tuning_fails_when_post_check_is_not_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reports = iter(
        [
            {
                "status": "calibration_required",
                "models": ["granite4:tiny-h"],
                "refresh_models": ["granite4:tiny-h"],
                "host_fingerprint": "host1234",
                "required_visible_threads": 8,
                "ready": False,
                "host_contract_created": False,
            },
            {
                "status": "calibration_required",
                "models": ["granite4:tiny-h"],
                "refresh_models": ["granite4:tiny-h"],
                "host_fingerprint": "host1234",
                "required_visible_threads": 8,
                "ready": False,
                "host_contract_created": False,
            },
        ]
    )
    monkeypatch.setattr(
        suite_certification,
        "assess_runtime_tuning",
        lambda *_args, **_kwargs: next(reports),
    )

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        suite_certification.subprocess,
        "run",
        lambda *args, **kwargs: _Completed(),
    )

    report = suite_certification.bootstrap_runtime_tuning(
        tmp_path,
        profile_name="gold",
        models=["granite4:tiny-h"],
    )

    assert report["status"] == "failed"
    assert report["reason"] == "post_check_not_ready"
    assert report["failed_models"] == ["granite4:tiny-h"]


def test_validate_tuning_contract_requires_quality_gate(tmp_path: Path) -> None:
    repo_root = tmp_path
    (repo_root / "audit/runner").mkdir(parents=True, exist_ok=True)
    for rel in suite_certification.HARNESS_FILES:
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test", encoding="utf-8")

    issues = suite_certification.validate_tuning_contract(
        {
            "host_contract_id": "host1234",
            "host_fingerprint": "host1234",
            "model": "granite4:tiny-h",
            "model_digest": "digest:granite4:tiny-h",
            "profile_schema_version": "native_qsg_suite.v1",
            "benchmark_harness_hash": suite_certification.benchmark_harness_hash(
                repo_root
            ),
            "contract_hashes": {"host_contract_sha256": "sha256:host"},
            "thread_config": {
                "decode_threads": 8,
                "batch_threads": 8,
                "ubatch": 32,
            },
            "continuous_config": {
                "scheduler_policy": "fcfs",
                "max_active_requests": 2,
                "batch_wait_timeout_ms": 2,
                "max_prefill_rows_per_iteration": 1024,
                "continuous_interleaved_streams": False,
            },
            "pager_config": {
                "state_page_rows": 128,
                "state_compaction_soft_threshold": 0.18,
                "state_compaction_hard_threshold": 0.30,
            },
            "objective_vector": {
                "decode_tps_median": 20.0,
                "ttft_ms_median": 40.0,
                "queue_wait_ms_p95": 2.0,
                "fairness": 1.0,
                "decode_goodput_tps": 20.0,
            },
            "admission": {
                "decision": "search",
                "budget_tier": "search",
                "invocation_source": "campaign",
                "workload_digest": "sha256:test",
            },
            "safe_envelope": {
                "fairness_floor": 0.9,
                "queue_wait_ms_p95_ceiling": 200.0,
                "quality_regression_policy": "fail_closed",
            },
        },
        host_contract={
            "contract_id": "host1234",
            "host_fingerprint": "host1234",
        },
        host_contract_sha256="sha256:host",
        model_name="granite4:tiny-h",
        model_contract={"digest": "digest:granite4:tiny-h"},
        profile_schema_version="native_qsg_suite.v1",
        repo_root=repo_root,
        workload_digest="sha256:test",
    )

    assert "quality_gate_missing" in issues


def test_should_bootstrap_runtime_tuning_respects_policy() -> None:
    report = {
        "ready": False,
        "admission_decision": "probe",
        "invocation_source": "repl_startup",
    }

    assert (
        suite_certification.resolve_runtime_tuning_bootstrap_policy() == "on_first_run"
    )
    assert (
        suite_certification.should_bootstrap_runtime_tuning(
            report,
            policy="explicit",
            has_prior_benchmark_evidence=False,
        )
        is False
    )
    assert (
        suite_certification.should_bootstrap_runtime_tuning(
            report,
            policy="on_first_run",
            has_prior_benchmark_evidence=False,
        )
        is True
    )
    assert (
        suite_certification.should_bootstrap_runtime_tuning(
            report,
            policy="on_first_run",
            has_prior_benchmark_evidence=True,
        )
        is False
    )


def test_mark_runtime_tuning_deferred_sets_deferred_status() -> None:
    report = {
        "status": "calibration_required",
        "models": ["granite4:tiny-h"],
        "refresh_models": ["granite4:tiny-h"],
        "host_fingerprint": "host1234",
        "required_visible_threads": 8,
        "ready": False,
        "tuning_state": "stale",
    }

    deferred = suite_certification.mark_runtime_tuning_deferred(
        report,
        reason="bootstrap_policy=explicit",
        policy="explicit",
    )

    assert deferred["status"] == "deferred"
    assert deferred["ready"] is False
    assert deferred["bootstrap_policy"] == "explicit"
    assert deferred["deferred_reason"] == "bootstrap_policy=explicit"
    assert deferred["admission_decision"] == "skip"
