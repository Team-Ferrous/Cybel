from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

sys.path.insert(0, os.getcwd())

from audit.runner import native_benchmark_runner as runner
from benchmarks import native_qsg_benchmark as bench
from audit.store.writer import read_ndjson, write_json_atomic


def _runtime_provenance() -> dict[str, object]:
    return {
        "captured_at": "2026-03-07T00:00:00Z",
        "repo_root": "/repo",
        "cwd": "/repo",
        "argv": ["python", "audit"],
        "git": {"commit": "abc", "dirty": False},
        "python": {
            "executable": "/venv/bin/python",
            "version": "3.12",
            "virtual_env": "/venv",
        },
        "host": {
            "hostname": "host",
            "machine": "x86_64",
            "platform": "linux",
            "cpu_model": "cpu",
            "logical_cpus": 8,
            "visible_threads": 8,
            "host_fingerprint": "fingerprint",
        },
        "env": {},
        "native_library": {},
    }


def _attempt_payload(*, run_id: str, attempt_id: str, model: str, run_index: int, warmup: bool) -> dict[str, object]:
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "model_id": model,
        "run_index": run_index,
        "warmup": warmup,
        "prompt_hash": "abc",
        "seed": 1,
        "thread_config": {
            "decode_threads": None,
            "batch_threads": None,
            "ubatch": None,
            "runtime_decode_threads": None,
            "runtime_batch_threads": None,
            "runtime_ubatch": None,
        },
        "host": {
            "affinity_visible_threads": 8,
            "runtime_affinity_visible_threads": 8,
            "launch_affinity_cpus": list(range(8)),
            "runtime_affinity_cpus": list(range(8)),
            "physical_core_count": 4,
            "logical_core_count": 8,
            "p_core_count": 0,
            "omp_max_threads": 8,
            "worker_cpu_mask": "",
            "orchestrator_cpu_mask": "",
            "affinity_policy": "close",
            "cpu_governor": "performance",
            "thp_mode": "madvise",
        },
        "sampling_config": {
            "profile": None,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "min_p": None,
            "presence_penalty": None,
            "repetition_penalty": None,
        },
        "throughput": {
            "wall_tokens_per_second": 12.0,
            "prefill_tps": 30.0,
            "decode_tps": 10.0,
            "e2e_tps": 8.0,
        },
        "latency": {
            "ttft_ms": 100.0,
            "p50_ms": 2.0,
            "p95_ms": 3.0,
        },
        "coherence": {"ok": True, "issues": [], "raw_ok": True, "raw_issues": []},
        "runtime": {
            "runtime_total_seconds": 1.0,
            "runtime_prefill_seconds": 0.4,
            "runtime_decode_seconds": 0.6,
            "graph_stage_ms": {"lm_head": 600.0},
            "graph_stage_calls": {"lm_head": 6},
            "context_stabilizer_enabled": False,
            "context_stabilizer_mode": None,
            "strict_path_stable": True,
            "native_backend_abi_match": True,
            "grover_enabled": True,
            "coconut_enabled": True,
            "perf_event_access": True,
            "perf_event_access_reason": "available:test",
            "python_hot_path_calls": 0,
            "numpy_hot_path_calls": 0,
            "python_attention_fallback_calls": 0,
            "python_ssm_fallback_calls": 0,
            "python_moe_fallback_calls": 0,
            "llama_cpp_hot_path_calls": 0,
            "batch_token_fallback_count": 0,
            "drift_latest": None,
            "drift_mean": None,
            "drift_max": None,
            "drift_decay_ratio": None,
            "drift_damped_blocks": None,
            "drift_pruned_blocks": None,
            "drift_active_tokens": None,
            "drift_overhead_percent": None,
            "stabilizer_seconds": None,
            "stabilizer_calls": None,
            "drift_auto_downgrade_events": None,
        },
        "provenance": {
            "native_build_id": None,
            "native_build_sha256": None,
            "loaded_native_library": None,
            "sanctioned_backend_path": "backend.so",
            "tokenizer_backend": "native",
            "backend_module": "core.native",
            "backend_module_library": "core.native._ops",
            "backend_module_loaded": True,
            "backend_module_marker_symbol": None,
            "backend_module_marker": 0,
            "full_qsg_enabled": True,
            "native_isa_baseline": "avx2",
            "native_backend_abi_match": True,
            "perf_event_access": True,
            "perf_event_access_reason": "available:test",
            "host_access": "user",
            "collect_hw_counters": "auto",
            "require_grover": False,
            "require_coconut": False,
            "autotune": "off",
            "cwd": "/repo",
            "argv": ["python", "benchmarks/native_qsg_benchmark.py"],
            "started_at": "2026-03-07T00:00:00Z",
            "finished_at": "2026-03-07T00:00:01Z",
            "duration_ms": 1000,
            "stdout_sha256": "sha256:abc",
            "stderr_sha256": "sha256:def",
            "benchmark_schema_version": bench.BENCHMARK_REPORT_SCHEMA,
            "report_failure_count": 0,
            "report_failure_keys": [],
        },
        "measurement": {
            "valid": True,
            "issues": [],
            "missing_signals": [],
        },
        "status": {"return_code": 0, "ok": True, "issues": []},
    }


def test_audit_schemas_validate_minimal_payloads() -> None:
    schema_dir = Path("audit/schemas")

    run_manifest_schema = json.loads((schema_dir / "run_manifest.schema.json").read_text(encoding="utf-8"))
    attempt_schema = json.loads((schema_dir / "attempt_record.schema.json").read_text(encoding="utf-8"))
    phase_schema = json.loads((schema_dir / "phase_trace.schema.json").read_text(encoding="utf-8"))
    summary_schema = json.loads((schema_dir / "summary.schema.json").read_text(encoding="utf-8"))
    failure_schema = json.loads((schema_dir / "failure_record.schema.json").read_text(encoding="utf-8"))
    checkpoint_schema = json.loads((schema_dir / "checkpoint.schema.json").read_text(encoding="utf-8"))

    jsonschema.validate(
        {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": "r1",
            "started_at": "2026-03-07T00:00:00Z",
            "repo_root": "/repo",
            "models": ["granite4:tiny-h"],
            "model_sequence": [{"index": 0, "model": "granite4:tiny-h"}],
            "thread_matrix": [
                {"decode_threads": 8, "batch_threads": 8, "ubatch": 16}
            ],
            "warmup_runs": 1,
            "measured_runs": 3,
            "prompt_hash": "abc",
            "planned_attempts": 1,
            "runtime_provenance": _runtime_provenance(),
            "model_contracts": {},
        },
        run_manifest_schema,
    )

    jsonschema.validate(
        _attempt_payload(
            run_id="r1",
            attempt_id="a1",
            model="granite4:tiny-h",
            run_index=0,
            warmup=False,
        ),
        attempt_schema,
    )

    jsonschema.validate(
        {
            "run_id": "r1",
            "attempt_id": "a1",
            "phase": "prefill",
            "duration_ms": 1.2,
            "calls": 1,
            "extra": {},
        },
        phase_schema,
    )

    jsonschema.validate(
        {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": "r1",
            "generated_at": "2026-03-07T00:00:00Z",
            "models": [],
            "top_stage_hotspots": [],
            "stage_origin_map": {},
            "baseline_floors": {},
            "overall_pass": True,
            "failure_count": 0,
            "failure_counts": {
                "total": 0,
                "gate_failure": 0,
                "execution_failure": 0,
            },
            "failed_attempt_ids": [],
            "completed_attempts": 0,
            "planned_attempts": 0,
            "pass": True,
        },
        summary_schema,
    )
    jsonschema.validate(
        {
            "schema_version": runner.SCHEMA_VERSION,
            "run_id": "r1",
            "generated_at": "2026-03-07T00:00:00Z",
            "models": [
                {
                    "model": "granite4:tiny-h",
                    "gates": {"run_stability": True},
                    "run_stability": {
                        "decode_tps_cv_pct": 0.0,
                        "e2e_tps_cv_pct": 0.0,
                        "ttft_ms_cv_pct": 0.0,
                        "decode_time_accounted_cv_pct": 0.0,
                    },
                    "runtime_drift": {
                        "drift_mean_mean": 0.1,
                        "drift_mean_max": 0.2,
                        "drift_max_mean": 0.3,
                        "drift_max_max": 0.4,
                        "drift_pruned_blocks_mean": 1.0,
                        "drift_pruned_blocks_max": 2.0,
                        "stabilizer_seconds_mean": 0.01,
                        "stabilizer_seconds_max": 0.02,
                        "drift_overhead_percent_mean": 1.5,
                        "drift_overhead_percent_max": 2.0,
                    },
                }
            ],
            "top_stage_hotspots": [],
            "stage_origin_map": {},
            "baseline_floors": {},
            "overall_pass": True,
            "failure_count": 0,
            "failure_counts": {
                "total": 0,
                "gate_failure": 0,
                "execution_failure": 0,
            },
            "failed_attempt_ids": [],
            "completed_attempts": 1,
            "planned_attempts": 1,
            "pass": True,
        },
        summary_schema,
    )

    jsonschema.validate(
        {
            "run_id": "r1",
            "attempt_id": "a1",
            "model": "granite4:tiny-h",
            "error": "x",
            "error_type": "RuntimeError",
            "failure_kind": "execution_failure",
            "gate_issues": [],
            "normalized_issues": ["error_type:RuntimeError"],
            "traceback": "",
            "timestamp": "2026-03-07T00:00:00Z",
        },
        failure_schema,
    )

    jsonschema.validate(
        {
            "run_id": "r1",
            "completed_attempt_ids": ["a1"],
            "last_attempt_id": None,
            "updated_at": "2026-03-07T00:00:00Z",
        },
        checkpoint_schema,
    )


def test_thread_matrix_order_is_deterministic() -> None:
    args = Namespace(
        thread_sweep=True,
        decode_thread_sweep="4,2",
        batch_thread_sweep="8",
        ubatch_sweep="16,8",
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
    )

    matrix = runner._thread_matrix(args)

    assert matrix == [
        (4, 8, 16),
        (4, 8, 8),
        (2, 8, 16),
        (2, 8, 8),
    ]


def test_last_attempt_id_sequence_orders_numeric_suffixes() -> None:
    last = runner._last_attempt_id_by_sequence(
        {
            "m00-c00-measure-r99-granite4_tiny-h",
            "m00-c00-measure-r100-granite4_tiny-h",
        }
    )
    assert last == "m00-c00-measure-r100-granite4_tiny-h"


def test_resume_manifest_schema_version_mismatch_raises() -> None:
    existing_manifest = {
        "schema_version": "native_qsg_audit.v0",
        "models": ["granite4:tiny-h"],
        "thread_matrix": [{"decode_threads": 8, "batch_threads": 8, "ubatch": 16}],
        "prompt_hash": "abc",
        "warmup_runs": 1,
        "measured_runs": 1,
    }
    with pytest.raises(RuntimeError, match="schema version"):
        runner._assert_resume_compatible(
            existing_manifest,
            models=["granite4:tiny-h"],
            matrix=[(8, 8, 16)],
            prompt_hash="abc",
            warmup_runs=1,
            measured_runs=1,
            benchmark_config={},
        )


def test_store_baseline_uses_run_scoped_immutable_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner, "host_fingerprint", lambda: "fp")
    summary_a = {"schema_version": runner.SCHEMA_VERSION, "models": []}
    summary_b = {"schema_version": runner.SCHEMA_VERSION, "models": [{"model": "x"}]}

    path_a = runner._store_baseline(summary_a, tmp_path, run_id="run-a")
    path_b = runner._store_baseline(summary_a, tmp_path, run_id="run-b")

    assert path_a.exists()
    assert path_b.exists()
    assert path_a != path_b
    assert runner._store_baseline(summary_a, tmp_path, run_id="run-a") == path_a
    with pytest.raises(RuntimeError, match="immutable baseline file"):
        runner._store_baseline(summary_b, tmp_path, run_id="run-a")


def test_resume_benchmark_config_mismatch_raises() -> None:
    existing_manifest = {
        "schema_version": runner.SCHEMA_VERSION,
        "models": ["granite4:tiny-h"],
        "thread_matrix": [{"decode_threads": 8, "batch_threads": 8, "ubatch": 16}],
        "prompt_hash": "abc",
        "warmup_runs": 1,
        "measured_runs": 1,
        "benchmark_config": {"max_new_tokens": 32},
    }
    with pytest.raises(RuntimeError, match="benchmark config"):
        runner._assert_resume_compatible(
            existing_manifest,
            models=["granite4:tiny-h"],
            matrix=[(8, 8, 16)],
            prompt_hash="abc",
            warmup_runs=1,
            measured_runs=1,
            benchmark_config={"max_new_tokens": 64},
        )


def test_runner_resume_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def _fake_execute_attempt(spec, *, repo_root: Path):  # noqa: ARG001
        calls.append(spec.attempt_id)
        attempt = _attempt_payload(
            run_id="resume-test",
            attempt_id=spec.attempt_id,
            model=spec.model,
            run_index=spec.run_index,
            warmup=bool(spec.warmup),
        )
        return attempt, [], {}

    monkeypatch.setattr(runner, "execute_attempt", _fake_execute_attempt)
    monkeypatch.setattr(runner, "capture_runtime_provenance", lambda _: _runtime_provenance())
    monkeypatch.setattr(
        runner,
        "capture_model_provenance",
        lambda model: {"model": model, "digest": "abc"},
    )

    base_args = Namespace(
        model=["granite4:tiny-h", "qwen3.5:4b"],
        prompt="hello",
        max_new_tokens=8,
        context_length=256,
        warmup_runs=1,
        runs=1,
        sampling_profile=None,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        thread_sweep=True,
        decode_thread_sweep="4",
        batch_thread_sweep="8",
        ubatch_sweep="16,8",
        out_root=str(tmp_path),
        run_id="resume-test",
        resume=False,
        force_rerun=False,
        skip_baseline=True,
        baseline_summary=None,
        baseline_mode=True,
    )

    monkeypatch.setattr(runner, "parse_args", lambda: base_args)
    rc1 = runner.main()
    assert rc1 == 0

    expected_attempts = 2 * 2 * 2
    assert len(calls) == expected_attempts

    resume_args = Namespace(**{**base_args.__dict__, "resume": True})
    monkeypatch.setattr(runner, "parse_args", lambda: resume_args)
    rc2 = runner.main()
    assert rc2 == 0

    assert len(calls) == expected_attempts


def test_runner_auto_baseline_mode_when_no_baseline_reference(
    tmp_path: Path, monkeypatch
) -> None:
    def _fake_execute_attempt(spec, *, repo_root: Path):  # noqa: ARG001
        attempt = _attempt_payload(
            run_id="auto-baseline",
            attempt_id=spec.attempt_id,
            model=spec.model,
            run_index=spec.run_index,
            warmup=bool(spec.warmup),
        )
        return attempt, [], {}

    monkeypatch.setattr(runner, "execute_attempt", _fake_execute_attempt)
    monkeypatch.setattr(runner, "capture_runtime_provenance", lambda _: _runtime_provenance())
    monkeypatch.setattr(
        runner,
        "capture_model_provenance",
        lambda model: {"model": model, "digest": "abc"},
    )

    args = Namespace(
        model=["granite4:tiny-h"],
        prompt="hello",
        max_new_tokens=8,
        context_length=256,
        warmup_runs=0,
        runs=1,
        sampling_profile=None,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        thread_sweep=False,
        decode_thread_sweep=None,
        batch_thread_sweep=None,
        ubatch_sweep=None,
        out_root=str(tmp_path),
        run_id="auto-baseline",
        resume=False,
        force_rerun=False,
        skip_baseline=True,
        baseline_summary=None,
        baseline_mode=False,
    )
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    assert runner.main() == 0


def test_runner_missing_explicit_baseline_path_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "capture_runtime_provenance", lambda _: _runtime_provenance())
    monkeypatch.setattr(
        runner,
        "capture_model_provenance",
        lambda model: {"model": model, "digest": "abc"},
    )

    args = Namespace(
        model=["granite4:tiny-h"],
        prompt="hello",
        max_new_tokens=8,
        context_length=256,
        warmup_runs=0,
        runs=1,
        sampling_profile=None,
        coherence_first=True,
        min_new_tokens_before_eos=12,
        require_openmp=True,
        require_avx2=True,
        require_mmap=False,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        thread_sweep=False,
        decode_thread_sweep=None,
        batch_thread_sweep=None,
        ubatch_sweep=None,
        out_root=str(tmp_path),
        run_id="missing-baseline",
        resume=False,
        force_rerun=False,
        skip_baseline=True,
        baseline_summary=str(tmp_path / "does-not-exist.json"),
        baseline_mode=False,
    )
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    with pytest.raises(RuntimeError, match="Baseline summary file not found"):
        runner.main()


def test_runner_propagates_attempt_gate_controls(tmp_path: Path, monkeypatch) -> None:
    captured: list[object] = []

    def _fake_execute_attempt(spec, *, repo_root: Path):  # noqa: ARG001
        captured.append(spec)
        attempt = _attempt_payload(
            run_id="gate-controls",
            attempt_id=spec.attempt_id,
            model=spec.model,
            run_index=spec.run_index,
            warmup=bool(spec.warmup),
        )
        return attempt, [], {}

    monkeypatch.setattr(runner, "execute_attempt", _fake_execute_attempt)
    monkeypatch.setattr(runner, "capture_runtime_provenance", lambda _: _runtime_provenance())
    monkeypatch.setattr(
        runner,
        "capture_model_provenance",
        lambda model: {"model": model, "digest": "abc"},
    )

    args = Namespace(
        model=["qwen3.5:4b"],
        prompt="Explain thread affinity and SIMD throughput.",
        max_new_tokens=32,
        context_length=1024,
        warmup_runs=0,
        runs=1,
        sampling_profile=None,
        coherence_first=True,
        min_new_tokens_before_eos=16,
        require_openmp=True,
        require_avx2=True,
        require_mmap=False,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        thread_sweep=False,
        decode_thread_sweep=None,
        batch_thread_sweep=None,
        ubatch_sweep=None,
        out_root=str(tmp_path),
        run_id="gate-controls",
        resume=False,
        force_rerun=False,
        skip_baseline=True,
        baseline_summary=None,
        baseline_mode=True,
    )
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    assert runner.main() == 0

    assert len(captured) == 1
    spec = captured[0]
    assert spec.coherence_first is True
    assert spec.min_new_tokens_before_eos == 16
    assert spec.require_openmp is True
    assert spec.require_avx2 is True
    assert spec.require_mmap is False


def test_resume_recovers_orphan_attempt_rows(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def _fake_execute_attempt(spec, *, repo_root: Path):  # noqa: ARG001
        calls.append(spec.attempt_id)
        attempt = _attempt_payload(
            run_id="orphan-test",
            attempt_id=spec.attempt_id,
            model=spec.model,
            run_index=spec.run_index,
            warmup=bool(spec.warmup),
        )
        phase = {
            "run_id": "orphan-test",
            "attempt_id": spec.attempt_id,
            "phase": "prefill",
            "duration_ms": 1.0,
            "calls": 1,
            "extra": {},
        }
        return attempt, [phase], {}

    monkeypatch.setattr(runner, "execute_attempt", _fake_execute_attempt)
    monkeypatch.setattr(runner, "capture_runtime_provenance", lambda _: _runtime_provenance())
    monkeypatch.setattr(
        runner,
        "capture_model_provenance",
        lambda model: {"model": model, "digest": "abc"},
    )

    args = Namespace(
        model=["granite4:tiny-h"],
        prompt="hello",
        max_new_tokens=8,
        context_length=256,
        warmup_runs=0,
        runs=1,
        sampling_profile=None,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        thread_sweep=False,
        decode_thread_sweep=None,
        batch_thread_sweep=None,
        ubatch_sweep=None,
        out_root=str(tmp_path),
        run_id="orphan-test",
        resume=False,
        force_rerun=False,
        skip_baseline=True,
        baseline_summary=None,
        baseline_mode=True,
    )

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    assert runner.main() == 0
    assert len(calls) == 1

    layout = runner.resolve_layout(Path(tmp_path), "orphan-test")
    # Simulate crash window: checkpoint lost completion state while attempt rows survived.
    write_json_atomic(
        layout.checkpoint_json,
        {
            "run_id": "orphan-test",
            "completed_attempt_ids": [],
            "last_attempt_id": None,
            "updated_at": "2026-03-07T00:00:00Z",
        },
    )

    resume_args = Namespace(**{**args.__dict__, "resume": True})
    monkeypatch.setattr(runner, "parse_args", lambda: resume_args)
    assert runner.main() == 0
    assert len(calls) == 2

    attempts = read_ndjson(layout.attempts_ndjson)
    assert len(attempts) == 1


def test_runner_classifies_subprocess_rc_only_as_execution_failure(
    tmp_path: Path, monkeypatch
) -> None:
    def _fake_execute_attempt(spec, *, repo_root: Path):  # noqa: ARG001
        attempt = _attempt_payload(
            run_id="rc-only",
            attempt_id=spec.attempt_id,
            model=spec.model,
            run_index=spec.run_index,
            warmup=bool(spec.warmup),
        )
        attempt["status"] = {
            "return_code": 2,
            "ok": False,
            "issues": ["subprocess_return_code=2"],
        }
        return attempt, [], {}

    monkeypatch.setattr(runner, "execute_attempt", _fake_execute_attempt)
    monkeypatch.setattr(runner, "capture_runtime_provenance", lambda _: _runtime_provenance())
    monkeypatch.setattr(
        runner,
        "capture_model_provenance",
        lambda model: {"model": model, "digest": "abc"},
    )

    args = Namespace(
        model=["granite4:tiny-h"],
        prompt="hello",
        max_new_tokens=8,
        context_length=256,
        warmup_runs=0,
        runs=1,
        sampling_profile=None,
        coherence_first=True,
        min_new_tokens_before_eos=12,
        require_openmp=True,
        require_avx2=True,
        require_mmap=False,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        thread_sweep=False,
        decode_thread_sweep=None,
        batch_thread_sweep=None,
        ubatch_sweep=None,
        out_root=str(tmp_path),
        run_id="rc-only",
        resume=False,
        force_rerun=False,
        skip_baseline=True,
        baseline_summary=None,
        baseline_mode=True,
    )
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    assert runner.main() == 1

    summary = json.loads((Path(tmp_path) / "rc-only" / "summary.json").read_text(encoding="utf-8"))
    assert int(summary["failure_counts"]["execution_failure"]) == 1


def test_failure_normalization_is_deduped_by_attempt() -> None:
    rows = [
        {
            "run_id": "r1",
            "attempt_id": "a1",
            "model": "m1",
            "error": "gate",
            "error_type": "GateFailure",
            "failure_kind": "gate_failure",
            "gate_issues": ["x", "x"],
            "normalized_issues": [],
            "traceback": "",
            "timestamp": "2026-03-07T00:00:00Z",
        },
        {
            "run_id": "r1",
            "attempt_id": "a1",
            "model": "m1",
            "error": "boom",
            "error_type": "RuntimeError",
            "failure_kind": "execution_failure",
            "gate_issues": [],
            "normalized_issues": [],
            "traceback": "tb",
            "timestamp": "2026-03-07T00:00:01Z",
        },
    ]

    normalized = runner._normalize_failure_rows(rows)
    assert len(normalized) == 1
    assert normalized[0]["attempt_id"] == "a1"
    assert normalized[0]["failure_kind"] == "execution_failure"
    assert "error_type:RuntimeError" in normalized[0]["normalized_issues"]


def test_build_summary_aggregates_runtime_drift_metrics() -> None:
    row_a = _attempt_payload(
        run_id="r1",
        attempt_id="a1",
        model="granite4:tiny-h",
        run_index=0,
        warmup=False,
    )
    row_b = _attempt_payload(
        run_id="r1",
        attempt_id="a2",
        model="granite4:tiny-h",
        run_index=1,
        warmup=False,
    )
    row_a["runtime"].update(
        {
            "drift_mean": 0.2,
            "drift_max": 0.6,
            "drift_pruned_blocks": 3,
            "stabilizer_seconds": 0.02,
            "drift_overhead_percent": 6.0,
        }
    )
    row_b["runtime"].update(
        {
            "drift_mean": 0.4,
            "drift_max": 0.8,
            "drift_pruned_blocks": 7,
            "stabilizer_seconds": 0.04,
            "drift_overhead_percent": 8.0,
        }
    )

    summary = runner._build_summary(
        [row_a, row_b],
        baseline_summary=None,
        baseline_mode=True,
        model_order=["granite4:tiny-h"],
    )
    model_summary = summary["models"][0]
    runtime_drift = model_summary["runtime_drift"]

    assert runtime_drift["drift_mean_mean"] == pytest.approx(0.3)
    assert runtime_drift["drift_mean_max"] == pytest.approx(0.4)
    assert runtime_drift["drift_max_mean"] == pytest.approx(0.7)
    assert runtime_drift["drift_max_max"] == pytest.approx(0.8)
    assert runtime_drift["drift_pruned_blocks_mean"] == pytest.approx(5.0)
    assert runtime_drift["drift_pruned_blocks_max"] == pytest.approx(7.0)
    assert runtime_drift["stabilizer_seconds_mean"] == pytest.approx(0.03)
    assert runtime_drift["stabilizer_seconds_max"] == pytest.approx(0.04)
    assert runtime_drift["drift_overhead_percent_mean"] == pytest.approx(7.0)
    assert runtime_drift["drift_overhead_percent_max"] == pytest.approx(8.0)


def test_build_summary_includes_stage_hotspots_and_baseline_deltas() -> None:
    row_a = _attempt_payload(
        run_id="r1",
        attempt_id="a1",
        model="granite4:tiny-h",
        run_index=0,
        warmup=False,
    )
    row_b = _attempt_payload(
        run_id="r1",
        attempt_id="a2",
        model="granite4:tiny-h",
        run_index=1,
        warmup=False,
    )
    row_a["runtime"].update(
        {
            "runtime_decode_seconds": 1.0,
            "graph_stage_ms": {"ffn_gate_up": 400.0, "lm_head": 100.0},
            "graph_stage_calls": {"ffn_gate_up": 40, "lm_head": 10},
        }
    )
    row_b["runtime"].update(
        {
            "runtime_decode_seconds": 1.0,
            "graph_stage_ms": {"ffn_gate_up": 300.0, "lm_head": 200.0},
            "graph_stage_calls": {"ffn_gate_up": 30, "lm_head": 20},
        }
    )

    baseline_summary = {
        "models": [
            {
                "model": "granite4:tiny-h",
                "decode_tps_p50": 10.0,
                "e2e_tps_p50": 8.0,
                "ttft_ms_p95": 100.0,
                "stage_hotspots": [
                    {
                        "stage": "ffn_gate_up",
                        "pct_of_decode": 20.0,
                        "total_ms_mean": 100.0,
                    }
                ],
            }
        ]
    }
    summary = runner._build_summary(
        [row_a, row_b],
        baseline_summary=baseline_summary,
        baseline_mode=False,
        model_order=["granite4:tiny-h"],
    )
    model_summary = summary["models"][0]

    assert model_summary["kernel_map_version"] == runner.KERNEL_MAP_VERSION
    assert model_summary["stage_hotspots"][0]["stage"] == "ffn_gate_up"
    assert model_summary["decode_time_accounted_pct"] == pytest.approx(50.0)
    assert model_summary["stage_hotspots"][0]["cpp_file"] == "core/native/quantized_matmul.cpp"
    assert model_summary["stage_hotspots"][0]["total_ms_stddev"] == pytest.approx(70.71)
    assert model_summary["stage_hotspots"][0]["total_ms_cv_pct"] == pytest.approx(20.2)
    assert model_summary["stage_hotspots"][0]["calls_stddev"] == pytest.approx(7.07)
    assert model_summary["stage_deltas"]["ffn_gate_up"]["baseline_pct"] == pytest.approx(20.0)
    assert model_summary["stage_deltas"]["ffn_gate_up"]["current_pct"] == pytest.approx(35.0)
    assert model_summary["stage_deltas"]["ffn_gate_up"]["delta_pct"] == pytest.approx(15.0)
    assert model_summary["run_stability"] == {
        "decode_tps_cv_pct": 0.0,
        "e2e_tps_cv_pct": 0.0,
        "ttft_ms_cv_pct": 0.0,
        "decode_time_accounted_cv_pct": 0.0,
    }
    assert model_summary["gates"]["run_stability"] is True


def test_build_summary_decode_accounting_overlap_fails_telemetry_gate() -> None:
    row = _attempt_payload(
        run_id="r1",
        attempt_id="a1",
        model="granite4:tiny-h",
        run_index=0,
        warmup=False,
    )
    row["runtime"].update(
        {
            "runtime_decode_seconds": 1.0,
            "graph_stage_ms": {"ffn_gate_up": 1200.0},
            "graph_stage_calls": {"ffn_gate_up": 12},
        }
    )

    summary = runner._build_summary(
        [row],
        baseline_summary=None,
        baseline_mode=True,
        model_order=["granite4:tiny-h"],
    )
    model_summary = summary["models"][0]

    assert model_summary["decode_time_accounted_pct"] == pytest.approx(120.0)
    assert model_summary["gates"]["decode_time_accounting_window"] is False
    assert model_summary["gates"]["run_stability"] is True
    assert "telemetry_gate_failed" in model_summary["gate_issues"]
    assert "telemetry_overlap_failure" in model_summary["gate_issues"]
    assert summary["overall_pass"] is False


def test_build_summary_run_stability_gate_fails_on_high_variance() -> None:
    row_a = _attempt_payload(
        run_id="r1",
        attempt_id="a1",
        model="granite4:tiny-h",
        run_index=0,
        warmup=False,
    )
    row_b = _attempt_payload(
        run_id="r1",
        attempt_id="a2",
        model="granite4:tiny-h",
        run_index=1,
        warmup=False,
    )
    row_a["throughput"].update({"decode_tps": 10.0, "e2e_tps": 8.0})
    row_b["throughput"].update({"decode_tps": 30.0, "e2e_tps": 24.0})
    row_a["latency"].update({"ttft_ms": 100.0})
    row_b["latency"].update({"ttft_ms": 300.0})
    row_a["runtime"].update(
        {
            "runtime_decode_seconds": 1.0,
            "graph_stage_ms": {"lm_head": 1000.0},
            "graph_stage_calls": {"lm_head": 10},
        }
    )
    row_b["runtime"].update(
        {
            "runtime_decode_seconds": 1.0,
            "graph_stage_ms": {"lm_head": 1000.0},
            "graph_stage_calls": {"lm_head": 10},
        }
    )

    summary = runner._build_summary(
        [row_a, row_b],
        baseline_summary=None,
        baseline_mode=True,
        model_order=["granite4:tiny-h"],
    )
    model_summary = summary["models"][0]

    assert model_summary["run_stability"]["decode_tps_cv_pct"] == pytest.approx(70.71)
    assert model_summary["run_stability"]["e2e_tps_cv_pct"] == pytest.approx(70.71)
    assert model_summary["run_stability"]["ttft_ms_cv_pct"] == pytest.approx(70.71)
    assert model_summary["run_stability"]["decode_time_accounted_cv_pct"] == pytest.approx(0.0)
    assert model_summary["gates"]["run_stability"] is False
    assert "run_stability_gate_failed" in model_summary["gate_issues"]
    assert "telemetry_gate_failed" in model_summary["gate_issues"]
    assert summary["overall_pass"] is False


def test_build_summary_emits_root_stage_hotspots_and_origin_map() -> None:
    row_a = _attempt_payload(
        run_id="r1",
        attempt_id="a1",
        model="granite4:tiny-h",
        run_index=0,
        warmup=False,
    )
    row_b = _attempt_payload(
        run_id="r1",
        attempt_id="a2",
        model="qwen3.5:4b",
        run_index=0,
        warmup=False,
    )
    row_a["runtime"].update(
        {
            "runtime_decode_seconds": 1.0,
            "graph_stage_ms": {"ffn_gate_up": 950.0},
            "graph_stage_calls": {"ffn_gate_up": 10},
        }
    )
    row_b["runtime"].update(
        {
            "runtime_decode_seconds": 1.0,
            "graph_stage_ms": {"lm_head": 1000.0},
            "graph_stage_calls": {"lm_head": 4},
        }
    )

    summary = runner._build_summary(
        [row_a, row_b],
        baseline_summary=None,
        baseline_mode=True,
        model_order=["granite4:tiny-h", "qwen3.5:4b"],
    )

    top_hotspots = summary["top_stage_hotspots"]
    stage_origin_map = summary["stage_origin_map"]

    assert top_hotspots[0]["model"] == "qwen3.5:4b"
    assert top_hotspots[0]["stage"] == "lm_head"
    assert top_hotspots[0]["cpp_file"] == "core/native/quantized_matmul.cpp"
    assert top_hotspots[0]["total_ms_stddev"] == pytest.approx(0.0)
    assert top_hotspots[0]["total_ms_cv_pct"] == pytest.approx(0.0)
    assert top_hotspots[0]["calls_stddev"] == pytest.approx(0.0)
    assert stage_origin_map["ffn_gate_up"]["cpp_file"] == "core/native/quantized_matmul.cpp"
    assert stage_origin_map["lm_head"]["models"] == ["qwen3.5:4b"]


def test_baseline_models_carries_stage_hotspots() -> None:
    baseline = runner._baseline_models(
        {
            "models": [
                {
                    "model": "qwen3.5:4b",
                    "decode_tps_p50": 4.8,
                    "e2e_tps_p50": 1.85,
                    "ttft_ms_p95": 12663.0,
                    "stage_hotspots": [{"stage": "ffn_gate_up", "pct_of_decode": 41.6}],
                }
            ]
        }
    )

    assert baseline["qwen3.5:4b"]["stage_hotspots"][0]["stage"] == "ffn_gate_up"


def test_benchmark_isolated_adapter_accepts_legacy_list_payload(monkeypatch) -> None:
    class _Completed:
        returncode = 0
        stdout = '[{"model":"granite4:tiny-h","run_index":0}]'
        stderr = ""

    monkeypatch.setattr(bench.subprocess, "run", lambda *args, **kwargs: _Completed())
    monkeypatch.setattr(bench, "BenchmarkResult", lambda **kwargs: SimpleNamespace(**kwargs))

    args = Namespace(
        prompt="hello",
        max_new_tokens=4,
        context_length=32,
        sampling_profile=None,
        min_new_tokens_before_eos=None,
        fast_native=False,
        disable_logits_processors=False,
        disable_token_penalties=False,
        force_parallel_decode=False,
        coherence_first=False,
    )
    result = bench._run_once_isolated(
        model="granite4:tiny-h",
        args=args,
        override_params={},
        run_index=7,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
    )
    assert result.model == "granite4:tiny-h"
    assert result.run_index == 7


def test_deprecated_runner_entrypoint_rejects_legacy_tuning_flags() -> None:
    rc = runner._deprecated_entrypoint(["--model", "granite4:tiny-h", "--runs", "3"])
    assert rc == 2


def test_deprecated_runner_entrypoint_maps_safe_flags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0

    def _stub_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(runner.subprocess, "run", _stub_run)

    rc = runner._deprecated_entrypoint(
        ["--profile", "smoke", "--run-id", "suite-run-1", "--resume", "--out-root", "audit"]
    )

    assert rc == 0
    cmd = list(captured["cmd"])
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("audit/runner/benchmark_suite.py")
    assert "--profile" in cmd and "smoke" in cmd
    assert "--run-id" in cmd and "suite-run-1" in cmd
    assert "--resume" in cmd
    assert "--out-root" in cmd and "audit" in cmd
