from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema

sys.path.insert(0, os.getcwd())

from audit.runner import attempt_executor
from audit.runtime_logging import SuiteEventLogger
from audit.runtime_logging import set_active_logger


def _row_payload() -> dict[str, object]:
    return {
        "decode_throughput_tps": 10.0,
        "end_to_end_throughput_tps": 8.0,
        "ttft_ms": 120.0,
        "measurement_valid": True,
        "coherence_valid": True,
        "native_build_id": "20260307T000000Z",
        "native_build_sha256": "sha256:abc",
        "loaded_native_library": "/tmp/libanvil_native_ops.so",
        "sanctioned_backend_path": "prompt -> native tokenizer -> NativeQSGEngine",
        "tokenizer_backend": "native",
        "backend_module": "qwen35",
        "backend_module_library": "/tmp/libanvil_backend_qwen35.so",
        "backend_module_loaded": True,
        "native_isa_baseline": "avx2",
        "native_backend_abi_match": True,
        "grover_enabled": True,
        "coconut_enabled": True,
        "perf_event_access": True,
        "perf_event_access_reason": "available:test",
        "full_qsg_enabled": True,
        "pmu_observed": True,
        "pmu_parse_error": "",
        "pmu_cycles": 1200000.0,
        "pmu_instructions": 2400000.0,
        "pmu_ipc": 2.0,
        "pmu_cache_references": 200000.0,
        "pmu_cache_misses": 10000.0,
        "pmu_cache_miss_rate": 0.05,
        "pmu_context_switches": 8.0,
        "pmu_cpu_migrations": 2.0,
        "pmu_page_faults": 16.0,
        "runtime_total_seconds": 1.0,
        "runtime_prefill_seconds": 0.4,
        "runtime_decode_seconds": 0.6,
        "graph_stage_ms": {},
        "graph_stage_calls": {},
        "context_stabilizer_enabled": False,
        "context_stabilizer_mode": None,
        "strict_path_stable": True,
        "python_hot_path_calls": 0,
        "numpy_hot_path_calls": 0,
        "python_attention_fallback_calls": 0,
        "python_ssm_fallback_calls": 0,
        "python_moe_fallback_calls": 0,
        "llama_cpp_hot_path_calls": 0,
        "batch_token_fallback_count": 0,
        "physical_core_count": 8,
        "logical_core_count": 16,
        "p_core_count": 8,
        "affinity_policy": "close",
        "omp_max_threads": 16,
        "worker_cpu_mask": "0,1,2,3,4,5,6,7",
        "orchestrator_cpu_mask": "8,9,10,11,12,13,14,15",
        "cpu_governor": "performance",
        "thp_mode": "madvise",
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
        "sampling_profile": "instruct_deterministic",
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "presence_penalty": 0.5,
        "repetition_penalty": 1.5,
        "per_token_latency_p50_ms": 2.0,
        "per_token_latency_p95_ms": 3.0,
        "arch_hidden_dim": 4096,
        "arch_num_layers": 32,
        "arch_num_heads": 32,
        "arch_num_kv_heads": 8,
        "arch_head_dim": 128,
        "arch_intermediate_dim": 14336,
        "arch_vocab_size": 151936,
        "arch_num_experts": 0,
        "arch_top_k_experts": 0,
        "arch_weight_qtype": "Q6_K",
        "parallel_decode": True,
        "speculative_decode": False,
        "generation_mode": "block_diffusion",
        "benchmark_label": "block_diffusion_candidate",
        "prompt_category": "canonical",
        "temperature_band": "deterministic",
        "accepted_parallel_tokens": 16,
        "rejected_parallel_tokens": 0,
        "draft_source": "block_diffusion_native",
        "blockwise_convergence_rate": 0.9,
        "quality_guard_triggered": False,
        "kernel_efficiency": {
            "ffn_gate_up": {
                "calls": 32,
                "gflops_per_second": 123.4,
            }
        },
        "tokens_per_second": 8.0,
        "effective_prefill_throughput_tps": 20.0,
        "sample_text": "Native telemetry proves the hot path stayed in C++ and produced readable output.",
        "raw_sample_text": "<think>internal</think> Native telemetry proves the hot path stayed in C++ and produced readable output.",
        "printable_ratio": 1.0,
        "repeated_8gram_ratio": 0.0,
        "utf8_valid": True,
        "leaked_control_text": False,
        "leaked_think_tags": False,
        "hot_path_proof": {
            "sanctioned_backend_path": "prompt -> native tokenizer -> NativeQSGEngine",
            "tokenizer_backend": "native",
        },
    }


def _report_payload() -> dict[str, object]:
    return {
        "schema_version": "native_qsg_benchmark.v6",
        "host": {
            "affinity_visible_threads": 16,
            "runtime_affinity_visible_threads": 16,
            "launch_affinity_cpus": list(range(16)),
            "runtime_affinity_cpus": list(range(16)),
        },
        "failure_count": 0,
        "failure_keys": [],
        "failures": {},
    }


def _fake_logged_subprocess_factory(
    *,
    captured: dict[str, object] | None = None,
    returncode: int = 0,
    stdout: str = "{}",
    stderr: str = "",
):
    def _fake_logged_subprocess(**kwargs):  # noqa: ANN001
        if captured is not None:
            captured["cmd"] = list(kwargs["cmd"])
            captured["env"] = dict(kwargs["env"])
        stdout_path = kwargs.get("stdout_path")
        stderr_path = kwargs.get("stderr_path")
        if stdout_path is not None:
            stdout_path.write_text(stdout, encoding="utf-8")
        if stderr_path is not None:
            stderr_path.write_text(stderr, encoding="utf-8")
        completed = SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
        completed.anvil_subprocess_metrics = {
            "sequence_id": 7,
            "tool_run_id": "attempt:test",
            "wall_time_ms": 12.5,
            "user_time_ms": 3.0,
            "sys_time_ms": 1.5,
            "max_rss_mb": 42.0,
        }
        completed.anvil_evidence_capsule = {"artifact_paths": {}}
        completed.anvil_evidence_path = str(
            (stdout_path.parent if stdout_path is not None else Path("."))
            / "evidence_capsule.json"
        )
        return completed

    return _fake_logged_subprocess


def test_execute_attempt_includes_cpu_gate_and_coherence_flags(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(captured=captured),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (_row_payload(), _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a1",
        model="qwen3.5:4b",
        prompt="Explain AVX2 and OpenMP throughput scaling in one paragraph.",
        max_new_tokens=64,
        context_length=2048,
        decode_threads=4,
        batch_threads=4,
        ubatch=32,
        sampling_profile="instruct_deterministic",
        coherence_first=True,
        min_new_tokens_before_eos=12,
        require_openmp=True,
        require_avx2=True,
        require_mmap=False,
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)
    cmd = captured["cmd"]

    assert "--coherence-first" in cmd
    assert "--min-new-tokens-before-eos" in cmd
    assert "12" in cmd
    assert "--require-openmp" in cmd
    assert "--require-avx2" in cmd
    assert "--require-mmap" not in cmd
    assert attempt["status"]["ok"] is True
    assert attempt["host"]["affinity_visible_threads"] == 16
    assert attempt["host"]["runtime_affinity_visible_threads"] == 16
    assert attempt["host"]["omp_max_threads"] == 16
    assert attempt["host"]["cpu_governor"] == "performance"
    assert attempt["runtime"]["strict_path_stable"] is True
    assert int(attempt["runtime"]["python_hot_path_calls"]) == 0
    assert bool(attempt["runtime"]["context_stabilizer_enabled"]) is False
    assert attempt["measurement"]["pmu"]["observed"] is True
    assert float(attempt["measurement"]["pmu"]["ipc"]) == 2.0
    assert float(attempt["measurement"]["pmu"]["context_switches"]) == 8.0
    assert int(attempt["architecture"]["hidden_dim"]) == 4096
    assert attempt["sample"]["text"].startswith("Native telemetry proves")
    assert attempt["generation_mode"] == "block_diffusion"
    assert attempt["accepted_parallel_tokens"] == 16
    assert attempt["draft_source"] == "block_diffusion_native"
    assert (
        float(attempt["kernel_efficiency"]["ffn_gate_up"]["gflops_per_second"]) == 123.4
    )

    schema = json.loads(
        Path("audit/schemas/attempt_record.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate({"run_id": "test-run", **attempt}, schema)


def test_execute_attempt_omits_optional_flags_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(captured=captured),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (_row_payload(), _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a2",
        model="granite4:tiny-h",
        prompt="Describe tensor cache locality.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        warmup=False,
        run_index=0,
    )
    attempt_executor.execute_attempt(spec, repo_root=tmp_path)
    cmd = captured["cmd"]

    assert "--coherence-first" not in cmd
    assert "--min-new-tokens-before-eos" not in cmd
    assert "--require-openmp" not in cmd
    assert "--require-avx2" not in cmd
    assert "--require-mmap" not in cmd


def test_execute_attempt_exports_non_ar_native_env_contract(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(captured=captured),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (_row_payload(), _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a2-nonar",
        model="granite4:tiny-h",
        prompt="Describe the native non autoregressive decode contract.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        force_parallel_decode=True,
        forbid_autoregressive_fallback=True,
        warmup=False,
        run_index=0,
    )

    attempt_executor.execute_attempt(spec, repo_root=tmp_path)
    env = dict(captured["env"])

    assert env["ANVIL_FORCE_PARALLEL_DECODE"] == "1"
    assert env["ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK"] == "1"
    assert env["ANVIL_PARALLEL_AR_RECOVERY_ENABLED"] == "0"


def test_execute_attempt_records_strict_native_env_receipt_in_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (_row_payload(), _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a2-receipt",
        model="granite4:tiny-h",
        prompt="Describe the strict native receipt.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        force_parallel_decode=True,
        forbid_autoregressive_fallback=True,
        warmup=False,
        run_index=0,
    )

    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert attempt["provenance"]["env_overrides"]["ANVIL_FORCE_PARALLEL_DECODE"] == "1"
    assert (
        attempt["provenance"]["env_overrides"]["ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK"]
        == "1"
    )
    assert (
        attempt["provenance"]["env_overrides"]["ANVIL_PARALLEL_AR_RECOVERY_ENABLED"]
        == "0"
    )


def test_execute_attempt_records_nonzero_subprocess_return_code(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(returncode=7),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (_row_payload(), _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a3",
        model="granite4:tiny-h",
        prompt="Describe cache locality.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert attempt["status"]["ok"] is False
    assert "subprocess_return_code=7" in attempt["status"]["issues"]


def test_execute_attempt_flags_strict_path_and_python_fallback_gates(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _row_payload()
    payload["strict_path_stable"] = False
    payload["python_hot_path_calls"] = 3
    payload["batch_token_fallback_count"] = 1

    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (payload, _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a4",
        model="granite4:tiny-h",
        prompt="Describe cache locality.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert attempt["status"]["ok"] is False
    assert "strict_path_stable=false" in attempt["status"]["issues"]
    assert "python_hot_path_calls=3" in attempt["status"]["issues"]
    assert "batch_token_fallback_count=1" in attempt["status"]["issues"]


def test_execute_attempt_requires_stabilizer_runtime_fields_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _row_payload()
    payload.update(
        {
            "context_stabilizer_enabled": True,
            "context_stabilizer_mode": "adaptive",
            "drift_latest": 0.1,
            "drift_mean": 0.1,
            "drift_max": 0.2,
            "drift_decay_ratio": 0.95,
            "drift_damped_blocks": 1,
            "drift_pruned_blocks": 0,
            "drift_active_tokens": 128,
            "drift_overhead_percent": 21.5,
            "stabilizer_seconds": None,
            "stabilizer_calls": 2,
            "drift_auto_downgrade_events": 0,
        }
    )

    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (payload, _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a5",
        model="granite4:tiny-h",
        prompt="Describe cache locality.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert attempt["status"]["ok"] is False
    assert "drift_overhead_percent=21.5>20.0" in attempt["status"]["issues"]
    assert "stabilizer_field_missing:stabilizer_seconds" in attempt["status"]["issues"]


def test_extract_flat_record_preserves_sample_and_hot_path_fields() -> None:
    row, report = attempt_executor._extract_flat_record(
        {
            "schema_version": "native_qsg_benchmark.v6",
            "results": [
                {
                    "identity": {"model": "granite4:tiny-h"},
                    "throughput": {
                        "decode_throughput_tps": 8.0,
                        "end_to_end_throughput_tps": 7.0,
                    },
                    "latency": {"ttft_ms": 120.0},
                    "quality": {
                        "coherence_valid": True,
                        "coherence_issues": [],
                    },
                    "measurement": {
                        "valid": True,
                        "issues": [],
                        "perf_stat_artifact": "/tmp/perf.stat",
                    },
                    "hot_path": {
                        "sanctioned_backend_path": "native",
                        "tokenizer_backend": "native",
                        "proof": {
                            "sanctioned_backend_path": "native",
                            "tokenizer_backend": "native",
                        },
                    },
                    "sample": {
                        "text": "Readable generated text.",
                        "raw_text": "<think>raw</think> Readable generated text.",
                    },
                    "status": {"issues": []},
                }
            ],
        }
    )

    assert report["schema_version"] == "native_qsg_benchmark.v6"
    assert row["sample_text"] == "Readable generated text."
    assert row["raw_sample_text"].startswith("<think>")
    assert row["hot_path_proof"]["sanctioned_backend_path"] == "native"
    assert row["measurement_valid"] is True


def test_execute_attempt_wraps_required_hw_counters_with_perf_stat(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _row_payload()
    for key in (
        "pmu_observed",
        "pmu_parse_error",
        "pmu_cycles",
        "pmu_instructions",
        "pmu_ipc",
        "pmu_cache_references",
        "pmu_cache_misses",
        "pmu_cache_miss_rate",
        "pmu_context_switches",
        "pmu_cpu_migrations",
        "pmu_page_faults",
    ):
        payload[key] = (
            False
            if key == "pmu_observed"
            else ("" if key == "pmu_parse_error" else None)
        )
    captured: dict[str, object] = {}

    def _fake_stream_subprocess_output(**kwargs):  # noqa: ANN003
        cmd = list(kwargs["cmd"])
        captured["cmd"] = cmd
        perf_path = Path(cmd[cmd.index("-o") + 1])
        perf_path.write_text(
            "\n".join(
                [
                    "1,234,567      cycles",
                    "2,469,134      instructions              #    2.00  insn per cycle",
                    "98,765         cache-references",
                    "4,938          cache-misses              #    5.00% of all cache refs",
                    "32             context-switches",
                    "7              cpu-migrations",
                    "12             page-faults",
                ]
            ),
            encoding="utf-8",
        )
        kwargs["stdout_path"].write_text("{}", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(
        attempt_executor,
        "_stream_subprocess_output",
        _fake_stream_subprocess_output,
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (payload, _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a-perf",
        model="granite4:tiny-h",
        prompt="Explain CPU counter collection.",
        max_new_tokens=16,
        context_length=1024,
        decode_threads=4,
        batch_threads=4,
        ubatch=16,
        sampling_profile="coding_deterministic",
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=True,
        require_avx2=True,
        require_mmap=False,
        host_access="privileged",
        collect_hw_counters="required",
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert captured["cmd"][:2] == ["perf", "stat"]
    assert attempt["artifact_paths"]["perf_stat_artifact"].endswith("perf_stat.txt")
    assert attempt["measurement"]["pmu"]["observed"] is True
    assert float(attempt["measurement"]["pmu"]["ipc"]) == 2.0
    assert float(attempt["measurement"]["pmu"]["context_switches"]) == 32.0
    assert attempt["status"]["ok"] is True


def test_execute_attempt_required_hw_counters_fail_when_perf_stat_missing(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _row_payload()
    for key in (
        "pmu_observed",
        "pmu_parse_error",
        "pmu_cycles",
        "pmu_instructions",
        "pmu_ipc",
        "pmu_cache_references",
        "pmu_cache_misses",
        "pmu_cache_miss_rate",
        "pmu_context_switches",
        "pmu_cpu_migrations",
        "pmu_page_faults",
    ):
        payload[key] = (
            False
            if key == "pmu_observed"
            else ("" if key == "pmu_parse_error" else None)
        )

    def _fake_stream_subprocess_output(**kwargs):  # noqa: ANN003
        perf_path = Path(kwargs["cmd"][kwargs["cmd"].index("-o") + 1])
        perf_path.write_text("", encoding="utf-8")
        kwargs["stdout_path"].write_text("{}", encoding="utf-8")
        kwargs["stderr_path"].write_text("", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(
        attempt_executor,
        "_stream_subprocess_output",
        _fake_stream_subprocess_output,
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (payload, _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a-perf-missing",
        model="granite4:tiny-h",
        prompt="Explain missing perf evidence.",
        max_new_tokens=16,
        context_length=1024,
        decode_threads=4,
        batch_threads=4,
        ubatch=16,
        sampling_profile="coding_deterministic",
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=True,
        require_avx2=True,
        require_mmap=False,
        host_access="privileged",
        collect_hw_counters="required",
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert attempt["measurement"]["pmu"]["observed"] is False
    assert attempt["measurement"]["pmu"]["parse_error"] == "empty_perf_stat_artifact"
    assert "pmu_observed=false(empty_perf_stat_artifact)" in attempt["status"]["issues"]
    assert attempt["status"]["ok"] is False


def test_execute_attempt_fails_on_empty_generated_output(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _row_payload()
    payload["sample_text"] = ""

    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (payload, _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a-empty",
        model="granite4:tiny-h",
        prompt="Describe cache locality.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert attempt["status"]["ok"] is False
    assert "empty_sample_text" in attempt["status"]["issues"]


def test_execute_attempt_requires_hot_path_proof_when_strict(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _row_payload()
    payload["hot_path_proof"] = {}

    monkeypatch.setattr(
        attempt_executor,
        "run_logged_subprocess",
        _fake_logged_subprocess_factory(),
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (payload, _report_payload()),
    )

    spec = attempt_executor.AttemptSpec(
        attempt_id="a-proof",
        model="granite4:tiny-h",
        prompt="Describe cache locality.",
        max_new_tokens=32,
        context_length=1024,
        decode_threads=None,
        batch_threads=None,
        ubatch=None,
        sampling_profile=None,
        coherence_first=False,
        min_new_tokens_before_eos=None,
        require_openmp=True,
        require_avx2=True,
        require_mmap=False,
        warmup=False,
        run_index=0,
    )
    attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)

    assert attempt["status"]["ok"] is False
    assert (
        "hot_path_proof_missing:sanctioned_backend_path" in attempt["status"]["issues"]
    )
    assert "hot_path_proof_missing:tokenizer_backend" in attempt["status"]["issues"]


def test_execute_attempt_persists_stream_logs_and_events(
    monkeypatch, tmp_path: Path
) -> None:
    def _fake_logged_subprocess(**kwargs):  # noqa: ANN001
        stdout_path = kwargs["stdout_path"]
        stderr_path = kwargs["stderr_path"]
        stdout_path.write_text("child-stdout\n", encoding="utf-8")
        stderr_path.write_text("child-stderr\n", encoding="utf-8")
        return SimpleNamespace(
            returncode=0, stdout="child-stdout\n", stderr="child-stderr\n"
        )

    monkeypatch.setattr(
        attempt_executor, "run_logged_subprocess", _fake_logged_subprocess
    )
    monkeypatch.setattr(attempt_executor, "_extract_json", lambda _: {})
    monkeypatch.setattr(
        attempt_executor,
        "_extract_flat_record",
        lambda _: (_row_payload(), _report_payload()),
    )

    logger = SuiteEventLogger(
        run_id="attempt-test",
        run_root=tmp_path,
        events_path=tmp_path / "events.ndjson",
        transcript_path=tmp_path / "terminal_transcript.log",
        console_log_path=tmp_path / "console.log",
        ui_mode="plain",
        log_level="trace",
    )
    logger.start()
    set_active_logger(logger)
    try:
        spec = attempt_executor.AttemptSpec(
            attempt_id="a6",
            model="granite4:tiny-h",
            prompt="Describe cache locality.",
            max_new_tokens=32,
            context_length=1024,
            decode_threads=4,
            batch_threads=8,
            ubatch=16,
            sampling_profile=None,
            coherence_first=False,
            min_new_tokens_before_eos=None,
            require_openmp=False,
            require_avx2=False,
            require_mmap=False,
            warmup=False,
            run_index=0,
            lane_id="canonical_all_on",
            artifact_paths={
                "stdout_log": str(tmp_path / "attempt" / "stdout.log"),
                "stderr_log": str(tmp_path / "attempt" / "stderr.log"),
            },
            artifacts_dir=str(tmp_path / "attempt"),
        )
        attempt, _, _ = attempt_executor.execute_attempt(spec, repo_root=tmp_path)
    finally:
        set_active_logger(None)
        logger.close()

    stdout_path = tmp_path / "attempt" / "stdout.log"
    stderr_path = tmp_path / "attempt" / "stderr.log"
    evidence_path = tmp_path / "attempt" / "evidence_capsule.json"
    assert stdout_path.read_text(encoding="utf-8") == "child-stdout\n"
    assert stderr_path.read_text(encoding="utf-8") == "child-stderr\n"
    assert attempt["artifact_paths"]["stdout_log"] == str(stdout_path)
    assert attempt["artifact_paths"]["stderr_log"] == str(stderr_path)
    assert attempt["artifact_paths"]["evidence_capsule"] == str(evidence_path)

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_schema = json.loads(
        Path("audit/schemas/evidence_capsule.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(evidence, evidence_schema)
    assert evidence["replay"]["inspectable_without_model"] is True
    assert evidence["replay"]["flight_recorder_timeline_path"] == str(
        tmp_path / "events.ndjson"
    )
    assert evidence["replay"]["terminal_transcript_path"] == str(
        tmp_path / "terminal_transcript.log"
    )
    assert evidence["benchmark_metrics"]["decode_tps"] == 10.0
    assert evidence["user_time_ms"] is None
    assert evidence["sys_time_ms"] is None
    assert evidence["max_rss_mb"] is None

    events = [
        json.loads(line)
        for line in (tmp_path / "events.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    event_types = {event["event_type"] for event in events}
    assert "attempt_launch" in event_types
    assert "attempt_payload_parsed" in event_types
    assert "gate_evaluation" in event_types
    assert "artifact_write" in event_types
