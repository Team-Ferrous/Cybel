from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit.coherence.scoring import evaluate_coherence, evaluate_signal_completeness
from audit.evidence_capsule import build_evidence_capsule
from audit.evidence_capsule import extract_compiler_diagnostics
from audit.evidence_capsule import write_evidence_capsule
from audit.runtime_logging import get_active_logger
from audit.runtime_logging import run_logged_subprocess


@dataclass(frozen=True)
class AttemptSpec:
    attempt_id: str
    model: str
    prompt: str
    max_new_tokens: int
    context_length: int
    decode_threads: int | None
    batch_threads: int | None
    ubatch: int | None
    sampling_profile: str | None
    coherence_first: bool
    min_new_tokens_before_eos: int | None
    require_openmp: bool
    require_avx2: bool
    require_mmap: bool
    host_access: str = "user"
    collect_hw_counters: str = "auto"
    require_grover: bool = False
    require_coconut: bool = False
    force_parallel_decode: bool = False
    forbid_autoregressive_fallback: bool = False
    autotune: str = "off"
    warmup: bool = False
    run_index: int = 0
    lane_id: str | None = None
    ablation_id: str | None = None
    feature_toggles: dict[str, Any] | None = None
    dataset_id: str | None = None
    prompt_id: str | None = None
    artifact_paths: dict[str, str] | None = None
    evaluation: dict[str, Any] | None = None
    env_overrides: dict[str, str] | None = None
    artifacts_dir: str | None = None


def _append_optional_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def _default_artifact_dir(repo_root: Path, attempt_id: str) -> Path:
    safe_attempt_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(attempt_id)).strip("._")
    if not safe_attempt_id:
        safe_attempt_id = "attempt"
    return repo_root / ".anvil" / "attempt_artifacts" / safe_attempt_id


_PERF_STAT_EVENT_MAP = {
    "cycles": "pmu_cycles",
    "instructions": "pmu_instructions",
    "cache-references": "pmu_cache_references",
    "cache-misses": "pmu_cache_misses",
    "context-switches": "pmu_context_switches",
    "cpu-migrations": "pmu_cpu_migrations",
    "page-faults": "pmu_page_faults",
}


def _pmu_metric_defaults() -> dict[str, Any]:
    return {
        "pmu_observed": False,
        "pmu_parse_error": "",
        "pmu_cycles": None,
        "pmu_instructions": None,
        "pmu_ipc": None,
        "pmu_cache_references": None,
        "pmu_cache_misses": None,
        "pmu_cache_miss_rate": None,
        "pmu_context_switches": None,
        "pmu_cpu_migrations": None,
        "pmu_page_faults": None,
    }


def _parse_perf_stat_number(raw: object) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_perf_stat_artifact(path: Path) -> dict[str, Any]:
    metrics = _pmu_metric_defaults()
    if not path.exists():
        metrics["pmu_parse_error"] = "perf_stat_artifact_missing"
        return metrics
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as exc:
        metrics["pmu_parse_error"] = f"perf_stat_read_failed:{type(exc).__name__}"
        return metrics
    if not raw.strip():
        metrics["pmu_parse_error"] = "empty_perf_stat_artifact"
        return metrics
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        event_name = parts[1]
        target_key = _PERF_STAT_EVENT_MAP.get(event_name)
        if target_key is None:
            continue
        value = _parse_perf_stat_number(parts[0])
        if value is None:
            continue
        metrics[target_key] = value
        metrics["pmu_observed"] = True
        if event_name == "instructions":
            ipc_match = re.search(r"([0-9][0-9.,]*)\s+insn per cycle", line)
            if ipc_match:
                metrics["pmu_ipc"] = _parse_perf_stat_number(ipc_match.group(1))
        if event_name == "cache-misses":
            miss_rate_match = re.search(r"([0-9][0-9.,]*)%\s+of all cache refs", line)
            if miss_rate_match:
                miss_rate_pct = _parse_perf_stat_number(miss_rate_match.group(1))
                if miss_rate_pct is not None:
                    metrics["pmu_cache_miss_rate"] = miss_rate_pct / 100.0
    cycles = metrics["pmu_cycles"]
    instructions = metrics["pmu_instructions"]
    if metrics["pmu_ipc"] is None and cycles and instructions:
        metrics["pmu_ipc"] = instructions / cycles if cycles > 0.0 else None
    cache_refs = metrics["pmu_cache_references"]
    cache_misses = metrics["pmu_cache_misses"]
    if (
        metrics["pmu_cache_miss_rate"] is None
        and cache_refs
        and cache_misses is not None
        and cache_refs > 0.0
    ):
        metrics["pmu_cache_miss_rate"] = cache_misses / cache_refs
    if not metrics["pmu_observed"] and not metrics["pmu_parse_error"]:
        metrics["pmu_parse_error"] = "perf_stat_counters_unavailable"
    return metrics


def _stream_subprocess_output(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    attempt_id: str,
    lane_id: str | None = None,
    model: str | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return run_logged_subprocess(
        cmd=cmd,
        cwd=cwd,
        env=env,
        source="attempt_executor",
        phase="attempt",
        lane=lane_id,
        attempt_id=attempt_id,
        model=model,
        timeout=1800,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def _extract_json(text: str) -> dict[str, Any] | list[dict[str, Any]]:
    stripped = str(text or "").strip()
    if not stripped:
        raise RuntimeError("benchmark subprocess emitted empty stdout")
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = stripped.find(start_char)
        end = stripped.rfind(end_char)
        if start >= 0 and end >= start:
            payload = stripped[start : end + 1]
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
    raise RuntimeError("benchmark subprocess stdout did not include parseable JSON")


def _extract_flat_record(
    parsed: dict[str, Any] | list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(parsed, list):
        if not parsed:
            raise RuntimeError("benchmark JSON list payload was empty")
        return dict(parsed[0]), {"raw_payload_type": "list"}

    if not isinstance(parsed, dict):
        raise RuntimeError("benchmark JSON payload must be dict or list")

    flat_results = list(parsed.get("flat_results") or [])
    if flat_results:
        row = dict(flat_results[0])
        results = list(parsed.get("results") or [])
        if results and isinstance(results[0], dict):
            structured = dict(results[0])
            sample = dict(structured.get("sample") or {})
            hot_path = dict(structured.get("hot_path") or {})
            if sample:
                row.setdefault("sample_text", sample.get("text"))
                row.setdefault("raw_sample_text", sample.get("raw_text"))
            if hot_path:
                row.setdefault("hot_path_proof", hot_path.get("proof"))
        return row, parsed

    results = list(parsed.get("results") or [])
    if results:
        structured = dict(results[0])
        merged: dict[str, Any] = {}
        for section_name, alias_map in (
            ("identity", {}),
            ("sampling", {}),
            ("throughput", {}),
            ("latency", {}),
            ("runtime", {}),
            ("threading", {}),
            ("hardware", {}),
            ("quality", {}),
            (
                "measurement",
                {
                    "valid": "measurement_valid",
                    "issues": "measurement_issues",
                    "perf_stat_artifact": "perf_stat_artifact",
                    "perf_c2c_artifact": "perf_c2c_artifact",
                    "numa_maps_artifact": "numa_maps_artifact",
                },
            ),
            (
                "hot_path",
                {
                    "proof": "hot_path_proof",
                },
            ),
            (
                "sample",
                {
                    "text": "sample_text",
                    "raw_text": "raw_sample_text",
                },
            ),
            ("architecture", {}),
            ("kernel_efficiency", {"": "kernel_efficiency"}),
            ("timecrystal", {}),
        ):
            section = structured.get(section_name)
            if not isinstance(section, dict):
                continue
            if "" in alias_map:
                merged[alias_map[""]] = dict(section)
            if section_name == "measurement":
                pmu = dict(section.get("pmu") or {})
                if pmu:
                    merged["pmu_observed"] = bool(pmu.get("observed", False))
                    merged["pmu_parse_error"] = pmu.get("parse_error")
                    for key in (
                        "cycles",
                        "instructions",
                        "ipc",
                        "cache_references",
                        "cache_misses",
                        "cache_miss_rate",
                        "context_switches",
                        "cpu_migrations",
                        "page_faults",
                    ):
                        merged[f"pmu_{key}"] = pmu.get(key)
            for key, value in section.items():
                target = alias_map.get(str(key), str(key))
                merged[target] = value
        status = dict(structured.get("status") or {})
        merged["coherence_issues"] = list(merged.get("coherence_issues") or [])
        merged["status_issues"] = list(status.get("issues") or [])
        return merged, parsed

    raise RuntimeError("benchmark JSON payload missing both flat_results and results")


def _build_phase_records(attempt_id: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    simple_phases = {
        "prefill": float(row.get("runtime_prefill_seconds", 0.0)) * 1000.0,
        "decode": float(row.get("runtime_decode_seconds", 0.0)) * 1000.0,
        "graph_prefill": float(row.get("graph_prefill_ms", 0.0)),
        "graph_decode": float(row.get("graph_decode_ms", 0.0)),
        "sampling": float(row.get("sample_ms", 0.0)),
        "logits_processor": float(row.get("logits_processor_ms", 0.0)),
        "penalty": float(row.get("penalty_ms", 0.0)),
        "suppression": float(row.get("suppression_ms", 0.0)),
        "scheduler_queue_wait": float(row.get("scheduler_queue_wait_ms", 0.0)),
        "scheduler_iteration": float(row.get("scheduler_iteration_ms", 0.0)),
    }
    for name, duration_ms in simple_phases.items():
        records.append(
            {
                "attempt_id": attempt_id,
                "phase": name,
                "duration_ms": max(0.0, float(duration_ms)),
                "calls": int(row.get(f"{name}_calls", 0) or 0),
                "extra": {},
            }
        )

    stage_ms = dict(row.get("graph_stage_ms") or {})
    stage_calls = dict(row.get("graph_stage_calls") or {})
    for stage, value in stage_ms.items():
        records.append(
            {
                "attempt_id": attempt_id,
                "phase": f"graph_stage:{stage}",
                "duration_ms": max(0.0, float(value)),
                "calls": int(stage_calls.get(stage, 0) or 0),
                "extra": {
                    "avg_ms": float(
                        dict(row.get("graph_stage_avg_ms") or {}).get(stage, 0.0)
                    ),
                },
            }
        )

    return records


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _evidence_summary(
    *,
    attempt_id: str,
    exit_code: int,
    gate_issues: list[str],
    decode_tps: Any,
    e2e_tps: Any,
    ttft_ms: Any,
) -> str:
    if gate_issues:
        return (
            f"attempt {attempt_id} failed with exit_code={exit_code}; "
            f"gate_issues={len(gate_issues)}"
        )
    return (
        f"attempt {attempt_id} passed with exit_code={exit_code}; "
        f"decode_tps={decode_tps}, e2e_tps={e2e_tps}, ttft_ms={ttft_ms}"
    )


def execute_attempt(
    spec: AttemptSpec,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    started_at = datetime.now(timezone.utc)
    logger = get_active_logger()
    cmd = [
        sys.executable,
        str((repo_root / "benchmarks" / "native_qsg_benchmark.py").resolve()),
        "--isolated-child",
        "--json",
        "--require-measurement-valid",
        "--require-utf8",
        "--model",
        spec.model,
        "--prompt",
        spec.prompt,
        "--max-new-tokens",
        str(int(spec.max_new_tokens)),
        "--context-length",
        str(int(spec.context_length)),
        "--runs",
        "1",
    ]
    _append_optional_arg(cmd, "--sampling-profile", spec.sampling_profile)
    if spec.coherence_first:
        cmd.append("--coherence-first")
    _append_optional_arg(
        cmd,
        "--min-new-tokens-before-eos",
        spec.min_new_tokens_before_eos,
    )
    if spec.require_openmp:
        cmd.append("--require-openmp")
    if spec.require_avx2:
        cmd.append("--require-avx2")
    if spec.require_mmap:
        cmd.append("--require-mmap")
    if spec.require_grover:
        cmd.append("--require-grover")
    if spec.require_coconut:
        cmd.append("--require-coconut")
    _append_optional_arg(cmd, "--host-access", spec.host_access)
    _append_optional_arg(cmd, "--collect-hw-counters", spec.collect_hw_counters)
    _append_optional_arg(cmd, "--autotune", spec.autotune)
    _append_optional_arg(cmd, "--decode-threads", spec.decode_threads)
    _append_optional_arg(cmd, "--batch-threads", spec.batch_threads)
    _append_optional_arg(cmd, "--ubatch", spec.ubatch)
    _append_optional_arg(cmd, "--artifacts-dir", spec.artifacts_dir)

    effective_env_overrides = {
        str(key): str(value)
        for key, value in (spec.env_overrides or {}).items()
        if value is not None
    }
    env = os.environ.copy()
    for key, value in effective_env_overrides.items():
        if value is None:
            continue
        env[str(key)] = str(value)
    if spec.force_parallel_decode:
        env["ANVIL_FORCE_PARALLEL_DECODE"] = "1"
        effective_env_overrides["ANVIL_FORCE_PARALLEL_DECODE"] = "1"
    if spec.forbid_autoregressive_fallback:
        env["ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK"] = "1"
        env["ANVIL_PARALLEL_AR_RECOVERY_ENABLED"] = "0"
        effective_env_overrides["ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK"] = "1"
        effective_env_overrides["ANVIL_PARALLEL_AR_RECOVERY_ENABLED"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    artifact_paths = dict(spec.artifact_paths or {})
    artifact_dir = Path(
        str(
            spec.artifacts_dir
            or artifact_paths.get("attempt_artifact_dir")
            or _default_artifact_dir(repo_root, spec.attempt_id)
        )
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    perf_stat_artifact = (
        artifact_dir / "perf_stat.txt"
        if str(spec.collect_hw_counters).strip().lower() == "required"
        else None
    )
    stdout_path = Path(
        str(artifact_paths.get("stdout_log") or (artifact_dir / "stdout.log"))
    )
    stderr_path = Path(
        str(artifact_paths.get("stderr_log") or (artifact_dir / "stderr.log"))
    )
    if logger is not None:
        logger.emit(
            level="debug",
            source="attempt_executor",
            event_type="attempt_launch",
            message=f"launching subprocess for {spec.attempt_id}",
            phase="attempt",
            lane=str(spec.lane_id or ""),
            attempt_id=spec.attempt_id,
            model=spec.model,
            payload={
                "thread_tuple": (
                    f"{spec.decode_threads or 'auto'}x{spec.batch_threads or 'auto'}x"
                    f"{spec.ubatch or 'auto'}"
                ),
                "perf_stat_artifact": (
                    str(perf_stat_artifact) if perf_stat_artifact is not None else ""
                ),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            },
        )
    wrapped_cmd = list(cmd)
    if perf_stat_artifact is not None:
        wrapped_cmd = [
            "perf",
            "stat",
            "-d",
            "-d",
            "-d",
            "-e",
            (
                "cycles,instructions,cache-references,cache-misses,"
                "context-switches,cpu-migrations,page-faults"
            ),
            "-o",
            str(perf_stat_artifact),
            "--",
            *cmd,
        ]
    completed = _stream_subprocess_output(
        cmd=wrapped_cmd,
        cwd=repo_root,
        env=env,
        attempt_id=spec.attempt_id,
        lane_id=spec.lane_id,
        model=spec.model,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    finished_at = datetime.now(timezone.utc)

    try:
        parsed = _extract_json(completed.stdout)
    except Exception as exc:
        stderr_excerpt = (completed.stderr or "")[-2000:]
        stdout_excerpt = (completed.stdout or "")[-2000:]
        raise RuntimeError(
            "benchmark subprocess did not emit a parseable JSON payload; "
            f"return_code={completed.returncode}, "
            f"stdout_excerpt={stdout_excerpt!r}, stderr_excerpt={stderr_excerpt!r}"
        ) from exc
    if logger is not None:
        logger.emit(
            level="debug",
            source="attempt_executor",
            event_type="attempt_payload_parsed",
            message=f"parsed benchmark payload for {spec.attempt_id}",
            phase="attempt",
            lane=str(spec.lane_id or ""),
            attempt_id=spec.attempt_id,
            model=spec.model,
        )
    row, report = _extract_flat_record(parsed)
    if perf_stat_artifact is not None:
        row["perf_stat_artifact"] = str(perf_stat_artifact)
        row.update(_parse_perf_stat_artifact(perf_stat_artifact))
    prompt_hash = hashlib.sha256(spec.prompt.encode("utf-8")).hexdigest()
    stdout_hash = hashlib.sha256((completed.stdout or "").encode("utf-8")).hexdigest()
    stderr_hash = hashlib.sha256((completed.stderr or "").encode("utf-8")).hexdigest()

    required_fields = (
        "decode_throughput_tps",
        "end_to_end_throughput_tps",
        "ttft_ms",
        "measurement_valid",
        "coherence_valid",
        "native_build_sha256",
        "loaded_native_library",
        "sanctioned_backend_path",
        "tokenizer_backend",
        "backend_module",
        "backend_module_library",
        "backend_module_loaded",
        "full_qsg_enabled",
        "strict_path_stable",
        "native_isa_baseline",
        "native_backend_abi_match",
        "grover_enabled",
        "coconut_enabled",
    )
    missing_signals: list[str] = [
        key
        for key in required_fields
        if row.get(key) is None
        or (isinstance(row.get(key), str) and not str(row.get(key)).strip())
    ]

    coherence = evaluate_coherence(row)
    completeness = evaluate_signal_completeness(row)
    report_failure_count = 0
    benchmark_failures: list[str] = []
    benchmark_failure_keys: list[str] = []
    benchmark_schema_version = ""
    host_report: dict[str, Any] = {}
    if isinstance(report, dict):
        benchmark_schema_version = str(report.get("schema_version") or "")
        host_report = dict(report.get("host") or {})
        report_failure_count = int(report.get("failure_count", 0) or 0)
        benchmark_failure_keys = sorted(
            str(key) for key in (report.get("failure_keys") or []) if str(key).strip()
        )
        failures = report.get("failures") or {}
        if isinstance(failures, dict):
            for _, issues in failures.items():
                if isinstance(issues, list):
                    benchmark_failures.extend(str(issue) for issue in issues if issue)
                elif issues:
                    benchmark_failures.append(str(issues))

    context_stabilizer_enabled = bool(row.get("context_stabilizer_enabled", False))
    context_stabilizer_mode = row.get("context_stabilizer_mode")
    if isinstance(context_stabilizer_mode, str) and not context_stabilizer_mode.strip():
        context_stabilizer_mode = None
    strict_path_stable = bool(row.get("strict_path_stable", False))
    python_hot_path_calls = int(row.get("python_hot_path_calls", 0) or 0)
    numpy_hot_path_calls = int(row.get("numpy_hot_path_calls", 0) or 0)
    python_attention_fallback_calls = int(
        row.get("python_attention_fallback_calls", 0) or 0
    )
    python_ssm_fallback_calls = int(row.get("python_ssm_fallback_calls", 0) or 0)
    python_moe_fallback_calls = int(row.get("python_moe_fallback_calls", 0) or 0)
    llama_cpp_hot_path_calls = int(row.get("llama_cpp_hot_path_calls", 0) or 0)
    batch_token_fallback_count = int(row.get("batch_token_fallback_count", 0) or 0)
    native_backend_abi_match = bool(row.get("native_backend_abi_match", False))
    grover_enabled = bool(row.get("grover_enabled", False))
    coconut_enabled = bool(row.get("coconut_enabled", False))
    perf_event_access = bool(row.get("perf_event_access", False))
    perf_event_access_reason = str(row.get("perf_event_access_reason", "") or "")
    drift_latest = _optional_float(row.get("drift_latest"))
    drift_mean = _optional_float(row.get("drift_mean"))
    drift_max = _optional_float(row.get("drift_max"))
    drift_decay_ratio = _optional_float(row.get("drift_decay_ratio"))
    drift_damped_blocks = _optional_int(row.get("drift_damped_blocks"))
    drift_pruned_blocks = _optional_int(row.get("drift_pruned_blocks"))
    drift_active_tokens = _optional_int(row.get("drift_active_tokens"))
    drift_overhead_percent = _optional_float(row.get("drift_overhead_percent"))
    stabilizer_seconds = _optional_float(row.get("stabilizer_seconds"))
    stabilizer_calls = _optional_int(row.get("stabilizer_calls"))
    drift_auto_downgrade_events = _optional_int(row.get("drift_auto_downgrade_events"))

    gate_issues: list[str] = []
    if not bool(row.get("measurement_valid", False)):
        gate_issues.append("measurement_valid=false")
    if not coherence.get("ok", False):
        gate_issues.extend(
            f"coherence:{issue}" for issue in coherence.get("issues", [])
        )
    if not completeness.get("ok", False):
        gate_issues.extend(
            f"missing_signal:{name}" for name in completeness.get("missing", [])
        )
    if report_failure_count > 0:
        gate_issues.extend(f"benchmark:{issue}" for issue in benchmark_failures)
        if not benchmark_failures:
            gate_issues.append(f"benchmark_failure_count={report_failure_count}")
    if not strict_path_stable:
        gate_issues.append("strict_path_stable=false")
    if not native_backend_abi_match:
        gate_issues.append("native_backend_abi_match=false")
    if spec.require_grover and not grover_enabled:
        gate_issues.append("grover_enabled=false")
    if spec.require_coconut and not coconut_enabled:
        gate_issues.append("coconut_enabled=false")
    if (
        str(spec.host_access).strip().lower() == "privileged"
        and str(spec.collect_hw_counters).strip().lower() == "required"
        and not perf_event_access
    ):
        gate_issues.append(
            f"perf_event_access=false({perf_event_access_reason or 'unknown'})"
        )
    if str(spec.collect_hw_counters).strip().lower() == "required" and not bool(
        row.get("pmu_observed", False)
    ):
        gate_issues.append(
            "pmu_observed=false("
            f"{str(row.get('pmu_parse_error') or 'perf_stat_unobserved')})"
        )
    for name, value in (
        ("python_hot_path_calls", python_hot_path_calls),
        ("numpy_hot_path_calls", numpy_hot_path_calls),
        ("python_attention_fallback_calls", python_attention_fallback_calls),
        ("python_ssm_fallback_calls", python_ssm_fallback_calls),
        ("python_moe_fallback_calls", python_moe_fallback_calls),
        ("llama_cpp_hot_path_calls", llama_cpp_hot_path_calls),
        ("batch_token_fallback_count", batch_token_fallback_count),
    ):
        if value != 0:
            gate_issues.append(f"{name}={value}")
    if context_stabilizer_enabled:
        if context_stabilizer_mode is None:
            gate_issues.append("context_stabilizer_mode_missing")
        if drift_overhead_percent is None:
            gate_issues.append("drift_overhead_percent_missing")
        elif drift_overhead_percent > 20.0:
            gate_issues.append(f"drift_overhead_percent={drift_overhead_percent}>20.0")
        for name, value in (
            ("drift_latest", drift_latest),
            ("drift_mean", drift_mean),
            ("drift_max", drift_max),
            ("drift_decay_ratio", drift_decay_ratio),
            ("drift_damped_blocks", drift_damped_blocks),
            ("drift_pruned_blocks", drift_pruned_blocks),
            ("drift_active_tokens", drift_active_tokens),
            ("stabilizer_seconds", stabilizer_seconds),
            ("stabilizer_calls", stabilizer_calls),
            ("drift_auto_downgrade_events", drift_auto_downgrade_events),
        ):
            if value is None:
                gate_issues.append(f"stabilizer_field_missing:{name}")
    if int(completed.returncode) != 0:
        gate_issues.append(f"subprocess_return_code={int(completed.returncode)}")
    sample_text = str(row.get("sample_text") or "")
    raw_sample_text = str(row.get("raw_sample_text") or "")
    if not sample_text.strip():
        gate_issues.append("empty_sample_text")
    printable_ratio = _optional_float(row.get("printable_ratio"))
    if printable_ratio is not None and printable_ratio < 0.95:
        gate_issues.append(f"printable_ratio={printable_ratio}<0.95")
    repeated_8gram_ratio = _optional_float(row.get("repeated_8gram_ratio"))
    if repeated_8gram_ratio is not None and repeated_8gram_ratio > 0.2:
        gate_issues.append(f"repeated_8gram_ratio={repeated_8gram_ratio}>0.2")
    if not bool(row.get("utf8_valid", True)):
        gate_issues.append("utf8_valid=false")
    if bool(row.get("leaked_control_text", False)):
        gate_issues.append("leaked_control_text=true")
    if bool(row.get("leaked_think_tags", False)):
        gate_issues.append("leaked_think_tags=true")
    strict_hot_path_required = any(
        (
            bool(spec.require_openmp),
            bool(spec.require_avx2),
            bool(spec.require_mmap),
            bool(spec.require_grover),
            bool(spec.require_coconut),
        )
    )
    hot_path_proof_raw = row.get("hot_path_proof")
    hot_path_proof = (
        dict(hot_path_proof_raw) if isinstance(hot_path_proof_raw, dict) else {}
    )
    if strict_hot_path_required:
        for key in ("sanctioned_backend_path", "tokenizer_backend"):
            if not str(hot_path_proof.get(key) or "").strip():
                gate_issues.append(f"hot_path_proof_missing:{key}")
    gate_issues = sorted(set(gate_issues))
    if logger is not None:
        logger.emit(
            level="warn" if gate_issues else "info",
            source="attempt_executor",
            event_type="gate_evaluation",
            message=f"gate evaluation {'failed' if gate_issues else 'passed'} for {spec.attempt_id}",
            phase="attempt",
            lane=str(spec.lane_id or ""),
            attempt_id=spec.attempt_id,
            model=spec.model,
            payload={
                "issue_count": len(gate_issues),
                "issues": gate_issues,
                "decode_tps": row.get("decode_throughput_tps"),
                "e2e_tps": row.get("end_to_end_throughput_tps"),
                "ttft_ms": row.get("ttft_ms"),
            },
        )

    attempt_record = {
        "attempt_id": spec.attempt_id,
        "model_id": spec.model,
        "run_index": int(spec.run_index),
        "warmup": bool(spec.warmup),
        "lane_id": str(spec.lane_id or "").strip() or None,
        "ablation_id": str(spec.ablation_id or "").strip() or None,
        "prompt_hash": prompt_hash,
        "seed": int(row.get("seed") or row.get("runtime_seed") or 0),
        "thread_config": {
            "decode_threads": spec.decode_threads,
            "batch_threads": spec.batch_threads,
            "ubatch": spec.ubatch,
            "runtime_decode_threads": row.get("decode_threads"),
            "runtime_batch_threads": row.get("batch_threads"),
            "runtime_ubatch": row.get("ubatch"),
        },
        "host": {
            "affinity_visible_threads": _optional_int(
                host_report.get("affinity_visible_threads")
            ),
            "runtime_affinity_visible_threads": _optional_int(
                host_report.get("runtime_affinity_visible_threads")
            ),
            "launch_affinity_cpus": [
                int(cpu)
                for cpu in list(host_report.get("launch_affinity_cpus") or [])
                if _optional_int(cpu) is not None
            ],
            "runtime_affinity_cpus": [
                int(cpu)
                for cpu in list(host_report.get("runtime_affinity_cpus") or [])
                if _optional_int(cpu) is not None
            ],
            "physical_core_count": _optional_int(row.get("physical_core_count")),
            "logical_core_count": _optional_int(row.get("logical_core_count")),
            "p_core_count": _optional_int(row.get("p_core_count")),
            "omp_max_threads": _optional_int(row.get("omp_max_threads")),
            "worker_cpu_mask": str(row.get("worker_cpu_mask", "") or ""),
            "orchestrator_cpu_mask": str(row.get("orchestrator_cpu_mask", "") or ""),
            "affinity_policy": str(row.get("affinity_policy", "") or ""),
            "cpu_governor": str(row.get("cpu_governor", "") or ""),
            "thp_mode": str(row.get("thp_mode", "") or ""),
        },
        "sampling_config": {
            "profile": row.get("sampling_profile"),
            "temperature": row.get("temperature"),
            "top_p": row.get("top_p"),
            "top_k": row.get("top_k"),
            "min_p": row.get("min_p"),
            "presence_penalty": row.get("presence_penalty"),
            "repetition_penalty": row.get("repetition_penalty"),
        },
        "throughput": {
            "wall_tokens_per_second": float(row.get("tokens_per_second", 0.0) or 0.0),
            "prefill_tps_raw": float(row.get("prefill_throughput_tps", 0.0) or 0.0),
            "prefill_tps": float(
                row.get("effective_prefill_throughput_tps", 0.0) or 0.0
            ),
            "effective_prefill_tps": float(
                row.get("effective_prefill_throughput_tps", 0.0) or 0.0
            ),
            "decode_tps": float(row.get("decode_throughput_tps", 0.0) or 0.0),
            "e2e_tps": float(row.get("end_to_end_throughput_tps", 0.0) or 0.0),
        },
        "latency": {
            "ttft_ms": float(row.get("ttft_ms", 0.0) or 0.0),
            "p10_ms": float(row.get("per_token_latency_p10_ms", 0.0) or 0.0),
            "p25_ms": float(row.get("per_token_latency_p25_ms", 0.0) or 0.0),
            "p50_ms": float(row.get("per_token_latency_p50_ms", 0.0) or 0.0),
            "p75_ms": float(row.get("per_token_latency_p75_ms", 0.0) or 0.0),
            "p95_ms": float(row.get("per_token_latency_p95_ms", 0.0) or 0.0),
            "p99_ms": float(row.get("per_token_latency_p99_ms", 0.0) or 0.0),
            "stddev_ms": float(row.get("per_token_latency_stddev_ms", 0.0) or 0.0),
            "min_ms": float(row.get("per_token_latency_min_ms", 0.0) or 0.0),
            "max_ms": float(row.get("per_token_latency_max_ms", 0.0) or 0.0),
        },
        "coherence": {
            "ok": bool(coherence.get("ok", False)),
            "issues": list(coherence.get("issues", [])),
            "raw_ok": bool(row.get("raw_coherence_valid", True)),
            "raw_issues": list(row.get("raw_coherence_issues") or []),
        },
        "sample": {
            "text": sample_text,
            "raw_text": raw_sample_text,
            "printable_ratio": printable_ratio,
            "repeated_8gram_ratio": repeated_8gram_ratio,
        },
        "runtime": {
            "runtime_total_seconds": float(
                row.get("runtime_total_seconds", 0.0) or 0.0
            ),
            "runtime_prefill_seconds": float(
                row.get("runtime_prefill_seconds", 0.0) or 0.0
            ),
            "runtime_decode_seconds": float(
                row.get("runtime_decode_seconds", 0.0) or 0.0
            ),
            "os_thread_migrations": int(row.get("os_thread_migrations", 0) or 0),
            "os_last_cpu": int(row.get("os_last_cpu", -1) or -1),
            "numa_strict": bool(row.get("numa_strict", False)),
            "numa_affinity_mode": str(row.get("numa_affinity_mode", "") or ""),
            "numa_hugepage": str(row.get("numa_hugepage", "") or ""),
            "numa_bind_policy": str(row.get("numa_bind_policy", "") or ""),
            "numa_first_touch": bool(row.get("numa_first_touch", False)),
            "graph_stage_ms": dict(row.get("graph_stage_ms") or {}),
            "graph_stage_calls": dict(row.get("graph_stage_calls") or {}),
            "graph_prefill_avg_ms": float(row.get("graph_prefill_avg_ms", 0.0) or 0.0),
            "graph_decode_avg_ms": float(row.get("graph_decode_avg_ms", 0.0) or 0.0),
            "sample_avg_ms": float(row.get("sample_avg_ms", 0.0) or 0.0),
            "logits_processor_avg_ms": float(
                row.get("logits_processor_avg_ms", 0.0) or 0.0
            ),
            "penalty_avg_ms": float(row.get("penalty_avg_ms", 0.0) or 0.0),
            "suppression_avg_ms": float(row.get("suppression_avg_ms", 0.0) or 0.0),
            "scheduler_queue_wait_ms": float(
                row.get("scheduler_queue_wait_ms", 0.0) or 0.0
            ),
            "scheduler_iteration_ms": float(
                row.get("scheduler_iteration_ms", 0.0) or 0.0
            ),
            "context_stabilizer_enabled": context_stabilizer_enabled,
            "context_stabilizer_mode": context_stabilizer_mode,
            "strict_path_stable": strict_path_stable,
            "native_backend_abi_match": native_backend_abi_match,
            "grover_enabled": grover_enabled,
            "coconut_enabled": coconut_enabled,
            "perf_event_access": perf_event_access,
            "perf_event_access_reason": perf_event_access_reason,
            "python_hot_path_calls": python_hot_path_calls,
            "numpy_hot_path_calls": numpy_hot_path_calls,
            "python_attention_fallback_calls": python_attention_fallback_calls,
            "python_ssm_fallback_calls": python_ssm_fallback_calls,
            "python_moe_fallback_calls": python_moe_fallback_calls,
            "llama_cpp_hot_path_calls": llama_cpp_hot_path_calls,
            "batch_token_fallback_count": batch_token_fallback_count,
            "hot_path_proof": hot_path_proof,
            "drift_latest": drift_latest,
            "drift_mean": drift_mean,
            "drift_max": drift_max,
            "drift_decay_ratio": drift_decay_ratio,
            "drift_damped_blocks": drift_damped_blocks,
            "drift_pruned_blocks": drift_pruned_blocks,
            "drift_active_tokens": drift_active_tokens,
            "drift_overhead_percent": drift_overhead_percent,
            "stabilizer_seconds": stabilizer_seconds,
            "stabilizer_calls": stabilizer_calls,
            "drift_auto_downgrade_events": drift_auto_downgrade_events,
        },
        "architecture": {
            "hidden_dim": int(row.get("arch_hidden_dim", 0) or 0),
            "num_layers": int(row.get("arch_num_layers", 0) or 0),
            "num_heads": int(row.get("arch_num_heads", 0) or 0),
            "num_kv_heads": int(row.get("arch_num_kv_heads", 0) or 0),
            "head_dim": int(row.get("arch_head_dim", 0) or 0),
            "intermediate_dim": int(row.get("arch_intermediate_dim", 0) or 0),
            "vocab_size": int(row.get("arch_vocab_size", 0) or 0),
            "num_experts": int(row.get("arch_num_experts", 0) or 0),
            "top_k_experts": int(row.get("arch_top_k_experts", 0) or 0),
            "weight_qtype": str(row.get("arch_weight_qtype", "") or ""),
        },
        "kernel_efficiency": dict(row.get("kernel_efficiency") or {}),
        "feature_toggles": dict(spec.feature_toggles or {}),
        "dataset_id": str(spec.dataset_id or "").strip() or None,
        "prompt_id": str(spec.prompt_id or "").strip() or None,
        "artifact_paths": {
            **dict(spec.artifact_paths or {}),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "perf_stat_artifact": str(row.get("perf_stat_artifact", "") or ""),
            "perf_c2c_artifact": str(row.get("perf_c2c_artifact", "") or ""),
            "numa_maps_artifact": str(row.get("numa_maps_artifact", "") or ""),
        },
        "evaluation": dict(spec.evaluation or {}),
        "parallel_decode": bool(row.get("parallel_decode", False)),
        "speculative_decode": bool(row.get("speculative_decode", False)),
        "generation_mode": str(row.get("generation_mode", "") or ""),
        "benchmark_label": str(row.get("benchmark_label", "") or ""),
        "prompt_category": str(row.get("prompt_category", "") or ""),
        "temperature_band": str(row.get("temperature_band", "") or ""),
        "accepted_parallel_tokens": int(row.get("accepted_parallel_tokens", 0) or 0),
        "rejected_parallel_tokens": int(row.get("rejected_parallel_tokens", 0) or 0),
        "draft_source": str(row.get("draft_source", "") or ""),
        "blockwise_convergence_rate": float(
            row.get("blockwise_convergence_rate", 0.0) or 0.0
        ),
        "quality_guard_triggered": bool(row.get("quality_guard_triggered", False)),
        "provenance": {
            "native_build_id": row.get("native_build_id"),
            "native_build_sha256": row.get("native_build_sha256"),
            "loaded_native_library": row.get("loaded_native_library"),
            "sanctioned_backend_path": row.get("sanctioned_backend_path"),
            "tokenizer_backend": row.get("tokenizer_backend"),
            "backend_module": row.get("backend_module"),
            "backend_module_library": row.get("backend_module_library"),
            "backend_module_loaded": bool(row.get("backend_module_loaded", False)),
            "backend_module_marker_symbol": row.get("backend_module_marker_symbol"),
            "backend_module_marker": int(row.get("backend_module_marker", 0) or 0),
            "full_qsg_enabled": bool(row.get("full_qsg_enabled", False)),
            "native_isa_baseline": str(row.get("native_isa_baseline", "") or ""),
            "native_backend_abi_match": native_backend_abi_match,
            "perf_event_access": perf_event_access,
            "perf_event_access_reason": perf_event_access_reason,
            "host_access": spec.host_access,
            "collect_hw_counters": spec.collect_hw_counters,
            "require_grover": spec.require_grover,
            "require_coconut": spec.require_coconut,
            "autotune": spec.autotune,
            "cwd": str(repo_root),
            "argv": list(cmd),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": int(
                max(
                    0.0,
                    (finished_at - started_at).total_seconds() * 1000.0,
                )
            ),
            "stdout_sha256": f"sha256:{stdout_hash}",
            "stderr_sha256": f"sha256:{stderr_hash}",
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "perf_stat_artifact": str(row.get("perf_stat_artifact", "") or ""),
            "perf_c2c_artifact": str(row.get("perf_c2c_artifact", "") or ""),
            "numa_maps_artifact": str(row.get("numa_maps_artifact", "") or ""),
            "benchmark_schema_version": benchmark_schema_version,
            "report_failure_count": report_failure_count,
            "report_failure_keys": benchmark_failure_keys,
            "env_overrides": dict(effective_env_overrides),
        },
        "measurement": {
            "valid": bool(row.get("measurement_valid", False)),
            "issues": list(row.get("measurement_issues") or []),
            "missing_signals": missing_signals,
            "perf_stat_artifact": str(row.get("perf_stat_artifact", "") or ""),
            "pmu": {
                "observed": bool(row.get("pmu_observed", False)),
                "parse_error": str(row.get("pmu_parse_error", "") or ""),
                "cycles": _optional_float(row.get("pmu_cycles")),
                "instructions": _optional_float(row.get("pmu_instructions")),
                "ipc": _optional_float(row.get("pmu_ipc")),
                "cache_references": _optional_float(row.get("pmu_cache_references")),
                "cache_misses": _optional_float(row.get("pmu_cache_misses")),
                "cache_miss_rate": _optional_float(row.get("pmu_cache_miss_rate")),
                "context_switches": _optional_float(row.get("pmu_context_switches")),
                "cpu_migrations": _optional_float(row.get("pmu_cpu_migrations")),
                "page_faults": _optional_float(row.get("pmu_page_faults")),
            },
        },
        "status": {
            "return_code": int(completed.returncode),
            "ok": int(completed.returncode) == 0 and not gate_issues,
            "issues": gate_issues,
        },
    }

    subprocess_metrics = dict(getattr(completed, "anvil_subprocess_metrics", {}) or {})
    base_evidence = dict(getattr(completed, "anvil_evidence_capsule", {}) or {})
    base_artifact_paths = dict(base_evidence.get("artifact_paths") or {})
    evidence_path = Path(
        str(
            attempt_record["artifact_paths"].get("evidence_capsule")
            or base_artifact_paths.get("evidence_capsule")
            or (artifact_dir / "evidence_capsule.json")
        )
    )
    attempt_record["artifact_paths"]["evidence_capsule"] = str(evidence_path)
    if base_artifact_paths.get("flight_recorder_timeline") and not attempt_record[
        "artifact_paths"
    ].get("flight_recorder_timeline"):
        attempt_record["artifact_paths"]["flight_recorder_timeline"] = str(
            base_artifact_paths["flight_recorder_timeline"]
        )
    if base_artifact_paths.get("terminal_transcript") and not attempt_record[
        "artifact_paths"
    ].get("terminal_transcript"):
        attempt_record["artifact_paths"]["terminal_transcript"] = str(
            base_artifact_paths["terminal_transcript"]
        )
    if logger is not None and not attempt_record["artifact_paths"].get(
        "flight_recorder_timeline"
    ):
        attempt_record["artifact_paths"]["flight_recorder_timeline"] = str(
            logger.events_path
        )
    if logger is not None and not attempt_record["artifact_paths"].get(
        "terminal_transcript"
    ):
        attempt_record["artifact_paths"]["terminal_transcript"] = str(
            logger.transcript_path
        )

    benchmark_metrics = {
        "wall_tokens_per_second": _optional_float(row.get("tokens_per_second")),
        "prefill_tps": _optional_float(row.get("prefill_throughput_tps")),
        "effective_prefill_tps": _optional_float(
            row.get("effective_prefill_throughput_tps")
        ),
        "decode_tps": _optional_float(row.get("decode_throughput_tps")),
        "e2e_tps": _optional_float(row.get("end_to_end_throughput_tps")),
        "ttft_ms": _optional_float(row.get("ttft_ms")),
        "queue_wait_ms": _optional_float(row.get("scheduler_queue_wait_ms")),
        "scheduler_iteration_ms": _optional_float(row.get("scheduler_iteration_ms")),
        "per_token_latency_p95_ms": _optional_float(
            row.get("per_token_latency_p95_ms")
        ),
        "accepted_parallel_tokens": _optional_float(
            row.get("accepted_parallel_tokens")
        ),
        "proposed_parallel_tokens": _optional_float(
            row.get("proposed_parallel_tokens")
        ),
        "measurement_valid": bool(row.get("measurement_valid", False)),
        "measurement_valid_marker": (
            "measurement_valid"
            if bool(row.get("measurement_valid", False))
            else "measurement_invalid"
        ),
        "coherence_ok": bool(coherence.get("ok", False)),
        "coherence_ok_marker": (
            "coherence_ok" if bool(coherence.get("ok", False)) else "coherence_failed"
        ),
        "strict_path_stable": strict_path_stable,
        "strict_path_stable_marker": (
            "strict_path_stable" if strict_path_stable else "strict_path_unstable"
        ),
        "gate_issue_count": float(len(gate_issues)),
        "benchmark_schema_version": benchmark_schema_version,
        "report_failure_count": float(report_failure_count),
    }
    evidence_capsule = build_evidence_capsule(
        sequence_id=int(subprocess_metrics.get("sequence_id") or 1),
        tool_run_id=str(
            subprocess_metrics.get("tool_run_id")
            or f"{spec.attempt_id}:benchmark_subprocess"
        ),
        source="attempt_executor",
        command=[str(part) for part in cmd],
        cwd=str(repo_root),
        exit_code=int(completed.returncode),
        wall_time_ms=float(
            subprocess_metrics.get("wall_time_ms")
            or max(0.0, (finished_at - started_at).total_seconds() * 1000.0)
        ),
        user_time_ms=_optional_float(subprocess_metrics.get("user_time_ms")),
        sys_time_ms=_optional_float(subprocess_metrics.get("sys_time_ms")),
        max_rss_mb=_optional_float(subprocess_metrics.get("max_rss_mb")),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        artifact_paths={
            str(key): str(value)
            for key, value in dict(attempt_record["artifact_paths"]).items()
            if str(value).strip()
        },
        failing_tests=[str(item) for item in benchmark_failure_keys],
        compiler_diagnostics=extract_compiler_diagnostics(completed.stderr or ""),
        benchmark_metrics=benchmark_metrics,
        summary=_evidence_summary(
            attempt_id=spec.attempt_id,
            exit_code=int(completed.returncode),
            gate_issues=gate_issues,
            decode_tps=row.get("decode_throughput_tps"),
            e2e_tps=row.get("end_to_end_throughput_tps"),
            ttft_ms=row.get("ttft_ms"),
        ),
        replay={
            "checkpoint_metadata_path": attempt_record["artifact_paths"].get(
                "checkpoint_metadata"
            ),
            "flight_recorder_timeline_path": attempt_record["artifact_paths"].get(
                "flight_recorder_timeline"
            ),
            "terminal_transcript_path": attempt_record["artifact_paths"].get(
                "terminal_transcript"
            ),
            "inspectable_without_model": True,
        },
        stdout_text=completed.stdout or "",
        stderr_text=completed.stderr or "",
    )
    write_evidence_capsule(evidence_path, evidence_capsule)
    if logger is not None:
        logger.emit_artifact(
            source="attempt_executor",
            kind="evidence_capsule",
            path=evidence_path,
            summary=f"persisted evidence capsule for {spec.attempt_id}",
            phase="attempt",
            lane=str(spec.lane_id or ""),
            attempt_id=spec.attempt_id,
            model=spec.model,
            level="debug" if attempt_record["status"]["ok"] else "warn",
        )

    return attempt_record, _build_phase_records(spec.attempt_id, row), report
