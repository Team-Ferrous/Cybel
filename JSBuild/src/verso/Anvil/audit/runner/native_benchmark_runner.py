from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from audit.provenance.capture import (
    capture_model_provenance,
    capture_runtime_provenance,
    host_fingerprint,
)
from audit.runner.common import load_optional_json_dict, read_json_dict
from audit.runner.attempt_executor import AttemptSpec, execute_attempt
from audit.store.layout import RunLayout, ensure_layout, resolve_layout
from audit.store.schema_validation import validate_payload
from audit.store.writer import (
    append_ndjson,
    read_ndjson,
    write_json_atomic,
    write_ndjson_atomic,
)


SCHEMA_VERSION = "native_qsg_audit.v2"
KERNEL_MAP_VERSION = "1.0"
DECODE_ACCOUNTING_MIN_PCT = 95.0
DECODE_ACCOUNTING_MAX_PCT = 105.0
ROOT_STAGE_HOTSPOT_LIMIT = 25

KERNEL_MAP: dict[str, dict[str, str | int]] = {
    "embedding_lookup": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "graph_forward_token_impl (inline embedding copy)",
        "kernel_class": "memory_copy",
        "weight_format": "fp32/fp16",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(hidden_dim)",
    },
    "attention_proj": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "dot_rows_q6k / simd_fused_qkv_matvec_quant",
        "kernel_class": "quantized_matvec",
        "weight_format": "Q6_K / Q6_K_R4",
        "block_bytes": 210,
        "elements_per_block": 256,
        "complexity": "O(hidden_dim * (q_heads + 2*kv_heads) * head_dim) per layer",
    },
    "attention_rope_kv": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "apply_rope_with_dim",
        "kernel_class": "elementwise_trig",
        "weight_format": "fp32",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(num_heads * head_dim) per layer",
    },
    "attention_decode": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "fused_gqa_attention_decode_paged",
        "kernel_class": "attention_score_softmax",
        "weight_format": "fp32 (KV cache)",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(num_heads * kv_len * head_dim) per layer",
    },
    "attention_out_proj": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "dot_rows_q6k",
        "kernel_class": "quantized_matvec",
        "weight_format": "Q6_K",
        "block_bytes": 210,
        "elements_per_block": 256,
        "complexity": "O(num_heads * head_dim * hidden_dim) per layer",
    },
    "ffn_gate_up": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "dot_rows_q6k / dot_rows_q6k_r4",
        "kernel_class": "quantized_matvec",
        "weight_format": "Q6_K / Q6_K_R4",
        "block_bytes": 210,
        "elements_per_block": 256,
        "complexity": "O(hidden_dim * 2 * intermediate_dim) per layer",
    },
    "ffn_down": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "dot_rows_q6k",
        "kernel_class": "quantized_matvec",
        "weight_format": "Q6_K",
        "block_bytes": 210,
        "elements_per_block": 256,
        "complexity": "O(intermediate_dim * hidden_dim) per layer",
    },
    "ffn_norm": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "rmsnorm_inplace",
        "kernel_class": "elementwise_norm",
        "weight_format": "fp32",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(hidden_dim) per layer",
    },
    "ssm": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "ssm_mamba2_step (composite)",
        "kernel_class": "ssm_recurrent",
        "weight_format": "fp32 (state)",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(ssm_state_dim * ssm_head_dim * num_ssm_heads) per layer",
    },
    "ssm_projection": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "dot_rows_q6k",
        "kernel_class": "quantized_matvec",
        "weight_format": "Q6_K",
        "block_bytes": 210,
        "elements_per_block": 256,
        "complexity": "O(hidden_dim * ssm_proj_dim) per layer",
    },
    "ssm_conv": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "ssm_conv_step",
        "kernel_class": "1d_convolution",
        "weight_format": "fp32",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(ssm_conv_dim * conv_kernel_size) per layer",
    },
    "ssm_recurrent": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "ssm_mamba2_step inner recurrence",
        "kernel_class": "diagonal_ssm_update",
        "weight_format": "fp32",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(ssm_state_dim * ssm_head_dim) per layer",
    },
    "ssm_output": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "dot_rows_q6k",
        "kernel_class": "quantized_matvec",
        "weight_format": "Q6_K",
        "block_bytes": 210,
        "elements_per_block": 256,
        "complexity": "O(ssm_out_dim * hidden_dim) per layer",
    },
    "moe": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "simd_fused_moe_ffn + granite_router_top_k_streaming",
        "kernel_class": "expert_dispatch_swiglu",
        "weight_format": "Q4_K / Q6_K",
        "block_bytes": 144,
        "elements_per_block": 256,
        "complexity": "O(top_k * (hidden_dim * intermediate_dim * 3)) per layer",
    },
    "lm_head": {
        "cpp_file": "core/native/quantized_matmul.cpp",
        "cpp_function": "dot_row_q6k_lm",
        "kernel_class": "quantized_matvec_lm",
        "weight_format": "Q6_K_LM (expanded, 276 bytes/block)",
        "block_bytes": 276,
        "elements_per_block": 256,
        "complexity": "O(hidden_dim * vocab_size) per token",
    },
    "sanitize": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "sanitize_output_inplace",
        "kernel_class": "elementwise_clamp",
        "weight_format": "fp32",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(hidden_dim)",
    },
    "final_norm": {
        "cpp_file": "core/native/model_graph.cpp",
        "cpp_function": "rmsnorm_inplace",
        "kernel_class": "elementwise_norm",
        "weight_format": "fp32",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(hidden_dim)",
    },
    "sampling": {
        "cpp_file": "core/native/simd_ops.cpp",
        "cpp_function": "simd_sample_token_f32",
        "kernel_class": "sampling",
        "weight_format": "fp32 (logits)",
        "block_bytes": 0,
        "elements_per_block": 0,
        "complexity": "O(vocab_size)",
    },
}


def _parse_int_list(raw: str | None) -> list[int | None]:
    if raw is None or not str(raw).strip():
        return [None]
    values: list[int | None] = []
    for part in str(raw).split(","):
        token = part.strip().lower()
        if not token or token in {"none", "null", "auto"}:
            values.append(None)
            continue
        values.append(int(token))
    return values or [None]


def _thread_matrix(args: argparse.Namespace) -> list[tuple[int | None, int | None, int | None]]:
    if not bool(getattr(args, "thread_sweep", False)):
        return [
            (
                getattr(args, "decode_threads", None),
                getattr(args, "batch_threads", None),
                getattr(args, "ubatch", None),
            )
        ]
    decodes = _parse_int_list(getattr(args, "decode_thread_sweep", None))
    batches = _parse_int_list(getattr(args, "batch_thread_sweep", None))
    ubatches = _parse_int_list(getattr(args, "ubatch_sweep", None))
    matrix: list[tuple[int | None, int | None, int | None]] = []
    for d in decodes:
        for b in batches:
            for u in ubatches:
                matrix.append((d, b, u))
    return matrix


def _normalize_models(raw_models: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in raw_models:
        model = str(raw).strip()
        if not model or model in seen:
            continue
        seen.add(model)
        ordered.append(model)
    if not ordered:
        raise RuntimeError("At least one non-empty model id is required.")
    return ordered


def _thread_matrix_payload(
    matrix: list[tuple[int | None, int | None, int | None]],
) -> list[dict[str, int | None]]:
    return [
        {"decode_threads": d, "batch_threads": b, "ubatch": u}
        for d, b, u in matrix
    ]


def _benchmark_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": int(args.max_new_tokens),
        "context_length": int(args.context_length),
        "sampling_profile": str(getattr(args, "sampling_profile", "") or ""),
        "audit_surface": str(getattr(args, "audit_surface", "both")),
        "coherence_first": bool(getattr(args, "coherence_first", True)),
        "min_new_tokens_before_eos": getattr(args, "min_new_tokens_before_eos", 12),
        "require_openmp": bool(getattr(args, "require_openmp", True)),
        "require_avx2": bool(getattr(args, "require_avx2", True)),
        "require_mmap": bool(getattr(args, "require_mmap", False)),
        "host_access": str(getattr(args, "host_access", "user")),
        "collect_hw_counters": str(getattr(args, "collect_hw_counters", "auto")),
        "require_grover": bool(getattr(args, "require_grover", False)),
        "require_coconut": bool(getattr(args, "require_coconut", False)),
        "autotune": str(getattr(args, "autotune", "off")),
    }


def _run_continuous_surface(
    *,
    repo_root: Path,
    layout: RunLayout,
    models: list[str],
    max_new_tokens: int,
) -> list[str]:
    artifacts: list[str] = []
    for model in models:
        safe_model = str(model).replace(":", "_").replace("/", "_")
        output_path = layout.root / f"continuous_{safe_model}.json"
        cmd = [
            sys.executable,
            str((repo_root / "benchmarks" / "continuous_qsg_benchmark.py").resolve()),
            "--model",
            model,
            "--max-new-tokens",
            str(int(max_new_tokens)),
            "--json-out",
            str(output_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Continuous QSG benchmark failed for "
                f"{model}: stdout={completed.stdout[-1000:]!r} stderr={completed.stderr[-1000:]!r}"
            )
        artifacts.append(str(output_path))
    return artifacts


def _attempt_plan(
    *,
    models: list[str],
    matrix: list[tuple[int | None, int | None, int | None]],
    warmup_runs: int,
    measured_runs: int,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        for combo_index, (decode_threads, batch_threads, ubatch) in enumerate(matrix):
            for warmup, runs in ((True, int(warmup_runs)), (False, int(measured_runs))):
                for run_index in range(runs):
                    plan.append(
                        {
                            "model_index": model_index,
                            "model": model,
                            "combo_index": combo_index,
                            "decode_threads": decode_threads,
                            "batch_threads": batch_threads,
                            "ubatch": ubatch,
                            "warmup": warmup,
                            "run_index": run_index,
                        }
                    )
    return plan


def _attempt_id(
    model_index: int,
    combo_index: int,
    warmup: bool,
    run_index: int,
    model: str,
) -> str:
    safe_model = str(model).replace(":", "_").replace("/", "_")
    stage = "warmup" if warmup else "measure"
    return f"m{model_index:02d}-c{combo_index:02d}-{stage}-r{run_index:02d}-{safe_model}"


_ATTEMPT_ID_RE = re.compile(
    r"^m(?P<model>\d+)-c(?P<combo>\d+)-(?P<stage>warmup|measure)-r(?P<run>\d+)-"
)


def _attempt_id_order_key(attempt_id: str) -> tuple[int, int, int, int, str]:
    text = str(attempt_id or "").strip()
    match = _ATTEMPT_ID_RE.match(text)
    if not match:
        return (10_000, 10_000, 10_000, 10_000, text)
    stage = 0 if match.group("stage") == "warmup" else 1
    return (
        int(match.group("model")),
        int(match.group("combo")),
        stage,
        int(match.group("run")),
        text,
    )


def _last_attempt_id_by_sequence(completed_attempt_ids: set[str]) -> str | None:
    if not completed_attempt_ids:
        return None
    return max(completed_attempt_ids, key=_attempt_id_order_key)


def _load_checkpoint(layout: RunLayout) -> dict[str, Any]:
    if not layout.checkpoint_json.exists():
        return {
            "run_id": layout.run_id,
            "completed_attempt_ids": [],
            "last_attempt_id": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    payload = read_json_dict(layout.checkpoint_json)
    validate_payload("checkpoint.schema.json", payload)
    return payload


def _write_json_validated(path: Path, schema_name: str, payload: dict[str, Any]) -> None:
    validate_payload(schema_name, payload)
    write_json_atomic(path, payload)


def _append_ndjson_validated(path: Path, schema_name: str, payload: dict[str, Any]) -> None:
    validate_payload(schema_name, payload)
    append_ndjson(path, payload)


def _rewrite_ndjson_validated(
    path: Path,
    schema_name: str,
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        validate_payload(schema_name, row)
    write_ndjson_atomic(path, rows)


def _checkpoint_payload(
    *,
    run_id: str,
    completed_attempt_ids: set[str],
    last_attempt_id: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "completed_attempt_ids": sorted(str(item) for item in completed_attempt_ids),
        "last_attempt_id": str(last_attempt_id) if last_attempt_id else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_checkpoint(
    layout: RunLayout,
    *,
    completed_attempt_ids: set[str],
    last_attempt_id: str | None,
) -> None:
    _write_json_validated(
        layout.checkpoint_json,
        "checkpoint.schema.json",
        _checkpoint_payload(
            run_id=layout.run_id,
            completed_attempt_ids=completed_attempt_ids,
            last_attempt_id=last_attempt_id,
        ),
    )


def _attempt_id_from_row(row: dict[str, Any]) -> str:
    return str(row.get("attempt_id") or "").strip()


def _dedupe_attempt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_attempt: dict[str, dict[str, Any]] = {}
    for row in rows:
        attempt_id = _attempt_id_from_row(row)
        if not attempt_id:
            continue
        by_attempt[attempt_id] = dict(row)
    return [by_attempt[attempt_id] for attempt_id in sorted(by_attempt)]


def _dedupe_phase_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        attempt_id = _attempt_id_from_row(row)
        phase = str(row.get("phase") or "").strip()
        if not attempt_id or not phase:
            continue
        by_key[(attempt_id, phase)] = dict(row)
    return [by_key[key] for key in sorted(by_key)]


def _normalize_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_attempt: dict[str, dict[str, Any]] = {}
    for row in rows:
        attempt_id = _attempt_id_from_row(row)
        if not attempt_id:
            continue
        normalized = dict(row)
        gate_issues = sorted(
            {str(issue) for issue in normalized.get("gate_issues") or [] if str(issue).strip()}
        )
        normalized_issues = sorted(
            {
                str(issue)
                for issue in normalized.get("normalized_issues") or []
                if str(issue).strip()
            }
        )
        if not normalized_issues:
            normalized_issues = list(gate_issues)
            error_type = str(normalized.get("error_type") or "").strip()
            if error_type:
                normalized_issues.append(f"error_type:{error_type}")
            normalized_issues = sorted(set(normalized_issues))

        failure_kind = str(normalized.get("failure_kind") or "").strip().lower()
        if failure_kind not in {"gate_failure", "execution_failure"}:
            failure_kind = "execution_failure"

        normalized["attempt_id"] = attempt_id
        normalized["gate_issues"] = gate_issues
        normalized["normalized_issues"] = normalized_issues
        normalized["failure_kind"] = failure_kind
        by_attempt[attempt_id] = normalized

    return [by_attempt[attempt_id] for attempt_id in sorted(by_attempt)]


def _reconcile_resume_state(
    layout: RunLayout,
    checkpoint: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    checkpoint_completed = {
        str(item)
        for item in (checkpoint.get("completed_attempt_ids") or [])
        if str(item).strip()
    }

    attempts = _dedupe_attempt_rows(read_ndjson(layout.attempts_ndjson))
    phases = _dedupe_phase_rows(read_ndjson(layout.phases_ndjson))
    failures = _normalize_failure_rows(read_ndjson(layout.failures_ndjson))

    attempt_ids = {_attempt_id_from_row(row) for row in attempts if _attempt_id_from_row(row)}
    failure_ids = {_attempt_id_from_row(row) for row in failures if _attempt_id_from_row(row)}
    phase_ids = {_attempt_id_from_row(row) for row in phases if _attempt_id_from_row(row)}

    persisted_terminal_ids = attempt_ids | failure_ids
    persisted_all_ids = persisted_terminal_ids | phase_ids

    orphan_ids = persisted_all_ids - checkpoint_completed
    if orphan_ids:
        attempts = [row for row in attempts if _attempt_id_from_row(row) not in orphan_ids]
        failures = [row for row in failures if _attempt_id_from_row(row) not in orphan_ids]
        phases = [row for row in phases if _attempt_id_from_row(row) not in orphan_ids]

    checkpoint_only_ids = checkpoint_completed - persisted_terminal_ids
    if checkpoint_only_ids:
        checkpoint_completed -= checkpoint_only_ids

    _rewrite_ndjson_validated(layout.attempts_ndjson, "attempt_record.schema.json", attempts)
    _rewrite_ndjson_validated(layout.phases_ndjson, "phase_trace.schema.json", phases)
    _rewrite_ndjson_validated(layout.failures_ndjson, "failure_record.schema.json", failures)

    _save_checkpoint(
        layout,
        completed_attempt_ids=checkpoint_completed,
        last_attempt_id=_last_attempt_id_by_sequence(checkpoint_completed),
    )
    return attempts, failures, checkpoint_completed


def _baseline_models(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(summary, dict):
        return {}
    models = summary.get("models") or []
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(models, list):
        return out
    for item in models:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model") or "")
        if not model:
            continue
        out[model] = {
            "decode_tps_p50": float(item.get("decode_tps_p50", 0.0) or 0.0),
            "e2e_tps_p50": float(item.get("e2e_tps_p50", 0.0) or 0.0),
            "ttft_ms_p95": float(item.get("ttft_ms_p95", 0.0) or 0.0),
            "stage_hotspots": list(item.get("stage_hotspots") or []),
            "os_thread_migrations_p50": float(
                ((item.get("runtime_locality") or {}) if isinstance(item, dict) else {}).get(
                    "os_thread_migrations_p50",
                    0.0,
                )
                or 0.0
            ),
        }
    return out


def _build_summary(
    attempt_rows: list[dict[str, Any]],
    *,
    baseline_summary: dict[str, Any] | None,
    baseline_mode: bool,
    model_order: list[str],
) -> dict[str, Any]:
    def _runtime_numeric_values(model_rows: list[dict[str, Any]], key: str) -> list[float]:
        values: list[float] = []
        for row in model_rows:
            runtime = row.get("runtime")
            if not isinstance(runtime, dict):
                continue
            raw = runtime.get(key)
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        return values

    def _mean_max(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        return float(statistics.fmean(values)), float(max(values))

    def _cv_pct(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = float(statistics.fmean(values))
        if mean <= 0.0:
            return 0.0
        return float(statistics.stdev(values) / mean * 100.0)

    measured = [row for row in attempt_rows if not bool(row.get("warmup", False))]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in measured:
        model = str(row.get("model_id", "")).strip()
        if not model:
            continue
        by_model.setdefault(model, []).append(row)

    order_index = {model: index for index, model in enumerate(model_order)}
    ordered_models = sorted(by_model.keys(), key=lambda model: (order_index.get(model, 10_000), model))

    model_summaries: list[dict[str, Any]] = []
    baseline_floors: dict[str, dict[str, float]] = {}
    baseline_reference = _baseline_models(baseline_summary)

    for model in ordered_models:
        rows = by_model[model]
        decode_values = [
            float((row.get("throughput") or {}).get("decode_tps", 0.0) or 0.0)
            for row in rows
        ]
        e2e_values = [
            float((row.get("throughput") or {}).get("e2e_tps", 0.0) or 0.0)
            for row in rows
        ]
        ttft_values = [
            float((row.get("latency") or {}).get("ttft_ms", 0.0) or 0.0)
            for row in rows
        ]

        decode_p50 = statistics.median(decode_values) if decode_values else 0.0
        e2e_p50 = statistics.median(e2e_values) if e2e_values else 0.0
        ttft_p95 = (
            statistics.quantiles(ttft_values, n=20)[18]
            if len(ttft_values) >= 2
            else (ttft_values[0] if ttft_values else 0.0)
        )
        drift_mean_mean, drift_mean_max = _mean_max(_runtime_numeric_values(rows, "drift_mean"))
        drift_max_mean, drift_max_max = _mean_max(_runtime_numeric_values(rows, "drift_max"))
        drift_pruned_blocks_mean, drift_pruned_blocks_max = _mean_max(
            _runtime_numeric_values(rows, "drift_pruned_blocks")
        )
        stabilizer_seconds_mean, stabilizer_seconds_max = _mean_max(
            _runtime_numeric_values(rows, "stabilizer_seconds")
        )
        drift_overhead_percent_mean, drift_overhead_percent_max = _mean_max(
            _runtime_numeric_values(rows, "drift_overhead_percent")
        )
        os_thread_migration_values = _runtime_numeric_values(rows, "os_thread_migrations")
        os_thread_migrations_p50 = (
            float(statistics.median(os_thread_migration_values))
            if os_thread_migration_values
            else 0.0
        )
        stage_hotspots = _compute_stage_hotspots(rows)
        decode_time_accounted_pct = _decode_time_accounted_pct(rows, stage_hotspots)
        decode_time_accounted_values = [
            _decode_time_accounted_pct([row], _compute_stage_hotspots([row])) for row in rows
        ]
        run_stability = {
            "decode_tps_cv_pct": round(_cv_pct(decode_values), 2),
            "e2e_tps_cv_pct": round(_cv_pct(e2e_values), 2),
            "ttft_ms_cv_pct": round(_cv_pct(ttft_values), 2),
            "decode_time_accounted_cv_pct": round(_cv_pct(decode_time_accounted_values), 2),
        }
        run_stability_gate = (
            True
            if len(rows) < 2
            else (
                float(run_stability["decode_tps_cv_pct"]) <= 20.0
                and float(run_stability["ttft_ms_cv_pct"]) <= 20.0
            )
        )
        decode_accounting_gate = bool(
            DECODE_ACCOUNTING_MIN_PCT
            <= decode_time_accounted_pct
            <= DECODE_ACCOUNTING_MAX_PCT
        )
        stage_deltas: dict[str, dict[str, float]] = {}

        baseline_ref = baseline_reference.get(model)
        baseline_decode = float(
            (baseline_ref or {}).get("decode_tps_p50", decode_p50) or decode_p50
        )
        baseline_e2e = float((baseline_ref or {}).get("e2e_tps_p50", e2e_p50) or e2e_p50)
        baseline_ttft = float((baseline_ref or {}).get("ttft_ms_p95", ttft_p95) or ttft_p95)
        baseline_migrations = float(
            (baseline_ref or {}).get("os_thread_migrations_p50", os_thread_migrations_p50)
            or os_thread_migrations_p50
        )
        if baseline_ref is not None:
            baseline_hotspots = baseline_ref.get("stage_hotspots") or []
            baseline_by_stage: dict[str, dict[str, Any]] = {}
            for hotspot in baseline_hotspots:
                if not isinstance(hotspot, dict):
                    continue
                stage_name = str(hotspot.get("stage") or "").strip()
                if stage_name:
                    baseline_by_stage[stage_name] = hotspot
            for hotspot in stage_hotspots:
                stage = str(hotspot.get("stage") or "").strip()
                if stage not in baseline_by_stage:
                    continue
                baseline_hotspot = baseline_by_stage[stage]
                baseline_pct = float(baseline_hotspot.get("pct_of_decode", 0.0) or 0.0)
                current_pct = float(hotspot.get("pct_of_decode", 0.0) or 0.0)
                baseline_ms = float(baseline_hotspot.get("total_ms_mean", 0.0) or 0.0)
                current_ms = float(hotspot.get("total_ms_mean", 0.0) or 0.0)
                stage_deltas[stage] = {
                    "baseline_pct": baseline_pct,
                    "current_pct": current_pct,
                    "delta_pct": round(current_pct - baseline_pct, 1),
                    "baseline_ms": baseline_ms,
                    "current_ms": current_ms,
                    "delta_ms": round(current_ms - baseline_ms, 2),
                }

        floors = {
            "decode_tps_floor": baseline_decode * 0.97,
            "e2e_tps_floor": baseline_e2e * 0.97,
            "ttft_ceiling_ms": baseline_ttft * 1.05,
        }
        baseline_floors[model] = floors

        absolute_gate = bool(
            decode_p50 >= floors["decode_tps_floor"]
            and e2e_p50 >= floors["e2e_tps_floor"]
            and ttft_p95 <= floors["ttft_ceiling_ms"]
        )

        relative_gate: bool | None = None
        if not baseline_mode and baseline_ref is not None:
            relative_gate = bool(
                decode_p50 >= baseline_decode * 1.08
                and e2e_p50 >= baseline_e2e * 1.08
                and ttft_p95 <= baseline_ttft * 0.95
            )
        migration_reduction_gate: bool | None = None
        if not baseline_mode and baseline_ref is not None and baseline_migrations > 0.0:
            migration_reduction_gate = bool(
                os_thread_migrations_p50 <= baseline_migrations * 0.20
            )

        telemetry_completeness_gate = all(
            bool((row.get("measurement") or {}).get("valid", False))
            and not list((row.get("measurement") or {}).get("missing_signals") or [])
            and bool((row.get("provenance") or {}).get("sanctioned_backend_path"))
            and bool((row.get("provenance") or {}).get("tokenizer_backend"))
            for row in rows
        )
        telemetry_gate = (
            telemetry_completeness_gate
            and decode_accounting_gate
            and run_stability_gate
            and (migration_reduction_gate is None or migration_reduction_gate)
        )
        coherence_gate = all(
            bool((row.get("coherence") or {}).get("ok", False))
            and bool((row.get("coherence") or {}).get("raw_ok", False))
            and not list((row.get("coherence") or {}).get("issues") or [])
            and not list((row.get("coherence") or {}).get("raw_issues") or [])
            for row in rows
        )

        model_gate_issues: list[str] = []
        if not absolute_gate:
            model_gate_issues.append("absolute_gate_failed")
        if not baseline_mode:
            if relative_gate is None:
                model_gate_issues.append("relative_gate_baseline_missing")
            elif not relative_gate:
                model_gate_issues.append("relative_gate_failed")
            if migration_reduction_gate is None:
                model_gate_issues.append("migration_reduction_baseline_missing")
            elif not migration_reduction_gate:
                model_gate_issues.append("migration_reduction_gate_failed")
        if not telemetry_gate:
            model_gate_issues.append("telemetry_gate_failed")
        if not decode_accounting_gate:
            model_gate_issues.append("telemetry_decode_accounting_failure")
            if decode_time_accounted_pct > DECODE_ACCOUNTING_MAX_PCT:
                model_gate_issues.append("telemetry_overlap_failure")
            elif decode_time_accounted_pct < DECODE_ACCOUNTING_MIN_PCT:
                model_gate_issues.append("telemetry_under_accounting_failure")
        if not run_stability_gate:
            model_gate_issues.append("run_stability_gate_failed")
        if not coherence_gate:
            model_gate_issues.append("coherence_gate_failed")

        model_pass = (
            absolute_gate and telemetry_gate and coherence_gate
            if baseline_mode
            else absolute_gate and bool(relative_gate) and telemetry_gate and coherence_gate
        )

        model_summaries.append(
            {
                "model": model,
                "runs": len(rows),
                "decode_tps_p50": decode_p50,
                "e2e_tps_p50": e2e_p50,
                "ttft_ms_p95": ttft_p95,
                "baseline": {
                    "decode_tps_p50": baseline_decode,
                    "e2e_tps_p50": baseline_e2e,
                    "ttft_ms_p95": baseline_ttft,
                },
                "floors": floors,
                "gates": {
                    "relative_uplift": relative_gate,
                    "absolute_floor": absolute_gate,
                    "telemetry_completeness": telemetry_completeness_gate,
                    "decode_time_accounting_window": decode_accounting_gate,
                    "decode_time_accounting_min_pct": DECODE_ACCOUNTING_MIN_PCT,
                    "decode_time_accounting_max_pct": DECODE_ACCOUNTING_MAX_PCT,
                    "run_stability": run_stability_gate,
                    "coherence_non_regression": coherence_gate,
                    "migration_reduction": migration_reduction_gate,
                },
                "run_stability": run_stability,
                "runtime_locality": {
                    "os_thread_migrations_p50": os_thread_migrations_p50,
                    "baseline_os_thread_migrations_p50": baseline_migrations,
                },
                "runtime_drift": {
                    "drift_mean_mean": drift_mean_mean,
                    "drift_mean_max": drift_mean_max,
                    "drift_max_mean": drift_max_mean,
                    "drift_max_max": drift_max_max,
                    "drift_pruned_blocks_mean": drift_pruned_blocks_mean,
                    "drift_pruned_blocks_max": drift_pruned_blocks_max,
                    "stabilizer_seconds_mean": stabilizer_seconds_mean,
                    "stabilizer_seconds_max": stabilizer_seconds_max,
                    "drift_overhead_percent_mean": drift_overhead_percent_mean,
                    "drift_overhead_percent_max": drift_overhead_percent_max,
                },
                "stage_hotspots": stage_hotspots[:15],
                "stage_deltas": stage_deltas,
                "kernel_map_version": KERNEL_MAP_VERSION,
                "decode_time_accounted_pct": decode_time_accounted_pct,
                "pass": model_pass,
                "gate_issues": model_gate_issues,
            }
        )

    top_stage_hotspots, stage_origin_map = _root_stage_views(
        model_summaries,
        model_order=model_order,
    )

    overall_pass = bool(model_summaries) and all(
        bool(model.get("pass", False)) for model in model_summaries
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": model_summaries,
        "top_stage_hotspots": top_stage_hotspots,
        "stage_origin_map": stage_origin_map,
        "baseline_floors": baseline_floors,
        "overall_pass": overall_pass,
    }


def _decode_time_accounted_pct(
    rows: list[dict[str, Any]],
    stage_hotspots: list[dict[str, Any]],
) -> float:
    decode_totals_ms: list[float] = []
    stage_sums_ms: list[float] = []
    for row in rows:
        runtime = row.get("runtime")
        if not isinstance(runtime, dict):
            continue
        decode_s = float(runtime.get("runtime_decode_seconds", 0.0) or 0.0)
        if decode_s > 0.0:
            decode_totals_ms.append(decode_s * 1000.0)
        stage_ms = runtime.get("graph_stage_ms")
        if isinstance(stage_ms, dict):
            stage_sum = 0.0
            for value in stage_ms.values():
                stage_sum += float(value or 0.0)
            if stage_sum > 0.0:
                stage_sums_ms.append(stage_sum)

    if decode_totals_ms:
        total_decode_ms = float(statistics.fmean(decode_totals_ms))
    elif stage_sums_ms:
        total_decode_ms = float(statistics.fmean(stage_sums_ms))
    else:
        total_decode_ms = 0.0

    if total_decode_ms <= 0.0:
        return 0.0

    accounted_ms = sum(
        float(hotspot.get("total_ms_mean", 0.0) or 0.0) for hotspot in stage_hotspots
    )
    return round((accounted_ms / total_decode_ms) * 100.0, 1)


def _compute_stage_hotspots(
    rows: list[dict[str, Any]],
    kernel_map: dict[str, dict[str, str | int]] | None = None,
) -> list[dict[str, Any]]:
    """Compute ranked stage hotspots from graph-stage runtime counters."""
    if kernel_map is None:
        kernel_map = KERNEL_MAP

    stage_totals: dict[str, list[float]] = {}
    stage_call_totals: dict[str, list[int]] = {}
    decode_totals_ms: list[float] = []
    stage_sums_ms: list[float] = []

    for row in rows:
        runtime = row.get("runtime")
        if not isinstance(runtime, dict):
            continue
        stage_ms = runtime.get("graph_stage_ms")
        stage_calls = runtime.get("graph_stage_calls")
        if not isinstance(stage_ms, dict):
            stage_ms = {}
        if not isinstance(stage_calls, dict):
            stage_calls = {}

        decode_s = float(runtime.get("runtime_decode_seconds", 0.0) or 0.0)
        if decode_s > 0.0:
            decode_totals_ms.append(decode_s * 1000.0)

        per_row_stage_sum = 0.0
        for stage, ms_val in stage_ms.items():
            stage_name = str(stage).strip()
            if not stage_name:
                continue
            stage_ms_value = float(ms_val or 0.0)
            stage_totals.setdefault(stage_name, []).append(stage_ms_value)
            per_row_stage_sum += stage_ms_value
            stage_call_totals.setdefault(stage_name, []).append(
                int(stage_calls.get(stage_name, 0) or 0)
            )
        if per_row_stage_sum > 0.0:
            stage_sums_ms.append(per_row_stage_sum)

    if decode_totals_ms:
        total_decode_ms = float(statistics.fmean(decode_totals_ms))
    elif stage_sums_ms:
        total_decode_ms = float(statistics.fmean(stage_sums_ms))
    else:
        total_decode_ms = 0.0

    hotspots: list[dict[str, Any]] = []
    for stage_name in sorted(stage_totals.keys()):
        ms_values = stage_totals[stage_name]
        call_values = stage_call_totals.get(stage_name, [0])
        total_ms_mean = float(statistics.fmean(ms_values)) if ms_values else 0.0
        calls_mean = float(statistics.fmean(call_values)) if call_values else 0.0
        total_ms_stddev = float(statistics.stdev(ms_values)) if len(ms_values) >= 2 else 0.0
        calls_stddev = float(statistics.stdev(call_values)) if len(call_values) >= 2 else 0.0
        total_ms_cv_pct = ((total_ms_stddev / total_ms_mean) * 100.0) if total_ms_mean > 0.0 else 0.0
        pct_of_decode = (
            (total_ms_mean / total_decode_ms * 100.0)
            if total_decode_ms > 0.0
            else 0.0
        )
        avg_ms_per_call = (total_ms_mean / calls_mean) if calls_mean > 0.0 else 0.0

        kernel_info = kernel_map.get(stage_name, {})
        hotspots.append(
            {
                "stage": stage_name,
                "total_ms_mean": round(total_ms_mean, 2),
                "total_ms_stddev": round(total_ms_stddev, 2),
                "total_ms_cv_pct": round(total_ms_cv_pct, 2),
                "pct_of_decode": round(pct_of_decode, 1),
                "calls_mean": round(calls_mean, 0),
                "calls_stddev": round(calls_stddev, 2),
                "avg_ms_per_call": round(avg_ms_per_call, 4),
                "cpp_file": str(kernel_info.get("cpp_file", "unknown")),
                "cpp_function": str(kernel_info.get("cpp_function", "unknown")),
                "kernel_class": str(kernel_info.get("kernel_class", "unknown")),
                "weight_format": str(kernel_info.get("weight_format", "unknown")),
                "complexity": str(kernel_info.get("complexity", "unknown")),
            }
        )

    hotspots.sort(key=lambda item: float(item.get("pct_of_decode", 0.0)), reverse=True)
    return hotspots


def _root_stage_views(
    model_summaries: list[dict[str, Any]],
    *,
    model_order: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    model_rank = {model: index for index, model in enumerate(model_order)}
    rows: list[dict[str, Any]] = []
    stage_origin: dict[str, dict[str, str]] = {}
    stage_models: dict[str, set[str]] = {}

    for model_summary in model_summaries:
        model = str(model_summary.get("model") or "").strip()
        if not model:
            continue
        for hotspot in model_summary.get("stage_hotspots") or []:
            if not isinstance(hotspot, dict):
                continue
            stage = str(hotspot.get("stage") or "").strip()
            if not stage:
                continue
            origin = {
                "cpp_file": str(hotspot.get("cpp_file") or "unknown"),
                "cpp_function": str(hotspot.get("cpp_function") or "unknown"),
                "kernel_class": str(hotspot.get("kernel_class") or "unknown"),
                "weight_format": str(hotspot.get("weight_format") or "unknown"),
                "complexity": str(hotspot.get("complexity") or "unknown"),
            }
            if stage not in stage_origin:
                stage_origin[stage] = origin
            stage_models.setdefault(stage, set()).add(model)

            rows.append(
                {
                    "model": model,
                    "stage": stage,
                    "total_ms_mean": float(hotspot.get("total_ms_mean", 0.0) or 0.0),
                    "total_ms_stddev": float(hotspot.get("total_ms_stddev", 0.0) or 0.0),
                    "total_ms_cv_pct": float(hotspot.get("total_ms_cv_pct", 0.0) or 0.0),
                    "pct_of_decode": float(hotspot.get("pct_of_decode", 0.0) or 0.0),
                    "calls_mean": float(hotspot.get("calls_mean", 0.0) or 0.0),
                    "calls_stddev": float(hotspot.get("calls_stddev", 0.0) or 0.0),
                    "avg_ms_per_call": float(hotspot.get("avg_ms_per_call", 0.0) or 0.0),
                    **origin,
                }
            )

    rows.sort(
        key=lambda item: (
            -float(item.get("pct_of_decode", 0.0)),
            model_rank.get(str(item.get("model") or ""), 10_000),
            str(item.get("stage") or ""),
            str(item.get("model") or ""),
        )
    )
    top_hotspots = rows[:ROOT_STAGE_HOTSPOT_LIMIT]

    ordered_origin_map: dict[str, dict[str, Any]] = {}
    for stage in sorted(stage_origin):
        ordered_origin_map[stage] = {
            **stage_origin[stage],
            "models": sorted(stage_models.get(stage, set())),
        }

    return top_hotspots, ordered_origin_map


def _store_baseline(summary: dict[str, Any], base_dir: Path, *, run_id: str) -> Path:
    fingerprint = host_fingerprint()
    date_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    baseline_dir = base_dir / "baselines" / fingerprint / date_stamp
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = baseline_dir / f"{str(run_id).strip() or 'baseline'}.json"
    if baseline_path.exists():
        existing = json.loads(baseline_path.read_text(encoding="utf-8"))
        if existing != summary:
            raise RuntimeError(
                f"Refusing to overwrite immutable baseline file: {baseline_path}"
            )
        return baseline_path
    write_json_atomic(baseline_path, summary)
    return baseline_path


def _assert_fresh_run(layout: RunLayout, *, force_rerun: bool) -> None:
    generated = [
        layout.manifest_json,
        layout.attempts_ndjson,
        layout.phases_ndjson,
        layout.summary_json,
        layout.failures_ndjson,
        layout.checkpoint_json,
    ]
    existing = [path for path in generated if path.exists()]
    if not existing:
        return
    if not force_rerun:
        paths = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            f"Run directory already has artifacts. Use --resume or --force-rerun: {paths}"
        )
    for path in existing:
        path.unlink()


def _assert_resume_compatible(
    existing_manifest: dict[str, Any],
    *,
    models: list[str],
    matrix: list[tuple[int | None, int | None, int | None]],
    prompt_hash: str,
    warmup_runs: int,
    measured_runs: int,
    benchmark_config: dict[str, Any],
) -> None:
    expected_matrix = _thread_matrix_payload(matrix)
    checks = {
        "schema_version": str(existing_manifest.get("schema_version") or ""),
        "models": list(existing_manifest.get("models") or []),
        "thread_matrix": list(existing_manifest.get("thread_matrix") or []),
        "prompt_hash": str(existing_manifest.get("prompt_hash") or ""),
        "warmup_runs": int(existing_manifest.get("warmup_runs") or 0),
        "measured_runs": int(existing_manifest.get("measured_runs") or 0),
    }
    if checks["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(
            "Resume manifest mismatch: schema version differs from current runner."
        )
    if checks["models"] != list(models):
        raise RuntimeError("Resume manifest mismatch: models differ from existing run.")
    if checks["thread_matrix"] != expected_matrix:
        raise RuntimeError("Resume manifest mismatch: thread matrix differs.")
    if checks["prompt_hash"] != prompt_hash:
        raise RuntimeError("Resume manifest mismatch: prompt hash differs.")
    if checks["warmup_runs"] != int(warmup_runs):
        raise RuntimeError("Resume manifest mismatch: warmup runs differ.")
    if checks["measured_runs"] != int(measured_runs):
        raise RuntimeError("Resume manifest mismatch: measured runs differ.")
    existing_config = existing_manifest.get("benchmark_config")
    if existing_config is not None and dict(existing_config) != dict(benchmark_config):
        raise RuntimeError("Resume manifest mismatch: benchmark config differs.")


def _load_json(path: Path | None) -> dict[str, Any] | None:
    return load_optional_json_dict(path)


def _failure_payload(
    *,
    run_id: str,
    attempt_id: str,
    model: str,
    error: str,
    error_type: str,
    gate_issues: list[str],
    failure_kind: str,
    tb: str | None = None,
) -> dict[str, Any]:
    normalized = sorted(
        {
            *(str(issue) for issue in gate_issues if str(issue).strip()),
            f"error_type:{str(error_type).strip()}" if str(error_type).strip() else "",
        }
    )
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "model": model,
        "error": str(error),
        "error_type": str(error_type),
        "failure_kind": failure_kind,
        "gate_issues": sorted({str(issue) for issue in gate_issues if str(issue).strip()}),
        "normalized_issues": [issue for issue in normalized if issue],
        "traceback": str(tb or ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument(
        "--prompt",
        default=(
            "In 3-4 sentences, explain how AVX2 and OpenMP improve CPU LLM inference "
            "throughput and latency, using concrete mechanisms and avoiding repetition."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument(
        "--audit-surface",
        choices=("native", "continuous", "both"),
        default="both",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sampling-profile", type=str, default=None)
    parser.add_argument(
        "--coherence-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-new-tokens-before-eos", type=int, default=12)
    parser.add_argument(
        "--require-openmp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-avx2",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--host-access",
        choices=("user", "privileged", "mixed"),
        default="user",
    )
    parser.add_argument(
        "--collect-hw-counters",
        choices=("off", "auto", "required"),
        default="auto",
    )
    parser.add_argument(
        "--require-grover",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--require-coconut",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--autotune",
        choices=("off", "explore", "locked"),
        default="off",
    )
    parser.add_argument("--decode-threads", type=int, default=None)
    parser.add_argument("--batch-threads", type=int, default=None)
    parser.add_argument("--ubatch", type=int, default=None)
    parser.add_argument("--thread-sweep", action="store_true")
    parser.add_argument("--decode-thread-sweep", type=str, default=None)
    parser.add_argument("--batch-thread-sweep", type=str, default=None)
    parser.add_argument("--ubatch-sweep", type=str, default=None)
    parser.add_argument("--out-root", type=str, default="audit")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--baseline-summary", type=str, default=None)
    parser.add_argument("--baseline-mode", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_root = (repo_root / args.out_root).resolve()
    layout = resolve_layout(out_root, args.run_id)
    ensure_layout(layout)

    models = _normalize_models(list(args.model))
    matrix = _thread_matrix(args)
    if not matrix:
        raise RuntimeError("Thread matrix resolved to an empty set.")

    warmup_runs = int(args.warmup_runs)
    measured_runs = int(args.runs)
    prompt_hash = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()
    benchmark_config = _benchmark_config_payload(args)

    plan = _attempt_plan(
        models=models,
        matrix=matrix,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    planned_attempts = len(plan)

    runtime_prov = capture_runtime_provenance(repo_root)
    model_contracts = {model: capture_model_provenance(model) for model in models}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": layout.run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "models": models,
        "model_sequence": [{"index": idx, "model": model} for idx, model in enumerate(models)],
        "thread_matrix": _thread_matrix_payload(matrix),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "benchmark_config": benchmark_config,
        "prompt_hash": prompt_hash,
        "planned_attempts": planned_attempts,
        "runtime_provenance": runtime_prov,
        "model_contracts": model_contracts,
    }

    if bool(getattr(args, "resume", False)):
        if not layout.manifest_json.exists():
            raise RuntimeError(
                f"Cannot resume: missing manifest file at {layout.manifest_json}"
            )
        existing_manifest = json.loads(layout.manifest_json.read_text(encoding="utf-8"))
        validate_payload("run_manifest.schema.json", existing_manifest)
        _assert_resume_compatible(
            existing_manifest,
            models=models,
            matrix=matrix,
            prompt_hash=prompt_hash,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            benchmark_config=benchmark_config,
        )
        manifest = existing_manifest
    else:
        _assert_fresh_run(layout, force_rerun=bool(getattr(args, "force_rerun", False)))
        _write_json_validated(layout.manifest_json, "run_manifest.schema.json", manifest)

    if bool(getattr(args, "resume", False)):
        checkpoint = _load_checkpoint(layout)
        attempt_rows, failure_rows, completed_attempt_ids = _reconcile_resume_state(
            layout,
            checkpoint,
        )
    else:
        attempt_rows = []
        failure_rows = []
        completed_attempt_ids: set[str] = set()
        _rewrite_ndjson_validated(layout.attempts_ndjson, "attempt_record.schema.json", [])
        _rewrite_ndjson_validated(layout.phases_ndjson, "phase_trace.schema.json", [])
        _rewrite_ndjson_validated(layout.failures_ndjson, "failure_record.schema.json", [])
        _save_checkpoint(
            layout,
            completed_attempt_ids=completed_attempt_ids,
            last_attempt_id=None,
        )

    for item in plan:
        attempt_id = _attempt_id(
            model_index=int(item["model_index"]),
            combo_index=int(item["combo_index"]),
            warmup=bool(item["warmup"]),
            run_index=int(item["run_index"]),
            model=str(item["model"]),
        )
        if attempt_id in completed_attempt_ids and not bool(getattr(args, "force_rerun", False)):
            continue

        spec = AttemptSpec(
            attempt_id=attempt_id,
            model=str(item["model"]),
            prompt=args.prompt,
            max_new_tokens=int(args.max_new_tokens),
            context_length=int(args.context_length),
            decode_threads=item["decode_threads"],
            batch_threads=item["batch_threads"],
            ubatch=item["ubatch"],
            sampling_profile=args.sampling_profile,
            coherence_first=bool(getattr(args, "coherence_first", True)),
            min_new_tokens_before_eos=getattr(args, "min_new_tokens_before_eos", 12),
            require_openmp=bool(getattr(args, "require_openmp", True)),
            require_avx2=bool(getattr(args, "require_avx2", True)),
            require_mmap=bool(getattr(args, "require_mmap", False)),
            host_access=str(getattr(args, "host_access", "user")),
            collect_hw_counters=str(getattr(args, "collect_hw_counters", "auto")),
            require_grover=bool(getattr(args, "require_grover", False)),
            require_coconut=bool(getattr(args, "require_coconut", False)),
            autotune=str(getattr(args, "autotune", "off")),
            warmup=bool(item["warmup"]),
            run_index=int(item["run_index"]),
        )

        try:
            attempt_record, phases, report = execute_attempt(
                spec,
                repo_root=repo_root,
            )
            attempt_record["run_id"] = layout.run_id
            attempt_record["measurement"]["report_type"] = (
                "dict" if isinstance(report, dict) else "list"
            )
            _append_ndjson_validated(
                layout.attempts_ndjson,
                "attempt_record.schema.json",
                attempt_record,
            )
            attempt_rows.append(attempt_record)

            for phase in phases:
                phase["run_id"] = layout.run_id
                _append_ndjson_validated(
                    layout.phases_ndjson,
                    "phase_trace.schema.json",
                    phase,
                )

            if not bool((attempt_record.get("status") or {}).get("ok", False)):
                status_payload = attempt_record.get("status") or {}
                status_code = int(status_payload.get("return_code", 0) or 0)
                status_issues = list(status_payload.get("issues") or [])
                non_return_code_issues = [
                    issue
                    for issue in status_issues
                    if not str(issue).startswith("subprocess_return_code=")
                ]
                failure_kind = "gate_failure"
                error = "Attempt failed gate checks."
                error_type = "GateFailure"
                if status_code != 0 and not non_return_code_issues:
                    failure_kind = "execution_failure"
                    error = "Benchmark subprocess exited non-zero."
                    error_type = "BenchmarkSubprocessError"
                failure = _failure_payload(
                    run_id=layout.run_id,
                    attempt_id=attempt_id,
                    model=spec.model,
                    error=error,
                    error_type=error_type,
                    gate_issues=status_issues,
                    failure_kind=failure_kind,
                )
                _append_ndjson_validated(
                    layout.failures_ndjson,
                    "failure_record.schema.json",
                    failure,
                )
                failure_rows.append(failure)
        except Exception as exc:
            failure = _failure_payload(
                run_id=layout.run_id,
                attempt_id=attempt_id,
                model=spec.model,
                error=str(exc),
                error_type=type(exc).__name__,
                gate_issues=[],
                failure_kind="execution_failure",
                tb=traceback.format_exc(),
            )
            _append_ndjson_validated(
                layout.failures_ndjson,
                "failure_record.schema.json",
                failure,
            )
            failure_rows.append(failure)

        completed_attempt_ids.add(attempt_id)
        _save_checkpoint(
            layout,
            completed_attempt_ids=completed_attempt_ids,
            last_attempt_id=attempt_id,
        )

    failure_rows = _normalize_failure_rows(failure_rows)
    _rewrite_ndjson_validated(layout.failures_ndjson, "failure_record.schema.json", failure_rows)

    if str(getattr(args, "audit_surface", "native")) in {"continuous", "both"}:
        _run_continuous_surface(
            repo_root=repo_root,
            layout=layout,
            models=models,
            max_new_tokens=int(args.max_new_tokens),
        )

    baseline_mode = bool(getattr(args, "baseline_mode", False))
    baseline_summary_path = (
        Path(str(args.baseline_summary)).resolve()
        if getattr(args, "baseline_summary", None)
        else None
    )
    if baseline_summary_path is not None and not baseline_summary_path.exists():
        raise RuntimeError(f"Baseline summary file not found: {baseline_summary_path}")
    baseline_summary = _load_json(baseline_summary_path)
    prior_summary = _load_json(layout.summary_json)
    if baseline_summary is None and prior_summary is not None and not baseline_mode:
        baseline_summary = prior_summary
    if baseline_summary is None and not baseline_mode:
        baseline_mode = True
    summary = _build_summary(
        attempt_rows,
        baseline_summary=baseline_summary,
        baseline_mode=baseline_mode,
        model_order=models,
    )

    failure_counts = {
        "total": len(failure_rows),
        "gate_failure": sum(
            1 for row in failure_rows if str(row.get("failure_kind")) == "gate_failure"
        ),
        "execution_failure": sum(
            1
            for row in failure_rows
            if str(row.get("failure_kind")) == "execution_failure"
        ),
    }

    summary["run_id"] = layout.run_id
    summary["failure_count"] = int(failure_counts["total"])
    summary["failure_counts"] = failure_counts
    summary["failed_attempt_ids"] = [
        str(row.get("attempt_id"))
        for row in failure_rows
        if str(row.get("attempt_id") or "").strip()
    ]
    summary["completed_attempts"] = len(completed_attempt_ids)
    summary["planned_attempts"] = planned_attempts
    summary["pass"] = bool(summary.get("overall_pass", False)) and failure_counts["total"] == 0
    _write_json_validated(layout.summary_json, "summary.schema.json", summary)

    if not bool(getattr(args, "skip_baseline", False)):
        existing_baseline_path = str((prior_summary or {}).get("baseline_path") or "").strip()
        if existing_baseline_path and Path(existing_baseline_path).exists():
            summary["baseline_path"] = existing_baseline_path
        else:
            baseline_path = _store_baseline(summary, out_root, run_id=layout.run_id)
            summary["baseline_path"] = str(baseline_path)
        _write_json_validated(layout.summary_json, "summary.schema.json", summary)

    return 1 if not bool(summary.get("pass", False)) else 0


def _deprecated_entrypoint(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    known, unknown = parser.parse_known_args(argv)

    forwarded: list[str] = []
    if known.profile:
        forwarded.extend(["--profile", str(known.profile)])
    if known.run_id:
        forwarded.extend(["--run-id", str(known.run_id)])
    if known.resume:
        forwarded.append("--resume")
    if known.out_root:
        forwarded.extend(["--out-root", str(known.out_root)])

    legacy_flags = [token for token in unknown if str(token).startswith("-")]
    if legacy_flags:
        print(
            "ERROR: direct native_benchmark_runner flags are deprecated.\n"
            "Use `./scripts/run_native_qsg_suite.sh` with profile-backed defaults.\n"
            f"Rejected flags: {', '.join(sorted(set(legacy_flags)))}",
            file=sys.stderr,
        )
        print(
            "Allowed compatibility flags: --profile, --run-id, --resume, --out-root",
            file=sys.stderr,
        )
        return 2

    repo_root = Path(__file__).resolve().parents[2]
    suite_script = repo_root / "audit" / "runner" / "benchmark_suite.py"
    cmd = [sys.executable, str(suite_script), *forwarded]
    print(
        "DEPRECATED: `python audit/runner/native_benchmark_runner.py` is not an operator entrypoint.\n"
        "Use: ./scripts/run_native_qsg_suite.sh",
        file=sys.stderr,
    )
    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(_deprecated_entrypoint(sys.argv[1:]))
