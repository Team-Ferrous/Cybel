#!/usr/bin/env python3
"""Native kernel microbenchmark harness and compatibility summarizer."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.native.native_qsg_engine import NativeQSGEngine


KERNEL_STAGE_MAP = {
    "embedding_lookup": ("embedding_lookup_seconds", None, None, None),
    "attention_proj": ("attention_proj_seconds", "attention_proj_flops", "attention_proj_bytes", "attention_calls"),
    "attention_rope_kv": ("attention_rope_kv_seconds", None, None, "attention_calls"),
    "attention_decode": ("attention_decode_seconds", None, None, "attention_calls"),
    "attention_out_proj": ("attention_out_proj_seconds", "attention_out_proj_flops", "attention_out_proj_bytes", "attention_calls"),
    "ffn_gate_up": ("ffn_gate_up_seconds", "ffn_gate_up_flops", "ffn_gate_up_bytes", "ffn_calls"),
    "ffn_down": ("ffn_down_seconds", "ffn_down_flops", "ffn_down_bytes", "ffn_calls"),
    "ssm_projection": ("ssm_projection_seconds", "ssm_projection_flops", "ssm_projection_bytes", "ssm_calls"),
    "ssm_conv": ("ssm_conv_seconds", None, None, "ssm_calls"),
    "ssm_recurrent": ("ssm_recurrent_seconds", None, None, "ssm_calls"),
    "ssm_output": ("ssm_output_seconds", "ssm_output_flops", "ssm_output_bytes", "ssm_calls"),
    "lm_head": ("lm_head_seconds", "lm_head_flops", "lm_head_bytes", "forward_token_calls"),
    "sanitize": ("sanitize_seconds", None, None, "forward_token_calls"),
}

KERNEL_SOURCE_MAP = {
    "embedding_lookup": ("core/native/model_graph.cpp", "graph_forward_token_impl"),
    "attention_proj": ("core/native/quantized_matmul.cpp", "dot_rows_q6k"),
    "attention_rope_kv": ("core/native/model_graph.cpp", "apply_rope_with_dim"),
    "attention_decode": ("core/native/model_graph.cpp", "fused_gqa_attention_decode_paged"),
    "attention_out_proj": ("core/native/quantized_matmul.cpp", "dot_rows_q6k"),
    "ffn_gate_up": ("core/native/quantized_matmul.cpp", "dot_rows_q6k"),
    "ffn_down": ("core/native/quantized_matmul.cpp", "dot_rows_q6k"),
    "ssm_projection": ("core/native/quantized_matmul.cpp", "dot_rows_q6k"),
    "ssm_conv": ("core/native/model_graph.cpp", "ssm_conv_step"),
    "ssm_recurrent": ("core/native/model_graph.cpp", "ssm_mamba2_step"),
    "ssm_output": ("core/native/quantized_matmul.cpp", "dot_rows_q6k"),
    "lm_head": ("core/native/quantized_matmul.cpp", "dot_row_q6k_lm"),
    "sanitize": ("core/native/model_graph.cpp", "sanitize_output_inplace"),
}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))),
    )
    return ordered[index]


def _recoverable_gain_pct(kernel_seconds: float, wall_seconds: float) -> float:
    if kernel_seconds <= 0.0 or wall_seconds <= 0.0:
        return 0.0
    return min(75.0, max(0.0, (kernel_seconds / wall_seconds) * 100.0 * 0.5))


def _first_float(perf_stats: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key not in perf_stats:
            continue
        raw = perf_stats.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _summary_rows_from_benchmark_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("flat_results") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    summaries: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        stage_ms = dict(row.get("graph_stage_ms") or {})
        decode_ms = float(row.get("graph_decode_ms", 0.0) or 0.0)
        for kernel in KERNEL_STAGE_MAP:
            total_ms = float(stage_ms.get(kernel, 0.0) or 0.0)
            summaries.append(
                {
                    "model": row.get("model"),
                    "kernel": kernel,
                    "total_ms": total_ms,
                    "decode_share_pct": (total_ms / decode_ms * 100.0) if decode_ms > 0.0 else 0.0,
                    "recoverable_gain_estimate_pct": _recoverable_gain_pct(total_ms / 1000.0, decode_ms / 1000.0),
                }
            )
    summaries.sort(key=lambda item: float(item["recoverable_gain_estimate_pct"]), reverse=True)
    return summaries


def _derive_token_sequence(engine: NativeQSGEngine, token_count: int) -> list[int]:
    base = engine.tokenize(
        "kernel microbenchmark harness uses native graph perf stats to isolate hotspot movement."
    )
    if not base:
        base = [1, 2, 3, 4]
    tokens: list[int] = []
    while len(tokens) < token_count:
        tokens.extend(base)
    return tokens[:token_count]


def _kernel_shapes(engine: NativeQSGEngine, lengths: list[int]) -> list[dict[str, Any]]:
    profile = engine.profile
    head_dim = int(profile.embedding_dim / max(1, profile.n_heads)) if profile.n_heads else 0
    return [
        {
            "token_count": int(length),
            "hidden_dim": int(profile.embedding_dim),
            "num_layers": int(profile.n_layers),
            "num_heads": int(profile.n_heads),
            "num_kv_heads": int(profile.n_kv_heads),
            "head_dim": int(head_dim),
            "vocab_size": int(profile.vocab_size),
        }
        for length in lengths
    ]


def run_kernel_microbench(
    *,
    model: str,
    context_length: int,
    warmups: int,
    iterations: int,
    target_kernels: list[str],
    synthetic_lengths: list[int],
    degraded_reason: str | None = None,
) -> dict[str, Any]:
    engine = NativeQSGEngine(model, context_length=int(context_length))
    try:
        graph = getattr(engine, "_model_graph", None)
        if graph is None:
            raise RuntimeError("NativeModelGraph is required for kernel microbenchmarking.")
        shapes = _kernel_shapes(engine, synthetic_lengths)
        kernel_runs: list[dict[str, Any]] = []
        for shape in shapes:
            tokens = _derive_token_sequence(engine, int(shape["token_count"]))
            for _ in range(max(0, int(warmups))):
                engine.reset_kv_cache()
                reset_perf = getattr(graph, "reset_perf_stats", None)
                if callable(reset_perf):
                    reset_perf()
                engine._get_logits_for_tokens(tokens)
            samples_by_kernel: dict[str, list[dict[str, Any]]] = {
                kernel: [] for kernel in target_kernels
            }
            for _ in range(max(1, int(iterations))):
                engine.reset_kv_cache()
                reset_perf = getattr(graph, "reset_perf_stats", None)
                if callable(reset_perf):
                    reset_perf()
                started = time.perf_counter()
                engine._get_logits_for_tokens(tokens)
                wall_seconds = time.perf_counter() - started
                perf_stats = dict(getattr(graph, "get_perf_stats", lambda: {})() or {})
                for kernel in target_kernels:
                    stage_name, flop_name, byte_name, call_name = KERNEL_STAGE_MAP[kernel]
                    kernel_seconds = float(perf_stats.get(stage_name, 0.0) or 0.0)
                    calls = int(perf_stats.get(call_name, 0) or 0) if call_name else 0
                    flops = float(perf_stats.get(flop_name, 0.0) or 0.0) if flop_name else 0.0
                    byte_count = float(perf_stats.get(byte_name, 0.0) or 0.0) if byte_name else 0.0
                    cycles = _first_float(perf_stats, [f"{kernel}_cycles", "cycles"])
                    instructions = _first_float(
                        perf_stats,
                        [f"{kernel}_instructions", "instructions"],
                    )
                    cache_misses = _first_float(
                        perf_stats,
                        [f"{kernel}_cache_misses", "cache_misses"],
                    )
                    cache_refs = _first_float(
                        perf_stats,
                        [f"{kernel}_cache_references", "cache_references"],
                    )
                    ipc = (
                        instructions / cycles
                        if instructions is not None and cycles is not None and cycles > 0.0
                        else None
                    )
                    cache_miss_rate = (
                        cache_misses / cache_refs
                        if cache_misses is not None and cache_refs is not None and cache_refs > 0.0
                        else None
                    )
                    samples_by_kernel[kernel].append(
                        {
                            "kernel_seconds": kernel_seconds,
                            "wall_seconds": wall_seconds,
                            "calls": calls,
                            "flops": flops,
                            "bytes": byte_count,
                            "cycles": cycles,
                            "instructions": instructions,
                            "ipc": ipc,
                            "cache_miss_rate": cache_miss_rate,
                        }
                    )
            for kernel in target_kernels:
                samples = samples_by_kernel[kernel]
                latencies_ms = [float(item["kernel_seconds"]) * 1000.0 for item in samples]
                wall_seconds_mean = statistics.mean(
                    [float(item["wall_seconds"]) for item in samples]
                )
                seconds_mean = statistics.mean(
                    [float(item["kernel_seconds"]) for item in samples]
                )
                calls_mean = statistics.mean([float(item["calls"]) for item in samples])
                flops_mean = statistics.mean([float(item["flops"]) for item in samples])
                bytes_mean = statistics.mean([float(item["bytes"]) for item in samples])
                stddev = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
                cv = (stddev / statistics.mean(latencies_ms) * 100.0) if latencies_ms and statistics.mean(latencies_ms) > 0.0 else 0.0
                cycles_values = [
                    float(item["cycles"])
                    for item in samples
                    if item.get("cycles") is not None
                ]
                instruction_values = [
                    float(item["instructions"])
                    for item in samples
                    if item.get("instructions") is not None
                ]
                ipc_values = [
                    float(item["ipc"])
                    for item in samples
                    if item.get("ipc") is not None
                ]
                cache_miss_rate_values = [
                    float(item["cache_miss_rate"])
                    for item in samples
                    if item.get("cache_miss_rate") is not None
                ]
                cpp_file, cpp_function = KERNEL_SOURCE_MAP[kernel]
                counter_degraded_reason = degraded_reason
                if (
                    counter_degraded_reason is None
                    and not cycles_values
                    and not instruction_values
                    and not ipc_values
                    and not cache_miss_rate_values
                ):
                    counter_degraded_reason = "hw_counters_unavailable"
                kernel_runs.append(
                    {
                        "model": model,
                        "kernel": kernel,
                        "shape": shape,
                        "iterations": int(iterations),
                        "warmups": int(warmups),
                        "p50_ms": _percentile(latencies_ms, 50.0),
                        "p95_ms": _percentile(latencies_ms, 95.0),
                        "stddev_ms": stddev,
                        "cv_pct": cv,
                        "wall_p50_ms": _percentile(
                            [float(item["wall_seconds"]) * 1000.0 for item in samples],
                            50.0,
                        ),
                        "calls_mean": calls_mean,
                        "gflops_per_second": (
                            (flops_mean / max(seconds_mean, 1.0e-30)) / 1.0e9
                            if flops_mean > 0.0 and seconds_mean > 0.0
                            else 0.0
                        ),
                        "gbytes_per_second": (
                            (bytes_mean / max(seconds_mean, 1.0e-30)) / 1.0e9
                            if bytes_mean > 0.0 and seconds_mean > 0.0
                            else 0.0
                        ),
                        "cycles": statistics.fmean(cycles_values) if cycles_values else None,
                        "instructions": (
                            statistics.fmean(instruction_values) if instruction_values else None
                        ),
                        "ipc": statistics.fmean(ipc_values) if ipc_values else None,
                        "cache_miss_rate": (
                            statistics.fmean(cache_miss_rate_values)
                            if cache_miss_rate_values
                            else None
                        ),
                        "degraded_reason": counter_degraded_reason,
                        "cpp_file": cpp_file,
                        "cpp_function": cpp_function,
                        "estimated_recoverable_gain_pct": _recoverable_gain_pct(
                            seconds_mean,
                            wall_seconds_mean,
                        ),
                    }
                )
        kernel_runs.sort(
            key=lambda item: (
                float(item["estimated_recoverable_gain_pct"]),
                float(item["p95_ms"]),
            ),
            reverse=True,
        )
        return {
            "schema_version": "native_kernel_microbench.v2",
            "model": model,
            "degraded_reason": degraded_reason,
            "runs": kernel_runs,
        }
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--synthetic-lengths", type=str, default="32,64,128,256")
    parser.add_argument(
        "--kernel",
        action="append",
        choices=tuple(KERNEL_STAGE_MAP),
        default=None,
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def _parse_lengths(raw: str) -> list[int]:
    lengths: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        lengths.append(max(1, int(part)))
    return list(dict.fromkeys(lengths))


def main() -> int:
    args = parse_args()
    if args.benchmark_json:
        payload = {
            "schema_version": "native_kernel_microbench.v1",
            "kernels": _summary_rows_from_benchmark_json(Path(args.benchmark_json)),
        }
    else:
        if not args.model:
            raise SystemExit("--model is required unless --benchmark-json is supplied")
        payload = run_kernel_microbench(
            model=str(args.model),
            context_length=int(args.context_length),
            warmups=int(args.warmups),
            iterations=int(args.iterations),
            target_kernels=list(args.kernel or list(KERNEL_STAGE_MAP)),
            synthetic_lengths=_parse_lengths(args.synthetic_lengths),
        )
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.json or args.json_out:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        runs = list(payload.get("runs") or payload.get("kernels") or [])
        for item in runs:
            print(
                f"{item.get('model','')}\t{item.get('kernel','')}\t"
                f"{float(item.get('p50_ms', item.get('total_ms', 0.0))):.4f}\t"
                f"{float(item.get('estimated_recoverable_gain_pct', item.get('recoverable_gain_estimate_pct', 0.0))):.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
