#!/usr/bin/env python3
"""Benchmark the continuous QSG scheduler surface through the native QSG adapter."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.qsg.config import QSGConfig
from core.qsg.ollama_adapter import OllamaQSGAdapter

SCHEMA_VERSION = "continuous_qsg_benchmark.v1"

PROMPT_CLASSES = {
    "short": "Explain in two sentences how AVX2 helps CPU inference.",
    "medium": (
        "Explain how AVX2, OpenMP, and NUMA locality interact in a CPU-native "
        "LLM inference stack, using concrete mechanisms."
    ),
    "long": (
        "Write a concise technical note describing how a strict AVX2/OpenMP "
        "native QSG pipeline should schedule decode work, batch prefill, and "
        "state paging while avoiding Python hot-path fallbacks and preserving "
        "throughput fairness across concurrent requests."
    ),
}


@dataclass
class RequestResult:
    request_id: str
    concurrency: int
    prompt_class: str
    scheduler_policy: str
    ttft_ms: float = 0.0
    tpot_ms_mean: float = 0.0
    generated_token_estimate: int = 0
    output_text: str = ""
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class MatrixResult:
    concurrency: int
    max_active_requests: int
    prompt_class: str
    scheduler_policy: str
    batch_wait_timeout_ms: int
    state_page_rows: int
    max_prefill_rows_per_iteration: int
    continuous_interleaved_streams: bool
    requests: list[RequestResult] = field(default_factory=list)
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)


def _default_concurrency() -> list[int]:
    logical = max(1, int(os.cpu_count() or 1))
    candidates = [1, 2, 4, 8]
    return [value for value in candidates if value <= logical] or [1]


def _request_options(max_new_tokens: int) -> dict[str, Any]:
    return {
        "num_predict": int(max_new_tokens),
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
    }


def _run_request(
    adapter: OllamaQSGAdapter,
    *,
    request_id: str,
    prompt: str,
    max_new_tokens: int,
) -> RequestResult:
    started = time.perf_counter()
    first_emit: float | None = None
    last_emit: float | None = None
    emitted_chunks = 0
    output_parts: list[str] = []
    try:
        for chunk in adapter.stream_generate(
            prompt, options=_request_options(max_new_tokens)
        ):
            now = time.perf_counter()
            if first_emit is None:
                first_emit = now
            last_emit = now
            emitted_chunks += max(1, len(str(chunk).split()))
            output_parts.append(str(chunk))
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            concurrency=0,
            prompt_class="",
            scheduler_policy="",
            error=str(exc),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
    duration_ms = (time.perf_counter() - started) * 1000.0
    ttft_ms = ((first_emit - started) * 1000.0) if first_emit is not None else 0.0
    tpot_ms_mean = 0.0
    if emitted_chunks > 1 and first_emit is not None and last_emit is not None:
        tpot_ms_mean = ((last_emit - first_emit) * 1000.0) / float(emitted_chunks - 1)
    return RequestResult(
        request_id=request_id,
        concurrency=0,
        prompt_class="",
        scheduler_policy="",
        ttft_ms=ttft_ms,
        tpot_ms_mean=tpot_ms_mean,
        generated_token_estimate=emitted_chunks,
        output_text="".join(output_parts),
        duration_ms=duration_ms,
    )


def _run_matrix(
    *,
    model: str,
    prompt_class: str,
    concurrency: int,
    max_active_requests: int,
    scheduler_policy: str,
    state_page_rows: int,
    batch_wait_timeout_ms: int,
    max_prefill_rows_per_iteration: int,
    continuous_interleaved_streams: bool,
    max_new_tokens: int,
    context_length: int | None,
    decode_threads: int | None,
    batch_threads: int | None,
    ubatch: int | None,
) -> MatrixResult:
    restore_env = {
        "ANVIL_NATIVE_CTX_CAP": os.environ.get("ANVIL_NATIVE_CTX_CAP"),
        "ANVIL_NUM_THREADS": os.environ.get("ANVIL_NUM_THREADS"),
        "ANVIL_NUM_THREADS_DECODE": os.environ.get("ANVIL_NUM_THREADS_DECODE"),
        "ANVIL_NUM_THREADS_BATCH": os.environ.get("ANVIL_NUM_THREADS_BATCH"),
        "ANVIL_NUM_UBATCH": os.environ.get("ANVIL_NUM_UBATCH"),
    }
    if context_length is not None:
        os.environ["ANVIL_NATIVE_CTX_CAP"] = str(max(1, int(context_length)))
    if decode_threads is not None or batch_threads is not None:
        base_threads = max(
            1,
            int(
                max(
                    int(decode_threads or 0),
                    int(batch_threads or 0),
                )
                or 1
            ),
        )
        os.environ["ANVIL_NUM_THREADS"] = str(base_threads)
    if decode_threads is not None:
        os.environ["ANVIL_NUM_THREADS_DECODE"] = str(max(1, int(decode_threads)))
    if batch_threads is not None:
        os.environ["ANVIL_NUM_THREADS_BATCH"] = str(max(1, int(batch_threads)))
    if ubatch is not None:
        os.environ["ANVIL_NUM_UBATCH"] = str(max(1, int(ubatch)))

    config = QSGConfig(
        continuous_batching_enabled=True,
        scheduler_policy=scheduler_policy,
        max_active_requests=max_active_requests,
        batch_wait_timeout_ms=batch_wait_timeout_ms,
        max_prefill_rows_per_iteration=max_prefill_rows_per_iteration,
        state_page_rows=state_page_rows,
        continuous_interleaved_streams=continuous_interleaved_streams,
        use_coconut=True,
        use_grover=True,
    )
    adapter = OllamaQSGAdapter(model, config=config)
    prompt = PROMPT_CLASSES[prompt_class]
    results: list[RequestResult] = []
    metrics_snapshot: dict[str, Any] = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    _run_request,
                    adapter,
                    request_id=f"{prompt_class}-{scheduler_policy}-{index}",
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                for index in range(concurrency)
            ]
            for future in concurrent.futures.as_completed(futures):
                item = future.result()
                item.concurrency = concurrency
                item.prompt_class = prompt_class
                item.scheduler_policy = scheduler_policy
                results.append(item)
        metrics_snapshot = dict(
            (adapter.get_runtime_status().get("continuous_batching") or {})
        )
    finally:
        try:
            continuous_engine = getattr(adapter, "_continuous_engine", None)
            if continuous_engine is not None:
                continuous_engine.shutdown()
        except Exception:
            pass
        for key, value in restore_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return MatrixResult(
        concurrency=concurrency,
        max_active_requests=max_active_requests,
        prompt_class=prompt_class,
        scheduler_policy=scheduler_policy,
        batch_wait_timeout_ms=batch_wait_timeout_ms,
        state_page_rows=state_page_rows,
        max_prefill_rows_per_iteration=max_prefill_rows_per_iteration,
        continuous_interleaved_streams=continuous_interleaved_streams,
        requests=sorted(results, key=lambda item: item.request_id),
        metrics_snapshot=metrics_snapshot,
    )


def _fairness_score(token_counts: list[int]) -> float:
    positive = [count for count in token_counts if count > 0]
    if len(positive) <= 1:
        return 1.0
    return min(positive) / max(positive)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--decode-threads", type=int, default=None)
    parser.add_argument("--batch-threads", type=int, default=None)
    parser.add_argument("--ubatch", type=int, default=None)
    parser.add_argument("--concurrency", type=str, default="1,2,4,8")
    parser.add_argument("--max-active-requests", type=str, default="")
    parser.add_argument(
        "--prompt-class", action="append", choices=tuple(PROMPT_CLASSES), default=None
    )
    parser.add_argument(
        "--scheduler-policy",
        action="append",
        choices=("fcfs", "priority"),
        default=None,
    )
    parser.add_argument("--state-page-rows", type=str, default="64,128,256")
    parser.add_argument("--batch-wait-timeout-ms", type=str, default="1,2,4")
    parser.add_argument(
        "--max-prefill-rows-per-iteration", type=str, default="512,1024"
    )
    parser.add_argument("--interleaved-streams", type=str, default="false,true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def _parse_int_csv(raw: str) -> list[int]:
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(max(1, int(part)))
    return list(dict.fromkeys(values))


def _parse_bool_csv(raw: str) -> list[bool]:
    values: list[bool] = []
    for part in str(raw).split(","):
        token = part.strip().lower()
        if not token:
            continue
        values.append(token in {"1", "true", "yes", "on"})
    return list(dict.fromkeys(values)) or [False]


def main() -> int:
    args = parse_args()
    prompt_classes = args.prompt_class or ["short", "medium", "long"]
    policies = args.scheduler_policy or ["fcfs", "priority"]
    state_page_rows_values = _parse_int_csv(args.state_page_rows)
    batch_wait_timeout_values = _parse_int_csv(args.batch_wait_timeout_ms)
    max_prefill_rows_values = _parse_int_csv(args.max_prefill_rows_per_iteration)
    interleaved_stream_values = _parse_bool_csv(args.interleaved_streams)
    matrix_results: list[MatrixResult] = []
    for concurrency in _parse_int_csv(args.concurrency):
        max_active_requests_values = _parse_int_csv(args.max_active_requests) or [
            concurrency
        ]
        for prompt_class in prompt_classes:
            for scheduler_policy in policies:
                for max_active_requests in max_active_requests_values:
                    for state_page_rows in state_page_rows_values:
                        for batch_wait_timeout_ms in batch_wait_timeout_values:
                            for max_prefill_rows in max_prefill_rows_values:
                                for interleaved_streams in interleaved_stream_values:
                                    matrix_results.append(
                                        _run_matrix(
                                            model=args.model,
                                            prompt_class=prompt_class,
                                            concurrency=concurrency,
                                            max_active_requests=max_active_requests,
                                            scheduler_policy=scheduler_policy,
                                            state_page_rows=state_page_rows,
                                            batch_wait_timeout_ms=batch_wait_timeout_ms,
                                            max_prefill_rows_per_iteration=max_prefill_rows,
                                            continuous_interleaved_streams=interleaved_streams,
                                            max_new_tokens=int(args.max_new_tokens),
                                            context_length=args.context_length,
                                            decode_threads=args.decode_threads,
                                            batch_threads=args.batch_threads,
                                            ubatch=args.ubatch,
                                        )
                                    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "results": [],
    }
    for item in matrix_results:
        request_ttfts = [
            request.ttft_ms for request in item.requests if not request.error
        ]
        request_tpots = [
            request.tpot_ms_mean for request in item.requests if not request.error
        ]
        token_counts = [
            request.generated_token_estimate
            for request in item.requests
            if not request.error
        ]
        payload["results"].append(
            {
                "concurrency": item.concurrency,
                "max_active_requests": item.max_active_requests,
                "prompt_class": item.prompt_class,
                "scheduler_policy": item.scheduler_policy,
                "batch_wait_timeout_ms": item.batch_wait_timeout_ms,
                "state_page_rows": item.state_page_rows,
                "max_prefill_rows_per_iteration": item.max_prefill_rows_per_iteration,
                "continuous_interleaved_streams": item.continuous_interleaved_streams,
                "ttft_ms_p50": (
                    statistics.median(request_ttfts) if request_ttfts else 0.0
                ),
                "ttft_ms_p95": max(request_ttfts) if request_ttfts else 0.0,
                "tpot_ms_p50": (
                    statistics.median(request_tpots) if request_tpots else 0.0
                ),
                "queue_wait_ms_p95": float(
                    item.metrics_snapshot.get("qsg_queue_wait_ms_p95", 0.0)
                ),
                "queue_wait_ms_p99": float(
                    item.metrics_snapshot.get("qsg_queue_wait_ms_p99", 0.0)
                ),
                "scheduler_iteration_ms_p95": float(
                    item.metrics_snapshot.get("qsg_scheduler_iteration_ms_p95", 0.0)
                ),
                "decode_tps_global": float(
                    item.metrics_snapshot.get("qsg_decode_tps_global", 0.0)
                ),
                "decode_goodput_tps": float(
                    item.metrics_snapshot.get("qsg_decode_tps_global", 0.0)
                )
                * _fairness_score(token_counts),
                "decode_tps_per_agent": dict(
                    item.metrics_snapshot.get("qsg_decode_tps_per_agent", {})
                ),
                "state_fragmentation_ratio": float(
                    item.metrics_snapshot.get("qsg_state_fragmentation_ratio", 0.0)
                ),
                "coconut_active_paths": int(
                    item.metrics_snapshot.get("qsg_coconut_active_paths", 0)
                ),
                "coconut_entropy_mean": float(
                    item.metrics_snapshot.get("qsg_coconut_entropy_mean", 0.0)
                ),
                "drift_overhead_percent": float(
                    item.metrics_snapshot.get("qsg_drift_overhead_percent", 0.0)
                ),
                "batched_prefill_token_id_calls": int(
                    item.metrics_snapshot.get("qsg_batched_prefill_token_id_calls", 0)
                ),
                "batched_prefill_token_id_tokens": int(
                    item.metrics_snapshot.get("qsg_batched_prefill_token_id_tokens", 0)
                ),
                "fairness": _fairness_score(token_counts),
                "request_results": [asdict(request) for request in item.requests],
                "continuous_metrics": item.metrics_snapshot,
            }
        )

    if args.json_out:
        output_path = pathlib.Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload["results"]:
            print(
                f"{item['concurrency']}\t{item['max_active_requests']}\t{item['prompt_class']}\t{item['scheduler_policy']}\t"
                f"{item['decode_tps_global']:.2f}\t{item['ttft_ms_p95']:.2f}\t"
                f"{item['queue_wait_ms_p95']:.2f}\t{item['fairness']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
