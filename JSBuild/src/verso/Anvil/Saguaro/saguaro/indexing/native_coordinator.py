"""Authoritative native-first indexing coordinator."""

from __future__ import annotations

import concurrent.futures
import gc
import multiprocessing
import os
import pickle
import resource
import sys
import time
from typing import Any

from saguaro.indexing.auto_scaler import plan_index_concurrency
from saguaro.indexing.engine import SharedProjectionManager
from saguaro.indexing.native_worker import process_batch_worker_native


def _run_process_batch_worker_native(args: tuple[Any, ...]) -> Any:
    return process_batch_worker_native(*args)


def _available_cpu_count() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    return max(1, os.cpu_count() or 1)


def _resolve_batch_size(batch_size: int) -> int:
    if batch_size <= 0:
        try:
            batch_size = max(1, int(os.getenv("SAGUARO_INDEX_BATCH_SIZE", "250")))
        except ValueError:
            batch_size = 250
    return max(1, int(batch_size))


def _parallel_file_batch_size(
    *,
    requested: int,
    file_count: int,
    workers: int,
) -> int:
    raw = os.getenv("SAGUARO_INDEX_FILE_BATCH_SIZE")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if file_count <= 0:
        return 1
    target_waves = max(4, int(workers) * 8)
    adaptive = max(1, (int(file_count) + target_waves - 1) // target_waves)
    max_batch_size = max(4, min(250, max(1, int(workers)) * 32))
    return max(1, min(int(requested), adaptive, max_batch_size))


def _batch_count(item_count: int, batch_size: int) -> int:
    resolved = _resolve_batch_size(batch_size)
    return max(0, (max(0, int(item_count)) + resolved - 1) // resolved)


def _file_size_bytes(path: str) -> int:
    try:
        return max(1, int(os.path.getsize(path)))
    except OSError:
        return 1


def _target_batch_bytes(*, paths: list[str], workers: int) -> int:
    raw = os.getenv("SAGUARO_INDEX_BATCH_BYTES_MB", "").strip()
    if raw:
        try:
            return max(64 * 1024, int(float(raw) * 1024 * 1024))
        except ValueError:
            pass
    total_bytes = sum(_file_size_bytes(path) for path in paths)
    target_waves = max(4, max(1, int(workers)) * 8)
    adaptive = total_bytes // target_waves if total_bytes > 0 else 0
    return max(256 * 1024, min(2 * 1024 * 1024, adaptive))


def _build_file_batches(
    items: list[str],
    *,
    batch_size: int,
    workers: int,
) -> list[list[str]]:
    if not items:
        return []
    resolved = _resolve_batch_size(batch_size)
    target_bytes = _target_batch_bytes(paths=items, workers=workers)
    ordered = sorted(
        {os.path.abspath(path) for path in items if path},
        key=lambda path: (_file_size_bytes(path), path),
        reverse=True,
    )
    batches: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for path in ordered:
        file_bytes = _file_size_bytes(path)
        if current and (
            len(current) >= resolved or current_bytes + file_bytes > target_bytes
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(path)
        current_bytes += file_bytes
        if file_bytes >= target_bytes:
            batches.append(current)
            current = []
            current_bytes = 0
    if current:
        batches.append(current)
    return batches

def _current_rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return 0.0


def _threads_per_worker() -> int:
    raw = os.getenv("SAGUARO_NATIVE_THREADS_PER_WORKER") or os.getenv(
        "SAGUARO_NATIVE_NUM_THREADS"
    )
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    # Cold indexing on large repos is parser-heavy; default to one native
    # thread per worker so we can saturate the host with more file workers.
    return 1


def _worker_count(batch_count: int, threads_per_worker: int) -> int:
    disable_parallel = str(
        os.getenv("SAGUARO_DISABLE_PARALLEL_INDEXING", "0")
    ).lower() in {"1", "true", "yes"}
    if disable_parallel:
        return 1
    raw = os.getenv("SAGUARO_INDEX_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cpu_count = _available_cpu_count()
    per_worker = max(1, threads_per_worker)
    capacity = max(1, cpu_count // per_worker)
    try:
        max_workers = max(1, int(os.getenv("SAGUARO_INDEX_MAX_WORKERS", str(cpu_count))))
    except ValueError:
        max_workers = cpu_count
    return max(1, min(batch_count, capacity, max_workers))


def _projection_bytes(*, vocab_size: int, active_dim: int) -> int:
    return int(vocab_size) * int(active_dim) * 4


def _store_path_aliases(repo_path: str, file_paths: list[str]) -> list[str]:
    aliases: set[str] = set()
    repo_root = os.path.abspath(repo_path)
    for file_path in file_paths:
        if not file_path:
            continue
        abs_path = os.path.abspath(file_path)
        aliases.add(abs_path)
        try:
            aliases.add(os.path.relpath(abs_path, repo_root).replace("\\", "/"))
        except ValueError:
            pass
    return sorted(aliases)


def _remove_store_rows(engine: Any, repo_path: str, file_paths: list[str]) -> None:
    aliases = _store_path_aliases(repo_path, file_paths)
    if not aliases:
        return
    if hasattr(engine.store, "remove_files"):
        engine.store.remove_files(aliases)
    else:
        for stale_path in aliases:
            if hasattr(engine.store, "remove_file"):
                engine.store.remove_file(stale_path)


def _single_runtime_mode() -> bool:
    raw = str(os.getenv("SAGUARO_INDEX_SINGLE_RUNTIME", "0")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", "", "auto"}:
        return False
    return False


def _ingest_spill_file(engine: Any, spill_path: str | None) -> tuple[int, int]:
    if not spill_path or not os.path.exists(spill_path):
        return 0, 0

    indexed_files = 0
    indexed_entities = 0
    try:
        with open(spill_path, "rb") as handle:
            while True:
                try:
                    meta_list, vectors = pickle.load(handle)
                except EOFError:
                    break
                file_count, entity_count = engine.ingest_worker_result(meta_list, vectors)
                indexed_files += file_count
                indexed_entities += entity_count
    finally:
        try:
            os.remove(spill_path)
        except OSError:
            pass
    return indexed_files, indexed_entities


def _ingest_batches_in_parallel(
    *,
    engine: Any,
    batches: list[list[str]],
    batch_count: int,
    workers: int,
    active_dim: int,
    total_dim: int,
    vocab_size: int,
    projection_resource: str,
    repo_path: str,
    num_threads: int,
    batch_capacity: int | None,
    max_total_text_chars: int | None,
    max_in_flight: int | None = None,
    enforce_quotas: bool | None = None,
) -> tuple[int, int, set[str], dict[str, float | int]]:
    indexed_files = 0
    indexed_entities = 0
    touched_files: set[str] = set()
    gc_interval = max(1, int(os.getenv("SAGUARO_INDEX_GC_INTERVAL", "8") or 8))
    completed_batches = 0
    metrics: dict[str, float | int] = {
        "parse_seconds": 0.0,
        "pipeline_seconds": 0.0,
        "files_with_entities": 0,
        "emitted_vector_bytes": 0,
        "queue_wait_seconds": 0.0,
        "queue_depth_max": 0,
        "quota_hits": 0,
        "entities_dropped_by_quota": 0,
        "text_chars_processed": 0,
        "peak_rss_mb": _current_rss_mb(),
    }
    batch_iter = iter(batches)
    context_name = _process_pool_context()
    if workers <= 1:
        for batch in batch_iter:
            submitted_at = time.perf_counter()
            streamed_entities = 0

            def _stream_ingest(meta_list: list[dict], vectors: list[Any]) -> None:
                nonlocal streamed_entities
                _file_count, entity_count = engine.ingest_worker_result(meta_list, vectors)
                streamed_entities += entity_count

            if batch:
                _remove_store_rows(engine, repo_path, list(batch))
            result = process_batch_worker_native(
                batch,
                active_dim,
                total_dim,
                vocab_size,
                projection_resource,
                None,
                repo_path,
                max(1, num_threads),
                batch_capacity,
                max_total_text_chars,
                enforce_quotas,
                _stream_ingest,
            )
            metrics["queue_depth_max"] = max(int(metrics["queue_depth_max"]), 1)
            if submitted_at > 0.0:
                metrics["queue_wait_seconds"] += max(
                    0.0, time.perf_counter() - submitted_at
                )
            if len(result) == 4:
                meta_list, vectors, touched, batch_metrics = result
            else:
                meta_list, vectors, touched = result
                batch_metrics = {}
            spill_path = str(batch_metrics.get("spill_path") or "")
            if meta_list and vectors is not None:
                file_count, entity_count = engine.ingest_worker_result(meta_list, vectors)
                indexed_files += file_count
                indexed_entities += entity_count
            elif spill_path:
                file_count, entity_count = _ingest_spill_file(engine, spill_path)
                indexed_files += file_count
                indexed_entities += entity_count
            else:
                indexed_files += int(batch_metrics.get("files_with_entities", 0) or 0)
                indexed_entities += int(streamed_entities)
            touched_files.update(touched)
            metrics["parse_seconds"] += float(batch_metrics.get("parse_seconds", 0.0) or 0.0)
            metrics["pipeline_seconds"] += float(
                batch_metrics.get("pipeline_seconds", 0.0) or 0.0
            )
            metrics["files_with_entities"] += int(
                batch_metrics.get("files_with_entities", 0) or 0
            )
            metrics["emitted_vector_bytes"] += int(
                batch_metrics.get("emitted_vector_bytes", 0) or 0
            )
            metrics["quota_hits"] += int(batch_metrics.get("quota_hits", 0) or 0)
            metrics["entities_dropped_by_quota"] += int(
                batch_metrics.get("entities_dropped_by_quota", 0) or 0
            )
            metrics["text_chars_processed"] += int(
                batch_metrics.get("text_chars_processed", 0) or 0
            )
            metrics["peak_rss_mb"] = max(
                float(metrics.get("peak_rss_mb", 0.0) or 0.0),
                _current_rss_mb(),
            )
            completed_batches += 1
            del meta_list, vectors
            if completed_batches % gc_interval == 0:
                gc.collect()
    else:
        ctx = multiprocessing.get_context(context_name)
        pool_size = max(1, min(batch_count, workers))
        target_in_flight = max(1, min(pool_size, int(max_in_flight or pool_size)))
        metrics["queue_depth_max"] = max(int(metrics["queue_depth_max"]), target_in_flight)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=pool_size,
            mp_context=ctx,
        ) as executor:
            batch_iter = iter(batches)
            pending: dict[concurrent.futures.Future[Any], float] = {}

            def _submit_next() -> bool:
                try:
                    batch = next(batch_iter)
                except StopIteration:
                    return False
                submitted_at = time.perf_counter()
                future = executor.submit(
                    process_batch_worker_native,
                    batch,
                    active_dim,
                    total_dim,
                    vocab_size,
                    projection_resource,
                    None,
                    repo_path,
                    max(1, num_threads),
                    batch_capacity,
                    max_total_text_chars,
                    enforce_quotas,
                )
                pending[future] = submitted_at
                metrics["queue_depth_max"] = max(
                    int(metrics["queue_depth_max"]),
                    len(pending),
                )
                return True

            for _ in range(target_in_flight):
                if not _submit_next():
                    break

            while pending:
                done, _ = concurrent.futures.wait(
                    list(pending.keys()),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    submitted_at = pending.pop(future, 0.0)
                    if submitted_at > 0.0:
                        metrics["queue_wait_seconds"] += max(
                            0.0,
                            time.perf_counter() - submitted_at,
                        )
                    result = future.result()
                    if len(result) == 4:
                        meta_list, vectors, touched, batch_metrics = result
                    else:
                        meta_list, vectors, touched = result
                        batch_metrics = {}
                    spill_path = str(batch_metrics.get("spill_path") or "")
                    if touched:
                        _remove_store_rows(engine, repo_path, list(touched))
                    if meta_list and vectors is not None:
                        file_count, entity_count = engine.ingest_worker_result(
                            meta_list, vectors
                        )
                    elif spill_path:
                        file_count, entity_count = _ingest_spill_file(
                            engine,
                            spill_path,
                        )
                    else:
                        file_count, entity_count = (0, 0)
                    indexed_files += file_count
                    indexed_entities += entity_count
                    touched_files.update(touched)
                    metrics["parse_seconds"] += float(
                        batch_metrics.get("parse_seconds", 0.0) or 0.0
                    )
                    metrics["pipeline_seconds"] += float(
                        batch_metrics.get("pipeline_seconds", 0.0) or 0.0
                    )
                    metrics["files_with_entities"] += int(
                        batch_metrics.get("files_with_entities", 0) or 0
                    )
                    metrics["emitted_vector_bytes"] += int(
                        batch_metrics.get("emitted_vector_bytes", 0) or 0
                    )
                    metrics["quota_hits"] += int(
                        batch_metrics.get("quota_hits", 0) or 0
                    )
                    metrics["entities_dropped_by_quota"] += int(
                        batch_metrics.get("entities_dropped_by_quota", 0) or 0
                    )
                    metrics["text_chars_processed"] += int(
                        batch_metrics.get("text_chars_processed", 0) or 0
                    )
                    metrics["peak_rss_mb"] = max(
                        float(metrics.get("peak_rss_mb", 0.0) or 0.0),
                        _current_rss_mb(),
                    )
                    completed_batches += 1
                    del meta_list, vectors
                    if completed_batches % gc_interval == 0:
                        gc.collect()
                while len(pending) < target_in_flight and _submit_next():
                    continue

    metrics["parse_seconds"] = round(float(metrics["parse_seconds"]), 6)
    metrics["pipeline_seconds"] = round(float(metrics["pipeline_seconds"]), 6)
    metrics["queue_wait_seconds"] = round(float(metrics["queue_wait_seconds"]), 6)
    metrics["peak_rss_mb"] = round(float(metrics["peak_rss_mb"]), 6)
    return indexed_files, indexed_entities, touched_files, metrics


def _process_pool_context() -> str:
    raw = str(os.getenv("SAGUARO_INDEX_MP_CONTEXT", "")).strip().lower()
    if raw:
        return raw
    if sys.platform.startswith("linux"):
        return "fork"
    return "spawn"


def run_native_index_coordinator(
    *,
    engine: Any,
    file_paths: list[str],
    remove_before_index: list[str] | None = None,
    batch_size: int = 250,
) -> dict[str, Any]:
    """Run the shared native indexing pipeline over the requested files."""
    paths = sorted({os.path.abspath(path) for path in file_paths if path})
    removals = sorted({os.path.abspath(path) for path in (remove_before_index or []) if path})
    if removals:
        for stale_path in removals:
            engine.tracker.state.pop(stale_path, None)

    if not paths:
        engine.commit()
        return {
            "indexed_files": 0,
            "indexed_entities": 0,
            "touched_files": [],
            "batches_processed": 0,
            "workers": 0,
            "threads_per_worker": 0,
            "mode": "noop",
            "parse_seconds": 0.0,
            "pipeline_seconds": 0.0,
            "files_with_entities": 0,
        }

    resolved_batch_size = _resolve_batch_size(batch_size)
    batch_count = _batch_count(len(paths), resolved_batch_size)
    single_runtime = _single_runtime_mode()
    projection_shared = True
    plan = plan_index_concurrency(
        candidate_files=len(paths),
        batch_count=batch_count,
        batch_size=resolved_batch_size,
        active_dim=int(engine.active_dim),
        total_dim=int(engine.total_dim),
        vocab_size=int(engine.vocab_size),
        threads_per_worker_hint=_threads_per_worker(),
        cpu_threads=_available_cpu_count(),
        projection_shared=projection_shared,
    )
    threads_per_worker = int(plan.get("threads_per_worker", _threads_per_worker()) or 1)
    workers = int(
        min(
            batch_count,
            max(1, int(plan.get("workers", _worker_count(batch_count, threads_per_worker)))),
        )
    )
    requested_worker_cap = max(1, workers)

    if single_runtime:
        workers = 1
        threads_per_worker = max(1, int(plan.get("cpu_threads", _available_cpu_count()) or 1))
        plan["workers"] = workers
        plan["threads_per_worker"] = threads_per_worker
        plan["total_threads"] = threads_per_worker
        plan["queue_depth_target"] = batch_count
        plan["quota_mode"] = "single_runtime_budgeted"
    else:
        resolved_batch_size = _parallel_file_batch_size(
            requested=resolved_batch_size,
            file_count=len(paths),
            workers=workers,
        )
        batch_count = _batch_count(len(paths), resolved_batch_size)
        plan = plan_index_concurrency(
            candidate_files=len(paths),
            batch_count=batch_count,
            batch_size=resolved_batch_size,
            active_dim=int(engine.active_dim),
            total_dim=int(engine.total_dim),
            vocab_size=int(engine.vocab_size),
            threads_per_worker_hint=threads_per_worker,
            cpu_threads=_available_cpu_count(),
            projection_shared=projection_shared,
        )
        threads_per_worker = int(
            plan.get("threads_per_worker", _threads_per_worker()) or 1
        )
        workers = int(
            min(
                batch_count,
                requested_worker_cap,
                max(
                    1,
                    int(
                        plan.get(
                            "workers",
                            _worker_count(batch_count, threads_per_worker),
                        )
                    ),
                ),
            )
        )
        plan["queue_depth_target"] = min(
            batch_count,
            max(
                workers,
                int(plan.get("queue_depth_target", workers * 2) or workers * 2),
            ),
        )

    batches = _build_file_batches(
        paths,
        batch_size=resolved_batch_size,
        workers=workers,
    )
    batch_count = len(batches)
    if batch_count > 0:
        plan = plan_index_concurrency(
            candidate_files=len(paths),
            batch_count=batch_count,
            batch_size=resolved_batch_size,
            active_dim=int(engine.active_dim),
            total_dim=int(engine.total_dim),
            vocab_size=int(engine.vocab_size),
            threads_per_worker_hint=threads_per_worker,
            cpu_threads=_available_cpu_count(),
            projection_shared=projection_shared,
        )
        if single_runtime:
            workers = 1
            threads_per_worker = max(
                1,
                int(plan.get("cpu_threads", _available_cpu_count()) or 1),
            )
            plan["workers"] = workers
            plan["threads_per_worker"] = threads_per_worker
            plan["total_threads"] = threads_per_worker
            plan["queue_depth_target"] = batch_count
            plan["quota_mode"] = "single_runtime_budgeted"
        else:
            threads_per_worker = int(
                plan.get("threads_per_worker", threads_per_worker) or threads_per_worker
            )
            workers = int(
                min(
                    batch_count,
                    requested_worker_cap,
                    max(
                        1,
                        int(
                            plan.get(
                                "workers",
                                _worker_count(batch_count, threads_per_worker),
                            )
                        ),
                    ),
                )
            )
            plan["queue_depth_target"] = min(
                batch_count,
                max(
                    workers,
                    int(plan.get("queue_depth_target", workers * 2) or workers * 2),
                ),
            )
            rebalanced_batches = _build_file_batches(
                paths,
                batch_size=resolved_batch_size,
                workers=workers,
            )
            if len(rebalanced_batches) != batch_count:
                batches = rebalanced_batches
                batch_count = len(batches)

    batch_capacity = int(plan.get("batch_capacity", 0) or 0) or None
    max_total_text_chars = int(plan.get("max_total_text_chars", 0) or 0) or None
    enforce_quotas = (
        True
        if (batch_capacity is not None and batch_capacity > 0)
        or (max_total_text_chars is not None and max_total_text_chars > 0)
        else None
    )
    max_in_flight = int(plan.get("queue_depth_target", workers * 2) or workers * 2)
    indexed_files = 0
    indexed_entities = 0
    touched_files: set[str] = set()
    import saguaro.indexing.native_worker as native_worker_module

    native_worker_module._close_worker_shm()
    native_worker_module._reset_worker_projection()
    engine.create_shared_projection()
    projection_manager = getattr(engine, "projection_manager", None)
    projection_resource = str(
        getattr(
            projection_manager,
            "resource_name",
            os.path.join(os.getcwd(), "projection_runtime.bin"),
        )
    )
    if single_runtime and projection_manager is not None:
        native_worker_module._worker_projection = engine.projection_manager.get_projection()
    try:
        file_count, entity_count, touched, metrics = _ingest_batches_in_parallel(
            engine=engine,
            batches=batches,
            batch_count=batch_count,
            workers=workers,
            active_dim=engine.active_dim,
            total_dim=engine.total_dim,
            vocab_size=engine.vocab_size,
            projection_resource=projection_resource,
            repo_path=str(getattr(engine, "repo_path", os.getcwd())),
            num_threads=threads_per_worker,
            batch_capacity=batch_capacity,
            max_total_text_chars=max_total_text_chars,
            max_in_flight=max_in_flight,
            enforce_quotas=enforce_quotas,
        )
        indexed_files += file_count
        indexed_entities += entity_count
        touched_files.update(touched)
        if touched_files:
            engine.tracker.update(sorted(touched_files), compute_hash=False)
        else:
            engine.tracker.save()
        engine.commit()
    finally:
        native_worker_module._close_worker_shm()
        native_worker_module._reset_worker_projection()
        engine.cleanup_shared_projection()
        gc.collect()

    return {
        "indexed_files": indexed_files,
        "indexed_entities": indexed_entities,
        "touched_files": sorted(touched_files),
        "batches_processed": batch_count,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "total_threads": int(plan.get("total_threads", workers * threads_per_worker)),
        "cpu_threads": int(plan.get("cpu_threads", _available_cpu_count())),
        "rss_budget_mb": None
        if single_runtime
        else float(plan.get("rss_budget_mb", 0.0) or 0.0),
        "predicted_rss_mb": float(plan.get("predicted_rss_mb", 0.0) or 0.0),
        "observed_rss_mb": float(metrics.get("peak_rss_mb", 0.0) or 0.0),
        "worker_rss_mb": float(plan.get("worker_rss_mb", 0.0) or 0.0),
        "projection_bytes": _projection_bytes(
            vocab_size=engine.vocab_size,
            active_dim=engine.active_dim,
        ),
        "projection_mb": float(plan.get("projection_mb", 0.0) or 0.0),
        "projection_shared": bool(plan.get("projection_shared", False)),
        "batch_capacity": int(plan.get("batch_capacity", 0) or 0),
        "max_total_text_chars": int(plan.get("max_total_text_chars", 0) or 0),
        "queue_depth_target": int(plan.get("queue_depth_target", 0) or 0),
        "quota_mode": str(plan.get("quota_mode", "")),
        "mode": "native_single_runtime" if single_runtime else "native_process_pool",
        "parse_seconds": float(metrics.get("parse_seconds", 0.0) or 0.0),
        "pipeline_seconds": float(metrics.get("pipeline_seconds", 0.0) or 0.0),
        "files_with_entities": int(metrics.get("files_with_entities", 0) or 0),
        "worker_emitted_bytes": int(metrics.get("emitted_vector_bytes", 0) or 0),
        "queue_wait_seconds": float(metrics.get("queue_wait_seconds", 0.0) or 0.0),
        "queue_depth_max": int(metrics.get("queue_depth_max", 0) or 0),
        "quota_hits": int(metrics.get("quota_hits", 0) or 0),
        "entities_dropped_by_quota": int(
            metrics.get("entities_dropped_by_quota", 0) or 0
        ),
        "text_chars_processed": int(metrics.get("text_chars_processed", 0) or 0),
    }
