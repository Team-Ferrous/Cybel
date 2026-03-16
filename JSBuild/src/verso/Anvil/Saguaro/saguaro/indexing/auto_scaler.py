"""Budget-aware autoscaling and runtime profiling for Saguaro indexing."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

from saguaro.utils.file_utils import build_corpus_manifest

logger = logging.getLogger(__name__)

_CACHE_VERSION = 2
_PROFILE_VERSION = 1
_MEMORY_HEADROOM_MB = 256.0


def _saguaro_dir(path: str) -> Path:
    return Path(path).resolve() / ".saguaro"


def _stats_cache_path(path: str) -> Path:
    return _saguaro_dir(path) / "repo_stats_cache.json"


def _perf_profile_path(path: str) -> Path:
    return _saguaro_dir(path) / "perf_profiles.json"


def _ensure_saguaro_dir(path: str) -> None:
    _saguaro_dir(path).mkdir(parents=True, exist_ok=True)


def _language_from_path(path: str) -> str | None:
    ext = Path(path).suffix.lower()
    return {
        ".py": "Python",
        ".cc": "C++",
        ".cpp": "C++",
        ".cxx": "C++",
        ".c": "C",
        ".h": "C/C++ Header",
        ".hpp": "C++ Header",
        ".hh": "C++ Header",
        ".hxx": "C++ Header",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".md": "Markdown",
    }.get(ext)


def count_loc(path: str) -> tuple[int, dict[str, int]]:
    """Approximate lines of code using the authoritative corpus manifest."""
    total_lines = 0
    language_breakdown: dict[str, int] = {}
    manifest = build_corpus_manifest(path)
    for file_path in manifest.files:
        lang = _language_from_path(file_path)
        if not lang:
            continue
        try:
            with open(file_path, "rb") as handle:
                lines = sum(1 for _ in handle)
        except OSError:
            continue
        total_lines += lines
        language_breakdown[lang] = language_breakdown.get(lang, 0) + lines
    return total_lines, language_breakdown


def _scan_repo_stats(path: str) -> tuple[int, int, dict[str, int]]:
    manifest = build_corpus_manifest(path)
    loc, languages = count_loc(path)
    return len(manifest.files), loc, languages


def calculate_ideal_dim(loc: int, enterprise_mode: bool = False) -> int:
    """Choose an active semantic dimension from corpus size."""
    if enterprise_mode or loc >= 2_000_000:
        return 16384
    if loc >= 500_000:
        return 12288
    if loc >= 10_000:
        return 8192
    return 4096


def allocate_dark_space(dim: int, buffer_ratio: float = 0.40) -> int:
    """Reserve logical darkspace without forcing physical hot-store padding."""
    minimum = max(
        1,
        int(math.ceil(dim * (1.0 + max(0.0, float(buffer_ratio))))),
    )
    power = 1
    while power < minimum:
        power <<= 1
    return power


def _predict_store_mb(*, candidate_files: int, active_dim: int) -> float:
    entities_per_file = 6.0 if candidate_files > 0 else 1.0
    rows = max(1.0, float(candidate_files) * entities_per_file)
    return rows * float(active_dim) * 4.0 / (1024.0 * 1024.0)


def _predict_projection_mb(*, vocab_size: int, active_dim: int) -> float:
    return float(vocab_size) * float(active_dim) * 4.0 / (1024.0 * 1024.0)


def index_rss_budget_mb(
    rss_budget_mb: float | None = None,
    *,
    cpu_threads: int | None = None,
    threads_per_worker: int = 1,
    worker_rss_mb: float = 0.0,
    projection_mb: float = 0.0,
    store_mb: float = 0.0,
) -> float:
    """Normalize explicit RSS budgets and compute a CPU-capacity default."""
    if rss_budget_mb is not None and float(rss_budget_mb) > 0.0:
        return round(float(rss_budget_mb), 3)
    resolved_cpu = max(1, int(cpu_threads or os.cpu_count() or 1))
    resolved_threads = max(1, int(threads_per_worker or 1))
    cpu_workers = max(1, resolved_cpu // resolved_threads)
    default_budget = max(
        _MEMORY_HEADROOM_MB + float(worker_rss_mb) * float(cpu_workers),
        _MEMORY_HEADROOM_MB + float(projection_mb) * 2.0 + float(store_mb) * 0.25,
    )
    return round(default_budget, 3)


def _cache_payload(
    *,
    path: str,
    candidate_files: int,
    loc: int,
    languages: dict[str, int],
) -> dict[str, Any]:
    active_dim = calculate_ideal_dim(loc)
    if candidate_files >= 1024:
        active_dim = min(active_dim, 2048)
    elif candidate_files >= 256:
        active_dim = min(active_dim, 4096)
    total_dim = allocate_dark_space(active_dim)
    predicted_store_mb = _predict_store_mb(
        candidate_files=candidate_files,
        active_dim=active_dim,
    )
    return {
        "cache_version": _CACHE_VERSION,
        "generated_at": time.time(),
        "path": str(Path(path).resolve()),
        "candidate_files": int(candidate_files),
        "loc": int(loc),
        "languages": dict(languages),
        "active_dim": int(active_dim),
        "total_dim": int(total_dim),
        "dark_space_ratio": 1.0 - (float(active_dim) / float(total_dim)),
        "predicted_store_mb": round(predicted_store_mb, 3),
        "autoscaler_contract_version": 2,
    }


def get_repo_stats_and_config(
    path: str,
    *,
    force_refresh: bool = False,
    ttl_seconds: int = 300,
) -> dict[str, float | int | dict[str, int]]:
    """Return cached, budget-aware repository sizing guidance."""
    resolved = str(Path(path).resolve())
    cache_path = _stats_cache_path(resolved)
    now = time.time()
    if not force_refresh and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        if (
            isinstance(payload, dict)
            and int(payload.get("cache_version", 0) or 0) == _CACHE_VERSION
            and float(payload.get("generated_at", 0.0) or 0.0)
            + max(0, int(ttl_seconds))
            >= now
        ):
            return payload

    logger.info("Analyzing repository at %s", resolved)
    candidate_files, loc, languages = _scan_repo_stats(resolved)
    payload = _cache_payload(
        path=resolved,
        candidate_files=candidate_files,
        loc=loc,
        languages=languages,
    )
    _ensure_saguaro_dir(resolved)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def record_runtime_profile(
    path: str,
    *,
    profile_kind: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Persist index/query runtime evidence for operator-visible tuning."""
    resolved = str(Path(path).resolve())
    _ensure_saguaro_dir(resolved)
    profile_path = _perf_profile_path(resolved)
    try:
        existing = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        existing = {"profile_version": _PROFILE_VERSION, "profiles": []}
    profiles = list(existing.get("profiles") or [])
    record = {
        "kind": str(profile_kind),
        "timestamp": time.time(),
        "metrics": dict(metrics),
    }
    profiles.append(record)
    existing = {
        "profile_version": _PROFILE_VERSION,
        "selected_runtime_layout": dict(metrics),
        "profiles": profiles[-16:],
    }
    profile_path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    return existing


def load_runtime_profile(path: str) -> dict[str, Any]:
    profile_path = _perf_profile_path(path)
    if not profile_path.exists():
        return {
            "profile_version": _PROFILE_VERSION,
            "profiles": [],
            "selected_runtime_layout": {},
        }
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "profile_version": _PROFILE_VERSION,
            "profiles": [],
            "selected_runtime_layout": {},
        }
    if not isinstance(payload, dict):
        return {
            "profile_version": _PROFILE_VERSION,
            "profiles": [],
            "selected_runtime_layout": {},
        }
    payload.setdefault("profiles", [])
    payload.setdefault("selected_runtime_layout", {})
    return payload


def calibrate_runtime_profile(
    path: str,
    *,
    cpu_threads: int | None = None,
    native_threads: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Persist an operator-visible runtime layout for Saguaro and Anvil."""
    existing = load_runtime_profile(path)
    selected_existing = dict(existing.get("selected_runtime_layout") or {})
    if selected_existing and not force:
        return existing

    stats = get_repo_stats_and_config(path)
    resolved_cpu = max(1, int(cpu_threads or os.cpu_count() or 1))
    resolved_native = max(1, int(native_threads or resolved_cpu))
    query_threads = max(1, min(resolved_native, 4))
    batch_size = 64 if int(stats.get("candidate_files", 0) or 0) >= 512 else 32
    candidate_files = max(1, int(stats.get("candidate_files", 0) or 0))
    batch_count = max(1, int(math.ceil(candidate_files / float(batch_size))))
    index_plan = plan_index_concurrency(
        candidate_files=candidate_files,
        batch_count=batch_count,
        batch_size=batch_size,
        active_dim=int(stats.get("active_dim", 4096) or 4096),
        total_dim=int(stats.get("total_dim", 8192) or 8192),
        vocab_size=int(stats.get("vocab_size", 16384) or 16384),
        threads_per_worker_hint=max(1, min(resolved_native, 2)),
        cpu_threads=resolved_cpu,
        projection_shared=True,
    )

    max_concurrent_saguaro_sessions = max(1, resolved_cpu // query_threads)
    max_parallel_agents = max(
        1,
        min(
            max_concurrent_saguaro_sessions,
            max(1, resolved_cpu // max(1, query_threads)),
        ),
    )
    max_parallel_anvil_instances = max(
        1,
        min(
            max_parallel_agents,
            max(1, resolved_cpu // max(query_threads * 2, 1)),
        ),
    )
    selected_runtime_layout = {
        "contract_version": 1,
        "cpu_threads": int(resolved_cpu),
        "native_threads": int(resolved_native),
        "query_threads": int(query_threads),
        "max_concurrent_saguaro_sessions": int(max_concurrent_saguaro_sessions),
        "max_parallel_agents": int(max_parallel_agents),
        "max_parallel_anvil_instances": int(max_parallel_anvil_instances),
        "index_workers": int(index_plan.get("workers", 1) or 1),
        "index_threads_per_worker": int(index_plan.get("threads_per_worker", 1) or 1),
        "index_batch_capacity": int(index_plan.get("batch_capacity", batch_size) or batch_size),
        "queue_depth_target": int(index_plan.get("queue_depth_target", 1) or 1),
        "rss_budget_mb": float(index_plan.get("rss_budget_mb", 0.0) or 0.0),
        "predicted_rss_mb": float(index_plan.get("predicted_rss_mb", 0.0) or 0.0),
        "calibration_source": "autoscaler",
    }
    return record_runtime_profile(
        path,
        profile_kind="runtime_layout",
        metrics=selected_runtime_layout,
    )


def plan_index_concurrency(
    *,
    candidate_files: int,
    batch_count: int,
    batch_size: int,
    active_dim: int,
    total_dim: int,
    vocab_size: int,
    threads_per_worker_hint: int = 1,
    cpu_threads: int | None = None,
    rss_budget_mb: float | None = None,
    projection_shared: bool = False,
) -> dict[str, int | float | str]:
    """Plan a bounded indexing layout from CPU and RSS budgets."""
    resolved_cpu = max(1, int(cpu_threads or os.cpu_count() or 1))
    resolved_threads = max(1, min(int(threads_per_worker_hint or 1), resolved_cpu))
    resolved_batches = max(0, int(batch_count or 0))
    resolved_batch_size = max(1, int(batch_size or 1))
    projection_mb = _predict_projection_mb(vocab_size=int(vocab_size), active_dim=int(active_dim))
    store_mb = _predict_store_mb(candidate_files=int(candidate_files), active_dim=int(active_dim))
    projection_worker_mb = 0.0 if projection_shared else projection_mb
    worker_rss_mb = round(
        projection_worker_mb
        + max(32.0, float(resolved_batch_size) * float(active_dim) * 0.0035),
        3,
    )
    if rss_budget_mb is None or float(rss_budget_mb) <= 0.0:
        rss_budget_mb = index_rss_budget_mb(
            None,
            cpu_threads=resolved_cpu,
            threads_per_worker=resolved_threads,
            worker_rss_mb=worker_rss_mb,
            projection_mb=projection_mb,
            store_mb=store_mb,
        )
        quota_mode = "default"
    else:
        rss_budget_mb = index_rss_budget_mb(float(rss_budget_mb))
        quota_mode = "budgeted"

    usable_budget = max(worker_rss_mb, rss_budget_mb - _MEMORY_HEADROOM_MB)
    budget_workers = max(
        1,
        int((usable_budget + 1.0e-9) / max(worker_rss_mb, 1.0)),
    )
    cpu_workers = max(1, resolved_cpu // resolved_threads)
    workers = min(cpu_workers, budget_workers, max(1, resolved_batches or 1))
    if candidate_files > 0 and workers == 0:
        workers = 1
    total_threads = workers * resolved_threads if workers else 0
    queue_depth_target = min(
        max(1, resolved_batches or 1),
        max(workers, workers * 2),
    )
    batch_capacity = max(32, resolved_batch_size)
    chars_per_text = max(2048, min(24000, int(total_dim) * 2))
    max_total_text_chars = batch_capacity * chars_per_text
    shared_projection_mb = projection_mb if projection_shared else 0.0
    predicted_rss_mb = round(
        shared_projection_mb + (worker_rss_mb * max(workers, 1)) + store_mb * 0.10,
        3,
    )
    return {
        "workers": int(workers),
        "threads_per_worker": int(resolved_threads),
        "total_threads": int(total_threads),
        "cpu_threads": int(resolved_cpu),
        "queue_depth_target": int(queue_depth_target),
        "rss_budget_mb": round(float(rss_budget_mb), 3),
        "predicted_rss_mb": predicted_rss_mb,
        "worker_rss_mb": worker_rss_mb,
        "projection_mb": round(projection_mb, 3),
        "projection_shared": bool(projection_shared),
        "store_mb": round(store_mb, 3),
        "batch_capacity": int(batch_capacity),
        "max_total_text_chars": int(max_total_text_chars),
        "quota_mode": quota_mode,
        "contract_version": 2,
    }
