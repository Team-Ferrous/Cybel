"""Benchmark audit harness for Saguaro indexing control-plane and ingest paths."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from saguaro.api import SaguaroAPI
from saguaro.indexing.auto_scaler import get_repo_stats_and_config

TRACE_SCHEMA_VERSION = 2
CORPUS_PRESETS = {
    "small": {"sample_files": 64, "functions_per_file": 4},
    "medium": {"sample_files": 256, "functions_per_file": 8},
    "large": {"sample_files": 1024, "functions_per_file": 12},
}


def _write_sample_repo(root: Path, file_count: int, functions_per_file: int) -> None:
    pkg = root / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    for index in range(file_count):
        lines = [f"def fn_{index}_{inner}():\n    return {index + inner}\n" for inner in range(functions_per_file)]
        (pkg / f"module_{index:04d}.py").write_text("".join(lines), encoding="utf-8")


def _time_call(fn, *args, **kwargs) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


def _run_audit(
    repo_path: Path,
    *,
    force_refresh: bool,
    warm_cache: bool,
    corpus_label: str = "custom",
) -> dict[str, Any]:
    repo_path = repo_path.resolve()
    api = SaguaroAPI(str(repo_path))

    stats1, stats_elapsed = _time_call(
        get_repo_stats_and_config,
        str(repo_path),
        force_refresh=force_refresh,
    )
    stats2, stats_cached_elapsed = _time_call(
        get_repo_stats_and_config,
        str(repo_path),
        force_refresh=False,
    )
    if warm_cache:
        _, stats_cached_elapsed = _time_call(
            get_repo_stats_and_config,
            str(repo_path),
            force_refresh=False,
        )

    index_result, index_elapsed = _time_call(
        api.index,
        path=".",
        force=False,
        incremental=True,
        prune_deleted=True,
    )

    execution = index_result.get("execution", {}) or {}
    execution_compact = dict(execution)
    memory = dict(execution_compact.get("memory") or {})
    ast_ledger = dict(memory.get("ast_ledger") or {})
    per_file = ast_ledger.get("per_file")
    if isinstance(per_file, dict):
        ast_ledger["per_file_count"] = len(per_file)
        ast_ledger.pop("per_file", None)
    if ast_ledger:
        memory["ast_ledger"] = ast_ledger
    if memory:
        execution_compact["memory"] = memory
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "corpus": corpus_label,
        "repo_path": str(repo_path),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count() or 1,
        },
        "env": {
            "SAGUARO_INDEX_WORKERS": os.getenv("SAGUARO_INDEX_WORKERS", ""),
            "SAGUARO_INDEX_MAX_WORKERS": os.getenv("SAGUARO_INDEX_MAX_WORKERS", ""),
            "SAGUARO_NATIVE_THREADS_PER_WORKER": os.getenv(
                "SAGUARO_NATIVE_THREADS_PER_WORKER", ""
            ),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", ""),
        },
        "timings": {
            "repo_stats_seconds": round(stats_elapsed, 6),
            "repo_stats_cached_seconds": round(stats_cached_elapsed, 6),
            "index_seconds": round(index_elapsed, 6),
        },
        "stats": stats1,
        "cached_stats_match": stats1 == stats2,
        "index": {
            "status": index_result.get("status"),
            "indexed_files": index_result.get("indexed_files", 0),
            "indexed_entities": index_result.get("indexed_entities", 0),
            "execution": execution_compact,
            "rss_budget_mb": execution_compact.get("rss_budget_mb", 0.0),
            "predicted_rss_mb": execution_compact.get("predicted_rss_mb", 0.0),
            "observed_rss_mb": execution_compact.get("observed_rss_mb", 0.0),
            "worker_emitted_bytes": execution_compact.get("worker_emitted_bytes", 0),
            "queue_wait_seconds": execution_compact.get("queue_wait_seconds", 0.0),
            "projection_bytes": execution_compact.get("projection_bytes", 0),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=None, help="Repo to benchmark.")
    parser.add_argument("--sample-files", type=int, default=200)
    parser.add_argument("--functions-per-file", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--warm-cache", action="store_true")
    parser.add_argument(
        "--corpus",
        choices=sorted(CORPUS_PRESETS.keys()),
        default=None,
        help="Use a named corpus preset (small/medium/large).",
    )
    args = parser.parse_args()

    temp_root: Path | None = None
    repo_path = args.repo
    if repo_path is None:
        temp_root = Path(tempfile.mkdtemp(prefix="saguaro-index-audit-"))
        repo_path = temp_root
        sample_files = int(args.sample_files)
        functions_per_file = int(args.functions_per_file)
        corpus_label = "custom"
        if args.corpus:
            preset = CORPUS_PRESETS[str(args.corpus)]
            sample_files = int(preset["sample_files"])
            functions_per_file = int(preset["functions_per_file"])
            corpus_label = str(args.corpus)
        _write_sample_repo(repo_path, sample_files, functions_per_file)
    else:
        corpus_label = str(args.corpus or "external")

    try:
        payload = _run_audit(
            repo_path,
            force_refresh=args.force_refresh,
            warm_cache=args.warm_cache,
            corpus_label=corpus_label,
        )
        rendered = json.dumps(payload, indent=2, sort_keys=True)
        print(rendered)
        if args.output is not None:
            args.output.write_text(rendered + "\n", encoding="utf-8")
    finally:
        if temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
