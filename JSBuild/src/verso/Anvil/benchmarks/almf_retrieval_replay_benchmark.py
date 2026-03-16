"""Run ALMF retrieval and replay benchmarks and emit a JSON summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

if __name__ == "__main__" and __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

from core.memory.fabric import (
    MemoryBenchmarkRunner,
    MemoryFabricStore,
    MemoryProjector,
    MemoryRetrievalPlanner,
)
from saguaro.storage.atomic_fs import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, help="ALMF sqlite database path")
    parser.add_argument("--campaign-id", required=True, help="Campaign to benchmark")
    parser.add_argument("--cases", required=True, help="JSON file containing benchmark cases")
    parser.add_argument(
        "--storage-root",
        default=None,
        help="Optional ALMF storage root override",
    )
    parser.add_argument(
        "--out-root",
        default="audit/runs",
        help="Directory to write benchmark artifacts",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id; defaults to an ALMF timestamped value",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = str(args.run_id or f"almf_{int(time.time())}")
    out_dir = Path(str(args.out_root)) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    store = MemoryFabricStore.from_db_path(
        str(args.db_path),
        storage_root=args.storage_root,
    )
    planner = MemoryRetrievalPlanner(store, MemoryProjector())
    runner = MemoryBenchmarkRunner(store, planner)
    cases = json.loads(Path(str(args.cases)).read_text(encoding="utf-8"))
    summary = runner.run_suite(campaign_id=str(args.campaign_id), cases=list(cases))
    summary["run_id"] = run_id
    summary["artifact_dir"] = str(out_dir)
    atomic_write_json(str(out_dir / "summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
