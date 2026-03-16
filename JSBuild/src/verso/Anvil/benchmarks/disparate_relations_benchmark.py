from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

from saguaro.api import SaguaroAPI


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_repo(root: Path) -> None:
    _write(
        root / "core" / "qsg" / "continuous_engine.py",
        "class QSGInferenceEngine:\n"
        "    def evaluation_pipeline(self) -> int:\n"
        "        return 1\n",
    )
    _write(
        root / "Saguaro" / "saguaro" / "query" / "pipeline.py",
        "class QueryPipeline:\n"
        "    def evaluation_pipeline(self) -> int:\n"
        "        return 2\n",
    )
    _write(
        root / "core" / "runtime" / "registry.py",
        "class TargetRegistry:\n"
        "    def register_target(self, name: str) -> str:\n"
        "        return name.lower()\n",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="saguaro-disparate-bench-") as tmp:
        repo = Path(tmp)
        _seed_repo(repo)
        api = SaguaroAPI(repo_path=str(repo))
        started = time.perf_counter()
        build = api.graph_build(path=".", incremental=False)
        build_ms = round((time.perf_counter() - started) * 1000.0, 3)
        started = time.perf_counter()
        payload = api.disparate_relations(limit=20)
        query_ms = round((time.perf_counter() - started) * 1000.0, 3)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "build": build,
                    "disparate_relation_count": payload.get("total_count", payload.get("count", 0)),
                    "families": payload.get("families", []),
                    "build_ms": build_ms,
                    "query_ms": query_ms,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
