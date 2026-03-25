from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from saguaro.services.comparative import ComparativeAnalysisService


def main() -> int:
    if len(sys.argv) > 1:
        suite_path = Path(sys.argv[1])
    else:
        suite_path = Path(__file__).with_name("comparative_suite.json")
    with suite_path.open(encoding="utf-8") as handle:
        suite = json.load(handle)

    service = ComparativeAnalysisService(os.getcwd())
    results = []
    for entry in list(suite.get("suites") or []):
        benchmark = service.benchmark_session(
            path=str(entry.get("fleet_root") or entry.get("target") or "."),
            ttl_hours=float(entry.get("ttl_hours", 24.0) or 24.0),
            batch_sizes=list(entry.get("batch_sizes") or [128]),
            iterations=int(entry.get("iterations", 1) or 1),
        )
        results.append(
            {
                "name": entry.get("name"),
                "benchmark": benchmark,
                "modernization": service.compare(
                    target=str(entry.get("target") or "."),
                    candidates=list(entry.get("candidates") or []),
                    fleet_root=str(entry.get("fleet_root") or "") or None,
                    top_k=int(entry.get("top_k", 8) or 8),
                    portfolio_top_n=int(entry.get("portfolio_top_n", 12) or 12),
                    export_datatables=True,
                )
                if entry.get("candidates") or entry.get("fleet_root")
                else {},
            }
        )
    print(json.dumps({"status": "ok", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
