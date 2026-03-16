from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_benchmark_suite_emits_modernization_results(tmp_path: Path) -> None:
    target = tmp_path / "target"
    candidate = tmp_path / "repo_analysis" / "candidate"
    _write(target / "auth.py", "def authenticate(name: str) -> str:\n    return name\n")
    _write(candidate / "token_auth.py", "def authenticate(name: str) -> str:\n    return name.strip()\n")
    suite_path = tmp_path / "comparative_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suites": [
                    {
                        "name": "smoke",
                        "target": str(target),
                        "candidates": [str(candidate)],
                        "top_k": 4,
                        "iterations": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "benchmarks/run_comparative_suite.py", str(suite_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["results"][0]["modernization"]["schema_version"] == "comparative_report.v3"
