from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import saguaro.api as api_module
from saguaro.agents.perception import TracePerception
from saguaro.analysis.complexity_analyzer import ComplexityAnalyzer
from saguaro.api import SaguaroAPI
from saguaro.indexing.backends import NumPyBackend


def test_trace_perception_complexity_detects_nested_loop_and_recursion(
    tmp_path: Path,
) -> None:
    (tmp_path / "algo.py").write_text(
        "def quadratic(xs):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        for y in xs:\n"
        "            total += x * y\n"
        "    return total\n\n"
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n",
        encoding="utf-8",
    )

    perception = TracePerception(repo_path=str(tmp_path))
    quad = perception.complexity_report("quadratic", file_path="algo.py")
    fib = perception.complexity_report("fib", file_path="algo.py")

    assert quad["status"] == "ok"
    assert quad["time_complexity"] in {"O(n^2)", "O(n^3)", "O(n^k)"}
    assert fib["status"] == "ok"
    assert fib["time_complexity"] == "O(2^n)"


def test_api_pipeline_complexity_surface(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    (tmp_path / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=False)
    report = api.complexity(pipeline="main.py", depth=2)

    assert report["status"] in {"ok", "no_match"}
    assert "total_complexity" in report


def test_api_pipeline_complexity_query_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    (tmp_path / "pipeline.py").write_text(
        "def forward(tokens):\n"
        "    for token in tokens:\n"
        "        _ = token\n"
        "    return tokens\n\n"
        "def decode(tokens):\n"
        "    return forward(tokens)\n",
        encoding="utf-8",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=False)
    report = api.complexity(pipeline="inference pipeline", depth=4)

    assert report["status"] == "ok"
    assert int(report.get("stage_count", 0)) >= 1
    total = report.get("total_complexity", {})
    assert "time_complexity" in total


def test_complexity_report_returns_unknown_for_missing_symbol(tmp_path: Path) -> None:
    (tmp_path / "algo.py").write_text(
        "def existing(xs):\n"
        "    return len(xs)\n",
        encoding="utf-8",
    )

    perception = TracePerception(repo_path=str(tmp_path))
    report = perception.complexity_report("missing", file_path="algo.py")

    assert report["status"] == "ok"
    assert report["time_complexity"] == "unknown"
    assert report["space_complexity"] == "unknown"
    assert report["confidence"] <= 0.25
    assert any("symbol_not_found" in item for item in report.get("evidence", []))


def test_complexity_analyzer_pipeline_accepts_dict_stage_payloads() -> None:
    analyzer = ComplexityAnalyzer(repo_path=".")
    trace = SimpleNamespace(
        stages=[
            {
                "name": "decode",
                "file": "core/native/model_graph_wrapper.py",
                "complexity": {
                    "time_complexity": "O(n)",
                    "space_complexity": "O(n)",
                    "amortized_time_complexity": "O(n)",
                    "worst_case_time_complexity": "O(n^2)",
                    "confidence": 0.7,
                    "evidence": ["loop detected"],
                },
            }
        ]
    )
    report = analyzer.analyze_pipeline(trace)
    assert report["time_complexity"] == "O(n)"
    assert report["worst_case_time_complexity"] == "O(n^2)"


def test_cli_help_exposes_trace_and_complexity_commands() -> None:
    trace_help = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "saguaro.cli", "trace", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert trace_help.returncode == 0
    assert "--max-stages" in trace_help.stdout

    complexity_help = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "saguaro.cli", "complexity", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert complexity_help.returncode == 0
    assert "--pipeline" in complexity_help.stdout
