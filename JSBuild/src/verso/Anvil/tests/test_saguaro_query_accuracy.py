from __future__ import annotations

import subprocess
from pathlib import Path

from saguaro.api import SaguaroAPI
from saguaro.query.benchmark import (
    derive_query_calibration,
    load_benchmark_cases,
    score_benchmark_results,
)
from saguaro.query.corpus_rules import canonicalize_rel_path, is_excluded_path


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "saguaro_accuracy"
    / "anvil_query_benchmark.json"
)


def _write_accuracy_repo(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "runtime_engine.py").write_text(
        '"""Runtime engine for telemetry-aware thread orchestration."""\n\n'
        "class RuntimeEngine:\n"
        "    def configure_threads(self, logical_threads: int) -> int:\n"
        "        reserve = self._decode_thread_headroom_reserve(logical_threads)\n"
        "        return max(1, logical_threads - reserve)\n\n"
        "    def annotate_telemetry(self, counters: dict[str, int]) -> dict[str, int]:\n"
        "        counters['annotated'] = 1\n"
        "        return counters\n\n"
        "    def _decode_thread_headroom_reserve(self, logical_threads: int) -> int:\n"
        "        return max(1, logical_threads // 4)\n",
        encoding="utf-8",
    )
    (pkg / "benchmark_runner.py").write_text(
        "from pkg.runtime_engine import RuntimeEngine\n\n"
        "def run_native_benchmark() -> int:\n"
        "    engine = RuntimeEngine()\n"
        "    return engine.configure_threads(16)\n",
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("# runtime notes\n", encoding="utf-8")


def _write_eval_fixture(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "tests" / "fixtures" / "saguaro_accuracy"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "anvil_query_benchmark.json").write_text(
        "[\n"
        '  {"category": "symbol_lookup", "query": "RuntimeEngine", "expected_paths": ["pkg/runtime_engine.py"]},\n'
        '  {"category": "benchmark_audit", "query": "native benchmark", "expected_paths": ["pkg/benchmark_runner.py"]}\n'
        "]\n",
        encoding="utf-8",
    )


def _git_init(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


def test_benchmark_fixture_covers_required_categories() -> None:
    cases = load_benchmark_cases(str(FIXTURE_PATH))

    assert len(cases) >= 60
    categories = {case["category"] for case in cases}
    assert {
        "native_qsg_runtime",
        "benchmark_audit",
        "query_engine",
        "symbol_lookup",
        "conceptual_lookup",
        "stale_behavior",
    } <= categories


def test_corpus_rules_exclude_generated_native_wrappers() -> None:
    assert is_excluded_path("core/native/split/kernels/thread_config_wrap.cpp")
    assert is_excluded_path("core/native/model_graph_wrapper.py")


def test_corpus_rules_preserve_authoritative_uppercase_saguaro_paths() -> None:
    assert (
        canonicalize_rel_path("Saguaro/src/ops/core_kernel.cc")
        == "Saguaro/src/ops/core_kernel.cc"
    )


def test_corpus_rules_load_repo_scan_policy(tmp_path: Path) -> None:
    standards = tmp_path / "standards"
    standards.mkdir(parents=True)
    (standards / "scan_exclusion_policy.yaml").write_text(
        "exclude_globs:\n  - generated/**\n",
        encoding="utf-8",
    )

    assert is_excluded_path("generated/out.cc", repo_path=str(tmp_path))
    assert not is_excluded_path("src/live.cc", repo_path=str(tmp_path))


def test_corpus_rules_exclude_restored_temp_tree() -> None:
    assert is_excluded_path("saguaro_restored+temp/indexing/native_worker.py")


def test_benchmark_scoring_counts_top1_and_top3_hits() -> None:
    cases = [
        {
            "category": "symbol_lookup",
            "query": "RuntimeEngine",
            "expected_paths": ["pkg/runtime_engine.py"],
        },
        {
            "category": "benchmark_audit",
            "query": "native benchmark",
            "expected_paths": ["pkg/benchmark_runner.py"],
        },
    ]
    results_by_query = {
        "RuntimeEngine": [{"file": "pkg/runtime_engine.py"}],
        "native benchmark": [
            {"file": "pkg/runtime_engine.py"},
            {"file": "pkg/benchmark_runner.py"},
        ],
    }

    score = score_benchmark_results(cases, results_by_query)

    assert score["top1_hits"] == 1
    assert score["top3_hits"] == 2
    assert score["categories"]["symbol_lookup"]["top1_precision"] == 1.0


def test_query_calibration_derives_bands_from_hits_and_misses() -> None:
    cases = [
        {
            "category": "symbol_lookup",
            "query": "RuntimeEngine",
            "expected_paths": ["pkg/runtime_engine.py"],
        },
        {
            "category": "symbol_lookup",
            "query": "BenchmarkRunner",
            "expected_paths": ["pkg/benchmark_runner.py"],
        },
        {
            "category": "conceptual_lookup",
            "query": "runtime telemetry",
            "expected_paths": ["pkg/runtime_engine.py"],
        },
        {
            "category": "conceptual_lookup",
            "query": "native benchmark",
            "expected_paths": ["pkg/benchmark_runner.py"],
        },
    ]
    results_by_query = {
        "RuntimeEngine": [
            {"file": "pkg/runtime_engine.py", "score": 1.8},
            {"file": "pkg/benchmark_runner.py", "score": 1.1},
        ],
        "BenchmarkRunner": [
            {"file": "pkg/benchmark_runner.py", "score": 1.6},
            {"file": "pkg/runtime_engine.py", "score": 1.0},
        ],
        "runtime telemetry": [
            {"file": "pkg/runtime_engine.py", "score": 1.1},
            {"file": "pkg/benchmark_runner.py", "score": 0.95},
        ],
        "native benchmark": [
            {"file": "pkg/runtime_engine.py", "score": 0.9},
            {"file": "pkg/benchmark_runner.py", "score": 0.82},
        ],
    }

    calibration = derive_query_calibration(cases, results_by_query)

    assert calibration["sample_size"] == 4
    assert calibration["high"]["min_score"] >= calibration["moderate"]["min_score"]
    assert calibration["high"]["min_margin"] >= calibration["moderate"]["min_margin"]


def test_query_pipeline_prefers_runtime_file_for_broad_runtime_query(
    tmp_path: Path,
) -> None:
    _write_accuracy_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)

    result = api.query(
        "runtime engine telemetry thread config",
        k=3,
        strategy="hybrid",
        explain=True,
        auto_refresh=True,
    )

    assert result["results"]
    top = result["results"][0]
    assert top["file"].endswith("pkg/runtime_engine.py")
    assert top["confidence"] in {"high", "moderate", "low", "abstain"}
    assert "score_breakdown" in top["explanation"]
    assert top["explanation"]["matched_features"]


def test_query_pipeline_auto_refreshes_dirty_targets(tmp_path: Path) -> None:
    _write_accuracy_repo(tmp_path)
    _git_init(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)

    runtime_file = tmp_path / "pkg" / "runtime_engine.py"
    runtime_file.write_text(
        runtime_file.read_text(encoding="utf-8") + "\n# dirty change\n",
        encoding="utf-8",
    )

    result = api.query(
        "runtime engine telemetry thread config",
        k=3,
        strategy="hybrid",
        explain=True,
        auto_refresh=True,
    )

    assert result["auto_refreshed"] is True
    assert "pkg/runtime_engine.py" in set(result["auto_refreshed_files"])
    assert "pkg/runtime_engine.py" not in set(result["stale_candidates"])
    assert result["results"]
    assert result["results"][0]["stale"] is False
    assert result["results"][0]["explanation"]["auto_refreshed"] is True


def test_query_pipeline_marks_dirty_targets_as_stale_when_auto_refresh_disabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SAGUARO_QUERY_AUTO_REFRESH", "0")
    _write_accuracy_repo(tmp_path)
    _git_init(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)

    runtime_file = tmp_path / "pkg" / "runtime_engine.py"
    runtime_file.write_text(
        runtime_file.read_text(encoding="utf-8") + "\n# dirty change\n",
        encoding="utf-8",
    )

    result = api.query(
        "runtime engine telemetry thread config",
        k=3,
        strategy="hybrid",
        explain=True,
    )

    assert result["auto_refreshed"] is False
    assert "pkg/runtime_engine.py" in set(result["stale_candidates"])
    assert result["results"]
    assert result["results"][0]["explanation"]["stale_candidates"]


def test_retrieval_quality_eval_respects_fixture_limit_and_persists_calibration(
    tmp_path: Path,
) -> None:
    _write_accuracy_repo(tmp_path)
    _write_eval_fixture(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)

    result = api.eval_run("retrieval_quality", k=3, limit=1)

    assert result["suite"] == "retrieval_quality"
    assert result["total_available_cases"] == 2
    assert result["total_cases"] == 1
    assert result["case_limit"] == 1
    calibration_path = tmp_path / ".saguaro" / "query_calibration.json"
    assert calibration_path.exists()
