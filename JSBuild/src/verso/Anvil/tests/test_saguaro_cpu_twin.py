from __future__ import annotations

from saguaro.cpu import CPUScanner


def test_cpu_twin_reports_cache_roofline_and_schedule_candidates(tmp_path) -> None:
    target = tmp_path / "core" / "simd" / "common" / "perf_utils.h"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i + 1];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    payload = CPUScanner(str(tmp_path)).scan(
        path="core/simd/common/perf_utils.h",
        arch="x86_64-avx2",
    )

    assert payload["status"] == "ok"
    assert payload["hotspot_count"] >= 1
    hotspot = payload["hotspots"][0]
    assert hotspot["analysis_engine"] in {"native_cpp", "python_fallback"}
    assert hotspot["cache"]["estimated_cache_lines_touched"] >= 1
    assert hotspot["cache"]["l3_risk"] in {"low", "medium", "high"}
    assert hotspot["roofline"]["bound"] in {"memory_bound", "balanced", "compute_bound"}
    assert hotspot["register_pressure"]["band"] in {"low", "medium", "high"}
    assert hotspot["schedule_twin"]["candidates"]
    assert hotspot["schedule_twin"]["selected"]["recipe"]["transforms"]
    assert hotspot["optimization_packet"]["schedule_recipe"]["name"] == hotspot["schedule_twin"]["selected"]["name"]
    assert (
        hotspot["proof_packet"]["runtime_witness"]["schedule_recipe"]["name"]
        == hotspot["schedule_twin"]["selected"]["name"]
    )


def test_cpu_twin_marks_reduction_candidates_for_tree_reduce(tmp_path) -> None:
    target = tmp_path / "core" / "simd" / "common" / "reduce.h"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "inline float reduce_sum(float* input, int n) {\n"
        "  float acc = 0.0f;\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    acc += input[i];\n"
        "  }\n"
        "  return acc;\n"
        "}\n",
        encoding="utf-8",
    )

    payload = CPUScanner(str(tmp_path)).scan(
        path="core/simd/common/reduce.h",
        arch="x86_64-avx2",
    )

    hotspot = payload["hotspots"][0]
    assert hotspot["loop_context"]["reduction"] is True
    assert any(
        candidate["name"] == "tree_reduce"
        for candidate in hotspot["schedule_twin"]["candidates"]
    )
    selected = hotspot["schedule_twin"]["selected"]
    assert selected["recipe"]["recipe_id"].startswith("schedule::")
    assert any(
        hint["kind"] == "tree_reduce"
        for hint in hotspot["optimization_packet"]["reduction_hints"]
    )
    assert hotspot["proof_packet"]["runtime_witness"]["reduction_hints"]
