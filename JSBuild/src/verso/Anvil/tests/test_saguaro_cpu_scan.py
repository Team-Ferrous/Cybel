from __future__ import annotations

import json

from saguaro.cpu import CPUScanner


def test_cpu_scan_reports_hotspots_and_schedule_candidates(tmp_path) -> None:
    target = tmp_path / "core" / "simd" / "common" / "perf_utils.h"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i];\n"
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
    assert hotspot["vectorization"]["legal"] is True
    assert hotspot["analysis_engine"] in {"native_cpp", "python_fallback"}
    assert hotspot["cache"]["estimated_cache_lines_touched"] >= 1
    assert hotspot["schedule_twin"]["candidates"]
    assert hotspot["schedule_twin"]["selected"]["recipe"]["transforms"]
    assert hotspot["optimization_packet"]["schedule_recipe"]["name"] == hotspot["schedule_twin"]["selected"]["name"]
    assert hotspot["proof_packet"]["capsule_id"].startswith("hotspot::")
    assert hotspot["proof_packet"]["completeness"]["complete"] is True
    assert payload["capsule_manifest"]["capsule_count"] >= 1
    assert payload["summary"]["proof_packets_complete"] >= 1
    assert hotspot["benchmark_priority"] > 0


def test_cpu_scan_skips_generated_build_outputs(tmp_path) -> None:
    kernel = tmp_path / "core" / "native" / "kernel.cpp"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    build_log = (
        tmp_path
        / "core"
        / "native"
        / "build-cmake-20260311"
        / "CMakeFiles"
        / "CMakeConfigureLog.yaml"
    )
    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.write_text(
        "The system is: Linux - 6.17.0-14-generic - x86_64\n"
        "Compiler: /usr/bin/c++\n",
        encoding="utf-8",
    )
    wrapper = tmp_path / "core" / "native" / "wrapper.py"
    wrapper.write_text(
        "STATE = {\n"
        "    'lanes': 8,\n"
        "    'width': 32,\n"
        "}\n",
        encoding="utf-8",
    )

    payload = CPUScanner(str(tmp_path)).scan(path="core/native", arch="x86_64-avx2")

    assert payload["files_scanned"] == 2
    assert payload["hotspot_count"] >= 1
    assert all("CMakeConfigureLog.yaml" not in item["file"] for item in payload["hotspots"])
    assert all(item["file"] != "core/native/wrapper.py" for item in payload["hotspots"])
    assert payload["hotspots"][0]["file"] == "core/native/kernel.cpp"


def test_cpu_scan_persists_hotspot_capsules_and_traceability(tmp_path) -> None:
    target = tmp_path / "core" / "native" / "kernel.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    payload = CPUScanner(str(tmp_path)).scan(path="core/native", arch="x86_64-avx2")

    manifest_path = tmp_path / ".anvil" / "validation" / "hotspot_capsule_manifest.json"
    traceability_path = tmp_path / "standards" / "traceability" / "TRACEABILITY.jsonl"

    assert manifest_path.exists()
    assert traceability_path.exists()
    assert payload["capsule_manifest"]["complete_capsule_count"] >= 1
    assert payload["hotspots"][0]["proof_packet_path"].startswith(
        ".anvil/validation/hotspot_capsules/"
    )
    trace_records = [
        json.loads(line)
        for line in traceability_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert trace_records
    assert all(record.get("design_ref") == "standards/AES.md" for record in trace_records)
