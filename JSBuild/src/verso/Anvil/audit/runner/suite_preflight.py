from __future__ import annotations

import ctypes.util
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit.provenance.capture import (
    capture_model_provenance,
    capture_runtime_provenance,
    host_fingerprint,
)
from audit.runtime_logging import get_active_logger
from audit.runtime_logging import run_logged_subprocess
from audit.runner.suite_certification import (
    benchmark_harness_hash,
    contract_readiness_state,
    ensure_host_contract,
    load_host_contract,
    load_tuning_contract,
    resolve_host_contract_id,
    sha256_file,
    summarize_tuning_states,
    validate_host_contract,
    validate_tuning_contract,
)
from audit.runner.suite_profiles import BenchmarkSuiteSpec
from core.native.ffi_boundary_manifest import resolve_ffi_boundary
from core.native.runtime_telemetry import build_runtime_capability_ledger


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    telemetry_dir: Path | None = None,
    label: str | None = None,
    timeout: int | float | None = None,
) -> tuple[int, str, str]:
    stdout_path = None
    stderr_path = None
    if telemetry_dir is not None and label:
        stdout_path = telemetry_dir / f"{label}.stdout.log"
        stderr_path = telemetry_dir / f"{label}.stderr.log"
    completed = run_logged_subprocess(
        cmd=cmd,
        cwd=cwd or Path.cwd(),
        env=os.environ.copy(),
        source="suite_preflight",
        phase="preflight",
        timeout=timeout,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _read_text(path: str) -> str:
    candidate = Path(path)
    if not candidate.exists():
        return ""
    return candidate.read_text(encoding="utf-8", errors="ignore").strip()


def _selected_mode(raw: str) -> str:
    for chunk in raw.split():
        if chunk.startswith("[") and chunk.endswith("]"):
            return chunk.strip("[]")
    return raw.splitlines()[0].strip() if raw.strip() else ""


def _cpu_governor() -> str:
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if base.exists():
        return base.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def _meminfo_map(raw: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parts = value.strip().split()
        if not parts:
            continue
        try:
            parsed[key.strip()] = int(parts[0])
        except ValueError:
            continue
    return parsed


def _perf_probe(tmp_dir: Path) -> dict[str, Any]:
    perf_path = shutil.which("perf")
    if not perf_path:
        return {
            "available": False,
            "reason": "perf_missing",
            "perf_path": "",
            "stat_ok": False,
            "record_ok": False,
        }
    perf_data = tmp_dir / "preflight_perf.data"
    stat_code, stat_out, stat_err = _run(
        [perf_path, "stat", "-o", "/dev/null", "true"],
        telemetry_dir=tmp_dir,
        label="perf_stat_probe",
    )
    record_code, record_out, record_err = _run(
        [perf_path, "record", "-o", str(perf_data), "--", "true"],
        telemetry_dir=tmp_dir,
        label="perf_record_probe",
    )
    return {
        "available": stat_code == 0 and record_code == 0,
        "reason": ";".join(
            part
            for part in (
                "" if stat_code == 0 else f"perf_stat_failed:{stat_err or stat_out}",
                (
                    ""
                    if record_code == 0
                    else f"perf_record_failed:{record_err or record_out}"
                ),
            )
            if part
        )
        or "ok",
        "perf_path": perf_path,
        "stat_ok": stat_code == 0,
        "record_ok": record_code == 0,
    }


def _preflight_strictness(spec: BenchmarkSuiteSpec) -> str:
    raw = str(getattr(spec, "preflight_strictness", "") or "").strip().lower()
    if raw in {"audit", "optimize", "certify"}:
        return raw
    return "certify"


def _saguaro_timeout_seconds(strictness: str) -> float:
    explicit = str(os.getenv("ANVIL_SAGUARO_HEALTH_TIMEOUT_SECONDS", "") or "").strip()
    if explicit:
        return float(explicit)
    if strictness == "optimize":
        return 90.0
    if strictness == "audit":
        return 45.0
    return 30.0


def _cpu_scan_timeout_seconds(strictness: str, timeout_seconds: float) -> float:
    explicit = str(
        os.getenv("ANVIL_SAGUARO_CPU_SCAN_TIMEOUT_SECONDS", "") or ""
    ).strip()
    if explicit:
        return float(explicit)
    if strictness == "optimize":
        return max(timeout_seconds, 90.0)
    return timeout_seconds


def _saguaro_cache_path(repo_root: Path) -> Path:
    return repo_root / ".anvil" / "cache" / "saguaro_health.json"


def _load_cached_saguaro_health(repo_root: Path) -> dict[str, Any] | None:
    path = _saguaro_cache_path(repo_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_cached_saguaro_health(repo_root: Path, payload: dict[str, Any]) -> None:
    path = _saguaro_cache_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _saguaro_probe(
    repo_root: Path,
    telemetry_dir: Path,
    *,
    strictness: str,
) -> dict[str, Any]:
    timeout_seconds = _saguaro_timeout_seconds(strictness)
    health_path = telemetry_dir / "saguaro_health.json"
    cpu_scan_timeout = _cpu_scan_timeout_seconds(strictness, timeout_seconds)
    cpu_scan_target = str(
        os.getenv("ANVIL_SAGUARO_CPU_SCAN_PATH", "core/simd/common/perf_utils.h")
        or "core/simd/common/perf_utils.h"
    )
    cpu_target_path = repo_root / cpu_scan_target
    if not cpu_target_path.exists():
        fallback = repo_root / "core" / "native" / "runtime_telemetry.py"
        cpu_scan_target = (
            "core/native/runtime_telemetry.py" if fallback.exists() else "."
        )
    machine = platform.machine().lower()
    arch = "arm64-neon" if machine in {"arm64", "aarch64"} else "x86_64-avx2"
    started = time.perf_counter()
    code, stdout, stderr = _run(
        ["saguaro", "health"],
        cwd=repo_root,
        telemetry_dir=telemetry_dir,
        label="saguaro_health",
        timeout=timeout_seconds,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if stdout:
        health_path.write_text(stdout + "\n", encoding="utf-8")
        _write_cached_saguaro_health(
            repo_root,
            {
                "stdout": stdout,
                "return_code": code,
                "captured_at_ms": elapsed_ms,
            },
        )
    cached = None
    cache_hit = False
    if code != 0 and strictness in {"audit", "optimize"}:
        cached = _load_cached_saguaro_health(repo_root)
        cache_hit = bool(cached and str(cached.get("stdout") or "").strip())
        if cache_hit and not stdout:
            stdout = str(cached.get("stdout") or "")
            health_path.write_text(stdout + "\n", encoding="utf-8")
    cpu_code, cpu_stdout, cpu_stderr = _run(
        [
            "saguaro",
            "cpu",
            "scan",
            "--path",
            cpu_scan_target,
            "--arch",
            arch,
            "--format",
            "json",
            "--limit",
            "8",
        ],
        cwd=repo_root,
        telemetry_dir=telemetry_dir,
        label="saguaro_cpu_scan",
        timeout=cpu_scan_timeout,
    )
    cpu_scan_path = telemetry_dir / "saguaro_cpu_scan.json"
    if cpu_stdout:
        cpu_scan_path.write_text(cpu_stdout + "\n", encoding="utf-8")
    return {
        "ok": code == 0 and cpu_code == 0,
        "return_code": code,
        "timeout_seconds": timeout_seconds,
        "elapsed_ms": elapsed_ms,
        "budget_mode": strictness,
        "stdout": stdout,
        "stderr": stderr,
        "cache_hit": cache_hit,
        "artifact": {"path": health_path.as_posix()},
        "health": {
            "ok": code == 0 or cache_hit,
            "return_code": code,
            "stdout": stdout,
            "stderr": stderr,
            "cache_hit": cache_hit,
            "elapsed_ms": elapsed_ms,
        },
        "cpu_scan_ok": cpu_code == 0,
        "cpu_scan_return_code": cpu_code,
        "cpu_scan_timeout_seconds": cpu_scan_timeout,
        "cpu_scan_stdout": cpu_stdout,
        "cpu_scan_stderr": cpu_stderr,
        "cpu_scan_target": cpu_scan_target,
        "cpu_scan_arch": arch,
        "cpu_scan_artifact": {"path": cpu_scan_path.as_posix()},
        "cpu_scan": {
            "ok": cpu_code == 0,
            "return_code": cpu_code,
            "target": cpu_scan_target,
            "arch": arch,
            "stdout": cpu_stdout,
            "stderr": cpu_stderr,
        },
    }


def _dependency_report(repo_root: Path) -> dict[str, Any]:
    pkg_config = shutil.which("pkg-config") or ""
    gcc = shutil.which("gcc") or ""
    binaries = {
        name: {"present": bool(path), "path": path}
        for name, path in (
            ("perf", shutil.which("perf") or ""),
            ("cpupower", shutil.which("cpupower") or ""),
            ("numactl", shutil.which("numactl") or ""),
            ("taskset", shutil.which("taskset") or ""),
            ("pkg-config", pkg_config),
            ("gcc", gcc),
        )
    }
    hwloc_pkg_ok = False
    hwloc_pkg_version = ""
    if pkg_config:
        try:
            completed = subprocess.run(
                [pkg_config, "--modversion", "hwloc"],
                cwd=str(repo_root),
                text=True,
                capture_output=True,
                check=False,
            )
            hwloc_pkg_ok = completed.returncode == 0
            hwloc_pkg_version = (completed.stdout or "").strip()
        except Exception:
            hwloc_pkg_ok = False
    return {
        "schema_version": "native_qsg_suite.dependency_report.v1",
        "binaries": binaries,
        "libraries": {
            "libnuma": bool(ctypes.util.find_library("numa")),
            "libhwloc": bool(ctypes.util.find_library("hwloc")),
        },
        "dev_metadata": {
            "hwloc_pkg_config": {
                "present": hwloc_pkg_ok,
                "version": hwloc_pkg_version,
            }
        },
    }


@dataclass(frozen=True)
class PreflightResult:
    payload: dict[str, Any]
    ok: bool


def _threading_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    threading = dict(payload.get("threading") or {})
    env = dict(payload.get("env") or {})
    host = dict(payload.get("host") or {})
    return {
        "visible_threads": int(
            threading.get("visible_threads", host.get("visible_threads", 0)) or 0
        ),
        "logical_cpus": int(host.get("logical_cpus", 0) or 0),
        "omp_num_threads": str(
            threading.get("omp_num_threads") or env.get("OMP_NUM_THREADS") or ""
        ),
        "omp_proc_bind": str(
            threading.get("omp_proc_bind") or env.get("OMP_PROC_BIND") or ""
        ),
        "omp_places": str(threading.get("omp_places") or env.get("OMP_PLACES") or ""),
    }


def _kv_cache_quantization_mode(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        env = dict(payload.get("env") or {})
        raw = str(env.get("ANVIL_KV_QUANT") or "").strip().lower()
        if not raw:
            continue
        if raw in {"0", "false", "off", "none", "fp32", "f32"}:
            return "fp32"
        if raw in {"1", "true", "on", "q8", "int8"}:
            return "q8"
        return raw
    return "fp32"


def _build_benchmark_metadata(
    *,
    spec: BenchmarkSuiteSpec,
    launch_runtime: dict[str, Any],
    runtime: dict[str, Any],
    models: dict[str, Any],
    governor: str,
    thp_mode: str,
    perf_paranoid: str,
    launch_affinity: list[int],
    post_adjustment_affinity: list[int],
) -> dict[str, Any]:
    runtime_host = dict(runtime.get("host") or {})
    launch_host = dict(launch_runtime.get("host") or {})
    runtime_memory = dict(runtime.get("memory") or {})
    runtime_numa = dict(runtime.get("numa") or {})
    kv_cache_quantization = _kv_cache_quantization_mode(runtime, launch_runtime)
    return {
        "git_sha": str(((runtime.get("git") or {}).get("commit")) or ""),
        "profile_name": spec.profile_name,
        "models": [
            {
                "model": model_name,
                "quantization_profile": str(
                    (dict(model_payload).get("quant_variant")) or ""
                ),
                "kv_cache_quantization": kv_cache_quantization,
                "digest": str((dict(model_payload).get("digest")) or ""),
                "strict_native_supported": bool(
                    dict(model_payload).get("strict_native_supported", False)
                ),
            }
            for model_name, model_payload in models.items()
        ],
        "launch": {
            "threading": _threading_metadata(launch_runtime),
            "affinity_cpus": launch_affinity,
        },
        "runtime": {
            "threading": _threading_metadata(runtime),
            "affinity_cpus": post_adjustment_affinity,
        },
        "host": {
            "host_fingerprint": str(runtime_host.get("host_fingerprint") or ""),
            "cpu_model": str(runtime_host.get("cpu_model") or ""),
            "machine": str(runtime_host.get("machine") or ""),
            "platform": str(runtime_host.get("platform") or ""),
            "kernel_release": str(runtime_host.get("kernel_release") or ""),
            "microcode_version": str(runtime_host.get("microcode_version") or ""),
            "cpu_governor": governor or str(runtime_host.get("cpu_governor") or ""),
            "huge_page_mode": thp_mode
            or str(runtime_host.get("transparent_hugepage_mode") or ""),
            "numa_policy": str(runtime_numa.get("policy") or ""),
            "numa_membind": str(runtime_numa.get("membind") or ""),
            "numa_cpubind": str(runtime_numa.get("cpubind") or ""),
            "numa_physcpubind": str(runtime_numa.get("physcpubind") or ""),
            "memory_speed_mt_s": runtime_memory.get("memory_speed_mt_s"),
            "memory_speed_source": str(runtime_memory.get("memory_speed_source") or ""),
            "launch_visible_threads": int(launch_host.get("visible_threads", 0) or 0),
            "runtime_visible_threads": int(runtime_host.get("visible_threads", 0) or 0),
        },
        "perf_event_paranoid": perf_paranoid,
    }


def _graph_preflight(
    *,
    repo_root: Path,
    telemetry_dir: Path,
    spec: BenchmarkSuiteSpec,
) -> dict[str, Any]:
    touched_native_paths = [
        path
        for path in (
            "core/native/native_qsg_engine.py",
            "core/native/runtime_telemetry.py",
            "audit/runner/suite_preflight.py",
            "audit/runner/benchmark_suite.py",
            "saguaro/services/platform.py",
        )
        if (repo_root / path).exists()
    ]
    if not touched_native_paths:
        payload = {
            "schema_version": "native_qsg_suite.graph_preflight.v1",
            "status": "skipped",
            "passed": True,
            "profile_name": spec.profile_name,
            "touched_files": [],
            "ffi_boundary_count": 0,
            "unresolved_boundaries": [],
            "resolved_by_manifest": [],
        }
    else:
        try:
            from saguaro.api import SaguaroAPI

            report = SaguaroAPI(repo_path=str(repo_root)).ffi_audit(path=".", limit=400)
            boundaries = [
                dict(item)
                for item in list(report.get("boundaries") or [])
                if str(item.get("file") or "").startswith(
                    ("core/native/", "audit/runner/", "saguaro/services/")
                )
            ]
            resolved_by_manifest: list[dict[str, Any]] = []
            unresolved = [
                {
                    "file": str(item.get("file") or ""),
                    "mechanism": str(item.get("mechanism") or ""),
                    "target": str(item.get("target") or ""),
                    "shared_object": str(item.get("shared_object") or ""),
                }
                for item in boundaries
                if not str(item.get("target") or "").strip()
                and not str(item.get("shared_object") or "").strip()
            ]
            filtered_unresolved: list[dict[str, Any]] = []
            for item in unresolved:
                manifest_entry = resolve_ffi_boundary(str(item.get("file") or ""))
                if manifest_entry:
                    resolved_by_manifest.append(
                        {
                            **item,
                            "shared_object": str(
                                manifest_entry.get("shared_object") or ""
                            ),
                            "symbols": list(manifest_entry.get("symbols") or []),
                        }
                    )
                    continue
                filtered_unresolved.append(item)
            payload = {
                "schema_version": "native_qsg_suite.graph_preflight.v1",
                "status": "covered",
                "passed": not filtered_unresolved,
                "profile_name": spec.profile_name,
                "touched_files": touched_native_paths,
                "ffi_boundary_count": len(boundaries),
                "unresolved_boundaries": filtered_unresolved,
                "resolved_by_manifest": resolved_by_manifest,
            }
        except Exception as exc:
            payload = {
                "schema_version": "native_qsg_suite.graph_preflight.v1",
                "status": "error",
                "passed": False,
                "profile_name": spec.profile_name,
                "touched_files": touched_native_paths,
                "ffi_boundary_count": 0,
                "unresolved_boundaries": [],
                "resolved_by_manifest": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
    artifact_path = telemetry_dir / "graph_preflight.json"
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    payload["artifact"] = str(artifact_path)
    return payload


def _admission_manifest(
    *,
    spec: BenchmarkSuiteSpec,
    failures: list[str],
    warnings: list[str],
    saguaro: dict[str, Any],
    perf: dict[str, Any],
    host_contract_sha256: str,
    tuning_contracts: dict[str, Any],
    graph_preflight: dict[str, Any],
    benchmark_harness_digest: str,
) -> dict[str, Any]:
    decision = "pass" if not failures else "fail"
    return {
        "schema_version": "native_qsg_suite.admission_manifest.v1",
        "profile_name": spec.profile_name,
        "decision": decision,
        "passed": decision == "pass",
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": list(failures),
        "warnings": list(warnings),
        "required_evidence": {
            "host_contract_sha256": str(host_contract_sha256 or ""),
            "benchmark_harness_hash": str(benchmark_harness_digest or ""),
            "saguaro_ok": bool(
                (saguaro.get("health") or {}).get("ok", saguaro.get("ok", False))
            ),
            "perf_ok": bool(perf.get("available", False)),
            "graph_preflight_ok": bool(graph_preflight.get("passed", False)),
        },
        "tuning_contracts": {
            model: {
                "valid": bool(details.get("valid", False)),
                "readiness_state": str(details.get("readiness_state") or ""),
                "issues": list(details.get("issues") or []),
                "path": str(details.get("path") or ""),
            }
            for model, details in sorted(tuning_contracts.items())
        },
    }


def _remediation_for_failure(failure: str) -> str:
    if failure == "virtualenv_missing":
        return "Activate the repo virtualenv before running the suite, for example `source venv/bin/activate`."
    if failure == "saguaro_health_failed":
        return "Run `source venv/bin/activate && saguaro health` and repair or rebuild the index until health is green."
    if failure == "saguaro_cpu_scan_failed":
        return "Run `source venv/bin/activate && saguaro cpu scan --path core/simd/common/perf_utils.h --arch x86_64-avx2 --format json` and resolve the scan/runtime breakage before benchmarking."
    if failure == "perf_unavailable":
        return "Lower `/proc/sys/kernel/perf_event_paranoid` and confirm both `perf stat true` and `perf record -- true` succeed."
    if failure.startswith("cpu_governor="):
        return "Set the CPU governor to `performance` on the benchmark host before running the strict benchmark suite."
    if failure.startswith("thp_mode="):
        return "Set Transparent Huge Pages to a permitted mode such as `never` or `madvise`."
    if failure.startswith("model_digest_invalid:"):
        model = failure.split(":", 1)[1]
        return f"Refresh or restage the Ollama weights for `{model}` so the audited digest matches the expected manifest."
    if failure.startswith("strict_native_unsupported:"):
        model = failure.split(":", 1)[1]
        return f"Re-export or replace `{model}` with a strict-native-compatible model package."
    if failure.startswith("memory_available_kb<"):
        return "Free memory on the benchmark host or lower the suite footprint before retrying."
    if failure.startswith("visible_threads<"):
        return (
            "Your launch context is CPU-limited. "
            "For certified runs, start the suite from a shell that already sees the full host thread budget."
        )
    if failure == "launch_affinity_constrained":
        return "Re-launch the suite from an unconstrained shell; certified and calibration runs refuse repaired affinity."
    if failure == "affinity_repair_attempted":
        return "Do not rely on automatic affinity repair for certified runs. Fix the launch context first."
    if failure.startswith("host_contract_missing:"):
        return "Add the certified host contract under `audit/contracts/hosts/` or select a profile without host certification."
    if failure.startswith("host_contract:"):
        return "Correct the host contract mismatch or use the matching certified machine before running the strict suite."
    if failure.startswith("tuning_contract_missing:"):
        model = failure.split(":", 1)[1]
        return f"Run `./scripts/run_native_qsg_suite.sh --profile calibrate` on the certified host to generate tuning for `{model}`."
    if failure.startswith("tuning_contract:"):
        return "Refresh calibration on the certified host so the tuning contract matches the current model digest and harness."
    if failure == "graph_preflight_failed":
        return "Review `telemetry/graph_preflight.json`, resolve unresolved native-boundary drift, then rerun preflight."
    return "Inspect preflight.json for the failing check and correct host state before re-running."


def run_preflight(
    *,
    repo_root: Path,
    spec: BenchmarkSuiteSpec,
    telemetry_dir: Path,
    launch_runtime: dict[str, Any],
    affinity_adjustment: dict[str, Any],
) -> PreflightResult:
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    logger = get_active_logger()
    strictness = _preflight_strictness(spec)
    runtime = capture_runtime_provenance(repo_root)
    if not dict(runtime.get("native_library") or {}):
        try:
            from core.native.native_ops import get_native_library_info

            runtime["native_library"] = get_native_library_info()
        except Exception as exc:
            runtime["native_library"] = {"error": str(exc)}
    models = {model: capture_model_provenance(model) for model in spec.models}
    if logger is not None:
        logger.emit(
            level="info",
            source="suite_preflight",
            event_type="preflight_start",
            message=f"starting preflight for profile {spec.profile_name}",
            phase="preflight",
        )
    saguaro = _saguaro_probe(repo_root, telemetry_dir, strictness=strictness)
    perf = _perf_probe(telemetry_dir)
    dependency_report = _dependency_report(repo_root)
    dependency_report_path = telemetry_dir / "dependency_report.json"
    dependency_report_path.write_text(
        json.dumps(dependency_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    thp_mode = _selected_mode(_read_text("/sys/kernel/mm/transparent_hugepage/enabled"))
    governor = _cpu_governor()
    perf_paranoid = _read_text("/proc/sys/kernel/perf_event_paranoid")
    numa_topology_path = telemetry_dir / "lscpu.json"
    lscpu_code, lscpu_out, lscpu_err = _run(
        ["lscpu", "--json"],
        telemetry_dir=telemetry_dir,
        label="lscpu",
    )
    if lscpu_out:
        numa_topology_path.write_text(lscpu_out, encoding="utf-8")
        if logger is not None:
            logger.emit(
                level="debug",
                source="suite_preflight",
                event_type="telemetry_artifact",
                message=f"wrote lscpu telemetry -> {numa_topology_path}",
                phase="preflight",
            )
    meminfo = _read_text("/proc/meminfo")
    meminfo_path = telemetry_dir / "meminfo.txt"
    if meminfo:
        meminfo_path.write_text(meminfo + "\n", encoding="utf-8")
        if logger is not None:
            logger.emit(
                level="debug",
                source="suite_preflight",
                event_type="telemetry_artifact",
                message=f"wrote meminfo telemetry -> {meminfo_path}",
                phase="preflight",
            )
    meminfo_values = _meminfo_map(meminfo)
    requirements = dict(spec.host_requirements or {})
    host = dict(runtime.get("host") or {})
    launch_host = dict((launch_runtime.get("host") or {}))
    host_contract: dict[str, Any] | None = None
    host_contract_path = ""
    host_contract_sha256 = ""
    certified_host_match = False
    tuning_contracts: dict[str, Any] = {}
    repair_allowed = bool(affinity_adjustment.get("repair_allowed", False))
    repair_attempted = bool(affinity_adjustment.get("attempted", False))
    repair_required = bool(affinity_adjustment.get("repair_required", False))
    launch_affinity = list(affinity_adjustment.get("before") or [])
    post_adjustment_affinity = list(affinity_adjustment.get("after") or launch_affinity)

    failures: list[str] = []
    warnings: list[str] = []
    virtual_env = str((runtime.get("python") or {}).get("virtual_env") or "")
    if not virtual_env:
        failures.append("virtualenv_missing")
    if spec.require_saguaro and not bool(
        (saguaro.get("health") or {}).get("ok", saguaro.get("ok", False))
    ):
        failures.append("saguaro_health_failed")
    if spec.require_saguaro and not bool(saguaro.get("cpu_scan_ok", True)):
        failures.append("saguaro_cpu_scan_failed")
    if spec.require_perf and not bool(perf.get("available")):
        failures.append("perf_unavailable")
    if not bool(
        (dependency_report.get("dev_metadata") or {})
        .get("hwloc_pkg_config", {})
        .get("present", False)
    ):
        warnings.append("dependency_hwloc_pkg_config_missing")
    if logger is not None:
        logger.emit(
            level="info",
            source="suite_preflight",
            event_type="host_state",
            message=(
                f"governor={governor or 'unknown'} thp={thp_mode or 'unknown'} "
                f"perf_ok={bool(perf.get('available', False))} "
                f"saguaro_ok={bool((saguaro.get('health') or {}).get('ok', saguaro.get('ok', False)))} "
                f"cpu_scan_ok={bool(saguaro.get('cpu_scan_ok', True))}"
            ),
            phase="preflight",
        )
    required_governor = str(requirements.get("cpu_governor") or "").strip()
    if required_governor and governor and governor != required_governor:
        failures.append(f"cpu_governor={governor}")
    allowed_thp_modes = {
        str(item).strip()
        for item in list(requirements.get("thp_modes") or [])
        if str(item).strip()
    }
    if allowed_thp_modes and thp_mode and thp_mode not in allowed_thp_modes:
        failures.append(f"thp_mode={thp_mode}")
    min_available_kb = int(requirements.get("min_mem_available_kb", 0) or 0)
    if min_available_kb > 0:
        available_kb = int(meminfo_values.get("MemAvailable", 0) or 0)
        if available_kb < min_available_kb:
            failures.append(f"memory_available_kb<{min_available_kb}")
    min_visible_threads = int(requirements.get("min_visible_threads", 0) or 0)
    if min_visible_threads > 0:
        visible_threads = int(
            host.get("visible_threads", host.get("logical_cpus", 1)) or 1
        )
        if visible_threads < min_visible_threads:
            warning = f"visible_threads<{min_visible_threads}"
            warnings.append(warning)
            if bool(requirements.get("enforce_min_visible_threads", False)):
                failures.append(warning)
    for model, contract in models.items():
        if not bool(contract.get("digest_validated", False)):
            failures.append(f"model_digest_invalid:{model}")
        if not bool(contract.get("strict_native_supported", False)):
            failures.append(f"strict_native_unsupported:{model}")

    resolved_host_contract_id = ""
    if spec.host_contract_id:
        resolved_host_contract_id = resolve_host_contract_id(
            spec.host_contract_id, launch_runtime
        )
        try:
            host_contract, host_contract_file, _ = ensure_host_contract(
                repo_root,
                contract_id=resolved_host_contract_id,
                launch_runtime=launch_runtime,
                cpu_governor=governor,
                thp_mode=thp_mode,
                allow_create=str(spec.host_contract_id or "").strip()
                in {"", "auto", "current"},
            )
            host_contract_path = str(host_contract_file)
            host_contract_sha256 = sha256_file(host_contract_file)
        except Exception:
            failures.append(
                f"host_contract_missing:{resolved_host_contract_id or spec.host_contract_id}"
            )
            host_contract = None
        if host_contract is not None:
            contract_issues = validate_host_contract(
                host_contract,
                launch_host=launch_host,
                governor=governor,
                thp_mode=thp_mode,
            )
            for issue in contract_issues:
                failures.append(f"host_contract:{issue}")
            certified_host_match = not contract_issues
            if spec.affinity_policy == "certified_exact":
                if repair_required:
                    failures.append("launch_affinity_constrained")
                if repair_attempted:
                    failures.append("affinity_repair_attempted")
        if logger is not None:
            logger.emit(
                level="debug",
                source="suite_preflight",
                event_type="host_contract",
                message=(
                    f"resolved host contract {resolved_host_contract_id or spec.host_contract_id} "
                    f"match={certified_host_match}"
                ),
                phase="preflight",
            )

    if (
        spec.tuning_contract_policy in {"required", "optional"}
        and host_contract is not None
    ):
        for model_name, contract in models.items():
            tuning_payload, tuning_path = load_tuning_contract(
                repo_root,
                str(host_contract.get("host_fingerprint") or ""),
                model_name,
            )
            if tuning_payload is None:
                issue = f"tuning_contract_missing:{model_name}"
                if spec.tuning_contract_policy == "required":
                    if strictness == "certify":
                        failures.append(issue)
                    else:
                        warnings.append(issue)
                tuning_contracts[model_name] = {
                    "path": str(tuning_path),
                    "valid": False,
                    "issues": ["missing"],
                    "readiness_state": "missing",
                }
                continue
            issues = validate_tuning_contract(
                tuning_payload,
                host_contract=host_contract,
                host_contract_sha256=host_contract_sha256,
                model_name=model_name,
                model_contract=contract,
                profile_schema_version=spec.schema_version,
                repo_root=repo_root,
                workload_digest="",
            )
            if issues:
                container = (
                    failures
                    if spec.tuning_contract_policy == "required"
                    and strictness == "certify"
                    else warnings
                )
                container.extend(
                    f"tuning_contract:{model_name}:{issue}" for issue in issues
                )
            tuning_contracts[model_name] = {
                "path": str(tuning_path),
                "valid": not issues,
                "issues": issues,
                "readiness_state": contract_readiness_state(issues),
                "thread_config": dict(tuning_payload.get("thread_config") or {}),
                "continuous_config": dict(
                    tuning_payload.get("continuous_config") or {}
                ),
                "pager_config": dict(tuning_payload.get("pager_config") or {}),
                "admission": dict(tuning_payload.get("admission") or {}),
                "model_digest": str(tuning_payload.get("model_digest") or ""),
                "benchmark_harness_hash": str(
                    tuning_payload.get("benchmark_harness_hash") or ""
                ),
            }
    elif (
        spec.tuning_contract_policy == "generate"
        and host_contract is not None
        and spec.affinity_policy == "certified_exact"
    ):
        expected_visible = int(host_contract.get("required_visible_threads", 0) or 0)
        if (
            expected_visible > 0
            and int(launch_host.get("visible_threads", 0) or 0) != expected_visible
        ):
            failures.append("launch_affinity_constrained")

    host["host_fingerprint_expected"] = host_fingerprint()
    host["python_execution_target"] = platform.python_implementation()
    certification_state = (
        "certified_candidate"
        if certified_host_match and not repair_required and not repair_attempted
        else "non_certified"
    )
    benchmark_metadata = _build_benchmark_metadata(
        spec=spec,
        launch_runtime=launch_runtime,
        runtime=runtime,
        models=models,
        governor=governor,
        thp_mode=thp_mode,
        perf_paranoid=perf_paranoid,
        launch_affinity=launch_affinity,
        post_adjustment_affinity=post_adjustment_affinity,
    )
    graph_preflight = _graph_preflight(
        repo_root=repo_root,
        telemetry_dir=telemetry_dir,
        spec=spec,
    )
    graph_preflight_required = strictness == "certify"
    if not bool(graph_preflight.get("passed", True)) and graph_preflight_required:
        failures.append("graph_preflight_failed")
    elif not bool(graph_preflight.get("passed", True)):
        warnings.append("graph_preflight_warning")
    tuning_state, tuning_state_groups = summarize_tuning_states(tuning_contracts)
    primary_model = str(spec.models[0] if spec.models else "")
    primary_tuning = dict(
        (tuning_contracts.get(primary_model) or {}).get("thread_config") or {}
    )
    native_library = dict(runtime.get("native_library") or {})
    capability_ledger = build_runtime_capability_ledger(
        {
            "model": primary_model,
            "digest": str((models.get(primary_model) or {}).get("digest") or ""),
            "native_isa_baseline": str(
                native_library.get("native_isa_baseline")
                or host.get("machine", platform.machine())
            ),
            "decode_threads": int(primary_tuning.get("decode_threads", 0) or 0),
            "batch_threads": int(primary_tuning.get("batch_threads", 0) or 0),
            "ubatch": int(primary_tuning.get("ubatch", 0) or 0),
            "generation_mode": (
                "parallel_hybrid" if spec.force_parallel_decode else "ar_baseline"
            ),
            "affinity_policy": str(spec.affinity_policy or ""),
            "parallel_decode_allowed": bool(spec.force_parallel_decode),
            "full_qsg_enabled": bool(spec.force_parallel_decode),
            "full_graph_enabled": bool(spec.require_saguaro),
            "qsg_processors_native_enabled": bool(spec.require_saguaro),
            "batched_prefill_native_enabled": bool(spec.require_saguaro),
            "native_backend_abi_match": bool(not failures),
            "perf_event_access": bool(perf.get("available", False)),
            "native_optional_isa_leaves": list(
                native_library.get("native_optional_isa_leaves", []) or []
            ),
            "native_compiled_with_amx": bool(
                native_library.get("native_compiled_with_amx", False)
            ),
            "native_runtime_amx_available": bool(
                native_library.get("native_runtime_amx_available", False)
            ),
        },
        host_fingerprint=str(host.get("host_fingerprint") or host_fingerprint()),
        certification_state=certification_state,
        source="suite_preflight",
    )
    capability_ledger_path = telemetry_dir / "capability_ledger.json"
    capability_ledger_path.write_text(
        json.dumps(capability_ledger, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    admission_manifest = _admission_manifest(
        spec=spec,
        failures=failures,
        warnings=warnings,
        saguaro=saguaro,
        perf=perf,
        host_contract_sha256=host_contract_sha256,
        tuning_contracts=tuning_contracts,
        graph_preflight=graph_preflight,
        benchmark_harness_digest=benchmark_harness_hash(repo_root),
    )
    admission_manifest_path = telemetry_dir / "admission_manifest.json"
    admission_manifest_path.write_text(
        json.dumps(admission_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "native_qsg_suite.preflight.v1",
        "ok": not failures,
        "failures": failures,
        "warnings": warnings,
        "preflight_strictness": strictness,
        "strict_host": spec.strict_host,
        "repo_root": str(repo_root),
        "launch_runtime": launch_runtime,
        "runtime": runtime,
        "models": models,
        "saguaro": saguaro,
        "perf": perf,
        "host": host,
        "launch_host": launch_host,
        "cpu_governor": governor,
        "thp_mode": thp_mode,
        "perf_event_paranoid": perf_paranoid,
        "benchmark_metadata": benchmark_metadata,
        "launch_affinity": launch_affinity,
        "post_adjustment_affinity": post_adjustment_affinity,
        "repair_allowed": repair_allowed,
        "repair_attempted": repair_attempted,
        "repair_required": repair_required,
        "certified_host_match": certified_host_match,
        "certification_state": certification_state,
        "host_contract": host_contract,
        "resolved_host_contract_id": resolved_host_contract_id,
        "host_contract_path": host_contract_path,
        "host_contract_sha256": host_contract_sha256,
        "tuning_contracts": tuning_contracts,
        "tuning_state": tuning_state,
        "tuning_state_groups": tuning_state_groups,
        "benchmark_harness_hash": benchmark_harness_hash(repo_root),
        "graph_preflight": {
            **graph_preflight,
            "artifact": str(graph_preflight.get("artifact") or ""),
        },
        "capability_ledger": {
            **capability_ledger,
            "artifact": str(capability_ledger_path),
        },
        "admission_manifest": {
            **admission_manifest,
            "artifact": str(admission_manifest_path),
        },
        "lscpu": {
            "ok": lscpu_code == 0,
            "artifact": str(numa_topology_path) if lscpu_out else "",
            "stderr": lscpu_err,
        },
        "memory": {
            "meminfo_artifact": str(meminfo_path) if meminfo else "",
            "meminfo": meminfo_values,
        },
        "dependency_report": {
            **dependency_report,
            "artifact": str(dependency_report_path),
        },
        "requirements": requirements,
        "remediations": {
            failure: _remediation_for_failure(failure) for failure in failures
        },
    }
    if logger is not None:
        logger.emit(
            level="info" if not failures else "warn",
            source="suite_preflight",
            event_type="preflight_complete",
            message=(
                f"preflight {'passed' if not failures else 'failed'} "
                f"failures={len(failures)} warnings={len(warnings)}"
            ),
            phase="preflight",
            payload={
                "failure_count": len(failures),
                "warning_count": len(warnings),
            },
        )
    return PreflightResult(payload=payload, ok=not failures)
