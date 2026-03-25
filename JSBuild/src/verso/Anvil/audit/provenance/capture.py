from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.model.model_contract import model_contract_snapshot, resolve_model_contract


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


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


def _cpuinfo_field(field_name: str) -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return ""
    needle = field_name.lower()
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip().lower() == needle:
            return value.strip()
    return ""


def _cpu_model() -> str:
    return _cpuinfo_field("model name") or platform.processor()


def _cpu_governor() -> str:
    return _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")


def _transparent_hugepage_mode() -> str:
    return _selected_mode(_read_text("/sys/kernel/mm/transparent_hugepage/enabled"))


def _parse_meminfo(raw: str) -> dict[str, int]:
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


def _detect_memory_speed_mt_s() -> tuple[int | None, str]:
    dmidecode_path = shutil.which("dmidecode")
    if not dmidecode_path:
        return None, "dmidecode_missing"
    try:
        completed = subprocess.run(
            [dmidecode_path, "--type", "17"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        return None, f"dmidecode_error:{exc}"
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        reason = stderr.splitlines()[0].strip() if stderr else "unavailable"
        return None, f"dmidecode_failed:{reason}"
    matches = [
        int(match)
        for match in re.findall(
            r"(?:Configured Memory Speed|Speed):\s*(\d+)\s*MT/s",
            completed.stdout,
        )
    ]
    if not matches:
        return None, "not_detected"
    return max(matches), "dmidecode"


def _numa_policy_snapshot() -> dict[str, Any]:
    numactl_path = shutil.which("numactl")
    payload: dict[str, Any] = {
        "available": False,
        "policy": "",
        "preferred": "",
        "physcpubind": "",
        "cpubind": "",
        "nodebind": "",
        "membind": "",
        "interleave": "",
        "source": "numactl_missing" if not numactl_path else "numactl",
    }
    if not numactl_path:
        return payload
    try:
        completed = subprocess.run(
            [numactl_path, "--show"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        payload["source"] = f"numactl_error:{exc}"
        return payload
    raw = "\n".join(
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    )
    if completed.returncode != 0:
        payload["source"] = f"numactl_failed:{raw.splitlines()[0] if raw else 'unknown'}"
        return payload
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = key.strip().lower().replace(" ", "_")
        if normalized in payload:
            payload[normalized] = value.strip()
    payload["available"] = True
    payload["raw"] = raw
    return payload


def _memory_snapshot() -> dict[str, Any]:
    meminfo = _parse_meminfo(_read_text("/proc/meminfo"))
    memory_speed_mt_s, memory_speed_source = _detect_memory_speed_mt_s()
    return {
        "mem_total_kb": int(meminfo.get("MemTotal", 0) or 0),
        "mem_available_kb": int(meminfo.get("MemAvailable", 0) or 0),
        "hugepages_total": int(meminfo.get("HugePages_Total", 0) or 0),
        "hugepages_free": int(meminfo.get("HugePages_Free", 0) or 0),
        "hugepage_size_kb": int(meminfo.get("Hugepagesize", 0) or 0),
        "memory_speed_mt_s": memory_speed_mt_s,
        "memory_speed_source": memory_speed_source,
    }


def host_fingerprint() -> str:
    payload = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "logical_cpus": int(os.cpu_count() or 1),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def capture_runtime_provenance(
    repo_root: Path,
    *,
    include_native_library: bool = False,
) -> dict[str, Any]:
    git_sha = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_dirty = bool(_run(["git", "status", "--porcelain"], cwd=repo_root))
    try:
        affinity = os.sched_getaffinity(0)
        visible_threads = int(len(affinity)) if affinity else int(os.cpu_count() or 1)
    except Exception:
        visible_threads = int(os.cpu_count() or 1)

    env_keys = sorted(
        key
        for key in os.environ
        if key.startswith("ANVIL_") or key.startswith("OMP_")
    )
    env_snapshot = {key: os.environ.get(key, "") for key in env_keys}
    memory = _memory_snapshot()
    numa = _numa_policy_snapshot()
    threading = {
        "omp_num_threads": env_snapshot.get("OMP_NUM_THREADS", ""),
        "omp_proc_bind": env_snapshot.get("OMP_PROC_BIND", ""),
        "omp_places": env_snapshot.get("OMP_PLACES", ""),
        "visible_threads": visible_threads,
    }

    native_lib_info: dict[str, Any] = {}
    if include_native_library:
        try:
            from core.native.native_ops import get_native_library_info

            native_lib_info = get_native_library_info()
        except Exception as exc:
            native_lib_info = {"error": str(exc)}

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "git": {
            "commit": git_sha,
            "dirty": git_dirty,
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "virtual_env": os.getenv("VIRTUAL_ENV", ""),
        },
        "host": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "kernel_release": platform.release(),
            "cpu_model": _cpu_model(),
            "microcode_version": _cpuinfo_field("microcode"),
            "logical_cpus": int(os.cpu_count() or 1),
            "visible_threads": visible_threads,
            "host_fingerprint": host_fingerprint(),
            "cpu_governor": _cpu_governor(),
            "transparent_hugepage_mode": _transparent_hugepage_mode(),
        },
        "env": env_snapshot,
        "threading": threading,
        "memory": memory,
        "numa": numa,
        "native_library": native_lib_info,
    }


def capture_model_provenance(model_name: str) -> dict[str, Any]:
    try:
        contract = resolve_model_contract(model_name)
    except Exception as exc:
        return {
            "model": model_name,
            "contract_error": str(exc),
        }
    return model_contract_snapshot(contract)
