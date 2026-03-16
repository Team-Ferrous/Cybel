from __future__ import annotations

import os
import platform
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from audit.provenance.capture import (
    capture_model_provenance,
    capture_runtime_provenance,
    host_fingerprint,
)
from core.native.runtime_telemetry import build_runtime_capability_ledger

HOST_CONTRACT_DIR = Path("audit/contracts/hosts")
TUNING_CONTRACT_DIR = Path("audit/contracts/tuning")
AUTO_HOST_CONTRACT_ID = "auto"
DEFAULT_RUNTIME_TUNING_BOOTSTRAP_POLICY = "on_first_run"
HARNESS_FILES = (
    "audit/runner/benchmark_suite.py",
    "audit/runner/attempt_executor.py",
    "audit/runner/suite_preflight.py",
    "audit/runner/suite_profiles.py",
    "benchmarks/native_qsg_benchmark.py",
    "benchmarks/native_kernel_microbench.py",
    "core/native/native_qsg_engine.py",
    "core/native/runtime_telemetry.py",
    "core/native/thread_config.cpp",
    "core/native/model_graph.cpp",
    "core/native/quantized_matmul.cpp",
    "core/native/tinyblas_kernels.cpp",
    "core/native/CMakeLists.txt",
)
AUTO_CALIBRATION_PROFILES = frozenset({"gold", "platinum"})
TUNING_DECISIONS = frozenset({"skip", "probe", "search", "revoke"})


def safe_model_name(model: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(model)
    )


def sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def parse_cpu_list(raw: str) -> set[int]:
    cpus: set[int] = set()
    text = str(raw or "").strip()
    if not text:
        return cpus
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            try:
                start = int(start_text)
                end = int(end_text)
            except Exception:
                continue
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
            continue
        try:
            cpus.add(int(token))
        except Exception:
            continue
    return {cpu for cpu in cpus if cpu >= 0}


def cpu_target_candidates() -> list[str]:
    candidates: list[str] = []
    for key in (
        "ANVIL_SUITE_TARGET_CPUS",
        "ANVIL_CPU_TARGET_CPUS",
        "GOMP_CPU_AFFINITY",
    ):
        value = str(os.getenv(key, "") or "").strip()
        if value:
            candidates.append(value)

    if "sched_getaffinity" in dir(os):
        try:
            affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        except Exception:
            affinity = []
        if affinity:
            candidates.append(",".join(str(cpu) for cpu in affinity))

    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "/sys/devices/system/cpu/online",
    ):
        candidate = Path(path)
        if candidate.exists():
            value = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            if value:
                candidates.append(value)
    logical = int(os.cpu_count() or 1)
    if logical > 0:
        candidates.append(f"0-{logical - 1}")
    return candidates


def best_cpu_target() -> set[int]:
    for raw in cpu_target_candidates():
        parsed = parse_cpu_list(raw)
        if parsed:
            return parsed
    return {0}


def expected_visible_threads(fallback: int | None = None) -> int:
    target = best_cpu_target()
    if target:
        return int(len(target))
    return max(1, int(fallback or os.cpu_count() or 1))


def benchmark_harness_hash(repo_root: Path) -> str:
    digest = hashlib.sha256()
    for raw_path in HARNESS_FILES:
        path = repo_root / raw_path
        digest.update(raw_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def host_contract_path(repo_root: Path, contract_id: str) -> Path:
    return repo_root / HOST_CONTRACT_DIR / f"{str(contract_id).strip()}.yaml"


def tuning_contract_path(repo_root: Path, host_fingerprint: str, model: str) -> Path:
    return (
        repo_root
        / TUNING_CONTRACT_DIR
        / str(host_fingerprint).strip()
        / f"{safe_model_name(model)}.json"
    )


def load_host_contract(
    repo_root: Path, contract_id: str
) -> tuple[dict[str, Any], Path]:
    path = host_contract_path(repo_root, contract_id)
    if not path.exists():
        raise RuntimeError(f"Host contract not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Host contract must decode to a mapping: {path}")
    return dict(payload), path


def resolve_host_contract_id(
    contract_id: str | None,
    launch_runtime: dict[str, Any] | None = None,
) -> str:
    raw = str(contract_id or "").strip()
    if raw and raw not in {AUTO_HOST_CONTRACT_ID, "current"}:
        return raw
    host = dict((launch_runtime or {}).get("host") or {})
    resolved = str(host.get("host_fingerprint") or "").strip()
    return resolved or host_fingerprint()


def build_host_contract_payload(
    *,
    contract_id: str,
    launch_runtime: dict[str, Any],
    cpu_governor: str = "",
    thp_mode: str = "",
    lscpu_payload: dict[str, Any] | list[Any] | None = None,
) -> dict[str, Any]:
    host = dict(launch_runtime.get("host") or {})
    logical_cpus = int(host.get("logical_cpus", 0) or 0) or max(
        1, int(os.cpu_count() or 1)
    )
    required_threads = expected_visible_threads(logical_cpus)
    payload: dict[str, Any] = {
        "schema_version": "native_qsg_suite.host_contract.v1",
        "contract_id": str(contract_id).strip(),
        "contract_origin": "auto_generated",
        "generated_at": str(launch_runtime.get("captured_at") or ""),
        "host_fingerprint": str(host.get("host_fingerprint") or contract_id),
        "logical_cpus": logical_cpus,
        "required_visible_threads": required_threads,
        "cpu_governor": str(cpu_governor or "").strip(),
        "thp_modes": [str(thp_mode).strip()] if str(thp_mode or "").strip() else [],
        "host": {
            "hostname": str(host.get("hostname") or platform.node()),
            "machine": str(host.get("machine") or platform.machine()),
            "platform": str(host.get("platform") or platform.platform()),
            "cpu_model": str(host.get("cpu_model") or ""),
        },
    }
    if lscpu_payload:
        payload["lscpu"] = lscpu_payload
    return payload


def write_host_contract(repo_root: Path, payload: dict[str, Any]) -> Path:
    contract_id = str(
        payload.get("contract_id") or payload.get("host_fingerprint") or ""
    ).strip()
    if not contract_id:
        raise RuntimeError("Host contract payload is missing contract_id.")
    path = host_contract_path(repo_root, contract_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def ensure_host_contract(
    repo_root: Path,
    *,
    contract_id: str | None,
    launch_runtime: dict[str, Any],
    cpu_governor: str = "",
    thp_mode: str = "",
    lscpu_payload: dict[str, Any] | list[Any] | None = None,
    allow_create: bool = False,
) -> tuple[dict[str, Any], Path, bool]:
    resolved_id = resolve_host_contract_id(contract_id, launch_runtime)
    path = host_contract_path(repo_root, resolved_id)
    if path.exists():
        payload, loaded_path = load_host_contract(repo_root, resolved_id)
        return payload, loaded_path, False
    if not allow_create:
        raise RuntimeError(f"Host contract not found: {path}")
    payload = build_host_contract_payload(
        contract_id=resolved_id,
        launch_runtime=launch_runtime,
        cpu_governor=cpu_governor,
        thp_mode=thp_mode,
        lscpu_payload=lscpu_payload,
    )
    written = write_host_contract(repo_root, payload)
    return dict(payload), written, True


def load_tuning_contract(
    repo_root: Path, host_fingerprint: str, model: str
) -> tuple[dict[str, Any] | None, Path]:
    path = tuning_contract_path(repo_root, host_fingerprint, model)
    if not path.exists():
        return None, path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Tuning contract must decode to an object: {path}")
    return dict(payload), path


def write_tuning_contract(
    repo_root: Path,
    *,
    host_fingerprint: str,
    model: str,
    payload: dict[str, Any],
) -> Path:
    path = tuning_contract_path(repo_root, host_fingerprint, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def model_digest(model_contract: dict[str, Any]) -> str:
    for key in ("digest", "expected_model_digest"):
        value = str(model_contract.get(key) or "").strip()
        if value:
            return value
    return ""


def validate_host_contract(
    contract: dict[str, Any],
    *,
    launch_host: dict[str, Any],
    governor: str,
    thp_mode: str,
) -> list[str]:
    issues: list[str] = []
    expected_fingerprint = str(
        contract.get("host_fingerprint") or contract.get("contract_id") or ""
    ).strip()
    if (
        expected_fingerprint
        and str(launch_host.get("host_fingerprint") or "").strip()
        != expected_fingerprint
    ):
        issues.append(f"host_fingerprint_mismatch:{expected_fingerprint}")

    expected_logical = int(contract.get("logical_cpus", 0) or 0)
    if (
        expected_logical > 0
        and int(launch_host.get("logical_cpus", 0) or 0) != expected_logical
    ):
        issues.append(f"logical_cpus_mismatch:{expected_logical}")

    expected_visible = int(contract.get("required_visible_threads", 0) or 0)
    if (
        expected_visible > 0
        and int(launch_host.get("visible_threads", 0) or 0) != expected_visible
    ):
        issues.append(f"required_visible_threads_mismatch:{expected_visible}")

    expected_governor = str(contract.get("cpu_governor") or "").strip()
    if expected_governor and governor and governor != expected_governor:
        issues.append(f"cpu_governor_contract_mismatch:{expected_governor}")

    allowed_thp = {
        str(item).strip()
        for item in list(contract.get("thp_modes") or [])
        if str(item).strip()
    }
    if allowed_thp and thp_mode and thp_mode not in allowed_thp:
        issues.append(f"thp_mode_contract_mismatch:{thp_mode}")
    return issues


def validate_tuning_contract(
    contract: dict[str, Any],
    *,
    host_contract: dict[str, Any],
    host_contract_sha256: str,
    model_name: str,
    model_contract: dict[str, Any],
    profile_schema_version: str,
    repo_root: Path,
    workload_digest: str = "",
) -> list[str]:
    issues: list[str] = []
    expected_host_id = str(
        host_contract.get("contract_id") or host_contract.get("host_fingerprint") or ""
    ).strip()
    expected_host_fingerprint = str(host_contract.get("host_fingerprint") or "").strip()
    expected_model_digest = model_digest(model_contract)
    expected_harness_hash = benchmark_harness_hash(repo_root)

    if str(contract.get("host_contract_id") or "").strip() != expected_host_id:
        issues.append(f"host_contract_id_mismatch:{expected_host_id}")
    if str(contract.get("host_fingerprint") or "").strip() != expected_host_fingerprint:
        issues.append(f"host_fingerprint_mismatch:{expected_host_fingerprint}")
    if str(contract.get("model") or "").strip() != str(model_name).strip():
        issues.append(f"model_mismatch:{model_name}")
    if str(contract.get("model_digest") or "").strip() != expected_model_digest:
        issues.append(f"model_digest_mismatch:{expected_model_digest}")
    if (
        str(contract.get("profile_schema_version") or "").strip()
        != str(profile_schema_version).strip()
    ):
        issues.append(f"profile_schema_version_mismatch:{profile_schema_version}")
    if (
        str(contract.get("benchmark_harness_hash") or "").strip()
        != expected_harness_hash
    ):
        issues.append(f"benchmark_harness_hash_mismatch:{expected_harness_hash}")

    contract_hashes = dict(contract.get("contract_hashes") or {})
    if (
        str(contract_hashes.get("host_contract_sha256") or "").strip()
        != host_contract_sha256
    ):
        issues.append(f"host_contract_sha256_mismatch:{host_contract_sha256}")

    thread_config = dict(contract.get("thread_config") or {})
    for key in ("decode_threads", "batch_threads", "ubatch"):
        value = thread_config.get(key)
        if not isinstance(value, int) or value <= 0:
            issues.append(f"thread_config_invalid:{key}")
    continuous_config = dict(contract.get("continuous_config") or {})
    for key in (
        "scheduler_policy",
        "max_active_requests",
        "batch_wait_timeout_ms",
        "max_prefill_rows_per_iteration",
        "continuous_interleaved_streams",
    ):
        if key not in continuous_config:
            issues.append(f"continuous_config_missing:{key}")
    if "scheduler_policy" in continuous_config and str(
        continuous_config.get("scheduler_policy") or ""
    ).strip().lower() not in {"fcfs", "priority"}:
        issues.append("continuous_config_invalid:scheduler_policy")
    for key in (
        "max_active_requests",
        "batch_wait_timeout_ms",
        "max_prefill_rows_per_iteration",
    ):
        value = continuous_config.get(key)
        if not isinstance(value, int) or value <= 0:
            issues.append(f"continuous_config_invalid:{key}")
    if not isinstance(continuous_config.get("continuous_interleaved_streams"), bool):
        issues.append("continuous_config_invalid:continuous_interleaved_streams")
    pager_config = dict(contract.get("pager_config") or {})
    for key in (
        "state_page_rows",
        "state_compaction_soft_threshold",
        "state_compaction_hard_threshold",
    ):
        if key not in pager_config:
            issues.append(f"pager_config_missing:{key}")
    if (
        not isinstance(pager_config.get("state_page_rows"), int)
        or int(pager_config.get("state_page_rows", 0) or 0) <= 0
    ):
        issues.append("pager_config_invalid:state_page_rows")
    for key in ("state_compaction_soft_threshold", "state_compaction_hard_threshold"):
        value = pager_config.get(key)
        if not isinstance(value, (int, float)) or float(value) <= 0.0:
            issues.append(f"pager_config_invalid:{key}")
    objective_vector = dict(contract.get("objective_vector") or {})
    for key in (
        "decode_tps_median",
        "ttft_ms_median",
        "queue_wait_ms_p95",
        "fairness",
        "decode_goodput_tps",
    ):
        if key not in objective_vector:
            issues.append(f"objective_vector_missing:{key}")
    admission = dict(contract.get("admission") or {})
    for key in ("decision", "budget_tier", "invocation_source", "workload_digest"):
        if not str(admission.get(key) or "").strip():
            issues.append(f"admission_missing:{key}")
    decision = str(admission.get("decision") or "").strip()
    if decision and decision not in TUNING_DECISIONS:
        issues.append(f"admission_invalid:decision:{decision}")
    if (
        workload_digest
        and str(admission.get("workload_digest") or "").strip()
        != str(workload_digest).strip()
    ):
        issues.append(f"workload_digest_mismatch:{workload_digest}")
    safe_envelope = dict(contract.get("safe_envelope") or {})
    for key in (
        "fairness_floor",
        "queue_wait_ms_p95_ceiling",
        "quality_regression_policy",
    ):
        if key not in safe_envelope:
            issues.append(f"safe_envelope_missing:{key}")
    quality_gate = dict(contract.get("quality_gate") or {})
    if not quality_gate:
        issues.append("quality_gate_missing")
    else:
        if not bool(quality_gate.get("passed", False)):
            issues.append("quality_gate_failed")
        evidence = dict(quality_gate.get("evidence") or {})
        if not evidence:
            issues.append("quality_gate_evidence_missing")
        else:
            for key in ("perplexity", "confidence", "coherence", "accuracy"):
                if not isinstance(evidence.get(key), dict):
                    issues.append(f"quality_gate_evidence_missing:{key}")
    return issues


def contract_readiness_state(issues: list[str]) -> str:
    normalized = [
        str(issue).strip() for issue in list(issues or []) if str(issue).strip()
    ]
    if not normalized:
        return "ready"
    if any(issue == "missing" for issue in normalized):
        return "missing"
    return "stale"


def summarize_tuning_states(
    tuning_contracts: dict[str, Any],
) -> tuple[str, dict[str, list[str]]]:
    grouped = {"ready": [], "stale": [], "missing": []}
    for model_name, payload in sorted((tuning_contracts or {}).items()):
        readiness_state = str(payload.get("readiness_state") or "").strip() or "stale"
        if readiness_state not in grouped:
            readiness_state = "stale"
        grouped[readiness_state].append(str(model_name))
    if grouped["missing"] and grouped["stale"]:
        return "mixed", grouped
    if grouped["missing"]:
        return "missing", grouped
    if grouped["stale"]:
        return "stale", grouped
    return "ready", grouped


def _normalize_profile_name(profile_name: str) -> str:
    normalized = str(profile_name or "").strip().lower()
    if normalized == "smoke":
        return "bronze"
    return normalized


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _workload_digest(spec: Any) -> str:
    payload = {
        "profile_name": str(getattr(spec, "profile_name", "") or ""),
        "models": list(getattr(spec, "models", []) or []),
        "canonical_max_new_tokens": int(
            getattr(getattr(spec, "scenario_pack", None), "canonical_max_new_tokens", 0)
            or 0
        ),
        "canonical_context_length": int(
            getattr(getattr(spec, "scenario_pack", None), "canonical_context_length", 0)
            or 0
        ),
        "continuous_concurrency": list(
            getattr(getattr(spec, "scenario_pack", None), "continuous_concurrency", [])
            or []
        ),
        "continuous_prompt_classes": list(
            getattr(
                getattr(spec, "scenario_pack", None), "continuous_prompt_classes", []
            )
            or []
        ),
        "continuous_scheduler_policies": list(
            getattr(
                getattr(spec, "scenario_pack", None),
                "continuous_scheduler_policies",
                [],
            )
            or []
        ),
        "calibration_surface": {
            "continuous_max_active_requests": list(
                getattr(
                    getattr(spec, "calibration_search", None),
                    "continuous_max_active_requests",
                    [],
                )
                or []
            ),
            "continuous_batch_wait_timeout_ms": list(
                getattr(
                    getattr(spec, "calibration_search", None),
                    "continuous_batch_wait_timeout_ms",
                    [],
                )
                or []
            ),
            "continuous_state_page_rows": list(
                getattr(
                    getattr(spec, "calibration_search", None),
                    "continuous_state_page_rows",
                    [],
                )
                or []
            ),
            "continuous_max_prefill_rows_per_iteration": list(
                getattr(
                    getattr(spec, "calibration_search", None),
                    "continuous_max_prefill_rows_per_iteration",
                    [],
                )
                or []
            ),
            "continuous_interleaved_streams": list(
                getattr(
                    getattr(spec, "calibration_search", None),
                    "continuous_interleaved_streams",
                    [],
                )
                or []
            ),
        },
    }
    return f"sha256:{_stable_hash(payload)}"


def _budget_tier_for_profile(
    profile_name: str,
    *,
    invocation_source: str,
    tuning_state: str,
) -> str:
    normalized = _normalize_profile_name(profile_name)
    if invocation_source == "repl_startup":
        return "probe"
    if normalized == "platinum":
        return "deep_search"
    if normalized in {"gold", "calibrate"}:
        return "search"
    if tuning_state == "stale":
        return "search"
    return "probe"


def _admission_decision_for_state(
    *,
    profile_name: str,
    invocation_source: str,
    ready: bool,
    tuning_state: str,
) -> str:
    if ready:
        return "skip"
    normalized = _normalize_profile_name(profile_name)
    if invocation_source == "repl_startup":
        return "revoke" if tuning_state in {"stale", "mixed"} else "probe"
    if normalized in AUTO_CALIBRATION_PROFILES or normalized == "calibrate":
        if tuning_state in {"stale", "mixed"}:
            return "revoke"
        return "search"
    return "skip"


def resolve_runtime_tuning_bootstrap_policy(policy: str | None = None) -> str:
    raw = (
        str(
            policy
            or os.getenv("ANVIL_RUNTIME_TUNING_BOOTSTRAP_POLICY")
            or DEFAULT_RUNTIME_TUNING_BOOTSTRAP_POLICY
        )
        .strip()
        .lower()
    )
    if raw in {"", "deferred"}:
        return DEFAULT_RUNTIME_TUNING_BOOTSTRAP_POLICY
    if raw in {"never", "explicit", "on_first_run", "background", "always"}:
        return raw
    return DEFAULT_RUNTIME_TUNING_BOOTSTRAP_POLICY


def has_benchmark_evidence(repo_root: Path) -> bool:
    runs_root = repo_root / "audit" / "runs"
    if not runs_root.exists():
        return False
    return any((run_dir / "summary.json").exists() for run_dir in runs_root.iterdir())


def should_bootstrap_runtime_tuning(
    report: dict[str, Any],
    *,
    policy: str | None = None,
    has_prior_benchmark_evidence: bool = False,
) -> bool:
    if bool(report.get("ready")):
        return False
    decision = str(report.get("admission_decision") or "").strip()
    if decision not in TUNING_DECISIONS or decision == "skip":
        return False
    resolved = resolve_runtime_tuning_bootstrap_policy(policy)
    invocation_source = str(report.get("invocation_source") or "").strip().lower()
    if invocation_source in {"campaign", "suite"}:
        return bool(report.get("auto_calibration_allowed", False))
    if resolved == "always":
        return True
    if resolved == "on_first_run":
        return not has_prior_benchmark_evidence
    return False


def mark_runtime_tuning_deferred(
    report: dict[str, Any],
    *,
    reason: str,
    policy: str | None = None,
) -> dict[str, Any]:
    updated = dict(report or {})
    updated["ready"] = False
    updated["status"] = "deferred"
    updated["tuning_state"] = "deferred"
    updated["deferred_reason"] = str(reason or "").strip() or "deferred"
    updated["bootstrap_policy"] = resolve_runtime_tuning_bootstrap_policy(policy)
    updated["admission_decision"] = "skip"
    return updated


def _capability_status_from_tuning(
    *,
    model_name: str,
    model_contract: dict[str, Any],
    thread_config: dict[str, Any],
    continuous_config: dict[str, Any],
    pager_config: dict[str, Any],
    admission: dict[str, Any],
    affinity_policy: str,
    ready: bool,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "digest": str(model_contract.get("digest") or ""),
        "native_isa_baseline": platform.machine(),
        "decode_threads": int(thread_config.get("decode_threads", 0) or 0),
        "batch_threads": int(thread_config.get("batch_threads", 0) or 0),
        "ubatch": int(thread_config.get("ubatch", 0) or 0),
        "generation_mode": "parallel_hybrid" if ready else "ar_baseline",
        "affinity_policy": affinity_policy,
        "scheduler_policy": str(continuous_config.get("scheduler_policy") or ""),
        "max_active_requests": int(
            continuous_config.get("max_active_requests", 0) or 0
        ),
        "batch_wait_timeout_ms": int(
            continuous_config.get("batch_wait_timeout_ms", 0) or 0
        ),
        "max_prefill_rows_per_iteration": int(
            continuous_config.get("max_prefill_rows_per_iteration", 0) or 0
        ),
        "continuous_interleaved_streams": bool(
            continuous_config.get("continuous_interleaved_streams", False)
        ),
        "state_page_rows": int(pager_config.get("state_page_rows", 0) or 0),
        "state_compaction_soft_threshold": float(
            pager_config.get("state_compaction_soft_threshold", 0.0) or 0.0
        ),
        "state_compaction_hard_threshold": float(
            pager_config.get("state_compaction_hard_threshold", 0.0) or 0.0
        ),
        "workload_digest": str(admission.get("workload_digest") or ""),
        "budget_tier": str(admission.get("budget_tier") or ""),
        "admission_decision": str(admission.get("decision") or ""),
        "parallel_decode_allowed": ready,
        "full_qsg_enabled": ready,
        "full_graph_enabled": ready,
        "qsg_processors_native_enabled": ready,
        "batched_prefill_native_enabled": ready,
        "native_backend_abi_match": ready,
        "perf_event_access": True,
    }


def _profile_path(repo_root: Path, profile_name: str) -> Path:
    return (
        repo_root
        / "audit"
        / "profiles"
        / f"native_qsg_{str(profile_name).strip()}.yaml"
    )


def assess_runtime_tuning(
    repo_root: Path,
    *,
    profile_name: str = "silver",
    models: list[str] | None = None,
    launch_runtime: dict[str, Any] | None = None,
    cpu_governor: str = "",
    thp_mode: str = "",
    allow_create_host_contract: bool = True,
    invocation_source: str = "suite",
) -> dict[str, Any]:
    from audit.runner.suite_profiles import load_suite_profile

    spec = load_suite_profile(_profile_path(repo_root, profile_name))
    if models:
        spec = replace(
            spec, models=[str(model) for model in models if str(model).strip()]
        )
    profile_name_normalized = _normalize_profile_name(spec.profile_name)
    workload_digest = _workload_digest(spec)
    runtime = launch_runtime or capture_runtime_provenance(repo_root)
    host = dict(runtime.get("host") or {})
    host_contract, host_contract_file, host_contract_created = ensure_host_contract(
        repo_root,
        contract_id=spec.host_contract_id,
        launch_runtime=runtime,
        cpu_governor=cpu_governor,
        thp_mode=thp_mode,
        allow_create=allow_create_host_contract
        and str(spec.host_contract_id or "").strip()
        in {"", AUTO_HOST_CONTRACT_ID, "current"},
    )
    host_contract_sha256 = sha256_file(host_contract_file)
    tuning_contracts: dict[str, Any] = {}
    missing_models: list[str] = []
    stale_models: list[str] = []
    refresh_models: list[str] = []
    model_contracts = {model: capture_model_provenance(model) for model in spec.models}
    for model_name, model_contract in model_contracts.items():
        tuning_payload, tuning_path = load_tuning_contract(
            repo_root,
            str(host_contract.get("host_fingerprint") or ""),
            model_name,
        )
        if tuning_payload is None:
            missing_models.append(model_name)
            refresh_models.append(model_name)
            tuning_contracts[model_name] = {
                "path": str(tuning_path),
                "status": "missing",
                "issues": ["missing"],
                "readiness_state": "missing",
            }
            continue
        issues = validate_tuning_contract(
            tuning_payload,
            host_contract=host_contract,
            host_contract_sha256=host_contract_sha256,
            model_name=model_name,
            model_contract=model_contract,
            profile_schema_version=spec.schema_version,
            repo_root=repo_root,
            workload_digest=workload_digest,
        )
        if issues:
            stale_models.append(model_name)
            refresh_models.append(model_name)
        tuning_contracts[model_name] = {
            "path": str(tuning_path),
            "status": "valid" if not issues else "stale",
            "issues": issues,
            "readiness_state": contract_readiness_state(issues),
            "thread_config": dict((tuning_payload or {}).get("thread_config") or {}),
            "continuous_config": dict(
                (tuning_payload or {}).get("continuous_config") or {}
            ),
            "pager_config": dict((tuning_payload or {}).get("pager_config") or {}),
            "admission": dict((tuning_payload or {}).get("admission") or {}),
            "benchmark_harness_hash": str(
                (tuning_payload or {}).get("benchmark_harness_hash") or ""
            ),
            "model_digest": str((tuning_payload or {}).get("model_digest") or ""),
        }
    required_visible_threads = int(
        host_contract.get("required_visible_threads", 0)
        or expected_visible_threads(int(host.get("logical_cpus", 0) or 0))
    )
    visible_threads = int(host.get("visible_threads", 0) or 0)
    ready = not refresh_models
    tuning_state, tuning_groups = summarize_tuning_states(tuning_contracts)
    budget_tier = _budget_tier_for_profile(
        profile_name_normalized,
        invocation_source=invocation_source,
        tuning_state=tuning_state,
    )
    admission_decision = _admission_decision_for_state(
        profile_name=profile_name_normalized,
        invocation_source=invocation_source,
        ready=ready,
        tuning_state=tuning_state,
    )
    auto_calibration_allowed = bool(
        invocation_source == "repl_startup"
        or profile_name_normalized in AUTO_CALIBRATION_PROFILES
    )
    primary_model = str(spec.models[0] if spec.models else "")
    primary_contract = dict(model_contracts.get(primary_model) or {})
    primary_tuning = dict(tuning_contracts.get(primary_model) or {})
    primary_thread_config = dict(primary_tuning.get("thread_config") or {})
    primary_continuous_config = dict(primary_tuning.get("continuous_config") or {})
    primary_pager_config = dict(primary_tuning.get("pager_config") or {})
    primary_admission = dict(primary_tuning.get("admission") or {})
    if not primary_admission:
        primary_admission = {
            "decision": admission_decision,
            "budget_tier": budget_tier,
            "invocation_source": invocation_source,
            "workload_digest": workload_digest,
        }
    status = (
        "ready"
        if ready
        else ("calibration_required" if admission_decision != "skip" else "deferred")
    )
    return {
        "schema_version": "native_qsg_suite.runtime_tuning.v2",
        "status": status,
        "profile_name": spec.profile_name,
        "models": list(spec.models),
        "invocation_source": str(invocation_source or "suite"),
        "host_fingerprint": str(
            host_contract.get("host_fingerprint") or host.get("host_fingerprint") or ""
        ),
        "host_contract_id": str(host_contract.get("contract_id") or ""),
        "host_contract_path": str(host_contract_file),
        "host_contract_created": bool(host_contract_created),
        "host_contract_sha256": host_contract_sha256,
        "required_visible_threads": required_visible_threads,
        "visible_threads": visible_threads,
        "benchmark_harness_hash": benchmark_harness_hash(repo_root),
        "workload_digest": workload_digest,
        "budget_tier": budget_tier,
        "admission_decision": admission_decision,
        "auto_calibration_allowed": auto_calibration_allowed,
        "tuning_contracts": tuning_contracts,
        "tuning_state": tuning_state,
        "tuning_state_groups": tuning_groups,
        "missing_models": missing_models,
        "stale_models": stale_models,
        "refresh_models": refresh_models,
        "capability_ledger": build_runtime_capability_ledger(
            _capability_status_from_tuning(
                model_name=primary_model,
                model_contract=primary_contract,
                thread_config=primary_thread_config,
                continuous_config=primary_continuous_config,
                pager_config=primary_pager_config,
                admission=primary_admission,
                affinity_policy=str(spec.affinity_policy or ""),
                ready=ready,
            ),
            host_fingerprint=str(
                host_contract.get("host_fingerprint")
                or host.get("host_fingerprint")
                or ""
            ),
            certification_state="ready" if ready else "calibration_required",
            source="suite_certification",
        ),
        "ready": ready,
    }


def bootstrap_runtime_tuning(
    repo_root: Path,
    *,
    profile_name: str = "silver",
    models: list[str] | None = None,
    calibrate_profile: str = "calibrate",
    auto_run: bool = True,
    stream_output: bool = False,
    invocation_source: str = "campaign",
) -> dict[str, Any]:
    if os.getenv("ANVIL_DISABLE_RUNTIME_TUNING_BOOTSTRAP") == "1":
        report = assess_runtime_tuning(
            repo_root,
            profile_name=profile_name,
            models=models,
            invocation_source=invocation_source,
        )
        report["status"] = "skipped"
        report["reason"] = "ANVIL_DISABLE_RUNTIME_TUNING_BOOTSTRAP=1"
        report["skipped"] = True
        return report

    report = assess_runtime_tuning(
        repo_root,
        profile_name=profile_name,
        models=models,
        invocation_source=invocation_source,
    )
    report["skipped"] = False
    if report.get("ready"):
        return report
    if not auto_run:
        report["status"] = "calibration_required"
        return report

    refresh_models = list(report.get("refresh_models") or report.get("models") or [])
    calibration_mode = str(report.get("budget_tier") or "search").strip() or "search"
    cmd = [
        sys.executable,
        "-m",
        "audit.runner.benchmark_suite",
        "--profile",
        calibrate_profile,
        "--calibration-mode",
        calibration_mode,
        "--calibration-source",
        str(invocation_source or "campaign"),
        "--calibration-target-profile",
        str(profile_name or ""),
    ]
    if refresh_models:
        cmd.extend(["--models", ",".join(refresh_models)])
    env = os.environ.copy()
    env["ANVIL_DISABLE_RUNTIME_TUNING_BOOTSTRAP"] = "1"
    completed = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        text=True,
        check=False,
        capture_output=not stream_output,
    )
    post_report = assess_runtime_tuning(
        repo_root,
        profile_name=profile_name,
        models=models,
        invocation_source=invocation_source,
    )
    post_report.update(
        {
            "command": cmd,
            "bootstrap_return_code": int(completed.returncode),
            "bootstrap_stdout": "" if stream_output else completed.stdout,
            "bootstrap_stderr": "" if stream_output else completed.stderr,
        }
    )
    if completed.returncode == 0 and post_report.get("ready"):
        post_report["status"] = "bootstrapped"
        return post_report
    post_report["status"] = "failed"
    if completed.returncode != 0:
        post_report["reason"] = "calibration_subprocess_failed"
    else:
        refresh_models = list(post_report.get("refresh_models") or [])
        post_report["reason"] = "post_check_not_ready"
        post_report["failed_models"] = refresh_models
    return post_report


def campaign_stage_sequence(profile_name: str) -> list[str]:
    normalized = _normalize_profile_name(profile_name)
    if normalized == "gold":
        return ["bronze", "silver", "gold-fast", "gold"]
    if normalized == "platinum":
        return ["bronze", "silver", "gold-fast", "gold", "platinum"]
    return [normalized] if normalized else ["silver"]


def format_runtime_tuning_summary(report: dict[str, Any]) -> list[str]:
    status = str(report.get("status") or "")
    models = ", ".join(
        str(item) for item in list(report.get("models") or []) if str(item).strip()
    )
    fingerprint = str(report.get("host_fingerprint") or "")
    threads = int(report.get("required_visible_threads", 0) or 0)
    tuning_state = str(report.get("tuning_state") or "").strip()
    decision = str(report.get("admission_decision") or "").strip()
    budget_tier = str(report.get("budget_tier") or "").strip()
    if status == "ready":
        if not bool(report.get("host_contract_created")):
            return []
        return [
            f"Runtime tuning host contract created for `{fingerprint}` ({threads} threads).",
        ]
    if status == "bootstrapped":
        return [
            f"Runtime tuning refreshed for `{models}` on `{fingerprint}` ({threads} threads).",
        ]
    if status == "calibration_required":
        refresh = ", ".join(
            str(item)
            for item in list(report.get("refresh_models") or [])
            if str(item).strip()
        )
        return [
            (
                f"Runtime tuning refresh required for `{refresh or models}` on "
                f"`{fingerprint}` (decision={decision or 'search'}, "
                f"budget={budget_tier or 'search'})."
            ),
        ]
    if status == "deferred":
        refresh = ", ".join(
            str(item)
            for item in list(report.get("refresh_models") or [])
            if str(item).strip()
        )
        reason = str(report.get("deferred_reason") or "deferred").strip()
        return [
            (
                f"Runtime tuning deferred for `{refresh or models}` on `{fingerprint}` "
                f"({threads} threads, state={tuning_state or 'deferred'}, "
                f"decision={decision or 'skip'})."
            ),
            f"reason: {reason}",
        ]
    if status == "failed":
        reason = str(report.get("reason") or "").strip()
        if reason == "post_check_not_ready":
            failed_models = ", ".join(
                str(item)
                for item in list(report.get("failed_models") or [])
                if str(item).strip()
            )
            line = f"contracts_not_ready:{failed_models or models}"
        else:
            detail = (
                str(
                    report.get("bootstrap_stderr")
                    or report.get("bootstrap_stdout")
                    or ""
                )
                .strip()
                .splitlines()
            )
            line = detail[-1] if detail else "calibration_failed"
        return [
            f"Runtime tuning refresh failed for `{models}` on `{fingerprint}`.",
            f"reason: {line}",
        ]
    if status == "skipped":
        return [f"Runtime tuning skipped: {report.get('reason') or 'disabled'}."]
    return []


def ensure_runtime_affinity() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attempted": False,
        "expanded": False,
        "before": [],
        "after": [],
        "target": [],
        "error": "",
    }
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        payload["error"] = "sched_affinity_unsupported"
        return payload
    try:
        before = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception as exc:
        payload["error"] = f"sched_getaffinity_failed:{type(exc).__name__}"
        return payload
    target_set = best_cpu_target()
    target = sorted(int(cpu) for cpu in target_set)
    payload["before"] = before
    payload["target"] = target
    if not target or target_set.issubset(set(before)):
        payload["after"] = before
        return payload
    payload["attempted"] = True
    try:
        os.sched_setaffinity(0, target_set)
        after = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception as exc:
        payload["error"] = f"sched_setaffinity_failed:{type(exc).__name__}:{exc}"
        payload["after"] = before
        return payload
    payload["after"] = after
    payload["expanded"] = len(after) > len(before)
    return payload
