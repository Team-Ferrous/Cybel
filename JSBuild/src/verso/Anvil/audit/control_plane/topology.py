from __future__ import annotations

from typing import Any

from audit.control_plane.mission import MissionContext
from audit.control_plane.reducers import now_iso, stable_hash

TOPOLOGY_PASSPORT_SCHEMA_VERSION = "native_qsg_suite.topology_passport.v1"


def topology_fingerprint_basis(
    *,
    spec: Any,
    preflight_payload: dict[str, Any],
) -> dict[str, Any]:
    runtime = dict(preflight_payload.get("runtime") or {})
    host = dict(preflight_payload.get("host") or runtime.get("host") or {})
    isolation = {
        "launch_affinity": list(preflight_payload.get("launch_affinity") or []),
        "runtime_affinity": list(preflight_payload.get("post_adjustment_affinity") or []),
        "cpu_governor": str(preflight_payload.get("cpu_governor") or ""),
        "thp_mode": str(preflight_payload.get("thp_mode") or ""),
    }
    return {
        "host_fingerprint": str(host.get("host_fingerprint") or ""),
        "cpu_model": str(host.get("cpu_model") or ""),
        "logical_cpus": int(host.get("logical_cpus", 0) or 0),
        "visible_threads": int(host.get("visible_threads", 0) or 0),
        "launch_affinity": isolation["launch_affinity"],
        "runtime_affinity": isolation["runtime_affinity"],
        "cpu_governor": isolation["cpu_governor"],
        "thp_mode": isolation["thp_mode"],
        "profile_name": spec.profile_name,
        "models": list(spec.models),
        "schema_version": spec.schema_version,
    }


def topology_hash_from_preflight(
    *,
    spec: Any,
    preflight_payload: dict[str, Any],
) -> str:
    return stable_hash(topology_fingerprint_basis(spec=spec, preflight_payload=preflight_payload))


def build_topology_passport(
    mission: MissionContext,
    *,
    saguaro_verify: dict[str, Any],
) -> dict[str, Any]:
    preflight = dict(mission.preflight_payload or {})
    runtime = dict(preflight.get("runtime") or mission.launch_runtime or {})
    host = dict(preflight.get("host") or runtime.get("host") or {})
    memory = dict(preflight.get("memory") or {})
    saguaro = dict(preflight.get("saguaro") or {})
    isolation = {
        "launch_affinity": list(preflight.get("launch_affinity") or []),
        "runtime_affinity": list(preflight.get("post_adjustment_affinity") or []),
        "repair_allowed": bool(preflight.get("repair_allowed", False)),
        "repair_attempted": bool(preflight.get("repair_attempted", False)),
        "repair_required": bool(preflight.get("repair_required", False)),
        "cpu_governor": str(preflight.get("cpu_governor") or ""),
        "thp_mode": str(preflight.get("thp_mode") or ""),
        "perf_event_paranoid": str(preflight.get("perf_event_paranoid") or ""),
    }
    topology_basis = topology_fingerprint_basis(
        spec=mission.spec,
        preflight_payload=mission.preflight_payload,
    )
    topology_hash = stable_hash(topology_basis)
    cohort_key = stable_hash(
        {
            "host_fingerprint": topology_basis["host_fingerprint"],
            "topology_hash": topology_hash,
            "models": list(mission.spec.models),
        }
    )[:16]
    return {
        "schema_version": TOPOLOGY_PASSPORT_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "profile_name": mission.spec.profile_name,
        "models": list(mission.spec.models),
        "host_fingerprint": str(host.get("host_fingerprint") or ""),
        "topology_hash": topology_hash,
        "cohort_key": cohort_key,
        "host": {
            "hostname": str(host.get("hostname") or ""),
            "machine": str(host.get("machine") or ""),
            "platform": str(host.get("platform") or ""),
            "cpu_model": str(host.get("cpu_model") or ""),
            "logical_cpus": int(host.get("logical_cpus", 0) or 0),
            "visible_threads": int(host.get("visible_threads", 0) or 0),
        },
        "isolation": isolation,
        "memory": {
            "meminfo_artifact": str(memory.get("meminfo_artifact") or ""),
            "mem_available_kb": int(
                (memory.get("meminfo") or {}).get("MemAvailable", 0) or 0
            ),
        },
        "saguaro": {
            "health_ok": bool(saguaro.get("ok", False)),
            "health_artifact": str((saguaro.get("artifact") or {}).get("path") or ""),
            "verify_status": str(saguaro_verify.get("status") or ""),
            "verify_available": bool(saguaro_verify.get("available", False)),
        },
        "comparator_hints": {
            "same_topology_key": cohort_key,
            "thread_shape": {
                "canonical_decode_threads": list(mission.spec.canonical_decode_threads),
                "canonical_batch_threads": list(mission.spec.canonical_batch_threads),
            },
        },
    }
