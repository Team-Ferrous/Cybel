"""Proof-carrying capsule helpers for CPU/math hotspot promotion."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any


def build_hotspot_capsule(
    hotspot: dict[str, Any],
    *,
    arch: str,
    scan_path: str,
    math_cache_path: str = "",
    generated_at_epoch: float | None = None,
) -> dict[str, Any]:
    """Build a proof-carrying capsule for a hotspot candidate."""

    generated_at_epoch = float(generated_at_epoch or time.time())
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated_at_epoch))
    source_file = str(hotspot.get("file") or "")
    selected_schedule = dict((hotspot.get("schedule_twin") or {}).get("selected") or {})
    current_blockers = list(selected_schedule.get("blockers") or [])
    benchmark_priority = float(hotspot.get("benchmark_priority", 0.0) or 0.0)
    proof_basis = (
        str(hotspot.get("id") or ""),
        source_file,
        str(hotspot.get("line_start") or 0),
        arch,
    )
    capsule_id = f"hotspot::{_stable_fragment('::'.join(proof_basis))}"
    benchmark_status = "candidate" if benchmark_priority > 0 else "deferred"
    capsule = {
        "capsule_id": capsule_id,
        "schema_version": "1",
        "generated_at": generated_at,
        "generated_at_epoch": generated_at_epoch,
        "source": {
            "id": str(hotspot.get("id") or ""),
            "file": source_file,
            "line_start": int(hotspot.get("line_start", 0) or 0),
            "expression": str(hotspot.get("expression") or ""),
            "language": str(hotspot.get("language") or ""),
        },
        "provenance": {
            "scan_path": str(scan_path or "."),
            "arch": str(arch or ""),
            "analysis_engine": str(hotspot.get("analysis_engine") or "python_fallback"),
            "native_runtime_version": str(hotspot.get("native_runtime_version") or ""),
            "math_cache_path": str(math_cache_path or ""),
        },
        "runtime_witness": {
            "cache": dict(hotspot.get("cache") or {}),
            "vectorization": dict(hotspot.get("vectorization") or {}),
            "prefetch": dict(hotspot.get("prefetch") or {}),
            "roofline": dict(hotspot.get("roofline") or {}),
            "register_pressure": dict(hotspot.get("register_pressure") or {}),
            "schedule_recipe": selected_schedule,
            "reduction_hints": list(
                hotspot.get("complexity_reduction_hints")
                or (hotspot.get("optimization_packet") or {}).get("reduction_hints")
                or []
            ),
        },
        "benchmark_result": {
            "priority": benchmark_priority,
            "status": benchmark_status,
            "measured_delta": None,
        },
        "rollback_rule": {
            "rule": "Demote this hotspot if schedule blockers reappear or verification evidence regresses.",
            "trigger_conditions": [
                "schedule_blockers_present",
                "verification_regression",
                "runtime_witness_drift",
            ],
            "current_blockers": current_blockers,
        },
        "artifact_paths": {},
    }
    missing_fields = [
        field
        for field, value in (
            ("source.file", source_file),
            ("source.expression", capsule["source"]["expression"]),
            ("provenance.arch", capsule["provenance"]["arch"]),
            ("benchmark_result.status", benchmark_status),
            ("rollback_rule.rule", capsule["rollback_rule"]["rule"]),
        )
        if _is_missing_value(value)
    ]
    contradictions = []
    if current_blockers:
        contradictions.append("schedule_blockers_present")
    capsule["completeness"] = {
        "complete": not missing_fields,
        "missing_fields": missing_fields,
        "contradictions": contradictions,
    }
    return capsule


def persist_hotspot_capsules(
    repo_path: str,
    hotspots: list[dict[str, Any]],
    *,
    arch: str,
    scan_path: str,
    math_cache_path: str = "",
    max_capsules: int = 5,
) -> dict[str, Any]:
    """Persist hotspot capsules and a manifest under .anvil/validation."""

    repo_root = os.path.abspath(repo_path)
    generated_at_epoch = time.time()
    validation_dir = os.path.join(repo_root, ".anvil", "validation")
    capsule_dir = os.path.join(validation_dir, "hotspot_capsules")
    manifest_path = os.path.join(validation_dir, "hotspot_capsule_manifest.json")
    traceability_path = os.path.join(
        repo_root,
        "standards",
        "traceability",
        "TRACEABILITY.jsonl",
    )
    os.makedirs(capsule_dir, exist_ok=True)
    os.makedirs(os.path.dirname(traceability_path), exist_ok=True)
    persisted: list[dict[str, Any]] = []
    for hotspot in list(hotspots or [])[: max(1, int(max_capsules or 5))]:
        capsule = build_hotspot_capsule(
            hotspot,
            arch=arch,
            scan_path=scan_path,
            math_cache_path=math_cache_path,
            generated_at_epoch=generated_at_epoch,
        )
        capsule_path = os.path.join(capsule_dir, f"{capsule['capsule_id']}.json")
        capsule["artifact_paths"] = {
            "capsule_path": os.path.relpath(capsule_path, repo_root).replace("\\", "/"),
            "traceability_path": os.path.relpath(traceability_path, repo_root).replace(
                "\\", "/"
            ),
            "manifest_path": os.path.relpath(manifest_path, repo_root).replace("\\", "/"),
        }
        with open(capsule_path, "w", encoding="utf-8") as handle:
            json.dump(capsule, handle, indent=2, sort_keys=True)
        _append_traceability_record(traceability_path, _traceability_record(capsule))
        persisted.append(capsule)

    manifest = {
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated_at_epoch)),
        "generated_at_epoch": generated_at_epoch,
        "scan_path": str(scan_path or "."),
        "arch": str(arch or ""),
        "capsule_count": len(persisted),
        "complete_capsule_count": sum(
            1 for capsule in persisted if capsule.get("completeness", {}).get("complete")
        ),
        "contradictory_capsule_count": sum(
            1
            for capsule in persisted
            if capsule.get("completeness", {}).get("contradictions")
        ),
        "capsules": persisted,
        "artifact_path": os.path.relpath(manifest_path, repo_root).replace("\\", "/"),
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def load_hotspot_capsule_manifest(repo_path: str) -> dict[str, Any]:
    """Load the persisted hotspot capsule manifest if present."""

    manifest_path = os.path.join(
        os.path.abspath(repo_path),
        ".anvil",
        "validation",
        "hotspot_capsule_manifest.json",
    )
    if not os.path.exists(manifest_path):
        return {
            "status": "missing",
            "capsule_count": 0,
            "complete_capsule_count": 0,
            "contradictory_capsule_count": 0,
            "capsules": [],
            "artifact_path": os.path.relpath(manifest_path, os.path.abspath(repo_path)).replace(
                "\\", "/"
            ),
        }
    with open(manifest_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {"status": "invalid", "capsules": []}


def _traceability_record(capsule: dict[str, Any]) -> dict[str, Any]:
    source = dict(capsule.get("source") or {})
    artifact_paths = dict(capsule.get("artifact_paths") or {})
    return {
        "aal": "AAL-2",
        "owner": "saguaro-hotspot-capsules",
        "design_ref": "standards/AES.md",
        "requirement_id": str(capsule.get("capsule_id") or ""),
        "trace_id": f"trace::{capsule.get('capsule_id')}",
        "timestamp": str(capsule.get("generated_at") or ""),
        "code_refs": [str(source.get("file") or "")],
        "graph_refs": [str(source.get("file") or "")],
        "test_refs": [],
        "verification_refs": [
            str(artifact_paths.get("capsule_path") or ""),
            str(artifact_paths.get("manifest_path") or ""),
        ],
        "statement": str(source.get("expression") or ""),
        "source_path": str(source.get("file") or ""),
        "line_start": int(source.get("line_start", 0) or 0),
    }


def _append_traceability_record(path: str, record: dict[str, Any]) -> None:
    trace_id = str(record.get("trace_id") or "")
    if trace_id and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if trace_id and f'"trace_id": "{trace_id}"' in line:
                    return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _stable_fragment(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False
