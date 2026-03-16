"""Committed index manifest helpers for Saguaro."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any

from saguaro.errors import SaguaroStateCorruptionError, SaguaroStateMismatchError

INDEX_MANIFEST_FILENAME = "index_manifest.json"
INDEX_MANIFEST_SCHEMA_VERSION = 1
INDEX_ARTIFACTS = {
    "vectors.bin": os.path.join("vectors", "vectors.bin"),
    "vectors/norms.bin": os.path.join("vectors", "norms.bin"),
    "vectors/metadata.json": os.path.join("vectors", "metadata.json"),
    "vectors/index_meta.json": os.path.join("vectors", "index_meta.json"),
    "tracking.json": "tracking.json",
    "index_stats.json": "index_stats.json",
    "index_schema.json": "index_schema.json",
    "graph/graph.json": os.path.join("graph", "graph.json"),
}


def manifest_path(root: str) -> str:
    return os.path.join(root, INDEX_MANIFEST_FILENAME)


def new_generation_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:12]}"


def artifact_paths(root: str) -> dict[str, str]:
    return {
        name: os.path.join(root, rel_path)
        for name, rel_path in INDEX_ARTIFACTS.items()
    }


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_artifact(path: str, rel_path: str) -> dict[str, Any]:
    stat = os.stat(path)
    return {
        "path": rel_path.replace("\\", "/"),
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256": file_sha256(path),
    }


def load_manifest(root: str) -> dict[str, Any]:
    path = manifest_path(root)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except Exception as exc:
        raise SaguaroStateCorruptionError(
            f"Unreadable index manifest at {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SaguaroStateCorruptionError(f"Invalid index manifest at {path}")
    return payload


def validate_manifest(root: str, manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest:
        return {"status": "missing", "mismatches": [], "artifacts": {}}

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SaguaroStateCorruptionError("Index manifest is missing artifact records")

    mismatches: list[str] = []
    live_artifacts: dict[str, Any] = {}
    for name, rel_path in INDEX_ARTIFACTS.items():
        expected = artifacts.get(name)
        path = os.path.join(root, rel_path)
        if not os.path.exists(path):
            mismatches.append(f"missing:{name}")
            continue
        current = {
            "path": rel_path.replace("\\", "/"),
            "size": int(os.path.getsize(path)),
            "mtime": float(os.path.getmtime(path)),
        }
        live_artifacts[name] = current
        if not isinstance(expected, dict):
            mismatches.append(f"manifest_missing:{name}")
            continue
        if int(expected.get("size", -1)) != current["size"]:
            mismatches.append(f"size:{name}")
        elif abs(float(expected.get("mtime", 0.0)) - current["mtime"]) > 1e-6:
            mismatches.append(f"mtime:{name}")

    status = "ready" if not mismatches else "mismatch"
    return {
        "status": status,
        "generation_id": manifest.get("generation_id"),
        "mismatches": mismatches,
        "artifacts": live_artifacts,
    }


def require_manifest_ready(root: str, manifest: dict[str, Any]) -> dict[str, Any]:
    validation = validate_manifest(root, manifest)
    if str(manifest.get("status") or "") != "ready":
        raise SaguaroStateMismatchError(
            f"Committed index manifest is not ready: {manifest.get('status')}"
        )
    if validation["status"] == "mismatch":
        raise SaguaroStateMismatchError(
            "Committed index manifest does not match live artifacts: "
            + ", ".join(validation["mismatches"])
        )
    return validation
