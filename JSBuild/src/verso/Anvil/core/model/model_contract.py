"""Fail-closed production model contract for native QSG."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.settings import PRODUCTION_MODEL_ALLOWLIST, PRODUCTION_MODEL_POLICY


@dataclass(frozen=True)
class ModelContract:
    canonical_name: str
    template_name: str
    strict_native_supported: bool
    manifest_path: Path
    blob_path: Path
    manifest_sha256: str | None
    expected_manifest_digest: str | None
    manifest_digest: str
    expected_model_digest: str
    expected_digest: str
    blob_size: int
    digest_validated: bool
    local_sha256: str | None
    quant_variant: str | None


def candidate_ollama_model_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.getenv("OLLAMA_MODELS")
    if env_root:
        roots.append(Path(env_root).expanduser())

    roots.extend(
        [
            Path("/usr/share/ollama/.ollama/models"),
            Path.home() / ".ollama" / "models",
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def canonicalize_model_name(model_name: str) -> str:
    ref = str(model_name or "").strip()
    if not ref:
        raise RuntimeError("Production native QSG requires a non-empty model name.")
    if (
        ref.endswith(".gguf")
        or ref.startswith(os.path.sep)
        or ref.startswith("./")
        or ref.startswith("../")
    ):
        raise RuntimeError(
            "Production native QSG resolves models from local Ollama manifests only. "
            f"Direct GGUF path '{model_name}' is not allowed."
        )

    host = "registry.ollama.ai"
    namespace = "library"
    tag = "latest"
    if ":" in ref:
        ref, maybe_tag = ref.rsplit(":", 1)
        if maybe_tag:
            tag = maybe_tag

    segments = [seg for seg in ref.split("/") if seg]
    if len(segments) > 1 and "." in segments[0]:
        host = segments[0]
        segments = segments[1:]

    if host != "registry.ollama.ai":
        raise RuntimeError(
            "Production native QSG only supports local Ollama registry manifests "
            f"from registry.ollama.ai; got '{host}'."
        )
    if len(segments) >= 2:
        namespace = segments[-2]
        name = segments[-1]
    elif len(segments) == 1:
        name = segments[0]
    else:
        raise RuntimeError(f"Could not parse production model name '{model_name}'.")

    if namespace != "library":
        raise RuntimeError(
            "Production native QSG only supports library namespace manifests; "
            f"got '{namespace}' for '{model_name}'."
        )
    canonical = f"{name}:{tag}"
    if canonical not in PRODUCTION_MODEL_ALLOWLIST:
        allowed = ", ".join(sorted(PRODUCTION_MODEL_ALLOWLIST))
        raise RuntimeError(
            f"Unsupported production model '{model_name}'. Supported models: {allowed}."
        )
    return canonical


def _resolve_manifest_path(canonical_name: str) -> Path:
    name, tag = canonical_name.rsplit(":", 1)
    manifest_rel = (
        Path("manifests") / "registry.ollama.ai" / "library" / name / tag
    )
    for root in candidate_ollama_model_roots():
        manifest_path = root / manifest_rel
        if manifest_path.exists():
            return manifest_path
    raise RuntimeError(
        f"Could not resolve local Ollama manifest for '{canonical_name}'. "
        "Ensure the model is already downloaded locally."
    )


def _resolve_model_layer(manifest_path: Path) -> tuple[str, int]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = data.get("layers") or []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        media = str(layer.get("mediaType") or "").lower()
        if "model" not in media and "gguf" not in media:
            continue
        digest = str(layer.get("digest") or "").strip()
        size = int(layer.get("size") or 0)
        if digest:
            return digest, size
    raise RuntimeError(
        f"Manifest '{manifest_path}' does not include a model GGUF layer."
    )


def _blob_path_from_digest(manifest_path: Path, digest: str) -> Path:
    root = manifest_path.parents[4]
    return root / "blobs" / digest.replace(":", "-")


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def resolve_model_contract(model_name: str) -> ModelContract:
    canonical_name = canonicalize_model_name(model_name)
    manifest_path = _resolve_manifest_path(canonical_name)
    manifest_sha256 = _compute_sha256(manifest_path)
    manifest_digest, blob_size = _resolve_model_layer(manifest_path)
    blob_path = _blob_path_from_digest(manifest_path, manifest_digest)
    if not blob_path.exists():
        raise RuntimeError(
            f"Local model blob for '{canonical_name}' is missing: {blob_path}"
        )
    if blob_size > 0:
        actual_size = int(blob_path.stat().st_size)
        if actual_size != blob_size:
            raise RuntimeError(
                f"Local model blob size mismatch for '{canonical_name}': "
                f"expected {blob_size}, found {actual_size}."
            )

    spec = dict(PRODUCTION_MODEL_ALLOWLIST[canonical_name])
    policy_raw = PRODUCTION_MODEL_POLICY.get(canonical_name)
    if not isinstance(policy_raw, dict) or not policy_raw:
        raise RuntimeError(
            f"Production contract for '{canonical_name}' is missing pinned policy metadata."
        )
    policy = dict(policy_raw)
    expected_manifest_digest = str(policy.get("expected_manifest_digest") or "").strip()
    expected_model_digest = str(policy.get("expected_model_digest") or "").strip()
    allowlist_digest = str(spec.get("digest") or "").strip()
    quant_variant = str(policy.get("quant_variant") or "").strip() or None
    if not expected_manifest_digest:
        raise RuntimeError(
            f"Production contract for '{canonical_name}' is missing an expected manifest digest."
        )
    template_name = str(spec.get("template") or "").strip().lower()
    if not expected_model_digest:
        raise RuntimeError(
            f"Production contract for '{canonical_name}' is missing an expected digest."
        )
    if allowlist_digest and allowlist_digest != expected_model_digest:
        raise RuntimeError(
            f"Production contract for '{canonical_name}' has inconsistent digest pins: "
            f"allowlist={allowlist_digest}, policy={expected_model_digest}."
        )
    if not template_name:
        raise RuntimeError(
            f"Production contract for '{canonical_name}' is missing a strict-native template."
        )
    if expected_manifest_digest and manifest_sha256 != expected_manifest_digest:
        raise RuntimeError(
            f"Unexpected manifest SHA-256 for '{canonical_name}': "
            f"expected {expected_manifest_digest}, found {manifest_sha256}."
        )
    if manifest_digest != expected_model_digest:
        raise RuntimeError(
            f"Unexpected manifest digest for '{canonical_name}': "
            f"expected {expected_model_digest}, found {manifest_digest}."
        )

    validate_sha = str(os.getenv("ANVIL_VALIDATE_MODEL_SHA256", "0")).strip() == "1"
    local_sha256 = _compute_sha256(blob_path) if validate_sha else None
    if local_sha256 is not None and local_sha256 != expected_model_digest:
        raise RuntimeError(
            f"Local blob SHA-256 mismatch for '{canonical_name}': "
            f"expected {expected_model_digest}, found {local_sha256}."
        )

    return ModelContract(
        canonical_name=canonical_name,
        template_name=template_name,
        strict_native_supported=True,
        manifest_path=manifest_path,
        blob_path=blob_path,
        manifest_sha256=manifest_sha256,
        expected_manifest_digest=expected_manifest_digest or None,
        manifest_digest=manifest_digest,
        expected_model_digest=expected_model_digest,
        expected_digest=expected_model_digest,
        blob_size=blob_size,
        digest_validated=True,
        local_sha256=local_sha256,
        quant_variant=quant_variant,
    )


def model_contract_snapshot(contract: ModelContract) -> dict[str, Any]:
    return {
        "model": contract.canonical_name,
        "template_name": contract.template_name,
        "manifest_path": str(contract.manifest_path),
        "blob_path": str(contract.blob_path),
        "manifest_sha256": contract.manifest_sha256,
        "expected_manifest_digest": contract.expected_manifest_digest,
        "digest": contract.expected_digest,
        "expected_model_digest": contract.expected_model_digest,
        "blob_size": int(contract.blob_size),
        "digest_validated": bool(contract.digest_validated),
        "sha256_validated": bool(contract.local_sha256 is not None),
        "quant_variant": contract.quant_variant,
        "strict_native_supported": bool(contract.strict_native_supported),
    }
