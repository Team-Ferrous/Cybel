from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.model.model_contract as contract_mod
from config.settings import PRODUCTION_MODEL_ALLOWLIST, PRODUCTION_MODEL_POLICY
from core.model.model_contract import canonicalize_model_name, resolve_model_contract


def test_canonicalize_model_name_rejects_unsupported_models() -> None:
    with pytest.raises(RuntimeError, match="Unsupported production model"):
        canonicalize_model_name("deepseek-coder:33b")


def test_resolve_model_contract_uses_local_manifest(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    digest = "sha256:deadbeef"
    manifest = (
        model_root / "manifests" / "registry.ollama.ai" / "library" / "qwen3.5" / "9b"
    )
    blob = model_root / "blobs" / digest.replace(":", "-")
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_text("stub")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": digest,
                        "size": blob.stat().st_size,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(contract_mod, "candidate_ollama_model_roots", lambda: [model_root])
    monkeypatch.setitem(
        PRODUCTION_MODEL_ALLOWLIST["qwen3.5:9b"],
        "digest",
        digest,
    )
    monkeypatch.setitem(
        PRODUCTION_MODEL_POLICY["qwen3.5:9b"],
        "expected_manifest_digest",
        contract_mod._compute_sha256(manifest),
    )
    monkeypatch.setitem(
        PRODUCTION_MODEL_POLICY["qwen3.5:9b"],
        "expected_model_digest",
        digest,
    )

    contract = resolve_model_contract("qwen3.5:9b")

    assert contract.canonical_name == "qwen3.5:9b"
    assert contract.blob_path == blob
    assert contract.expected_digest == digest
    assert contract.expected_model_digest == digest
    assert contract.expected_manifest_digest == contract_mod._compute_sha256(manifest)
    assert contract.template_name == "chatml"
    assert contract.strict_native_supported is True


def test_resolve_model_contract_requires_local_manifest(monkeypatch) -> None:
    monkeypatch.setattr(contract_mod, "candidate_ollama_model_roots", lambda: [])
    with pytest.raises(RuntimeError, match="Could not resolve local Ollama manifest"):
        resolve_model_contract("granite4:tiny-h")
