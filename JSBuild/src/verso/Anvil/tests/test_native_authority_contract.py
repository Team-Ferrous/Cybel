from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from saguaro.native.loader import core_library_candidates, repo_root
from saguaro.ops import fused_text_tokenizer


def _load_yaml(path: str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_native_authority_manifest_assigns_uppercase_saguaro_kernel_root() -> None:
    manifest = _load_yaml("standards/native_authority_manifest.yaml")

    layers = {layer["layer_id"]: layer for layer in manifest["layers"]}
    assert layers["saguaro_native_kernels"]["authority"] == "Saguaro"
    assert "Saguaro/src" in layers["saguaro_native_kernels"]["public_interfaces"]
    assert "saguaro/native/ops" in layers["saguaro_native_kernels"]["compatibility_roots"]
    comparative_policy = manifest["comparative_native_rewrite_policy"]
    assert comparative_policy["preferred_native_authority"] == "Saguaro"
    assert comparative_policy["wrapper_policy"] == "thin_python_wrapper_only"


def test_op_lineage_tracks_authoritative_and_compatibility_paths() -> None:
    payload = _load_yaml("standards/op_lineage.yaml")
    ops = {row["op_id"]: row for row in payload["ops"]}

    assert "tensor_stream_pool" in ops
    assert ops["tensor_stream_pool"]["authoritative"]["source"] == "Saguaro/src/ops/common/tensor_stream_pool.cc"
    assert "core/simd/common/tensor_stream_pool.cc" in ops["tensor_stream_pool"]["compatibility"]


def test_scan_exclusion_policy_covers_generated_native_build_noise() -> None:
    payload = _load_yaml("standards/scan_exclusion_policy.yaml")
    patterns = set(payload["exclude_globs"])

    assert "_legacy_saguaro_to_remove/**" in patterns
    assert "saguaro/native/build.stale-*/**" in patterns
    assert "Saguaro/build/**" in patterns
    assert "core/native/build/**" in patterns


def test_shim_expiry_ledger_tracks_root_launcher_and_simd_mirror_tree() -> None:
    payload = _load_yaml("standards/shim_expiry.yaml")
    shims = {row["shim_id"]: row for row in payload["shims"]}

    assert shims["main_py_launcher"]["replacement"] == "anvil.py"
    assert shims["core_simd_mirror_tree"]["replacement"] == "Saguaro/src"


def test_native_loader_repo_root_points_at_anvil_checkout() -> None:
    assert repo_root() == Path(__file__).resolve().parents[1]

    candidates = core_library_candidates()
    assert candidates
    assert candidates[0] == repo_root() / "Saguaro" / "build" / "_saguaro_core.so"


def test_fused_text_tokenizer_accepts_canonical_trie_aliases() -> None:
    marker = object()

    assert (
        fused_text_tokenizer._resolve_trie_create_op(
            SimpleNamespace(saguaro_trie_create=marker)
        )
        is marker
    )
    assert (
        fused_text_tokenizer._resolve_trie_create_op(
            SimpleNamespace(SAGUAROTrieCreate=marker)
        )
        is marker
    )
