"""Unified native capability posture reporting."""

from __future__ import annotations

import time
from typing import Any, Callable


class NativeCapabilityRegistry:
    """Probe native surfaces and return one health manifest."""

    def __init__(self, repo_root: str = ".") -> None:
        self.repo_root = repo_root

    def build_manifest(self) -> dict[str, Any]:
        capabilities = [
            self._probe("native_ops", self._native_ops_manifest),
            self._probe("fast_attention", self._fast_attention_manifest),
            self._probe("native_tokenizer", self._tokenizer_manifest),
            self._probe("native_indexer", self._native_indexer_manifest),
        ]
        available = sum(1 for item in capabilities if item["status"] == "available")
        degraded = sum(1 for item in capabilities if item["status"] != "available")
        return {
            "schema_version": "native_capability_manifest.v1",
            "generated_at": time.time(),
            "repo_root": self.repo_root,
            "summary": {
                "capability_count": len(capabilities),
                "available_count": available,
                "degraded_count": degraded,
            },
            "capabilities": capabilities,
        }

    def _probe(
        self,
        capability: str,
        loader: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            payload = loader()
            return {
                "capability": capability,
                "status": "available",
                "fallback_reason": "",
                **payload,
            }
        except Exception as exc:
            return {
                "capability": capability,
                "status": "degraded",
                "fallback_reason": str(exc),
            }

    @staticmethod
    def _native_ops_manifest() -> dict[str, Any]:
        from core.native.native_ops import get_native_library_info

        return {
            "provider": "core.native.native_ops",
            "details": get_native_library_info(),
        }

    @staticmethod
    def _fast_attention_manifest() -> dict[str, Any]:
        from core.native.fast_attention_wrapper import FastAttention

        attention = FastAttention()
        return {
            "provider": "core.native.fast_attention_wrapper.FastAttention",
            "details": {
                "available": attention.available,
                "mqa_available": attention.mqa_available,
            },
        }

    @staticmethod
    def _tokenizer_manifest() -> dict[str, Any]:
        from core.native import native_tokenizer

        native_tokenizer._lib()
        return {
            "provider": "core.native.native_tokenizer",
            "details": {"symbols_ready": True},
        }

    @staticmethod
    def _native_indexer_manifest() -> dict[str, Any]:
        from saguaro.indexing.native_indexer_bindings import NativeIndexer

        indexer = NativeIndexer()
        return {
            "provider": "saguaro.indexing.native_indexer_bindings.NativeIndexer",
            "details": {
                "library_path": getattr(indexer, "_lib_path", ""),
                "manifest_path": str((getattr(indexer, "_manifest", {}) or {}).get("_path", "")),
                "version": indexer.version(),
            },
        }
