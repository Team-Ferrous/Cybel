from __future__ import annotations

import os
from typing import Any


class BridgeSynthesizer:
    """Synthesize probable cross-language bridge edges from FFI findings."""

    def synthesize(self, ffi_patterns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        providers = [p for p in ffi_patterns if str(p.get("role") or "") == "provider"]
        consumers = [p for p in ffi_patterns if str(p.get("role") or "") == "consumer"]
        if not providers or not consumers:
            return []

        bridges: dict[str, dict[str, Any]] = {}
        for consumer in sorted(consumers, key=self._pattern_sort_key):
            for provider in sorted(providers, key=self._pattern_sort_key):
                if consumer.get("id") == provider.get("id"):
                    continue
                confidence, reason = self._bridge_confidence(consumer, provider)
                if confidence <= 0.0:
                    continue

                src = str(consumer.get("id") or "")
                dst = str(provider.get("id") or "")
                line = int(consumer.get("line") or 0)
                edge_id = f"{src}->{dst}::ffi_bridge::{line}"
                bridges[edge_id] = {
                    "id": edge_id,
                    "from": src,
                    "to": dst,
                    "relation": "ffi_bridge",
                    "file": consumer.get("file"),
                    "line": line,
                    "confidence": round(confidence, 3),
                    "reason": reason,
                    "boundary_pair": [
                        str(consumer.get("boundary_type") or ""),
                        str(provider.get("boundary_type") or ""),
                    ],
                    "shared_object": str(
                        consumer.get("shared_object") or provider.get("shared_object") or ""
                    ),
                    "source": "bridge_synthesizer",
                }

        return [bridges[edge_id] for edge_id in sorted(bridges)]

    def _bridge_confidence(
        self, consumer: dict[str, Any], provider: dict[str, Any]
    ) -> tuple[float, str]:
        consumer_hint = str(consumer.get("library_hint") or "").strip().lower()
        provider_hint = str(provider.get("library_hint") or "").strip().lower()
        consumer_conf = float(consumer.get("confidence") or 0.0)
        provider_conf = float(provider.get("confidence") or 0.0)
        base = (consumer_conf + provider_conf) / 2.0

        consumer_so = str(consumer.get("shared_object") or "").strip().lower()
        provider_so = str(provider.get("shared_object") or "").strip().lower()
        if consumer_so and provider_so and consumer_so == provider_so:
            return min(0.99, base * 0.8 + 0.22), "shared_object_match"

        consumer_boundary = str(consumer.get("boundary_type") or "").strip().lower()
        provider_boundary = str(provider.get("boundary_type") or "").strip().lower()
        if (
            consumer_boundary.startswith(("ctypes.", "cffi.", "cgo.", "jni.", "napi."))
            and provider_boundary
            in {"pybind11.module", "python_c_api", "extern_c_export", "pyo3.binding"}
        ):
            return min(0.96, base * 0.72 + 0.2), "typed_boundary_compatible"

        if consumer_hint and provider_hint and consumer_hint == provider_hint:
            return min(0.99, base * 0.75 + 0.25), "library_hint_match"

        consumer_tokens = self._file_tokens(str(consumer.get("file") or ""))
        provider_tokens = self._file_tokens(str(provider.get("file") or ""))
        overlap = consumer_tokens.intersection(provider_tokens)
        if overlap:
            return min(0.95, base * 0.65 + 0.18), "file_token_overlap"

        same_dir = os.path.dirname(str(consumer.get("file") or "")) == os.path.dirname(
            str(provider.get("file") or "")
        )
        if same_dir:
            return min(0.85, base * 0.55 + 0.12), "same_directory"

        return 0.0, ""

    def _file_tokens(self, rel_file: str) -> set[str]:
        stem = os.path.splitext(os.path.basename(rel_file))[0]
        tokens = {part.strip().lower() for part in stem.replace("-", "_").split("_")}
        return {token for token in tokens if token and len(token) > 1}

    @staticmethod
    def _pattern_sort_key(pattern: dict[str, Any]) -> tuple[str, int, str]:
        return (
            str(pattern.get("file") or ""),
            int(pattern.get("line") or 0),
            str(pattern.get("id") or ""),
        )
