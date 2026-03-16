"""Projection pipelines for ALMF memory objects."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

import numpy as np

from core.memory.fabric.models import MemoryObject
from core.memory.fabric.store import MemoryFabricStore
from domains.memory_management.hd.compression import HDCompressor


class MemoryProjector:
    """Build deterministic retrieval projections for memory objects."""

    def __init__(
        self,
        *,
        dense_dim: int = 64,
        token_dim: int = 32,
        hd_dimension: int = 10000,
    ) -> None:
        self.dense_dim = int(max(8, dense_dim))
        self.token_dim = int(max(8, token_dim))
        self.hd = HDCompressor(dimension=hd_dimension)

    @staticmethod
    def text_for(memory: MemoryObject | dict[str, object]) -> str:
        summary = str(getattr(memory, "summary_text", "") or memory.get("summary_text") or "")
        payload = getattr(memory, "payload_json", None)
        if payload is None and isinstance(memory, dict):
            payload = memory.get("payload_json") or {}
        return (summary + "\n" + str(payload or "")).strip()

    def dense_embedding(self, text: str) -> np.ndarray:
        tokens = _tokenize(text)
        vector = np.zeros((self.dense_dim,), dtype=np.float32)
        if not tokens:
            return vector
        for token in tokens:
            index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dense_dim
            sign = 1.0 if index % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector

    def multivector(self, text: str, *, max_tokens: int = 32) -> np.ndarray:
        tokens = _tokenize(text)[: max(1, int(max_tokens))]
        if not tokens:
            return np.zeros((0, self.token_dim), dtype=np.float32)
        vectors = np.zeros((len(tokens), self.token_dim), dtype=np.float32)
        for row_idx, token in enumerate(tokens):
            for idx, value in enumerate(_token_bytes(token)):
                vectors[row_idx, idx % self.token_dim] += (float(value) / 255.0) - 0.5
            norm = np.linalg.norm(vectors[row_idx])
            if norm > 0:
                vectors[row_idx] /= norm
        return vectors

    def hd_bundle(self, text: str) -> np.ndarray:
        return np.asarray(self.hd.encode_context(text), dtype=np.int8)

    def latent_tensor(self, text: str, *, hidden_dim: int = 16) -> np.ndarray:
        dense = self.dense_embedding(text)
        if hidden_dim <= dense.shape[0]:
            return dense[:hidden_dim].reshape(1, hidden_dim)
        tensor = np.zeros((1, hidden_dim), dtype=np.float32)
        tensor[0, : dense.shape[0]] = dense
        return tensor

    def project_memory(
        self,
        store: MemoryFabricStore,
        memory: MemoryObject | dict[str, object],
        *,
        include_multivector: bool = False,
        include_hd_bundle: bool = True,
    ) -> None:
        memory_id = str(getattr(memory, "memory_id", "") or memory.get("memory_id") or "")
        if not memory_id:
            raise ValueError("memory_id is required for projection")
        text = self.text_for(memory)
        dense = self.dense_embedding(text)
        store.put_embedding(
            memory_id,
            embedding_family="almf-dense",
            embedding_version="v1",
            vector=dense,
        )
        if include_multivector:
            store.put_multivector(
                memory_id,
                embedding_family="almf-token",
                vectors=self.multivector(text),
                indexing_mode="token",
            )
        if include_hd_bundle:
            store.put_hd_bundle(
                memory_id,
                bundle_family="almf-hd",
                bundle_version="v1",
                bundle=self.hd_bundle(text),
            )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _token_bytes(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()
