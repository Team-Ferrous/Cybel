"""High-precision reranking helpers for memory retrieval."""

from __future__ import annotations

import math
import re
from typing import Any, Dict

import numpy as np

from core.memory.fabric.projectors import MemoryProjector
from core.memory.fabric.store import MemoryFabricStore


class MemoryReranker:
    """Rerank candidates with lexical and multivector signals."""

    def __init__(self, store: MemoryFabricStore, projector: MemoryProjector) -> None:
        self.store = store
        self.projector = projector

    def rerank(self, query_text: str, candidates: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        query_tokens = set(_tokenize(query_text))
        query_multivector = self.projector.multivector(query_text)
        rescored: list[tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            summary_text = str(candidate.get("summary_text") or "")
            lexical_tokens = set(_tokenize(summary_text))
            lexical_overlap = (
                len(query_tokens & lexical_tokens) / max(1, len(query_tokens | lexical_tokens))
            )
            multivector_score = 0.0
            memory_id = str(candidate.get("memory_id") or "")
            stored_multivector = self.store.get_multivector(memory_id)
            if (
                stored_multivector is not None
                and stored_multivector.size
                and query_multivector.size
            ):
                multivector_score = _max_cosine(query_multivector, stored_multivector)
            dense_score = float(candidate.get("_dense_score") or 0.0)
            total = (dense_score * 0.55) + (lexical_overlap * 0.25) + (multivector_score * 0.20)
            rescored.append((total, {**candidate, "_score": total}))
        rescored.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in rescored]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _max_cosine(query_vectors: np.ndarray, candidate_vectors: np.ndarray) -> float:
    best = 0.0
    for query_vector in query_vectors:
        query_norm = np.linalg.norm(query_vector)
        if query_norm <= 0:
            continue
        for candidate_vector in candidate_vectors:
            candidate_norm = np.linalg.norm(candidate_vector)
            if candidate_norm <= 0:
                continue
            score = float(np.dot(query_vector, candidate_vector) / (query_norm * candidate_norm))
            best = max(best, score)
    return best
