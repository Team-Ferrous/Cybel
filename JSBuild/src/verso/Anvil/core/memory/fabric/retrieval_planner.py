"""Retrieval planning for ALMF memory objects."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

import numpy as np

from core.memory.fabric.models import MemoryReadRecord
from core.memory.fabric.projectors import MemoryProjector
from core.memory.fabric.reranker import MemoryReranker
from core.memory.fabric.store import MemoryFabricStore


class MemoryRetrievalPlanner:
    """Hybrid retrieval over canonical, dense, and multivector projections."""

    def __init__(self, store: MemoryFabricStore, projector: MemoryProjector) -> None:
        self.store = store
        self.projector = projector
        self.reranker = MemoryReranker(store, projector)

    def retrieve(
        self,
        *,
        campaign_id: str,
        query_text: str,
        planner_mode: str = "hybrid",
        memory_kinds: Optional[Iterable[str]] = None,
        repo_context: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        candidates = self.store.list_memories(
            campaign_id,
            memory_kinds=memory_kinds,
            repo_context=repo_context,
        )
        if not candidates and query_text:
            candidates = self.store.search_text(
                campaign_id,
                query_text,
                memory_kinds=memory_kinds,
                limit=max(limit * 4, 20),
            )
        query_embedding = self.projector.dense_embedding(query_text)
        scored: list[Dict[str, Any]] = []
        for candidate in candidates:
            memory_id = str(candidate.get("memory_id") or "")
            dense = self.store.get_embedding(memory_id)
            if dense is None:
                dense = self.projector.dense_embedding(self.projector.text_for(candidate))
            score = _cosine(query_embedding, dense)
            if query_text and query_text.lower() in str(candidate.get("summary_text") or "").lower():
                score += 0.15
            scored.append({**candidate, "_dense_score": score})
        scored.sort(key=lambda item: float(item.get("_dense_score") or 0.0), reverse=True)
        reranked = self.reranker.rerank(query_text, scored[: max(limit * 4, 8)])
        result_items = reranked[: max(1, int(limit))]
        latency_ms = (time.perf_counter() - started) * 1000.0
        read = MemoryReadRecord(
            campaign_id=campaign_id,
            query_kind="memory_retrieval",
            query_text=query_text,
            planner_mode=planner_mode,
            result_memory_ids_json=[str(item.get("memory_id") or "") for item in result_items],
            latency_ms=latency_ms,
        )
        self.store.record_read(read)
        return {
            "read_id": read.read_id,
            "planner_mode": planner_mode,
            "query_text": query_text,
            "results": result_items,
            "latency_ms": latency_ms,
        }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))
