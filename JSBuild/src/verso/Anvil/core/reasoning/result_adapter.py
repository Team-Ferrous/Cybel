from __future__ import annotations

from typing import Any, Optional

import numpy as np


class ResultEmbeddingAdapter:
    """
    Adapts discrete tool output back into a continuous latent state.
    """

    def __init__(
        self,
        brain: Any,
        embedding_dim: int = 4096,
        residual_weight: float = 0.7,
        max_chars: int = 2000,
    ) -> None:
        self.brain = brain
        self.embedding_dim = int(max(1, embedding_dim))
        self.residual_weight = float(min(1.0, max(0.0, residual_weight)))
        self.max_chars = int(max(64, max_chars))

    def project_text(self, text: str) -> np.ndarray:
        if not hasattr(self.brain, "embeddings"):
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

        truncated = (text or "")[: self.max_chars]
        try:
            emb = np.asarray(self.brain.embeddings(truncated), dtype=np.float32)
        except Exception:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

        emb = self._to_2d(emb)
        emb = self._coerce_dim(emb)
        return self._normalize(emb)

    def inject(
        self,
        tool_result: str,
        pre_tool_state: Optional[np.ndarray],
        tool_name: str,
    ) -> np.ndarray:
        projected = self.project_text(f"[{tool_name}] {tool_result}")
        if pre_tool_state is None:
            return projected

        prev = self._coerce_dim(
            self._to_2d(np.asarray(pre_tool_state, dtype=np.float32))
        )
        merged = self.residual_weight * prev + (1.0 - self.residual_weight) * projected
        return self._normalize(merged)

    @staticmethod
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr.reshape(arr.shape[0], -1)

    def _coerce_dim(self, arr: np.ndarray) -> np.ndarray:
        if arr.shape[1] == self.embedding_dim:
            return arr
        if arr.shape[1] > self.embedding_dim:
            return arr[:, : self.embedding_dim]
        pad = np.zeros(
            (arr.shape[0], self.embedding_dim - arr.shape[1]), dtype=np.float32
        )
        return np.concatenate([arr, pad], axis=1)

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms <= 1e-8, 1.0, norms)
        return arr / norms
