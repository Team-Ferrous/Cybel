from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import hashlib
import re

import numpy as np


@dataclass(frozen=True)
class ToolIntentSignal:
    tool_name: str
    confidence: float
    scores: Dict[str, float]


class ToolIntentClassifier:
    """
    Lightweight latent-space tool-intent classifier.

    It combines latent similarity against deterministic tool prototypes with
    keyword priors from the active context.
    """

    _KEYWORD_PRIORS = {
        "saguaro_query": (
            "find",
            "search",
            "discover",
            "investigate",
            "analyze",
            "architecture",
            "flow",
            "roadmap",
            "where",
            "which",
        ),
        "skeleton": ("skeleton", "outline", "structure", "class", "function"),
        "slice": ("slice", "method", "symbol", "definition"),
        "read_file": ("read", "inspect", "open", "contents"),
        "read_files": ("read files", "inspect files", "multiple files"),
        "find_by_name": ("filename", "file name", "match", "glob"),
        "grep_search": ("grep", "regex", "pattern", "search text"),
    }

    def __init__(
        self,
        embedding_dim: int = 4096,
        tool_names: Optional[Iterable[str]] = None,
        threshold: float = 0.82,
        latent_weight: float = 0.65,
        keyword_weight: float = 0.35,
    ) -> None:
        self.embedding_dim = int(max(8, embedding_dim))
        self.threshold = float(threshold)
        self.latent_weight = float(latent_weight)
        self.keyword_weight = float(keyword_weight)
        self.tool_names = list(tool_names or [])
        self._prototypes: Dict[str, np.ndarray] = {}
        for tool_name in self.tool_names:
            self._prototypes[tool_name] = self._make_prototype(tool_name)

    def detect(
        self,
        hidden_state: np.ndarray,
        context_text: str = "",
        allowed_tools: Optional[Iterable[str]] = None,
    ) -> Optional[ToolIntentSignal]:
        tools = list(allowed_tools or self.tool_names)
        if not tools:
            return None

        state = self._normalize(self._coerce_dim(self._flatten(hidden_state)))
        latent_scores = self._latent_scores(state, tools)
        keyword_scores = self._keyword_scores(context_text, tools)

        combined = {
            name: (
                self.latent_weight * latent_scores.get(name, 0.0)
                + self.keyword_weight * keyword_scores.get(name, 0.0)
            )
            for name in tools
        }
        probabilities = self._softmax_dict(combined)
        if not probabilities:
            return None

        best_tool = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[best_tool])
        if confidence < self.threshold:
            return None

        return ToolIntentSignal(
            tool_name=best_tool,
            confidence=confidence,
            scores=probabilities,
        )

    def _latent_scores(self, state: np.ndarray, tools: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for tool_name in tools:
            prototype = self._prototypes.get(tool_name)
            if prototype is None:
                prototype = self._make_prototype(tool_name)
                self._prototypes[tool_name] = prototype
            scores[tool_name] = float(np.dot(state, prototype))
        return scores

    def _keyword_scores(self, context_text: str, tools: List[str]) -> Dict[str, float]:
        lowered = (context_text or "").lower()
        scores: Dict[str, float] = {}
        for tool_name in tools:
            keywords = self._KEYWORD_PRIORS.get(tool_name, ())
            if not keywords:
                scores[tool_name] = 0.0
                continue
            hits = sum(
                1 for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", lowered)
            )
            scores[tool_name] = min(1.0, hits / max(1.0, len(keywords) * 0.35))
        return scores

    def _make_prototype(self, tool_name: str) -> np.ndarray:
        digest = hashlib.sha256(tool_name.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**32 - 1)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        return self._normalize(vec)

    @staticmethod
    def _flatten(hidden_state: np.ndarray) -> np.ndarray:
        arr = np.asarray(hidden_state, dtype=np.float32)
        if arr.ndim == 0:
            return np.zeros((8,), dtype=np.float32)
        return arr.reshape(-1)

    def _coerce_dim(self, vec: np.ndarray) -> np.ndarray:
        if vec.shape[0] == self.embedding_dim:
            return vec
        if vec.shape[0] > self.embedding_dim:
            return vec[: self.embedding_dim]
        padded = np.zeros((self.embedding_dim,), dtype=np.float32)
        padded[: vec.shape[0]] = vec
        return padded

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return np.zeros_like(vec)
        return vec / norm

    @staticmethod
    def _softmax_dict(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        names = list(scores.keys())
        values = np.asarray([scores[name] for name in names], dtype=np.float32)
        values -= float(np.max(values))
        exps = np.exp(values)
        denom = float(np.sum(exps))
        if denom <= 1e-12:
            uniform = 1.0 / float(len(names))
            return {name: uniform for name in names}
        probs = exps / denom
        return {name: float(prob) for name, prob in zip(names, probs)}
