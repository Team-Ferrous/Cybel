from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class LatticeCandidate:
    name: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgramLattice:
    """Rerank component candidates using bundled fragment vectors."""

    def __init__(self, *, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def encode_fragment(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in str(text or "").lower().replace("/", " ").replace("_", " ").split():
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            slot = digest[0] % self.dimensions
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[slot] += sign
        return self._normalize(vector)

    def bundle(self, fragments: list[str]) -> list[float]:
        vector = [0.0] * self.dimensions
        for fragment in fragments:
            encoded = self.encode_fragment(fragment)
            vector = [lhs + rhs for lhs, rhs in zip(vector, encoded)]
        return self._normalize(vector)

    def rerank_candidates(
        self,
        query_fragments: list[str],
        candidates: list[dict[str, Any]],
    ) -> list[LatticeCandidate]:
        query_vector = self.bundle(query_fragments)
        ranked: list[LatticeCandidate] = []
        for candidate in candidates:
            terms = list(candidate.get("terms") or [])
            name = str(candidate.get("name") or candidate.get("qualified_name") or "")
            candidate_vector = self.bundle([name, *[str(item) for item in terms]])
            ranked.append(
                LatticeCandidate(
                    name=name,
                    score=round(self._cosine(query_vector, candidate_vector), 4),
                    metadata=dict(candidate),
                )
            )
        return sorted(ranked, key=lambda item: (-item.score, item.name))

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return [item / norm for item in vector]

    @staticmethod
    def _cosine(lhs: list[float], rhs: list[float]) -> float:
        return sum(a * b for a, b in zip(lhs, rhs))

