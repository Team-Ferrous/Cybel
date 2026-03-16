from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class SearchBudget:
    max_expansions: int
    max_depth: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class CandidateMeritScore:
    expression: str
    score: float
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PortfolioSearch:
    """Bounded search over a tiny arithmetic grammar for fallback synthesis."""

    def search(
        self,
        *,
        cases: list[dict[str, Any]],
        validator: Callable[[str, list[dict[str, Any]]], bool],
        budget: SearchBudget,
    ) -> CandidateMeritScore:
        candidates = [
            "value",
            "max(lower, min(upper, value))",
            "(value - lower) / (upper - lower)",
            "min(upper, max(lower, value))",
        ][: budget.max_expansions]
        best = CandidateMeritScore(expression=candidates[0], score=0.0, reasons=["fallback_default"])
        for expression in candidates:
            passed = validator(expression, cases)
            score = 1.0 if passed else self._score_partial(expression, cases)
            reasons = ["validated"] if passed else ["bounded_candidate"]
            candidate = CandidateMeritScore(expression=expression, score=score, reasons=reasons)
            if candidate.score > best.score:
                best = candidate
            if passed:
                return candidate
        return best

    @staticmethod
    def _score_partial(expression: str, cases: list[dict[str, Any]]) -> float:
        score = 0.0
        if "min" in expression or "max" in expression:
            score += 0.4
        if "/" in expression:
            score += 0.2
        score += min(0.4, len(cases) * 0.05)
        return round(score, 3)

