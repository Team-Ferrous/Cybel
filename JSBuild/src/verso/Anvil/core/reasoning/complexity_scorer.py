from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class ComplexityProfile:
    score: int
    coconut_paths: int
    coconut_steps: int
    coconut_frequency: str
    subagent_coconut: bool
    recommended_context_budget: int


class ComplexityScorer:
    """Heuristic request complexity scorer used to tune reasoning intensity."""

    _HIGH_COMPLEXITY_TERMS = {
        "architecture",
        "end-to-end",
        "multi-file",
        "multi file",
        "orchestrate",
        "refactor",
        "migrate",
        "overhaul",
        "redesign",
        "distributed",
        "dependency",
        "cross-module",
        "cross module",
        "protocol",
        "tradeoff",
    }
    _MODERATE_COMPLEXITY_TERMS = {
        "implement",
        "optimize",
        "improve",
        "debug",
        "investigate",
        "integrate",
        "design",
        "performance",
        "reasoning",
        "context",
    }
    _DEPTH_TERMS = {"why", "how", "architecture", "flow", "interaction", "tradeoff"}

    def score_request(
        self,
        user_input: str,
        referenced_files: Optional[Iterable[str]] = None,
        question_type: Optional[str] = None,
    ) -> ComplexityProfile:
        text = (user_input or "").strip()
        lowered = text.lower()
        score = 1.0

        # Length signal.
        word_count = len(re.findall(r"\w+", text))
        if word_count > 25:
            score += 1.0
        if word_count > 55:
            score += 1.0

        # Keyword signal.
        high_hits = sum(1 for term in self._HIGH_COMPLEXITY_TERMS if term in lowered)
        moderate_hits = sum(
            1 for term in self._MODERATE_COMPLEXITY_TERMS if term in lowered
        )
        score += min(4.0, high_hits * 2.0)
        score += min(2.4, moderate_hits * 0.8)

        # Code reference/file-count signal.
        inline_file_refs = re.findall(r"[\w./-]+\.(?:py|cc|cpp|h|hpp|js|ts|md|json)", text)
        file_count = len(set(inline_file_refs))
        if referenced_files:
            file_count += len({f for f in referenced_files if f})
        if file_count >= 2:
            score += 2.0
        if file_count >= 5:
            score += 2.5

        # Question depth signal.
        depth_hits = sum(1 for term in self._DEPTH_TERMS if term in lowered)
        if depth_hits:
            score += min(2.7, depth_hits * 0.9)
        if question_type == "architecture":
            score += 1.5

        final_score = max(1, min(10, int(round(score))))
        return self._profile_from_score(final_score)

    def _profile_from_score(self, score: int) -> ComplexityProfile:
        if score <= 3:
            return ComplexityProfile(
                score=score,
                coconut_paths=4,
                coconut_steps=2,
                coconut_frequency="none",
                subagent_coconut=False,
                recommended_context_budget=120000,
            )
        if score <= 5:
            return ComplexityProfile(
                score=score,
                coconut_paths=6,
                coconut_steps=3,
                coconut_frequency="synthesis_only",
                subagent_coconut=False,
                recommended_context_budget=180000,
            )
        if score <= 7:
            return ComplexityProfile(
                score=score,
                coconut_paths=8,
                coconut_steps=4,
                coconut_frequency="per_phase",
                subagent_coconut=True,
                recommended_context_budget=280000,
            )
        return ComplexityProfile(
            score=score,
            coconut_paths=12 if score >= 9 else 10,
            coconut_steps=8 if score >= 9 else 6,
            coconut_frequency="per_step",
            subagent_coconut=True,
            recommended_context_budget=380000,
        )
