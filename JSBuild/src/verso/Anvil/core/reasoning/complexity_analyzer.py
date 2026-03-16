from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Optional


@dataclass(frozen=True)
class AdaptiveComplexityProfile:
    complexity_score: float
    estimated_files: int
    reasoning_depth: int
    subagent_count: int
    subagent_coconut: bool
    max_steps_per_agent: int
    coconut_depth: int
    tool_intensity: float


class TaskComplexityAnalyzer:
    """Adaptive complexity model used for subagent/depth guidance."""

    _HIGH_TERMS = {
        "refactor",
        "migrate",
        "integrate",
        "end-to-end",
        "orchestrate",
        "roadmap",
        "architecture",
        "multi-agent",
    }
    _MEDIUM_TERMS = {
        "analyze",
        "investigate",
        "implement",
        "debug",
        "optimize",
        "pipeline",
    }

    def analyze(
        self,
        query: str,
        candidate_files: Optional[Iterable[str]] = None,
        previous_entropy: Optional[float] = None,
    ) -> AdaptiveComplexityProfile:
        text = (query or "").strip().lower()
        token_count = len(re.findall(r"\w+", text))
        inline_file_refs = re.findall(
            r"[\w./-]+\.(?:py|cc|cpp|c|h|hpp|js|ts|md|json|yaml|yml)",
            text,
        )
        referenced = set(inline_file_refs)
        if candidate_files:
            referenced.update(str(p) for p in candidate_files if p)
        estimated_files = max(1, len(referenced))

        high_hits = sum(1 for term in self._HIGH_TERMS if term in text)
        medium_hits = sum(1 for term in self._MEDIUM_TERMS if term in text)

        length_score = min(1.0, token_count / 220.0)
        file_score = min(1.0, estimated_files / 12.0)
        keyword_score = min(1.0, 0.22 * high_hits + 0.12 * medium_hits)
        entropy_score = (
            float(previous_entropy) if previous_entropy is not None else 0.45
        )
        entropy_score = max(0.0, min(1.0, entropy_score))

        complexity = (
            0.35 * keyword_score
            + 0.2 * length_score
            + 0.2 * file_score
            + 0.25 * entropy_score
        )
        complexity = max(0.0, min(1.0, complexity))

        return AdaptiveComplexityProfile(
            complexity_score=complexity,
            estimated_files=estimated_files,
            reasoning_depth=self._reasoning_depth(complexity),
            subagent_count=self._subagent_count(complexity),
            subagent_coconut=complexity >= 0.55,
            max_steps_per_agent=self._max_steps(complexity),
            coconut_depth=self._coconut_depth(complexity),
            tool_intensity=self._tool_intensity(complexity),
        )

    @staticmethod
    def _reasoning_depth(complexity: float) -> int:
        if complexity < 0.3:
            return 1
        if complexity < 0.5:
            return 2
        if complexity < 0.7:
            return 4
        return 6

    @staticmethod
    def _subagent_count(complexity: float) -> int:
        if complexity < 0.25:
            return 1
        if complexity < 0.45:
            return 2
        if complexity < 0.7:
            return 3
        return 4

    @staticmethod
    def _max_steps(complexity: float) -> int:
        return max(4, min(24, int(round(4 + complexity * 20))))

    @staticmethod
    def _coconut_depth(complexity: float) -> int:
        return max(2, min(16, int(round(2 + complexity * 14))))

    @staticmethod
    def _tool_intensity(complexity: float) -> float:
        return max(0.2, min(3.0, 0.4 + complexity * 2.2))
