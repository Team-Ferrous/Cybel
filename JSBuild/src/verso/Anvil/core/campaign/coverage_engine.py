"""Coverage tracking and research stop-proof generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CoverageAreaStatus:
    area: str
    count: int = 0
    weighted_score: float = 0.0
    max_confidence: float = 0.0
    avg_confidence: float = 0.0
    dominant_topics: list[str] = field(default_factory=list)
    source_scopes: list[str] = field(default_factory=list)
    sufficient: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoverageEngine:
    """Computes research topic coverage, sufficiency, and stop proofs."""

    REQUIRED_AREAS = [
        "architecture",
        "algorithms_and_math",
        "hardware_fit",
        "implementation_patterns",
        "testing_and_verification",
        "packaging_and_deployment",
        "observability_and_telemetry",
        "market_and_features",
    ]

    AREA_KEYWORDS = {
        "architecture": {"architecture", "design", "interface", "api", "module"},
        "algorithms_and_math": {"algorithm", "math", "numerical", "proof", "inference"},
        "hardware_fit": {"hardware", "simd", "avx", "openmp", "native", "cpu", "benchmark"},
        "implementation_patterns": {"implementation", "code", "refactor", "runtime", "subsystem"},
        "testing_and_verification": {"test", "verification", "audit", "reliability", "correctness"},
        "packaging_and_deployment": {"package", "deploy", "release", "distribution", "build"},
        "observability_and_telemetry": {"telemetry", "observability", "metrics", "trace", "profiling"},
        "market_and_features": {"market", "feature", "competitor", "product", "ux"},
    }

    def __init__(self, sufficiency_threshold: float = 0.75) -> None:
        self.sufficiency_threshold = sufficiency_threshold

    def coverage_matrix(self, evidence: list[dict[str, Any]] | Any) -> dict[str, int]:
        statuses = self.coverage_details(evidence)
        return {area: status.count for area, status in statuses.items()}

    def coverage_details(
        self,
        evidence: list[dict[str, Any]] | Any,
    ) -> dict[str, CoverageAreaStatus]:
        items = list(evidence)
        statuses = {
            area: CoverageAreaStatus(area=area)
            for area in self.REQUIRED_AREAS
        }
        confidence_totals = {area: 0.0 for area in self.REQUIRED_AREAS}
        topic_frequency = {area: {} for area in self.REQUIRED_AREAS}
        for item in items:
            areas = self._classify_areas(item)
            confidence = self._bounded_float(item.get("confidence"), default=0.5)
            novelty = self._bounded_float(item.get("novelty_score"), default=0.3)
            complexity = self._bounded_float(item.get("complexity_score"), default=0.3)
            weight = confidence * 0.55 + novelty * 0.25 + complexity * 0.20
            topic = str(item.get("topic") or "general")
            source_scope = str(item.get("source_scope") or item.get("repo_scope") or "unknown")
            for area in areas:
                status = statuses[area]
                status.count += 1
                status.weighted_score += weight
                status.max_confidence = max(status.max_confidence, confidence)
                confidence_totals[area] += confidence
                if source_scope not in status.source_scopes:
                    status.source_scopes.append(source_scope)
                topic_frequency[area][topic] = topic_frequency[area].get(topic, 0) + 1

        for area, status in statuses.items():
            if status.count:
                status.avg_confidence = confidence_totals[area] / status.count
                ranked_topics = sorted(
                    topic_frequency[area].items(),
                    key=lambda item: (-item[1], item[0]),
                )
                status.dominant_topics = [topic for topic, _ in ranked_topics[:3]]
            status.sufficient = (
                status.count > 0
                and status.weighted_score >= self.sufficiency_threshold
                and status.avg_confidence >= 0.45
            )
        return statuses

    def stop_proof(
        self,
        evidence: list[dict[str, Any]] | Any,
        unknowns: list[dict[str, Any]] | Any,
        *,
        yield_history: list[float] | None = None,
        impact_deltas: list[float] | None = None,
        frontier_size: int = 0,
        impact_threshold: float = 0.15,
    ) -> dict[str, Any]:
        items = list(evidence)
        statuses = self.coverage_details(items)
        sufficient = all(status.sufficient for status in statuses.values())
        unresolved = [
            item
            for item in unknowns
            if item.get("current_status", item.get("status")) not in {"resolved", "accepted", "answered", "waived"}
        ]
        high_yield_history = list(yield_history or [])
        recent_yield = high_yield_history[-3:]
        low_yield = bool(recent_yield) and max(recent_yield) < 0.2
        impact_window = list(impact_deltas or [])
        impact_below_threshold = bool(impact_window) and max(impact_window[-3:]) < impact_threshold
        topic_gaps = [
            area
            for area, status in statuses.items()
            if not status.sufficient
        ]
        stop_allowed = (
            sufficient
            and not unresolved
            and frontier_size == 0
            and (low_yield or not high_yield_history)
            and (impact_below_threshold or not impact_window)
        )
        return {
            "coverage": {area: status.count for area, status in statuses.items()},
            "coverage_details": {
                area: status.to_dict() for area, status in statuses.items()
            },
            "coverage_sufficient": sufficient,
            "remaining_unknowns": len(unresolved),
            "open_unknowns": unresolved,
            "frontier_size": frontier_size,
            "yield_history": high_yield_history,
            "low_yield_streak": low_yield,
            "impact_deltas": impact_window,
            "impact_below_threshold": impact_below_threshold,
            "topic_gaps": topic_gaps,
            "stop_allowed": stop_allowed,
            "stop_reasons": [
                "topic coverage reached configured sufficiency threshold"
                if sufficient
                else f"topic coverage still incomplete for: {', '.join(topic_gaps)}",
                "remaining unknowns resolved or accepted"
                if not unresolved
                else f"{len(unresolved)} unknowns remain unresolved",
                "frontier queue exhausted"
                if frontier_size == 0
                else f"{frontier_size} frontier items remain queued",
                "recent source yield is below threshold"
                if low_yield or not high_yield_history
                else "recent source yield is still material",
                "roadmap impact drift is below threshold"
                if impact_below_threshold or not impact_window
                else "new evidence is still changing roadmap direction materially",
            ],
        }

    def _classify_areas(self, item: dict[str, Any]) -> set[str]:
        haystack = " ".join(
            [
                str(item.get("topic") or ""),
                str(item.get("summary") or ""),
                str(item.get("title") or ""),
                str(item.get("source_scope") or ""),
                str(item.get("repo_scope") or ""),
            ]
        ).lower()
        matches = {
            area
            for area, keywords in self.AREA_KEYWORDS.items()
            if any(keyword in haystack for keyword in keywords)
        }
        return matches or {"implementation_patterns"}

    @staticmethod
    def _bounded_float(value: Any, *, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, number))
