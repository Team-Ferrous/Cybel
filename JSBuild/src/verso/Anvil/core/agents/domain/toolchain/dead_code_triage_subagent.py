"""Dead-code triage specialist for Saguaro deadcode reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


@dataclass(frozen=True)
class DeadCodeTriageDecision:
    """Normalized disposition for one dead-code candidate."""

    symbol: str
    file: str
    confidence: float
    disposition: str
    rationale: str
    suggested_owner: str
    suggested_action: str


class DeadCodeTriageSubagent(DomainSpecialistSubagent):
    """Triages Saguaro deadcode candidates into actionable buckets."""

    system_prompt = """You are Anvil's DeadCodeTriageSubagent.

Mission:
- Run Saguaro deadcode and unwired to triage each candidate into
  rewireable, obsolete, or manual review.

Focus on:
- false-positive reduction for dynamic registration paths
- isolated feature islands that are internally connected but not wired to roots
- explicit wiring recommendations for salvageable symbols
- safe removal preconditions for truly dead code
"""
    tools = [*DomainSpecialistSubagent.governance_tools(), "deadcode", "unwired"]

    @staticmethod
    def parse_deadcode_payload(payload: str | dict[str, Any]) -> dict[str, Any]:
        """Parse JSON payload returned by the deadcode tool."""
        if isinstance(payload, dict):
            return payload

        text = str(payload or "").strip()
        if not text:
            return {"threshold": 0.5, "count": 0, "candidates": []}

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("deadcode payload did not contain JSON object")

        return json.loads(text[start : end + 1])

    @classmethod
    def triage_from_payload(
        cls,
        payload: str | dict[str, Any],
        *,
        max_candidates: int = 250,
    ) -> dict[str, Any]:
        """Parse a raw payload and produce a triage summary."""
        report = cls.parse_deadcode_payload(payload)
        return cls.triage_report(report, max_candidates=max_candidates)

    @classmethod
    def triage_report(
        cls,
        report: dict[str, Any],
        *,
        max_candidates: int = 250,
    ) -> dict[str, Any]:
        """Classify report candidates into actionable disposition buckets."""
        candidates = list(report.get("candidates") or [])
        decisions = [
            cls.classify_candidate(candidate)
            for candidate in candidates[: max(0, int(max_candidates))]
        ]

        buckets: dict[str, list[dict[str, Any]]] = {
            "rewire_candidate": [],
            "likely_obsolete": [],
            "needs_manual_review": [],
        }
        for decision in decisions:
            buckets.setdefault(decision.disposition, []).append(asdict(decision))

        return {
            "threshold": report.get("threshold"),
            "input_count": len(candidates),
            "analyzed_count": len(decisions),
            "summary": {key: len(values) for key, values in buckets.items()},
            "rewire_candidates": buckets["rewire_candidate"],
            "likely_obsolete": buckets["likely_obsolete"],
            "needs_manual_review": buckets["needs_manual_review"],
        }

    @classmethod
    def classify_candidate(cls, candidate: dict[str, Any]) -> DeadCodeTriageDecision:
        """Classify a single deadcode candidate into a disposition."""
        symbol = str(candidate.get("symbol") or "").strip()
        file_path = str(candidate.get("file") or "").strip()
        module = str(candidate.get("module") or "").strip()
        confidence = cls._normalize_confidence(candidate.get("confidence"))
        is_public = bool(candidate.get("public"))
        is_dynamic_file = bool(candidate.get("dynamic_file"))
        lowered_path = file_path.replace("\\", "/").lower()

        if not symbol or not file_path:
            return DeadCodeTriageDecision(
                symbol=symbol or "<unknown>",
                file=file_path or "<unknown>",
                confidence=confidence,
                disposition="needs_manual_review",
                rationale="candidate metadata incomplete",
                suggested_owner="ImplementationEngineerSubagent",
                suggested_action="Re-run deadcode and inspect raw analyzer entry.",
            )

        if is_dynamic_file:
            return DeadCodeTriageDecision(
                symbol=symbol,
                file=file_path,
                confidence=confidence,
                disposition="rewire_candidate",
                rationale="dynamic file paths often evade static reference analysis",
                suggested_owner="ImplementationEngineerSubagent",
                suggested_action=(
                    "Trace runtime registration and add an explicit callsite test."
                ),
            )

        if symbol.endswith("Subagent"):
            return DeadCodeTriageDecision(
                symbol=symbol,
                file=file_path,
                confidence=confidence,
                disposition="rewire_candidate",
                rationale=(
                    "specialist classes are commonly routed by registry, not direct "
                    "imports"
                ),
                suggested_owner="CampaignDirectorSubagent",
                suggested_action=(
                    "Wire role through SpecialistRegistry catalog, prompt map, and "
                    "routing rules."
                ),
            )

        if "/core/aes/checks/" in lowered_path:
            return DeadCodeTriageDecision(
                symbol=symbol,
                file=file_path,
                confidence=confidence,
                disposition="rewire_candidate",
                rationale=(
                    "AES check functions are frequently loaded via policy and metadata"
                ),
                suggested_owner="AESSentinelSubagent",
                suggested_action=(
                    "Confirm rule binding path and add a verification test for this "
                    "check."
                ),
            )

        if is_public and "/core/agents/domain/" in lowered_path:
            return DeadCodeTriageDecision(
                symbol=symbol,
                file=file_path,
                confidence=confidence,
                disposition="rewire_candidate",
                rationale=(
                    "public domain specialists must be discoverable by routing "
                    "contracts"
                ),
                suggested_owner="CampaignDirectorSubagent",
                suggested_action=(
                    "Ensure specialist appears in catalog and prompt-key mapping."
                ),
            )

        if is_public and confidence >= 0.7:
            return DeadCodeTriageDecision(
                symbol=symbol,
                file=file_path,
                confidence=confidence,
                disposition="rewire_candidate",
                rationale=(
                    "public symbol likely intended as extension point despite static "
                    "miss"
                ),
                suggested_owner="ImplementationEngineerSubagent",
                suggested_action=(
                    "Locate intended integration path or deprecate explicitly."
                ),
            )

        if symbol.startswith("_") and not is_public and confidence >= 0.7:
            return DeadCodeTriageDecision(
                symbol=symbol,
                file=file_path,
                confidence=confidence,
                disposition="likely_obsolete",
                rationale="private symbol with high deadcode confidence",
                suggested_owner="ImplementationEngineerSubagent",
                suggested_action=(
                    "Delete behind targeted tests and verify with saguaro verify."
                ),
            )

        if confidence < 0.65 or "No static references found" not in str(
            candidate.get("reason") or ""
        ):
            return DeadCodeTriageDecision(
                symbol=symbol,
                file=file_path,
                confidence=confidence,
                disposition="needs_manual_review",
                rationale="low confidence or non-standard deadcode signal",
                suggested_owner="TestAuditSubagent",
                suggested_action=(
                    "Inspect call graph and execution traces before deletion."
                ),
            )

        return DeadCodeTriageDecision(
            symbol=symbol,
            file=file_path,
            confidence=confidence,
            disposition="likely_obsolete",
            rationale=f"high-confidence deadcode candidate in module {module}",
            suggested_owner="ImplementationEngineerSubagent",
            suggested_action="Stage removal and run regression suite.",
        )

    @staticmethod
    def _normalize_confidence(raw: Any) -> float:
        try:
            return round(float(raw), 4)
        except (TypeError, ValueError):
            return 0.0
