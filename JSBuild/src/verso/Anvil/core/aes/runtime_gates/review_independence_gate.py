from __future__ import annotations

from pathlib import Path

import yaml

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate


class ReviewIndependenceGate(RuntimeGate):
    gate_id = "review_independence_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return ["evidence_bundle.json"]

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return compliance_context.aal in {"AAL-0", "AAL-1"}

    def schema_for_artifact(
        self, artifact: str, compliance_context: ComplianceContext
    ) -> str | None:
        return (
            "evidence_bundle.schema.json" if artifact == "evidence_bundle.json" else None
        )

    def validate_artifact_payload(
        self,
        artifact: str,
        payload: object,
        thresholds: dict[str, object],
        compliance_context: ComplianceContext,
    ) -> str | None:
        del thresholds
        if artifact != "evidence_bundle.json" or not isinstance(payload, dict):
            return None

        review_signoffs = payload.get("review_signoffs")
        if not isinstance(review_signoffs, list):
            return "review_signoffs must be a list"
        approved_reviewers = [
            str(item.get("reviewer", "")).strip()
            for item in review_signoffs
            if isinstance(item, dict) and str(item.get("decision", "")).lower() == "approved"
        ]
        unique_reviewers = {reviewer for reviewer in approved_reviewers if reviewer}
        required = self._required_review_count(compliance_context.aal)
        if len(unique_reviewers) < required:
            return (
                f"{compliance_context.aal} requires {required} approvals; "
                f"found {len(unique_reviewers)}"
            )

        author = str(payload.get("author", "")).strip()
        if author and author in unique_reviewers:
            return "review_signoffs must be independent from the author"

        return None

    def _required_review_count(self, aal: str) -> int:
        matrix_path = self._schema_root.parent / "review_matrix.yaml"
        if not matrix_path.exists():
            return 0
        try:
            payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return 0
        aal_levels = payload.get("aal_levels", {}) if isinstance(payload, dict) else {}
        config = aal_levels.get(aal, {}) if isinstance(aal_levels, dict) else {}
        if not isinstance(config, dict):
            return 0
        try:
            return int(config.get("independent_reviews", 0) or 0)
        except (TypeError, ValueError):
            return 0
