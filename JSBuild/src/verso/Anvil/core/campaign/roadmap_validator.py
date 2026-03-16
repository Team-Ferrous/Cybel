"""Static validation for roadmap phase packets."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class RoadmapValidator:
    """Validates roadmap items and phase packets before development starts."""

    REQUIRED_PHASE_FIELDS = (
        "objective",
        "repo_scope",
        "owning_specialist_type",
        "allowed_writes",
        "telemetry_contract",
        "required_evidence",
        "rollback_criteria",
        "promotion_gate",
    )
    REQUIRED_SYNTHESIS_BUNDLE_FIELDS = (
        "spec_path",
        "replay_tape_path",
        "proof_capsule_path",
        "verification_passed",
    )

    def validate(
        self,
        items: Iterable[Dict[str, Any]],
        phase_packets: Iterable[Dict[str, Any]],
    ) -> List[str]:
        errors: List[str] = []
        seen_ids: set[str] = set()
        for item in items:
            item_id = str(item.get("item_id") or "")
            if not item_id:
                errors.append("Roadmap item missing item_id")
            elif item_id in seen_ids:
                errors.append(f"Duplicate roadmap item_id: {item_id}")
            else:
                seen_ids.add(item_id)
            if not item.get("telemetry_contract"):
                errors.append(f"Roadmap item missing telemetry contract: {item_id}")
            if not item.get("exit_gate"):
                errors.append(f"Roadmap item missing promotion gate: {item_id}")

        for packet in phase_packets:
            phase_id = str(packet.get("phase_id") or "unknown")
            if not packet.get("tasks"):
                continue
            for field in self.REQUIRED_PHASE_FIELDS:
                value = packet.get(field)
                if value in (None, "", [], {}):
                    errors.append(f"Phase packet '{phase_id}' missing {field}")
        return errors

    def validate_synthesis_promotion_bundle(self, bundle: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        payload = dict(bundle or {})
        for field in self.REQUIRED_SYNTHESIS_BUNDLE_FIELDS:
            value = payload.get(field)
            if value in (None, "", [], {}):
                errors.append(f"missing_{field}")
        if payload.get("verification_passed") is not True:
            errors.append("verification_not_passed")
        if not payload.get("benchmark_summary"):
            errors.append("missing_benchmark_summary")
        if not payload.get("roadmap_validation"):
            errors.append("missing_roadmap_validation")
        return errors

    def summarize_synthesis_promotion(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        errors = self.validate_synthesis_promotion_bundle(bundle)
        return {
            "allowed": not errors,
            "errors": errors,
            "bundle": dict(bundle or {}),
        }
