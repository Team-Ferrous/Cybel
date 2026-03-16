"""Evidence-backed governance proposal generation."""

from __future__ import annotations

import json
import os
import time
from typing import Any


class RuleProposalEngine:
    """Aggregate repeated failures and success patterns into draft rules."""

    def __init__(self, state_store, standards_dir: str = "standards/governance") -> None:
        self.state_store = state_store
        self.standards_dir = os.path.abspath(standards_dir)
        os.makedirs(self.standards_dir, exist_ok=True)

    def build(self, campaign_id: str) -> dict[str, Any]:
        telemetry = self.state_store.list_telemetry(campaign_id)
        verification_failures = [
            item
            for item in telemetry
            if item.get("telemetry_kind") == "verification_lane"
            and not bool(item.get("all_passed", False))
        ]
        verification_successes = [
            item
            for item in telemetry
            if item.get("telemetry_kind") == "verification_lane"
            and bool(item.get("all_passed", False))
        ]
        task_packets = self.state_store.list_task_packets(campaign_id)
        proposals: list[dict[str, Any]] = []
        if verification_failures:
            proposals.append(
                {
                    "rule_id": f"{campaign_id}:require_verification_lane",
                    "kind": "verification_failure_cluster",
                    "support_count": len(verification_failures),
                    "false_positive_estimate": round(
                        len(verification_successes)
                        / max(1, len(verification_successes) + len(verification_failures)),
                        3,
                    ),
                    "statement": "Require the verification lane before campaign promotion.",
                }
            )
        if task_packets:
            proposals.append(
                {
                    "rule_id": f"{campaign_id}:task_packet_obligations",
                    "kind": "success_pattern",
                    "support_count": len(task_packets),
                    "false_positive_estimate": 0.0,
                    "statement": "Keep task packet output obligations explicit and machine-checked.",
                }
            )
        payload = {
            "campaign_id": campaign_id,
            "proposal_count": len(proposals),
            "proposals": proposals,
        }
        path = os.path.join(self.standards_dir, f"{campaign_id}_rule_proposals.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        payload["path"] = path
        return payload

    def adopt(
        self,
        campaign_id: str,
        rule_id: str,
        *,
        approved_by: str,
        notes: str = "",
    ) -> dict[str, Any]:
        proposals = self.build(campaign_id)
        match = next(
            (
                item
                for item in proposals.get("proposals", [])
                if str(item.get("rule_id") or "") == str(rule_id)
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Unknown rule proposal: {rule_id}")
        adoption = {
            "campaign_id": campaign_id,
            "rule_id": rule_id,
            "approved_by": approved_by,
            "notes": notes,
            "approved_at": time.time(),
            "proposal": match,
            "requires_human_approval": True,
        }
        path = os.path.join(self.standards_dir, f"{campaign_id}_rule_adoptions.json")
        payload = self._append_record(path, adoption)
        payload["path"] = path
        return adoption

    def record_outcome(
        self,
        campaign_id: str,
        rule_id: str,
        *,
        outcome_status: str,
        regression_delta: float = 0.0,
        notes: str = "",
    ) -> dict[str, Any]:
        outcome = {
            "campaign_id": campaign_id,
            "rule_id": rule_id,
            "outcome_status": outcome_status,
            "regression_delta": float(regression_delta),
            "notes": notes,
            "recorded_at": time.time(),
        }
        path = os.path.join(self.standards_dir, f"{campaign_id}_rule_outcomes.json")
        self._append_record(path, outcome)
        outcome["path"] = path
        return outcome

    def status(self, campaign_id: str) -> dict[str, Any]:
        proposals = self.build(campaign_id)
        adoptions = self._load_records(
            os.path.join(self.standards_dir, f"{campaign_id}_rule_adoptions.json")
        )
        outcomes = self._load_records(
            os.path.join(self.standards_dir, f"{campaign_id}_rule_outcomes.json")
        )
        adopted_ids = {str(item.get("rule_id") or "") for item in adoptions}
        return {
            "campaign_id": campaign_id,
            "proposal_count": int(proposals.get("proposal_count", 0)),
            "adoption_count": len(adoptions),
            "outcome_count": len(outcomes),
            "adopted_rule_ids": sorted(item for item in adopted_ids if item),
            "adoptions": adoptions,
            "outcomes": outcomes,
        }

    @staticmethod
    def _load_records(path: str) -> list[dict[str, Any]]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload.get("records") or [])

    def _append_record(self, path: str, record: dict[str, Any]) -> dict[str, Any]:
        payload = {"records": self._load_records(path)}
        payload["records"].append(record)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return payload
