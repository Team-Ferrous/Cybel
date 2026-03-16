"""Audit-loop support for autonomy campaigns."""

from __future__ import annotations

import uuid
from typing import Dict, List


class CampaignAuditEngine:
    """Run deterministic audit convergence checks from campaign artifacts."""

    def __init__(self, state_store, campaign_id: str):
        self.state_store = state_store
        self.campaign_id = campaign_id

    def run(self, *, scope: str = "final") -> Dict[str, object]:
        artifacts = self.state_store.list_artifacts(self.campaign_id)
        telemetry = self.state_store.list_telemetry(self.campaign_id)
        roadmap = self.state_store.list_roadmap_items(self.campaign_id)
        verification_lane = [
            item for item in telemetry if item.get("telemetry_kind") == "verification_lane"
        ]
        shadow_preflight = [
            item for item in telemetry if item.get("telemetry_kind") == "shadow_preflight"
        ]

        findings: List[Dict[str, object]] = []
        if not any(item["family"] == "roadmap_final" for item in artifacts):
            findings.append(
                {
                    "finding_id": f"finding_{uuid.uuid4().hex[:12]}",
                    "audit_run_id": "",
                    "campaign_id": self.campaign_id,
                    "severity": "high",
                    "category": "documentation_completeness",
                    "summary": "Final roadmap artifact is missing.",
                    "details": {},
                }
            )
        if roadmap and not telemetry:
            findings.append(
                {
                    "finding_id": f"finding_{uuid.uuid4().hex[:12]}",
                    "audit_run_id": "",
                    "campaign_id": self.campaign_id,
                    "severity": "medium",
                    "category": "observability",
                    "summary": "Roadmap exists but no telemetry has been recorded.",
                    "details": {"roadmap_items": len(roadmap)},
                }
            )
        if telemetry and not shadow_preflight:
            findings.append(
                {
                    "finding_id": f"finding_{uuid.uuid4().hex[:12]}",
                    "audit_run_id": "",
                    "campaign_id": self.campaign_id,
                    "severity": "medium",
                    "category": "preflight_coverage",
                    "summary": "Experiment telemetry exists without shadow preflight coverage.",
                    "details": {"telemetry_events": len(telemetry)},
                }
            )
        unresolved_counterexamples = sum(
            len(list(item.get("counterexamples") or []))
            for item in verification_lane
            if not bool(item.get("all_passed", False))
            or bool(item.get("promotion_blocked"))
        )
        if unresolved_counterexamples:
            findings.append(
                {
                    "finding_id": f"finding_{uuid.uuid4().hex[:12]}",
                    "audit_run_id": "",
                    "campaign_id": self.campaign_id,
                    "severity": "medium",
                    "category": "verification_closure",
                    "summary": "Verification counterexample debt remains unresolved.",
                    "details": {"counterexample_count": unresolved_counterexamples},
                }
            )

        audit_run_id = f"audit_{uuid.uuid4().hex[:12]}"
        for finding in findings:
            finding["audit_run_id"] = audit_run_id

        summary = {
            "finding_count": len(findings),
            "material_findings": len([item for item in findings if item["severity"] in {"high", "critical"}]),
            "unresolved_risk_count": len(
                [item for item in findings if item["severity"] in {"medium", "high", "critical"}]
            ),
        }
        self.state_store.record_audit_run(
            {
                "audit_run_id": audit_run_id,
                "campaign_id": self.campaign_id,
                "scope": scope,
                "status": "failed" if findings else "passed",
                "summary": summary,
            }
        )
        if findings:
            self.state_store.record_audit_findings(findings)
        return {"audit_run_id": audit_run_id, "summary": summary, "findings": findings}
