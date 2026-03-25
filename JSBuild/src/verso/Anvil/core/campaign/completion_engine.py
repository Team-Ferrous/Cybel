"""Closure proof generation for autonomy campaigns."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict

from core.campaign.workspace import CampaignWorkspace


class CompletionEngine:
    """Build reproducible closure proofs from campaign state."""

    def __init__(
        self,
        workspace: CampaignWorkspace,
        state_store,
        campaign_id: str,
        *,
        event_store=None,
        memory_fabric=None,
    ):
        self.workspace = workspace
        self.state_store = state_store
        self.campaign_id = campaign_id
        self.event_store = event_store
        self.memory_fabric = memory_fabric

    def build_proof(
        self,
        *,
        replay_export: Dict[str, Any] | None = None,
    ) -> Dict[str, object]:
        artifacts = self.state_store.list_artifacts(self.campaign_id)
        audits = self.state_store.list_audit_runs(self.campaign_id)
        audit_findings = self.state_store.list_audit_findings(self.campaign_id)
        telemetry = self.state_store.list_telemetry(self.campaign_id)
        checks = self.state_store.list_completion_checks(self.campaign_id)
        roadmap = self.state_store.list_roadmap_items(self.campaign_id)
        replay = replay_export or (
            self.event_store.export_run(self.campaign_id) if self.event_store is not None else {}
        )
        latent_packages = (
            self.memory_fabric.list_latent_packages(campaign_id=self.campaign_id)
            if self.memory_fabric is not None
            else []
        )
        verification_lane = [
            item for item in telemetry if item.get("telemetry_kind") == "verification_lane"
        ]

        open_findings = [
            finding for finding in audit_findings if finding["severity"] in {"high", "critical"}
        ]
        unresolved_risk_count = len(open_findings) + sum(
            len(list(item.get("counterexamples") or []))
            + (1 if item.get("promotion_blocked") else 0)
            for item in verification_lane
        )
        closure_status = (
            "provable"
            if unresolved_risk_count == 0 and replay
            else "blocked"
        )
        capsule = dict(replay.get("mission_capsule") or {})
        proof = {
            "campaign_id": self.campaign_id,
            "completed_roadmap_items": len([item for item in roadmap if item.get("status") in {"completed", "planned"}]),
            "artifact_inventory": [artifact["artifact_id"] for artifact in artifacts],
            "audit_run_count": len(audits),
            "open_material_findings": len(open_findings),
            "telemetry_event_count": len(telemetry),
            "completion_checks": checks,
            "replay_hash": str((replay.get("replay") or {}).get("deterministic_hash") or ""),
            "capsule_id": str(capsule.get("capsule_id") or ""),
            "mission_capsule": capsule,
            "latent_packages": [
                {
                    "latent_package_id": str(item.get("latent_package_id") or ""),
                    "memory_id": str(item.get("memory_id") or ""),
                    "capture_stage": str(item.get("capture_stage") or ""),
                }
                for item in latent_packages[:10]
            ],
            "safety_case": dict(replay.get("safety_case") or {}),
            "safety_case_node_count": len((replay.get("safety_case") or {}).get("nodes") or []),
            "unresolved_risk_count": unresolved_risk_count,
            "closure_status": closure_status,
            "closure_allowed": not open_findings and unresolved_risk_count == 0,
        }
        return proof

    def persist_proof(self, proof: Dict[str, Any] | None = None) -> str:
        proof = proof or self.build_proof()
        artifact_dir = self.workspace.artifact_family_dir("closure")
        target = os.path.join(artifact_dir, f"closure_proof_{uuid.uuid4().hex[:12]}.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(proof, handle, indent=2, default=str)
        return target
