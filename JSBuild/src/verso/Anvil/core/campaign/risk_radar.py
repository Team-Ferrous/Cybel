"""Roadmap risk scoring and repo twin snapshot helpers."""

from __future__ import annotations

import os
import time
from typing import Any


class RoadmapRiskRadar:
    """Generate deterministic risk fields from campaign evidence."""

    PHASE_TEST_HINTS = {
        "questionnaire": ["tests/test_campaign_autonomy_runner.py"],
        "research": ["tests/test_campaign_autonomy_runner.py"],
        "eid": ["tests/test_campaign_autonomy_runner.py"],
        "development": [
            "tests/test_task_packet_executor_contract.py",
            "tests/test_campaign_daemon_manager.py",
        ],
        "analysis_upgrade": ["tests/test_campaign_control_kernel.py"],
    }

    def __init__(self, state_store, event_store, state_ledger) -> None:
        self.state_store = state_store
        self.event_store = event_store
        self.state_ledger = state_ledger

    def analyze(
        self,
        campaign_id: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        telemetry = self.state_store.list_telemetry(campaign_id)
        verification_failures = [
            item
            for item in telemetry
            if item.get("telemetry_kind") == "verification_lane"
            and not bool(item.get("all_passed", False))
        ]
        experiment_lanes = [
            item for item in telemetry if item.get("telemetry_kind") == "experiment_lane"
        ]
        event_count = len(self.event_store.events(run_id=campaign_id, limit=1000))
        watermark = self.state_ledger.delta_watermark()
        changed_path_count = len(list(watermark.get("changed_paths") or []))

        scored_items: list[dict[str, Any]] = []
        for item in items:
            dependency_score = min(1.0, len(list(item.get("depends_on") or [])) / 3.0)
            repo_scope_score = min(1.0, len(list(item.get("repo_scope") or [])) / 3.0)
            event_score = min(1.0, event_count / 25.0)
            failure_score = min(1.0, len(verification_failures) / 3.0)
            churn_score = min(1.0, changed_path_count / 10.0)
            risk_score = round(
                (
                    dependency_score * 0.2
                    + repo_scope_score * 0.15
                    + event_score * 0.15
                    + failure_score * 0.3
                    + churn_score * 0.2
                ),
                3,
            )
            predicted_failure_classes = []
            if dependency_score >= 0.34:
                predicted_failure_classes.append("dependency_ordering")
            if failure_score > 0.0:
                predicted_failure_classes.append("verification_regression")
            if str(item.get("type") or "") == "experiment_lane":
                predicted_failure_classes.append("workspace_isolation")
            if not predicted_failure_classes:
                predicted_failure_classes.append("integration_drift")
            predicted_tests = list(
                dict.fromkeys(
                    self.PHASE_TEST_HINTS.get(
                        str(item.get("phase_id") or ""),
                        ["tests/test_campaign_control_kernel.py"],
                    )
                )
            )
            scored_items.append(
                {
                    "item_id": item.get("item_id"),
                    "title": item.get("title"),
                    "risk_score": risk_score,
                    "risk_level": self._risk_level(risk_score),
                    "blast_radius_score": round(
                        (dependency_score + repo_scope_score + churn_score) / 3.0,
                        3,
                    ),
                    "predicted_tests": predicted_tests,
                    "predicted_failure_classes": predicted_failure_classes,
                    "signals": {
                        "dependency_score": dependency_score,
                        "repo_scope_score": repo_scope_score,
                        "event_score": event_score,
                        "verification_failure_score": failure_score,
                        "churn_score": churn_score,
                        "event_count": event_count,
                        "changed_path_count": changed_path_count,
                        "experiment_lane_count": len(experiment_lanes),
                    },
                }
            )
        scored_items.sort(
            key=lambda item: (-float(item["risk_score"]), str(item["item_id"]))
        )
        return {
            "campaign_id": campaign_id,
            "generated_at": time.time(),
            "watermark": watermark,
            "event_count": event_count,
            "verification_failure_count": len(verification_failures),
            "items": scored_items,
            "summary": {
                "item_count": len(scored_items),
                "high_risk_count": len(
                    [item for item in scored_items if item["risk_level"] == "high"]
                ),
                "top_item": scored_items[0]["item_id"] if scored_items else "",
            },
        }

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.67:
            return "high"
        if score >= 0.34:
            return "medium"
        return "low"


class RepoTwinBuilder:
    """Persist repo twin snapshots at controlled phase transitions."""

    def __init__(self, workspace, event_store, state_ledger) -> None:
        self.workspace = workspace
        self.event_store = event_store
        self.state_ledger = state_ledger

    def capture(
        self,
        campaign_id: str,
        *,
        label: str,
        snapshot: dict[str, Any],
        roadmap_risk: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        twin = {
            "campaign_id": campaign_id,
            "label": label,
            "generated_at": time.time(),
            "workspace_root": self.workspace.root_dir,
            "roadmap_risk_summary": (roadmap_risk or {}).get("summary", {}),
            "snapshot": snapshot,
            "state_ledger": self.state_ledger.delta_watermark(),
            "event_excerpt": self.event_store.events(run_id=campaign_id, limit=50),
        }
        target = os.path.join(
            self.workspace.root_dir,
            "artifacts",
            "telemetry",
            f"repo_twin_{label}.json",
        )
        os.makedirs(os.path.dirname(target), exist_ok=True)
        self.workspace.write_json(
            os.path.relpath(target, self.workspace.root_dir),
            twin,
        )
        twin["path"] = target
        return twin
