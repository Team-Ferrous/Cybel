"""Deterministic architect arbitration for overlapping repo work."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ArchitectDecision:
    decision_id: str
    leader_id: str
    council: List[str]
    merged_execution_contract: Dict[str, Any]
    issued_at: float


class ArchitectPlane:
    """Elect a local architect and synthesize one binding execution contract."""

    def __init__(self, *, instance_id: str, event_store=None, state_ledger=None) -> None:
        self.instance_id = instance_id
        self.event_store = event_store
        self.state_ledger = state_ledger

    def elect_leader(
        self,
        peers: Iterable[Dict[str, Any]],
        *,
        campaign_id: str = "",
    ) -> Dict[str, Any]:
        candidates = [
            self._candidate_rank(peer, campaign_id=campaign_id)
            for peer in peers
            if str(peer.get("instance_id") or "").strip()
        ]
        if not any(candidate["instance_id"] == self.instance_id for candidate in candidates):
            candidates.append(
                self._candidate_rank(
                    {"instance_id": self.instance_id, "connected": True},
                    campaign_id=campaign_id,
                )
            )
        ordered = sorted(
            candidates,
            key=lambda item: (
                -item.get("effective_score", item["score"]),
                item["staleness_penalty"],
                item["instance_id"],
            ),
        )
        leader = ordered[0]
        result = {
            "leader_id": leader["instance_id"],
            "campaign_id": campaign_id,
            "council": [item["instance_id"] for item in ordered[:3]],
            "candidates": ordered,
        }
        self._emit("architect.elected", result)
        return result

    def arbitrate(
        self,
        *,
        local_plan: Dict[str, Any],
        remote_plans: List[Dict[str, Any]],
        presence: Optional[Dict[str, Any]] = None,
        ownership_snapshot: Optional[Dict[str, Any]] = None,
        campaign_id: str = "",
    ) -> ArchitectDecision:
        leader = self.elect_leader(list((presence or {}).get("peers") or []), campaign_id=campaign_id)
        all_plans = [dict(local_plan), *[dict(plan) for plan in remote_plans]]
        merged_tasks: List[Dict[str, Any]] = []
        seen = set()
        claimed_files = {
            file_path
            for file_path, records in dict(
                (ownership_snapshot or {}).get("file_owners") or {}
            ).items()
            if records
        }
        handoffs: List[Dict[str, Any]] = []
        file_assignments: Dict[str, List[str]] = {}

        for plan in all_plans:
            actor = str(plan.get("instance_id") or plan.get("agent_id") or "unknown")
            candidate_files = sorted(
                {
                    str(item)
                    for item in list(plan.get("files") or plan.get("context_files") or [])
                    if str(item).strip()
                }
            )
            safe_files = [path for path in candidate_files if path not in claimed_files]
            blocked_files = [path for path in candidate_files if path in claimed_files]
            if safe_files:
                file_assignments[actor] = safe_files
            if blocked_files:
                handoffs.append(
                    {
                        "actor": actor,
                        "files": blocked_files,
                        "instruction": "defer_to_existing_owner",
                    }
                )
            for task in list(plan.get("tasks") or []):
                key = (
                    str(task.get("id") or ""),
                    str(task.get("task") or task.get("instruction") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_tasks.append(dict(task))

        contract = {
            "leader_id": leader["leader_id"],
            "campaign_id": campaign_id,
            "task_count": len(merged_tasks),
            "tasks": merged_tasks,
            "file_assignments": file_assignments,
            "handoffs": handoffs,
            "ownership_rebalance_required": bool(handoffs),
        }
        decision = ArchitectDecision(
            decision_id=f"arch_{uuid.uuid4().hex[:12]}",
            leader_id=leader["leader_id"],
            council=list(leader["council"]),
            merged_execution_contract=contract,
            issued_at=time.time(),
        )
        self._emit(
            "architect.arbitrated",
            {
                "decision_id": decision.decision_id,
                "leader_id": decision.leader_id,
                "campaign_id": campaign_id,
                "task_count": len(merged_tasks),
                "ownership_rebalance_required": bool(handoffs),
            },
        )
        return decision

    def snapshot(self, *, presence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        peers = list((presence or {}).get("peers") or [])
        leader = self.elect_leader(peers) if peers else {"leader_id": self.instance_id, "council": [self.instance_id], "candidates": []}
        return {
            "instance_id": self.instance_id,
            "leader_id": leader["leader_id"],
            "council": leader["council"],
            "peer_count": len(peers),
            "watermark": self.state_ledger.delta_watermark() if self.state_ledger is not None else {},
        }

    def _candidate_rank(
        self, peer: Dict[str, Any], *, campaign_id: str = ""
    ) -> Dict[str, Any]:
        instance_id = str(peer.get("instance_id") or "")
        same_campaign = str(peer.get("campaign_id") or "") == campaign_id if campaign_id else True
        verification = str(peer.get("verification_state") or "")
        staleness_penalty = max(
            0.0,
            time.time() - float(peer.get("last_seen") or time.time()),
        )
        score = 0.0
        score += 2.0 if peer.get("connected", True) else 0.0
        score += 1.5 if same_campaign else 0.0
        score += 1.0 if verification in {"ready", "verified", "promotable"} else 0.0
        score += float(peer.get("analysis_capacity") or 0.0)
        score += float(peer.get("verification_capacity") or 0.0)
        score -= min(1.0, float(peer.get("active_claim_count") or 0.0) / 10.0)
        effective_score = score - min(2.5, staleness_penalty / 60.0)
        return {
            "instance_id": instance_id,
            "score": round(score, 3),
            "effective_score": round(effective_score, 3),
            "staleness_penalty": round(staleness_penalty, 3),
            "campaign_id": str(peer.get("campaign_id") or ""),
        }

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.event_store is None:
            return
        try:
            self.event_store.emit(event_type=event_type, payload=payload, source="ArchitectPlane")
        except Exception:
            pass
