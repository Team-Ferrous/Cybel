"""Collaboration negotiation primitives."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.collaboration.task_announcer import OverlapResult


@dataclass
class NegotiationResponse:
    proposal_id: str
    status: str
    message: str
    merged_plan: Optional[dict] = None


@dataclass
class MergedPlan:
    plan_id: str
    summary: str
    tasks: List[dict]
    file_assignments: Dict[str, List[str]]
    handoffs: List[dict] = field(default_factory=list)


class CollaborationNegotiator:
    """Creates, receives, and synthesizes collaboration proposals."""

    def __init__(self, event_store=None):
        self.event_store = event_store
        self.proposals: Dict[str, dict] = {}

    def _emit(self, event_type: str, payload: dict) -> None:
        if self.event_store is None:
            return
        try:
            self.event_store.emit(event_type=event_type, payload=payload, source="collaboration")
        except Exception:
            pass

    def propose_collaboration(
        self,
        overlap: OverlapResult,
        local_context: str,
        local_plan: dict,
    ) -> str:
        proposal_id = uuid.uuid4().hex[:12]
        proposal = {
            "proposal_id": proposal_id,
            "created_at": time.time(),
            "overlap": overlap.__dict__,
            "local_context": local_context,
            "local_plan": local_plan,
            "status": "proposed",
        }
        self.proposals[proposal_id] = proposal
        self._emit("collaboration.proposed", proposal)
        return proposal_id

    def receive_proposal(self, proposal: dict) -> NegotiationResponse:
        proposal_id = str(proposal.get("proposal_id") or uuid.uuid4().hex[:12])
        overlap = proposal.get("overlap") or {}
        overlap_type = str(overlap.get("overlap_type") or "complementary")

        if overlap_type == "conflicting":
            response = NegotiationResponse(
                proposal_id=proposal_id,
                status="counter_proposal",
                message="Conflict detected. Recommend split by file ownership boundaries.",
            )
            self._emit("collaboration.counter_proposal", response.__dict__)
            return response

        merged = self.synthesize_plan(
            local_plan=proposal.get("local_plan") or {},
            remote_plan=proposal.get("remote_plan") or {},
            negotiation_context=proposal,
        )
        response = NegotiationResponse(
            proposal_id=proposal_id,
            status="accepted",
            message="Collaboration accepted.",
            merged_plan={
                "plan_id": merged.plan_id,
                "summary": merged.summary,
                "tasks": merged.tasks,
                "file_assignments": merged.file_assignments,
                "handoffs": merged.handoffs,
            },
        )
        self._emit("collaboration.accepted", response.__dict__)
        return response

    def synthesize_plan(
        self,
        local_plan: dict,
        remote_plan: dict,
        negotiation_context: dict,
    ) -> MergedPlan:
        local_tasks = list(local_plan.get("tasks") or [])
        remote_tasks = list(remote_plan.get("tasks") or [])

        seen = set()
        merged_tasks: List[dict] = []
        for task in local_tasks + remote_tasks:
            task_id = str(task.get("id") or task.get("task") or uuid.uuid4().hex[:8])
            key = (task_id, str(task.get("task") or task.get("instruction") or ""))
            if key in seen:
                continue
            seen.add(key)
            merged_tasks.append(dict(task))

        file_assignments: Dict[str, List[str]] = {
            "local": list(local_plan.get("files") or []),
            "remote": list(remote_plan.get("files") or []),
        }

        overlap = negotiation_context.get("overlap") or {}
        shared_files = sorted(
            set(overlap.get("local_files") or []) | set(overlap.get("remote_files") or [])
        )
        if shared_files:
            file_assignments["shared"] = shared_files

        handoffs = []
        if shared_files:
            handoffs.append(
                {
                    "type": "shared_files",
                    "files": shared_files,
                    "rule": "Coordinate before write operations.",
                }
            )

        return MergedPlan(
            plan_id=uuid.uuid4().hex[:12],
            summary="Merged collaboration plan with duplicate tasks removed.",
            tasks=merged_tasks,
            file_assignments=file_assignments,
            handoffs=handoffs,
        )

    @staticmethod
    def score_semantic_conflict(local_plan: dict, remote_plan: dict) -> dict:
        local_symbols = set(local_plan.get("context_symbols") or [])
        remote_symbols = set(remote_plan.get("context_symbols") or [])
        local_files = set(local_plan.get("files") or local_plan.get("context_files") or [])
        remote_files = set(remote_plan.get("files") or remote_plan.get("context_files") or [])
        symbol_overlap = sorted(local_symbols & remote_symbols)
        file_overlap = sorted(local_files & remote_files)
        score = 0.0
        if symbol_overlap:
            score += 0.7
        if file_overlap:
            score += 0.3
        return {
            "score": round(min(1.0, score), 3),
            "symbol_overlap": symbol_overlap,
            "file_overlap": file_overlap,
            "risk_level": "high" if score >= 0.7 else ("medium" if score > 0 else "low"),
        }
