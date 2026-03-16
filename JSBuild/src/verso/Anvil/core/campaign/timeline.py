"""Campaign mission timeline assembly and export."""

from __future__ import annotations

import json
import os
from typing import Any


class MissionTimelineAssembler:
    """Assemble one replayable campaign timeline from persistent signals."""

    def __init__(self, state_store, event_store=None) -> None:
        self.state_store = state_store
        self.event_store = event_store

    def assemble(self, campaign_id: str) -> dict[str, Any]:
        transitions = [
            dict(row)
            for row in self.state_store.fetchall(
                """
                SELECT from_state, to_state, reason, payload_json, created_at
                FROM campaign_state_transitions
                WHERE campaign_id = ?
                ORDER BY created_at ASC
                """,
                (campaign_id,),
            )
        ]
        for item in transitions:
            item["payload_json"] = json.loads(item.get("payload_json") or "{}")

        phase_artifacts = self.state_store.list_phase_artifacts(campaign_id)
        task_packets = self.state_store.list_task_packets(campaign_id)
        telemetry = self.state_store.list_telemetry(campaign_id)
        memory_reads = [
            {
                **dict(row),
                "result_memory_ids_json": json.loads(
                    dict(row).get("result_memory_ids_json") or "[]"
                ),
            }
            for row in self.state_store.fetchall(
                """
                SELECT *
                FROM memory_reads
                WHERE campaign_id = ?
                ORDER BY created_at ASC
                """,
                (campaign_id,),
            )
        ]
        read_ids = [item["read_id"] for item in memory_reads]
        feedback = []
        if read_ids:
            rows = self.state_store.fetchall(
                """
                SELECT *
                FROM memory_feedback
                WHERE read_id IN (%s)
                ORDER BY created_at ASC
                """
                % ",".join("?" for _ in read_ids),
                tuple(read_ids),
            )
            feedback = [
                {
                    **dict(row),
                    "outcome_json": json.loads(dict(row).get("outcome_json") or "{}"),
                }
                for row in rows
            ]
        events = (
            self.event_store.events(run_id=campaign_id, limit=1000)
            if self.event_store is not None
            else []
        )
        return {
            "campaign_id": campaign_id,
            "summary": {
                "transition_count": len(transitions),
                "artifact_count": len(phase_artifacts),
                "task_packet_count": len(task_packets),
                "telemetry_count": len(telemetry),
                "memory_read_count": len(memory_reads),
                "memory_feedback_count": len(feedback),
                "event_count": len(events),
            },
            "transitions": transitions,
            "phase_artifacts": phase_artifacts,
            "task_packets": task_packets,
            "telemetry": telemetry,
            "memory_reads": memory_reads,
            "memory_feedback": feedback,
            "events": events,
        }

    def persist(self, campaign_id: str, workspace_root: str) -> dict[str, Any]:
        payload = self.assemble(campaign_id)
        path = os.path.join(workspace_root, "artifacts", "telemetry", "mission_timeline.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        payload["path"] = path
        return payload
