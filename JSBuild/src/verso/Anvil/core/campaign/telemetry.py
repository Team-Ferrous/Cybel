"""Campaign telemetry contracts and execution records."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TelemetryContract:
    required_metrics: List[str] = field(default_factory=lambda: ["wall_time"])
    optional_metrics: List[str] = field(default_factory=list)


class CampaignTelemetry:
    """Record material execution telemetry into the campaign store."""

    def __init__(self, state_store, campaign_id: str):
        self.state_store = state_store
        self.campaign_id = campaign_id

    def start_span(
        self,
        *,
        telemetry_kind: str,
        task_packet_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        return {
            "span_id": str(uuid.uuid4()),
            "telemetry_kind": telemetry_kind,
            "task_packet_id": task_packet_id,
            "metadata": dict(metadata or {}),
            "started_at": time.time(),
        }

    def finish_span(
        self,
        span: Dict[str, object],
        *,
        metrics: Dict[str, object],
        status: str = "completed",
    ) -> Dict[str, object]:
        completed_at = time.time()
        payload = {
            **span,
            "completed_at": completed_at,
            "status": status,
            "duration_seconds": completed_at - float(span["started_at"]),
            "metrics": dict(metrics),
        }
        self.state_store.record_telemetry(
            self.campaign_id,
            task_packet_id=span.get("task_packet_id"),
            telemetry_kind=str(span["telemetry_kind"]),
            payload=payload,
        )
        return payload

    def summarize(self) -> Dict[str, object]:
        events = self.state_store.list_telemetry(self.campaign_id)
        durations = []
        kinds: Dict[str, int] = {}
        for event in events:
            durations.append(float(event.get("duration_seconds", 0.0)))
            kind = str(event.get("telemetry_kind", "unknown"))
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "event_count": len(events),
            "kinds": kinds,
            "total_duration_seconds": sum(durations),
        }
