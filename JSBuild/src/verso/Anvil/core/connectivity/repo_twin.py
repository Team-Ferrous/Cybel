"""Synthetic repo digital twin for connectivity-aware operations."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional


class ConnectivityRepoTwin:
    """Synthesize a federated operational picture from connectivity primitives."""

    def __init__(
        self,
        *,
        state_ledger,
        presence_service=None,
        ownership_registry=None,
        telemetry=None,
        event_store=None,
        architect_plane=None,
    ) -> None:
        self.state_ledger = state_ledger
        self.presence_service = presence_service
        self.ownership_registry = ownership_registry
        self.telemetry = telemetry
        self.event_store = event_store
        self.architect_plane = architect_plane

    def capture(
        self,
        *,
        label: str,
        campaign_id: str = "",
        roadmap_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        presence = (
            self.presence_service.snapshot() if self.presence_service is not None else {}
        )
        ownership = (
            self.ownership_registry.get_status_snapshot()
            if self.ownership_registry is not None
            else {"total_claimed_files": 0, "file_owners": {}}
        )
        telemetry_summary = (
            self.telemetry.summarize() if self.telemetry is not None else {"event_count": 0}
        )
        architect = (
            self.architect_plane.snapshot(presence=presence)
            if self.architect_plane is not None
            else {}
        )
        watermark = self.state_ledger.delta_watermark()
        events = (
            self.event_store.events(run_id=campaign_id, limit=20)
            if self.event_store is not None and campaign_id
            else []
        )
        blocked_promotions = len(
            [
                event
                for event in events
                if bool(event.get("payload", {}).get("promotion_blocked"))
            ]
        )
        return {
            "label": label,
            "generated_at": time.time(),
            "campaign_id": campaign_id,
            "presence": presence,
            "ownership": ownership,
            "state_ledger": watermark,
            "telemetry": telemetry_summary,
            "architect": architect,
            "roadmap_summary": dict(roadmap_summary or {}),
            "events": events,
            "summary": {
                "peer_count": int(presence.get("peer_count", 0)),
                "claimed_file_count": int(ownership.get("total_claimed_files", 0)),
                "delta_logical_clock": int(watermark.get("logical_clock", 0) or 0),
                "blocked_promotion_count": blocked_promotions,
                "stale_peer_count": len(
                    [
                        peer
                        for peer in list(presence.get("peers") or [])
                        if not bool(peer.get("connected"))
                    ]
                ),
            },
        }
