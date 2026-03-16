"""Repo-scoped presence envelopes for collaborative Anvil instances."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional


class RepoPresenceService:
    """Maintain local repo presence and synthesize peer awareness snapshots."""

    def __init__(
        self,
        *,
        instance_registry,
        peer_discovery=None,
        peer_transport=None,
        ownership_registry=None,
        campaign_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        capability_getter: Optional[Callable[[], Dict[str, Any]]] = None,
        trust_zone: str = "internal",
    ) -> None:
        self.instance_registry = instance_registry
        self.peer_discovery = peer_discovery
        self.peer_transport = peer_transport
        self.ownership_registry = ownership_registry
        self.campaign_getter = campaign_getter
        self.capability_getter = capability_getter
        self.trust_zone = trust_zone

    def refresh(
        self,
        *,
        campaign_id: str = "",
        phase_id: str = "",
        lane_id: str = "",
        verification_state: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ownership = self._ownership_snapshot()
        capability = self._capability_snapshot()
        campaign = self._campaign_snapshot()
        verification = verification_state or str(
            campaign.get("verification_state")
            or ("ready" if capability["verification_capacity"] >= 0.5 else "warming")
        )
        identity = self.instance_registry.update_presence(
            trust_zone=self.trust_zone,
            discovery_method=getattr(self.peer_discovery, "method", "filesystem"),
            transport_provider=getattr(self.peer_transport, "provider_name", "in_memory"),
            current_campaign_id=campaign_id or str(campaign.get("campaign_id") or ""),
            current_phase_id=phase_id or str(campaign.get("phase_id") or ""),
            lane_id=lane_id or str(campaign.get("lane_id") or ""),
            active_claim_count=int(ownership.get("total_claimed_files", 0)),
            verification_state=verification,
            analysis_capacity=float(capability["analysis_capacity"]),
            verification_capacity=float(capability["verification_capacity"]),
            runtime_symbol_digest=str(capability.get("runtime_symbol_digest") or ""),
            metadata={
                "last_presence_refresh": str(time.time()),
                **dict(metadata or {}),
            },
        )
        if self.peer_discovery is not None:
            self.peer_discovery.instance = identity
            self.peer_discovery.refresh()
        peers = (
            self.peer_discovery.get_peers(same_project_only=True)
            if self.peer_discovery is not None
            else []
        )
        if self.peer_transport is not None:
            self.peer_transport.sync_peers(peers)
        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        local = self.instance_registry.identity
        peers = (
            self.peer_discovery.get_peers(same_project_only=True)
            if self.peer_discovery is not None
            else []
        )
        peer_rows = [self._peer_row(peer) for peer in peers]
        peer_rows.sort(
            key=lambda row: (
                row["promotable"] is False,
                row["phase_id"] or "zzz",
                -float(row["analysis_capacity"]),
                row["instance_id"],
            )
        )
        return {
            "generated_at": time.time(),
            "local": self._peer_row(local, include_connection=False),
            "peer_count": len(peer_rows),
            "transport_provider": getattr(self.peer_transport, "provider_name", "none"),
            "discovery_method": getattr(self.peer_discovery, "method", "none"),
            "peers": peer_rows,
        }

    def build_prompt_context(self) -> Dict[str, Any]:
        snapshot = self.snapshot()
        peers = snapshot["peers"]
        promotable = [peer for peer in peers if peer["promotable"]]
        return {
            "local_campaign_id": snapshot["local"]["campaign_id"],
            "local_phase_id": snapshot["local"]["phase_id"],
            "local_claim_count": snapshot["local"]["active_claim_count"],
            "peer_count": snapshot["peer_count"],
            "promotable_peer_count": len(promotable),
            "transport_provider": snapshot["transport_provider"],
            "trust_zone": snapshot["local"]["trust_zone"],
            "connected_peers": [peer["instance_id"] for peer in peers if peer["connected"]],
        }

    def _peer_row(self, peer, *, include_connection: bool = True) -> Dict[str, Any]:
        connected = False
        if include_connection and self.peer_transport is not None:
            connected = peer.instance_id in self.peer_transport.connections
        return {
            "instance_id": peer.instance_id,
            "hostname": peer.hostname,
            "user": peer.user,
            "listen_address": peer.listen_address,
            "campaign_id": getattr(peer, "current_campaign_id", ""),
            "phase_id": getattr(peer, "current_phase_id", ""),
            "lane_id": getattr(peer, "lane_id", ""),
            "active_claim_count": int(getattr(peer, "active_claim_count", 0)),
            "trust_zone": getattr(peer, "trust_zone", "internal"),
            "transport_provider": getattr(peer, "transport_provider", "unknown"),
            "verification_state": getattr(peer, "verification_state", "unknown"),
            "analysis_capacity": float(getattr(peer, "analysis_capacity", 0.0)),
            "verification_capacity": float(getattr(peer, "verification_capacity", 0.0)),
            "runtime_symbol_digest": getattr(peer, "runtime_symbol_digest", ""),
            "repo_branch": getattr(peer, "repo_branch", ""),
            "repo_dirty": bool(getattr(peer, "repo_dirty", False)),
            "connected": connected,
            "promotable": bool(getattr(peer, "promotable", False)),
            "last_seen": float(getattr(peer, "last_seen", 0.0)),
        }

    def _ownership_snapshot(self) -> Dict[str, Any]:
        if self.ownership_registry is None:
            return {"total_claimed_files": 0}
        try:
            return dict(self.ownership_registry.get_status_snapshot() or {})
        except Exception:
            return {"total_claimed_files": 0}

    def _campaign_snapshot(self) -> Dict[str, Any]:
        if not callable(self.campaign_getter):
            return {}
        try:
            return dict(self.campaign_getter() or {})
        except Exception:
            return {}

    def _capability_snapshot(self) -> Dict[str, Any]:
        default = {
            "analysis_capacity": 0.5,
            "verification_capacity": 0.5,
            "runtime_symbol_digest": "",
        }
        if not callable(self.capability_getter):
            return default
        try:
            payload = dict(self.capability_getter() or {})
        except Exception:
            return default
        return {
            **default,
            **payload,
        }
