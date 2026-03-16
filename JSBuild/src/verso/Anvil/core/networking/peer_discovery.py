"""Peer discovery for cross-instance collaboration."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from core.networking.instance_identity import AnvilInstance


class MDNSDiscovery:
    """Placeholder mDNS discovery for environments without zeroconf dependency."""

    SERVICE_TYPE = "_anvil._tcp.local."

    def register(self, instance: AnvilInstance) -> None:
        _ = instance

    def browse(self) -> List[AnvilInstance]:
        return []


class RendezvousDiscovery:
    """Placeholder rendezvous discovery implementation."""

    def __init__(self, rendezvous_url: Optional[str] = None):
        self.rendezvous_url = rendezvous_url

    def register(self, instance: AnvilInstance) -> None:
        _ = instance

    def browse(self) -> List[AnvilInstance]:
        return []


class FileSystemDiscovery:
    """Shared-filesystem peer discovery using ``.anvil/peers/*.json`` records."""

    def __init__(self, instance: AnvilInstance, peers_dir: str = ".anvil/peers", stale_after_seconds: int = 60):
        self.instance = instance
        self.peers_dir = Path(peers_dir)
        self.stale_after_seconds = int(stale_after_seconds)
        self.peers_dir.mkdir(parents=True, exist_ok=True)

    def _peer_file(self, instance_id: str) -> Path:
        return self.peers_dir / f"{instance_id}.json"

    def announce(self) -> None:
        payload = self.instance.to_dict()
        payload["last_seen"] = time.time()
        self._peer_file(self.instance.instance_id).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def cleanup_stale(self) -> None:
        now = time.time()
        for path in self.peers_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                age = now - float(payload.get("last_seen", 0.0))
                if age > self.stale_after_seconds:
                    path.unlink(missing_ok=True)
            except Exception:
                path.unlink(missing_ok=True)

    def browse(self) -> List[AnvilInstance]:
        self.cleanup_stale()
        peers: List[AnvilInstance] = []
        for path in self.peers_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                peer = AnvilInstance.from_dict(payload)
                if peer.instance_id == self.instance.instance_id:
                    continue
                peers.append(peer)
            except Exception:
                continue
        return peers


class PeerDiscovery:
    """Multi-strategy peer discovery with graceful fallback."""

    def __init__(
        self,
        instance: AnvilInstance,
        method: str = "auto",
        shared_peers_dir: str = ".anvil/peers",
        rendezvous_url: Optional[str] = None,
        heartbeat_interval_seconds: int = 10,
    ):
        self.instance = instance
        self.method = method
        self.peers: Dict[str, AnvilInstance] = {}
        self.heartbeat_interval_seconds = int(heartbeat_interval_seconds)

        self.mdns = MDNSDiscovery()
        self.filesystem = FileSystemDiscovery(instance=instance, peers_dir=shared_peers_dir)
        self.rendezvous = RendezvousDiscovery(rendezvous_url=rendezvous_url)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _discover_once(self) -> None:
        discovered: List[AnvilInstance] = []

        if self.method in {"auto", "mdns"}:
            self.mdns.register(self.instance)
            discovered.extend(self.mdns.browse())

        if self.method in {"auto", "filesystem"}:
            self.filesystem.announce()
            discovered.extend(self.filesystem.browse())

        if self.method in {"auto", "rendezvous"}:
            self.rendezvous.register(self.instance)
            discovered.extend(self.rendezvous.browse())

        merged = {
            peer.instance_id: peer
            for peer in discovered
            if peer.instance_id != self.instance.instance_id
        }
        self.peers = merged

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                self._discover_once()
                self._stop.wait(self.heartbeat_interval_seconds)

        self._thread = threading.Thread(target=_loop, daemon=True, name="PeerDiscovery")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def refresh(self) -> None:
        self._discover_once()

    def get_peers(self, same_project_only: bool = True) -> List[AnvilInstance]:
        peers = list(self.peers.values())
        if not same_project_only:
            return peers
        return [peer for peer in peers if peer.project_hash == self.instance.project_hash]

    def status(self) -> Dict[str, object]:
        peers = self.get_peers(same_project_only=True)
        return {
            "method": self.method,
            "peer_count": len(peers),
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "peers": [peer.presence_summary() for peer in peers],
        }
