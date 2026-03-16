"""Provider-backed peer transport for cross-instance messaging."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.networking.instance_identity import AnvilInstance


@dataclass
class PeerMessage:
    type: str
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    version: str = "1.0"
    correlation_id: Optional[str] = None


@dataclass
class PeerConnection:
    peer: AnvilInstance
    connected_at: float = field(default_factory=time.time)
    last_message_at: float = field(default_factory=time.time)


class PeerTransportProvider:
    name = "provider"
    requires_polling = False

    def register(self, transport: "PeerTransport") -> None:
        _ = transport

    async def connect(self, transport: "PeerTransport", peer: AnvilInstance) -> None:
        _ = (transport, peer)

    async def send(self, transport: "PeerTransport", peer_id: str, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    def poll(self, transport: "PeerTransport") -> None:
        _ = transport

    def close(self, transport: "PeerTransport") -> None:
        _ = transport


_IN_MEMORY_TRANSPORTS: Dict[str, "PeerTransport"] = {}


class InMemoryTransportProvider(PeerTransportProvider):
    name = "in_memory"

    def register(self, transport: "PeerTransport") -> None:
        _IN_MEMORY_TRANSPORTS[transport.instance.instance_id] = transport

    async def send(
        self, transport: "PeerTransport", peer_id: str, payload: Dict[str, Any]
    ) -> None:
        remote = _IN_MEMORY_TRANSPORTS.get(peer_id)
        if remote is None:
            raise RuntimeError(f"Peer '{peer_id}' is not connected")
        remote._deliver(payload)

    def close(self, transport: "PeerTransport") -> None:
        _IN_MEMORY_TRANSPORTS.pop(transport.instance.instance_id, None)


class FileSystemTransportProvider(PeerTransportProvider):
    name = "filesystem"
    requires_polling = True

    def __init__(self, root_dir: str = ".anvil/transport") -> None:
        self.root_dir = Path(root_dir)
        self.inbox_dir = self.root_dir / "inbox"
        self.inbox_dir.mkdir(parents=True, exist_ok=True)

    def register(self, transport: "PeerTransport") -> None:
        self._inbox_path(transport.instance.instance_id).touch(exist_ok=True)

    async def send(
        self, transport: "PeerTransport", peer_id: str, payload: Dict[str, Any]
    ) -> None:
        envelope = dict(payload)
        envelope.setdefault("message_id", uuid.uuid4().hex)
        with self._inbox_path(peer_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope, sort_keys=True))
            handle.write("\n")

    def poll(self, transport: "PeerTransport") -> None:
        inbox = self._inbox_path(transport.instance.instance_id)
        if not inbox.exists():
            return
        last_offset = transport._provider_state.get("offset", 0)
        with inbox.open("r", encoding="utf-8") as handle:
            handle.seek(last_offset)
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                transport._deliver(payload)
            transport._provider_state["offset"] = handle.tell()

    def _inbox_path(self, peer_id: str) -> Path:
        return self.inbox_dir / f"{peer_id}.jsonl"


def resolve_transport_provider(
    provider: str | None = None,
    *,
    transport_root: str = ".anvil/transport",
) -> PeerTransportProvider:
    selected = str(provider or "in_memory").strip().lower()
    if selected == "filesystem":
        return FileSystemTransportProvider(root_dir=transport_root)
    return InMemoryTransportProvider()


class PeerTransport:
    """Async transport façade with provider-backed delivery."""

    def __init__(
        self,
        instance: AnvilInstance,
        tls_config=None,
        *,
        provider: str | None = None,
        transport_root: str = ".anvil/transport",
        poll_interval_seconds: float = 0.1,
    ):
        self.instance = instance
        self.tls_config = tls_config
        self.connections: Dict[str, PeerConnection] = {}
        self._handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._provider_state: Dict[str, Any] = {}
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.provider = resolve_transport_provider(
            provider,
            transport_root=transport_root,
        )
        self.provider_name = self.provider.name
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.provider.register(self)
        self._start_poll_loop()

    async def connect(self, peer: AnvilInstance):
        self.connections[peer.instance_id] = PeerConnection(peer=peer)
        await self.provider.connect(self, peer)

    async def send(self, peer_id: str, message):
        if isinstance(message, PeerMessage):
            payload = {
                "type": message.type,
                "sender_id": message.sender_id,
                "payload": message.payload,
                "timestamp": message.timestamp,
                "version": message.version,
                "correlation_id": message.correlation_id,
            }
        else:
            payload = dict(message)
            payload.setdefault("sender_id", self.instance.instance_id)
            payload.setdefault("timestamp", time.time())
            payload.setdefault("version", "1.0")
        payload.setdefault("transport_provider", self.provider_name)

        conn = self.connections.get(peer_id)
        if conn is not None:
            conn.last_message_at = time.time()

        await self.provider.send(self, peer_id, payload)

    async def broadcast(self, message, filter_fn=None):
        peer_ids = list(self.connections.keys())
        for peer_id in peer_ids:
            if filter_fn is not None and not filter_fn(peer_id):
                continue
            await self.send(peer_id, message)

    def on_message(self, handler: Callable[[Dict[str, Any]], None]):
        self._handlers.append(handler)

    def sync_peers(self, peers: List[AnvilInstance]) -> None:
        known = {peer.instance_id: peer for peer in peers}
        for peer_id in list(self.connections.keys()):
            if peer_id not in known:
                self.connections.pop(peer_id, None)
        for peer_id, peer in known.items():
            if peer_id == self.instance.instance_id:
                continue
            self.connections.setdefault(peer_id, PeerConnection(peer=peer))

    def _deliver(self, message: Dict[str, Any]) -> None:
        sender_id = message.get("sender_id")
        if sender_id in self.connections:
            self.connections[sender_id].last_message_at = time.time()

        for handler in list(self._handlers):
            handler(message)

    def _start_poll_loop(self) -> None:
        if not self.provider.requires_polling:
            return
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()

        def _loop() -> None:
            while not self._stop.is_set():
                self.provider.poll(self)
                self._stop.wait(self.poll_interval_seconds)

        self._thread = threading.Thread(
            target=_loop,
            name=f"PeerTransport:{self.instance.instance_id}",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.provider.close(self)
        self.connections.clear()
        self._handlers.clear()
