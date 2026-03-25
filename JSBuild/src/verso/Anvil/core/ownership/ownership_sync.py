"""Cross-instance ownership synchronization protocol."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Dict

from core.ownership.ownership_crdt import LWWEntry, OwnershipCRDT


class OwnershipSyncProtocol:
    """Synchronizes ownership state across peers using CRDT deltas."""

    OWNERSHIP_SNAPSHOT_REQUEST = "OWNERSHIP_SNAPSHOT_REQUEST"
    OWNERSHIP_SNAPSHOT = "OWNERSHIP_SNAPSHOT"
    OWNERSHIP_DELTA = "OWNERSHIP_DELTA"
    OWNERSHIP_CLAIM = "OWNERSHIP_CLAIM"
    OWNERSHIP_RELEASE = "OWNERSHIP_RELEASE"

    def __init__(self, crdt: OwnershipCRDT, transport, message_bus=None):
        self.crdt = crdt
        self.transport = transport
        self.message_bus = message_bus
        self.peer_clocks: Dict[str, int] = {}
        self.transport.on_message(self._on_transport_message)

    @staticmethod
    def _run_async(coro) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    async def on_peer_connected(self, peer_id: str) -> None:
        await self.transport.send(
            peer_id,
            {
                "type": self.OWNERSHIP_SNAPSHOT_REQUEST,
                "sender_id": self.crdt.instance_id,
                "payload": {"clock": self.crdt.clock.value},
            },
        )

    async def on_ownership_change(self, entry: LWWEntry) -> None:
        message_type = self.OWNERSHIP_RELEASE if entry.is_tombstone else self.OWNERSHIP_CLAIM
        await self.transport.broadcast(
            {
                "type": message_type,
                "sender_id": self.crdt.instance_id,
                "payload": entry.to_dict(),
            }
        )

    async def handle_remote_delta(self, peer_id: str, delta: Dict) -> None:
        entries = {
            path: LWWEntry.from_dict(entry)
            for path, entry in (delta or {}).items()
        }
        self.crdt.merge(entries)
        self.peer_clocks[peer_id] = max(
            self.peer_clocks.get(peer_id, 0),
            max((entry.timestamp for entry in entries.values()), default=0),
        )
        self._publish_update("ownership.delta", peer_id, entries)

    async def handle_remote_claim(self, peer_id: str, claim: Dict) -> None:
        entry = LWWEntry.from_dict(claim)
        self.crdt.merge({entry.file_path: entry})
        self.peer_clocks[peer_id] = max(self.peer_clocks.get(peer_id, 0), entry.timestamp)
        event = "ownership.release" if entry.is_tombstone else "ownership.claim"
        self._publish_update(event, peer_id, {entry.file_path: entry})

    def _publish_update(self, topic: str, peer_id: str, entries: Dict[str, LWWEntry]) -> None:
        if self.message_bus is None:
            return
        try:
            self.message_bus.publish(
                topic="ownership.claims",
                sender=f"peer:{peer_id}",
                payload={
                    "topic": topic,
                    "peer_id": peer_id,
                    "entries": {path: entry.to_dict() for path, entry in entries.items()},
                },
            )
        except Exception:
            pass

    def _on_transport_message(self, message) -> None:
        if inspect.isawaitable(message):
            self._run_async(self._handle_message(awaitable=message))
            return
        self._run_async(self._handle_message(message=message))

    async def _handle_message(self, message=None, awaitable=None) -> None:
        if awaitable is not None:
            message = await awaitable
        if message is None:
            return

        msg_type = message.get("type")
        peer_id = message.get("sender_id")
        payload = message.get("payload") or {}

        if msg_type == self.OWNERSHIP_SNAPSHOT_REQUEST:
            await self.transport.send(
                peer_id,
                {
                    "type": self.OWNERSHIP_SNAPSHOT,
                    "sender_id": self.crdt.instance_id,
                    "payload": {
                        "clock": self.crdt.clock.value,
                        "state": {
                            path: entry.to_dict() for path, entry in self.crdt.state.items()
                        },
                    },
                },
            )
            return

        if msg_type == self.OWNERSHIP_SNAPSHOT:
            state = {
                path: LWWEntry.from_dict(entry)
                for path, entry in (payload.get("state") or {}).items()
            }
            self.crdt.merge(state)
            self.peer_clocks[peer_id] = int(payload.get("clock", 0))
            self._publish_update("ownership.snapshot", peer_id, state)
            return

        if msg_type == self.OWNERSHIP_DELTA:
            await self.handle_remote_delta(peer_id, payload)
            return

        if msg_type in {self.OWNERSHIP_CLAIM, self.OWNERSHIP_RELEASE}:
            await self.handle_remote_claim(peer_id, payload)
            return

    def snapshot_json(self) -> str:
        return json.dumps(
            {path: entry.to_dict() for path, entry in self.crdt.state.items()},
            sort_keys=True,
        )
