"""CRDT primitives for distributed file ownership."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LWWEntry:
    file_path: str
    owner_agent_id: str
    owner_instance_id: str
    mode: str
    timestamp: int
    instance_id: str
    is_tombstone: bool = False
    updated_at: float = field(default_factory=time.time)

    def __gt__(self, other: "LWWEntry") -> bool:
        if self.timestamp != other.timestamp:
            return self.timestamp > other.timestamp
        return self.instance_id > other.instance_id

    def to_dict(self) -> Dict[str, object]:
        return {
            "file_path": self.file_path,
            "owner_agent_id": self.owner_agent_id,
            "owner_instance_id": self.owner_instance_id,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "instance_id": self.instance_id,
            "is_tombstone": self.is_tombstone,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "LWWEntry":
        return LWWEntry(
            file_path=str(data["file_path"]),
            owner_agent_id=str(data.get("owner_agent_id", "")),
            owner_instance_id=str(data.get("owner_instance_id", "")),
            mode=str(data.get("mode", "exclusive")),
            timestamp=int(data.get("timestamp", 0)),
            instance_id=str(data.get("instance_id", "")),
            is_tombstone=bool(data.get("is_tombstone", False)),
            updated_at=float(data.get("updated_at", time.time())),
        )


class LamportClock:
    """Lamport logical clock for causal ordering."""

    def __init__(self):
        self.value = 0

    def increment(self) -> int:
        self.value += 1
        return self.value

    def update(self, received: int) -> int:
        self.value = max(self.value, int(received)) + 1
        return self.value


class OwnershipCRDT:
    """LWW-register CRDT map keyed by file path."""

    def __init__(self, instance_id: str):
        self.instance_id = instance_id
        self.clock = LamportClock()
        self.state: Dict[str, LWWEntry] = {}

    def claim(self, file_path: str, agent_id: str, mode: str = "exclusive") -> LWWEntry:
        existing = self.state.get(file_path)
        if existing is not None and existing.is_tombstone:
            raise RuntimeError(
                f"Cannot reclaim '{file_path}' until tombstone GC completes."
            )

        timestamp = self.clock.increment()
        entry = LWWEntry(
            file_path=file_path,
            owner_agent_id=agent_id,
            owner_instance_id=self.instance_id,
            mode=mode,
            timestamp=timestamp,
            instance_id=self.instance_id,
            is_tombstone=False,
        )
        self.state[file_path] = entry
        return entry

    def merge(self, remote_state: Dict[str, LWWEntry]) -> None:
        for path, remote_entry in remote_state.items():
            if isinstance(remote_entry, dict):
                remote_entry = LWWEntry.from_dict(remote_entry)
            local_entry = self.state.get(path)
            if local_entry is None or remote_entry > local_entry:
                self.state[path] = remote_entry
                self.clock.update(remote_entry.timestamp)

    def release(self, file_path: str, agent_id: str) -> LWWEntry:
        timestamp = self.clock.increment()
        current = self.state.get(file_path)
        owner_instance_id = self.instance_id
        if current is not None:
            owner_instance_id = current.owner_instance_id

        tombstone = LWWEntry(
            file_path=file_path,
            owner_agent_id=agent_id,
            owner_instance_id=owner_instance_id,
            mode="released",
            timestamp=timestamp,
            instance_id=self.instance_id,
            is_tombstone=True,
        )
        self.state[file_path] = tombstone
        return tombstone

    def diff(self, other_clock: int) -> Dict[str, LWWEntry]:
        baseline = int(other_clock)
        return {
            path: entry
            for path, entry in self.state.items()
            if entry.timestamp > baseline
        }

    def snapshot(self) -> bytes:
        payload = {
            "instance_id": self.instance_id,
            "clock": self.clock.value,
            "state": {path: entry.to_dict() for path, entry in self.state.items()},
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def load_snapshot(self, snapshot: bytes) -> None:
        payload = json.loads(snapshot.decode("utf-8"))
        entries = {
            path: LWWEntry.from_dict(entry)
            for path, entry in (payload.get("state") or {}).items()
        }
        self.merge(entries)
        self.clock.value = max(self.clock.value, int(payload.get("clock", 0)))

    def gc_tombstones(self, max_age_seconds: Optional[float] = None) -> None:
        now = time.time()
        if max_age_seconds is None:
            stale = [path for path, entry in self.state.items() if entry.is_tombstone]
        else:
            stale = [
                path
                for path, entry in self.state.items()
                if entry.is_tombstone and (now - entry.updated_at) > max_age_seconds
            ]
        for path in stale:
            self.state.pop(path, None)
