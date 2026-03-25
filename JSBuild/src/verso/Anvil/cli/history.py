import json
import os
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from core.anvil_db import get_anvil_db
from core.memory.fabric import MemoryFabricStore, MemoryProjector
from core.serialization import SerializableMixin


class ConversationHistory(SerializableMixin):
    """Manages conversation history and persistence."""

    def __init__(
        self,
        history_file: Optional[str] = "history.json",
        messages: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        db_path: Optional[str] = ".anvil/anvil.db",
        persist_db: bool = True,
    ):
        self.history_file = history_file
        self.messages: List[Dict[str, Any]] = messages if messages is not None else []
        self.session_id = session_id or uuid.uuid4().hex
        self.db_path = db_path
        self.persist_db = persist_db and bool(db_path)
        self._start_monotonic = time.monotonic()
        self._db = get_anvil_db(db_path) if self.persist_db else None
        self._memory_fabric = (
            MemoryFabricStore.from_db_path(db_path)
            if self.persist_db and db_path
            else None
        )
        self._memory_projector = MemoryProjector()

        if self._db is not None:
            self._db.ensure_session(self.session_id)

        if history_file and messages is None:
            self.load()

    def load(self) -> None:
        """Load history from DB first, then file fallback."""
        if self._db is not None:
            db_messages = self._db.load_messages(self.session_id)
            if db_messages:
                self.messages = db_messages
                return

        if not self.history_file:
            return

        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.messages = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.messages = []

    def save(self) -> None:
        """Save current history to JSON file for backward compatibility."""
        if not self.history_file:
            return
        try:
            os.makedirs(os.path.dirname(self.history_file) or ".", exist_ok=True)
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.messages, f, indent=2)
        except IOError as e:
            print(f"Failed to save history: {e}")

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Add a message to history and persist audit metadata."""
        wall_clock = datetime.now().astimezone().isoformat()
        monotonic_elapsed_ms = int((time.monotonic() - self._start_monotonic) * 1000)

        message = {
            "role": role,
            "content": content,
            "timestamp": wall_clock,
            "monotonic_elapsed_ms": monotonic_elapsed_ms,
        }
        if metadata:
            message.update(metadata)

        self.messages.append(message)
        self.save()

        if self._db is not None:
            db_metadata = dict(metadata)
            self._db.append_message(
                session_id=self.session_id,
                role=role,
                content=content,
                timestamp=wall_clock,
                monotonic_elapsed_ms=monotonic_elapsed_ms,
                metadata=db_metadata,
                summary_message_id=db_metadata.get("summary_message_id"),
            )
            self._db.log_timeline_event(
                session_id=self.session_id,
                event_type=f"message:{role}",
                wall_clock=wall_clock,
                monotonic_elapsed_ms=monotonic_elapsed_ms,
                payload={
                    "role": role,
                    "chars": len(content or ""),
                    "has_metadata": bool(metadata),
                },
            )
        if self._memory_fabric is not None:
            memory = self._memory_fabric.create_memory(
                memory_kind="conversation_turn",
                payload_json={
                    "role": role,
                    "content": content,
                    "timestamp": wall_clock,
                    "metadata": dict(metadata),
                },
                campaign_id="conversation",
                workspace_id=self.session_id,
                session_id=self.session_id,
                source_system="conversation_history",
                summary_text=f"{role}: {content[:120]}",
                observed_at=time.time(),
            )
            self._memory_fabric.register_alias(
                memory.memory_id,
                "messages",
                f"{self.session_id}:{len(self.messages) - 1}",
                campaign_id="conversation",
            )
            self._memory_projector.project_memory(
                self._memory_fabric,
                memory,
                include_multivector=True,
            )

    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages

    def get_timeline(self, limit: int = 200) -> List[Dict[str, Any]]:
        if self._db is not None:
            events = self._db.get_timeline(self.session_id, limit=limit)
            return [
                {
                    "event_type": event.event_type,
                    "wall_clock": event.wall_clock,
                    "monotonic_elapsed_ms": event.monotonic_elapsed_ms,
                    "payload": event.payload,
                }
                for event in events
            ]

        # Fallback timeline from message metadata.
        output: List[Dict[str, Any]] = []
        for message in self.messages[-limit:]:
            output.append(
                {
                    "event_type": f"message:{message.get('role', 'unknown')}",
                    "wall_clock": message.get("timestamp"),
                    "monotonic_elapsed_ms": message.get("monotonic_elapsed_ms"),
                    "payload": {"chars": len(message.get("content", ""))},
                }
            )
        return output

    def export_audit(self, output_path: str) -> Optional[str]:
        if self._db is None:
            return None
        return self._db.export_audit(self.session_id, output_path)

    def clear(self) -> None:
        self.messages = []
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "history_file": self.history_file,
            "messages": self.messages,
            "session_id": self.session_id,
            "db_path": self.db_path,
            "persist_db": self.persist_db,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            history_file=data.get("history_file"),
            messages=data.get("messages", []),
            session_id=data.get("session_id"),
            db_path=data.get("db_path", ".anvil/anvil.db"),
            persist_db=data.get("persist_db", True),
        )
