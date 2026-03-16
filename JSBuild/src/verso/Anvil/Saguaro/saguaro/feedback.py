"""Utilities for feedback."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Provide FeedbackEntry support."""
    query_text: str
    timestamp: float
    context_id: str  # ID from ContextItem
    action: str  # "used", "ignored", "rejected"
    outcome: str | None = None  # "success", "failure"


@dataclass
class FeedbackSession:
    """Provide FeedbackSession support."""
    id: str
    timestamp: float
    entries: list[FeedbackEntry]

    def to_json(self) -> str:
        """Handle to json."""
        return json.dumps(asdict(self), indent=2)


class FeedbackStore:
    """Provide FeedbackStore support."""
    def __init__(self, saguaro_dir: str) -> None:
        """Initialize the instance."""
        self.feedback_dir = os.path.join(saguaro_dir, "feedback")
        os.makedirs(self.feedback_dir, exist_ok=True)

    def log_feedback(
        self, query: str, context_items: list[dict[str, Any]], outcome: str = "unknown"
    ) -> str:
        """Logs a feedback session.
        context_items should be a list of dicts with keys: {'id', 'action'}.
        """
        entries = []
        ts = time.time()

        for item in context_items:
            entries.append(
                FeedbackEntry(
                    query_text=query,
                    timestamp=ts,
                    context_id=item.get("id", "unknown"),
                    action=item.get("action", "unknown"),
                    outcome=outcome,
                )
            )

        session_id = f"fb_{int(ts)}_{os.getpid()}"
        session = FeedbackSession(id=session_id, timestamp=ts, entries=entries)

        self._save(session)
        return session_id

    def _save(self, session: FeedbackSession) -> None:
        path = os.path.join(self.feedback_dir, f"{session.id}.json")
        with open(path, "w") as f:
            f.write(session.to_json())

    def get_stats(self) -> dict[str, Any]:
        """Returns aggregate stats on usage."""
        total_sessions = 0
        used_count = 0
        ignored_count = 0

        if not os.path.exists(self.feedback_dir):
            return {"total": 0}

        for fname in os.listdir(self.feedback_dir):
            if fname.endswith(".json"):
                total_sessions += 1
                try:
                    with open(os.path.join(self.feedback_dir, fname)) as f:
                        data = json.load(f)
                        for entry in data.get("entries", []):
                            if entry.get("action") == "used":
                                used_count += 1
                            elif entry.get("action") == "ignored":
                                ignored_count += 1
                except Exception:
                    pass

        return {
            "total_sessions": total_sessions,
            "items_used": used_count,
            "items_ignored": ignored_count,
            "utilization_rate": (
                used_count / (used_count + ignored_count)
                if (used_count + ignored_count) > 0
                else 0
            ),
        }
