"""Utilities for objects."""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Task:
    """Provide Task support."""
    id: str
    description: str
    target_files: list[str]
    type: str = "refactor"  # refactor, cleanup, migration
    status: str = "pending"  # pending, in_progress, complete, failed
    dependencies: list[str] = field(default_factory=list)  # Task IDs
    risk_score: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass
class Plan:
    """Provide Plan support."""
    id: str
    goal: str
    scope: str
    tasks: list[Task] = field(default_factory=list)
    risk_summary: str = "Unknown"

    def add_task(
        self, description: str, targets: list[str], type: str = "refactor"
    ) -> Task:
        """Handle add task."""
        t = Task(
            id=str(uuid.uuid4())[:8],
            description=description,
            target_files=targets,
            type=type,
        )
        self.tasks.append(t)
        return t
