"""Utilities for graph."""

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class TaskNode:
    """Provide TaskNode support."""
    id: str
    description: str
    status: str  # pending, running, completed, failed
    dependencies: list[str]
    assigned_agent: str | None = None


class TaskGraph:
    """Persists refactor plans as DO-graphs (Dependency/Operation)
    that survive agent restarts.
    """

    def __init__(self, persistence_path: str = ".saguaro/task_graph.json") -> None:
        """Initialize the instance."""
        self.persistence_path = persistence_path
        self.nodes: dict[str, TaskNode] = {}
        self._load()

    def add_task(self, task: TaskNode) -> None:
        """Handle add task."""
        self.nodes[task.id] = task
        self._save()

    def get_task(self, task_id: str) -> TaskNode | None:
        """Get task."""
        return self.nodes.get(task_id)

    def get_ready_tasks(self) -> list[TaskNode]:
        """Returns tasks whose dependencies are met."""
        ready = []
        for task in self.nodes.values():
            if task.status != "pending":
                continue
            deps_met = all(
                self.nodes[d].status == "completed"
                for d in task.dependencies
                if d in self.nodes
            )
            if deps_met:
                ready.append(task)
        return ready

    def mark_complete(self, task_id: str) -> None:
        """Handle mark complete."""
        if task_id in self.nodes:
            self.nodes[task_id].status = "completed"
            self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
        data = {k: asdict(v) for k, v in self.nodes.items()}
        with open(self.persistence_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path) as f:
                data = json.load(f)
                for k, v in data.items():
                    self.nodes[k] = TaskNode(**v)
        except Exception:
            self.nodes = {}
