from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class TaskType(Enum):
    RESEARCH = "RESEARCH"
    IMPLEMENTATION = "IMPLEMENTATION"
    VERIFICATION = "VERIFICATION"
    REASONING = "REASONING"


@dataclass
class TaskUnit:
    """A discrete unit of work within the agentic loop."""

    instruction: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    type: TaskType = TaskType.IMPLEMENTATION
    context_files: List[str] = field(default_factory=list)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # List of IDs
    semantic_context: Dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED, BLOCKED
    result: Optional[Dict[str, Any]] = None
    assigned_agent_id: Optional[str] = None
    owned_files: List[str] = field(default_factory=list)
    read_files: List[str] = field(default_factory=list)
    phase_id: Optional[str] = None
    ownership_workset_id: Optional[str] = None


class TaskGraph:
    """Manages a collection of TaskUnits and their dependencies."""

    def __init__(self):
        self.tasks: Dict[str, TaskUnit] = {}

    def add_task(self, task: TaskUnit):
        self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[TaskUnit]:
        return self.tasks.get(task_id)

    def get_ready_tasks(self) -> List[TaskUnit]:
        """Returns tasks whose dependencies are all COMPLETED."""
        ready = []
        for task in self.tasks.values():
            if task.status != "PENDING":
                continue

            deps_satisfied = True
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status != "COMPLETED":
                    deps_satisfied = False
                    break

            if deps_satisfied:
                ready.append(task)
        return ready

    def is_complete(self) -> bool:
        return all(t.status == "COMPLETED" for t in self.tasks.values())

    def to_json(self) -> Dict[str, Any]:
        return {
            "tasks": {
                tid: {
                    "id": t.id,
                    "parent_id": t.parent_id,
                    "type": t.type.value,
                    "instruction": t.instruction,
                    "context_files": t.context_files,
                    "output_schema": t.output_schema,
                    "dependencies": t.dependencies,
                    "semantic_context": t.semantic_context,
                    "status": t.status,
                    "result": t.result,
                    "assigned_agent_id": t.assigned_agent_id,
                    "owned_files": t.owned_files,
                    "read_files": t.read_files,
                    "phase_id": t.phase_id,
                    "ownership_workset_id": t.ownership_workset_id,
                }
                for tid, t in self.tasks.items()
            }
        }
