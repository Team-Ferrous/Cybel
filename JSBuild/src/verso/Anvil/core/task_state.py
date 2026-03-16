"""
Task State Management - Tracks task boundaries, progress, and artifacts.

Implements the task boundary system inspired by Antigravity's architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from enum import Enum

from core.agent_mode import AgentMode


class TaskItemStatus(Enum):
    """Status markers for task checklist items."""

    TODO = "[ ]"
    IN_PROGRESS = "[/]"
    DONE = "[x]"
    BLOCKED = "[!]"


@dataclass
class TaskItem:
    """A single item in the task checklist."""

    text: str
    status: TaskItemStatus = TaskItemStatus.TODO
    indent: int = 0

    def render(self) -> str:
        """Render as markdown checkbox."""
        indent_str = "  " * self.indent
        return f"{indent_str}{self.status.value} {self.text}"


@dataclass
class TaskState:
    """
    Represents the current state of an agent's task.

    Inspired by Antigravity's task boundaries with:
    - TaskName: Human-readable identifier
    - TaskStatus: What agent is ABOUT TO DO (forward-looking)
    - TaskSummary: What has been accomplished (backward-looking)
    - Mode: Current operational mode
    """

    name: str  # Human-readable task identifier
    mode: AgentMode = AgentMode.IDLE  # Current operational mode
    status: str = ""  # What agent is ABOUT TO DO
    summary: str = ""  # What has been accomplished
    step_count: int = 0  # Current iteration
    max_steps: int = 15  # Maximum allowed iterations
    started_at: datetime = field(default_factory=datetime.now)
    artifacts: List[str] = field(default_factory=list)  # Paths to created artifacts
    checklist: List[TaskItem] = field(default_factory=list)

    # Approval state
    plan_approved: bool = False
    blocked_on_user: bool = False

    # Dashboard Widgets Data
    context_tokens: int = 0
    context_limit: int = 400000
    active_tools: List[str] = field(default_factory=list)
    sub_agents: dict = field(default_factory=dict)  # Name -> Status

    # Backtrack tracking for verification loops
    verification_attempts: int = 0
    max_verification_attempts: int = 3

    def add_artifact(self, path: str) -> None:
        """Register a created artifact."""
        if path not in self.artifacts:
            self.artifacts.append(path)

    def add_checklist_item(self, text: str, indent: int = 0) -> int:
        """Add an item to the checklist. Returns the item index."""
        item = TaskItem(text=text, indent=indent)
        self.checklist.append(item)
        return len(self.checklist) - 1

    def mark_item_in_progress(self, index: int) -> None:
        """Mark a checklist item as in progress."""
        if 0 <= index < len(self.checklist):
            self.checklist[index].status = TaskItemStatus.IN_PROGRESS

    def mark_item_complete(self, index: int) -> None:
        """Mark a checklist item as complete."""
        if 0 <= index < len(self.checklist):
            self.checklist[index].status = TaskItemStatus.DONE

    def mark_item_blocked(self, index: int) -> None:
        """Mark a checklist item as blocked."""
        if 0 <= index < len(self.checklist):
            self.checklist[index].status = TaskItemStatus.BLOCKED

    def render_checklist(self) -> str:
        """Render the full checklist as markdown."""
        if not self.checklist:
            return "No tasks defined."
        return "\n".join(item.render() for item in self.checklist)

    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.checklist:
            return 0.0
        done = sum(1 for item in self.checklist if item.status == TaskItemStatus.DONE)
        return (done / len(self.checklist)) * 100

    def can_backtrack(self) -> bool:
        """Check if we can attempt another verification cycle."""
        return self.verification_attempts < self.max_verification_attempts

    def increment_verification(self) -> None:
        """Track a verification attempt."""
        self.verification_attempts += 1

    def duration(self) -> float:
        """Return task duration in seconds."""
        return (datetime.now() - self.started_at).total_seconds()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "mode": self.mode.name,
            "status": self.status,
            "summary": self.summary,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "started_at": self.started_at.isoformat(),
            "artifacts": self.artifacts,
            "checklist": [
                {"text": item.text, "status": item.status.value, "indent": item.indent}
                for item in self.checklist
            ],
            "plan_approved": self.plan_approved,
            "blocked_on_user": self.blocked_on_user,
            "verification_attempts": self.verification_attempts,
            "progress_percentage": self.progress_percentage(),
            "duration_seconds": self.duration(),
        }


class TaskStateManager:
    """Manages task state transitions and persistence."""

    def __init__(self):
        self.current_task: Optional[TaskState] = None
        self.task_history: List[TaskState] = []

    def start_task(
        self,
        name: str,
        mode: AgentMode = AgentMode.PLANNING,
        status: str = "",
        max_steps: int = 15,
    ) -> TaskState:
        """Start a new task, archiving any existing one."""
        if self.current_task:
            self.task_history.append(self.current_task)

        self.current_task = TaskState(
            name=name, mode=mode, status=status, max_steps=max_steps
        )
        return self.current_task

    def update_boundary(
        self,
        name: Optional[str] = None,
        mode: Optional[AgentMode] = None,
        status: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> None:
        """Update the current task boundary."""
        if not self.current_task:
            return

        if name and name != "%SAME%":
            self.current_task.name = name
        if mode:
            self.current_task.mode = mode
        if status:
            self.current_task.status = status
        if summary:
            self.current_task.summary = summary

    def increment_step(self) -> int:
        """Increment step counter. Returns new step count."""
        if self.current_task:
            self.current_task.step_count += 1
            return self.current_task.step_count
        return 0

    def is_at_limit(self) -> bool:
        """Check if we've reached max steps."""
        if not self.current_task:
            return False
        return self.current_task.step_count >= self.current_task.max_steps

    def get_state(self) -> Optional[TaskState]:
        """Get current task state."""
        return self.current_task

    def end_task(self) -> Optional[TaskState]:
        """End the current task and archive it."""
        if self.current_task:
            self.current_task.mode = AgentMode.IDLE
            self.task_history.append(self.current_task)
            ended = self.current_task
            self.current_task = None
            return ended
        return None
