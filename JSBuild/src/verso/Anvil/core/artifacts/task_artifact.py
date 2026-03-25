"""
Task Artifact - Living checklist for current work.

Implements the task.md artifact with [ ], [/], [x] markers.
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum


class TaskItemStatus(Enum):
    """Status markers for task checklist items."""

    TODO = "[ ]"
    IN_PROGRESS = "[/]"
    DONE = "[x]"
    BLOCKED = "[!]"
    SKIPPED = "[-]"


@dataclass
class TaskItem:
    """A single item in the task checklist."""

    text: str
    status: TaskItemStatus = TaskItemStatus.TODO
    indent: int = 0
    notes: Optional[str] = None

    def render(self) -> str:
        """Render as markdown checkbox line."""
        indent_str = "  " * self.indent
        line = f"{indent_str}{self.status.value} {self.text}"
        if self.notes:
            line += f" _{self.notes}_"
        return line


@dataclass
class TaskSection:
    """A section grouping related task items."""

    title: str
    items: List[TaskItem] = field(default_factory=list)

    def add_item(self, text: str, indent: int = 0) -> int:
        """Add item to section. Returns item index."""
        item = TaskItem(text=text, indent=indent)
        self.items.append(item)
        return len(self.items) - 1

    def render(self) -> str:
        """Render section as markdown."""
        lines = [f"### {self.title}", ""]
        lines.extend(item.render() for item in self.items)
        return "\n".join(lines)

    def progress(self) -> tuple:
        """Return (done, total) counts."""
        done = sum(1 for item in self.items if item.status == TaskItemStatus.DONE)
        return done, len(self.items)


class TaskArtifact:
    """
    Living checklist for current work.

    Creates and manages task.md artifact with:
    - Sections for organizing related tasks
    - Status markers (TODO, IN_PROGRESS, DONE, BLOCKED, SKIPPED)
    - Progress tracking
    - Auto-save on changes
    """

    FILENAME = "task.md"

    def __init__(self, artifact_dir: Path):
        """
        Initialize task artifact.

        Args:
            artifact_dir: Directory for storing artifacts
        """
        self.artifact_dir = Path(artifact_dir)
        self.path = self.artifact_dir / self.FILENAME

        self.title: str = "Task Checklist"
        self.objective: str = ""
        self.sections: List[TaskSection] = []
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()

        # Ensure directory exists
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def set_objective(self, objective: str) -> None:
        """Set the task objective."""
        self.objective = objective
        self.updated_at = datetime.now()

    def add_section(self, title: str) -> TaskSection:
        """Add a new section."""
        section = TaskSection(title=title)
        self.sections.append(section)
        self.updated_at = datetime.now()
        return section

    def add_item(self, text: str, section_index: int = -1, indent: int = 0) -> tuple:
        """
        Add an item to a section.

        Args:
            text: Item text
            section_index: Section to add to (-1 for last section, creates default if none)
            indent: Indentation level

        Returns:
            (section_index, item_index) tuple
        """
        if not self.sections:
            self.add_section("Tasks")

        section = self.sections[section_index]
        item_index = section.add_item(text, indent)
        self.updated_at = datetime.now()
        return (
            len(self.sections) - 1 if section_index == -1 else section_index
        ), item_index

    def mark_in_progress(self, section_index: int, item_index: int) -> None:
        """Mark an item as in progress."""
        if 0 <= section_index < len(self.sections):
            section = self.sections[section_index]
            if 0 <= item_index < len(section.items):
                section.items[item_index].status = TaskItemStatus.IN_PROGRESS
                self.updated_at = datetime.now()

    def mark_complete(
        self, section_index: int, item_index: int, notes: str = None
    ) -> None:
        """Mark an item as complete."""
        if 0 <= section_index < len(self.sections):
            section = self.sections[section_index]
            if 0 <= item_index < len(section.items):
                section.items[item_index].status = TaskItemStatus.DONE
                if notes:
                    section.items[item_index].notes = notes
                self.updated_at = datetime.now()

    def mark_blocked(
        self, section_index: int, item_index: int, reason: str = None
    ) -> None:
        """Mark an item as blocked."""
        if 0 <= section_index < len(self.sections):
            section = self.sections[section_index]
            if 0 <= item_index < len(section.items):
                section.items[item_index].status = TaskItemStatus.BLOCKED
                if reason:
                    section.items[item_index].notes = reason
                self.updated_at = datetime.now()

    def mark_skipped(
        self, section_index: int, item_index: int, reason: str = None
    ) -> None:
        """Mark an item as skipped."""
        if 0 <= section_index < len(self.sections):
            section = self.sections[section_index]
            if 0 <= item_index < len(section.items):
                section.items[item_index].status = TaskItemStatus.SKIPPED
                if reason:
                    section.items[item_index].notes = reason
                self.updated_at = datetime.now()

    def progress(self) -> dict:
        """Get overall progress statistics."""
        total = 0
        done = 0
        in_progress = 0
        blocked = 0

        for section in self.sections:
            for item in section.items:
                total += 1
                if item.status == TaskItemStatus.DONE:
                    done += 1
                elif item.status == TaskItemStatus.IN_PROGRESS:
                    in_progress += 1
                elif item.status == TaskItemStatus.BLOCKED:
                    blocked += 1

        return {
            "total": total,
            "done": done,
            "in_progress": in_progress,
            "blocked": blocked,
            "percentage": (done / total * 100) if total > 0 else 0,
        }

    def render(self) -> str:
        """Render the full task artifact as markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"**Objective:** {self.objective}" if self.objective else "",
            "",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
            f"**Updated:** {self.updated_at.strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        # Progress bar
        prog = self.progress()
        if prog["total"] > 0:
            bar_length = 20
            filled = int(prog["percentage"] / 100 * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            lines.append(
                f"**Progress:** [{bar}] {prog['percentage']:.1f}% ({prog['done']}/{prog['total']})"
            )
            lines.append("")

        # Sections
        for section in self.sections:
            lines.append(section.render())
            lines.append("")

        return "\n".join(lines)

    def save(self) -> Path:
        """Save artifact to disk."""
        with open(self.path, "w") as f:
            f.write(self.render())
        return self.path

    @classmethod
    def load(cls, path: Path) -> "TaskArtifact":
        """Load artifact from disk (basic parsing)."""
        artifact = cls(path.parent)

        if not path.exists():
            return artifact

        with open(path, "r") as f:
            content = f.read()

        # Basic parsing - extract title and items
        lines = content.split("\n")
        current_section = None

        for line in lines:
            if line.startswith("# "):
                artifact.title = line[2:].strip()
            elif line.startswith("**Objective:**"):
                artifact.objective = line.split(":", 1)[1].strip()
            elif line.startswith("### "):
                current_section = artifact.add_section(line[4:].strip())
            elif line.strip().startswith(("[ ]", "[/]", "[x]", "[!]", "[-]")):
                # Parse task item
                stripped = line.strip()
                status_str = stripped[:3]
                text = stripped[4:].strip()

                # Map status
                status_map = {
                    "[ ]": TaskItemStatus.TODO,
                    "[/]": TaskItemStatus.IN_PROGRESS,
                    "[x]": TaskItemStatus.DONE,
                    "[!]": TaskItemStatus.BLOCKED,
                    "[-]": TaskItemStatus.SKIPPED,
                }
                status = status_map.get(status_str, TaskItemStatus.TODO)

                # Calculate indent
                indent = (len(line) - len(line.lstrip())) // 2

                if current_section:
                    item = TaskItem(text=text, status=status, indent=indent)
                    current_section.items.append(item)

        artifact.path = path
        return artifact
