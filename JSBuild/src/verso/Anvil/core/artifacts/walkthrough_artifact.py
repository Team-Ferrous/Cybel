"""
Walkthrough Artifact - Proof of work after task completion.

Documents completed work with:
- Summary of changes
- Files modified/created
- Tests run and results
- Thinking chain summary
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class FileChange:
    """Record of a file change."""

    path: str
    change_type: str  # created, modified, deleted
    lines_added: int = 0
    lines_removed: int = 0
    summary: str = ""

    def render(self) -> str:
        """Render as markdown."""
        icon = {"created": "➕", "modified": "📝", "deleted": "🗑️"}.get(
            self.change_type, "📄"
        )
        stats = (
            f"+{self.lines_added} -{self.lines_removed}"
            if self.lines_added or self.lines_removed
            else ""
        )
        line = f"- {icon} `{self.path}`"
        if stats:
            line += f" ({stats})"
        if self.summary:
            line += f" - {self.summary}"
        return line


@dataclass
class TestResult:
    """Result of a test run."""

    name: str
    passed: bool
    message: str = ""
    duration_ms: int = 0

    def render(self) -> str:
        """Render as markdown."""
        icon = "✅" if self.passed else "❌"
        line = f"- {icon} {self.name}"
        if self.duration_ms:
            line += f" ({self.duration_ms}ms)"
        if self.message:
            line += f" - {self.message}"
        return line


class WalkthroughArtifact:
    """
    Proof of work artifact documenting completed tasks.

    Creates walkthrough.md with:
    - Summary of changes made
    - Files modified/created/deleted
    - Tests run and results
    - Key thinking points from the chain
    - Overall outcome
    """

    FILENAME = "walkthrough.md"

    def __init__(self, artifact_dir: Path):
        """
        Initialize walkthrough artifact.

        Args:
            artifact_dir: Directory for storing artifacts
        """
        self.artifact_dir = Path(artifact_dir)
        self.path = self.artifact_dir / self.FILENAME

        self.title: str = "Task Walkthrough"
        self.objective: str = ""
        self.summary: str = ""
        self.outcome: str = "completed"  # completed, partial, failed

        self.file_changes: List[FileChange] = []
        self.test_results: List[TestResult] = []
        self.key_decisions: List[str] = []
        self.lessons_learned: List[str] = []

        self.started_at: Optional[datetime] = None
        self.completed_at: datetime = datetime.now()
        self.duration_seconds: float = 0
        self.tool_calls_count: int = 0
        self.thinking_blocks_count: int = 0

        # Ensure directory exists
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def set_objective(self, objective: str) -> None:
        """Set the original objective."""
        self.objective = objective

    def set_summary(self, summary: str) -> None:
        """Set the summary of what was accomplished."""
        self.summary = summary

    def set_outcome(self, outcome: str) -> None:
        """Set the outcome (completed, partial, failed)."""
        self.outcome = outcome

    def set_timing(self, started_at: datetime, completed_at: datetime = None) -> None:
        """Set timing information."""
        self.started_at = started_at
        self.completed_at = completed_at or datetime.now()
        self.duration_seconds = (self.completed_at - self.started_at).total_seconds()

    def set_stats(self, tool_calls: int = 0, thinking_blocks: int = 0) -> None:
        """Set execution statistics."""
        self.tool_calls_count = tool_calls
        self.thinking_blocks_count = thinking_blocks

    def add_file_change(
        self,
        path: str,
        change_type: str,
        lines_added: int = 0,
        lines_removed: int = 0,
        summary: str = "",
    ) -> None:
        """Record a file change."""
        self.file_changes.append(
            FileChange(
                path=path,
                change_type=change_type,
                lines_added=lines_added,
                lines_removed=lines_removed,
                summary=summary,
            )
        )

    def add_test_result(
        self, name: str, passed: bool, message: str = "", duration_ms: int = 0
    ) -> None:
        """Record a test result."""
        self.test_results.append(
            TestResult(
                name=name, passed=passed, message=message, duration_ms=duration_ms
            )
        )

    def add_key_decision(self, decision: str) -> None:
        """Record a key decision made during execution."""
        self.key_decisions.append(decision)

    def add_lesson(self, lesson: str) -> None:
        """Record a lesson learned."""
        self.lessons_learned.append(lesson)

    def tests_summary(self) -> Dict[str, int]:
        """Get test results summary."""
        passed = sum(1 for t in self.test_results if t.passed)
        failed = sum(1 for t in self.test_results if not t.passed)
        return {"passed": passed, "failed": failed, "total": len(self.test_results)}

    def files_summary(self) -> Dict[str, int]:
        """Get file changes summary."""
        created = sum(1 for f in self.file_changes if f.change_type == "created")
        modified = sum(1 for f in self.file_changes if f.change_type == "modified")
        deleted = sum(1 for f in self.file_changes if f.change_type == "deleted")
        return {
            "created": created,
            "modified": modified,
            "deleted": deleted,
            "total": len(self.file_changes),
        }

    def render(self) -> str:
        """Render the full walkthrough as markdown."""
        outcome_icon = {"completed": "✅", "partial": "🟡", "failed": "❌"}.get(
            self.outcome, "⚪"
        )

        lines = [
            f"# {self.title}",
            "",
            f"**Outcome:** {outcome_icon} {self.outcome.upper()}",
            "",
        ]

        # Timing
        if self.started_at:
            lines.append(
                f"**Started:** {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        lines.append(
            f"**Completed:** {self.completed_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if self.duration_seconds:
            minutes = int(self.duration_seconds // 60)
            seconds = int(self.duration_seconds % 60)
            lines.append(f"**Duration:** {minutes}m {seconds}s")

        lines.extend(["", "---", ""])

        # Objective
        lines.extend(["## Objective", "", self.objective, ""])

        # Summary
        lines.extend(["## Summary", "", self.summary, ""])

        # Stats
        lines.extend(
            [
                "## Execution Stats",
                "",
                f"- **Tool calls:** {self.tool_calls_count}",
                f"- **Thinking blocks:** {self.thinking_blocks_count}",
                "",
            ]
        )

        # File changes
        if self.file_changes:
            fs = self.files_summary()
            lines.extend(
                [
                    "## Files Changed",
                    "",
                    f"Created: {fs['created']} | Modified: {fs['modified']} | Deleted: {fs['deleted']}",
                    "",
                ]
            )
            for change in self.file_changes:
                lines.append(change.render())
            lines.append("")

        # Test results
        if self.test_results:
            ts = self.tests_summary()
            lines.extend(
                [
                    "## Test Results",
                    "",
                    f"Passed: {ts['passed']} | Failed: {ts['failed']} | Total: {ts['total']}",
                    "",
                ]
            )
            for result in self.test_results:
                lines.append(result.render())
            lines.append("")

        # Key decisions
        if self.key_decisions:
            lines.extend(["## Key Decisions", ""])
            for i, decision in enumerate(self.key_decisions, 1):
                lines.append(f"{i}. {decision}")
            lines.append("")

        # Lessons learned
        if self.lessons_learned:
            lines.extend(["## Lessons Learned", ""])
            for lesson in self.lessons_learned:
                lines.append(f"- {lesson}")
            lines.append("")

        return "\n".join(lines)

    def save(self) -> Path:
        """Save artifact to disk."""
        with open(self.path, "w") as f:
            f.write(self.render())
        return self.path

    @classmethod
    def load(cls, path: Path) -> "WalkthroughArtifact":
        """Load artifact from disk."""
        artifact = cls(path.parent)
        artifact.path = path

        if path.exists():
            with open(path, "r") as f:
                content = f.read()

            # Basic extraction of outcome
            if "COMPLETED" in content:
                artifact.outcome = "completed"
            elif "PARTIAL" in content:
                artifact.outcome = "partial"
            elif "FAILED" in content:
                artifact.outcome = "failed"

        return artifact
