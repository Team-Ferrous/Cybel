"""
Plan Artifact - Implementation plan requiring user approval.

Creates and manages implementation_plan.md with:
- Goal description
- User review requirements
- Proposed changes grouped by component
- Verification plan
"""

from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class ProposedChange:
    """A single proposed change."""

    file_path: str
    description: str
    change_type: str = "modify"  # create, modify, delete
    risk_level: str = "low"  # low, medium, high

    def render(self) -> str:
        """Render as markdown list item."""
        risk_badge = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(
            self.risk_level, "⚪"
        )
        type_badge = {"create": "➕", "modify": "📝", "delete": "🗑️"}.get(
            self.change_type, "📄"
        )
        return f"- {type_badge} `{self.file_path}` - {self.description} {risk_badge}"


@dataclass
class ChangeGroup:
    """A group of related changes."""

    component: str
    description: str
    changes: List[ProposedChange] = field(default_factory=list)

    def add_change(
        self,
        file_path: str,
        description: str,
        change_type: str = "modify",
        risk_level: str = "low",
    ) -> None:
        """Add a change to the group."""
        self.changes.append(
            ProposedChange(
                file_path=file_path,
                description=description,
                change_type=change_type,
                risk_level=risk_level,
            )
        )

    def render(self) -> str:
        """Render as markdown section."""
        lines = [f"#### {self.component}", "", self.description, ""]
        lines.extend(change.render() for change in self.changes)
        return "\n".join(lines)


@dataclass
class ReviewItem:
    """An item requiring user review/decision."""

    category: str  # warning, decision, assumption
    description: str

    def render(self) -> str:
        """Render as markdown."""
        icon = {"warning": "⚠️", "decision": "❓", "assumption": "💡"}.get(
            self.category, "📌"
        )
        return f"- {icon} **{self.category.upper()}:** {self.description}"


class PlanArtifact:
    """
    Implementation plan artifact requiring user approval.

    Creates implementation_plan.md with:
    - Goal description
    - Review items (warnings/decisions/assumptions)
    - Proposed changes grouped by component
    - Verification plan
    - Approval state tracking
    """

    FILENAME = "implementation_plan.md"

    def __init__(self, artifact_dir: Path):
        """
        Initialize plan artifact.

        Args:
            artifact_dir: Directory for storing artifacts
        """
        self.artifact_dir = Path(artifact_dir)
        self.path = self.artifact_dir / self.FILENAME

        self.title: str = "Implementation Plan"
        self.goal: str = ""
        self.review_items: List[ReviewItem] = []
        self.change_groups: List[ChangeGroup] = []
        self.verification_steps: List[str] = []

        self.created_at: datetime = datetime.now()
        self.approved: bool = False
        self.approved_at: Optional[datetime] = None
        self.approved_by: str = ""

        # Ensure directory exists
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def set_goal(self, goal: str) -> None:
        """Set the implementation goal."""
        self.goal = goal

    def add_warning(self, description: str) -> None:
        """Add a warning for user review."""
        self.review_items.append(
            ReviewItem(category="warning", description=description)
        )

    def add_decision(self, description: str) -> None:
        """Add a decision point for user."""
        self.review_items.append(
            ReviewItem(category="decision", description=description)
        )

    def add_assumption(self, description: str) -> None:
        """Add an assumption for user to verify."""
        self.review_items.append(
            ReviewItem(category="assumption", description=description)
        )

    def add_change_group(self, component: str, description: str) -> ChangeGroup:
        """Add a new change group."""
        group = ChangeGroup(component=component, description=description)
        self.change_groups.append(group)
        return group

    def add_verification_step(self, step: str) -> None:
        """Add a verification step."""
        self.verification_steps.append(step)

    def approve(self, approved_by: str = "user") -> None:
        """Mark the plan as approved."""
        self.approved = True
        self.approved_at = datetime.now()
        self.approved_by = approved_by

    def reject(self, reason: str = "") -> None:
        """Mark the plan as rejected."""
        self.approved = False
        self.approved_at = None
        if reason:
            self.add_warning(f"REJECTED: {reason}")

    def total_changes(self) -> int:
        """Count total changes across all groups."""
        return sum(len(group.changes) for group in self.change_groups)

    def high_risk_changes(self) -> List[ProposedChange]:
        """Get all high-risk changes."""
        high_risk = []
        for group in self.change_groups:
            high_risk.extend(c for c in group.changes if c.risk_level == "high")
        return high_risk

    def render(self) -> str:
        """Render the full plan as markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M')}",
        ]

        # Approval status
        if self.approved:
            lines.append(
                f"**Status:** ✅ APPROVED by {self.approved_by} at {self.approved_at.strftime('%Y-%m-%d %H:%M')}"
            )
        else:
            lines.append("**Status:** ⏳ PENDING APPROVAL")

        lines.extend(["", "---", "", "## Goal", "", self.goal, ""])

        # Review items
        if self.review_items:
            lines.extend(["## ⚠️ User Review Required", ""])
            for item in self.review_items:
                lines.append(item.render())
            lines.append("")

        # Proposed changes
        lines.extend(["## Proposed Changes", ""])
        lines.append(f"**Total files:** {self.total_changes()}")

        high_risk = self.high_risk_changes()
        if high_risk:
            lines.append(f"**High-risk changes:** {len(high_risk)} 🔴")

        lines.append("")

        for group in self.change_groups:
            lines.append(group.render())
            lines.append("")

        # Verification plan
        if self.verification_steps:
            lines.extend(["## Verification Plan", ""])
            for i, step in enumerate(self.verification_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        # Approval section
        if not self.approved:
            lines.extend(
                [
                    "---",
                    "",
                    "## Approval",
                    "",
                    "To approve this plan, respond with: **approve**",
                    "",
                    "To request changes, describe what should be modified.",
                    "",
                ]
            )

        return "\n".join(lines)

    def save(self) -> Path:
        """Save artifact to disk."""
        with open(self.path, "w") as f:
            f.write(self.render())
        return self.path

    @classmethod
    def load(cls, path: Path) -> "PlanArtifact":
        """Load artifact from disk (basic parsing)."""
        artifact = cls(path.parent)
        artifact.path = path

        if not path.exists():
            return artifact

        with open(path, "r") as f:
            content = f.read()

        # Check if approved
        if "✅ APPROVED" in content:
            artifact.approved = True

        # Extract goal (between "## Goal" and next ##)
        # Extract goal (between "## Goal" and next ##)
        lines = content.split("\n")
        in_goal = False
        goal_lines = []

        # Parsing state
        current_group = None
        in_changes = False

        for line in lines:
            line_stripped = line.strip()

            # Goal Section
            if line.startswith("## Goal"):
                in_goal = True
                in_changes = False
                current_group = None
                continue
            elif in_goal:
                if line.startswith("## "):
                    in_goal = False
                else:
                    if line_stripped:
                        goal_lines.append(line)

            # Proposed Changes Section
            if line.startswith("## Proposed Changes"):
                in_changes = True
                continue

            if in_changes:
                if (
                    line.startswith("## ")
                    and not line.startswith("## Proposed Changes")
                    and not line.startswith("#### ")
                ):
                    # End of changes section (start of Verification or Approval)
                    in_changes = False
                    current_group = None

                elif line.startswith("#### "):
                    # New Component Group
                    component_name = line.replace("#### ", "").strip()
                    current_group = artifact.add_change_group(component_name, "")

                elif line_stripped.startswith("- ") and current_group:
                    # Parse change item
                    # Format: - 📝 `file_path` - description 🟢
                    # We need to extract file_path and description loosely

                    # 1. Determine type
                    change_type = "modify"
                    if "➕" in line:
                        change_type = "create"
                    elif "📝" in line:
                        change_type = "modify"
                    elif "🗑️" in line:
                        change_type = "delete"

                    # 2. Determine risk
                    risk_level = "low"
                    if "🔴" in line:
                        risk_level = "high"
                    elif "🟡" in line:
                        risk_level = "medium"

                    # 3. Extract path (between backticks if present)
                    file_path = "unknown"
                    description = line_stripped

                    if "`" in line:
                        parts = line.split("`")
                        if len(parts) >= 3:
                            file_path = parts[1]
                            # Description is usually after the path and " - "
                            desc_part = parts[2]
                            if desc_part.startswith(" - "):
                                description = desc_part[3:].strip()
                            else:
                                description = desc_part.strip()

                            # Remove risk badge from description end
                            for badge in ["🔴", "🟡", "🟢", "⚪"]:
                                description = description.replace(badge, "").strip()

                    change = ProposedChange(
                        file_path=file_path,
                        description=description,
                        change_type=change_type,
                        risk_level=risk_level,
                    )
                    current_group.changes.append(change)

        artifact.goal = "\n".join(goal_lines).strip()

        return artifact
