"""
Artifact Manager - Centralized management of all task artifacts.

Handles:
- Artifact directory management
- Task, Plan, Walkthrough artifact lifecycle
- Approval state persistence
- Archive management
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

from core.artifacts.task_artifact import TaskArtifact
from core.artifacts.plan_artifact import PlanArtifact
from core.artifacts.walkthrough_artifact import WalkthroughArtifact


class ArtifactManager:
    """
    Manages the lifecycle of task artifacts.

    Directory structure:
    .anvil/artifacts/
    ├── current/
    │   ├── task.md
    │   ├── implementation_plan.md
    │   └── walkthrough.md
    ├── archive/
    │   └── <timestamp>_<task_name>/
    ├── approvals/
    │   └── <plan_id>.json
    └── thinking_chains/
        └── <task_id>_thinking.json
    """

    DEFAULT_BASE_DIR = ".anvil/artifacts"

    def __init__(self, base_dir: str = None):
        """
        Initialize artifact manager.

        Args:
            base_dir: Base directory for artifacts (defaults to .anvil/artifacts)
        """
        self.base_dir = Path(base_dir) if base_dir else Path(self.DEFAULT_BASE_DIR)
        self.current_dir = self.base_dir / "current"
        self.archive_dir = self.base_dir / "archive"
        self.approvals_dir = self.base_dir / "approvals"
        self.thinking_dir = self.base_dir / "thinking_chains"

        # Create directories
        for dir_path in [
            self.current_dir,
            self.archive_dir,
            self.approvals_dir,
            self.thinking_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Current artifacts (lazy loaded)
        self._task: Optional[TaskArtifact] = None
        self._plan: Optional[PlanArtifact] = None
        self._walkthrough: Optional[WalkthroughArtifact] = None

    # --- Task Artifact ---

    def get_task(self) -> TaskArtifact:
        """Get or create current task artifact."""
        if self._task is None:
            task_path = self.current_dir / TaskArtifact.FILENAME
            if task_path.exists():
                self._task = TaskArtifact.load(task_path)
            else:
                self._task = TaskArtifact(self.current_dir)
        return self._task

    def create_task(self, title: str, objective: str) -> TaskArtifact:
        """Create a new task artifact."""
        self._task = TaskArtifact(self.current_dir)
        self._task.title = title
        self._task.set_objective(objective)
        self._task.save()
        return self._task

    def save_task(self) -> Optional[Path]:
        """Save current task artifact."""
        if self._task:
            return self._task.save()
        return None

    @property
    def task_path(self) -> Path:
        """Get task artifact path."""
        return self.current_dir / TaskArtifact.FILENAME

    # --- Plan Artifact ---

    def get_plan(self) -> PlanArtifact:
        """Get or create current plan artifact."""
        if self._plan is None:
            plan_path = self.current_dir / PlanArtifact.FILENAME
            if plan_path.exists():
                self._plan = PlanArtifact.load(plan_path)
            else:
                self._plan = PlanArtifact(self.current_dir)
        return self._plan

    def create_plan(self, title: str, goal: str) -> PlanArtifact:
        """Create a new plan artifact."""
        self._plan = PlanArtifact(self.current_dir)
        self._plan.title = title
        self._plan.set_goal(goal)
        self._plan.save()
        return self._plan

    def save_plan(self) -> Optional[Path]:
        """Save current plan artifact."""
        if self._plan:
            return self._plan.save()
        return None

    def load_plan(self) -> Optional[PlanArtifact]:
        """Load plan from disk."""
        plan_path = self.current_dir / PlanArtifact.FILENAME
        if plan_path.exists():
            self._plan = PlanArtifact.load(plan_path)
            return self._plan
        return None

    def approve_plan(self, approved_by: str = "user") -> bool:
        """Approve the current plan."""
        plan = self.get_plan()
        plan.approve(approved_by)
        plan.save()

        # Also save approval record
        approval_record = {
            "approved_at": datetime.now().isoformat(),
            "approved_by": approved_by,
            "plan_path": str(plan.path),
        }
        approval_path = (
            self.approvals_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(approval_path, "w") as f:
            json.dump(approval_record, f, indent=2)

        return True

    def is_plan_approved(self) -> bool:
        """Check if current plan is approved."""
        plan = self.get_plan()
        return plan.approved

    @property
    def plan_path(self) -> Path:
        """Get plan artifact path."""
        return self.current_dir / PlanArtifact.FILENAME

    # --- Walkthrough Artifact ---

    def get_walkthrough(self) -> WalkthroughArtifact:
        """Get or create current walkthrough artifact."""
        if self._walkthrough is None:
            wt_path = self.current_dir / WalkthroughArtifact.FILENAME
            if wt_path.exists():
                self._walkthrough = WalkthroughArtifact.load(wt_path)
            else:
                self._walkthrough = WalkthroughArtifact(self.current_dir)
        return self._walkthrough

    def create_walkthrough(self, objective: str) -> WalkthroughArtifact:
        """Create a new walkthrough artifact."""
        self._walkthrough = WalkthroughArtifact(self.current_dir)
        self._walkthrough.set_objective(objective)
        return self._walkthrough

    def save_walkthrough(self) -> Optional[Path]:
        """Save current walkthrough artifact."""
        if self._walkthrough:
            return self._walkthrough.save()
        return None

    @property
    def walkthrough_path(self) -> Path:
        """Get walkthrough artifact path."""
        return self.current_dir / WalkthroughArtifact.FILENAME

    # --- Thinking Chains ---

    def save_thinking_chain(self, chain, task_id: str = None) -> Path:
        """Save a thinking chain to file."""
        task_id = task_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.thinking_dir / f"{task_id}_thinking.json"
        chain.save(str(path))
        return path

    def load_thinking_chain(self, task_id: str):
        """Load a thinking chain by task ID."""
        from core.thinking import ThinkingChain

        path = self.thinking_dir / f"{task_id}_thinking.json"
        if path.exists():
            return ThinkingChain.load(str(path))
        return None

    def list_thinking_chains(self) -> List[str]:
        """List all saved thinking chain IDs."""
        chains = []
        for path in self.thinking_dir.glob("*_thinking.json"):
            task_id = path.stem.replace("_thinking", "")
            chains.append(task_id)
        return sorted(chains, reverse=True)

    # --- Mermaid Artifacts ---

    def save_mermaid_diagram(self, name: str, content: str) -> Path:
        """Save a Mermaid diagram artifact."""
        if not name.endswith(".mermaid"):
            name += ".mermaid"
        path = self.current_dir / name
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    # --- Archive ---

    def archive_current(self, task_name: str = None) -> Path:
        """Archive current artifacts and start fresh."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = (task_name or "task").replace(" ", "_").replace("/", "_")[:50]
        archive_name = f"{timestamp}_{safe_name}"
        archive_path = self.archive_dir / archive_name

        # Move current directory contents to archive
        if self.current_dir.exists() and any(self.current_dir.iterdir()):
            shutil.copytree(self.current_dir, archive_path)

            # Clear current
            for item in self.current_dir.iterdir():
                if item.is_file():
                    item.unlink()

        # Reset lazy-loaded artifacts
        self._task = None
        self._plan = None
        self._walkthrough = None

        return archive_path

    def list_archives(self) -> List[Dict[str, str]]:
        """List all archived tasks."""
        archives = []
        for path in sorted(self.archive_dir.iterdir(), reverse=True):
            if path.is_dir():
                # Parse name
                parts = path.name.split("_", 2)
                if len(parts) >= 2:
                    timestamp = f"{parts[0]}_{parts[1]}"
                    name = parts[2] if len(parts) > 2 else "unnamed"

                    archives.append(
                        {"path": str(path), "timestamp": timestamp, "name": name}
                    )
        return archives

    def restore_archive(self, archive_name: str) -> bool:
        """Restore an archived task to current."""
        archive_path = self.archive_dir / archive_name

        if not archive_path.exists():
            return False

        # Archive current first
        if any(self.current_dir.iterdir()):
            self.archive_current("before_restore")

        # Copy archive to current
        for item in archive_path.iterdir():
            if item.is_file():
                shutil.copy2(item, self.current_dir / item.name)

        # Reset lazy-loaded
        self._task = None
        self._plan = None
        self._walkthrough = None

        return True

    # --- Utilities ---

    def clear_current(self) -> None:
        """Clear all current artifacts without archiving."""
        for item in self.current_dir.iterdir():
            if item.is_file():
                item.unlink()

        self._task = None
        self._plan = None
        self._walkthrough = None

    def mark_step_complete(self, step_id: int | str) -> None:
        """Mark a step as complete in the task artifact."""
        task = self.get_task()
        if task.sections:
            # Assume step_id is (section_index, item_index) or just item_index in first section
            if isinstance(step_id, tuple):
                task.mark_complete(step_id[0], step_id[1])
            else:
                task.mark_complete(0, int(step_id))
            task.save()

    def get_all_artifact_paths(self) -> List[Path]:
        """Get paths to all current artifacts."""
        paths = []
        for artifact in [
            TaskArtifact.FILENAME,
            PlanArtifact.FILENAME,
            WalkthroughArtifact.FILENAME,
        ]:
            path = self.current_dir / artifact
            if path.exists():
                paths.append(path)
        return paths
