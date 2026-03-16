"""Utilities for surgeon."""

from typing import Any

from saguaro.agents.base import Agent
from saguaro.context import Context


class SurgeonAgent(Agent):
    """The Surgeon Agent executes specific, high-risk code modifications.
    It "checks out" a task and performs atomic refactors.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(name="Surgeon", role="surgeon")

    def run(self, context: Context, task_id: str = "", **kwargs) -> dict[str, Any]:
        """Handle run."""
        self.log_activity(f"Scrubbing in for task {task_id}")

        # 1. Lease/Lock the necessary files (Coordination)
        # 2. Perform Refactor
        # 3. Release Locks

        return {"status": "success", "files_modified": []}

    def perform_refactor(self, plan: dict[str, Any]) -> None:
        """Handle perform refactor."""
        self.log_activity(
            f"Executing refactor plan: {plan.get('description', 'Unknown')}"
        )
        # Logic to apply patches
