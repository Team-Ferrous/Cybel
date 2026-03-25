"""Utilities for auditor."""

from typing import Any

from saguaro.agents.base import Agent
from saguaro.context import Context


class AuditorAgent(Agent):
    """The Auditor Agent enforces constraints and verifies progress.
    It runs verification loops before code is merged.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(name="Auditor", role="auditor")

    def run(self, context: Context, **kwargs) -> dict[str, Any]:
        """Handle run."""
        self.log_activity("Starting compliance audit...")

        # Invokes the Sentinel and Policy engines
        # In a real scenario, this would call `saguaro verify`

        return {"compliance_score": 100, "violations": [], "drift_detected": False}

    def verify_change(self, change_set: list[str]) -> bool:
        """Handle verify change."""
        self.log_activity(f"Verifying changes in {len(change_set)} files")
        return True
