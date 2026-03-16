from typing import List
from core.agents.subagent import SubAgent


class VerifierAgent(SubAgent):
    """
    Specialized subagent for verifying the aggregate changes.
    """

    def __init__(self, task: str = "Verify codebase changes", **kwargs):
        # Extract components for SubAgent
        parent_name = kwargs.get("parent_name", "Orchestrator")
        brain = kwargs.get("brain")
        console = kwargs.get("console")

        super().__init__(
            task=task, parent_name=parent_name, brain=brain, console=console
        )

        self.system_prompt = """
You are the Quality Assurance and Verification Agent. Your role is to ensure that the cumulative changes made to the codebase are correct, robust, and fulfill the original objective.

### RESPONSIBILITIES
1. Run all relevant tests.
2. Perform static analysis (linting, type checking).
3. Review changes for logical consistency.
4. If failures occur, pinpoint the cause and suggest fixes.
"""

    def verify(self, objective: str, change_summaries: List[str]) -> bool:
        """Verifies the changes against the original objective."""
        summary_text = "\n".join([f"- {s}" for s in change_summaries])
        prompt = f"""
ORIGINAL OBJECTIVE: {objective}
CHANGES IMPLEMENTED:
{summary_text}

Verify that these changes are correct and complete. Run tests and check for regression.
"""
        result = self.run(mission_override=prompt)
        # In a real implementation, we'd check for test results in the tool output.
        # For now, we'll assume success if no glaring errors are reported.
        response_text = (
            result.get("full_response", "") if isinstance(result, dict) else str(result)
        )
        return "SUCCESS" in response_text.upper()
