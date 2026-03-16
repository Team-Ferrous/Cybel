from typing import Any, Dict, Optional
from core.loops.phases.base_phase import BasePhase
from core.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionPhase(BasePhase):
    """
    Phase 3: Execution - Generate action plan and execute tools.
    """

    def execute(
        self, user_input: str, context: Dict[str, Any], dashboard: Optional[Any] = None
    ) -> Dict[str, Any]:
        logger.info("Executing Execution Phase")

        request_type = context.get("request_type")
        results = {
            "success": True,
            "tool_outputs": [],
            "errors": [],
            "files_written": [],
            "files_edited": [],
            "commands_run": [],
            "verification": {"passed": True, "issues": []},
        }

        if request_type in ["modification", "creation", "deletion"]:
            if dashboard:
                dashboard.update_phase(
                    "Execution",
                    status="in_progress",
                    message="Generating action plan...",
                )
            else:
                self.console.print("[cyan]Phase 2: Generating action plan...[/cyan]")

            # Generate action plan (logic from _handle_action)
            action_context = context.get("evidence", {})
            action_plan = self.loop._generate_action_plan(user_input, action_context)
            results["action_plan"] = action_plan

            # Execute the action (logic from _handle_action)
            if not dashboard:
                self.console.print("[cyan]Phase 3: Executing action...[/cyan]")

            if self.loop.enhanced_mode:
                execution_results = self.loop._execute_action_enhanced(
                    action_plan, user_input, dashboard=dashboard
                )
            else:
                execution_results = self.loop._execute_action_basic(action_plan)

            results.update(execution_results)

            if dashboard:
                dashboard.update_phase("Execution", status="completed")
        else:
            if dashboard:
                dashboard.update_phase(
                    "Execution",
                    status="skipped",
                    message="Research task - skipping execution",
                )

        return results
