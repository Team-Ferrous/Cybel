import json
from typing import Any, List
from core.agents.subagent import SubAgent
from core.orchestrator.graph import TaskUnit, TaskType

try:
    from saguaro.synthesis.spec import SpecLowerer
    from saguaro.synthesis.spec_lint import lint_sagspec
except Exception:  # pragma: no cover - synthesis package may not be present during bootstrap
    SpecLowerer = None
    lint_sagspec = None


class PlannerAgent(SubAgent):
    """
    Specialized subagent for decomposing objectives into task graphs.
    """

    def __init__(self, task: str = "Decompose objective into task graph", **kwargs):
        # Extract components for SubAgent
        parent_name = kwargs.get("parent_name", "Orchestrator")
        brain = kwargs.get("brain")
        console = kwargs.get("console")

        super().__init__(
            task=task, parent_name=parent_name, brain=brain, console=console
        )

        self.env_info = kwargs.get("env_info", {})
        self.system_prompt = """
You are the Lead Architect and Planner. Your task is to decompose a complex engineering objective into a sequence of discrete, actionable TaskUnits.

### GUIDELINES
1. **DECOMPOSITION**: Break the problem into Research, Implementation, and Verification phases.
2. **DEPENDENCIES**: Identify which tasks must complete before others can start.
3. **CONTEXT**: Specify which files are relevant for each task.
4. **SEMANTIC AWARENESS**: Use your knowledge of the codebase to be specific.

### OUTPUT FORMAT
You MUST output your plan as a JSON object within a <task_graph> tag.
Example:
<task_graph>
{
  "tasks": [
    {
      "id": "research_1",
      "type": "RESEARCH",
      "instruction": "Analyze the auth logic in auth.py",
      "context_files": ["auth.py"],
      "dependencies": []
    },
    {
      "id": "impl_1",
      "type": "IMPLEMENTATION",
      "instruction": "Add OAuth2 support to auth.py",
      "context_files": ["auth.py"],
      "dependencies": ["research_1"]
    }
  ]
}
</task_graph>
"""

    @staticmethod
    def _lint_allows_spec(lint_result: Any) -> bool:
        if lint_result is None:
            return False
        if isinstance(lint_result, dict):
            return bool(
                lint_result.get("is_valid")
                or lint_result.get("valid")
                or not lint_result.get("errors")
            )
        return bool(
            getattr(lint_result, "is_valid", False)
            or getattr(lint_result, "valid", False)
            or not getattr(lint_result, "errors", [])
        )

    def _lower_objective_to_spec(self, objective: str) -> tuple[Any | None, Any | None]:
        if SpecLowerer is None or lint_sagspec is None:
            return None, None
        lowerer = SpecLowerer()
        if hasattr(lowerer, "lower_objective"):
            spec = lowerer.lower_objective(objective, origin="planner_agent")
        else:
            spec = lowerer.lower(objective)
        lint_result = lint_sagspec(spec)
        return spec, lint_result

    @staticmethod
    def _collect_spec_target_files(spec: Any) -> list[str]:
        if spec is None:
            return []
        if isinstance(spec, dict):
            targets = spec.get("target_files") or spec.get("files") or []
            return [str(item) for item in targets if str(item).strip()]
        targets = getattr(spec, "target_files", None) or getattr(spec, "files", None) or []
        return [str(item) for item in targets if str(item).strip()]

    @staticmethod
    def _collect_spec_verification(spec: Any) -> list[str]:
        if spec is None:
            return []
        verification = None
        if isinstance(spec, dict):
            verification = spec.get("verification")
        else:
            verification = getattr(spec, "verification", None)
        if isinstance(verification, dict):
            commands = verification.get("commands") or verification.get("steps") or []
            return [str(item) for item in commands if str(item).strip()]
        commands = getattr(verification, "commands", None) or getattr(
            verification, "steps", None
        ) or []
        return [str(item) for item in commands if str(item).strip()]

    def _task_units_from_spec(self, spec: Any) -> List[TaskUnit]:
        target_files = self._collect_spec_target_files(spec)
        verification = self._collect_spec_verification(spec)
        tasks: list[TaskUnit] = [
            TaskUnit(
                id="spec_review",
                type=TaskType.RESEARCH,
                instruction="Review and validate the deterministic synthesis specification",
                context_files=target_files,
                dependencies=[],
            ),
            TaskUnit(
                id="spec_implement",
                type=TaskType.IMPLEMENTATION,
                instruction="Implement the bounded deterministic synthesis specification",
                context_files=target_files,
                dependencies=["spec_review"],
            ),
        ]
        if verification:
            tasks.append(
                TaskUnit(
                    id="spec_verify",
                    type=TaskType.VERIFICATION,
                    instruction="Run the specification-defined verification commands",
                    context_files=target_files,
                    dependencies=["spec_implement"],
                )
            )
        return tasks

    def plan(self, objective: str) -> List[TaskUnit]:
        """Runs the planning loop with self-critique (Ultrathink)."""
        spec, lint_result = self._lower_objective_to_spec(objective)
        if spec is not None and self._lint_allows_spec(lint_result):
            return self._task_units_from_spec(spec)

        # 1. Initial Plan
        self.console.print("[dim]Phase 1: Generating initial plan...[/dim]")
        result = self.run(mission_override=f"Plan the following objective: {objective}")
        initial_plan = (
            result.get("full_response", "") if isinstance(result, dict) else str(result)
        )

        # 2. Critique & Refine (2 iterations of improvement)
        current_plan = initial_plan
        for i in range(2):
            self.console.print(
                f"[dim]Phase {i+2}: Critiquing and refining plan...[/dim]"
            )
            critique_prompt = f"""
REVIEW THE CURRENT PLAN:
{current_plan}

OBJECTIVE:
{objective}

CRITIQUE:
1. Identify any missing dependencies.
2. Spot potential edge cases or safety risks.
3. Simplify overly complex steps.
4. Ensure all requested functionality is covered.

Provide an UPDATED <task_graph> reflecting these improvements.
"""
            result = self.run(mission_override=critique_prompt)
            current_plan = (
                result.get("full_response", "")
                if isinstance(result, dict)
                else str(result)
            )

        # 3. Final Parse
        content = current_plan

        # Extract JSON from <task_graph> tag
        import re

        match = re.search(r"<task_graph>(.*?)</task_graph>", content, re.DOTALL)
        if not match:
            # Fallback for old models or missed tags
            match = re.search(r"(\{.*?\})", content, re.DOTALL)

        if match:
            try:
                data = json.loads(match.group(1))
                tasks = []
                for t_data in data.get("tasks", []):
                    tasks.append(
                        TaskUnit(
                            id=t_data.get("id"),
                            type=TaskType(t_data.get("type", "IMPLEMENTATION")),
                            instruction=t_data.get("instruction"),
                            context_files=t_data.get("context_files", []),
                            dependencies=t_data.get("dependencies", []),
                        )
                    )
                return tasks
            except Exception as e:
                self.console.print(f"[red]Failed to parse task graph: {e}[/red]")

        return []
