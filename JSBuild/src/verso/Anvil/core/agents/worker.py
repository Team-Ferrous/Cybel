from core.agents.subagent import SubAgent
from core.orchestrator.graph import TaskUnit


class WorkerAgent(SubAgent):
    """
    Specialized subagent for executing a single TaskUnit.
    """

    def __init__(self, task_unit: TaskUnit, **kwargs):
        # SubAgent expects a string task
        task_str = f"Execute Task: {task_unit.instruction}"

        # Extract components for SubAgent
        parent_name = kwargs.get("parent_name", "Orchestrator")
        brain = kwargs.get("brain")
        console = kwargs.get("console")

        super().__init__(
            task=task_str, parent_name=parent_name, brain=brain, console=console
        )

        self.task_unit = task_unit
        self.system_prompt = f"""
You are a Specialized Developer Agent. Your mission is to complete the following task:
TASK ID: {task_unit.id}
TYPE: {task_unit.type.value}
INSTRUCTION: {task_unit.instruction}
CONTEXT FILES: {', '.join(task_unit.context_files)}

### PROTOCOL
1. Focus EXCLUSIVELY on this task.
2. Read the context files first to understand the current state.
3. Implement the requested changes or research.
4. Verify your work locally if possible.
5. When finished, provide a concise summary of what you accomplished.
"""

    def execute(self) -> str:
        """Executes the task and returns a summary."""
        result = self.run()
        return result.get("summary", "Task completed without summary.")
