from typing import List
from cli.commands.base import SlashCommand
from core.agents.repo_analyzer import RepoAnalysisSubagent
from core.agents.researcher import ResearchSubagent
from core.agents.debugger import DebugSubagent
from core.agents.implementor import ImplementationSubagent
from core.agents.planner_agent import PlanningSubagent
from core.agents.tester import TestingSubagent
from rich.panel import Panel


class SubagentCommand(SlashCommand):
    """Base class for subagent commands"""

    def __init__(self, agent_class, name, description, aliases=None):
        self.agent_class = agent_class
        self._name = name
        self._description = description
        self._aliases = aliases or []

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def aliases(self):
        return self._aliases

    def execute(self, args: List[str], context):
        if not args:
            context.console.print(f"[yellow]Usage: /{self.name} <task>[/yellow]")
            return

        task = " ".join(args)

        # Spawn subagent
        try:
            # SubAgent init expects: task, parent_name, brain, console
            subagent = self.agent_class(
                task=task,
                parent_name=context.name,
                brain=context.brain,
                console=context.console,
            )

            # Run with progress display
            result = subagent.run()

            # Display artifact (simple render for now)
            if isinstance(result, dict) and "response" in result:
                context.console.print(
                    Panel(
                        result["response"],
                        title=f"{self.name.capitalize()} Result",
                        border_style="cyan",
                    )
                )
            else:
                context.console.print(
                    f"[bold cyan]SubAgent Result:[/bold cyan]\n{result}"
                )
        except Exception as e:
            context.console.print(f"[red]Error executing subagent:[/red] {e}")


# Implementation classes
class AnalyzeCommand(SubagentCommand):
    def __init__(self):
        super().__init__(
            RepoAnalysisSubagent,
            "analyze",
            "Deep repository analysis",
            ["repo", "codebase"],
        )


class ResearchCommand(SubagentCommand):
    def __init__(self):
        super().__init__(
            ResearchSubagent, "research", "Comprehensive research", ["find", "search"]
        )


class DebugCommand(SubagentCommand):
    def __init__(self):
        super().__init__(
            DebugSubagent,
            "debug",
            "Root cause analysis and fixing",
            ["fix", "diagnose"],
        )


class ImplementCommand(SubagentCommand):
    def __init__(self):
        super().__init__(
            ImplementationSubagent,
            "implement",
            "Feature implementation",
            ["build", "create"],
        )


class PlanCommand(SubagentCommand):
    def __init__(self):
        super().__init__(
            PlanningSubagent, "plan", "Comprehensive planning", ["design", "architect"]
        )

    def execute(self, args: List[str], context):
        if not args:
            # Fallback to showing plan artifact if no args
            from core.artifacts.manager import ArtifactManager

            try:
                artifacts = ArtifactManager()
                plan = artifacts.get_plan()
                if plan.goal:
                    context.console.print(plan.render())
                    return
            except Exception:
                pass
            context.console.print(
                f"[yellow]Usage: /{self.name} <objective> (or no args to view current plan)[/yellow]"
            )
            return

        super().execute(args, context)


class TestCommand(SubagentCommand):
    def __init__(self):
        super().__init__(
            TestingSubagent,
            "test",
            "Comprehensive test generation",
            ["tests", "coverage"],
        )
