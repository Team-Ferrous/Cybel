"""
Agentic Thinking Commands - Slash commands for enhanced agentic loop control.

Commands:
- /mode [planning|execution|verification] - Switch agent mode
- /loop [simple|enhanced] - Force loop type
- /thinking [on|off|show] - Control thinking visibility
- /plan - Show current implementation plan
- /task - Show current task checklist
- /artifacts - List all current artifacts
"""

from typing import List, Optional, Any
from cli.commands.base import SlashCommand


class LoopCommand(SlashCommand):
    """Force loop type for next task."""

    @property
    def name(self) -> str:
        return "loop"

    @property
    def aliases(self) -> List[str]:
        return ["l"]

    @property
    def description(self) -> str:
        return "Force loop type: /loop [simple|enhanced|auto]"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if not args:
            # Show current setting
            current = getattr(context, "force_loop_type", "auto")
            return f"Current loop type: {current}"

        loop_type = args[0].lower()
        if loop_type not in ["simple", "enhanced", "auto"]:
            return "Invalid loop type. Use: simple, enhanced, or auto"

        context.force_loop_type = loop_type if loop_type != "auto" else None
        return f"Loop type set to: {loop_type}"


class PlanCommand(SlashCommand):
    """Show current implementation plan."""

    @property
    def name(self) -> str:
        return "plan"

    @property
    def description(self) -> str:
        return "Show current implementation plan"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.artifacts import ArtifactManager

        try:
            artifacts = ArtifactManager()
            plan = artifacts.get_plan()

            if plan.goal:
                context.console.print(plan.render())
                return None
            else:
                return "No implementation plan found. Start a task to create one."
        except Exception as e:
            return f"Error loading plan: {e}"


class TaskCommand(SlashCommand):
    """Show current task checklist."""

    @property
    def name(self) -> str:
        return "task"

    @property
    def aliases(self) -> List[str]:
        return ["checklist"]

    @property
    def description(self) -> str:
        return "Show current task checklist"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.artifacts import ArtifactManager

        try:
            artifacts = ArtifactManager()
            task = artifacts.get_task()

            if task.sections:
                context.console.print(task.render())
                return None
            else:
                return "No active task. Start a task to create a checklist."
        except Exception as e:
            return f"Error loading task: {e}"


class ArtifactsCommand(SlashCommand):
    """List all current artifacts."""

    @property
    def name(self) -> str:
        return "artifacts"

    @property
    def aliases(self) -> List[str]:
        return ["art"]

    @property
    def description(self) -> str:
        return "List all current artifacts"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.artifacts import ArtifactManager
        import os

        try:
            artifacts = ArtifactManager()

            if args and args[0] == "show":
                if len(args) < 2:
                    return "Usage: /artifacts show <filename>"

                fname = args[1]
                path = artifacts.current_dir / fname
                if not path.exists():
                    return f"Artifact not found: {fname}"

                content = path.read_text(encoding="utf-8")
                from rich.syntax import Syntax
                from rich.panel import Panel

                lexer = "markdown" if fname.endswith(".md") else "python"
                syntax = Syntax(content, lexer, theme="monokai", line_numbers=True)
                context.console.print(
                    Panel(syntax, title=f"Artifact: {fname}", border_style="cyan")
                )
                return None

            paths = artifacts.get_all_artifact_paths()

            if paths:
                lines = ["[bold]Current Artifacts:[/bold]"]
                for path in paths:
                    fname = os.path.basename(path)
                    lines.append(f"  📄 [cyan]{fname}[/cyan]")
                lines.append(
                    "\n[dim]Use '/artifacts show <name>' to view content.[/dim]"
                )
                return "\n".join(lines)
            else:
                return "No artifacts found."
        except Exception as e:
            return f"Error listing/showing artifacts: {e}"


class ArchiveCommand(SlashCommand):
    """Archive current task and start fresh."""

    @property
    def name(self) -> str:
        return "archive"

    @property
    def description(self) -> str:
        return "Archive current task artifacts"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.artifacts import ArtifactManager

        try:
            artifacts = ArtifactManager()

            # Get task name if provided
            task_name = " ".join(args) if args else None

            archive_path = artifacts.archive_current(task_name)
            return f"Archived to: {archive_path}"
        except Exception as e:
            return f"Error archiving: {e}"


class ArchivesCommand(SlashCommand):
    """List archived tasks."""

    @property
    def name(self) -> str:
        return "archives"

    @property
    def description(self) -> str:
        return "List archived tasks"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.artifacts import ArtifactManager

        try:
            artifacts = ArtifactManager()
            archives = artifacts.list_archives()

            if archives:
                lines = ["Archived Tasks:"]
                for arch in archives[:10]:  # Show last 10
                    lines.append(f"  📦 {arch['timestamp']} - {arch['name']}")
                if len(archives) > 10:
                    lines.append(f"  ... and {len(archives) - 10} more")
                return "\n".join(lines)
            else:
                return "No archives found."
        except Exception as e:
            return f"Error listing archives: {e}"


class VerifyCommand(SlashCommand):
    """Run verification suite."""

    @property
    def name(self) -> str:
        return "verify"

    @property
    def aliases(self) -> List[str]:
        return ["v"]

    @property
    def description(self) -> str:
        return "Run verification suite (syntax, lint, types, tests)"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from tools.verify import run_all_verifications

        path = args[0] if args else "."

        context.console.print("[dim]Running verification suite...[/dim]")
        results = run_all_verifications(path)

        # Display results
        from rich.panel import Panel

        lines = [results.summary(), ""]

        for r in results.results:
            status = "✅" if r.passed else "❌"
            lines.append(f"{status} {r.tool.upper()}: {r.message}")

            if r.details:
                for detail in r.details[:3]:
                    lines.append(f"   {detail}")
                if len(r.details) > 3:
                    lines.append(f"   ... and {len(r.details) - 3} more")

        border = "green" if results.all_passed else "red"
        context.console.print(
            Panel("\n".join(lines), title="Verification Results", border_style=border)
        )

        return None


class ApproveCommand(SlashCommand):
    """Approve the current implementation plan."""

    @property
    def name(self) -> str:
        return "approve"

    @property
    def description(self) -> str:
        return "Approve the current implementation plan"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.artifacts import ArtifactManager

        try:
            artifacts = ArtifactManager()

            if artifacts.is_plan_approved():
                return "Plan is already approved."

            artifacts.approve_plan("user")
            return "✅ Plan approved! Agent can proceed with execution."
        except Exception as e:
            return f"Error approving plan: {e}"


class ChainsCommand(SlashCommand):
    """List saved thinking chains."""

    @property
    def name(self) -> str:
        return "chains"

    @property
    def description(self) -> str:
        return "List saved thinking chains"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        from core.artifacts import ArtifactManager

        try:
            artifacts = ArtifactManager()
            chains = artifacts.list_thinking_chains()

            if chains:
                lines = ["Thinking Chains:"]
                for chain_id in chains[:10]:
                    lines.append(f"  🧠 {chain_id}")
                if len(chains) > 10:
                    lines.append(f"  ... and {len(chains) - 10} more")
                return "\n".join(lines)
            else:
                return "No thinking chains saved."
        except Exception as e:
            return f"Error listing chains: {e}"
