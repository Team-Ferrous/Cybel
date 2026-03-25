"""
Task View - Rich terminal renderer for task state.

Renders task state in a structured panel separate from chat output.
"""

from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED

from typing import List
from core.task_state import TaskState
from core.agent_mode import AgentMode


class TaskView:
    """
    Renders task state in a structured panel, separate from chat.

    Features:
    - Mode-colored borders
    - Progress bar
    - Status and summary display
    - Artifact list
    - Collapsible checklist
    """

    def __init__(self, console: Console = None):
        """
        Initialize task view.

        Args:
            console: Rich console for output
        """
        self.console = console or Console()
        self.show_checklist = True
        self.show_artifacts = True
        self.compact_mode = False

    def _mode_color(self, mode: AgentMode) -> str:
        """Get color for mode."""
        return {
            AgentMode.PLANNING: "mode.planning",
            AgentMode.EXECUTION: "mode.execution",
            AgentMode.VERIFICATION: "mode.verification",
            AgentMode.IDLE: "mode.idle",
        }.get(mode, "white")

    def _mode_badge(self, mode: AgentMode) -> str:
        """Get styled mode badge."""
        emoji = {
            AgentMode.PLANNING: "📋",
            AgentMode.EXECUTION: "⚡",
            AgentMode.VERIFICATION: "✅",
            AgentMode.IDLE: "💤",
        }.get(mode, "❓")
        return f"{emoji} {mode.name}"

    def _render_progress_bar(self, current: int, total: int, width: int = 25) -> str:
        """Render a simple text progress bar with a more premium look."""
        if total == 0:
            return "░" * width

        filled = int((current / total) * width)
        return (
            "━" * filled + "╸" + "░" * (width - filled - 1)
            if filled < width
            else "━" * width
        )

    def _render_context_meter(self, current: int, limit: int) -> str:
        """Render a context window usage meter."""
        width = 20
        percentage = min(100.0, (current / limit) * 100.0) if limit > 0 else 0
        filled = int((percentage / 100.0) * width)

        color = "green"
        if percentage > 80:
            color = "red"
        elif percentage > 60:
            color = "yellow"

        bar = "█" * filled + "░" * (width - filled)
        return f"[{color}]{bar}[/{color}] {current/1000:.1f}k/{limit/1000:.0f}k ({percentage:.1f}%)"

    def _render_sub_agents(self, sub_agents: dict) -> List[str]:
        """Render active sub-agents."""
        lines = []
        if not sub_agents:
            return ["[dim]No active sub-agents[/dim]"]

        for name, status in sub_agents.items():
            lines.append(f"  [bold yellow]🤖 {name}:[/bold yellow] {status}")
        return lines

    def _render_content(self, state: TaskState) -> str:
        """Render the inner content of the task panel."""
        lines = []

        # Row 1: Status and Context
        lines.append(f"[bold]Status:[/bold] {state.status}")
        lines.append(
            f"[bold]Context:[/bold] {self._render_context_meter(state.context_tokens, state.context_limit)}"
        )
        lines.append("")

        # Progress
        progress = state.progress_percentage()
        bar = self._render_progress_bar(int(progress), 100)
        lines.append(
            f"[dim]Step {state.step_count}/{state.max_steps}[/dim] [{bar}] {progress:.0f}%"
        )
        lines.append("")

        # Summary (if any)
        if state.summary:
            lines.append("[bold]Summary:[/bold]")
            lines.append(f"[dim]{state.summary}[/dim]")
            lines.append("")

        # Sub-Agents Widget
        if state.sub_agents:
            lines.append("[bold]Sub-Agents:[/bold]")
            lines.extend(self._render_sub_agents(state.sub_agents))
            lines.append("")

        # Tool Activity Widget
        if state.active_tools:
            lines.append(
                f"[bold]Active Tools:[/bold] [cyan]{', '.join(state.active_tools)}[/cyan]"
            )
            lines.append("")

        # Checklist (condensed)
        if self.show_checklist and state.checklist:
            done = sum(1 for item in state.checklist if item.status.value == "[x]")
            total = len(state.checklist)
            lines.append(f"[bold]Checklist:[/bold] {done}/{total} complete")

            if not self.compact_mode:
                # Show first 5 items
                for item in state.checklist[:5]:
                    status_color = {
                        "[ ]": "white",
                        "[/]": "yellow",
                        "[x]": "green",
                        "[!]": "red",
                    }.get(item.status.value, "white")
                    lines.append(
                        f"  [{status_color}]{item.status.value}[/{status_color}] {item.text[:50]}"
                    )

                if len(state.checklist) > 5:
                    lines.append(
                        f"  [dim]... and {len(state.checklist) - 5} more[/dim]"
                    )
            lines.append("")

        # Artifacts
        if self.show_artifacts and state.artifacts:
            lines.append(f"[bold]Artifacts:[/bold] {len(state.artifacts)}")
            if not self.compact_mode:
                for artifact in state.artifacts[:3]:
                    # Extract filename for cleaner display
                    import os

                    fname = os.path.basename(artifact)
                    lines.append(f"  [artifact]📄 {fname}[/artifact]")
                if len(state.artifacts) > 3:
                    lines.append(
                        f"  [dim]... and {len(state.artifacts) - 3} more[/dim]"
                    )
            lines.append("")

        # Duration
        duration = state.duration()
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        lines.append(f"[dim]Duration: {minutes}m {seconds}s[/dim]")

        # Blocked indicator
        if state.blocked_on_user:
            lines.append("")
            lines.append("[bold red]⏸️ BLOCKED - Waiting for user input[/bold red]")

        return "\n".join(lines)

    def render(self, state: TaskState) -> Panel:
        """
        Create a Rich Panel for the task state.

        Args:
            state: Current task state

        Returns:
            Rich Panel with task information
        """
        content = self._render_content(state)

        mode_badge = self._mode_badge(state.mode)

        return Panel(
            content,
            title=f"[bold]{state.name}[/bold]",
            subtitle=f"Mode: {mode_badge}",
            border_style=self._mode_color(state.mode),
            box=ROUNDED,
        )

    def update(self, state: TaskState) -> None:
        """
        Update the display with new state.

        Args:
            state: New task state to display
        """
        panel = self.render(state)
        self.console.print(panel)

    def render_compact(self, state: TaskState) -> str:
        """
        Render a compact one-line status.

        Args:
            state: Current task state

        Returns:
            Single line status string
        """
        mode_emoji = {
            AgentMode.PLANNING: "📋",
            AgentMode.EXECUTION: "⚡",
            AgentMode.VERIFICATION: "✅",
            AgentMode.IDLE: "💤",
        }.get(state.mode, "❓")

        progress = state.progress_percentage()
        bar = self._render_progress_bar(int(progress), 100, width=10)

        return f"{mode_emoji} {state.name[:30]} [{bar}] {state.status[:40]}"

    def print_compact(self, state: TaskState) -> None:
        """Print compact status line."""
        self.console.print(self.render_compact(state))


def render_mode_transition(
    console: Console, from_mode: AgentMode, to_mode: AgentMode
) -> None:
    """
    Render a mode transition announcement.

    Args:
        console: Rich console
        from_mode: Previous mode
        to_mode: New mode
    """
    from_badge = f"[{from_mode.color}]{from_mode.name}[/{from_mode.color}]"
    to_badge = f"[{to_mode.color}]{to_mode.name}[/{to_mode.color}]"

    console.print(f"\n{'─' * 40}")
    console.print(f"Mode Transition: {from_badge} → {to_badge}")
    console.print(f"[dim]{to_mode.description}[/dim]")
    console.print(f"{'─' * 40}\n")
