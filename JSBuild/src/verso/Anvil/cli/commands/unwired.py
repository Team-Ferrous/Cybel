"""REPL command for Saguaro unwired feature analysis."""

from __future__ import annotations

from typing import Any, List

from rich.panel import Panel

from cli.commands.base import SlashCommand


class UnwiredCommand(SlashCommand):
    """Expose `saguaro unwired` through a dedicated REPL slash command."""

    @property
    def name(self) -> str:
        return "unwired"

    @property
    def aliases(self) -> list[str]:
        return ["feature_islands"]

    @property
    def description(self) -> str:
        return "Detect isolated unwired feature clusters via Saguaro."

    def execute(self, args: List[str], context: Any) -> None:
        cmd = "unwired"
        if args:
            cmd = f"{cmd} {' '.join(args)}"

        result = context.saguaro.execute_command(cmd)
        context.console.print(
            Panel(result, title="Unwired Analysis", border_style="magenta")
        )
        context.history.add_message("system", f"Unwired command result:\n{result}")
        return None
