from typing import List, Any
from cli.commands.base import SlashCommand
import sys


class HelpCommand(SlashCommand):
    name = "help"
    description = "Show available commands"

    def execute(self, args: List[str], context: Any) -> str:
        return context.command_registry.get_help_text()


class ClearCommand(SlashCommand):
    name = "clear"
    description = "Clear conversation history"

    def execute(self, args: List[str], context: Any) -> str:
        context.history.messages = (
            []
        )  # Assuming history has direct access or simple list
        # Re-add system prompt or similar if needed, but context_manager handles window
        return "Conversation history cleared."


class ExitCommand(SlashCommand):
    name = "exit"
    aliases = ["quit"]
    description = "Exit the application"

    def execute(self, args: List[str], context: Any) -> str:
        print("Goodbye.")
        sys.exit(0)


class SaguaroCommand(SlashCommand):
    name = "saguaro"
    description = "Execute a raw Saguaro command"

    def execute(self, args: List[str], context: Any) -> str:
        cmd = " ".join(args)
        if not cmd:
            return "Usage: /saguaro <command>"

        result = context.saguaro.execute_command(cmd)
        from rich.panel import Panel

        context.console.print(
            Panel(result, title="Saguaro Output", border_style="blue")
        )
        # Add to history as system info?
        context.history.add_message(
            "system", f"Saguaro Command '{cmd}' Result:\n{result}"
        )
        return None  # Already printed
