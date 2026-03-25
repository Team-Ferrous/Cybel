from typing import List, Optional, Any
from rich.table import Table

from cli.commands.base import SlashCommand
from core.approval import ApprovalMode


class ModeCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "mode"

    @property
    def description(self) -> str:
        return "Set approval mode (suggest, auto-edit, full-auto, paranoid)"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if not args:
            return f"Current Mode: {context.approval_manager.mode.value}"

        mode_str = args[0].lower()
        try:
            mode = ApprovalMode(mode_str)
            context.approval_manager.set_mode(mode)
            # Update config too for persistence?
            context.config.set("approval_mode", mode.value)
            context.config.save()
            return None  # set_mode prints its own confirmation
        except ValueError:
            return f"Invalid mode: {mode_str}. Valid modes: {[m.value for m in ApprovalMode]}"


class SettingsCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "settings"

    @property
    def description(self) -> str:
        return "View or modify configuration settings"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if not args:
            # Show all settings
            table = Table(title="Current Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")

            for k, v in context.config.config.items():
                table.add_row(k, str(v))

            context.console.print(table)
            return None

        if len(args) < 2:
            return "Usage: /settings <key> <value>"

        key = args[0]
        value = args[1]

        # Simple type conversion
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)

        context.config.set(key, value)
        context.config.save()
        return f"Set configuration '{key}' to '{value}'"
