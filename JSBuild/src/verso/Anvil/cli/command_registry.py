from typing import Dict, Optional, Any
from cli.commands.base import SlashCommand


class CommandRegistry:
    """Registry for dispatching slash commands."""

    def __init__(self):
        self.commands: Dict[str, SlashCommand] = {}
        self._categories: Dict[str, str] = {}
        self._category_order: list[str] = []

    def register(self, command: SlashCommand, *, category: str | None = None):
        resolved_category = str(category or getattr(command, "category", "general"))
        self.commands[command.name] = command
        self._categories[command.name] = resolved_category
        for alias in command.aliases:
            self.commands[alias] = command
            self._categories[alias] = resolved_category
        if resolved_category not in self._category_order:
            self._category_order.append(resolved_category)

    def get_command(self, name: str) -> Optional[SlashCommand]:
        return self.commands.get(name)

    def dispatch(self, input_line: str, context: Any) -> bool:
        """
        Parses and executes a command if input starts with /.
        Returns True if a command was executed, False otherwise.
        """
        if not input_line.strip().startswith("/"):
            return False

        parts = input_line.strip().split()
        cmd_name = parts[0][1:]  # Remove slash
        args = parts[1:]

        command = self.get_command(cmd_name)
        if command:
            try:
                result = command.execute(args, context)
                if result:
                    context.console.print(f"[bold green]System:[/bold green] {result}")
            except Exception as e:
                context.console.print(f"[bold red]Command error:[/bold red] {e}")
            return True
        else:
            context.console.print(f"[bold red]Unknown command:[/bold red] /{cmd_name}")
            return True  # It was a command attempt, so we consumed it

    def get_help_text(self) -> str:
        lines = ["Available Commands:"]
        grouped: Dict[str, list[SlashCommand]] = {}
        seen = set()
        for name, cmd in self.commands.items():
            if cmd in seen:
                continue
            seen.add(cmd)
            grouped.setdefault(self._categories.get(name, "general"), []).append(cmd)
        for category in self._category_order:
            commands = grouped.get(category)
            if not commands:
                continue
            lines.append(f"\n{category.title()}:")
            for cmd in sorted(commands, key=lambda item: item.name):
                aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
                lines.append(f"  /{cmd.name}{aliases:<10} - {cmd.description}")
        return "\n".join(lines)
