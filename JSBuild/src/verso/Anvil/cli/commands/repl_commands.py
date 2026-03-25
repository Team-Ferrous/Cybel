from typing import List, Any
import os
from cli.commands.base import SlashCommand
from rich.panel import Panel
from rich.table import Table


class LogsCommand(SlashCommand):
    name = "logs"
    description = "View or follow agent logs"

    def execute(self, args: List[str], context: Any) -> str:
        log_path = ".anvil/logs/anvil.log"
        if not os.path.exists(log_path):
            return f"Log file not found at {log_path}"

        num_lines = 20
        if args:
            try:
                num_lines = int(args[0])
            except ValueError:
                pass

        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                tail = "".join(lines[-num_lines:])
                context.console.print(
                    Panel(tail, title="Agent Logs (Tail)", border_style="dim")
                )
        except Exception as e:
            return f"Error reading logs: {e}"

        return None


class CoconutCommand(SlashCommand):
    name = "coconut"
    description = "Show COCONUT status and backend info"

    def execute(self, args: List[str], context: Any) -> str:
        if not hasattr(context, "loop_orchestrator"):
            return "Loop orchestrator not found in context."

        thinking_system = context.loop_orchestrator.thinking_system
        if not thinking_system or not thinking_system.coconut_enabled:
            return "COCONUT is disabled."

        coconut = thinking_system.coconut
        if not coconut:
            return "COCONUT is enabled but not initialized."

        info = coconut.get_device_info()

        table = Table(title="COCONUT Configuration", border_style="magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Backend", info.get("backend", "Unknown"))
        table.add_row("Device", str(info.get("device", "Unknown")))
        table.add_row("Paths", str(info.get("num_paths", "Unknown")))
        table.add_row("Steps", str(info.get("thought_steps", "Unknown")))

        if "saguaro_ops" in info:
            table.add_row(
                "Saguaro Ops", "✓ Enabled" if info["saguaro_ops"] else "○ Disabled"
            )

        if "gpu_memory_mb" in info:
            table.add_row("GPU Memory", f"{info['gpu_memory_mb']:.2f} MB")

        context.console.print(table)
        return None
