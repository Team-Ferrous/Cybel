from typing import List, Optional, Any
from cli.commands.base import SlashCommand
from core.model_manager import ModelManager
from rich.table import Table


class ModelsCommand(SlashCommand):
    def __init__(self):
        self.manager = ModelManager()

    @property
    def name(self) -> str:
        return "models"

    @property
    def description(self) -> str:
        return "Manage and benchmark Ollama models"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if not args:
            models = self.manager.list_models()
            if not models:
                return "No models found or Ollama is not running."

            table = Table(title="Ollama Models")
            table.add_column("Name", style="cyan")
            table.add_column("ID", style="magenta")
            table.add_column("Size", style="green")

            for m in models:
                table.add_row(m["name"], m["id"], m["size"])

            from rich.console import Console

            console = Console()
            console.print(table)
            return None

        sub = args[0]
        if sub == "pull" and len(args) > 1:
            m_name = args[1]
            print(f"Pulling {m_name}...")
            self.manager.pull_model(m_name)
            return "Pull attempt finished."
        elif sub == "rm" and len(args) > 1:
            m_name = args[1]
            self.manager.remove_model(m_name)
            return f"Removed {m_name}"
        elif sub == "benchmark" and len(args) > 1:
            m_name = args[1]
            print(f"Benchmarking {m_name}...")
            stats = self.manager.benchmark_model(m_name)
            return f"Benchmark results for {m_name}:\nTPS: {stats['tps']:.2f}\nDuration: {stats['duration']:.2f}s"

        return "Usage: /models [list|pull <name>|rm <name>|benchmark <name>]"
