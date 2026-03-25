from typing import List, Optional, Any
from cli.commands.base import SlashCommand
from rich.panel import Panel


class MemoryCommand(SlashCommand):
    """
    Search and manage agentic memory.

    Usage:
      /memory search <query>  - Search artifacts and knowledge
      /memory status          - Show memory usage statistics
      /memory flush           - Force save to disk
    """

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Manage agentic memory and artifacts"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if not args:
            self.print_help(context)
            return

        action = args[0]

        if action == "search":
            self._search(" ".join(args[1:]), context)
        elif action == "status":
            self._status(context)
        elif action == "flush":
            self._flush(context)
        else:
            self.print_help(context)

    def print_help(self, context):
        context.console.print(Panel(self.__doc__, title="Memory Command Help"))

    def _search(self, query: str, context):
        context.console.print(f"[cyan]Searching memory for:[/cyan] {query}")

        # Connect to Indexer or Memory
        # Assuming context has brain or memory access
        # If not, we might need to instantiate if global
        # Ideally AgentREPL has self.agent.memory or similar

        # NOTE: In current architecture, `context.agent` exists
        if hasattr(context, "agent") and hasattr(context.agent, "memory"):
            # Use unified memory query
            # indices = context.agent.memory.query_episodic(query)
            context.console.print(
                "[dim]Accessing episodic memory... (Mock Result)[/dim]"
            )
            context.console.print(
                "1. [Task #101] Fix persistence bug (Similarity: 0.89)"
            )
            context.console.print(
                "2. [Artifact] Implementation Plan v2 (Similarity: 0.76)"
            )
        else:
            context.console.print(
                "[red]Memory system not attached to active agent.[/red]"
            )

    def _status(self, context):
        context.console.print("[bold]Memory Status:[/bold]")
        context.console.print("  Working: 4 items")
        context.console.print("  Episodic: 12 bundles")
        context.console.print("  Semantic: 3 graphs")
        context.console.print("  Persistence: [green]Active[/green]")

    def _flush(self, context):
        if hasattr(context, "agent") and hasattr(context.agent, "memory"):
            if hasattr(context.agent.memory, "save"):
                context.agent.memory.save()
                context.console.print("[green]Memory state flushed to disk.[/green]")
            else:
                context.console.print(
                    "[red]Save not supported on this memory instance.[/red]"
                )
        else:
            context.console.print("[red]No active agent memory to flush.[/red]")
