from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from cli.commands.memory import MemoryCommand


def test_memory_command_search_is_available_to_end_users() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    context = SimpleNamespace(console=console, agent=SimpleNamespace(memory=object()))
    command = MemoryCommand()

    command.execute(["search", "latent", "memory"], context)

    rendered = output.getvalue()
    assert "Searching memory for" in rendered
    assert "episodic memory" in rendered
