from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from cli.commands.memory import MemoryCommand


class _Memory:
    def __init__(self):
        self.saved = False

    def save(self):
        self.saved = True


def test_memory_command_flush_and_status_render() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=120)
    memory = _Memory()
    context = SimpleNamespace(console=console, agent=SimpleNamespace(memory=memory))
    command = MemoryCommand()

    command.execute(["status"], context)
    command.execute(["flush"], context)

    rendered = output.getvalue()
    assert "Memory Status" in rendered
    assert memory.saved is True
