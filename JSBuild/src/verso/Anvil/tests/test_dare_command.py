from types import SimpleNamespace
from unittest.mock import MagicMock

from cli.commands.dare import DareCommand


def _make_context(tmp_path):
    return SimpleNamespace(
        console=MagicMock(),
        root_dir=str(tmp_path),
        brain=None,
        ownership_registry=None,
    )


def test_dare_command_analyze_and_status(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    context = _make_context(tmp_path)
    command = DareCommand()

    assert command.execute(["analyze", str(tmp_path)], context) is True
    assert hasattr(context, "_dare_pipeline")
    assert command.execute(["status"], context) is True


def test_dare_command_kb_report(tmp_path):
    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
    context = _make_context(tmp_path)
    command = DareCommand()
    command.execute(["analyze", str(tmp_path)], context)

    assert command.execute(["kb", "report"], context) is True
