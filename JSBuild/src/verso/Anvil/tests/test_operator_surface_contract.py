from __future__ import annotations

from cli.command_registry import CommandRegistry
from cli.commands.basic import ClearCommand, HelpCommand
from cli import repl as repl_cli
import anvil
import main
import json


def test_command_registry_groups_help_by_category() -> None:
    registry = CommandRegistry()
    registry.register(HelpCommand(), category="mission")
    registry.register(ClearCommand(), category="diagnostics")

    text = registry.get_help_text()

    assert "Mission:" in text
    assert "Diagnostics:" in text
    assert "/help" in text
    assert "/clear" in text


def test_deprecated_launchers_route_through_repl(monkeypatch, capsys) -> None:
    calls: list[list[str] | None] = []

    def _fake(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(main, "anvil_main", _fake)
    monkeypatch.setattr(anvil, "_load_repl_main", lambda: _fake)

    assert main.main(["--plan-only", "ship it"]) == 0
    assert anvil.main(["--prompt", "ship it"]) == 0

    stderr = capsys.readouterr().err
    assert "Deprecated launcher" in stderr
    assert calls == [["--plan-only", "ship it"], ["--prompt", "ship it"]]


def test_mission_subcommand_supports_detach(tmp_path, monkeypatch, capsys) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(tmp_path)

    assert (
        repl_cli.main(
            [
                "mission",
                "--detach",
                "--format",
                "json",
                "--root-dir",
                str(repo),
                "Ship",
                "the",
                "campaign",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["detached"] is True
    assert payload["campaign_id"]
    assert payload["worker"]["state"] == "running"
