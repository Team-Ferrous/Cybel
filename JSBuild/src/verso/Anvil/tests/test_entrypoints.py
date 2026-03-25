from __future__ import annotations

import main


def test_main_py_delegates_to_anvil_launcher(monkeypatch, capsys) -> None:
    calls: list[list[str] | None] = []

    def _fake(argv: list[str] | None) -> int:
        calls.append(argv)
        return 7

    monkeypatch.setattr(main, "anvil_main", _fake)

    assert main.main(["--mission", "cleanup"]) == 7
    assert calls == [["--mission", "cleanup"]]
    assert "Deprecated launcher" in capsys.readouterr().err
