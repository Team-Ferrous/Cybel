from __future__ import annotations

from pathlib import Path

from saguaro.analysis.entry_points import EntryPointDetector


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_entrypoints_filters_legacy_toolchain_and_tests(tmp_path: Path) -> None:
    _write(
        tmp_path / "main.py",
        "if __name__ == '__main__':\n"
        "    print('ok')\n",
    )
    _write(
        tmp_path / "tests" / "test_entry.py",
        "if __name__ == '__main__':\n"
        "    print('test')\n",
    )
    _write(
        tmp_path / ".anvil" / "toolchains" / "llvm.py",
        "if __name__ == '__main__':\n"
        "    print('toolchain')\n",
    )
    _write(
        tmp_path / "Saguaro" / "saguaro" / "cli.py",
        "if __name__ == '__main__':\n"
        "    print('legacy')\n",
    )

    detected = EntryPointDetector(str(tmp_path)).detect()
    files = {Path(item["file"]).resolve().relative_to(tmp_path.resolve()).as_posix() for item in detected}

    assert "main.py" in files
    assert not any(path.startswith("tests/") for path in files)
    assert not any(path.startswith(".anvil/toolchains/") for path in files)
    assert not any(path.startswith("Saguaro/") for path in files)


def test_entrypoints_detects_cli_route_and_main(tmp_path: Path) -> None:
    _write(
        tmp_path / "app.py",
        "if __name__ == '__main__':\n"
        "    print('ok')\n"
        "\n"
        "@click.command()\n"
        "def run():\n"
        "    return 1\n",
    )
    _write(
        tmp_path / "api.py",
        "@router.get('/health')\n"
        "def health():\n"
        "    return {'ok': True}\n",
    )

    detected = EntryPointDetector(str(tmp_path)).detect()
    types = {item["type"] for item in detected}

    assert "main_block" in types
    assert "cli_command" in types
    assert "api_route" in types
