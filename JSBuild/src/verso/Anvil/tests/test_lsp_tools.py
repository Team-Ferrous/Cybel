from pathlib import Path

from tools.lsp import LSPTools


def test_lsp_tools_definition_and_diagnostics(tmp_path: Path):
    root = tmp_path
    good = root / "good.py"
    bad = root / "bad.py"

    good.write_text(
        "def target_function():\n"
        "    return 42\n",
        encoding="utf-8",
    )
    bad.write_text(
        "def broken(\n"
        "    return 1\n",
        encoding="utf-8",
    )

    lsp = LSPTools(root_dir=str(root))

    definition = lsp.get_definition(symbol="target_function")
    assert "good.py:1" in definition

    diagnostics = lsp.get_diagnostics(file_path="bad.py")
    assert "syntax-error" in diagnostics

    summary = lsp.get_diagnostics()
    assert "Found" in summary
