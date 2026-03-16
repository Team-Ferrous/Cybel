from __future__ import annotations

from pathlib import Path

from saguaro.synthesis.contract_harvester import ContractHarvester


def test_contract_harvester_mines_docstrings_assertions_and_doc_refs(tmp_path: Path) -> None:
    python_file = tmp_path / "sample.py"
    markdown_file = tmp_path / "sample.md"
    python_file.write_text(
        "def clamp(value, lower, upper):\n"
        "    \"\"\"Clamp into bounds.\"\"\"\n"
        "    assert lower <= upper\n"
        "    return max(lower, min(upper, value))\n",
        encoding="utf-8",
    )
    markdown_file.write_text("Use `clamp` for bounded math.\n", encoding="utf-8")

    harvested = ContractHarvester().harvest_paths([python_file, markdown_file])

    assert any(item.symbol == "clamp" and item.contract_type == "docstring" for item in harvested)
    assert any(item.contract_type == "assertion" for item in harvested)
    assert any(item.contract_type == "doc_reference" for item in harvested)

