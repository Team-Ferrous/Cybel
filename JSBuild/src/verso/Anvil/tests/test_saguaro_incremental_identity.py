from __future__ import annotations

from pathlib import Path

from saguaro.parsing.parser import SAGUAROParser


def _function_segment_ids(parser: SAGUAROParser, path: Path) -> dict[str, str]:
    entities = parser.parse_file(str(path))
    return {
        entity.name: str(entity.metadata.get("segment_id") or "")
        for entity in entities
        if entity.type in {"function", "method"} and entity.name
    }


def test_segment_identity_stable_on_noop_and_line_shift(tmp_path: Path) -> None:
    source = tmp_path / "identity_demo.py"
    source.write_text(
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n",
        encoding="utf-8",
    )
    parser = SAGUAROParser()

    first = _function_segment_ids(parser, source)
    second = _function_segment_ids(parser, source)
    assert first == second

    # Shift function lines without changing function content.
    source.write_text(
        "\n\n# moved by formatting\n\n"
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n",
        encoding="utf-8",
    )
    shifted = _function_segment_ids(parser, source)

    assert shifted.get("alpha") == first.get("alpha")
    assert shifted.get("beta") == first.get("beta")
