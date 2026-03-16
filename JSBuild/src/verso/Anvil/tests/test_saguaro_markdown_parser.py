from __future__ import annotations

from saguaro.parsing.markdown import MarkdownStructureParser


def test_markdown_parser_builds_sections_and_blocks() -> None:
    parser = MarkdownStructureParser()
    document = parser.parse(
        """
# Roadmap

Intro paragraph.

## Requirements

- [x] The parser MUST preserve headings.
- The parser SHOULD preserve task metadata.

> Quotes MAY remain single blocks.

| Surface | Rule |
| --- | --- |
| Parser | MUST keep stable IDs |

```python
print("ignored by requirements")
```
""".strip(),
        source_path="specs/roadmap.md",
    )

    sections = document.nodes_of_kind("section")
    assert [section.title for section in sections] == ["Roadmap", "Requirements"]

    lists = document.nodes_of_kind("list")
    assert len(lists) == 1
    items = document.nodes_of_kind("list_item")
    assert len(items) == 2
    assert items[0].checked is True
    assert items[0].text == "The parser MUST preserve headings."

    quote = document.nodes_of_kind("blockquote")[0]
    assert quote.text == "Quotes MAY remain single blocks."

    table = document.nodes_of_kind("table")[0]
    assert table.header == ["Surface", "Rule"]
    assert table.rows == [["Parser", "MUST keep stable IDs"]]

    code = document.nodes_of_kind("code_fence")[0]
    assert code.language == "python"
    assert "ignored by requirements" in code.text


def test_markdown_parser_keeps_section_path_for_children() -> None:
    parser = MarkdownStructureParser()
    document = parser.parse(
        """
# Parent
## Child
Paragraph text.
""".strip(),
        source_path="docs/sample.md",
    )

    paragraph = document.nodes_of_kind("paragraph")[0]
    assert paragraph.section_path == ("Parent", "Child")
