"""Lightweight structural Markdown parser for roadmap and requirements docs."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterator

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
_FENCE_RE = re.compile(r"^([`~]{3,})([^\n`]*)$")
_ULIST_RE = re.compile(r"^([ \t]*)([-+*])[ \t]+(.*)$")
_OLIST_RE = re.compile(r"^([ \t]*)(\d+)[.)][ \t]+(.*)$")
_TASK_RE = re.compile(r"^\[( |x|X)\][ \t]+(.*)$")
_TABLE_SEPARATOR_RE = re.compile(r"^[ \t|:\-]+$")


@dataclass(slots=True)
class MarkdownNode:
    """Represent a structural markdown node."""

    kind: str
    line_start: int
    line_end: int
    text: str = ""
    children: list["MarkdownNode"] = field(default_factory=list)
    level: int | None = None
    title: str | None = None
    ordered: bool | None = None
    checked: bool | None = None
    language: str | None = None
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    section_path: tuple[str, ...] = field(default_factory=tuple)

    def walk(self) -> Iterator["MarkdownNode"]:
        """Yield this node and all descendants."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass(slots=True)
class MarkdownDocument:
    """Provide a parsed markdown document."""

    source_path: str | None
    text: str
    root: MarkdownNode

    def walk(self) -> Iterator[MarkdownNode]:
        """Yield document nodes depth-first."""
        yield from self.root.walk()

    def nodes_of_kind(self, kind: str) -> list[MarkdownNode]:
        """Return all nodes matching ``kind``."""
        return [node for node in self.walk() if node.kind == kind]


@dataclass(slots=True)
class ParsedMarkdownView:
    """Compatibility view used by roadmap-facing consumers."""

    profile: str
    nodes: list[MarkdownNode]
    blocks: list[MarkdownNode]
    equations: list[MarkdownNode]
    code_fences: list[MarkdownNode]

    def walk(self) -> Iterator[MarkdownNode]:
        """Yield the flattened structural view."""
        for group in (self.nodes, self.blocks, self.equations, self.code_fences):
            for node in group:
                yield node


class MarkdownStructureParser:
    """Parse markdown into a small structural tree."""

    def parse(self, text: str, *, source_path: str | None = None) -> MarkdownDocument:
        """Parse ``text`` and return a document tree."""
        lines = text.splitlines()
        root = MarkdownNode(
            kind="document",
            line_start=1,
            line_end=max(1, len(lines)),
            text="",
            section_path=(),
        )
        section_stack: list[MarkdownNode] = [root]
        index = 0
        while index < len(lines):
            line_number = index + 1
            line = lines[index]
            stripped = line.strip()
            if not stripped:
                index += 1
                continue

            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                while len(section_stack) > 1 and (section_stack[-1].level or 0) >= level:
                    section_stack.pop()
                parent = section_stack[-1]
                section = MarkdownNode(
                    kind="section",
                    line_start=line_number,
                    line_end=line_number,
                    title=title,
                    text=title,
                    level=level,
                    section_path=parent.section_path + (title,),
                )
                parent.children.append(section)
                section_stack.append(section)
                index += 1
                continue

            parent = section_stack[-1]

            fenced = self._consume_code_fence(lines, index, parent.section_path)
            if fenced is not None:
                parent.children.append(fenced[0])
                index = fenced[1]
                continue

            table = self._consume_table(lines, index, parent.section_path)
            if table is not None:
                parent.children.append(table[0])
                index = table[1]
                continue

            list_block = self._consume_list(lines, index, parent.section_path)
            if list_block is not None:
                parent.children.append(list_block[0])
                index = list_block[1]
                continue

            quote = self._consume_blockquote(lines, index, parent.section_path)
            if quote is not None:
                parent.children.append(quote[0])
                index = quote[1]
                continue

            paragraph = self._consume_paragraph(lines, index, parent.section_path)
            parent.children.append(paragraph[0])
            index = paragraph[1]

        self._close_sections(root)
        return MarkdownDocument(source_path=source_path, text=text, root=root)

    def _consume_code_fence(
        self,
        lines: list[str],
        index: int,
        section_path: tuple[str, ...],
    ) -> tuple[MarkdownNode, int] | None:
        opener = _FENCE_RE.match(lines[index].rstrip())
        if opener is None:
            return None
        fence = opener.group(1)
        fence_char = fence[0]
        language = opener.group(2).strip() or None
        start = index
        content: list[str] = []
        index += 1
        while index < len(lines):
            current = lines[index].rstrip()
            if current.startswith(fence_char * len(fence)):
                node = MarkdownNode(
                    kind="code_fence",
                    line_start=start + 1,
                    line_end=index + 1,
                    text="\n".join(content).strip("\n"),
                    language=language,
                    section_path=section_path,
                )
                return node, index + 1
            content.append(lines[index])
            index += 1
        node = MarkdownNode(
            kind="code_fence",
            line_start=start + 1,
            line_end=len(lines),
            text="\n".join(content).strip("\n"),
            language=language,
            section_path=section_path,
        )
        return node, len(lines)

    def _consume_table(
        self,
        lines: list[str],
        index: int,
        section_path: tuple[str, ...],
    ) -> tuple[MarkdownNode, int] | None:
        if index + 1 >= len(lines):
            return None
        header_line = lines[index]
        separator_line = lines[index + 1]
        if "|" not in header_line or "|" not in separator_line:
            return None
        if not _TABLE_SEPARATOR_RE.match(separator_line.replace(" ", "")):
            return None
        rows = [self._split_table_row(header_line)]
        end = index + 2
        body_rows: list[list[str]] = []
        while end < len(lines):
            current = lines[end]
            if not current.strip() or "|" not in current:
                break
            body_rows.append(self._split_table_row(current))
            end += 1
        node = MarkdownNode(
            kind="table",
            line_start=index + 1,
            line_end=end,
            header=rows[0],
            rows=body_rows,
            text="\n".join(lines[index:end]),
            section_path=section_path,
        )
        return node, end

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [cell.strip() for cell in stripped.split("|")]

    def _consume_list(
        self,
        lines: list[str],
        index: int,
        section_path: tuple[str, ...],
    ) -> tuple[MarkdownNode, int] | None:
        match = _ULIST_RE.match(lines[index]) or _OLIST_RE.match(lines[index])
        if match is None:
            return None
        ordered = bool(_OLIST_RE.match(lines[index]))
        start = index
        list_node = MarkdownNode(
            kind="list",
            line_start=index + 1,
            line_end=index + 1,
            ordered=ordered,
            section_path=section_path,
        )
        item_index = index
        while item_index < len(lines):
            line = lines[item_index]
            item_match = _ULIST_RE.match(line) if not ordered else _OLIST_RE.match(line)
            if item_match is None:
                break
            indent = len(item_match.group(1).replace("\t", "    "))
            item_text = item_match.group(3).strip()
            task_match = _TASK_RE.match(item_text)
            checked: bool | None = None
            if task_match is not None:
                checked = task_match.group(1).lower() == "x"
                item_text = task_match.group(2).strip()
            continuation: list[str] = []
            lookahead = item_index + 1
            while lookahead < len(lines):
                current = lines[lookahead]
                current_indent = len(current) - len(current.lstrip(" \t"))
                if not current.strip():
                    continuation.append("")
                    lookahead += 1
                    continue
                if _HEADING_RE.match(current) or _FENCE_RE.match(current.rstrip()):
                    break
                if _ULIST_RE.match(current) or _OLIST_RE.match(current):
                    break
                if current.lstrip().startswith(">") or self._looks_like_table_row(current):
                    break
                if current_indent <= indent:
                    break
                continuation.append(current.strip())
                lookahead += 1
            content = "\n".join([item_text, *continuation]).strip()
            list_node.children.append(
                MarkdownNode(
                    kind="list_item",
                    line_start=item_index + 1,
                    line_end=max(item_index + 1, lookahead),
                    text=content,
                    ordered=ordered,
                    checked=checked,
                    section_path=section_path,
                )
            )
            item_index = lookahead
        list_node.line_end = max(list_node.line_start, item_index)
        return list_node, item_index

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        stripped = line.strip()
        return "|" in stripped and not stripped.startswith("```") and not stripped.startswith("~~~")

    def _consume_blockquote(
        self,
        lines: list[str],
        index: int,
        section_path: tuple[str, ...],
    ) -> tuple[MarkdownNode, int] | None:
        if not lines[index].lstrip().startswith(">"):
            return None
        start = index
        content: list[str] = []
        while index < len(lines):
            line = lines[index]
            if not line.lstrip().startswith(">"):
                break
            text = line.lstrip()[1:]
            content.append(text.lstrip())
            index += 1
        node = MarkdownNode(
            kind="blockquote",
            line_start=start + 1,
            line_end=index,
            text="\n".join(content).strip(),
            section_path=section_path,
        )
        return node, index

    def _consume_paragraph(
        self,
        lines: list[str],
        index: int,
        section_path: tuple[str, ...],
    ) -> tuple[MarkdownNode, int]:
        start = index
        content: list[str] = []
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                break
            if _HEADING_RE.match(line) or _FENCE_RE.match(line.rstrip()):
                break
            if _ULIST_RE.match(line) or _OLIST_RE.match(line):
                break
            if line.lstrip().startswith(">"):
                break
            if self._consume_table(lines, index, section_path) is not None:
                break
            content.append(line.strip())
            index += 1
        node = MarkdownNode(
            kind="paragraph",
            line_start=start + 1,
            line_end=max(start + 1, index),
            text="\n".join(content).strip(),
            section_path=section_path,
        )
        return node, index

    def _close_sections(self, node: MarkdownNode) -> int:
        if not node.children:
            return node.line_end
        end = node.line_end
        for child in node.children:
            end = max(end, self._close_sections(child))
        node.line_end = max(node.line_end, end)
        return node.line_end


def parse_markdown(text: str, *, file_path: str, profile: str | None = None) -> ParsedMarkdownView:
    """Compatibility helper returning a flattened document summary."""
    document = MarkdownStructureParser().parse(text, source_path=file_path)
    nodes = [node for node in document.walk() if node.kind == "section"]
    blocks = [
        node
        for node in document.walk()
        if node.kind in {"paragraph", "list_item", "blockquote", "table"}
    ]
    equations = [
        node
        for node in document.walk()
        if node.kind == "code_fence" and (node.language or "").lower() in {"math", "latex"}
    ]
    code_fences = [node for node in document.walk() if node.kind == "code_fence"]
    return ParsedMarkdownView(
        profile=profile or "readme",
        nodes=nodes,
        blocks=blocks,
        equations=equations,
        code_fences=code_fences,
    )
