"""Recover runtime symbols from test output and native wrapper surfaces."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_TRACEBACK_FRAME_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+), in (?P<symbol>[^\n]+)'
)
_PYTEST_NODE_RE = re.compile(
    r"(?m)^(?P<nodeid>(?:tests?/)?[A-Za-z0-9_./-]+::[A-Za-z0-9_\[\]-]+)"
)
_LIB_SYMBOL_PATTERNS = (
    re.compile(r'getattr\(\s*lib\s*,\s*"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)"\s*,'),
    re.compile(r'hasattr\(\s*lib\s*,\s*"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)"\s*\)'),
    re.compile(r'lib\.(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\.(?:argtypes|restype)\b'),
    re.compile(
        r'_read_(?:char_ptr|int)_symbol\(\s*lib\s*,\s*"(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)"'
    ),
)
_EXPORT_SYMBOL_RE = re.compile(
    r'(?:extern\s+"C"\s+)?(?:const\s+)?(?:[A-Za-z_:][A-Za-z0-9_:<>\s\*&,\[\]]*?\s+)?'
    r'(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*\('
)
_LINE_EXPORT_RE = re.compile(
    r'(?m)^[ \t]*(?:inline\s+)?(?:static\s+)?(?:constexpr\s+)?(?:extern\s+"C"\s+)?'
    r'(?:const\s+)?(?:[A-Za-z_:][A-Za-z0-9_:<>\s\*&,\[\]]*?\s+)?'
    r'(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)\s*\('
)
_CONTROL_SYMBOLS = {"if", "for", "while", "switch", "return", "sizeof"}
_WRAPPER_SUFFIXES = (".py",)
_NATIVE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx")


@dataclass(frozen=True, slots=True)
class RuntimeSymbolMatch:
    """A recovered runtime symbol reference."""

    file_path: str
    symbol: str
    line: int
    confidence: float
    source: str
    runtime_hint: str
    corpus_id: str = "primary"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "symbol": self.symbol,
            "line": self.line,
            "confidence": self.confidence,
            "source": self.source,
            "runtime_hint": self.runtime_hint,
            "corpus_id": self.corpus_id,
            "qualified_symbol_id": f"{self.corpus_id}:{self.file_path}:runtime_symbol:{self.symbol}",
        }


class RuntimeSymbolResolver:
    """Resolve runtime frames and failing tests into repo symbols."""

    def __init__(self, repo_root: str | Path = ".") -> None:
        self.repo_root = Path(repo_root).resolve()

    def resolve_output(self, text: str) -> list[dict[str, Any]]:
        matches: list[RuntimeSymbolMatch] = []
        matches.extend(self._resolve_traceback_frames(text))
        matches.extend(self._resolve_pytest_nodes(text))
        deduped: dict[tuple[str, str, int], RuntimeSymbolMatch] = {}
        for match in matches:
            key = (match.file_path, match.symbol, match.line)
            incumbent = deduped.get(key)
            if incumbent is None or incumbent.confidence < match.confidence:
                deduped[key] = match
        return sorted(
            (item.to_dict() for item in deduped.values()),
            key=lambda entry: (str(entry.get("file_path") or ""), int(entry.get("line") or 0), str(entry.get("symbol") or "")),
        )

    def build_symbol_manifest(self, *, persist: bool = False) -> dict[str, Any]:
        """Build a static wrapper-to-export manifest for native runtime symbols."""
        referenced = self._wrapper_symbol_references()
        exported = self._native_exported_symbols()
        referenced_names = {item["symbol"] for item in referenced}
        exported_names = set(exported)
        matched_names = referenced_names.intersection(exported_names)
        unresolved = sorted(referenced_names - exported_names)
        payload = {
            "referenced_symbol_count": len(referenced_names),
            "exported_symbol_count": len(exported_names),
            "matched_symbol_count": len(matched_names),
            "coverage_percent": round(
                (len(matched_names) / max(len(referenced_names), 1)) * 100.0,
                1,
            ),
            "referenced_symbols": referenced,
            "exported_symbols": sorted(exported_names),
            "matched_symbols": sorted(matched_names),
            "unresolved_symbols": unresolved,
        }
        if persist:
            artifact_path = self.repo_root / ".saguaro" / "runtime_symbols.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["artifact_path"] = str(artifact_path)
        return payload

    def _resolve_traceback_frames(self, text: str) -> list[RuntimeSymbolMatch]:
        matches: list[RuntimeSymbolMatch] = []
        for frame in _TRACEBACK_FRAME_RE.finditer(text):
            file_path = self._normalize_repo_path(frame.group("file"))
            if not file_path:
                continue
            line = int(frame.group("line"))
            runtime_hint = frame.group("symbol").strip()
            resolved_symbol = self._symbol_for_line(file_path, line, runtime_hint)
            matches.append(
                RuntimeSymbolMatch(
                    file_path=file_path,
                    symbol=resolved_symbol or runtime_hint,
                    line=line,
                    confidence=0.93 if resolved_symbol and resolved_symbol != runtime_hint else 0.78,
                    source="traceback",
                    runtime_hint=runtime_hint,
                )
            )
        return matches

    def _resolve_pytest_nodes(self, text: str) -> list[RuntimeSymbolMatch]:
        matches: list[RuntimeSymbolMatch] = []
        for node in _PYTEST_NODE_RE.finditer(text):
            nodeid = node.group("nodeid")
            file_part = nodeid.split("::", 1)[0]
            file_path = self._normalize_repo_path(file_part)
            if not file_path:
                continue
            symbol_hint = nodeid.split("::")[-1]
            matches.append(
                RuntimeSymbolMatch(
                    file_path=file_path,
                    symbol=symbol_hint,
                    line=1,
                    confidence=0.74,
                    source="pytest",
                    runtime_hint=nodeid,
                )
            )
        return matches

    def _normalize_repo_path(self, raw_path: str) -> str | None:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                return candidate.resolve().relative_to(self.repo_root).as_posix()
            except ValueError:
                return None
        candidate = (self.repo_root / candidate).resolve()
        if not candidate.exists():
            return None
        return candidate.relative_to(self.repo_root).as_posix()

    def _symbol_for_line(
        self,
        file_path: str,
        line: int,
        runtime_hint: str,
    ) -> str | None:
        absolute_path = self.repo_root / file_path
        if absolute_path.suffix != ".py" or not absolute_path.exists():
            return runtime_hint or None
        try:
            tree = ast.parse(absolute_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return runtime_hint or None

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []
                self.best: tuple[int, str] | None = None

            def _consider(self, node: ast.AST, name: str) -> None:
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", start)
                if start is None or end is None or not (start <= line <= end):
                    return
                qualified = ".".join([*self.stack, name]) if self.stack else name
                span = end - start
                if self.best is None or span < self.best[0]:
                    self.best = (span, qualified)

            def visit_ClassDef(self, node: ast.ClassDef) -> Any:
                self._consider(node, node.name)
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                self._consider(node, node.name)
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                self.visit_FunctionDef(node)

        visitor = Visitor()
        visitor.visit(tree)
        return visitor.best[1] if visitor.best else runtime_hint or None

    def _wrapper_symbol_references(self) -> list[dict[str, Any]]:
        matches: dict[tuple[str, str], dict[str, Any]] = {}
        native_root = self.repo_root / "core" / "native"
        if not native_root.exists():
            return []
        for path in native_root.rglob("*"):
            if not path.is_file() or path.suffix not in _WRAPPER_SUFFIXES:
                continue
            rel_path = path.relative_to(self.repo_root).as_posix()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in _LIB_SYMBOL_PATTERNS:
                for match in pattern.finditer(text):
                    symbol = str(match.group("symbol") or "").strip()
                    if not symbol:
                        continue
                    key = (rel_path, symbol)
                    if key in matches:
                        continue
                    line = text.count("\n", 0, match.start()) + 1
                    matches[key] = {
                        "file": rel_path,
                        "symbol": symbol,
                        "line": line,
                        "corpus_id": "primary",
                        "qualified_symbol_id": f"primary:{rel_path}:runtime_symbol:{symbol}",
                    }
        return sorted(
            matches.values(),
            key=lambda item: (str(item["file"]), str(item["symbol"])),
        )

    def _native_exported_symbols(self) -> list[str]:
        symbols: set[str] = set()
        native_root = self.repo_root / "core" / "native"
        if not native_root.exists():
            return []
        for path in native_root.rglob("*"):
            if not path.is_file() or path.suffix not in _NATIVE_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            in_extern_block = False
            brace_depth = 0
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                opened_block = False
                if not in_extern_block and 'extern "C"' in line and "{" in line:
                    in_extern_block = True
                    brace_depth = line.count("{") - line.count("}")
                    opened_block = True
                if in_extern_block or 'extern "C"' in line:
                    for match in _LINE_EXPORT_RE.finditer(line):
                        symbol = str(match.group("symbol") or "").strip()
                        if symbol and symbol not in _CONTROL_SYMBOLS:
                            symbols.add(symbol)
                if in_extern_block and not opened_block:
                    brace_depth += line.count("{") - line.count("}")
                    if brace_depth <= 0:
                        in_extern_block = False
                        brace_depth = 0
                elif in_extern_block and brace_depth <= 0:
                    in_extern_block = False
                    brace_depth = 0
            for match in _LINE_EXPORT_RE.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.start())
                if line_end < 0:
                    line_end = len(text)
                line = text[line_start:line_end]
                if 'extern "C"' not in line and "extern" not in line:
                    continue
                symbol = str(match.group("symbol") or "").strip()
                if symbol and symbol not in _CONTROL_SYMBOLS:
                    symbols.add(symbol)
        return sorted(symbols)
