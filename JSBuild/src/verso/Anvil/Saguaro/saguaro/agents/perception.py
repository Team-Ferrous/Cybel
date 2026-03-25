"""Utilities for perception."""

import ast
import contextlib
import json
import logging
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from saguaro.analysis.cfg_builder import CFGBuilder
from saguaro.analysis.complexity_analyzer import ComplexityAnalyzer
from saguaro.analysis.ffi_scanner import FFIScanner
from saguaro.analysis.flop_counter import FLOPCounter
from saguaro.analysis.trace_output import TraceOutputFormatter
from saguaro.parsing.parser import SAGUAROParser

logger = logging.getLogger(__name__)

try:
    from tree_sitter_languages import get_parser

    TREE_SITTER_AVAILABLE = True
except Exception:
    get_parser = None
    TREE_SITTER_AVAILABLE = False


_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".vue": "javascript",
    ".svelte": "javascript",
    ".astro": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "java",
    ".sc": "java",
    ".groovy": "java",
    ".fs": "csharp",
    ".fsx": "csharp",
    ".vb": "csharp",
    ".swift": "swift",
    ".dart": "swift",
    ".zig": "rust",
    ".nim": "rust",
    ".php": "php",
    ".phtml": "php",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".ru": "ruby",
    ".pl": "perl",
    ".pm": "perl",
    ".lua": "lua",
    ".ex": "ruby",
    ".exs": "ruby",
    ".erl": "ruby",
    ".hrl": "ruby",
    ".clj": "ruby",
    ".cljs": "ruby",
    ".cljc": "ruby",
    ".edn": "ruby",
    ".lisp": "ruby",
    ".lsp": "ruby",
    ".scm": "ruby",
    ".rkt": "ruby",
    ".r": "ruby",
    ".jl": "ruby",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "shell",
    ".psm1": "shell",
    ".bat": "shell",
    ".cmd": "shell",
    ".cmake": "cmake",
    ".sql": "sql",
    ".proto": "proto",
    ".thrift": "proto",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".less": "css",
    ".gradle": "java",
    ".sbt": "java",
    ".bazel": "bazel",
    ".bzl": "bazel",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".env": "ini",
    ".properties": "ini",
    ".v": "cpp",
    ".sv": "cpp",
    ".vhd": "cpp",
    ".vhdl": "cpp",
    ".hs": "ruby",
    ".lhs": "ruby",
    ".ml": "ruby",
    ".mli": "ruby",
    ".f": "cpp",
    ".f90": "cpp",
    ".f95": "cpp",
    ".adb": "cpp",
    ".ads": "cpp",
    ".ada": "cpp",
    ".cob": "cpp",
    ".cbl": "cpp",
    ".sol": "cpp",
    ".vy": "cpp",
}
_LANGUAGE_BY_FILENAME = {
    "cmakelists.txt": "cmake",
    "makefile": "make",
    "dockerfile": "docker",
    "build": "bazel",
    "build.bazel": "bazel",
    "workspace": "bazel",
    "workspace.bazel": "bazel",
    "jenkinsfile": "java",
    "meson.build": "cmake",
    "meson_options.txt": "cmake",
    "rakefile": "ruby",
    "gemfile": "ruby",
    "vagrantfile": "ruby",
    "procfile": "shell",
    "justfile": "make",
    "gnumakefile": "make",
    ".env": "ini",
}


def _detect_language_for_path(path: str) -> str:
    content = ""
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            content = f.read(4096)
    except Exception:
        content = ""
    return SAGUAROParser().detect_language(path, content)


@dataclass
class SymbolLocation:
    """Provide SymbolLocation support."""

    file_path: str
    name: str
    symbol_type: str
    start_line: int
    end_line: int
    parent: str | None = None


class FileReader:
    """Reads complete files or selected line ranges for SSAI consumption."""

    def __init__(self, repo_path: str = ".") -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)

    def read(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Handle read."""
        full_path = self._resolve_path(file_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(full_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        start = 1 if start_line is None else max(1, start_line)
        end = total_lines if end_line is None else min(total_lines, end_line)

        if end < start:
            start, end = end, start

        content = "".join(lines[start - 1 : end])
        return {
            "path": os.path.relpath(full_path, self.repo_path),
            "content": content,
            "total_lines": total_lines,
            "range": [start, end],
        }

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.repo_path, path)


class DirectoryExplorer:
    """Directory and module-level structural exploration."""

    def __init__(self, repo_path: str = ".") -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)

    def list_directory(
        self,
        path: str,
        recursive: bool = False,
        extensions: list[str] | None = None,
    ) -> dict[str, Any]:
        """List directory."""
        base = self._resolve_path(path)
        if not os.path.exists(base):
            raise FileNotFoundError(f"Path not found: {path}")

        if os.path.isfile(base):
            rel = os.path.relpath(base, self.repo_path)
            return {
                "path": rel,
                "recursive": recursive,
                "entries": [{"path": rel, "type": "file"}],
            }

        ext_filter = set(extensions or [])
        entries: list[dict[str, str]] = []

        for root, dirs, files in os.walk(base):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", ".saguaro", "venv", "node_modules", "__pycache__"}
            ]

            rel_root = os.path.relpath(root, self.repo_path)
            if rel_root != ".":
                entries.append({"path": rel_root, "type": "directory"})

            for name in files:
                if ext_filter:
                    _, ext = os.path.splitext(name)
                    if ext not in ext_filter:
                        continue
                full = os.path.join(root, name)
                entries.append(
                    {"path": os.path.relpath(full, self.repo_path), "type": "file"}
                )

            if not recursive:
                break

        entries.sort(key=lambda e: (e["type"], e["path"]))
        return {
            "path": os.path.relpath(base, self.repo_path),
            "recursive": recursive,
            "entries": entries,
        }

    def module_structure(self, path: str) -> dict[str, Any]:
        """Handle module structure."""
        base = self._resolve_path(path)
        if not os.path.exists(base):
            raise FileNotFoundError(f"Path not found: {path}")

        modules = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", "venv", "node_modules", "__pycache__"}
            ]
            rel_root = os.path.relpath(root, self.repo_path)
            is_package = "__init__.py" in files

            py_files = [f for f in files if f.endswith(".py") and f != "__init__.py"]
            init_exports = (
                self._parse_init_exports(os.path.join(root, "__init__.py"))
                if is_package
                else []
            )

            if is_package or py_files:
                modules.append(
                    {
                        "module_path": rel_root,
                        "is_package": is_package,
                        "python_files": sorted(py_files),
                        "exports": init_exports,
                    }
                )

        modules.sort(key=lambda m: m["module_path"])
        return {
            "path": os.path.relpath(base, self.repo_path),
            "modules": modules,
        }

    def _parse_init_exports(self, init_path: str) -> list[str]:
        if not os.path.exists(init_path):
            return []

        try:
            with open(init_path, encoding="utf-8", errors="ignore") as f:
                tree = ast.parse(f.read())

            exports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(
                                        elt.value, str
                                    ):
                                        exports.append(elt.value)
            return sorted(set(exports))
        except Exception:
            return []

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.repo_path, path)


class SkeletonGenerator:
    """Generates multi-language structural skeletons."""

    def __init__(self) -> None:
        self._parser = SAGUAROParser()

    TS_NODE_TYPES = {
        "python": {
            "function_definition": "function",
            "class_definition": "class",
        },
        "c": {
            "function_definition": "function",
            "struct_specifier": "struct",
        },
        "cpp": {
            "function_definition": "function",
            "class_specifier": "class",
            "struct_specifier": "struct",
            "namespace_definition": "namespace",
        },
        "javascript": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
            "lexical_declaration": "variable",
        },
        "typescript": {
            "function_declaration": "function",
            "method_definition": "method",
            "class_declaration": "class",
            "interface_declaration": "interface",
            "type_alias_declaration": "type",
            "enum_declaration": "enum",
            "lexical_declaration": "variable",
        },
    }

    def generate(self, file_path: str) -> dict[str, Any]:
        """Handle generate."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path = os.path.abspath(file_path)
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        language = self._parser.detect_language(file_path, content)
        entities = self._parser.parse_file(file_path)

        dependency_graph = {
            "imports": [],
            "exports": [],
            "internal_edges": [],
        }
        symbols: list[dict[str, Any]] = []
        for entity in entities:
            if entity.type == "dependency_graph":
                with contextlib.suppress(Exception):
                    payload = json.loads(entity.content or "{}")
                    if isinstance(payload, dict):
                        dependency_graph = {
                            "imports": list(payload.get("imports") or []),
                            "exports": list(payload.get("exports") or []),
                            "internal_edges": list(payload.get("internal_edges") or []),
                        }
                continue
            if entity.type in {"file", "file_summary"}:
                continue
            if entity.metadata.get("chunk_role") == "section":
                continue
            symbols.append(
                {
                    "name": entity.name,
                    "type": entity.type,
                    "line_start": entity.start_line,
                    "line_end": entity.end_line,
                }
            )

        module_constants = [
            symbol
            for symbol in symbols
            if symbol.get("type") in {"constant", "variable"}
            and self._is_module_constant_name(str(symbol.get("name", "")))
        ]
        skeleton = {
            "type": "skeleton",
            "file_path": file_path,
            "language": language,
            "loc": len(content.splitlines()),
            "symbols": symbols,
            "imports": list(dependency_graph.get("imports") or []),
            "module_constants": module_constants,
            "dependency_graph": dependency_graph,
        }
        if not symbols and not self._parser.supports_structural_language(language):
            skeleton["note"] = "No explicit structural parser backend for this language."
        elif not symbols:
            skeleton["note"] = "No structural symbols found."
        return skeleton

    def _detect_language(self, path: str) -> str:
        return _detect_language_for_path(path)

    def _parse_python(self, content: str, skeleton: dict[str, Any]) -> None:
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            skeleton["error"] = str(e)
            return

        import_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ""
                for alias in node.names:
                    import_names.append(f"{module}.{alias.name}")

        symbols: list[dict[str, Any]] = []
        module_constants: list[dict[str, Any]] = []
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                constants = self._visit_python_constant(node)
                if constants:
                    symbols.extend(constants)
                    module_constants.extend(constants)
                continue
            symbol = self._visit_python_node(node)
            if symbol:
                symbols.append(symbol)

        exports = [
            symbol.get("name", "")
            for symbol in symbols
            if isinstance(symbol, dict) and symbol.get("name")
        ]
        edges = self._build_python_dependency_edges(
            tree=tree,
            exported_symbols=set(exports),
            constant_symbols={item.get("name", "") for item in module_constants},
        )

        skeleton["symbols"] = symbols
        skeleton["module_constants"] = module_constants
        skeleton["imports"] = sorted(set(import_names))
        skeleton["dependency_graph"] = {
            "imports": sorted(set(import_names)),
            "exports": exports,
            "internal_edges": edges,
        }

    def _visit_python_node(self, node: ast.AST) -> dict[str, Any] | None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._visit_python_function(node)
        if isinstance(node, ast.ClassDef):
            return self._visit_python_class(node)
        return None

    def _visit_python_function(self, node: ast.AST) -> dict[str, Any]:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        args = []
        for arg in node.args.args:
            name = arg.arg
            if arg.annotation and hasattr(ast, "unparse"):
                with contextlib.suppress(Exception):
                    name = f"{name}: {ast.unparse(arg.annotation)}"
            args.append(name)

        signature = (
            f"{'async def' if is_async else 'def'} {node.name}({', '.join(args)})"
        )
        if node.returns and hasattr(ast, "unparse"):
            with contextlib.suppress(Exception):
                signature += f" -> {ast.unparse(node.returns)}"

        return {
            "name": node.name,
            "type": "function",
            "signature": signature,
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno),
            "docstring": ast.get_docstring(node),
        }

    def _visit_python_class(self, node: ast.ClassDef) -> dict[str, Any]:
        children = []
        for item in node.body:
            child = self._visit_python_node(item)
            if child:
                if child["type"] == "function":
                    child["type"] = "method"
                children.append(child)

        return {
            "name": node.name,
            "type": "class",
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno),
            "docstring": ast.get_docstring(node),
            "children": children,
        }

    def _visit_python_constant(self, node: ast.AST) -> list[dict[str, Any]]:
        names: list[str] = []
        value_node = None

        if isinstance(node, ast.Assign):
            value_node = node.value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            value_node = node.value
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)

        if not names:
            return []

        symbols: list[dict[str, Any]] = []
        for name in names:
            if not self._is_module_constant_name(name):
                continue
            symbols.append(
                {
                    "name": name,
                    "type": "constant",
                    "signature": f"{name} = {self._python_value_preview(value_node)}",
                    "line_start": getattr(node, "lineno", 1),
                    "line_end": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                    "docstring": None,
                }
            )
        return symbols

    @staticmethod
    def _is_module_constant_name(name: str) -> bool:
        if not name or name.startswith("__"):
            return False
        return name.isupper() or name.endswith("_CONFIG")

    @staticmethod
    def _python_value_preview(value: ast.AST | None) -> str:
        if value is None:
            return "<unset>"
        if hasattr(ast, "unparse"):
            try:
                text = ast.unparse(value).replace("\n", " ").strip()
                if len(text) > 96:
                    return text[:93] + "..."
                return text
            except Exception:
                return "<expr>"
        return "<expr>"

    def _build_python_dependency_edges(
        self,
        tree: ast.Module,
        exported_symbols: set[str],
        constant_symbols: set[str],
    ) -> list[dict[str, Any]]:
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        if not exported_symbols:
            return []

        def add_edge(source: str, target: str, relation: str, line: int) -> None:
            if not source or not target or source == target:
                return
            key = (source, target, relation)
            if key not in edges:
                edges[key] = {
                    "from": source,
                    "to": target,
                    "relation": relation,
                    "line": int(line or 1),
                }

        def call_name(expr: ast.AST, *, class_name: str | None = None) -> str | None:
            if isinstance(expr, ast.Name):
                qualified = f"{class_name}.{expr.id}" if class_name else ""
                if qualified and qualified in exported_symbols:
                    return qualified
                return expr.id
            if isinstance(expr, ast.Attribute):
                if (
                    class_name
                    and isinstance(expr.value, ast.Name)
                    and expr.value.id in {"self", "cls"}
                ):
                    return f"{class_name}.{expr.attr}"
                return expr.attr
            return None

        def scan_body(
            source_name: str, nodes: list[ast.stmt], *, class_name: str | None = None
        ) -> None:
            for node in nodes:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        callee = call_name(inner.func, class_name=class_name)
                        if callee and callee in exported_symbols:
                            add_edge(
                                source_name,
                                callee,
                                "calls",
                                getattr(inner, "lineno", getattr(node, "lineno", 1)),
                            )
                    elif isinstance(inner, ast.Name) and isinstance(
                        inner.ctx, ast.Load
                    ):
                        if inner.id in constant_symbols:
                            add_edge(
                                source_name,
                                inner.id,
                                "reads",
                                getattr(inner, "lineno", getattr(node, "lineno", 1)),
                            )

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scan_body(node.name, list(node.body))
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scan_body(
                            f"{node.name}.{child.name}",
                            list(child.body),
                            class_name=node.name,
                        )

        out = list(edges.values())
        out.sort(
            key=lambda item: (
                item.get("from", ""),
                item.get("line", 0),
                item.get("to", ""),
            )
        )
        return out

    def _extract_ts_symbols(
        self, root_node: Any, content: str, language: str
    ) -> list[dict[str, Any]]:
        symbol_map = self.TS_NODE_TYPES.get(language, {})
        symbols = []

        def visit(node: Any, parent: str | None = None) -> None:
            node_type = getattr(node, "type", "")
            symbol_type = symbol_map.get(node_type)

            if symbol_type:
                name = self._node_name(node, content)
                line_start = getattr(node, "start_point", (0, 0))[0] + 1
                line_end = getattr(node, "end_point", (0, 0))[0] + 1
                item = {
                    "name": name or f"<{symbol_type}>",
                    "type": symbol_type,
                    "line_start": line_start,
                    "line_end": line_end,
                }
                if parent:
                    item["parent"] = parent
                symbols.append(item)
                parent = name or parent

            for child in getattr(node, "children", []):
                visit(child, parent)

        visit(root_node)

        dedup = {}
        for sym in symbols:
            key = (sym.get("name"), sym.get("type"), sym.get("line_start"))
            dedup[key] = sym
        out = list(dedup.values())
        out.sort(key=lambda s: (s.get("line_start", 0), s.get("name", "")))
        return out

    def _node_name(self, node: Any, content: str) -> str | None:
        name_node = None
        try:
            name_node = node.child_by_field_name("name")
        except Exception:
            name_node = None

        if name_node is None:
            for child in getattr(node, "children", []):
                if child.type in {
                    "identifier",
                    "type_identifier",
                    "property_identifier",
                }:
                    name_node = child
                    break

        if name_node is None:
            return None

        return content[name_node.start_byte : name_node.end_byte].strip() or None

    def _extract_imports(self, content: str, language: str) -> list[str]:
        imports: list[str] = []
        lines = content.splitlines()

        if language == "python":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    imports.append(stripped)
        elif language in {"javascript", "typescript"}:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("export "):
                    imports.append(stripped)
        elif language in {"c", "cpp"}:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#include"):
                    imports.append(stripped)
        elif language in {"go", "java", "kotlin", "swift"}:
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("import "):
                    imports.append(stripped)
        elif language == "rust":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("use "):
                    imports.append(stripped)
        elif language == "csharp":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("using "):
                    imports.append(stripped)
        elif language == "ruby":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("require ") or stripped.startswith(
                    "require_relative "
                ):
                    imports.append(stripped)
        elif language == "php":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("use ") or stripped.startswith(
                    ("require", "include")
                ):
                    imports.append(stripped)
        elif language == "proto":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("import "):
                    imports.append(stripped)
        elif language == "graphql":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#import "):
                    imports.append(stripped)
        elif language == "make":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("include ", "-include ")):
                    imports.append(stripped)
        elif language == "docker":
            for line in lines:
                stripped = line.strip()
                if stripped.upper().startswith("FROM "):
                    imports.append(stripped)
        elif language == "bazel":
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("load(", "package(", "workspace(")):
                    imports.append(stripped)
        elif language == "shell":
            for line in lines:
                stripped = line.strip()
                if (
                    stripped.startswith("source ")
                    or stripped.startswith(". ")
                    or stripped.startswith("export ")
                ):
                    imports.append(stripped)
        elif language == "cmake":
            for line in lines:
                stripped = line.strip()
                lowered = stripped.lower()
                if lowered.startswith(
                    ("include(", "find_package(", "add_subdirectory(")
                ):
                    imports.append(stripped)

        return sorted(set(imports))

    def _regex_fallback_symbols(
        self, content: str, language: str
    ) -> list[dict[str, Any]]:
        symbols: list[dict[str, Any]] = []
        lines = content.splitlines()

        patterns = []
        if language in {
            "c",
            "cpp",
            "go",
            "rust",
            "java",
            "csharp",
            "kotlin",
            "swift",
            "php",
        }:
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:class|struct|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
                    ),
                    "class",
                ),
                (
                    re.compile(
                        r"^\s*[A-Za-z_][A-Za-z0-9_:<>\s\*&]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{"
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*func\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                    "function",
                ),
            ]
        elif language in {"javascript", "typescript"}:
            patterns = [
                (re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"), "class"),
                (
                    re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
                    "function",
                ),
                (
                    re.compile(
                        r"^\s*(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\("
                    ),
                    "function",
                ),
            ]
        elif language in {"ruby", "perl", "lua"}:
            patterns = [
                (
                    re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_!?=]*)"),
                    "function",
                ),
                (
                    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_:]*)"),
                    "class",
                ),
            ]
        elif language == "sql":
            patterns = [
                (
                    re.compile(
                        r"^\s*create\s+(?:or\s+replace\s+)?(?:function|procedure|view|table)\s+([A-Za-z_][A-Za-z0-9_.]*)",
                        re.IGNORECASE,
                    ),
                    "symbol",
                )
            ]
        elif language == "proto":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:message|service|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
                    ),
                    "type",
                )
            ]
        elif language == "graphql":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:type|interface|enum|input|scalar|union)\s+([A-Za-z_][A-Za-z0-9_]*)"
                    ),
                    "type",
                )
            ]
        elif language == "make":
            patterns = [
                (
                    re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*:(?![=])"),
                    "target",
                ),
                (
                    re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*[:+?]?=\s*"),
                    "variable",
                ),
            ]
        elif language == "docker":
            patterns = [
                (
                    re.compile(
                        r"^\s*(FROM|RUN|CMD|ENTRYPOINT|COPY|ADD|ENV|ARG|WORKDIR|EXPOSE)\b",
                        re.IGNORECASE,
                    ),
                    "instruction",
                )
            ]
        elif language == "bazel":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:cc_|py_|java_|go_|rust_|sh_|genrule|test_suite)[A-Za-z0-9_]*\s*\(",
                        re.IGNORECASE,
                    ),
                    "rule",
                )
            ]
        elif language == "shell":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(\s*\))?\s*\{"
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*.+$"),
                    "constant",
                ),
            ]
        elif language == "cmake":
            patterns = [
                (
                    re.compile(
                        r"^\s*(?:add_library|add_executable)\s*\(\s*([A-Za-z_][A-Za-z0-9_.-]*)",
                        re.IGNORECASE,
                    ),
                    "target",
                ),
                (
                    re.compile(
                        r"^\s*(?:function|macro)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)",
                        re.IGNORECASE,
                    ),
                    "function",
                ),
                (
                    re.compile(r"^\s*set\s*\(\s*([A-Z][A-Z0-9_]+)", re.IGNORECASE),
                    "variable",
                ),
            ]
        else:
            patterns = [
                (re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("), "function")
            ]

        for idx, line in enumerate(lines, start=1):
            for pattern, sym_type in patterns:
                match = pattern.search(line)
                if match:
                    symbols.append(
                        {
                            "name": match.group(1),
                            "type": sym_type,
                            "line_start": idx,
                            "line_end": idx,
                        }
                    )
                    break

        return symbols


class SliceGenerator:
    """Generates symbol slices with imports and dependency context."""

    def __init__(self, repo_path: str, *, saguaro_dir: str | None = None) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.abspath(
            saguaro_dir or os.path.join(self.repo_path, ".saguaro")
        )
        self.metadata_path = os.path.join(self.saguaro_dir, "vectors", "metadata.json")
        self.metadata: list[dict[str, Any]] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = []

    def generate(
        self,
        symbol_name: str,
        depth: int = 1,
        preferred_file: str | None = None,
    ) -> dict[str, Any]:
        """Handle generate."""
        depth = max(1, int(depth or 1))
        preferred_abs = None
        if preferred_file:
            preferred_abs = (
                preferred_file
                if os.path.isabs(preferred_file)
                else os.path.join(self.repo_path, preferred_file)
            )

        candidates = self._resolve_symbol_candidates(
            symbol_name, preferred_file=preferred_abs
        )
        if not candidates:
            return {
                "error": "Symbol not found",
                "type": "INDEX_MISS",
                "symbol": symbol_name,
                "suggestion": (
                    f"Symbol '{symbol_name}' was not found. Try `saguaro query \"{symbol_name}\" --k 5` "
                    "or rebuild index with `saguaro index --path .`."
                ),
                "fallback_allowed": False,
                "recovery_steps": [
                    f'saguaro query "{symbol_name}" --k 5',
                    "saguaro health",
                    "saguaro index --path .",
                ],
            }
        if preferred_abs is None and len(candidates) > 1:
            return {
                "error": "Symbol is ambiguous in the primary corpus.",
                "type": "SYMBOL_AMBIGUOUS",
                "symbol": symbol_name,
                "corpus_id": "primary",
                "matches": [self._location_match_payload(item) for item in candidates[:10]],
            }
        location = candidates[0]

        source = self._read_lines(
            location.file_path, location.start_line, location.end_line
        )
        imports = self._get_imports(location.file_path)
        focus_path = os.path.relpath(location.file_path, self.repo_path)
        focus_qid = self._qualified_symbol_id(location)

        result: dict[str, Any] = {
            "type": "slice",
            "focus_symbol": symbol_name,
            "corpus_id": "primary",
            "qualified_symbol_id": focus_qid,
            "depth": depth,
            "content": [
                {
                    "role": "focus",
                    "name": location.name,
                    "file": focus_path,
                    "type": location.symbol_type,
                    "line_start": location.start_line,
                    "line_end": location.end_line,
                    "corpus_id": "primary",
                    "qualified_symbol_id": focus_qid,
                    "code": source,
                }
            ],
        }

        if location.parent:
            parent_loc = self._resolve_symbol(
                location.parent, preferred_file=location.file_path
            )
            if parent_loc:
                parent_code = self._read_lines(
                    parent_loc.file_path, parent_loc.start_line, parent_loc.end_line
                )
                result["content"].append(
                    {
                        "role": "dependency",
                        "relation": "parent_class",
                        "name": parent_loc.name,
                        "file": os.path.relpath(parent_loc.file_path, self.repo_path),
                        "type": parent_loc.symbol_type,
                        "corpus_id": "primary",
                        "qualified_symbol_id": self._qualified_symbol_id(parent_loc),
                        "code": parent_code,
                    }
                )

        for imp in imports:
            result["content"].append(
                {
                    "role": "dependency",
                    "relation": "import",
                    "name": imp,
                    "file": os.path.relpath(location.file_path, self.repo_path),
                    "signature": imp,
                }
            )

        if depth > 1:
            for dep in self._find_called_symbols(source, location.file_path, depth - 1):
                result["content"].append(dep)

        return result

    def _resolve_symbol(
        self, symbol_name: str, preferred_file: str | None = None
    ) -> SymbolLocation | None:
        candidates = self._resolve_symbol_candidates(
            symbol_name, preferred_file=preferred_file
        )
        return candidates[0] if candidates else None

    def _resolve_symbol_candidates(
        self, symbol_name: str, preferred_file: str | None = None
    ) -> list[SymbolLocation]:
        class_name = None
        member_name = None
        candidates: list[SymbolLocation] = []
        seen: set[tuple[str, str, str, int, int, str | None]] = set()

        def add_candidate(loc: SymbolLocation | None) -> None:
            if loc is None:
                return
            key = (
                os.path.abspath(loc.file_path),
                loc.name,
                loc.symbol_type,
                int(loc.start_line),
                int(loc.end_line),
                loc.parent,
            )
            if key in seen:
                return
            seen.add(key)
            candidates.append(loc)

        if "." in symbol_name:
            parts = [p for p in symbol_name.split(".") if p]
            if len(parts) >= 2:
                class_name = parts[-2]
                member_name = parts[-1]
            else:
                lhs, rhs = symbol_name.rsplit(".", 1)
                class_name = lhs
                member_name = rhs

            if preferred_file and member_name and class_name:
                add_candidate(
                    self._find_in_python_file(
                    preferred_file, member_name, class_name=class_name
                    )
                )

            # Support file-qualified targets like "core/agent.py._stream_response"
            # or module-qualified targets like "core.agent.BaseAgent._stream_response".
            hinted_file = (
                self._resolve_file_hint(parts[:-1]) if len(parts) > 1 else None
            )
            if hinted_file and member_name:
                add_candidate(self._find_in_file(hinted_file, member_name))

            # Root fix for symbol divergence: resolve class.method without requiring
            # a preferred file by searching class ownership across repository files.
            if class_name and member_name:
                add_candidate(
                    self._find_class_member_globally(
                    class_name=class_name,
                    member_name=member_name,
                    preferred_file=preferred_file,
                    )
                )

        for item in self.metadata:
            item_name = item.get("name")
            item_parent = item.get("parent")
            is_exact = item_name == symbol_name
            is_class_member = (
                class_name is not None
                and member_name is not None
                and item_name == member_name
                and item_parent == class_name
            )
            if is_exact or is_class_member:
                file_path = item.get("file")
                if not file_path:
                    continue
                abs_path = (
                    file_path
                    if os.path.isabs(file_path)
                    else os.path.join(self.repo_path, file_path)
                )
                abs_path = os.path.abspath(abs_path)
                if not abs_path.startswith(self.repo_path):
                    continue
                if preferred_file and abs_path != os.path.abspath(preferred_file):
                    continue
                if is_class_member and member_name and abs_path.endswith(".py"):
                    precise = self._find_in_python_file(
                        file_path=abs_path,
                        symbol_name=member_name,
                        class_name=class_name,
                    )
                else:
                    precise = self._find_in_file(
                        abs_path,
                        member_name if is_class_member and member_name else symbol_name,
                    )
                if precise is not None:
                    add_candidate(precise)
                    continue
                start_line = int(item.get("line", 1))
                end_line = int(item.get("end_line", item.get("line", 1)))
                if end_line < start_line:
                    end_line = start_line
                add_candidate(
                    SymbolLocation(
                        file_path=abs_path,
                        name=item.get("name", symbol_name),
                        symbol_type=item.get("type", "symbol"),
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", ".saguaro", "venv", "node_modules", "__pycache__"}
            ]
            for name in files:
                file_path = os.path.join(root, name)
                if _detect_language_for_path(file_path) == "unknown":
                    continue
                if preferred_file and os.path.abspath(file_path) != os.path.abspath(
                    preferred_file
                ):
                    continue

                loc = self._find_in_file(file_path, symbol_name)
                add_candidate(loc)

        candidates.sort(
            key=lambda item: (
                os.path.relpath(item.file_path, self.repo_path),
                int(item.start_line),
                item.name,
            )
        )
        return candidates

    def _location_match_payload(self, location: SymbolLocation) -> dict[str, Any]:
        return {
            "corpus_id": "primary",
            "name": location.name,
            "type": location.symbol_type,
            "file": os.path.relpath(location.file_path, self.repo_path),
            "line_start": int(location.start_line),
            "line_end": int(location.end_line),
            "qualified_symbol_id": self._qualified_symbol_id(location),
        }

    def _qualified_symbol_id(self, location: SymbolLocation) -> str:
        rel_path = os.path.relpath(location.file_path, self.repo_path).replace("\\", "/")
        return f"primary:{rel_path}:{location.symbol_type}:{location.name}"

    def _resolve_file_hint(self, parts: list[str]) -> str | None:
        if not parts:
            return None

        joined = ".".join(parts)
        slash_joined = "/".join(parts)
        candidates = [joined, slash_joined]
        if not joined.endswith(".py"):
            candidates.extend([f"{joined}.py", f"{slash_joined}.py"])

        for candidate in candidates:
            full = (
                candidate
                if os.path.isabs(candidate)
                else os.path.join(self.repo_path, candidate)
            )
            if os.path.exists(full) and os.path.isfile(full):
                return os.path.abspath(full)
        return None

    def _find_class_member_globally(
        self,
        class_name: str,
        member_name: str,
        preferred_file: str | None = None,
    ) -> SymbolLocation | None:
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", ".saguaro", "venv", "node_modules", "__pycache__"}
            ]
            for name in files:
                if not name.endswith(".py"):
                    continue
                file_path = os.path.join(root, name)
                if preferred_file and os.path.abspath(file_path) != os.path.abspath(
                    preferred_file
                ):
                    continue

                loc = self._find_in_python_file(
                    file_path=file_path,
                    symbol_name=member_name,
                    class_name=class_name,
                )
                if loc:
                    return loc
        return None

    def _find_in_file(self, file_path: str, symbol_name: str) -> SymbolLocation | None:
        lang = _detect_language_for_path(file_path)
        if lang == "python":
            return self._find_in_python_file(file_path, symbol_name)
        return self._find_in_tree_sitter_file(file_path, symbol_name, lang)

    def _find_in_python_file(
        self, file_path: str, symbol_name: str, class_name: str | None = None
    ) -> SymbolLocation | None:
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                tree = ast.parse(f.read())
        except Exception:
            return None

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if node.name == symbol_name:
                    return SymbolLocation(
                        file_path=file_path,
                        name=node.name,
                        symbol_type="class",
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                    )

                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == symbol_name
                    ):
                        if class_name and node.name != class_name:
                            continue
                        return SymbolLocation(
                            file_path=file_path,
                            name=child.name,
                            symbol_type="method",
                            start_line=child.lineno,
                            end_line=getattr(child, "end_lineno", child.lineno),
                            parent=node.name,
                        )

            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == symbol_name
            ):
                return SymbolLocation(
                    file_path=file_path,
                    name=node.name,
                    symbol_type="function",
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                )

            if class_name is None and isinstance(node, (ast.Assign, ast.AnnAssign)):
                assignment_names: list[str] = []
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            assignment_names.append(target.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ):
                    assignment_names.append(node.target.id)
                if symbol_name in assignment_names:
                    return SymbolLocation(
                        file_path=file_path,
                        name=symbol_name,
                        symbol_type="constant",
                        start_line=getattr(node, "lineno", 1),
                        end_line=getattr(
                            node, "end_lineno", getattr(node, "lineno", 1)
                        ),
                    )

        return None

    def _find_in_tree_sitter_file(
        self, file_path: str, symbol_name: str, language: str
    ) -> SymbolLocation | None:
        if not TREE_SITTER_AVAILABLE or language not in {
            "javascript",
            "typescript",
            "c",
            "cpp",
        }:
            return None

        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            parser = get_parser(language)
            tree = parser.parse(content.encode("utf-8"))
        except Exception:
            return None

        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            name = self._node_name(node, content)
            if name == symbol_name:
                return SymbolLocation(
                    file_path=file_path,
                    name=symbol_name,
                    symbol_type=node.type,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                )
            stack.extend(reversed(getattr(node, "children", [])))

        return None

    def _node_name(self, node: Any, content: str) -> str | None:
        try:
            name_node = node.child_by_field_name("name")
        except Exception:
            name_node = None

        if name_node is None:
            for child in getattr(node, "children", []):
                if child.type in {
                    "identifier",
                    "type_identifier",
                    "property_identifier",
                }:
                    name_node = child
                    break

        if name_node is None:
            return None
        return content[name_node.start_byte : name_node.end_byte].strip() or None

    def _read_lines(self, file_path: str, start: int, end: int) -> str:
        if not os.path.exists(file_path):
            return "<File not found>"

        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        start_idx = max(0, start - 1)
        end_idx = min(len(lines), max(start, end))
        return "".join(lines[start_idx:end_idx])

    def _get_imports(self, file_path: str) -> list[str]:
        lang = _detect_language_for_path(file_path)
        if lang == "python":
            try:
                with open(file_path, encoding="utf-8", errors="replace") as f:
                    tree = ast.parse(f.read())
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(f"import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module if node.module else ""
                        for alias in node.names:
                            imports.append(f"from {module} import {alias.name}")
                return sorted(set(imports))
            except Exception:
                return []

        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if (
                lang in {"javascript", "typescript"}
                and stripped.startswith("import ")
                or lang in {"c", "cpp"}
                and stripped.startswith("#include")
                or lang == "shell"
                and (
                    stripped.startswith("source ")
                    or stripped.startswith(". ")
                    or stripped.startswith("export ")
                )
                or lang == "cmake"
                and stripped.lower().startswith(
                    ("include(", "find_package(", "add_subdirectory(")
                )
            ):
                lines.append(stripped)
        return sorted(set(lines))

    def _find_called_symbols(
        self, source: str, file_path: str, depth: int
    ) -> list[dict[str, Any]]:
        if depth <= 0:
            return []

        try:
            tree = ast.parse(source)
        except Exception:
            return []

        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)

        dependencies = []
        for call in sorted(calls):
            loc = self._resolve_symbol(call, preferred_file=file_path)
            if not loc:
                continue
            code = self._read_lines(loc.file_path, loc.start_line, loc.end_line)
            dependencies.append(
                {
                    "role": "dependency",
                    "relation": "call",
                    "name": loc.name,
                    "file": os.path.relpath(loc.file_path, self.repo_path),
                    "type": loc.symbol_type,
                    "code": code,
                }
            )

        return dependencies


class TracePerception:
    """Agent-facing trace and complexity perception surfaces."""

    _FFI_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "ctypes",
            re.compile(
                r"\bctypes\.(?:CDLL|PyDLL|WinDLL)\b|ctypes\.cdll\.LoadLibrary\(",
                re.IGNORECASE,
            ),
        ),
        (
            "cffi",
            re.compile(r"\bffi\.dlopen\(|\bcffi\b|\bFFI\s*\(", re.IGNORECASE),
        ),
        (
            "pybind11",
            re.compile(r"\bpybind11\b|\bPYBIND11_MODULE\b", re.IGNORECASE),
        ),
        (
            "cython",
            re.compile(r"\bcdef\s+extern\b|\bcpdef\b", re.IGNORECASE),
        ),
        (
            "napi",
            re.compile(r"\bffi-napi\b|\bnode-addon-api\b|\bnapi_[a-z_]+\b"),
        ),
        (
            "cgo",
            re.compile(r'^\s*import\s+"C"\s*$', re.IGNORECASE),
        ),
        (
            "jni",
            re.compile(r"\bSystem\.loadLibrary\(|\bJNIEXPORT\b", re.IGNORECASE),
        ),
        (
            "wasm",
            re.compile(r"\bWebAssembly\.(?:instantiate|Module)\b", re.IGNORECASE),
        ),
    )

    def __init__(self, repo_path: str = ".") -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.join(self.repo_path, ".saguaro")
        self._cfg_builder = CFGBuilder()
        self._complexity = ComplexityAnalyzer(repo_path=self.repo_path)
        self._flops = FLOPCounter(repo_path=self.repo_path)
        self._ffi_scanner = FFIScanner(repo_path=self.repo_path)
        self._formatter = TraceOutputFormatter()

    def build_cfg(self, file_path: str, symbol: str | None = None) -> dict[str, Any]:
        """Build a lightweight control-flow graph for a file or symbol."""
        full = self._resolve_path(file_path)
        if not os.path.exists(full):
            return {
                "status": "error",
                "message": f"File not found: {file_path}",
                "nodes": [],
                "edges": [],
            }

        language = _detect_language_for_path(full)
        if language != "python":
            return {
                "status": "unsupported_language",
                "message": f"CFG extraction is only implemented for Python. Received: {language}",
                "file": os.path.relpath(full, self.repo_path),
                "language": language,
                "nodes": [],
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
            }

        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source, filename=full)
        except Exception as exc:
            return {
                "status": "error",
                "message": str(exc),
                "nodes": [],
                "edges": [],
            }

        body = list(tree.body)
        target_name = None
        if symbol:
            target_name = symbol.split(".")[-1].strip()
            resolved = self._resolve_python_body(tree, symbol)
            if resolved is not None:
                body = list(resolved)

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        counter = 0

        def add_node(kind: str, label: str, line: int) -> str:
            nonlocal counter
            counter += 1
            node_id = f"n{counter}"
            nodes.append(
                {
                    "id": node_id,
                    "kind": kind,
                    "label": label,
                    "line": int(line or 1),
                }
            )
            return node_id

        def link_many(src_ids: list[str], dst_id: str, relation: str) -> None:
            for src in src_ids:
                edges.append(
                    {
                        "from": src,
                        "to": dst_id,
                        "relation": relation,
                    }
                )

        def walk_block(statements: list[ast.stmt], incoming: list[str]) -> list[str]:
            tails = list(incoming)
            for stmt in statements:
                if isinstance(stmt, ast.If):
                    cond_id = add_node("if", "if", getattr(stmt, "lineno", 1))
                    link_many(tails, cond_id, "next")
                    true_tails = walk_block(list(stmt.body), [cond_id])
                    false_tails = (
                        walk_block(list(stmt.orelse), [cond_id])
                        if stmt.orelse
                        else [cond_id]
                    )
                    for tail in true_tails:
                        edges.append({"from": cond_id, "to": tail, "relation": "true"})
                    if stmt.orelse:
                        for tail in false_tails:
                            edges.append(
                                {"from": cond_id, "to": tail, "relation": "false"}
                            )
                    tails = list(dict.fromkeys(true_tails + false_tails))
                    continue

                if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                    loop_label = (
                        "for" if isinstance(stmt, (ast.For, ast.AsyncFor)) else "while"
                    )
                    loop_id = add_node("loop", loop_label, getattr(stmt, "lineno", 1))
                    link_many(tails, loop_id, "next")
                    body_tails = walk_block(list(stmt.body), [loop_id])
                    for tail in body_tails:
                        edges.append({"from": tail, "to": loop_id, "relation": "back"})
                    orelse_tails = (
                        walk_block(list(stmt.orelse), [loop_id]) if stmt.orelse else []
                    )
                    tails = [loop_id] + orelse_tails
                    continue

                if isinstance(stmt, ast.Return):
                    node_id = add_node("return", "return", getattr(stmt, "lineno", 1))
                    link_many(tails, node_id, "next")
                    tails = [node_id]
                    continue

                kind = type(stmt).__name__.lower()
                node_id = add_node(kind, kind, getattr(stmt, "lineno", 1))
                link_many(tails, node_id, "next")
                tails = [node_id]
            return tails

        entry_id = add_node("entry", "entry", 1)
        exits = walk_block(body, [entry_id])
        exit_id = add_node("exit", "exit", max([1] + [n["line"] for n in nodes]))
        for tail in exits:
            edges.append({"from": tail, "to": exit_id, "relation": "next"})

        return {
            "status": "ok",
            "type": "cfg",
            "file": os.path.relpath(full, self.repo_path),
            "language": "python",
            "symbol": symbol,
            "symbol_hint": target_name,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    def ffi_boundaries(
        self,
        path: str = ".",
        limit: int = 200,
    ) -> dict[str, Any]:
        """Scan repository files for likely FFI boundaries."""
        root = self._resolve_path(path)
        if not os.path.exists(root):
            return {
                "status": "error",
                "message": f"Path not found: {path}",
                "boundaries": [],
                "count": 0,
            }

        mechanism_by_kind = {
            "ctypes_load_library": "ctypes",
            "cffi_dlopen": "cffi",
            "cpp_pybind_module": "pybind11",
            "cpp_python_capi": "capi",
            "extern_c_export": "capi",
            "go_cgo": "cgo",
            "rust_pyo3": "pyo3",
            "tensorflow_load_op_library": "tensorflow_custom_op",
        }

        findings: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str]] = set()
        scanned_files = 0
        for file_path in self._iter_source_files(root):
            scanned_files += 1
            rel_file = os.path.relpath(file_path, self.repo_path).replace("\\", "/")
            try:
                with open(file_path, encoding="utf-8", errors="replace") as handle:
                    source = handle.read()
            except Exception:
                continue

            patterns = self._ffi_scanner.scan_file(rel_file, source)
            for pattern in patterns:
                kind = str(pattern.get("kind") or "")
                mechanism = mechanism_by_kind.get(kind, kind)
                snippet = str(pattern.get("evidence") or "")[:180]
                key = (rel_file, int(pattern.get("line") or 0), mechanism)
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    {
                        "file": rel_file,
                        "line": int(pattern.get("line") or 0),
                        "mechanism": mechanism,
                        "snippet": snippet,
                        "target": str(pattern.get("library_hint") or "") or None,
                        "confidence": float(pattern.get("confidence") or 0.0),
                        "boundary_type": str(pattern.get("boundary_type") or ""),
                        "from_language": str(pattern.get("source_language") or ""),
                        "to_language": str(pattern.get("target_language") or ""),
                        "typed_boundary": dict(pattern.get("typed_boundary") or {}),
                        "type_map": dict(pattern.get("type_map") or {}),
                        "typing_extraction": dict(pattern.get("typing_extraction") or {}),
                        "shared_object": str(pattern.get("shared_object") or ""),
                        "shared_object_resolution": dict(
                            pattern.get("shared_object_resolution") or {}
                        ),
                    }
                )
                if len(findings) >= max(1, int(limit)):
                    break
            if len(findings) < max(1, int(limit)):
                for idx, line in enumerate(source.splitlines(), start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    for mechanism, pattern in self._FFI_PATTERNS:
                        if not pattern.search(raw):
                            continue
                        key = (rel_file, idx, mechanism)
                        if key in seen:
                            break
                        seen.add(key)
                        findings.append(
                            {
                                "file": rel_file,
                                "line": idx,
                                "mechanism": mechanism,
                                "snippet": raw[:180],
                                "target": self._extract_ffi_target(raw),
                                "confidence": 0.65,
                            }
                        )
                        break
                    if len(findings) >= max(1, int(limit)):
                        break
            if len(findings) >= max(1, int(limit)):
                break

        return {
            "status": "ok",
            "count": len(findings),
            "scanned_files": scanned_files,
            "boundaries": findings,
        }

    def trace_pipeline(
        self,
        entry_point: str | None = None,
        *,
        query: str | None = None,
        depth: int = 20,
        max_stages: int = 128,
        include_complexity: bool = True,
    ) -> dict[str, Any]:
        """Trace execution path through the persisted code graph."""
        payload = self._load_code_graph()
        graph = payload.get("graph") or {}
        nodes = self._graph_items(graph.get("nodes"))
        edges = self._graph_items(graph.get("edges"))
        active_payload = payload

        if not nodes:
            return {
                "status": "missing_graph",
                "message": (
                    "No persisted code graph found. Run `saguaro index --path .` or `saguaro graph build`."
                ),
                "stages": [],
                "edges": [],
                "graph_path": payload.get("graph_path"),
            }

        seed_ids = self._resolve_seed_nodes(nodes, graph, entry_point=entry_point, query=query)
        if not seed_ids:
            fallback_path = os.path.join(self.saguaro_dir, "graph", "graph.json")
            if payload.get("graph_path") != fallback_path:
                fallback = self._load_graph_file(fallback_path)
                if fallback:
                    fallback_graph = fallback.get("graph") or {}
                    fallback_nodes = self._graph_items(fallback_graph.get("nodes"))
                    fallback_edges = self._graph_items(fallback_graph.get("edges"))
                    fallback_seed_ids = self._resolve_seed_nodes(
                        fallback_nodes,
                        fallback_graph,
                        entry_point=entry_point,
                        query=query,
                    )
                    if fallback_seed_ids:
                        active_payload = fallback
                        graph = fallback_graph
                        nodes = fallback_nodes
                        edges = fallback_edges
                        seed_ids = fallback_seed_ids

        if not seed_ids:
            return {
                "status": "no_match",
                "entry_point": entry_point,
                "query": query,
                "stages": [],
                "edges": [],
                "graph_path": active_payload.get("graph_path"),
            }

        out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        module_file_nodes = self._module_file_node_index(nodes)
        for edge in edges.values():
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if not src:
                continue
            projected = dict(edge)
            if dst.startswith("external::"):
                resolved = self._resolve_external_node_id(
                    dst=dst,
                    module_file_nodes=module_file_nodes,
                )
                if resolved:
                    projected["to"] = resolved
            out_edges[src].append(projected)

        limit = max(1, int(max_stages))
        max_depth = max(0, int(depth))
        queue: deque[tuple[str, int, dict[str, Any] | None]] = deque(
            (seed, 0, None) for seed in seed_ids
        )
        visited = set(seed_ids)
        stages: list[dict[str, Any]] = []
        traversed_edges: list[dict[str, Any]] = []

        while queue and len(stages) < limit:
            node_id, dist, via = queue.popleft()
            node = nodes.get(node_id)
            if not node:
                continue
            stage = {
                "id": node_id,
                "name": str(node.get("qualified_name") or node.get("name") or node_id),
                "type": str(node.get("type") or "unknown"),
                "file": str(node.get("file") or ""),
                "line": int(node.get("line", 0) or 0),
                "depth": dist,
            }
            if via:
                stage["from"] = str(via.get("from") or "")
                stage["relation"] = str(via.get("relation") or "related")
            if include_complexity and stage["file"].endswith(".py"):
                complexity_payload = self.complexity_report(
                    symbol=str(node.get("qualified_name") or node.get("name") or ""),
                    file_path=stage["file"],
                )
                if isinstance(complexity_payload, dict):
                    complexity_payload.setdefault(
                        "time",
                        complexity_payload.get("time_complexity", "unknown"),
                    )
                    complexity_payload.setdefault(
                        "space",
                        complexity_payload.get("space_complexity", "unknown"),
                    )
                    complexity_payload.setdefault(
                        "amortized_time",
                        complexity_payload.get("amortized_time_complexity", "unknown"),
                    )
                    complexity_payload.setdefault(
                        "worst_case_time",
                        complexity_payload.get("worst_case_time_complexity", "unknown"),
                    )
                stage["complexity"] = complexity_payload
            stages.append(stage)

            if dist >= max_depth:
                continue

            for edge in out_edges.get(node_id, []):
                nxt = str(edge.get("to") or "")
                if not nxt:
                    continue
                traversed_edges.append(
                    {
                        "from": node_id,
                        "to": nxt,
                        "relation": str(edge.get("relation") or "related"),
                    }
                )
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1, edge))

        ffi_report = self.ffi_boundaries(path=".", limit=400)
        stage_files = {
            str(item.get("file") or "") for item in stages if item.get("file")
        }
        ffi_in_trace = [
            item
            for item in ffi_report.get("boundaries", [])
            if str(item.get("file") or "") in stage_files
        ]

        return {
            "status": "ok",
            "entry_point": entry_point,
            "query": query,
            "graph_path": active_payload.get("graph_path"),
            "seed_count": len(seed_ids),
            "stage_count": len(stages),
            "stages": stages,
            "edges": traversed_edges[: limit * 4],
            "ffi_boundaries": ffi_in_trace,
            "total_complexity": self._summarize_trace_complexity(stages),
        }

    def pipeline_complexity(
        self,
        entry_point: str,
        *,
        depth: int = 20,
    ) -> dict[str, Any]:
        """Estimate aggregate complexity for a traced pipeline."""
        target = str(entry_point or "").strip()
        query_like = (
            " " in target
            and not any(
                marker in target
                for marker in (":", "/", "\\", ".py", ".ts", ".js", ".cpp", ".cc")
            )
        )
        if query_like:
            traced = self.trace_pipeline(
                query=target,
                depth=depth,
                include_complexity=True,
            )
        else:
            traced = self.trace_pipeline(
                entry_point=target,
                depth=depth,
                include_complexity=True,
            )
        if traced.get("status") == "no_match" and target:
            traced = self.trace_pipeline(
                query=target,
                depth=depth,
                include_complexity=True,
            )
        aggregate = self._complexity.analyze_pipeline(
            type("TracePayload", (), {"stages": list(traced.get("stages") or [])})()
        )
        total_complexity = (
            aggregate
            if aggregate
            else traced.get("total_complexity", {})
        )
        return {
            "status": traced.get("status", "unknown"),
            "entry_point": entry_point,
            "stage_count": int(traced.get("stage_count", 0) or 0),
            "total_complexity": total_complexity,
            "trace": traced,
        }

    def complexity_report(
        self,
        symbol: str,
        *,
        file_path: str | None = None,
        include_flops: bool = False,
    ) -> dict[str, Any]:
        """Estimate computational complexity for a Python symbol."""
        target_file = self._resolve_symbol_file(symbol, file_hint=file_path)
        if not target_file:
            return {
                "status": "not_found",
                "symbol": symbol,
                "time_complexity": "unknown",
                "space_complexity": "unknown",
                "confidence": 0.0,
                "evidence": ["symbol_not_found"],
            }

        full = self._resolve_path(target_file)
        if not os.path.exists(full):
            return {
                "status": "missing_file",
                "symbol": symbol,
                "file": target_file,
                "time_complexity": "unknown",
                "space_complexity": "unknown",
                "confidence": 0.0,
                "evidence": ["file_missing"],
            }
        if _detect_language_for_path(full) != "python":
            return {
                "status": "unsupported_language",
                "symbol": symbol,
                "file": target_file,
                "time_complexity": "unknown",
                "space_complexity": "unknown",
                "confidence": 0.2,
                "evidence": ["unsupported_language"],
            }

        try:
            with open(full, encoding="utf-8", errors="replace") as handle:
                source = handle.read()
        except Exception:
            return {
                "status": "parse_error",
                "symbol": symbol,
                "file": target_file,
                "time_complexity": "unknown",
                "space_complexity": "unknown",
                "confidence": 0.0,
                "evidence": ["file_parse_error"],
            }
        cfg_payload = self._cfg_builder.build(target_file, source)
        cfg = {
            "nodes": list(cfg_payload.get("nodes") or []),
            "edges": list(cfg_payload.get("edges") or []),
        }
        normalized_symbol = self._normalize_symbol_for_ast(symbol)
        estimate = self._complexity.analyze_function(
            symbol=normalized_symbol,
            file_path=target_file,
            cfg=cfg,
        )

        report: dict[str, Any] = {
            "status": "ok",
            "symbol": symbol,
            "file": os.path.relpath(full, self.repo_path),
            "time_complexity": estimate.time_complexity,
            "amortized_time_complexity": estimate.amortized_time_complexity,
            "worst_case_time_complexity": estimate.worst_case_time_complexity,
            "space_complexity": estimate.space_complexity,
            "parameterized_variables": dict(estimate.parameterized_variables or {}),
            "confidence": float(estimate.confidence),
            "features": {
                "loop_nesting": int(estimate.loop_depth),
                "recursive": bool(estimate.has_recursion),
                "evidence_count": len(estimate.evidence),
                "operation_count": len(estimate.dominant_operations),
            },
            "evidence": list(estimate.evidence),
            "dominant_operations": list(estimate.dominant_operations),
        }
        if include_flops:
            flop_estimates = self._flops.count_function(
                file_path=target_file,
                symbol=normalized_symbol,
            )
            if flop_estimates:
                report["estimated_flops"] = {
                    "count": len(flop_estimates),
                    "operations": [
                        {
                            "operation": item.operation,
                            "formula": item.formula,
                            "estimated_flops": item.estimated_flops,
                            "line": item.line,
                            "confidence": item.confidence,
                        }
                        for item in flop_estimates
                    ],
                }
            else:
                report["estimated_flops"] = None
        return report

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.repo_path, path))

    def _iter_source_files(self, root: str) -> list[str]:
        if os.path.isfile(root):
            return [root]
        accepted = {
            ".py",
            ".pyx",
            ".pxd",
            ".c",
            ".cc",
            ".cpp",
            ".h",
            ".hpp",
            ".rs",
            ".go",
            ".js",
            ".ts",
            ".tsx",
            ".java",
        }
        files: list[str] = []
        for base, dirs, names in os.walk(root):
            rel_base = os.path.relpath(base, root).replace("\\", "/")
            if rel_base != "." and rel_base.startswith("."):
                continue
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d
                not in {
                    ".git",
                    ".saguaro",
                    ".anvil",
                    "venv",
                    ".venv",
                    "node_modules",
                    "__pycache__",
                    "build",
                    "dist",
                    "Saguaro",
                }
            ]
            for name in names:
                suffix = os.path.splitext(name)[1].lower()
                if suffix not in accepted:
                    continue
                files.append(os.path.join(base, name))
        return sorted(files)

    @staticmethod
    def _extract_ffi_target(line: str) -> str | None:
        match = re.search(r"""['"]([^'"]+\.(?:so|dll|dylib|pyd|wasm))['"]""", line)
        if match:
            return match.group(1)
        return None

    def _fallback_linear_cfg(self, file_path: str, *, language: str) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                lines = [line.rstrip("\n") for line in f]
        except Exception as exc:
            return {"status": "error", "message": str(exc), "nodes": [], "edges": []}

        previous_id = None
        for idx, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            node_id = f"n{len(nodes) + 1}"
            nodes.append(
                {
                    "id": node_id,
                    "kind": "stmt",
                    "label": line.strip()[:80],
                    "line": idx,
                }
            )
            if previous_id:
                edges.append({"from": previous_id, "to": node_id, "relation": "next"})
            previous_id = node_id

        return {
            "status": "ok",
            "type": "cfg",
            "file": os.path.relpath(file_path, self.repo_path),
            "language": language,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "fallback": True,
        }

    def _load_code_graph(self) -> dict[str, Any]:
        candidate_paths = [
            os.path.join(self.saguaro_dir, "graph", "code_graph.json"),
            os.path.join(self.saguaro_dir, "code_graph.json"),
            os.path.join(self.saguaro_dir, "graph", "graph.json"),
        ]
        for candidate in candidate_paths:
            loaded = self._load_graph_file(candidate)
            if loaded:
                return loaded
        return {"graph_path": None, "graph": {}}

    def _load_graph_file(self, path: str) -> dict[str, Any] | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle) or {}
        except Exception:
            return None
        graph = raw.get("graph") if isinstance(raw, dict) and "graph" in raw else raw
        if not isinstance(graph, dict):
            return None
        return {"graph_path": path, "graph": graph}

    @staticmethod
    def _graph_items(payload: Any) -> dict[str, dict[str, Any]]:
        if isinstance(payload, dict):
            return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
        if isinstance(payload, list):
            out: dict[str, dict[str, Any]] = {}
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("id") or f"item_{idx}")
                out[node_id] = dict(item)
            return out
        return {}

    @staticmethod
    def _module_file_node_index(
        nodes: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for node_id, node in nodes.items():
            if str(node.get("type") or "") != "file":
                continue
            rel_file = str(node.get("file") or "").replace("\\", "/")
            if not rel_file.endswith(".py"):
                continue
            module = rel_file[: -len(".py")].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            if module:
                out[module] = node_id
        return out

    @staticmethod
    def _resolve_external_node_id(
        *,
        dst: str,
        module_file_nodes: dict[str, str],
    ) -> str | None:
        if not dst.startswith("external::"):
            return None
        reference = dst[len("external::") :].strip().lstrip(".")
        if not reference:
            return None
        parts = [part for part in reference.split(".") if part]
        while parts:
            candidate = ".".join(parts)
            node_id = module_file_nodes.get(candidate)
            if node_id:
                return node_id
            parts = parts[:-1]
        return None

    def _resolve_seed_nodes(
        self,
        nodes: dict[str, dict[str, Any]],
        graph: dict[str, Any],
        *,
        entry_point: str | None,
        query: str | None,
    ) -> list[str]:
        files = self._graph_items(graph.get("files"))
        if entry_point:
            normalized = entry_point.strip().replace("\\", "/")
            target = normalized.lower()
            rel = normalized
            if rel in files:
                return self._curated_file_seed_nodes(
                    file_entry=files[rel],
                    nodes=nodes,
                )
            rel_py = rel.replace(".", "/")
            if not rel_py.endswith(".py"):
                rel_py = f"{rel_py}.py"
            if rel_py in files:
                return self._curated_file_seed_nodes(
                    file_entry=files[rel_py],
                    nodes=nodes,
                )

            candidates = self._entry_point_candidates(normalized)
            scored: list[tuple[int, str, str, str]] = []
            explicit_test_context = any(
                token in target for token in ("test", "tests", "_test")
            )
            for node_id, node in nodes.items():
                node_name = str(node.get("name") or "")
                qualified = str(node.get("qualified_name") or "")
                node_file = str(node.get("file") or "").replace("\\", "/")
                node_type = str(node.get("type") or "").lower()
                hay = " ".join(
                    [
                        node_name,
                        qualified,
                        node_file,
                    ]
                ).lower()

                score = 0
                if target and target in hay:
                        score += 5
                for candidate in candidates:
                    cand = candidate.lower()
                    if not cand:
                        continue
                    if cand == qualified.lower():
                        score += 8
                    elif cand == node_name.lower():
                        score += 6
                    elif cand == node_file.lower():
                        score += 7
                    elif cand in hay:
                        score += 3
                if node_type in {"function", "method", "class"}:
                    score += 4
                elif node_type == "file":
                    score += 1
                elif node_type.startswith(("dfg_", "cfg_")):
                    score -= 4
                elif node_type in {"external", "dependency_graph"}:
                    score -= 2
                if node_type == "external" and "external" not in target:
                    continue
                if not node_file and node_type != "file":
                    score -= 5
                if (
                    node_file
                    and self._is_noisy_path(node_file)
                    and not explicit_test_context
                ):
                    score -= 3
                if score > 0:
                    scored.append((score, node_id, node_file, node_type))

            if scored:
                scored.sort(key=lambda item: (-item[0], item[2], item[1]))
                top_score = scored[0][0]
                selected: list[str] = []
                per_file: dict[str, int] = {}
                for score, node_id, node_file, node_type in scored:
                    if score < max(4, top_score - 3):
                        continue
                    if node_file:
                        count = per_file.get(node_file, 0)
                        if count >= 2 and node_type != "file":
                            continue
                        per_file[node_file] = count + 1
                    selected.append(node_id)
                    if len(selected) >= 8:
                        break
                if selected:
                    return selected
            return []

        if query:
            terms = [term for term in re.findall(r"[a-z0-9_]{2,}", query.lower())]
            expanded_terms = set(terms)
            if "inference" in expanded_terms or "infer" in expanded_terms:
                expanded_terms.update({"decode", "forward", "generate", "token"})
            if "training" in expanded_terms or "train" in expanded_terms:
                expanded_terms.update({"loss", "backward", "optimizer", "step"})
            explicit_test_context = any(
                token in expanded_terms for token in ("test", "tests", "_test")
            )
            scored: list[tuple[int, str, str, str]] = []
            for node_id, node in nodes.items():
                node_name = str(node.get("name") or "")
                qualified = str(node.get("qualified_name") or "")
                node_file = str(node.get("file") or "").replace("\\", "/")
                node_type = str(node.get("type") or "").lower()
                hay = " ".join(
                    [
                        node_name,
                        qualified,
                        node_file,
                    ]
                ).lower()
                score = 0
                for term in expanded_terms:
                    if term == qualified.lower() or term == node_name.lower():
                        score += 7
                    elif re.search(rf"\b{re.escape(term)}\b", hay):
                        score += 3
                    elif term in hay:
                        score += 1
                if "inference" in expanded_terms or "infer" in expanded_terms:
                    if any(
                        token in hay
                        for token in (
                            "forward",
                            "decode",
                            "generate",
                            "token",
                            "logits",
                            "model",
                        )
                    ):
                        score += 3
                    if any(
                        token in node_file.lower()
                        for token in (
                            "/index/",
                            "indexing/",
                            "vector_store",
                            "embedding",
                        )
                    ):
                        score -= 2
                    if node_file.startswith("core/"):
                        score += 3
                    if any(
                        token in node_file.lower()
                        for token in (
                            "model",
                            "native",
                            "tokenizer",
                            "decode",
                        )
                    ):
                        score += 2
                    if any(
                        token in node_file.lower()
                        for token in (
                            "saguaro/indexing/",
                            "saguaro/storage/",
                            "vector_store",
                        )
                    ):
                        score -= 6
                if "training" in expanded_terms or "train" in expanded_terms:
                    if any(
                        token in hay
                        for token in (
                            "loss",
                            "backward",
                            "optimizer",
                            "step",
                        )
                    ):
                        score += 3
                if node_type in {"function", "method", "class"}:
                    score += 4
                elif node_type == "file":
                    score += 1
                elif node_type.startswith(("dfg_", "cfg_")):
                    score -= 4
                elif node_type in {"external", "dependency_graph"}:
                    score -= 2
                if node_type == "external" and not (
                    "external" in expanded_terms
                    or "import" in expanded_terms
                    or "dependency" in expanded_terms
                ):
                    continue
                if not node_file and node_type != "file":
                    score -= 5
                if (
                    node_file
                    and self._is_noisy_path(node_file)
                    and not explicit_test_context
                ):
                    score -= 4
                if score > 0:
                    scored.append((score, node_id, node_file, node_type))
            if scored:
                scored.sort(key=lambda item: (-item[0], item[2], item[1]))
                selected: list[str] = []
                per_file: dict[str, int] = {}
                for score, node_id, node_file, node_type in scored:
                    if selected and score < max(3, scored[0][0] - 3):
                        continue
                    if node_file:
                        count = per_file.get(node_file, 0)
                        if count >= 2 and node_type != "file":
                            continue
                        per_file[node_file] = count + 1
                    selected.append(node_id)
                    if len(selected) >= 8:
                        break
                if selected:
                    return selected
            return []

        fallback = [
            node_id
            for node_id, node in nodes.items()
            if str(node.get("type") or "") == "file"
        ]
        if fallback:
            return fallback[:3]
        return list(nodes.keys())[:3]

    @staticmethod
    def _curated_file_seed_nodes(
        *,
        file_entry: dict[str, Any],
        nodes: dict[str, dict[str, Any]],
    ) -> list[str]:
        ordered: list[str] = []
        fallback: list[str] = []
        preferred_names = {"main", "run", "entry", "forward", "predict", "infer"}
        for node_id in sorted(set(file_entry.get("nodes") or [])):
            node = nodes.get(str(node_id))
            if not node:
                continue
            node_type = str(node.get("type") or "").lower()
            node_name = str(node.get("name") or "").lower()
            if node_type in {"function", "method"}:
                if node_name in preferred_names:
                    ordered.insert(0, str(node_id))
                else:
                    ordered.append(str(node_id))
            elif node_type == "class":
                ordered.append(str(node_id))
            elif node_type == "file":
                fallback.append(str(node_id))
        seed_ids = ordered[:6]
        if fallback:
            seed_ids = fallback[:1] + seed_ids
        return list(dict.fromkeys(seed_ids))

    @staticmethod
    def _is_noisy_path(path: str) -> bool:
        low = str(path or "").replace("\\", "/").lower()
        if not low:
            return False
        noisy_tokens = (
            "/test/",
            "/tests/",
            "/docs/",
            "/doc/",
            "/examples/",
            "/example/",
            "test_",
            "_test.",
            ".spec.",
        )
        return any(token in low for token in noisy_tokens)

    @staticmethod
    def _entry_point_candidates(entry_point: str) -> list[str]:
        value = (entry_point or "").strip()
        if not value:
            return []

        out: set[str] = {value}
        normalized = value.replace(":", ".")
        out.add(normalized)
        out.add(normalized.replace("/", "."))
        out.add(normalized.replace(".", "/"))

        if ":" in value:
            module, _, symbol = value.partition(":")
            module = module.strip()
            symbol = symbol.strip()
            if module:
                out.add(module)
                out.add(module.replace(".", "/"))
                out.add(f"{module.replace('.', '/')}.py")
            if symbol:
                out.add(symbol)
                parts = [part for part in symbol.split(".") if part]
                if parts:
                    out.add(parts[-1])
                    if len(parts) >= 2:
                        out.add(".".join(parts[-2:]))
        elif "." in normalized:
            dotted = [part for part in normalized.split(".") if part]
            if len(dotted) >= 2:
                module = ".".join(dotted[:-1])
                symbol = dotted[-1]
                out.add(symbol)
                out.add(module)
                out.add(module.replace(".", "/"))
                out.add(f"{module.replace('.', '/')}.py")

        parts = [part for part in normalized.split(".") if part]
        if parts:
            out.add(parts[-1])
            if len(parts) >= 2:
                out.add(".".join(parts[-2:]))
            out.add(f"{parts[-1]}.py")
            out.add(f"{'/'.join(parts)}.py")

        return [candidate for candidate in sorted(out) if candidate]

    @staticmethod
    def _summarize_trace_complexity(stages: list[dict[str, Any]]) -> dict[str, Any]:
        ranking = {
            "O(1)": 0,
            "O(log n)": 1,
            "O(n)": 2,
            "O(n log n)": 3,
            "O(n^2)": 4,
            "O(n^3)": 5,
            "O(n^k)": 6,
            "O(2^n)": 7,
            "unknown": -1,
        }
        best = "unknown"
        best_rank = -1
        confidence_values: list[float] = []
        for stage in stages:
            comp = stage.get("complexity")
            if not isinstance(comp, dict):
                continue
            candidate = str(
                comp.get("time_complexity")
                or comp.get("time")
                or "unknown"
            )
            rank = ranking.get(candidate, ranking.get("O(n^k)", 6))
            if rank > best_rank:
                best = candidate
                best_rank = rank
            try:
                confidence_values.append(float(comp.get("confidence") or 0.0))
            except Exception:
                pass
        return {
            "time_complexity": best,
            "stages_with_complexity": sum(
                1 for stage in stages if isinstance(stage.get("complexity"), dict)
            ),
            "confidence": round(
                sum(confidence_values) / len(confidence_values), 3
            )
            if confidence_values
            else 0.0,
        }

    def _resolve_symbol_file(
        self, symbol: str, file_hint: str | None = None
    ) -> str | None:
        if file_hint:
            return file_hint

        selector = str(symbol or "").strip().replace("\\", "/")
        if not selector:
            return None

        needles = {
            selector.lower(),
            selector.replace(":", ".").lower(),
            self._normalize_symbol_for_ast(selector).lower(),
            selector.split(".")[-1].lower(),
        }
        needles = {item for item in needles if item}

        candidate_paths = [
            os.path.join(self.saguaro_dir, "graph", "code_graph.json"),
            os.path.join(self.saguaro_dir, "code_graph.json"),
            os.path.join(self.saguaro_dir, "graph", "graph.json"),
        ]
        best_file = None
        best_score = 0
        for path in candidate_paths:
            loaded = self._load_graph_file(path)
            if not loaded:
                continue
            graph = loaded.get("graph") or {}
            nodes = self._graph_items(graph.get("nodes"))
            for node in nodes.values():
                node_name = str(node.get("name") or "").lower()
                qualified = str(node.get("qualified_name") or "").lower()
                file_name = str(node.get("file") or "")
                if not file_name:
                    continue
                hay = " ".join([node_name, qualified, file_name.lower()])
                score = 0
                for needle in needles:
                    if needle == qualified:
                        score += 8
                    elif needle == node_name:
                        score += 6
                    elif needle in hay:
                        score += 3
                if score > best_score:
                    best_score = score
                    best_file = file_name
        if best_file:
            return best_file

        # Entry-point style selector: module.path:Class.method
        if ":" in selector:
            module, _, _tail = selector.partition(":")
            module = module.strip().replace(".", "/")
            if module:
                module_path = f"{module}.py" if not module.endswith(".py") else module
                full = self._resolve_path(module_path)
                if os.path.exists(full):
                    return module_path
        return None

    @staticmethod
    def _normalize_symbol_for_ast(symbol: str) -> str:
        raw = str(symbol or "").strip()
        if ":" in raw:
            _module, _, tail = raw.partition(":")
            raw = tail.strip() or raw
        parts = [part for part in raw.split(".") if part]
        if not parts:
            return raw
        if len(parts) >= 2 and parts[-2][:1].isupper():
            return f"{parts[-2]}.{parts[-1]}"
        return parts[-1]

    @staticmethod
    def _resolve_python_body(tree: ast.Module, symbol: str) -> list[ast.stmt] | None:
        node = TracePerception._resolve_python_symbol_node(tree, symbol)
        if node is None:
            return None
        if isinstance(node, ast.Module):
            return list(node.body)
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            return list(node.body)
        return None

    @staticmethod
    def _resolve_python_symbol_node(tree: ast.Module, symbol: str) -> ast.AST | None:
        if not symbol:
            return tree
        parts = [part for part in symbol.split(".") if part]
        if not parts:
            return tree
        target = parts[-1]
        parent = parts[-2] if len(parts) >= 2 else None

        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == target
            ):
                return node
            if isinstance(node, ast.ClassDef):
                if node.name == target:
                    return node
                if parent and node.name != parent:
                    continue
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == target
                    ):
                        return child
        return None

    @staticmethod
    def _python_complexity_metrics(node: ast.AST, symbol: str) -> dict[str, Any]:
        loop_depth = 0
        branch_count = 0
        call_names: list[str] = []

        def visit(inner: ast.AST, depth: int = 0) -> None:
            nonlocal loop_depth, branch_count
            if isinstance(inner, (ast.For, ast.AsyncFor, ast.While)):
                loop_depth = max(loop_depth, depth + 1)
                for child in ast.iter_child_nodes(inner):
                    visit(child, depth + 1)
                return
            if isinstance(inner, ast.If):
                branch_count += 1
            if isinstance(inner, ast.Call):
                if isinstance(inner.func, ast.Name):
                    call_names.append(inner.func.id)
                elif isinstance(inner.func, ast.Attribute):
                    call_names.append(inner.func.attr)
            for child in ast.iter_child_nodes(inner):
                visit(child, depth)

        visit(node, 0)

        symbol_tail = symbol.split(".")[-1]
        recursive = symbol_tail in call_names
        uses_sort = any(name in {"sorted", "sort"} for name in call_names)

        if recursive:
            time_complexity = "O(2^n)"
        elif loop_depth <= 0:
            time_complexity = "O(n log n)" if uses_sort else "O(1)"
        elif loop_depth == 1:
            time_complexity = "O(n)"
        elif loop_depth == 2:
            time_complexity = "O(n^2)"
        elif loop_depth == 3:
            time_complexity = "O(n^3)"
        else:
            time_complexity = "O(n^k)"

        space_complexity = "O(1)"
        if recursive or loop_depth > 0:
            space_complexity = "O(n)"
        features = {
            "loop_nesting": loop_depth,
            "branches": branch_count,
            "recursive": recursive,
            "sort_usage": uses_sort,
            "call_count": len(call_names),
        }
        ops = []
        if loop_depth > 0:
            ops.append({"op": "loop", "complexity": time_complexity})
        if recursive:
            ops.append({"op": "recursion", "complexity": "O(2^n)"})
        if uses_sort:
            ops.append({"op": "sort", "complexity": "O(n log n)"})

        confidence = 0.4
        confidence += min(loop_depth * 0.15, 0.3)
        if recursive:
            confidence += 0.2
        if uses_sort:
            confidence += 0.1
        confidence = round(max(0.1, min(confidence, 0.95)), 2)

        estimated_flops: dict[str, Any] | None = None
        matmul_like = sum(
            1 for call in call_names if call in {"matmul", "mm", "einsum", "conv2d"}
        )
        if matmul_like:
            estimated_flops = {
                "model": "heuristic",
                "value": f"~{matmul_like} * O(n^3)",
            }

        return {
            "time": time_complexity,
            "space": space_complexity,
            "confidence": confidence,
            "features": features,
            "ops": ops,
            "flops": estimated_flops,
        }
