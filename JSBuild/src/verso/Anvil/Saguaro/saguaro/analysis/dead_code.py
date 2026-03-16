"""Dead Code Analyzer
Conservative detector for likely-unused top-level Python symbols.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from saguaro.utils.file_utils import get_code_files

logger = logging.getLogger(__name__)

_DYNAMIC_CALL_NAMES = {"getattr", "setattr", "hasattr", "delattr", "eval", "exec"}
_IGNORE_DIR_PREFIXES = (
    "venv/",
    ".venv/",
    ".git/",
    ".saguaro/",
    "node_modules/",
    "build/",
    "dist/",
    "Saguaro/",
    "xeditor-monorepo",
)


@dataclass(frozen=True)
class DefinitionRecord:
    """Provide DefinitionRecord support."""

    name: str
    file_path: str
    module: str
    line: int
    is_public: bool
    in_init_module: bool
    decorators: tuple[str, ...]


class DeadCodeAnalyzer:
    """Provide DeadCodeAnalyzer support."""

    def __init__(self, repo_path: str, include_tests: bool = False) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.include_tests = include_tests
        self.definitions: dict[str, DefinitionRecord] = {}
        self.references: set[str] = set()
        self.exported_names: set[str] = set()
        self.dynamic_files: set[str] = set()
        self._graph_payload: dict[str, Any] | None = None

    def analyze(self) -> list[dict]:
        """Scans codebase for definitions and references to find likely unused symbols.
        Returns candidates with conservative confidence scores.
        """
        self._scan_repo()
        graph_usage = self._graph_usage_hints()
        referenced_symbols = set(graph_usage.get("referenced_symbols", set()))
        file_imported = set(graph_usage.get("imported_files", set()))
        graph_enabled = bool(graph_usage.get("graph_enabled", False))

        candidates = []
        for _key, definition in self.definitions.items():
            if definition.name in self.references:
                continue
            if definition.name in self.exported_names:
                continue
            rel_file = os.path.relpath(definition.file_path, self.repo_path).replace(
                "\\", "/"
            )
            if (rel_file, definition.name) in referenced_symbols:
                continue
            if definition.is_public and rel_file in file_imported:
                # File-level imports imply potential external/public symbol use.
                continue
            if self._is_ignored(definition):
                continue

            score = self._calculate_confidence(definition)
            candidates.append(
                {
                    "symbol": definition.name,
                    "module": definition.module,
                    "file": definition.file_path,
                    "line": definition.line,
                    "confidence": score,
                    "reason": "No static references found",
                    "dynamic_file": definition.file_path in self.dynamic_files,
                    "public": definition.is_public,
                    "graph_assisted": graph_enabled,
                }
            )

        return sorted(candidates, key=lambda item: item["confidence"], reverse=True)

    def _scan_repo(self) -> None:
        for path in self._iter_python_files():
            self._parse_file(path)

    def _iter_python_files(self) -> list[str]:
        tracked = self._git_tracked_python_files()
        if tracked is not None:
            return tracked

        all_files = get_code_files(self.repo_path, exclusions=[])
        return [
            path
            for path in all_files
            if path.endswith(".py") and self._is_in_scope(path)
        ]

    def _git_tracked_python_files(self) -> list[str] | None:
        if not os.path.isdir(os.path.join(self.repo_path, ".git")):
            return None

        try:
            result = subprocess.run(
                ["git", "ls-files", "--", "*.py"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return None

        if result.returncode != 0:
            return None

        files = []
        for line in result.stdout.splitlines():
            rel = line.strip().replace("\\", "/")
            if not rel:
                continue
            abs_path = os.path.join(self.repo_path, rel)
            if not os.path.isfile(abs_path):
                continue
            if not self._is_in_scope(abs_path):
                continue
            files.append(abs_path)
        return files

    def _is_in_scope(self, abs_path: str) -> bool:
        rel = os.path.relpath(abs_path, self.repo_path).replace("\\", "/")
        if rel.startswith(".."):
            return False
        if not self.include_tests and rel.startswith("tests/"):
            return False
        return all(not rel.startswith(prefix) for prefix in _IGNORE_DIR_PREFIXES)

    def _parse_file(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=path)
        except Exception:
            return

        rel = os.path.relpath(path, self.repo_path).replace("\\", "/")
        module = rel[:-3].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        in_init = os.path.basename(path) == "__init__.py"

        # Collect top-level definitions and __all__ exports.
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorators = tuple(
                    self._decorator_name(item) for item in stmt.decorator_list
                )
                key = f"{module}:{stmt.name}"
                self.definitions[key] = DefinitionRecord(
                    name=stmt.name,
                    file_path=path,
                    module=module,
                    line=getattr(stmt, "lineno", 0),
                    is_public=not stmt.name.startswith("_"),
                    in_init_module=in_init,
                    decorators=tuple(name for name in decorators if name),
                )
            self._collect_all_exports(stmt)

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                self.references.add(node.id)
            elif isinstance(node, ast.Attribute):
                self.references.add(node.attr)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.references.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    self.references.add(alias.asname or alias.name)
            elif isinstance(node, ast.Call):
                self._collect_dynamic_reference(node, path)

    def _collect_all_exports(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
            if any(t.id == "__all__" for t in targets):
                self._consume_string_container(stmt.value)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__":
                self._consume_string_container(stmt.value)

    def _consume_string_container(self, value: ast.AST | None) -> None:
        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return
        for item in value.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                self.exported_names.add(item.value)

    def _collect_dynamic_reference(self, node: ast.Call, file_path: str) -> None:
        call_name = ""
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            call_name = node.func.attr

        if call_name in _DYNAMIC_CALL_NAMES:
            self.dynamic_files.add(file_path)
            if len(node.args) >= 2:
                maybe_name = node.args[1]
                if isinstance(maybe_name, ast.Constant) and isinstance(
                    maybe_name.value, str
                ):
                    self.references.add(maybe_name.value)
        elif call_name in {"globals", "locals", "vars"}:
            self.dynamic_files.add(file_path)

    def _decorator_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return self._decorator_name(node.func)
        return ""

    def _is_ignored(self, definition: DefinitionRecord) -> bool:
        name = definition.name
        if name.startswith("__") and name.endswith("__"):
            return True
        if name in {"main", "setup", "teardown"}:
            return True
        if definition.in_init_module and definition.is_public:
            # Public package exports are often imported dynamically.
            return True
        decorator_blocklist = {"fixture", "command", "group", "hookimpl"}
        return bool(any(d in decorator_blocklist for d in definition.decorators))

    def _graph_usage_hints(self) -> dict[str, Any]:
        payload = self._load_code_graph()
        graph = payload.get("graph") or {}
        nodes = self._graph_items(graph.get("nodes"))
        edges = self._graph_items(graph.get("edges"))
        if not nodes or not edges:
            return {
                "graph_enabled": False,
                "referenced_symbols": set(),
                "imported_files": set(),
            }

        referenced_symbols: set[tuple[str, str]] = set()
        imported_files: set[str] = set()
        for edge in edges.values():
            src_id = str(edge.get("from") or "")
            dst_id = str(edge.get("to") or "")
            relation = str(edge.get("relation") or "")
            src = nodes.get(src_id, {})
            dst = nodes.get(dst_id, {})

            src_file = str(src.get("file") or "").replace("\\", "/")
            dst_file = str(dst.get("file") or "").replace("\\", "/")
            dst_name = str(dst.get("name") or "").strip()
            src_name = str(src.get("name") or "").strip()

            if relation == "imports" and dst_file and dst_file != src_file:
                imported_files.add(dst_file)

            if not dst_file or not dst_name:
                continue
            if src_file == dst_file and src_name == dst_name:
                continue
            referenced_symbols.add((dst_file, dst_name))

        return {
            "graph_enabled": True,
            "referenced_symbols": referenced_symbols,
            "imported_files": imported_files,
        }

    def _load_code_graph(self) -> dict[str, Any]:
        if self._graph_payload is not None:
            return self._graph_payload
        candidates = [
            os.path.join(self.repo_path, ".saguaro", "graph", "code_graph.json"),
            os.path.join(self.repo_path, ".saguaro", "code_graph.json"),
            os.path.join(self.repo_path, ".saguaro", "graph", "graph.json"),
        ]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, encoding="utf-8") as f:
                    raw = json.load(f) or {}
            except Exception:
                continue
            graph = (
                raw.get("graph") if isinstance(raw, dict) and "graph" in raw else raw
            )
            if isinstance(graph, dict):
                self._graph_payload = {"graph_path": candidate, "graph": graph}
                return self._graph_payload
        self._graph_payload = {"graph_path": None, "graph": {}}
        return self._graph_payload

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

    def _calculate_confidence(self, definition: DefinitionRecord) -> float:
        """Calculates a conservative confidence score (0.0 - 1.0) that code is dead."""
        score = 0.95 if not definition.is_public else 0.7

        if definition.file_path in self.dynamic_files:
            score -= 0.35
        if definition.in_init_module:
            score -= 0.3
        if "tests" in definition.file_path.replace("\\", "/"):
            score -= 0.2
        if definition.decorators:
            score -= 0.2

        return max(0.0, min(1.0, score))
