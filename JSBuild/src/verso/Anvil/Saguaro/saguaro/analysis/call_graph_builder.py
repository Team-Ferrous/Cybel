from __future__ import annotations

import ast
from typing import Any


class CallGraphBuilder:
    """Build a lightweight call graph for Python files."""

    def build(self, rel_file: str, source: str) -> dict[str, list[dict[str, Any]]]:
        if not rel_file.endswith(".py"):
            return {"nodes": [], "edges": []}

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"nodes": [], "edges": []}

        walker = _CallGraphWalker(rel_file)
        walker.visit(tree)
        return walker.to_payload()


class _CallGraphWalker(ast.NodeVisitor):
    def __init__(self, rel_file: str) -> None:
        self.rel_file = rel_file
        self.scope: list[str] = []
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.defined_functions: dict[str, str] = {}

    def to_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "nodes": [self.nodes[node_id] for node_id in sorted(self.nodes)],
            "edges": [self.edges[edge_id] for edge_id in sorted(self.edges)],
        }

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return self._visit_callable(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self._visit_callable(node)

    def _visit_callable(self, node: ast.AST) -> Any:
        name = getattr(node, "name", "<callable>")
        qualified_name = ".".join([*self.scope, name]) if self.scope else name
        node_id = self._symbol_node_id(qualified_name)
        line = int(getattr(node, "lineno", 1) or 1)
        self.nodes[node_id] = {
            "id": node_id,
            "type": "callable",
            "name": name,
            "qualified_name": qualified_name,
            "file": self.rel_file,
            "line": line,
            "end_line": int(getattr(node, "end_lineno", line) or line),
            "source": "call_graph",
        }
        self.defined_functions[qualified_name] = node_id
        self.defined_functions[name] = node_id

        self.scope.append(name)
        for stmt in getattr(node, "body", []):
            self.visit(stmt)
        self.scope.pop()
        return None

    def visit_Call(self, node: ast.Call) -> Any:
        caller_name = self._current_function()
        if caller_name is not None:
            caller_id = self.defined_functions.get(caller_name)
            if caller_id is None:
                caller_id = self._symbol_node_id(caller_name)
                self.nodes[caller_id] = {
                    "id": caller_id,
                    "type": "callable",
                    "name": caller_name.split(".")[-1],
                    "qualified_name": caller_name,
                    "file": self.rel_file,
                    "line": int(getattr(node, "lineno", 1) or 1),
                    "end_line": int(getattr(node, "lineno", 1) or 1),
                    "source": "call_graph",
                }

            callee_name = self._call_name(node.func)
            callee_id, confidence = self._resolve_callee(callee_name)
            edge_line = int(getattr(node, "lineno", 1) or 1)
            edge_id = f"{caller_id}->{callee_id}::calls::{edge_line}"
            self.edges[edge_id] = {
                "id": edge_id,
                "from": caller_id,
                "to": callee_id,
                "relation": "calls",
                "line": edge_line,
                "confidence": confidence,
                "callee": callee_name,
                "source": "call_graph",
            }

        self.generic_visit(node)
        return None

    def _resolve_callee(self, callee_name: str) -> tuple[str, float]:
        cleaned = callee_name.strip()
        if not cleaned:
            unknown_id = f"external::{self.rel_file}::<dynamic>"
            self.nodes.setdefault(
                unknown_id,
                {
                    "id": unknown_id,
                    "type": "external_symbol",
                    "name": "<dynamic>",
                    "qualified_name": "<dynamic>",
                    "file": None,
                    "line": 0,
                    "end_line": 0,
                    "source": "call_graph",
                },
            )
            return unknown_id, 0.4

        if cleaned in self.defined_functions:
            return self.defined_functions[cleaned], 0.95

        if "." in cleaned:
            leaf = cleaned.split(".")[-1]
            if leaf in self.defined_functions:
                return self.defined_functions[leaf], 0.8
            confidence = 0.65
        else:
            confidence = 0.55

        external_id = f"external::{self.rel_file}::{cleaned}"
        self.nodes.setdefault(
            external_id,
            {
                "id": external_id,
                "type": "external_symbol",
                "name": cleaned.split(".")[-1],
                "qualified_name": cleaned,
                "file": None,
                "line": 0,
                "end_line": 0,
                "source": "call_graph",
            },
        )
        return external_id, confidence

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = _CallGraphWalker._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return _CallGraphWalker._call_name(node.func)
        return ""

    def _current_function(self) -> str | None:
        if not self.scope:
            return None
        return ".".join(self.scope)

    def _symbol_node_id(self, qualified_name: str) -> str:
        return f"{self.rel_file}::{qualified_name}::callable"
