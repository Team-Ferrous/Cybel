from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class _FunctionSymbol:
    qualified_name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    entry_id: str
    exit_id: str


@dataclass(slots=True)
class _CallSite:
    caller: str
    callee_name: str
    resolved_callee: str | None
    line: int
    col: int


class ICFGBuilder:
    """Build a lightweight interprocedural CFG for Python files."""

    def build(self, rel_file: str, source: str) -> dict[str, list[dict[str, Any]]]:
        if not rel_file.endswith(".py"):
            return {"nodes": [], "edges": []}

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"nodes": [], "edges": []}

        functions = self._collect_functions(tree, rel_file)
        if not functions:
            return {"nodes": [], "edges": []}

        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}

        for symbol in functions.values():
            start_line = int(getattr(symbol.node, "lineno", 1) or 1)
            end_line = int(getattr(symbol.node, "end_lineno", start_line) or start_line)
            nodes[symbol.entry_id] = {
                "id": symbol.entry_id,
                "type": "icfg_entry",
                "name": f"{symbol.qualified_name}::entry",
                "function": symbol.qualified_name,
                "file": rel_file,
                "line": start_line,
                "end_line": start_line,
                "source": "icfg",
            }
            nodes[symbol.exit_id] = {
                "id": symbol.exit_id,
                "type": "icfg_exit",
                "name": f"{symbol.qualified_name}::exit",
                "function": symbol.qualified_name,
                "file": rel_file,
                "line": end_line,
                "end_line": end_line,
                "source": "icfg",
            }

        calls = self._collect_calls(functions)
        recursive_components = self._recursive_component_map(functions, calls)

        for call in calls:
            caller_symbol = functions[call.caller]
            callsite_id = self._callsite_id(rel_file, call.caller, call.line, call.col)
            returnsite_id = self._returnsite_id(rel_file, call.caller, call.line, call.col)
            nodes[callsite_id] = {
                "id": callsite_id,
                "type": "icfg_callsite",
                "name": call.callee_name or "<dynamic>",
                "function": call.caller,
                "file": rel_file,
                "line": int(call.line),
                "end_line": int(call.line),
                "source": "icfg",
            }
            nodes[returnsite_id] = {
                "id": returnsite_id,
                "type": "icfg_returnsite",
                "name": "return_site",
                "function": call.caller,
                "file": rel_file,
                "line": int(call.line),
                "end_line": int(call.line),
                "source": "icfg",
            }

            self._add_edge(
                edges,
                caller_symbol.entry_id,
                callsite_id,
                "icfg_reach",
                call.line,
            )
            self._add_edge(edges, callsite_id, returnsite_id, "icfg_next", call.line)
            self._add_edge(
                edges,
                returnsite_id,
                caller_symbol.exit_id,
                "icfg_rejoin",
                call.line,
            )

            if call.resolved_callee and call.resolved_callee in functions:
                callee_symbol = functions[call.resolved_callee]
                is_recursive = self._is_recursive_edge(
                    call.caller,
                    call.resolved_callee,
                    recursive_components,
                )
                self._add_edge(
                    edges,
                    callsite_id,
                    callee_symbol.entry_id,
                    "icfg_call",
                    call.line,
                    recursive_backedge=is_recursive,
                )
                self._add_edge(
                    edges,
                    callee_symbol.exit_id,
                    returnsite_id,
                    "icfg_return",
                    call.line,
                    recursive_backedge=is_recursive,
                )
                if is_recursive:
                    self._add_edge(
                        edges,
                        callee_symbol.exit_id,
                        caller_symbol.entry_id,
                        "icfg_recursive_backedge",
                        call.line,
                        recursive_backedge=True,
                    )
            else:
                external_name = call.callee_name.strip() or "<dynamic>"
                external_id = f"external::{rel_file}::{external_name}::icfg"
                nodes.setdefault(
                    external_id,
                    {
                        "id": external_id,
                        "type": "icfg_external_symbol",
                        "name": external_name,
                        "function": "<external>",
                        "file": rel_file,
                        "line": 0,
                        "end_line": 0,
                        "source": "icfg",
                    },
                )
                self._add_edge(
                    edges,
                    callsite_id,
                    external_id,
                    "icfg_call_external",
                    call.line,
                )
                self._add_edge(
                    edges,
                    external_id,
                    returnsite_id,
                    "icfg_return_external",
                    call.line,
                )

        return {
            "nodes": [nodes[node_id] for node_id in sorted(nodes)],
            "edges": [edges[edge_id] for edge_id in sorted(edges)],
        }

    def _collect_functions(
        self,
        tree: ast.AST,
        rel_file: str,
    ) -> dict[str, _FunctionSymbol]:
        found: dict[str, _FunctionSymbol] = {}

        def walk(body: list[ast.stmt], scope: list[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    walk(list(node.body), [*scope, node.name])
                    continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = ".".join([*scope, node.name]) if scope else node.name
                    entry_id = f"{rel_file}::{qualified}::entry::icfg"
                    exit_id = f"{rel_file}::{qualified}::exit::icfg"
                    found[qualified] = _FunctionSymbol(
                        qualified_name=qualified,
                        node=node,
                        entry_id=entry_id,
                        exit_id=exit_id,
                    )
                    walk(list(node.body), [*scope, node.name])

        walk(list(getattr(tree, "body", [])), [])
        return found

    def _collect_calls(
        self,
        functions: dict[str, _FunctionSymbol],
    ) -> list[_CallSite]:
        simple_name_map: dict[str, list[str]] = {}
        for qualified in functions:
            simple_name_map.setdefault(qualified.split(".")[-1], []).append(qualified)

        calls: list[_CallSite] = []
        for symbol in functions.values():
            for call in self._iter_calls(symbol.node):
                callee_name = self._call_name(call.func)
                calls.append(
                    _CallSite(
                        caller=symbol.qualified_name,
                        callee_name=callee_name,
                        resolved_callee=self._resolve_callee(
                            caller=symbol.qualified_name,
                            callee_name=callee_name,
                            functions=functions,
                            simple_name_map=simple_name_map,
                        ),
                        line=int(getattr(call, "lineno", getattr(symbol.node, "lineno", 1)) or 1),
                        col=int(getattr(call, "col_offset", 0) or 0),
                    )
                )
        return calls

    def _iter_calls(
        self,
        function_node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[ast.Call]:
        calls: list[ast.Call] = []

        def visit(node: ast.AST) -> None:
            if node is not function_node and isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                return
            if isinstance(node, ast.Call):
                calls.append(node)
            for child in ast.iter_child_nodes(node):
                visit(child)

        for stmt in function_node.body:
            visit(stmt)
        return calls

    def _resolve_callee(
        self,
        *,
        caller: str,
        callee_name: str,
        functions: dict[str, _FunctionSymbol],
        simple_name_map: dict[str, list[str]],
    ) -> str | None:
        cleaned = callee_name.strip()
        if not cleaned:
            return None
        if cleaned in functions:
            return cleaned

        caller_parts = caller.split(".")
        for idx in range(len(caller_parts), 0, -1):
            candidate = ".".join([*caller_parts[: idx - 1], cleaned])
            if candidate in functions:
                return candidate

        leaf = cleaned.split(".")[-1]
        matches = simple_name_map.get(leaf, [])
        if len(matches) == 1:
            return matches[0]
        return None

    def _recursive_component_map(
        self,
        functions: dict[str, _FunctionSymbol],
        calls: list[_CallSite],
    ) -> dict[str, int]:
        adjacency: dict[str, set[str]] = {name: set() for name in functions}
        for call in calls:
            if call.resolved_callee in functions:
                adjacency[call.caller].add(call.resolved_callee)

        index = 0
        stack: list[str] = []
        index_by_node: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        on_stack: set[str] = set()
        components: list[list[str]] = []

        def strongconnect(node: str) -> None:
            nonlocal index
            index_by_node[node] = index
            lowlink[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in adjacency.get(node, set()):
                if neighbor not in index_by_node:
                    strongconnect(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], index_by_node[neighbor])

            if lowlink[node] != index_by_node[node]:
                return

            component: list[str] = []
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(component)

        for node in adjacency:
            if node not in index_by_node:
                strongconnect(node)

        component_map: dict[str, int] = {}
        for idx, component in enumerate(components):
            if len(component) > 1:
                for member in component:
                    component_map[member] = idx
                continue
            member = component[0]
            if member in adjacency.get(member, set()):
                component_map[member] = idx
        return component_map

    @staticmethod
    def _is_recursive_edge(
        caller: str,
        callee: str,
        component_map: dict[str, int],
    ) -> bool:
        return caller in component_map and callee in component_map and component_map[caller] == component_map[callee]

    @staticmethod
    def _callsite_id(rel_file: str, caller: str, line: int, col: int) -> str:
        return f"{rel_file}::{caller}::callsite::{int(line)}:{int(col)}::icfg"

    @staticmethod
    def _returnsite_id(rel_file: str, caller: str, line: int, col: int) -> str:
        return f"{rel_file}::{caller}::returnsite::{int(line)}:{int(col)}::icfg"

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = ICFGBuilder._call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        if isinstance(node, ast.Call):
            return ICFGBuilder._call_name(node.func)
        return ""

    @staticmethod
    def _add_edge(
        edges: dict[str, dict[str, Any]],
        src: str,
        dst: str,
        relation: str,
        line: int,
        *,
        recursive_backedge: bool = False,
    ) -> None:
        edge_id = f"{src}->{dst}::{relation}::{int(line)}"
        if recursive_backedge:
            edge_id = f"{edge_id}::recursive"
        edges[edge_id] = {
            "id": edge_id,
            "from": src,
            "to": dst,
            "relation": relation,
            "line": int(line),
            "source": "icfg",
            "recursive_backedge": bool(recursive_backedge),
        }
