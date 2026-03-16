from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CFGNode:
    """Serializable CFG node with optional concurrency metadata."""

    id: str
    type: str
    name: str
    function: str
    file: str
    line: int
    end_line: int
    source: str = "cfg"
    concurrency_role: str | None = None
    concurrency_group: str | None = None
    async_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "function": self.function,
            "file": self.file,
            "line": int(self.line),
            "end_line": int(self.end_line),
            "source": self.source,
            "concurrency_role": self.concurrency_role,
            "concurrency_group": self.concurrency_group,
            "async_state": self.async_state,
        }


@dataclass(slots=True)
class CFGEdge:
    """Serializable CFG edge with optional concurrency metadata."""

    id: str
    from_id: str
    to_id: str
    relation: str
    line: int
    source: str = "cfg"
    concurrency_marker: str | None = None
    recursive_backedge: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from": self.from_id,
            "to": self.to_id,
            "relation": self.relation,
            "line": int(self.line),
            "source": self.source,
            "concurrency_marker": self.concurrency_marker,
            "recursive_backedge": bool(self.recursive_backedge),
        }


@dataclass(slots=True)
class _FlowResult:
    first_node: str | None
    tails: list[str]
    raisers: list[str]


class CFGBuilder:
    """Build a lightweight, deterministic control-flow graph for Python files."""

    def build(self, rel_file: str, source: str) -> dict[str, list[dict[str, Any]]]:
        if not rel_file.endswith(".py"):
            return {"nodes": [], "edges": []}

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"nodes": [], "edges": []}

        nodes: dict[str, CFGNode] = {}
        edges: dict[str, CFGEdge] = {}

        for function_node, qualified_name in self._iter_functions(tree):
            entry_id = self._node_id(rel_file, f"{qualified_name}::entry", "cfg", function_node.lineno)
            exit_line = int(getattr(function_node, "end_lineno", function_node.lineno) or function_node.lineno)
            exit_id = self._node_id(rel_file, f"{qualified_name}::exit", "cfg", exit_line)
            nodes[entry_id] = CFGNode(
                id=entry_id,
                type="cfg_entry",
                name=f"{qualified_name}::entry",
                function=qualified_name,
                file=rel_file,
                line=int(function_node.lineno),
                end_line=int(function_node.lineno),
            )
            nodes[exit_id] = CFGNode(
                id=exit_id,
                type="cfg_exit",
                name=f"{qualified_name}::exit",
                function=qualified_name,
                file=rel_file,
                line=exit_line,
                end_line=exit_line,
            )

            flow = self._emit_block(
                rel_file=rel_file,
                function_name=qualified_name,
                statements=function_node.body,
                incoming=[entry_id],
                first_relation="cfg_next",
                nodes=nodes,
                edges=edges,
            )
            if not flow.first_node:
                self._add_edge(edges, entry_id, exit_id, "cfg_next", int(function_node.lineno))
            else:
                for tail in flow.tails:
                    self._add_edge(edges, tail, exit_id, "cfg_next", exit_line)

            for nested in ast.walk(function_node):
                if isinstance(nested, (ast.Return, ast.Raise)):
                    stmt_id = self._stmt_id(rel_file, qualified_name, nested)
                    if stmt_id in nodes:
                        self._add_edge(
                            edges,
                            stmt_id,
                            exit_id,
                            "cfg_terminate",
                            int(getattr(nested, "lineno", function_node.lineno) or function_node.lineno),
                        )

        return {
            "nodes": [nodes[node_id].to_dict() for node_id in sorted(nodes)],
            "edges": [edges[edge_id].to_dict() for edge_id in sorted(edges)],
        }

    def _emit_block(
        self,
        *,
        rel_file: str,
        function_name: str,
        statements: list[ast.stmt],
        incoming: list[str],
        first_relation: str,
        nodes: dict[str, CFGNode],
        edges: dict[str, CFGEdge],
    ) -> _FlowResult:
        if not statements:
            return _FlowResult(first_node=None, tails=list(incoming), raisers=[])

        first_node: str | None = None
        pending = list(incoming)
        relation = first_relation
        block_raisers: list[str] = []

        for stmt in statements:
            stmt_id = self._stmt_id(rel_file, function_name, stmt)
            line = int(getattr(stmt, "lineno", 1) or 1)
            end_line = int(getattr(stmt, "end_lineno", line) or line)
            nodes[stmt_id] = CFGNode(
                id=stmt_id,
                type="cfg_stmt",
                name=type(stmt).__name__,
                function=function_name,
                file=rel_file,
                line=line,
                end_line=end_line,
            )
            if first_node is None:
                first_node = stmt_id
            for src in pending:
                self._add_edge(edges, src, stmt_id, relation, line)

            own_raisers = [stmt_id] if self._stmt_can_raise(stmt) else []
            control_exit = stmt_id
            if self._supports_concurrency_markers(stmt):
                control_exit = self._emit_concurrency_markers(
                    rel_file=rel_file,
                    function_name=function_name,
                    stmt=stmt,
                    stmt_id=stmt_id,
                    nodes=nodes,
                    edges=edges,
                )

            if isinstance(stmt, ast.If):
                then_flow = self._emit_block(
                    rel_file=rel_file,
                    function_name=function_name,
                    statements=stmt.body,
                    incoming=[stmt_id],
                    first_relation="cfg_true",
                    nodes=nodes,
                    edges=edges,
                )
                else_flow = (
                    self._emit_block(
                        rel_file=rel_file,
                        function_name=function_name,
                        statements=stmt.orelse,
                        incoming=[stmt_id],
                        first_relation="cfg_false",
                        nodes=nodes,
                        edges=edges,
                    )
                    if stmt.orelse
                    else _FlowResult(first_node=None, tails=[stmt_id], raisers=[])
                )
                pending = self._dedupe(then_flow.tails + else_flow.tails)
                block_raisers.extend(self._dedupe(own_raisers + then_flow.raisers + else_flow.raisers))
                relation = "cfg_next"
                continue

            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                loop_flow = self._emit_block(
                    rel_file=rel_file,
                    function_name=function_name,
                    statements=stmt.body,
                    incoming=[stmt_id],
                    first_relation="cfg_loop",
                    nodes=nodes,
                    edges=edges,
                )
                for loop_tail in loop_flow.tails:
                    self._add_edge(edges, loop_tail, stmt_id, "cfg_back_edge", line)
                else_flow = (
                    self._emit_block(
                        rel_file=rel_file,
                        function_name=function_name,
                        statements=stmt.orelse,
                        incoming=[stmt_id],
                        first_relation="cfg_else",
                        nodes=nodes,
                        edges=edges,
                    )
                    if stmt.orelse
                    else _FlowResult(first_node=None, tails=[stmt_id], raisers=[])
                )
                pending = self._dedupe([stmt_id] + else_flow.tails)
                block_raisers.extend(self._dedupe(own_raisers + loop_flow.raisers + else_flow.raisers))
                relation = "cfg_next"
                continue

            if isinstance(stmt, ast.Try):
                try_flow = self._emit_try(
                    rel_file=rel_file,
                    function_name=function_name,
                    stmt=stmt,
                    try_node_id=stmt_id,
                    nodes=nodes,
                    edges=edges,
                )
                pending = self._dedupe(try_flow.tails)
                block_raisers.extend(self._dedupe(own_raisers + try_flow.raisers))
                relation = "cfg_next"
                continue

            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                with_flow = self._emit_with(
                    rel_file=rel_file,
                    function_name=function_name,
                    stmt=stmt,
                    with_node_id=stmt_id,
                    nodes=nodes,
                    edges=edges,
                )
                pending = self._dedupe(with_flow.tails)
                block_raisers.extend(self._dedupe(own_raisers + with_flow.raisers))
                relation = "cfg_next"
                continue

            if isinstance(stmt, ast.Return):
                pending = []
                relation = "cfg_next"
                continue

            if isinstance(stmt, ast.Raise):
                block_raisers.extend([stmt_id])
                pending = []
                relation = "cfg_next"
                continue

            pending = [control_exit]
            block_raisers.extend(self._dedupe(own_raisers))
            relation = "cfg_next"

        return _FlowResult(
            first_node=first_node,
            tails=self._dedupe(pending),
            raisers=self._dedupe(block_raisers),
        )

    def _emit_try(
        self,
        *,
        rel_file: str,
        function_name: str,
        stmt: ast.Try,
        try_node_id: str,
        nodes: dict[str, CFGNode],
        edges: dict[str, CFGEdge],
    ) -> _FlowResult:
        try_line = int(getattr(stmt, "lineno", 1) or 1)
        body_flow = self._emit_block(
            rel_file=rel_file,
            function_name=function_name,
            statements=list(stmt.body),
            incoming=[try_node_id],
            first_relation="cfg_try",
            nodes=nodes,
            edges=edges,
        )

        handler_entries: list[str] = []
        handler_tails: list[str] = []
        handler_raisers: list[str] = []
        for idx, handler in enumerate(stmt.handlers):
            handler_line = int(getattr(handler, "lineno", try_line) or try_line)
            handler_id = self._node_id(
                rel_file,
                f"{function_name}::except::{handler_line}:{idx}",
                "cfg",
                handler_line,
            )
            nodes[handler_id] = CFGNode(
                id=handler_id,
                type="cfg_except",
                name="ExceptHandler",
                function=function_name,
                file=rel_file,
                line=handler_line,
                end_line=int(getattr(handler, "end_lineno", handler_line) or handler_line),
            )
            self._add_edge(edges, try_node_id, handler_id, "cfg_except", handler_line)
            handler_flow = self._emit_block(
                rel_file=rel_file,
                function_name=function_name,
                statements=list(handler.body),
                incoming=[handler_id],
                first_relation="cfg_next",
                nodes=nodes,
                edges=edges,
            )
            handler_entries.append(handler_id)
            handler_tails.extend(handler_flow.tails if handler_flow.tails else [handler_id])
            handler_raisers.extend(handler_flow.raisers)

        for raised_node in body_flow.raisers:
            for handler_id in handler_entries:
                self._add_edge(edges, raised_node, handler_id, "cfg_exception", try_line)

        final_incoming = self._dedupe(body_flow.tails + handler_tails)
        propagated_raisers = self._dedupe(body_flow.raisers + handler_raisers)
        if stmt.finalbody:
            finally_line = int(
                getattr(stmt.finalbody[0], "lineno", getattr(stmt, "end_lineno", try_line))
                if stmt.finalbody
                else getattr(stmt, "end_lineno", try_line)
            )
            finally_id = self._node_id(
                rel_file,
                f"{function_name}::finally::{finally_line}",
                "cfg",
                finally_line,
            )
            nodes[finally_id] = CFGNode(
                id=finally_id,
                type="cfg_finally",
                name="Finally",
                function=function_name,
                file=rel_file,
                line=finally_line,
                end_line=finally_line,
            )
            if final_incoming:
                for src in final_incoming:
                    self._add_edge(edges, src, finally_id, "cfg_finally", finally_line)
            else:
                self._add_edge(edges, try_node_id, finally_id, "cfg_finally", finally_line)

            for raised_node in propagated_raisers:
                self._add_edge(edges, raised_node, finally_id, "cfg_finally_exception", finally_line)

            final_flow = self._emit_block(
                rel_file=rel_file,
                function_name=function_name,
                statements=list(stmt.finalbody),
                incoming=[finally_id],
                first_relation="cfg_next",
                nodes=nodes,
                edges=edges,
            )
            return _FlowResult(
                first_node=body_flow.first_node,
                tails=self._dedupe(final_flow.tails if final_flow.tails else [finally_id]),
                raisers=self._dedupe(propagated_raisers + final_flow.raisers),
            )

        return _FlowResult(
            first_node=body_flow.first_node,
            tails=final_incoming,
            raisers=propagated_raisers,
        )

    def _emit_with(
        self,
        *,
        rel_file: str,
        function_name: str,
        stmt: ast.With | ast.AsyncWith,
        with_node_id: str,
        nodes: dict[str, CFGNode],
        edges: dict[str, CFGEdge],
    ) -> _FlowResult:
        line = int(getattr(stmt, "lineno", 1) or 1)
        body_flow = self._emit_block(
            rel_file=rel_file,
            function_name=function_name,
            statements=list(stmt.body),
            incoming=[with_node_id],
            first_relation="cfg_with",
            nodes=nodes,
            edges=edges,
        )
        cleanup_line = int(getattr(stmt, "end_lineno", line) or line)
        cleanup_id = self._node_id(
            rel_file,
            f"{function_name}::with_cleanup::{line}",
            "cfg",
            cleanup_line,
        )
        nodes[cleanup_id] = CFGNode(
            id=cleanup_id,
            type="cfg_with_finally",
            name="WithCleanup",
            function=function_name,
            file=rel_file,
            line=cleanup_line,
            end_line=cleanup_line,
        )
        self._add_edge(edges, with_node_id, cleanup_id, "cfg_with_finally", cleanup_line)
        for tail in body_flow.tails:
            self._add_edge(edges, tail, cleanup_id, "cfg_with_finally", cleanup_line)
        for raised_node in body_flow.raisers:
            self._add_edge(
                edges,
                raised_node,
                cleanup_id,
                "cfg_with_finally_exception",
                cleanup_line,
            )
        return _FlowResult(
            first_node=body_flow.first_node,
            tails=[cleanup_id],
            raisers=self._dedupe(body_flow.raisers),
        )

    def _emit_concurrency_markers(
        self,
        *,
        rel_file: str,
        function_name: str,
        stmt: ast.stmt,
        stmt_id: str,
        nodes: dict[str, CFGNode],
        edges: dict[str, CFGEdge],
    ) -> str:
        line = int(getattr(stmt, "lineno", 1) or 1)
        col = int(getattr(stmt, "col_offset", 0) or 0)
        group = f"{function_name}:{line}:{col}"
        control_exit = stmt_id

        if any(isinstance(node, ast.Await) for node in ast.walk(stmt)):
            suspend_id = self._node_id(
                rel_file,
                f"{function_name}::suspend::{line}:{col}",
                "cfg",
                line,
            )
            resume_id = self._node_id(
                rel_file,
                f"{function_name}::resume::{line}:{col}",
                "cfg",
                line,
            )
            nodes[suspend_id] = CFGNode(
                id=suspend_id,
                type="cfg_concurrency",
                name="Suspend",
                function=function_name,
                file=rel_file,
                line=line,
                end_line=line,
                concurrency_role="suspend",
                concurrency_group=group,
                async_state="suspended",
            )
            nodes[resume_id] = CFGNode(
                id=resume_id,
                type="cfg_concurrency",
                name="Resume",
                function=function_name,
                file=rel_file,
                line=line,
                end_line=line,
                concurrency_role="resume",
                concurrency_group=group,
                async_state="resumed",
            )
            self._add_edge(
                edges,
                stmt_id,
                suspend_id,
                "cfg_suspend",
                line,
                concurrency_marker="suspend",
            )
            self._add_edge(
                edges,
                suspend_id,
                resume_id,
                "cfg_resume",
                line,
                concurrency_marker="resume",
            )
            control_exit = resume_id

        for idx, call in enumerate(self._collect_calls(stmt, suffix="create_task")):
            call_line = int(getattr(call, "lineno", line) or line)
            fork_id = self._node_id(
                rel_file,
                f"{function_name}::fork::{call_line}:{idx}",
                "cfg",
                call_line,
            )
            task_id = self._node_id(
                rel_file,
                f"{function_name}::task::{call_line}:{idx}",
                "cfg",
                call_line,
            )
            nodes[fork_id] = CFGNode(
                id=fork_id,
                type="cfg_concurrency",
                name="Fork",
                function=function_name,
                file=rel_file,
                line=call_line,
                end_line=call_line,
                concurrency_role="fork",
                concurrency_group=group,
                async_state="spawned",
            )
            nodes[task_id] = CFGNode(
                id=task_id,
                type="cfg_concurrency",
                name="Task",
                function=function_name,
                file=rel_file,
                line=call_line,
                end_line=call_line,
                concurrency_role="task",
                concurrency_group=group,
                async_state="running",
            )
            self._add_edge(
                edges,
                stmt_id,
                fork_id,
                "cfg_fork",
                call_line,
                concurrency_marker="fork",
            )
            self._add_edge(
                edges,
                fork_id,
                task_id,
                "cfg_spawn",
                call_line,
                concurrency_marker="fork",
            )

        gather_calls = self._collect_calls(stmt, suffix="gather")
        if gather_calls:
            gather_line = int(getattr(gather_calls[0], "lineno", line) or line)
            join_id = self._node_id(
                rel_file,
                f"{function_name}::join::{gather_line}:{col}",
                "cfg",
                gather_line,
            )
            nodes[join_id] = CFGNode(
                id=join_id,
                type="cfg_concurrency",
                name="Join",
                function=function_name,
                file=rel_file,
                line=gather_line,
                end_line=gather_line,
                concurrency_role="join",
                concurrency_group=group,
                async_state="joined",
            )
            self._add_edge(
                edges,
                control_exit,
                join_id,
                "cfg_join",
                gather_line,
                concurrency_marker="join",
            )
            control_exit = join_id

        return control_exit

    def _iter_functions(self, tree: ast.AST) -> list[tuple[ast.AST, str]]:
        found: list[tuple[ast.AST, str]] = []

        def walk(body: list[ast.stmt], scope: list[str]) -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    walk(node.body, [*scope, node.name])
                    continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified_name = ".".join([*scope, node.name]) if scope else node.name
                    found.append((node, qualified_name))
                    walk(node.body, [*scope, node.name])

        walk(getattr(tree, "body", []), [])
        return found

    @staticmethod
    def _stmt_id(rel_file: str, function_name: str, stmt: ast.stmt) -> str:
        line = int(getattr(stmt, "lineno", 1) or 1)
        col = int(getattr(stmt, "col_offset", 0) or 0)
        return f"{rel_file}::{function_name}::{type(stmt).__name__}::{line}:{col}"

    @staticmethod
    def _node_id(rel_file: str, name: str, kind: str, line: int) -> str:
        return f"{rel_file}::{name}::{kind}::{int(line)}"

    @staticmethod
    def _add_edge(
        edges: dict[str, CFGEdge],
        src: str,
        dst: str,
        relation: str,
        line: int,
        *,
        concurrency_marker: str | None = None,
        recursive_backedge: bool = False,
    ) -> None:
        edge_id = f"{src}->{dst}::{relation}::{int(line)}"
        if concurrency_marker:
            edge_id = f"{edge_id}::{concurrency_marker}"
        if recursive_backedge:
            edge_id = f"{edge_id}::recursive"
        edges[edge_id] = CFGEdge(
            id=edge_id,
            from_id=src,
            to_id=dst,
            relation=relation,
            line=int(line),
            concurrency_marker=concurrency_marker,
            recursive_backedge=recursive_backedge,
        )

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        return [value for value in dict.fromkeys(values) if value]

    @staticmethod
    def _supports_concurrency_markers(stmt: ast.stmt) -> bool:
        return not isinstance(
            stmt,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.With,
                ast.AsyncWith,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Match,
            ),
        )

    @staticmethod
    def _collect_calls(stmt: ast.stmt, *, suffix: str) -> list[ast.Call]:
        matched: list[ast.Call] = []
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            call_name = CFGBuilder._call_name(node.func)
            if call_name.endswith(suffix):
                matched.append(node)
        return matched

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = CFGBuilder._call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return CFGBuilder._call_name(node.func)
        return ""

    @staticmethod
    def _stmt_can_raise(stmt: ast.stmt) -> bool:
        if isinstance(stmt, (ast.Pass, ast.Break, ast.Continue, ast.Global, ast.Nonlocal)):
            return False
        if isinstance(stmt, ast.Return):
            return False
        if isinstance(stmt, ast.Raise):
            return True
        for node in ast.walk(stmt):
            if isinstance(
                node,
                (
                    ast.Call,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.Subscript,
                    ast.Attribute,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Compare,
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                return True
        return isinstance(
            stmt,
            (
                ast.Assert,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Delete,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.If,
                ast.Try,
                ast.With,
                ast.AsyncWith,
            ),
        )
