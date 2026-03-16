from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaintFlow:
    """Represents one unsanitized source-to-sink taint path."""

    source: str
    sink: str
    variable: str
    source_line: int
    sink_line: int
    scope: str


class TaintTracker:
    """Lightweight taint tracker with source/sink/sanitizer heuristics."""

    DEFAULT_SOURCES = frozenset(
        {
            "input",
            "getpass.getpass",
            "os.getenv",
            "request.args.get",
            "request.form.get",
            "request.get_json",
            "sys.argv",
            "environ.get",
        }
    )
    DEFAULT_SINKS = frozenset(
        {
            "eval",
            "exec",
            "os.system",
            "subprocess.run",
            "subprocess.call",
            "subprocess.popen",
            "subprocess.check_output",
            "pickle.loads",
            "yaml.load",
        }
    )
    DEFAULT_SANITIZERS = frozenset(
        {
            "escape",
            "html.escape",
            "shlex.quote",
            "bleach.clean",
            "quote",
            "sanitize",
            "clean",
        }
    )

    def __init__(
        self,
        *,
        sources: set[str] | None = None,
        sinks: set[str] | None = None,
        sanitizers: set[str] | None = None,
    ) -> None:
        self.sources = frozenset(
            part.strip().lower()
            for part in (sources if sources is not None else self.DEFAULT_SOURCES)
            if part and part.strip()
        )
        self.sinks = frozenset(
            part.strip().lower()
            for part in (sinks if sinks is not None else self.DEFAULT_SINKS)
            if part and part.strip()
        )
        self.sanitizers = frozenset(
            part.strip().lower()
            for part in (
                sanitizers if sanitizers is not None else self.DEFAULT_SANITIZERS
            )
            if part and part.strip()
        )

    def analyze(self, rel_file: str, source: str) -> list[TaintFlow]:
        if not rel_file.endswith(".py"):
            return []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        return self.analyze_from_tree(rel_file, tree)

    def analyze_from_tree(self, rel_file: str, tree: ast.AST) -> list[TaintFlow]:
        analyzer = _TaintAnalyzer(tracker=self, rel_file=rel_file)
        analyzer.walk_block(getattr(tree, "body", []), env={}, scope="<module>")
        return analyzer.flows()

    def analyze_interprocedural(
        self,
        rel_file: str,
        source: str,
        *,
        call_resolver: Any | None = None,
    ) -> list[TaintFlow]:
        # Interprocedural support is intentionally a no-op stub for now.
        _ = call_resolver
        return self.analyze(rel_file, source)

    def _is_source(self, call_name: str) -> bool:
        return self._matches_hint(call_name, self.sources)

    def _is_sink(self, call_name: str) -> bool:
        return self._matches_hint(call_name, self.sinks)

    def _is_sanitizer(self, call_name: str) -> bool:
        return self._matches_hint(call_name, self.sanitizers)

    @staticmethod
    def _matches_hint(call_name: str, hints: frozenset[str]) -> bool:
        normalized = call_name.strip().lower()
        if not normalized:
            return False
        return any(normalized == hint or normalized.endswith(hint) for hint in hints)


class TypeInferencer:
    """Flow-sensitive, heuristic type inferencer for Python symbols."""

    DEFAULT_TENSOR_FACTORIES = frozenset(
        {
            "tensor",
            "as_tensor",
            "from_numpy",
            "array",
            "zeros",
            "ones",
            "rand",
            "randn",
            "empty",
            "full",
        }
    )

    def __init__(self, *, tensor_factories: set[str] | None = None) -> None:
        self.tensor_factories = frozenset(
            part.strip().lower()
            for part in (
                tensor_factories
                if tensor_factories is not None
                else self.DEFAULT_TENSOR_FACTORIES
            )
            if part and part.strip()
        )

    def analyze(self, rel_file: str, source: str) -> dict[str, dict[str, tuple[str, ...]]]:
        if not rel_file.endswith(".py"):
            return {}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {}
        return self.analyze_from_tree(rel_file, tree)

    def analyze_from_tree(
        self,
        rel_file: str,
        tree: ast.AST,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        _ = rel_file
        analyzer = _TypeAnalyzer(tensor_factories=self.tensor_factories)
        analyzer.walk_block(getattr(tree, "body", []), env={}, scope="<module>")
        return analyzer.serialized()

    def analyze_interprocedural(
        self,
        rel_file: str,
        source: str,
        *,
        call_resolver: Any | None = None,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        # Interprocedural support is intentionally a no-op stub for now.
        _ = call_resolver
        return self.analyze(rel_file, source)


class DFGBuilder:
    """Build a deterministic intra-file data-flow graph for Python code."""

    def __init__(
        self,
        *,
        taint_tracker: TaintTracker | None = None,
        type_inferencer: TypeInferencer | None = None,
    ) -> None:
        self.taint_tracker = taint_tracker or TaintTracker()
        self.type_inferencer = type_inferencer or TypeInferencer()

    def build(self, rel_file: str, source: str) -> dict[str, list[dict[str, Any]]]:
        if not rel_file.endswith(".py"):
            return {"nodes": [], "edges": []}

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return {"nodes": [], "edges": []}

        visitor = _DFGVisitor(rel_file)
        visitor.visit(tree)
        payload = visitor.to_payload()

        inferred = self.type_inferencer.analyze_from_tree(rel_file, tree)
        self._apply_type_hints(payload.get("nodes", []), inferred)

        taint_flows = self.taint_tracker.analyze_from_tree(rel_file, tree)
        self._append_taint_payload(payload, rel_file, taint_flows)

        return {
            "nodes": sorted(
                payload.get("nodes", []),
                key=lambda item: str(item.get("id") or ""),
            ),
            "edges": sorted(
                payload.get("edges", []),
                key=lambda item: str(item.get("id") or ""),
            ),
        }

    def infer_types(self, rel_file: str, source: str) -> dict[str, dict[str, tuple[str, ...]]]:
        return self.type_inferencer.analyze(rel_file, source)

    def analyze_taint(self, rel_file: str, source: str) -> list[TaintFlow]:
        return self.taint_tracker.analyze(rel_file, source)

    def build_interprocedural(
        self,
        rel_file: str,
        source: str,
        *,
        call_resolver: Any | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        # Interprocedural support is intentionally a fallback stub.
        _ = call_resolver
        return self.build(rel_file, source)

    @staticmethod
    def _apply_type_hints(
        nodes: list[dict[str, Any]],
        inferred: dict[str, dict[str, tuple[str, ...]]],
    ) -> None:
        for node in nodes:
            if str(node.get("type") or "") != "dfg_def":
                continue
            scope = str(node.get("scope") or "")
            name = str(node.get("name") or "")
            hints = inferred.get(scope, {}).get(name, ())
            clean_hints = [hint for hint in hints if hint != "unknown"]
            if clean_hints:
                node["type_hints"] = clean_hints

    @staticmethod
    def _append_taint_payload(
        payload: dict[str, list[dict[str, Any]]],
        rel_file: str,
        flows: list[TaintFlow],
    ) -> None:
        if not flows:
            return

        nodes = payload.setdefault("nodes", [])
        edges = payload.setdefault("edges", [])
        node_index = {str(item.get("id") or "") for item in nodes}
        edge_index = {str(item.get("id") or "") for item in edges}

        for index, flow in enumerate(flows):
            src_id = (
                f"{rel_file}::{flow.scope}::{flow.source}::"
                f"taint_source::{flow.source_line}:{index}"
            )
            if src_id not in node_index:
                nodes.append(
                    {
                        "id": src_id,
                        "type": "dfg_taint_source",
                        "name": flow.source,
                        "scope": flow.scope,
                        "file": rel_file,
                        "line": int(flow.source_line),
                        "end_line": int(flow.source_line),
                        "source": "dfg_taint",
                    }
                )
                node_index.add(src_id)

            sink_id = (
                f"{rel_file}::{flow.scope}::{flow.sink}::"
                f"taint_sink::{flow.sink_line}:{index}"
            )
            if sink_id not in node_index:
                nodes.append(
                    {
                        "id": sink_id,
                        "type": "dfg_taint_sink",
                        "name": flow.sink,
                        "scope": flow.scope,
                        "file": rel_file,
                        "line": int(flow.sink_line),
                        "end_line": int(flow.sink_line),
                        "source": "dfg_taint",
                    }
                )
                node_index.add(sink_id)

            edge_id = f"{src_id}->{sink_id}::dfg_taint_flow::{int(flow.sink_line)}"
            if edge_id in edge_index:
                continue
            edges.append(
                {
                    "id": edge_id,
                    "from": src_id,
                    "to": sink_id,
                    "relation": "dfg_taint_flow",
                    "line": int(flow.sink_line),
                    "variable": flow.variable,
                    "source": "dfg_taint",
                }
            )
            edge_index.add(edge_id)


class _DFGVisitor(ast.NodeVisitor):
    def __init__(self, rel_file: str) -> None:
        self.rel_file = rel_file
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.scope: list[str] = ["<module>"]
        self.last_definition: dict[tuple[str, str], str] = {}

    def to_payload(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "nodes": [self.nodes[node_id] for node_id in sorted(self.nodes)],
            "edges": [self.edges[edge_id] for edge_id in sorted(self.edges)],
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._enter_scope(node.name)
        for arg in node.args.args + node.args.kwonlyargs:
            self._define_symbol(
                arg.arg,
                int(getattr(arg, "lineno", node.lineno) or node.lineno),
                int(getattr(arg, "col_offset", 0) or 0),
            )
        if node.args.vararg:
            self._define_symbol(
                node.args.vararg.arg,
                int(getattr(node.args.vararg, "lineno", node.lineno) or node.lineno),
                int(getattr(node.args.vararg, "col_offset", 0) or 0),
            )
        if node.args.kwarg:
            self._define_symbol(
                node.args.kwarg.arg,
                int(getattr(node.args.kwarg, "lineno", node.lineno) or node.lineno),
                int(getattr(node.args.kwarg, "col_offset", 0) or 0),
            )
        for stmt in node.body:
            self.visit(stmt)
        self._leave_scope()
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._enter_scope(node.name)
        for stmt in node.body:
            self.visit(stmt)
        self._leave_scope()
        return None

    def visit_Assign(self, node: ast.Assign) -> Any:
        source_defs = self._consume_expression(node.value)
        for target in node.targets:
            self._assign_target(target, source_defs, node)
        return None

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        source_defs = self._consume_expression(node.value) if node.value is not None else []
        self._assign_target(node.target, source_defs, node)
        return None

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        left_defs = self._consume_expression(node.target)
        right_defs = self._consume_expression(node.value)
        source_defs = [*left_defs, *right_defs]
        self._assign_target(node.target, source_defs, node)
        return None

    def visit_For(self, node: ast.For) -> Any:
        source_defs = self._consume_expression(node.iter)
        self._assign_target(node.target, source_defs, node)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        return None

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        return self.visit_For(node)

    def visit_With(self, node: ast.With) -> Any:
        for item in node.items:
            source_defs = self._consume_expression(item.context_expr)
            if item.optional_vars is not None:
                self._assign_target(item.optional_vars, source_defs, node)
        for stmt in node.body:
            self.visit(stmt)
        return None

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:
        return self.visit_With(node)

    def visit_Expr(self, node: ast.Expr) -> Any:
        self._consume_expression(node.value)
        return None

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is not None:
            self._consume_expression(node.value)
        return None

    def visit_If(self, node: ast.If) -> Any:
        self._consume_expression(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        return None

    def visit_While(self, node: ast.While) -> Any:
        self._consume_expression(node.test)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        return None

    def visit_Try(self, node: ast.Try) -> Any:
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            if handler.type is not None:
                self._consume_expression(handler.type)
            if handler.name:
                self._define_symbol(
                    handler.name,
                    int(getattr(handler, "lineno", node.lineno) or node.lineno),
                    0,
                )
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)
        return None

    def _consume_expression(self, node: ast.AST | None) -> list[str]:
        if node is None:
            return []
        source_defs: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                source_defs.extend(
                    self._register_use(
                        child.id,
                        int(getattr(child, "lineno", 1) or 1),
                        int(getattr(child, "col_offset", 0) or 0),
                    )
                )
        return source_defs

    def _assign_target(self, target: ast.AST, source_defs: list[str], anchor: ast.AST) -> None:
        for name_node in self._iter_target_names(target):
            line = int(getattr(name_node, "lineno", getattr(anchor, "lineno", 1)) or 1)
            col = int(getattr(name_node, "col_offset", 0) or 0)
            def_id = self._define_symbol(name_node.id, line, col)
            for src in source_defs:
                self._add_edge(src, def_id, "dfg_depends_on", line, variable=name_node.id)

    def _iter_target_names(self, node: ast.AST) -> list[ast.Name]:
        names: list[ast.Name] = []
        if isinstance(node, ast.Name):
            names.append(node)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for child in node.elts:
                names.extend(self._iter_target_names(child))
        elif isinstance(node, ast.Attribute):
            # Attribute writes are outside local symbol tracking scope.
            return names
        else:
            for child in ast.iter_child_nodes(node):
                names.extend(self._iter_target_names(child))
        return names

    def _register_use(self, symbol: str, line: int, col: int) -> list[str]:
        use_id = self._node_id(symbol, "use", line, col)
        self.nodes[use_id] = {
            "id": use_id,
            "type": "dfg_use",
            "name": symbol,
            "scope": self._scope_name(),
            "file": self.rel_file,
            "line": line,
            "end_line": line,
            "source": "dfg",
        }
        dependencies: list[str] = []
        local_key = (self._scope_name(), symbol)
        module_key = ("<module>", symbol)
        for key in (local_key, module_key):
            def_id = self.last_definition.get(key)
            if def_id:
                dependencies.append(def_id)
                self._add_edge(def_id, use_id, "dfg_reaches", line, variable=symbol)
        return dependencies

    def _define_symbol(self, symbol: str, line: int, col: int) -> str:
        def_id = self._node_id(symbol, "def", line, col)
        self.nodes[def_id] = {
            "id": def_id,
            "type": "dfg_def",
            "name": symbol,
            "scope": self._scope_name(),
            "file": self.rel_file,
            "line": line,
            "end_line": line,
            "source": "dfg",
        }
        self.last_definition[(self._scope_name(), symbol)] = def_id
        return def_id

    def _add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        line: int,
        *,
        variable: str,
    ) -> None:
        edge_id = f"{src}->{dst}::{relation}::{int(line)}"
        self.edges[edge_id] = {
            "id": edge_id,
            "from": src,
            "to": dst,
            "relation": relation,
            "line": int(line),
            "variable": variable,
            "source": "dfg",
        }

    def _enter_scope(self, name: str) -> None:
        self.scope.append(name)

    def _leave_scope(self) -> None:
        if len(self.scope) > 1:
            self.scope.pop()

    def _scope_name(self) -> str:
        return ".".join(self.scope)

    def _node_id(self, symbol: str, role: str, line: int, col: int) -> str:
        return (
            f"{self.rel_file}::{self._scope_name()}::{symbol}::"
            f"{role}::{int(line)}:{int(col)}"
        )


class _TaintAnalyzer:
    def __init__(self, *, tracker: TaintTracker, rel_file: str) -> None:
        self.tracker = tracker
        self.rel_file = rel_file
        self._flows: dict[tuple[str, str, str, int, int, str], TaintFlow] = {}

    def flows(self) -> list[TaintFlow]:
        return sorted(
            self._flows.values(),
            key=lambda item: (
                int(item.sink_line),
                int(item.source_line),
                item.scope,
                item.sink,
                item.source,
                item.variable,
            ),
        )

    def walk_block(self, statements: list[ast.stmt], env: dict[str, dict[str, int]], scope: str) -> None:
        for stmt in statements:
            self.walk_stmt(stmt, env, scope)

    def walk_stmt(self, stmt: ast.stmt, env: dict[str, dict[str, int]], scope: str) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_scope = f"{scope}.{stmt.name}"
            fn_env: dict[str, dict[str, int]] = {}
            for arg in list(stmt.args.args) + list(stmt.args.kwonlyargs):
                fn_env[arg.arg] = {}
            if stmt.args.vararg:
                fn_env[stmt.args.vararg.arg] = {}
            if stmt.args.kwarg:
                fn_env[stmt.args.kwarg.arg] = {}
            self.walk_block(stmt.body, fn_env, fn_scope)
            return

        if isinstance(stmt, ast.ClassDef):
            class_scope = f"{scope}.{stmt.name}"
            self.walk_block(stmt.body, self.clone_env(env), class_scope)
            return

        if isinstance(stmt, ast.Assign):
            sources = self.eval_expr(stmt.value, env, scope)
            for target in stmt.targets:
                self.assign_target(env, target, sources)
            return

        if isinstance(stmt, ast.AnnAssign):
            sources = self.eval_expr(stmt.value, env, scope) if stmt.value is not None else {}
            self.assign_target(env, stmt.target, sources)
            return

        if isinstance(stmt, ast.AugAssign):
            right = self.eval_expr(stmt.value, env, scope)
            if isinstance(stmt.target, ast.Name):
                current = env.get(stmt.target.id, {})
                merged = self.merge_source_maps(current, right)
                if merged:
                    env[stmt.target.id] = merged
                else:
                    env.pop(stmt.target.id, None)
            else:
                self.eval_expr(stmt.target, env, scope)
            return

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            iter_sources = self.eval_expr(stmt.iter, env, scope)
            loop_env = self.clone_env(env)
            self.assign_target(loop_env, stmt.target, iter_sources)
            self.walk_block(stmt.body, loop_env, scope)
            else_env = self.clone_env(env)
            self.walk_block(stmt.orelse, else_env, scope)
            env.clear()
            env.update(self.merge_envs(env, loop_env, else_env))
            return

        if isinstance(stmt, ast.With):
            with_env = self.clone_env(env)
            for item in stmt.items:
                context_sources = self.eval_expr(item.context_expr, with_env, scope)
                if item.optional_vars is not None:
                    self.assign_target(with_env, item.optional_vars, context_sources)
            self.walk_block(stmt.body, with_env, scope)
            env.clear()
            env.update(self.merge_envs(env, with_env))
            return

        if isinstance(stmt, ast.AsyncWith):
            self.walk_stmt(ast.With(items=stmt.items, body=stmt.body, type_comment=None), env, scope)
            return

        if isinstance(stmt, ast.If):
            self.eval_expr(stmt.test, env, scope)
            body_env = self.clone_env(env)
            else_env = self.clone_env(env)
            self.walk_block(stmt.body, body_env, scope)
            self.walk_block(stmt.orelse, else_env, scope)
            env.clear()
            env.update(self.merge_envs(body_env, else_env))
            return

        if isinstance(stmt, ast.While):
            self.eval_expr(stmt.test, env, scope)
            loop_env = self.clone_env(env)
            self.walk_block(stmt.body, loop_env, scope)
            else_env = self.clone_env(env)
            self.walk_block(stmt.orelse, else_env, scope)
            env.clear()
            env.update(self.merge_envs(env, loop_env, else_env))
            return

        if isinstance(stmt, ast.Try):
            body_env = self.clone_env(env)
            self.walk_block(stmt.body, body_env, scope)
            orelse_env = self.clone_env(body_env)
            self.walk_block(stmt.orelse, orelse_env, scope)
            handler_envs: list[dict[str, dict[str, int]]] = []
            for handler in stmt.handlers:
                handler_env = self.clone_env(env)
                if handler.type is not None:
                    self.eval_expr(handler.type, handler_env, scope)
                if handler.name:
                    handler_env.pop(handler.name, None)
                self.walk_block(handler.body, handler_env, scope)
                handler_envs.append(handler_env)
            merged = self.merge_envs(body_env, orelse_env, *handler_envs)
            final_env = self.clone_env(merged)
            self.walk_block(stmt.finalbody, final_env, scope)
            env.clear()
            env.update(self.merge_envs(env, final_env))
            return

        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self.eval_expr(stmt.value, env, scope)
            return

        if isinstance(stmt, ast.Expr):
            self.eval_expr(stmt.value, env, scope)
            return

    def eval_expr(self, node: ast.AST | None, env: dict[str, dict[str, int]], scope: str) -> dict[str, int]:
        if node is None:
            return {}
        if isinstance(node, ast.Name):
            return dict(env.get(node.id, {}))
        if isinstance(node, ast.Attribute):
            return self.eval_expr(node.value, env, scope)
        if isinstance(node, ast.Constant):
            return {}
        if isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            arg_sources: dict[str, int] = {}
            for arg in node.args:
                arg_sources = self.merge_source_maps(arg_sources, self.eval_expr(arg, env, scope))
            for keyword in node.keywords:
                if keyword.value is not None:
                    arg_sources = self.merge_source_maps(
                        arg_sources,
                        self.eval_expr(keyword.value, env, scope),
                    )

            if self.tracker._is_sanitizer(call_name):
                return {}

            if self.tracker._is_sink(call_name) and arg_sources:
                sink_line = int(getattr(node, "lineno", 1) or 1)
                variable = self.first_tainted_variable(node, env)
                for source_name, source_line in sorted(arg_sources.items()):
                    key = (
                        source_name,
                        call_name or "<sink>",
                        variable,
                        int(source_line),
                        sink_line,
                        scope,
                    )
                    self._flows[key] = TaintFlow(
                        source=source_name,
                        sink=call_name or "<sink>",
                        variable=variable,
                        source_line=int(source_line),
                        sink_line=sink_line,
                        scope=scope,
                    )

            if self.tracker._is_source(call_name):
                source_line = int(getattr(node, "lineno", 1) or 1)
                return {call_name or "<source>": source_line}

            return arg_sources

        if isinstance(node, ast.BinOp):
            return self.merge_source_maps(
                self.eval_expr(node.left, env, scope),
                self.eval_expr(node.right, env, scope),
            )
        if isinstance(node, ast.BoolOp):
            merged: dict[str, int] = {}
            for value in node.values:
                merged = self.merge_source_maps(merged, self.eval_expr(value, env, scope))
            return merged
        if isinstance(node, ast.UnaryOp):
            return self.eval_expr(node.operand, env, scope)
        if isinstance(node, ast.Compare):
            merged = self.eval_expr(node.left, env, scope)
            for comp in node.comparators:
                merged = self.merge_source_maps(merged, self.eval_expr(comp, env, scope))
            return merged
        if isinstance(node, ast.IfExp):
            return self.merge_source_maps(
                self.eval_expr(node.body, env, scope),
                self.eval_expr(node.orelse, env, scope),
            )
        if isinstance(node, ast.Subscript):
            return self.merge_source_maps(
                self.eval_expr(node.value, env, scope),
                self.eval_expr(node.slice, env, scope),
            )
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            merged = {}
            for part in node.elts:
                merged = self.merge_source_maps(merged, self.eval_expr(part, env, scope))
            return merged
        if isinstance(node, ast.Dict):
            merged = {}
            for key, value in zip(node.keys, node.values):
                merged = self.merge_source_maps(merged, self.eval_expr(key, env, scope))
                merged = self.merge_source_maps(merged, self.eval_expr(value, env, scope))
            return merged
        if isinstance(node, ast.JoinedStr):
            merged = {}
            for part in node.values:
                merged = self.merge_source_maps(merged, self.eval_expr(part, env, scope))
            return merged
        if isinstance(node, ast.FormattedValue):
            return self.eval_expr(node.value, env, scope)
        if isinstance(node, ast.Lambda):
            return {}
        if isinstance(node, ast.Await):
            return self.eval_expr(node.value, env, scope)
        return {}

    def first_tainted_variable(self, call_node: ast.Call, env: dict[str, dict[str, int]]) -> str:
        names: list[str] = []
        for arg in call_node.args:
            for child in ast.walk(arg):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and env.get(child.id):
                    names.append(child.id)
        for keyword in call_node.keywords:
            if keyword.value is None:
                continue
            for child in ast.walk(keyword.value):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and env.get(child.id):
                    names.append(child.id)
        if not names:
            return "<expr>"
        return ",".join(sorted(set(names)))

    def assign_target(self, env: dict[str, dict[str, int]], target: ast.AST, sources: dict[str, int]) -> None:
        target_names = self.iter_target_names(target)
        for target_name in target_names:
            if sources:
                env[target_name] = dict(sources)
            else:
                env.pop(target_name, None)

    @staticmethod
    def iter_target_names(node: ast.AST) -> list[str]:
        names: list[str] = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                names.extend(_TaintAnalyzer.iter_target_names(elt))
        elif isinstance(node, ast.Starred):
            names.extend(_TaintAnalyzer.iter_target_names(node.value))
        return names

    @staticmethod
    def clone_env(env: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        return {name: dict(values) for name, values in env.items()}

    @staticmethod
    def merge_source_maps(*items: dict[str, int]) -> dict[str, int]:
        merged: dict[str, int] = {}
        for mapping in items:
            for source_name, line in mapping.items():
                current = merged.get(source_name)
                if current is None or int(line) < current:
                    merged[source_name] = int(line)
        return merged

    @staticmethod
    def merge_envs(*envs: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
        merged: dict[str, dict[str, int]] = {}
        for env in envs:
            for symbol, sources in env.items():
                current = merged.get(symbol, {})
                merged[symbol] = _TaintAnalyzer.merge_source_maps(current, sources)
        return merged


class _TypeAnalyzer:
    def __init__(self, *, tensor_factories: frozenset[str]) -> None:
        self.tensor_factories = tensor_factories
        self._results: dict[str, dict[str, set[str]]] = {}

    def serialized(self) -> dict[str, dict[str, tuple[str, ...]]]:
        payload: dict[str, dict[str, tuple[str, ...]]] = {}
        for scope in sorted(self._results):
            scope_payload: dict[str, tuple[str, ...]] = {}
            for name in sorted(self._results[scope]):
                scope_payload[name] = tuple(sorted(self._results[scope][name]))
            if scope_payload:
                payload[scope] = scope_payload
        return payload

    def walk_block(self, statements: list[ast.stmt], env: dict[str, set[str]], scope: str) -> None:
        for stmt in statements:
            self.walk_stmt(stmt, env, scope)
        self.record(scope, env)

    def walk_stmt(self, stmt: ast.stmt, env: dict[str, set[str]], scope: str) -> None:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            env[stmt.name] = {"callable"}
            fn_scope = f"{scope}.{stmt.name}"
            fn_env: dict[str, set[str]] = {}
            for arg in list(stmt.args.args) + list(stmt.args.kwonlyargs):
                fn_env[arg.arg] = {"unknown"}
            if stmt.args.vararg:
                fn_env[stmt.args.vararg.arg] = {"unknown"}
            if stmt.args.kwarg:
                fn_env[stmt.args.kwarg.arg] = {"unknown"}
            self.walk_block(stmt.body, fn_env, fn_scope)
            return

        if isinstance(stmt, ast.ClassDef):
            env[stmt.name] = {"type"}
            class_scope = f"{scope}.{stmt.name}"
            self.walk_block(stmt.body, env=self.clone_env(env), scope=class_scope)
            return

        if isinstance(stmt, ast.Assign):
            value_types = self.infer_expr(stmt.value, env)
            for target in stmt.targets:
                self.assign_target(env, target, value_types)
            return

        if isinstance(stmt, ast.AnnAssign):
            value_types = self.infer_expr(stmt.value, env) if stmt.value is not None else set()
            annotation_types = self.infer_annotation(stmt.annotation)
            self.assign_target(env, stmt.target, value_types | annotation_types)
            return

        if isinstance(stmt, ast.AugAssign):
            right_types = self.infer_expr(stmt.value, env)
            if isinstance(stmt.target, ast.Name):
                left_types = set(env.get(stmt.target.id, {"unknown"}))
                env[stmt.target.id] = self.merge_types(left_types, right_types)
            else:
                self.infer_expr(stmt.target, env)
            return

        if isinstance(stmt, ast.Expr):
            self.infer_expr(stmt.value, env)
            return

        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self.infer_expr(stmt.value, env)
            return

        if isinstance(stmt, ast.If):
            self.infer_expr(stmt.test, env)
            body_env = self.clone_env(env)
            else_env = self.clone_env(env)
            narrowing = self.extract_isinstance_narrowing(stmt.test)
            if narrowing is not None:
                symbol, types = narrowing
                self.apply_true_narrowing(body_env, symbol, types)
                self.apply_false_narrowing(else_env, symbol, types)
            self.walk_block(stmt.body, body_env, scope)
            self.walk_block(stmt.orelse, else_env, scope)
            env.clear()
            env.update(self.merge_envs(body_env, else_env))
            return

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            iter_types = self.infer_expr(stmt.iter, env)
            loop_env = self.clone_env(env)
            self.assign_target(loop_env, stmt.target, self.element_types(iter_types))
            self.walk_block(stmt.body, loop_env, scope)
            else_env = self.clone_env(env)
            self.walk_block(stmt.orelse, else_env, scope)
            env.clear()
            env.update(self.merge_envs(env, loop_env, else_env))
            return

        if isinstance(stmt, ast.While):
            self.infer_expr(stmt.test, env)
            loop_env = self.clone_env(env)
            self.walk_block(stmt.body, loop_env, scope)
            else_env = self.clone_env(env)
            self.walk_block(stmt.orelse, else_env, scope)
            env.clear()
            env.update(self.merge_envs(env, loop_env, else_env))
            return

        if isinstance(stmt, ast.With):
            with_env = self.clone_env(env)
            for item in stmt.items:
                self.infer_expr(item.context_expr, with_env)
                if item.optional_vars is not None:
                    self.assign_target(with_env, item.optional_vars, {"unknown"})
            self.walk_block(stmt.body, with_env, scope)
            env.clear()
            env.update(self.merge_envs(env, with_env))
            return

        if isinstance(stmt, ast.AsyncWith):
            self.walk_stmt(ast.With(items=stmt.items, body=stmt.body, type_comment=None), env, scope)
            return

        if isinstance(stmt, ast.Try):
            body_env = self.clone_env(env)
            self.walk_block(stmt.body, body_env, scope)
            orelse_env = self.clone_env(body_env)
            self.walk_block(stmt.orelse, orelse_env, scope)
            handler_envs: list[dict[str, set[str]]] = []
            for handler in stmt.handlers:
                handler_env = self.clone_env(env)
                if handler.type is not None:
                    self.infer_expr(handler.type, handler_env)
                if handler.name:
                    handler_env[handler.name] = {"unknown"}
                self.walk_block(handler.body, handler_env, scope)
                handler_envs.append(handler_env)
            merged = self.merge_envs(body_env, orelse_env, *handler_envs)
            self.walk_block(stmt.finalbody, merged, scope)
            env.clear()
            env.update(self.merge_envs(env, merged))
            return

    def infer_expr(self, node: ast.AST | None, env: dict[str, set[str]]) -> set[str]:
        if node is None:
            return {"unknown"}
        if isinstance(node, ast.Constant):
            return self.constant_type(node.value)
        if isinstance(node, ast.Name):
            return set(env.get(node.id, {"unknown"}))
        if isinstance(node, ast.List):
            return {"list"}
        if isinstance(node, ast.Tuple):
            return {"tuple"}
        if isinstance(node, ast.Set):
            return {"set"}
        if isinstance(node, ast.Dict):
            return {"dict"}
        if isinstance(node, ast.ListComp):
            return {"list"}
        if isinstance(node, ast.SetComp):
            return {"set"}
        if isinstance(node, ast.DictComp):
            return {"dict"}
        if isinstance(node, ast.GeneratorExp):
            return {"iterator"}
        if isinstance(node, ast.JoinedStr):
            return {"str"}
        if isinstance(node, ast.FormattedValue):
            return {"str"}
        if isinstance(node, ast.Call):
            return self.infer_call(node, env)
        if isinstance(node, ast.Attribute):
            return self.attribute_type(node)
        if isinstance(node, ast.Subscript):
            base = self.infer_expr(node.value, env)
            if "tensor" in base:
                return {"tensor"}
            if base & {"list", "tuple", "set", "dict"}:
                return {"unknown"}
            return {"unknown"}
        if isinstance(node, ast.BinOp):
            left = self.infer_expr(node.left, env)
            right = self.infer_expr(node.right, env)
            if "str" in left and "str" in right and isinstance(node.op, ast.Add):
                return {"str"}
            numeric = {"int", "float", "bool"}
            if left <= numeric and right <= numeric:
                return {"float"} if "float" in left or "float" in right else {"int"}
            return {"unknown"}
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return {"bool"}
            operand = self.infer_expr(node.operand, env)
            if "float" in operand:
                return {"float"}
            if "int" in operand or "bool" in operand:
                return {"int"}
            return {"unknown"}
        if isinstance(node, (ast.BoolOp, ast.Compare)):
            return {"bool"}
        if isinstance(node, ast.IfExp):
            return self.merge_types(
                self.infer_expr(node.body, env),
                self.infer_expr(node.orelse, env),
            )
        if isinstance(node, ast.Await):
            return self.infer_expr(node.value, env)
        return {"unknown"}

    def infer_call(self, node: ast.Call, env: dict[str, set[str]]) -> set[str]:
        call_name = _call_name(node.func)
        normalized = call_name.lower()
        leaf = normalized.split(".")[-1]

        if leaf in {"int", "float", "str", "bool", "bytes", "list", "tuple", "set", "dict"}:
            return {leaf}
        if leaf in {"len", "sum"}:
            return {"int"}
        if leaf == "isinstance":
            return {"bool"}
        if leaf in self.tensor_factories:
            return {"tensor"}
        if normalized.startswith(("torch.", "tensorflow.", "tf.", "jax.numpy.", "numpy.")) and leaf in self.tensor_factories:
            return {"tensor"}
        if leaf == "array" and normalized.startswith(("numpy.", "jax.numpy.", "torch.")):
            return {"tensor"}

        for arg in node.args:
            self.infer_expr(arg, env)
        for keyword in node.keywords:
            if keyword.value is not None:
                self.infer_expr(keyword.value, env)
        return {"unknown"}

    @staticmethod
    def attribute_type(node: ast.Attribute) -> set[str]:
        attr = node.attr.lower()
        if attr == "shape":
            return {"tuple"}
        if attr in {"ndim", "size"}:
            return {"int"}
        if attr == "dtype":
            return {"str"}
        return {"unknown"}

    def infer_annotation(self, node: ast.AST | None) -> set[str]:
        if node is None:
            return set()
        if isinstance(node, ast.Name):
            normalized = self.normalize_type_name(node.id)
            return {normalized} if normalized else set()
        if isinstance(node, ast.Attribute):
            normalized = self.normalize_type_name(_call_name(node))
            return {normalized} if normalized else set()
        if isinstance(node, ast.Subscript):
            return self.infer_annotation(node.value)
        if isinstance(node, ast.Tuple):
            merged: set[str] = set()
            for elt in node.elts:
                merged |= self.infer_annotation(elt)
            return merged
        if isinstance(node, ast.Constant) and node.value is None:
            return {"none"}
        return set()

    def extract_isinstance_narrowing(self, node: ast.AST | None) -> tuple[str, set[str]] | None:
        if not isinstance(node, ast.Call):
            return None
        if _call_name(node.func).lower() != "isinstance":
            return None
        if len(node.args) < 2:
            return None
        subject = node.args[0]
        if not isinstance(subject, ast.Name):
            return None
        narrowed_types = self.infer_annotation(node.args[1])
        if not narrowed_types:
            return None
        return subject.id, narrowed_types

    def apply_true_narrowing(self, env: dict[str, set[str]], symbol: str, narrowed: set[str]) -> None:
        current = set(env.get(symbol, {"unknown"}))
        if current == {"unknown"}:
            env[symbol] = set(narrowed)
            return
        overlap = current & narrowed
        env[symbol] = overlap if overlap else set(narrowed)

    def apply_false_narrowing(self, env: dict[str, set[str]], symbol: str, narrowed: set[str]) -> None:
        current = set(env.get(symbol, set()))
        if not current:
            return
        remainder = current - narrowed
        if remainder:
            env[symbol] = remainder

    def assign_target(self, env: dict[str, set[str]], target: ast.AST, types: set[str]) -> None:
        cleaned = set(types) if types else {"unknown"}
        for name in self.iter_target_names(target):
            env[name] = cleaned

    @staticmethod
    def iter_target_names(node: ast.AST) -> list[str]:
        names: list[str] = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                names.extend(_TypeAnalyzer.iter_target_names(elt))
        elif isinstance(node, ast.Starred):
            names.extend(_TypeAnalyzer.iter_target_names(node.value))
        return names

    @staticmethod
    def constant_type(value: Any) -> set[str]:
        if value is None:
            return {"none"}
        if isinstance(value, bool):
            return {"bool"}
        if isinstance(value, int):
            return {"int"}
        if isinstance(value, float):
            return {"float"}
        if isinstance(value, str):
            return {"str"}
        if isinstance(value, bytes):
            return {"bytes"}
        return {"unknown"}

    @staticmethod
    def normalize_type_name(name: str) -> str | None:
        leaf = name.split(".")[-1].strip().lower()
        if leaf in {"int", "float", "str", "bool", "bytes", "list", "tuple", "set", "dict", "callable", "type"}:
            return leaf
        if leaf in {"none", "nonetype"}:
            return "none"
        if leaf in {"tensor", "ndarray"}:
            return "tensor"
        return None

    @staticmethod
    def element_types(iter_types: set[str]) -> set[str]:
        if "tensor" in iter_types:
            return {"tensor"}
        if iter_types & {"list", "tuple", "set", "dict", "iterator"}:
            return {"unknown"}
        return {"unknown"}

    def record(self, scope: str, env: dict[str, set[str]]) -> None:
        bucket = self._results.setdefault(scope, {})
        for symbol, types in env.items():
            current = bucket.get(symbol, set())
            bucket[symbol] = self.merge_types(current, types)

    @staticmethod
    def merge_types(*items: set[str]) -> set[str]:
        merged: set[str] = set()
        for item in items:
            merged |= set(item)
        return merged or {"unknown"}

    @staticmethod
    def clone_env(env: dict[str, set[str]]) -> dict[str, set[str]]:
        return {symbol: set(types) for symbol, types in env.items()}

    @staticmethod
    def merge_envs(*envs: dict[str, set[str]]) -> dict[str, set[str]]:
        merged: dict[str, set[str]] = {}
        for env in envs:
            for symbol, types in env.items():
                merged[symbol] = _TypeAnalyzer.merge_types(merged.get(symbol, set()), types)
        return merged


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return ""
