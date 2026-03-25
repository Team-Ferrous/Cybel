"""Heuristic algorithmic complexity estimation for symbols and pipelines.

The analyzer is static and best-effort: it prioritizes robust fallback behavior over
strict precision when source, CFG, or call-graph data is incomplete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import os
import re
from typing import Any


@dataclass(slots=True)
class ComplexityEstimate:
    """Estimated computational complexity for a code region."""

    symbol: str
    file_path: str
    time_complexity: str
    space_complexity: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    loop_depth: int = 0
    has_recursion: bool = False
    dominant_operations: list[dict[str, Any]] = field(default_factory=list)
    amortized_time_complexity: str | None = None
    worst_case_time_complexity: str | None = None
    parameterized_variables: dict[str, str] = field(default_factory=dict)


class ComplexityAnalyzer:
    """Estimates algorithmic complexity from CFGs and AST patterns."""

    OPERATION_COSTS: dict[str, str] = {
        "sorted": "O(n log n)",
        "sort": "O(n log n)",
        "bisect": "O(log n)",
        "binary_search": "O(log n)",
        "dict_lookup": "O(1) amortized",
        "set_lookup": "O(1) amortized",
        "matmul": "O(n^3)",
        "conv2d": "O(n^2 * k^2)",
        "attention": "O(n^2 * d)",
    }

    def __init__(self, repo_path: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path or ".")

    def analyze_function(
        self, symbol: str, file_path: str, cfg: dict[str, Any]
    ) -> ComplexityEstimate:
        """Estimate complexity for a single function."""
        abs_path = file_path if os.path.isabs(file_path) else os.path.join(self.repo_path, file_path)
        source = self._read_text(abs_path)
        fn_node = self._find_function_node(source, symbol)
        symbol_found = fn_node is not None

        evidence: list[str] = []
        dominant: list[dict[str, Any]] = []

        cfg_nodes = list((cfg or {}).get("nodes") or [])
        cfg_edges = list((cfg or {}).get("edges") or [])
        call_graph = (cfg or {}).get("call_graph") or {}

        loop_depth = self._count_loop_nesting(cfg_nodes)
        if loop_depth == 0 and fn_node is not None:
            loop_depth = self._ast_loop_depth(fn_node)
        if loop_depth > 0:
            evidence.append(f"loop nesting depth detected: {loop_depth}")
        if not symbol_found and symbol:
            evidence.append(f"symbol_not_found: {symbol}")

        recursion_type = self._detect_recursion_type(call_graph, symbol)
        if recursion_type is None and fn_node is not None:
            recursion_type = self._detect_recursion_from_ast(fn_node, symbol)
        has_recursion = recursion_type is not None
        if has_recursion:
            evidence.append(f"recursive pattern detected: {recursion_type}")

        if fn_node is not None:
            dominant = self._analyze_data_structure_ops(fn_node)
        elif source:
            dominant = self._quick_text_ops(source)

        for op in dominant[:8]:
            op_name = str(op.get("op") or "operation")
            op_line = int(op.get("line") or 0)
            evidence.append(f"{op_name} at line {op_line}")

        time_complexity = self._estimate_time_complexity(
            loop_depth=loop_depth,
            recursion_type=recursion_type,
            dominant_ops=dominant,
            cfg_edges=cfg_edges,
            source=source,
        )
        amortized_time_complexity = self._estimate_amortized_time(
            baseline=time_complexity,
            dominant_ops=dominant,
            source=source,
        )
        worst_case_time_complexity = self._estimate_worst_case_time(
            baseline=time_complexity,
            dominant_ops=dominant,
            loop_depth=loop_depth,
            recursion_type=recursion_type,
        )
        space_complexity = self._estimate_space_complexity(
            fn_node=fn_node,
            loop_depth=loop_depth,
            dominant_ops=dominant,
            source=source,
        )
        parameterized_variables = self._parameterized_variable_mapping(
            fn_node=fn_node,
            source=source,
        )

        confidence = self._estimate_confidence(
            has_source=bool(source),
            has_cfg=bool(cfg_nodes),
            has_fn_node=fn_node is not None,
            evidence_count=len(evidence),
            loop_depth=loop_depth,
            has_recursion=has_recursion,
        )
        if not symbol_found and symbol:
            time_complexity = "unknown"
            amortized_time_complexity = "unknown"
            worst_case_time_complexity = "unknown"
            space_complexity = "unknown"
            confidence = min(confidence, 0.25)

        if not evidence:
            evidence.append("fallback heuristic used due to limited structural evidence")

        return ComplexityEstimate(
            symbol=symbol,
            file_path=abs_path,
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            confidence=confidence,
            evidence=sorted(set(evidence))[:20],
            loop_depth=loop_depth,
            has_recursion=has_recursion,
            dominant_operations=dominant[:12],
            amortized_time_complexity=amortized_time_complexity,
            worst_case_time_complexity=worst_case_time_complexity,
            parameterized_variables=parameterized_variables,
        )

    def analyze_pipeline(self, trace: Any) -> dict[str, Any]:
        """Estimate aggregate complexity for an entire pipeline."""
        stage_estimates: list[ComplexityEstimate] = []
        evidence: list[str] = []

        for stage in list(getattr(trace, "stages", []) or []):
            if isinstance(stage, dict):
                stage_complexity = stage.get("complexity")
                stage_name = str(stage.get("name") or "stage")
                stage_file = str(stage.get("file_path") or stage.get("file") or "")
            else:
                stage_complexity = getattr(stage, "complexity", None)
                stage_name = str(getattr(stage, "name", "stage"))
                stage_file = str(
                    getattr(stage, "file_path", "")
                    or getattr(stage, "file", "")
                    or ""
                )
            if isinstance(stage_complexity, dict) and (
                stage_complexity.get("time")
                or stage_complexity.get("time_complexity")
            ):
                estimate = ComplexityEstimate(
                    symbol=stage_name,
                    file_path=stage_file,
                    time_complexity=str(
                        stage_complexity.get("time")
                        or stage_complexity.get("time_complexity")
                        or "unknown"
                    ),
                    space_complexity=str(
                        stage_complexity.get("space")
                        or stage_complexity.get("space_complexity")
                        or "unknown"
                    ),
                    confidence=float(stage_complexity.get("confidence") or 0.5),
                    evidence=[str(item) for item in stage_complexity.get("evidence") or []],
                    loop_depth=int(stage_complexity.get("loop_depth") or 0),
                    has_recursion=bool(stage_complexity.get("has_recursion") or False),
                    dominant_operations=list(stage_complexity.get("dominant_operations") or []),
                    amortized_time_complexity=str(
                        stage_complexity.get("amortized_time")
                        or stage_complexity.get("amortized_time_complexity")
                        or stage_complexity.get("time")
                        or stage_complexity.get("time_complexity")
                        or "unknown"
                    ),
                    worst_case_time_complexity=str(
                        stage_complexity.get("worst_case_time")
                        or stage_complexity.get("worst_case_time_complexity")
                        or stage_complexity.get("time")
                        or stage_complexity.get("time_complexity")
                        or "unknown"
                    ),
                    parameterized_variables=dict(stage_complexity.get("parameterized_variables") or {}),
                )
            else:
                estimate = self.analyze_function(
                    symbol=stage_name,
                    file_path=stage_file,
                    cfg={},
                )
            stage_estimates.append(estimate)
            evidence.extend(estimate.evidence)

        time_terms = [item.time_complexity for item in stage_estimates]
        amortized_terms = [
            str(item.amortized_time_complexity or item.time_complexity)
            for item in stage_estimates
        ]
        worst_terms = [
            str(item.worst_case_time_complexity or item.time_complexity)
            for item in stage_estimates
        ]
        space_terms = [item.space_complexity for item in stage_estimates]
        aggregate_time = self._combine_sequential(time_terms)
        aggregate_amortized = self._combine_sequential(amortized_terms)
        aggregate_worst = self._combine_sequential(worst_terms)
        aggregate_space = self._combine_sequential(space_terms)
        parameterized_variables: dict[str, str] = {}
        for estimate in stage_estimates:
            for key, value in estimate.parameterized_variables.items():
                if key and value and key not in parameterized_variables:
                    parameterized_variables[str(key)] = str(value)

        confidence = 0.0
        if stage_estimates:
            confidence = sum(item.confidence for item in stage_estimates) / len(stage_estimates)

        return {
            "time_complexity": aggregate_time,
            "amortized_time_complexity": aggregate_amortized,
            "worst_case_time_complexity": aggregate_worst,
            "space_complexity": aggregate_space,
            "confidence": round(confidence, 3),
            "evidence": sorted(set(evidence))[:30],
            "parameterized_variables": parameterized_variables,
            "stage_estimates": [asdict(item) for item in stage_estimates],
        }

    def _estimate_amortized_time(
        self,
        *,
        baseline: str,
        dominant_ops: list[dict[str, Any]],
        source: str,
    ) -> str:
        if any(
            str(item.get("op") or "").lower() in {"dict_lookup", "set_lookup"}
            for item in dominant_ops
        ):
            return "O(1) amortized"
        if "append(" in source.lower() and "for " in source.lower():
            return "O(1) amortized"
        return str(baseline or "O(1)")

    def _estimate_worst_case_time(
        self,
        *,
        baseline: str,
        dominant_ops: list[dict[str, Any]],
        loop_depth: int,
        recursion_type: str | None,
    ) -> str:
        if recursion_type == "branching":
            return "O(2^n)"
        if any(
            str(item.get("op") or "").lower() in {"dict_lookup", "set_lookup"}
            for item in dominant_ops
        ):
            if loop_depth > 0:
                return self._loop_expr(loop_depth + 1)
            return "O(n)"
        return str(baseline or "O(1)")

    def _count_loop_nesting(self, cfg_nodes: list[dict[str, Any]]) -> int:
        """Estimate loop nesting depth from CFG node metadata."""
        if not cfg_nodes:
            return 0

        max_depth = 0
        for node in cfg_nodes:
            kind = str(node.get("type") or node.get("kind") or "").lower()
            if "loop" in kind or kind in {"for", "while", "asyncfor"}:
                depth = int(node.get("loop_depth") or 1)
                max_depth = max(max_depth, depth)

        if max_depth > 0:
            return max_depth

        # Fallback: infer from parent depth hints.
        parent_depth = 0
        for node in cfg_nodes:
            parent_depth = max(parent_depth, int(node.get("depth") or 0))
        return max(0, parent_depth)

    def _detect_recursion_type(self, call_graph: dict[str, Any], symbol: str) -> str | None:
        """Detect recursion kind from call graph, if available."""
        if not call_graph or not symbol:
            return None

        callees = []
        if isinstance(call_graph, dict):
            raw = call_graph.get(symbol)
            if isinstance(raw, (list, tuple, set)):
                callees = [str(item) for item in raw]

        self_calls = [callee for callee in callees if callee == symbol]
        if len(self_calls) >= 2:
            return "branching"
        if len(self_calls) == 1:
            return "linear"
        return None

    def _analyze_data_structure_ops(self, ast_node: Any) -> list[dict[str, Any]]:
        """Extract dominant operation hints from AST."""
        if ast_node is None:
            return []

        ops: list[dict[str, Any]] = []

        for node in ast.walk(ast_node):
            if isinstance(node, ast.Call):
                call_name = self._call_name(node)
                lower = call_name.lower()
                if lower in {"sorted", "sort"}:
                    ops.append({"op": "sort", "complexity": "O(n log n)", "line": getattr(node, "lineno", 0)})
                elif "bisect" in lower or "binary_search" in lower:
                    ops.append({"op": "binary_search", "complexity": "O(log n)", "line": getattr(node, "lineno", 0)})
                elif lower in {"get", "setdefault"}:
                    ops.append({"op": "dict_lookup", "complexity": "O(1) amortized", "line": getattr(node, "lineno", 0)})
                elif lower in {"matmul", "mm", "bmm", "dot", "einsum"}:
                    ops.append({"op": "matmul", "complexity": "O(n^3)", "line": getattr(node, "lineno", 0)})
                elif "conv" in lower:
                    ops.append({"op": "conv2d", "complexity": "O(n^2 * k^2)", "line": getattr(node, "lineno", 0)})
                elif "attention" in lower:
                    ops.append({"op": "attention", "complexity": "O(n^2 * d)", "line": getattr(node, "lineno", 0)})

            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                ops.append({"op": "matmul", "complexity": "O(n^3)", "line": getattr(node, "lineno", 0)})

        return ops

    def _combine_sequential(self, complexities: list[str]) -> str:
        """Combine sequential complexities by dominant term."""
        clean = [str(item or "O(1)") for item in complexities if str(item or "").strip()]
        if not clean:
            return "O(1)"
        return max(clean, key=self._complexity_rank)

    def _combine_nested(self, outer: str, inner: str) -> str:
        """Combine nested complexities with a simple symbolic multiplication."""
        o = str(outer or "O(1)")
        i = str(inner or "O(1)")
        if o == "O(1)":
            return i
        if i == "O(1)":
            return o

        # Handle common polynomial forms O(n^a) * O(n^b) = O(n^(a+b)).
        o_pow = self._extract_power(o)
        i_pow = self._extract_power(i)
        if o_pow is not None and i_pow is not None:
            total = o_pow + i_pow
            return "O(n)" if total == 1 else f"O(n^{total})"

        if "log n" in o.lower() and "n" in i.lower():
            return "O(n log n)"
        if "log n" in i.lower() and "n" in o.lower():
            return "O(n log n)"

        return max([o, i], key=self._complexity_rank)

    def _estimate_time_complexity(
        self,
        *,
        loop_depth: int,
        recursion_type: str | None,
        dominant_ops: list[dict[str, Any]],
        cfg_edges: list[dict[str, Any]],
        source: str,
    ) -> str:
        if recursion_type == "branching":
            return "O(2^n)"
        if recursion_type == "linear":
            if loop_depth > 0:
                return self._combine_nested(self._loop_expr(loop_depth), "O(n)")
            return "O(n)"

        op_terms = [str(item.get("complexity") or "") for item in dominant_ops]
        if any("n log n" in term.lower() for term in op_terms):
            if loop_depth > 0:
                return self._combine_nested(self._loop_expr(loop_depth), "O(n log n)")
            return "O(n log n)"

        if self._looks_like_binary_search(source, cfg_edges):
            return "O(log n)"

        if loop_depth > 0:
            return self._loop_expr(loop_depth)

        if op_terms:
            return max(op_terms, key=self._complexity_rank)
        return "O(1)"

    def _estimate_space_complexity(
        self,
        *,
        fn_node: ast.AST | None,
        loop_depth: int,
        dominant_ops: list[dict[str, Any]],
        source: str,
    ) -> str:
        alloc_count = 0
        if fn_node is not None:
            for node in ast.walk(fn_node):
                if isinstance(node, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)):
                    alloc_count += 1
                if isinstance(node, ast.Call):
                    name = self._call_name(node).lower()
                    if name in {"list", "dict", "set", "tuple", "zeros", "ones", "empty", "array", "tensor"}:
                        alloc_count += 1

        if alloc_count >= 4 or "append(" in source:
            return "O(n)"

        if any(str(item.get("op") or "").lower() in {"matmul", "attention", "conv2d"} for item in dominant_ops):
            return "O(n * d)"

        if loop_depth >= 2 and ("cache" in source.lower() or "memo" in source.lower()):
            return "O(n^2)"

        return "O(1)"

    @staticmethod
    def _estimate_confidence(
        *,
        has_source: bool,
        has_cfg: bool,
        has_fn_node: bool,
        evidence_count: int,
        loop_depth: int,
        has_recursion: bool,
    ) -> float:
        score = 0.25
        if has_source:
            score += 0.25
        if has_cfg:
            score += 0.15
        if has_fn_node:
            score += 0.15
        score += min(0.15, evidence_count * 0.02)
        if loop_depth > 0:
            score += 0.05
        if has_recursion:
            score += 0.05
        return round(min(1.0, max(0.05, score)), 3)

    @staticmethod
    def _loop_expr(depth: int) -> str:
        if depth <= 0:
            return "O(1)"
        if depth == 1:
            return "O(n)"
        return f"O(n^{depth})"

    @staticmethod
    def _extract_power(expr: str) -> int | None:
        text = (expr or "").replace(" ", "").lower()
        if text == "o(n)":
            return 1
        match = re.search(r"o\(n\^([0-9]+)\)", text)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _complexity_rank(expr: str) -> int:
        text = (expr or "").replace(" ", "").lower()
        if "2^n" in text or "exp" in text:
            return 100
        match = re.search(r"n\^([0-9]+)", text)
        if match:
            return 20 + int(match.group(1))
        if "nlogn" in text:
            return 15
        if "n*d" in text:
            return 12
        if "n" in text and "log" not in text:
            return 10
        if "logn" in text or "log" in text:
            return 5
        if "1" in text:
            return 1
        return 0

    @staticmethod
    def _looks_like_binary_search(source: str, cfg_edges: list[dict[str, Any]]) -> bool:
        text = (source or "").lower()
        if "mid" in text and ("left" in text or "right" in text):
            if "while" in text or "for" in text:
                return True
        for edge in cfg_edges:
            rel = str(edge.get("relation") or "").lower()
            if rel in {"halves", "divide", "bisect"}:
                return True
        return False

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        target = node.func
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
        return ""

    @staticmethod
    def _read_text(path: str) -> str:
        if not path:
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        except Exception:
            return ""

    def _find_function_node(self, source: str, symbol: str) -> ast.AST | None:
        if not source:
            return None
        try:
            tree = ast.parse(source)
        except Exception:
            return None

        symbol = (symbol or "").strip()
        class_hint = None
        fn_hint = symbol
        if "." in symbol:
            class_hint, _, fn_hint = symbol.rpartition(".")

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_hint:
                return node
            if isinstance(node, ast.ClassDef):
                if class_hint and node.name != class_hint:
                    continue
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == fn_hint:
                        return child

        if symbol and str(symbol).strip():
            return None

        # Fallback: first function-like node for anonymous analysis.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node
        return None

    @staticmethod
    def _ast_loop_depth(fn_node: ast.AST) -> int:
        max_depth = 0

        def walk(node: ast.AST, depth: int) -> None:
            nonlocal max_depth
            is_loop = isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.ListComp, ast.DictComp, ast.SetComp))
            next_depth = depth + 1 if is_loop else depth
            max_depth = max(max_depth, next_depth)
            for child in ast.iter_child_nodes(node):
                walk(child, next_depth)

        walk(fn_node, 0)
        return max_depth

    @staticmethod
    def _detect_recursion_from_ast(fn_node: ast.AST, symbol: str) -> str | None:
        fn_name = (symbol or "").split(".")[-1]
        calls = 0
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == fn_name:
                        calls += 1
                elif isinstance(node.func, ast.Attribute):
                    # Count method recursion only when self.<fn_name>(...) is invoked.
                    if (
                        node.func.attr == fn_name
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                    ):
                        calls += 1
        if calls >= 2:
            return "branching"
        if calls == 1:
            return "linear"
        return None

    @staticmethod
    def _quick_text_ops(source: str) -> list[dict[str, Any]]:
        text = (source or "").lower()
        ops: list[dict[str, Any]] = []
        if "sorted(" in text or ".sort(" in text:
            ops.append({"op": "sort", "complexity": "O(n log n)", "line": 0})
        if "matmul" in text or "@" in text:
            ops.append({"op": "matmul", "complexity": "O(n^3)", "line": 0})
        if "attention" in text:
            ops.append({"op": "attention", "complexity": "O(n^2 * d)", "line": 0})
        return ops

    @staticmethod
    def _parameterized_variable_mapping(
        *,
        fn_node: ast.AST | None,
        source: str,
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if fn_node is not None and isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [str(arg.arg) for arg in fn_node.args.args if str(arg.arg)]
            if params:
                first = params[0]
                mapping["n"] = f"len({first})"
            if len(params) >= 2:
                second = params[1]
                mapping["m"] = f"len({second})"
            for param in params:
                low = param.lower()
                if low in {"seq_len", "length", "tokens", "timesteps"}:
                    mapping.setdefault("t", param)
                if low in {"dim", "hidden", "d_model", "width"}:
                    mapping.setdefault("d", param)
                if low in {"batch", "batch_size", "b"}:
                    mapping.setdefault("b", param)
        text = (source or "").lower()
        if "head" in text:
            mapping.setdefault("h", "num_heads")
        if "channel" in text:
            mapping.setdefault("c", "channels")
        return mapping


__all__ = ["ComplexityAnalyzer", "ComplexityEstimate"]
