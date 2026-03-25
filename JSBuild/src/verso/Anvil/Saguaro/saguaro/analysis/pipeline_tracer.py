"""Pipeline reconstruction over typed code-graph relations.

This module provides static, heuristic pipeline tracing for graph payloads that
follow the proposed CodeGraph contract:

- ``graph["nodes"]``: mapping of node_id -> node payload
- ``graph["edges"]``: mapping of edge_id -> edge payload
- edges include a typed ``relation`` field

The implementation is intentionally robust to partially-populated graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any


@dataclass(slots=True)
class PipelineStage:
    """A stage in a traced pipeline."""

    id: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    function_signature: str
    description: str
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    complexity: dict[str, Any] | None = None
    children: list[str] = field(default_factory=list)
    calls_ffi: bool = False
    stage_index: int = 0
    annotations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineTrace:
    """A complete pipeline trace."""

    name: str
    entry_point: str
    stages: list[PipelineStage] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    total_complexity: dict[str, Any] | None = None
    languages_involved: list[str] = field(default_factory=list)
    ffi_boundaries: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PipelineTracer:
    """Reconstruct execution-oriented pipelines from a typed code graph."""

    _CALL_RELATIONS = {
        "call",
        "calls",
        "invokes",
        "invoke",
        "dispatches",
        "executes",
        "runs",
        "applies",
        "imports",  # fallback: treat file import chain as coarse pipeline flow
    }
    _DATA_RELATIONS = {
        "data",
        "data_flow",
        "passes",
        "transforms",
        "returns",
        "reads",
        "writes",
        "yields",
        "next",
        "true",
        "false",
        "back",
        "branch",
        "conditional",
        "condition",
        "loop",
    }
    _EXTERNAL_PREFIX = "external::"

    def __init__(self, repo_path: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path or ".")

    def trace(
        self,
        entry_point: str,
        code_graph: dict[str, Any],
        *,
        max_depth: int = 50,
        include_stdlib: bool = False,
        include_complexity: bool = True,
    ) -> PipelineTrace:
        """Trace a complete pipeline from an entry point."""
        graph = code_graph.get("graph") if isinstance(code_graph, dict) else None
        payload = graph if isinstance(graph, dict) else code_graph
        node_map = dict(payload.get("nodes") or {})

        trace = PipelineTrace(name=f"Pipeline<{entry_point}>", entry_point=entry_point)
        if not node_map:
            trace.warnings.append("Code graph has no nodes; returning empty pipeline trace.")
            return trace

        start_node = self._resolve_entry_node(entry_point=entry_point, nodes=node_map)
        if not start_node:
            trace.warnings.append(
                f"Entry point '{entry_point}' was not found in graph nodes; returning empty trace."
            )
            return trace

        stages, flow_edges, ffi_boundaries = self._forward_walk(
            start_node=start_node,
            graph=payload,
            max_depth=max_depth,
            include_stdlib=include_stdlib,
        )

        self._propagate_data_shapes(stages=stages, dfg={"edges": flow_edges})
        if include_complexity:
            self._annotate_complexity(stages=stages)

        for stage in stages:
            stage.children = [
                edge.get("to", "")
                for edge in flow_edges
                if edge.get("from") == stage.id and edge.get("to")
            ]
            stage.calls_ffi = any(
                boundary.get("host_stage_id") == stage.id for boundary in ffi_boundaries
            )
            outgoing = [edge for edge in flow_edges if edge.get("from") == stage.id]
            incoming = [edge for edge in flow_edges if edge.get("to") == stage.id]
            self._annotate_stage_control_flow(stage=stage, outgoing=outgoing, incoming=incoming)

        trace.stages = stages
        trace.edges = flow_edges
        trace.ffi_boundaries = ffi_boundaries
        trace.languages_involved = sorted({s.language for s in stages if s.language})
        trace.total_complexity = self._aggregate_complexity(stages)
        return trace

    def trace_by_query(
        self,
        query: str,
        code_graph: dict[str, Any],
        vector_store: Any,
    ) -> list[PipelineTrace]:
        """Find and trace pipelines matching a semantic query."""
        graph = code_graph.get("graph") if isinstance(code_graph, dict) else None
        payload = graph if isinstance(graph, dict) else code_graph
        nodes = dict(payload.get("nodes") or {})
        q = (query or "").strip().lower()
        if not q or not nodes:
            return []

        candidates: list[str] = []
        if vector_store is not None and hasattr(vector_store, "query"):
            try:
                raw = vector_store.query(q, k=8)
            except Exception:
                raw = []
            for item in raw or []:
                for key in ("node_id", "entity_id", "qualified_name", "name"):
                    value = str((item or {}).get(key) or "").strip()
                    if value:
                        candidates.append(value)

        terms = [term for term in re.findall(r"[a-z][a-z0-9_]{2,}", q) if term]
        if not candidates:
            for node_id, node in nodes.items():
                haystack = " ".join(
                    [
                        node_id,
                        str(node.get("name") or ""),
                        str(node.get("qualified_name") or ""),
                        str(node.get("file") or ""),
                    ]
                ).lower()
                score = sum(1 for t in terms if t in haystack)
                if score > 0:
                    candidates.append(node_id)

        traces: list[PipelineTrace] = []
        seen_entry: set[str] = set()
        for candidate in candidates:
            entry = self._resolve_entry_node(candidate, nodes)
            if not entry or entry in seen_entry:
                continue
            seen_entry.add(entry)
            traces.append(self.trace(entry, payload))
            if len(traces) >= 5:
                break
        return traces

    def _forward_walk(
        self,
        start_node: str,
        graph: dict[str, Any],
        *,
        max_depth: int,
        include_stdlib: bool,
    ) -> tuple[list[PipelineStage], list[dict[str, Any]], list[dict[str, Any]]]:
        """Walk call/data relations forward from an entry node."""
        nodes = dict(graph.get("nodes") or {})
        edges = list((graph.get("edges") or {}).values())
        by_src: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            src = str(edge.get("from") or "")
            by_src.setdefault(src, []).append(edge)

        ordered_ids: list[str] = []
        flow_edges: list[dict[str, Any]] = []
        ffi_boundaries: list[dict[str, Any]] = []

        queue: list[tuple[str, int]] = [(start_node, 0)]
        seen_depth: dict[str, int] = {start_node: 0}

        while queue:
            node_id, depth = queue.pop(0)
            if node_id not in ordered_ids:
                ordered_ids.append(node_id)
            if depth >= max_depth:
                continue

            for edge in by_src.get(node_id, []):
                relation = str(edge.get("relation") or "related").lower()
                if not self._is_flow_relation(relation):
                    continue

                dst = str(edge.get("to") or "")
                if not dst:
                    continue

                dst_node = nodes.get(dst, {})
                if not include_stdlib and self._is_external_node(dst, dst_node):
                    ffi = self._ffi_boundary(node_id, dst, edge, nodes)
                    if ffi:
                        ffi_boundaries.append(ffi)
                    continue

                edge_data = str(edge.get("data") or edge.get("raw") or relation)
                flow_edges.append(
                    {
                        "from": node_id,
                        "to": dst,
                        "data": edge_data,
                        "relation": relation,
                        **self._edge_metadata(edge=edge, relation=relation, data=edge_data),
                    }
                )

                old_depth = seen_depth.get(dst)
                if old_depth is None or depth + 1 < old_depth:
                    seen_depth[dst] = depth + 1
                    queue.append((dst, depth + 1))

                ffi = self._ffi_boundary(node_id, dst, edge, nodes)
                if ffi:
                    ffi_boundaries.append(ffi)

        stages: list[PipelineStage] = []
        for idx, node_id in enumerate(ordered_ids):
            node = dict(nodes.get(node_id) or {})
            stage = self._stage_from_node(node_id=node_id, node=node, stage_index=idx)
            stages.append(stage)

        id_set = {stage.id for stage in stages}
        filtered_edges = [e for e in flow_edges if e.get("from") in id_set and e.get("to") in id_set]
        dedup = {
            (
                e["from"],
                e["to"],
                e["relation"],
                str(e.get("condition") or ""),
                str(e.get("label") or ""),
            ): e
            for e in filtered_edges
        }
        dedup_ffi = {
            (
                item.get("host_stage_id"),
                item.get("guest_stage_id"),
                item.get("mechanism"),
            ): item
            for item in ffi_boundaries
        }
        return stages, list(dedup.values()), list(dedup_ffi.values())

    def _propagate_data_shapes(self, stages: list[PipelineStage], dfg: dict[str, Any]) -> None:
        """Apply conservative shape propagation over traced stages."""
        if not stages:
            return
        try:
            from saguaro.analysis.shape_propagator import ShapePropagator

            trace = PipelineTrace(name="shape-pass", entry_point="", stages=stages, edges=dfg.get("edges", []))
            ShapePropagator().propagate(trace=trace, source_files={})
        except Exception:
            # Keep trace generation resilient even when shape inference fails.
            return

    @staticmethod
    def _detect_pipeline_boundaries(call_chain: list[str]) -> list[list[str]]:
        """Split a call chain into boundary-delimited segments.

        Heuristic: a boundary appears when a pseudo-node contains ``external::``.
        """
        if not call_chain:
            return []
        segments: list[list[str]] = [[]]
        for node_id in call_chain:
            if node_id.startswith("external::") and segments[-1]:
                segments.append([])
                continue
            segments[-1].append(node_id)
        return [seg for seg in segments if seg]

    def _annotate_from_docstrings(self, stage: PipelineStage) -> None:
        """Populate description from available text fields if currently blank."""
        if stage.description:
            return
        signature = stage.function_signature or stage.name
        stage.description = f"Stage derived from static call graph for `{signature}`."

    def _annotate_complexity(self, stages: list[PipelineStage]) -> None:
        """Attach per-stage complexity estimates when source context exists."""
        try:
            from saguaro.analysis.complexity_analyzer import ComplexityAnalyzer
        except Exception:
            return

        analyzer = ComplexityAnalyzer(repo_path=self.repo_path)
        for stage in stages:
            if not stage.file_path:
                continue
            abs_path = (
                stage.file_path
                if os.path.isabs(stage.file_path)
                else os.path.join(self.repo_path, stage.file_path)
            )
            estimate = analyzer.analyze_function(
                symbol=stage.name,
                file_path=abs_path,
                cfg={},
            )
            stage.complexity = {
                "time": estimate.time_complexity,
                "space": estimate.space_complexity,
                "confidence": estimate.confidence,
                "evidence": list(estimate.evidence),
                "loop_depth": int(estimate.loop_depth),
                "has_recursion": bool(estimate.has_recursion),
                "dominant_operations": list(estimate.dominant_operations),
                "amortized_time": str(estimate.amortized_time_complexity or estimate.time_complexity),
                "worst_case_time": str(estimate.worst_case_time_complexity or estimate.time_complexity),
                "parameterized_variables": dict(estimate.parameterized_variables),
            }

    @staticmethod
    def _aggregate_complexity(stages: list[PipelineStage]) -> dict[str, Any] | None:
        """Aggregate stage complexity into a trace-level summary."""
        if not stages:
            return None
        known = [s.complexity for s in stages if isinstance(s.complexity, dict)]
        if not known:
            return None

        best_time = "O(1)"
        best_space = "O(1)"
        best_amortized = "O(1)"
        best_worst = "O(1)"
        confidence_sum = 0.0
        confidence_count = 0
        evidence: list[str] = []
        parameterized: dict[str, str] = {}

        for item in known:
            time_expr = str(item.get("time") or "O(1)")
            space_expr = str(item.get("space") or "O(1)")
            amortized_expr = str(item.get("amortized_time") or time_expr)
            worst_expr = str(item.get("worst_case_time") or time_expr)
            if PipelineTracer._complexity_rank(time_expr) > PipelineTracer._complexity_rank(best_time):
                best_time = time_expr
            if PipelineTracer._complexity_rank(space_expr) > PipelineTracer._complexity_rank(best_space):
                best_space = space_expr
            if PipelineTracer._complexity_rank(amortized_expr) > PipelineTracer._complexity_rank(best_amortized):
                best_amortized = amortized_expr
            if PipelineTracer._complexity_rank(worst_expr) > PipelineTracer._complexity_rank(best_worst):
                best_worst = worst_expr
            conf = item.get("confidence")
            if isinstance(conf, (int, float)):
                confidence_sum += float(conf)
                confidence_count += 1
            for line in item.get("evidence") or []:
                text = str(line).strip()
                if text:
                    evidence.append(text)
            raw_vars = item.get("parameterized_variables")
            if isinstance(raw_vars, dict):
                for key, value in raw_vars.items():
                    var = str(key or "").strip()
                    mapping = str(value or "").strip()
                    if var and mapping and var not in parameterized:
                        parameterized[var] = mapping

        return {
            "time": best_time,
            "space": best_space,
            "amortized_time": best_amortized,
            "worst_case_time": best_worst,
            "confidence": round(confidence_sum / max(1, confidence_count), 3),
            "evidence": sorted(set(evidence))[:20],
            "parameterized_variables": parameterized,
        }

    @staticmethod
    def _complexity_rank(expr: str) -> int:
        text = str(expr or "").replace(" ", "").lower()
        if "2^n" in text or "exp" in text:
            return 100
        match = re.search(r"n\^([0-9]+)", text)
        if match:
            return 20 + int(match.group(1))
        if "nlogn" in text:
            return 15
        if "n" in text and "log" not in text:
            return 10
        if "log" in text:
            return 5
        return 1

    def _resolve_entry_node(self, entry_point: str, nodes: dict[str, dict[str, Any]]) -> str | None:
        """Resolve user-provided entry point to a graph node id."""
        needle = (entry_point or "").strip()
        if not needle:
            return None
        if needle in nodes:
            return needle

        exact_candidates: list[str] = []
        fuzzy_candidates: list[str] = []
        for node_id, node in nodes.items():
            name = str(node.get("name") or "")
            qualified = str(node.get("qualified_name") or "")
            entity_id = str(node.get("entity_id") or "")
            file_path = str(node.get("file") or "")
            joined = " ".join([node_id, name, qualified, entity_id, file_path])
            if needle == name or needle == qualified or needle == entity_id:
                exact_candidates.append(node_id)
            elif needle.lower() in joined.lower():
                fuzzy_candidates.append(node_id)

            if ":" in needle and file_path:
                file_hint, _, symbol_hint = needle.partition(":")
                if file_hint and file_hint in file_path and symbol_hint in {name, qualified}:
                    exact_candidates.append(node_id)

        if exact_candidates:
            return sorted(exact_candidates)[0]
        if fuzzy_candidates:
            fuzzy_candidates.sort(key=len)
            return fuzzy_candidates[0]
        return None

    def _stage_from_node(self, node_id: str, node: dict[str, Any], stage_index: int) -> PipelineStage:
        """Convert a graph node payload to a PipelineStage."""
        file_path = str(node.get("file") or "")
        language = self._language_for_file(file_path)
        name = str(node.get("qualified_name") or node.get("name") or node_id)
        start = int(node.get("line") or 0)
        end = int(node.get("end_line") or start)
        signature = f"{name}()" if name else node_id
        description = str(node.get("docstring") or "").strip()

        stage = PipelineStage(
            id=node_id,
            name=name or node_id,
            file_path=file_path,
            start_line=start,
            end_line=max(start, end),
            language=language,
            function_signature=signature,
            description=description,
            stage_index=stage_index,
        )
        self._annotate_from_docstrings(stage)
        return stage

    @staticmethod
    def _language_for_file(file_path: str) -> str:
        lower = file_path.lower()
        if lower.endswith(".py"):
            return "python"
        if lower.endswith((".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh")):
            return "cpp"
        if lower.endswith((".rs",)):
            return "rust"
        if lower.endswith((".js", ".jsx", ".mjs", ".cjs")):
            return "javascript"
        if lower.endswith((".ts", ".tsx", ".mts", ".cts")):
            return "typescript"
        if lower.endswith(".go"):
            return "go"
        if not file_path:
            return "external"
        return "unknown"

    def _ffi_boundary(
        self,
        src_id: str,
        dst_id: str,
        edge: dict[str, Any],
        nodes: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Create an FFI boundary record when an edge crosses language/runtime."""
        src_node = nodes.get(src_id, {})
        dst_node = nodes.get(dst_id, {})
        src_lang = self._language_for_file(str(src_node.get("file") or ""))
        dst_lang = self._language_for_file(str(dst_node.get("file") or ""))
        relation = str(edge.get("relation") or "related")

        external = self._is_external_node(dst_id, dst_node)
        cross_lang = src_lang != dst_lang and dst_lang != "unknown"
        if not external and not cross_lang:
            return None

        return {
            "host_stage_id": src_id,
            "guest_stage_id": dst_id,
            "host": {
                "file": src_node.get("file"),
                "name": src_node.get("qualified_name") or src_node.get("name") or src_id,
                "language": src_lang,
            },
            "guest": {
                "file": dst_node.get("file"),
                "name": dst_node.get("qualified_name") or dst_node.get("name") or dst_id,
                "language": dst_lang,
            },
            "mechanism": "external" if external else relation,
            "confidence": 0.85 if external else 0.65,
            "evidence": [
                f"edge relation={relation}",
                f"language transition {src_lang} -> {dst_lang}",
            ],
        }

    def _is_external_node(self, node_id: str, node: dict[str, Any]) -> bool:
        if node_id.startswith(self._EXTERNAL_PREFIX):
            return True
        node_type = str(node.get("type") or "")
        return node_type == "external"

    def _annotate_stage_control_flow(
        self,
        *,
        stage: PipelineStage,
        outgoing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> None:
        tags: set[str] = set(stage.annotations or [])
        if any(bool(edge.get("conditional")) for edge in outgoing):
            tags.add("branch_stage")
        if any(bool(edge.get("loop")) for edge in outgoing):
            tags.add("loop_stage")
        if any(bool(edge.get("loop")) for edge in incoming):
            tags.add("loop_target")
        stage.annotations = sorted(tags)

    @classmethod
    def _is_flow_relation(cls, relation: str) -> bool:
        rel = str(relation or "").strip().lower()
        if not rel:
            return False
        if rel in cls._CALL_RELATIONS or rel in cls._DATA_RELATIONS:
            return True
        return rel.startswith(("cfg_", "dfg_", "call_")) or any(
            token in rel for token in ("branch", "condition", "loop", "back", "true", "false")
        )

    def _edge_metadata(self, *, edge: dict[str, Any], relation: str, data: str) -> dict[str, Any]:
        label = self._edge_label(edge=edge, relation=relation, data=data)
        condition = self._extract_condition(label)
        rel = str(relation or "").lower()
        label_low = label.lower()
        conditional = rel in {"true", "false", "branch", "conditional", "condition"} or (
            any(tag in rel for tag in ("if", "cond", "switch", "case"))
            or any(tag in label_low for tag in ("if ", "when ", "else", "true", "false", "cond"))
        )
        loop = rel in {"back", "loop"} or any(
            tag in rel or tag in label_low for tag in ("loop", "back", "iterate", "repeat")
        )
        return {
            "label": label,
            "conditional": bool(conditional),
            "condition": condition,
            "loop": bool(loop),
        }

    @staticmethod
    def _edge_label(edge: dict[str, Any], relation: str, data: str) -> str:
        explicit = str(edge.get("label") or edge.get("condition") or "").strip()
        if explicit:
            return explicit
        text = str(data or "").strip()
        if text and text != relation:
            return text[:120]
        return str(relation or "flow")

    @staticmethod
    def _extract_condition(label: str) -> str | None:
        text = str(label or "").strip()
        if not text:
            return None
        low = text.lower()
        if low in {"true", "false"}:
            return low
        match = re.search(r"\bif\s+(.+)", text, re.IGNORECASE)
        if match:
            return str(match.group(1) or "").strip()[:120] or None
        if ":" in text and any(tag in low for tag in ("cond", "case", "branch")):
            return text.split(":", 1)[1].strip()[:120] or None
        return None
