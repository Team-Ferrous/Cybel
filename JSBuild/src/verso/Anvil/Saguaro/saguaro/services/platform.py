"""Platform services for graph, query, evidence, research, and app surfaces."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from saguaro.analysis.bridge_synthesizer import BridgeSynthesizer
from saguaro.analysis.call_graph_builder import CallGraphBuilder
from saguaro.analysis.cfg_builder import CFGBuilder
from saguaro.analysis.dfg_builder import DFGBuilder
from saguaro.analysis.disparate_relations import DisparateRelationSynthesizer
from saguaro.analysis.ffi_scanner import FFIScanner
from saguaro.coverage import CoverageReporter
from saguaro.parsing.parser import CodeEntity, SAGUAROParser
from saguaro.query.benchmark import (
    derive_query_calibration,
    load_benchmark_cases,
    persist_query_calibration,
    score_benchmark_results,
)
from saguaro.query.corpus_rules import (
    canonicalize_rel_path,
    classify_file_role,
    filter_indexable_files,
    is_excluded_path,
)
from saguaro.query.pipeline import QueryPipeline
from saguaro.storage.atomic_fs import atomic_write_json
from saguaro.storage.index_state import load_manifest, manifest_path, snapshot_artifact
from saguaro.storage.vector_store import VectorStore
from saguaro.utils.entity_ids import entity_identity
from saguaro.utils.file_utils import get_code_files

try:
    from core.aes.governance import GovernanceEngine
except ModuleNotFoundError:  # pragma: no cover - CLI/package fallback
    class GovernanceEngine:  # type: ignore[override]
        """Minimal fallback governance tier helper for standalone Saguaro entrypoints."""

        @staticmethod
        def get_tier(aal: str, action_type: str) -> Any:
            """Get tier."""
            class _Tier:
                value = "advisory"

            return _Tier()


def _safe_relpath(path: str, root: str) -> str:
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return rel.replace("\\", "/")


def _normalize_domains(domain: str | list[str] | None) -> list[str]:
    if domain is None:
        return ["universal"]
    if isinstance(domain, str):
        values = [item.strip() for item in domain.split(",") if item.strip()]
        return values or ["universal"]
    return [str(item).strip() for item in domain if str(item).strip()] or ["universal"]


def _expanded_search_terms(text: str) -> set[str]:
    raw = str(text or "")
    if not raw:
        return set()
    terms = {
        token
        for token in re.findall(r"[a-z][a-z0-9_]{2,}", raw.lower())
        if len(token) >= 3
    }
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw)
    expanded = re.sub(r"[/_.:-]+", " ", expanded)
    for token in re.findall(r"[a-z][a-z0-9_]{1,}", expanded.lower()):
        if len(token) >= 3:
            terms.add(token)
        for part in re.split(r"[_\-.]+", token):
            if len(part) >= 3:
                terms.add(part)
    return terms


def _aes_envelope(
    *,
    action_type: str,
    aal: str = "AAL-3",
    domain: str | list[str] | None = None,
    repo_role: str = "analysis_local",
    evidence_links: list[str] | None = None,
    promotable: bool = False,
) -> dict[str, Any]:
    engine = GovernanceEngine()
    normalized_aal = str(aal or "AAL-3").upper()
    normalized_domains = _normalize_domains(domain)
    return {
        "aal": normalized_aal,
        "domain": normalized_domains,
        "action_type": action_type,
        "repo_role": repo_role,
        "governance_tier": engine.get_tier(normalized_aal, action_type).value,
        "evidence_links": list(evidence_links or []),
        "promotable": bool(promotable),
    }


class ParseService:
    """Parser and coverage facade."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.parser = SAGUAROParser()
        self.coverage = CoverageReporter(self.repo_path)

    def parse_file(self, file_path: str) -> list[CodeEntity]:
        """Handle parse file."""
        return self.parser.parse_file(file_path)

    def coverage_report(self) -> dict[str, Any]:
        """Handle coverage report."""
        return self.coverage.generate_report()


class GraphService:
    """Build and query a local repository graph."""

    GRAPH_SCHEMA_VERSION = 4

    def __init__(
        self,
        repo_path: str,
        parse_service: ParseService,
        *,
        saguaro_dir: str | None = None,
    ) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.abspath(
            saguaro_dir or os.path.join(self.repo_path, ".saguaro")
        )
        self.graph_dir = os.path.join(self.saguaro_dir, "graph")
        self.graph_path = os.path.join(self.graph_dir, "graph.json")
        self.code_graph_path = os.path.join(self.graph_dir, "code_graph.json")
        self.parse_service = parse_service
        self.cfg_builder = CFGBuilder()
        self.dfg_builder = DFGBuilder()
        self.call_graph_builder = CallGraphBuilder()
        self.ffi_scanner = FFIScanner(repo_path=self.repo_path)
        self.bridge_synthesizer = BridgeSynthesizer()
        self.disparate_relation_synthesizer = DisparateRelationSynthesizer()
        self._graph_cache_lock = threading.Lock()
        self._graph_cache: tuple[float, dict[str, Any]] | None = None
        self._graph_write_lock = threading.Lock()

    def build(
        self,
        path: str = ".",
        incremental: bool = True,
        changed_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle build."""
        os.makedirs(self.graph_dir, exist_ok=True)
        target_path = os.path.abspath(
            path if os.path.isabs(path) else os.path.join(self.repo_path, path)
        )
        explicit_files, deleted_files = self._resolve_changed_files(changed_files)
        full_rebuild = not incremental or explicit_files is None
        all_files = (
            self._discover_files(target_path)
            if full_rebuild
            else sorted(explicit_files)
        )

        graph = self._empty_graph()
        if not full_rebuild and os.path.exists(self.graph_path):
            graph = self.load_graph()
            drop_set = {_safe_relpath(p, self.repo_path) for p in all_files}
            drop_set.update({_safe_relpath(p, self.repo_path) for p in deleted_files})
            graph = self._drop_files(graph, drop_set)

        module_index = self._module_index(self._discover_files(self.repo_path))
        parsed_files = 0
        for file_path in all_files:
            rel_file = _safe_relpath(file_path, self.repo_path)
            entities = self.parse_service.parse_file(file_path)
            if not entities:
                continue
            parsed_files += 1
            self._ingest_file(graph, rel_file, entities, module_index)
        self._rebuild_bridge_edges(graph)
        self._rebuild_indices(graph)

        coverage = self.parse_service.coverage_report()
        graph["generated_at"] = time.time()
        graph["generated_fmt"] = time.ctime(graph["generated_at"])
        self._refresh_stats(
            graph,
            parsed_files=parsed_files if full_rebuild else len(graph.get("files", {})),
            parser_coverage_percent=float(coverage.get("coverage_percent", 0.0)),
            total_files=int(coverage.get("total_files", 0) or 0),
            incremental=bool(not full_rebuild),
        )

        # Persist the normalized graph shape so later readers do not rewrite the
        # committed artifact and invalidate the index manifest.
        graph, _changed = self._normalize_graph(graph)
        self._write_graph(graph)

        return {
            "status": "ok",
            "graph_path": self.graph_path,
            "files": graph["stats"]["files"],
            "nodes": graph["stats"]["nodes"],
            "edges": graph["stats"]["edges"],
            "graph_coverage_percent": graph["stats"]["graph_coverage_percent"],
            "incremental": graph["stats"]["incremental"],
        }

    def query(
        self,
        symbol: str | None = None,
        file: str | None = None,
        relation: str | None = None,
        depth: int = 1,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Handle query."""
        graph = self.load_graph()
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})
        files = graph.get("files", {})

        if not nodes:
            return {
                "status": "empty",
                "nodes": [],
                "edges": [],
                "count": 0,
                "graph_path": self.graph_path,
            }

        seed_ids: set[str] = set()
        if file:
            rel_file = file
            if os.path.isabs(file):
                rel_file = _safe_relpath(file, self.repo_path)
            seed_ids.update(files.get(rel_file, {}).get("nodes", []))

        if symbol:
            needle = symbol.lower()
            symbol_index = graph.get("symbol_index", {})
            seed_ids.update(symbol_index.get(needle, []))
            if not seed_ids:
                for node_id, node in nodes.items():
                    haystack = " ".join(
                        [
                            str(node.get("name", "")),
                            str(node.get("qualified_name", "")),
                        ]
                    ).lower()
                    if needle in haystack:
                        seed_ids.add(node_id)

        if not seed_ids:
            seed_ids = set(list(nodes.keys())[: min(limit, len(nodes))])

        adjacency = defaultdict(list)
        for edge_id, edge in edges.items():
            if relation and edge.get("relation") != relation:
                continue
            adjacency[edge["from"]].append(edge_id)
            adjacency[edge["to"]].append(edge_id)

        seen_nodes = set(seed_ids)
        seen_edges: set[str] = set()
        queue = deque((node_id, 0) for node_id in seed_ids)
        while queue:
            node_id, dist = queue.popleft()
            if dist >= max(0, int(depth)):
                continue
            for edge_id in adjacency.get(node_id, []):
                seen_edges.add(edge_id)
                edge = edges[edge_id]
                neighbor = edge["to"] if edge["from"] == node_id else edge["from"]
                if neighbor not in seen_nodes:
                    seen_nodes.add(neighbor)
                    queue.append((neighbor, dist + 1))

        ordered_nodes = sorted(
            (
                nodes[node_id]
                for node_id in seen_nodes
                if node_id in nodes and nodes[node_id].get("file")
            ),
            key=lambda item: (
                str(item.get("file") or ""),
                int(item.get("line", 0) or 0),
                str(item.get("name") or ""),
            ),
        )[:limit]
        ordered_edges = [
            edges[edge_id]
            for edge_id in sorted(seen_edges)
            if edge_id in edges
            and edge_get_node(edges[edge_id], "from") in {n["id"] for n in ordered_nodes}
            and edge_get_node(edges[edge_id], "to") in {n["id"] for n in ordered_nodes}
        ]
        return {
            "status": "ok",
            "nodes": ordered_nodes,
            "edges": ordered_edges,
            "count": len(ordered_nodes),
            "depth": depth,
            "relation": relation,
            "graph_path": self.graph_path,
            "stats": graph.get("stats", {}),
        }

    def export(self) -> dict[str, Any]:
        """Handle export."""
        graph = self.load_graph()
        return {
            "status": "ok",
            "graph_path": self.graph_path,
            "graph": graph,
        }

    def disparate_relations(
        self,
        *,
        relation: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return native disparate relations from the persisted graph."""
        graph = self.load_graph()
        payload = self.disparate_relation_synthesizer.relation_summary(
            graph,
            relation=relation,
            limit=limit,
        )
        payload["graph_path"] = self.graph_path
        return payload

    def summary(self) -> dict[str, Any]:
        """Handle summary."""
        if not os.path.exists(self.graph_path):
            return {"status": "missing"}
        graph = self.load_graph()
        stats = graph.get("stats", {})
        return {
            "status": "ready",
            "graph_path": self.graph_path,
            "generated_at": graph.get("generated_at"),
            "generated_fmt": graph.get("generated_fmt"),
            "files": stats.get("files", 0),
            "nodes": stats.get("nodes", 0),
            "edges": stats.get("edges", 0),
            "graph_coverage_percent": stats.get("graph_coverage_percent", 0.0),
            "cfg_edges": stats.get("cfg_edges", 0),
            "dfg_edges": stats.get("dfg_edges", 0),
            "call_edges": stats.get("call_edges", 0),
            "ffi_patterns": stats.get("ffi_patterns", 0),
            "bridge_edges": stats.get("bridge_edges", 0),
            "disparate_relation_edges": stats.get("disparate_relation_edges", 0),
        }

    def load_graph(self) -> dict[str, Any]:
        """Load graph."""
        if not os.path.exists(self.graph_path):
            return self._empty_graph()
        try:
            graph_mtime = os.path.getmtime(self.graph_path)
        except OSError:
            graph_mtime = -1.0
        with self._graph_cache_lock:
            if self._graph_cache is not None and self._graph_cache[0] == graph_mtime:
                return self._graph_cache[1]
        try:
            with open(self.graph_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            corrupt_path = f"{self.graph_path}.corrupt-{int(time.time())}"
            try:
                os.replace(self.graph_path, corrupt_path)
            except OSError:
                pass
            normalized = self._empty_graph()
            with self._graph_cache_lock:
                self._graph_cache = (graph_mtime, normalized)
            return normalized
        if "nodes" not in data:
            normalized = self._empty_graph()
            with self._graph_cache_lock:
                self._graph_cache = (graph_mtime, normalized)
            return normalized
        normalized, changed = self._normalize_graph(data)
        if changed:
            self._write_graph(normalized)
        else:
            with self._graph_cache_lock:
                self._graph_cache = (graph_mtime, normalized)
        return normalized

    def _write_graph(self, graph: dict[str, Any]) -> None:
        os.makedirs(self.graph_dir, exist_ok=True)
        unique_suffix = f".tmp.{os.getpid()}.{threading.get_ident()}"
        with self._graph_write_lock:
            for path in (self.graph_path, self.code_graph_path):
                tmp_path = f"{path}{unique_suffix}"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(graph, f, indent=2, sort_keys=True)
                os.replace(tmp_path, path)
            stats_payload = dict(graph.get("stats") or {})
            stats_path = os.path.join(self.graph_dir, "graph_stats.json")
            stats_tmp = f"{stats_path}{unique_suffix}"
            with open(stats_tmp, "w", encoding="utf-8") as f:
                json.dump(stats_payload, f, indent=2, sort_keys=True)
            os.replace(stats_tmp, stats_path)
            manifest = load_manifest(self.saguaro_dir)
            if isinstance(manifest.get("artifacts"), dict):
                manifest["artifacts"]["graph/graph.json"] = snapshot_artifact(
                    self.graph_path, os.path.join("graph", "graph.json")
                )
                summary = dict(manifest.get("summary") or {})
                summary.update(
                    {
                        "graph_files": int(stats_payload.get("files", 0) or 0),
                        "graph_nodes": int(stats_payload.get("nodes", 0) or 0),
                        "graph_edges": int(stats_payload.get("edges", 0) or 0),
                    }
                )
                manifest["summary"] = summary
                atomic_write_json(
                    manifest_path(self.saguaro_dir),
                    manifest,
                    indent=2,
                    sort_keys=True,
                )
        try:
            graph_mtime = os.path.getmtime(self.graph_path)
        except OSError:
            graph_mtime = time.time()
        with self._graph_cache_lock:
            self._graph_cache = (graph_mtime, graph)

    def _discover_files(self, root_path: str) -> list[str]:
        return filter_indexable_files(
            get_code_files(root_path),
            repo_path=self.repo_path,
        )

    def _resolve_changed_files(
        self, changed_files: list[str] | None
    ) -> tuple[list[str] | None, list[str]]:
        if changed_files is None:
            return None, []
        resolved: list[str] = []
        deleted: list[str] = []
        for path in changed_files:
            if not path:
                continue
            full = path if os.path.isabs(path) else os.path.join(self.repo_path, path)
            full = os.path.abspath(full)
            if os.path.exists(full):
                resolved.append(full)
            else:
                deleted.append(full)
        return sorted(set(resolved)), sorted(set(deleted))

    def _empty_graph(self) -> dict[str, Any]:
        return {
            "version": self.GRAPH_SCHEMA_VERSION,
            "repo_path": self.repo_path,
            "generated_at": None,
            "generated_fmt": None,
            "nodes": {},
            "edges": {},
            "files": {},
            "ffi_patterns": {},
            "symbol_index": {},
            "term_index": {},
            "entity_to_node": {},
            "stats": {},
        }

    def _normalize_graph(
        self, graph: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        changed = False
        normalized = dict(graph)
        template = self._empty_graph()
        for key, default in template.items():
            if key not in normalized:
                normalized[key] = default
                changed = True

        if int(normalized.get("version", 0) or 0) != self.GRAPH_SCHEMA_VERSION:
            normalized["version"] = self.GRAPH_SCHEMA_VERSION
            self._rebuild_indices(normalized)
            changed = True

        node_map = normalized.get("nodes", {})
        for node_id, node in node_map.items():
            if "id" not in node:
                node["id"] = node_id
                changed = True
            node_name = str(node.get("name") or "")
            qualified_name = str(node.get("qualified_name") or "").strip()
            line = int(node.get("line", 0) or 0)
            file_path = str(node.get("file") or "")
            if file_path and not node.get("entity_id"):
                symbol_name = qualified_name or node_name or node_id
                node["entity_id"] = f"{file_path}:{symbol_name}:{line}"
                changed = True

        if (
            not normalized.get("symbol_index")
            or not normalized.get("term_index")
            or not normalized.get("entity_to_node")
        ):
            self._rebuild_indices(normalized)
            changed = True

        original_stats = dict(normalized.get("stats", {}))
        self._refresh_stats(
            normalized,
            parsed_files=int(
                original_stats.get(
                    "parsed_files",
                    len(normalized.get("files", {})),
                )
                or len(normalized.get("files", {}))
            ),
            parser_coverage_percent=(
                float(original_stats.get("parser_coverage_percent", 0.0))
                if "parser_coverage_percent" in original_stats
                else None
            ),
            total_files=None,
            incremental=(
                bool(original_stats.get("incremental", False))
                if "incremental" in original_stats
                else None
            ),
        )
        if normalized.get("stats", {}) != original_stats:
            changed = True
        return normalized, changed

    def _drop_files(self, graph: dict[str, Any], rel_files: set[str]) -> dict[str, Any]:
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})
        files = graph.get("files", {})
        ffi_patterns = graph.get("ffi_patterns", {})
        drop_nodes: set[str] = set()
        drop_edges: set[str] = set()
        for rel_file in rel_files:
            file_entry = files.pop(rel_file, None)
            if not file_entry:
                continue
            drop_nodes.update(file_entry.get("nodes", []))
            drop_edges.update(file_entry.get("edges", []))
            for ffi_id in file_entry.get("ffi_patterns", []):
                ffi_patterns.pop(str(ffi_id), None)

        for node_id in list(drop_nodes):
            nodes.pop(node_id, None)

        for edge_id, edge in list(edges.items()):
            if edge_id in drop_edges or edge.get("file") in rel_files:
                edges.pop(edge_id, None)
                continue
            if edge.get("from") in drop_nodes or edge.get("to") in drop_nodes:
                edges.pop(edge_id, None)

        return graph

    def _module_index(self, files: list[str]) -> dict[str, str]:
        index: dict[str, str] = {}
        for file_path in files:
            rel_file = _safe_relpath(file_path, self.repo_path)
            for module_hint in self._module_hints(rel_file):
                index.setdefault(module_hint, rel_file)
        return index

    @staticmethod
    def _module_hints(rel_file: str) -> list[str]:
        normalized = rel_file.replace("\\", "/")
        base = os.path.basename(normalized)
        stem = os.path.splitext(normalized)[0]
        base_stem = os.path.splitext(base)[0]
        hints: list[str] = []
        if rel_file.endswith(".py"):
            module = normalized[:-3]
            if module.endswith("/__init__"):
                module = module[: -len("/__init__")]
            hints.append(module.replace("/", "."))
        elif rel_file.endswith((".js", ".jsx", ".ts", ".tsx", ".go")):
            hints.append(stem.replace("/", "."))
        elif rel_file.endswith(
            (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".m", ".mm")
        ):
            hints.extend([normalized, stem, base, base_stem])
        return [hint for hint in dict.fromkeys(hints) if hint]

    @staticmethod
    def _module_hint(rel_file: str) -> str | None:
        hints = GraphService._module_hints(rel_file)
        return hints[0] if hints else None

    def _ingest_file(
        self,
        graph: dict[str, Any],
        rel_file: str,
        entities: list[CodeEntity],
        module_index: dict[str, str],
    ) -> None:
        graph.setdefault("nodes", {})
        graph.setdefault("edges", {})
        graph.setdefault("files", {})
        graph.setdefault("ffi_patterns", {})
        file_nodes: list[str] = []
        file_edges: list[str] = []
        file_ffi_patterns: list[str] = []
        node_lookup: dict[str, str] = {}
        dep_payload: dict[str, Any] | None = None
        abs_path = os.path.join(self.repo_path, rel_file)
        source = self._read_text(abs_path)

        for entity in entities:
            if entity.type == "dependency_graph":
                try:
                    dep_payload = json.loads(entity.content)
                except Exception:
                    dep_payload = None
                continue
            node_id = self._node_id(rel_file, entity.name, entity.type, entity.start_line)
            identity = entity_identity(
                self.repo_path,
                os.path.join(self.repo_path, rel_file),
                entity.name,
                entity.type,
                entity.start_line,
            )
            metadata = dict(getattr(entity, "metadata", {}) or {})
            graph["nodes"][node_id] = {
                "id": node_id,
                "name": identity["display_name"],
                "qualified_name": identity["qualified_name"],
                "entity_id": identity["entity_id"],
                "type": entity.type,
                "file": rel_file,
                "line": entity.start_line,
                "end_line": entity.end_line,
                "language": str(metadata.get("language") or self.parse_service.parser.detect_language(rel_file, source)),
                "file_role": metadata.get("file_role"),
                "chunk_role": metadata.get("chunk_role"),
                "role_tags": list(metadata.get("role_tags") or []),
                "feature_families": list(metadata.get("feature_families") or []),
                "boundary_markers": list(metadata.get("boundary_markers") or []),
                "signature_fingerprint": str(metadata.get("signature_fingerprint") or ""),
                "structural_fingerprint": str(metadata.get("structural_fingerprint") or ""),
                "parser_uncertainty": str(metadata.get("parser_uncertainty") or "unknown"),
                "terms": list(metadata.get("terms") or []),
                "metadata": metadata,
            }
            file_nodes.append(node_id)
            node_lookup.setdefault(entity.name, node_id)
            node_lookup.setdefault(identity["display_name"], node_id)

        file_node_id = next(
            (node_id for node_id in file_nodes if graph["nodes"][node_id]["type"] == "file"),
            None,
        )
        if file_node_id is None:
            file_node_id = self._node_id(rel_file, os.path.basename(rel_file), "file", 1)
            identity = entity_identity(
                self.repo_path,
                abs_path,
                os.path.basename(rel_file),
                "file",
                1,
            )
            graph["nodes"][file_node_id] = {
                "id": file_node_id,
                "name": identity["display_name"],
                "qualified_name": identity["qualified_name"],
                "entity_id": identity["entity_id"],
                "type": "file",
                "file": rel_file,
                "line": 1,
                "end_line": max(1, source.count("\n") + 1),
                "language": self.parse_service.parser.detect_language(rel_file, source),
                "role_tags": [],
                "feature_families": [],
                "boundary_markers": [],
                "signature_fingerprint": "",
                "structural_fingerprint": "",
                "parser_uncertainty": "medium",
                "terms": [],
            }
            file_nodes.append(file_node_id)

        if dep_payload:
            for raw_import in dep_payload.get("imports", []):
                target = self._resolve_import_target(raw_import, module_index)
                import_node_id = target or f"external::{raw_import}"
                if import_node_id.startswith("external::") and import_node_id not in graph["nodes"]:
                    graph["nodes"][import_node_id] = {
                        "id": import_node_id,
                        "name": raw_import,
                        "type": "external",
                        "file": None,
                        "line": 0,
                        "end_line": 0,
                    }
                edge_id = self._edge_id(file_node_id or rel_file, import_node_id, "imports", 1)
                graph["edges"][edge_id] = {
                    "id": edge_id,
                    "from": file_node_id or rel_file,
                    "to": import_node_id,
                    "relation": "imports",
                    "line": 1,
                    "file": rel_file,
                    "raw": raw_import,
                }
                file_edges.append(edge_id)

            for item in dep_payload.get("internal_edges", []):
                src_name = str(item.get("from", "")).strip()
                dst_name = str(item.get("to", "")).strip()
                if not src_name or not dst_name:
                    continue
                src_id = node_lookup.get(src_name)
                if src_id is None:
                    src_id = self._synthetic_symbol(graph, rel_file, src_name)
                    node_lookup[src_name] = src_id
                    file_nodes.append(src_id)
                dst_id = node_lookup.get(dst_name)
                if dst_id is None:
                    dst_id = self._synthetic_symbol(graph, rel_file, dst_name)
                    node_lookup[dst_name] = dst_id
                    file_nodes.append(dst_id)
                edge_id = self._edge_id(
                    src_id,
                    dst_id,
                    str(item.get("relation", "related")),
                    int(item.get("line", 1) or 1),
                )
                graph["edges"][edge_id] = {
                    "id": edge_id,
                    "from": src_id,
                    "to": dst_id,
                    "relation": str(item.get("relation", "related")),
                    "line": int(item.get("line", 1) or 1),
                    "file": rel_file,
                }
                file_edges.append(edge_id)

        for builder in (self.cfg_builder, self.dfg_builder, self.call_graph_builder):
            payload = builder.build(rel_file, source)
            for node in payload.get("nodes", []):
                node_id = str(node.get("id") or "")
                if not node_id:
                    continue
                merged = dict(node)
                merged.setdefault("file", rel_file)
                graph["nodes"][node_id] = merged
                file_nodes.append(node_id)
            for edge in payload.get("edges", []):
                edge_id = str(edge.get("id") or "")
                if not edge_id:
                    continue
                merged = dict(edge)
                merged.setdefault("file", rel_file)
                graph["edges"][edge_id] = merged
                file_edges.append(edge_id)

        for pattern in self.ffi_scanner.scan_file(rel_file, source):
            pattern_id = str(pattern.get("id") or "")
            if not pattern_id:
                continue
            graph["ffi_patterns"][pattern_id] = dict(pattern)
            graph["nodes"][pattern_id] = {
                "id": pattern_id,
                "name": str(pattern.get("kind") or "ffi_pattern"),
                "qualified_name": str(pattern.get("kind") or "ffi_pattern"),
                "type": "ffi_pattern",
                "file": rel_file,
                "line": int(pattern.get("line") or 0),
                "end_line": int(pattern.get("line") or 0),
                "confidence": float(pattern.get("confidence") or 0.0),
                "source": "ffi_scanner",
            }
            file_nodes.append(pattern_id)
            file_ffi_patterns.append(pattern_id)
            edge_line = int(pattern.get("line") or 0)
            edge_id = self._edge_id(file_node_id, pattern_id, "ffi_detected", edge_line)
            graph["edges"][edge_id] = {
                "id": edge_id,
                "from": file_node_id,
                "to": pattern_id,
                "relation": "ffi_detected",
                "line": edge_line,
                "file": rel_file,
                "confidence": float(pattern.get("confidence") or 0.0),
                "source": "ffi_scanner",
            }
            file_edges.append(edge_id)

        graph["files"][rel_file] = {
            "nodes": sorted(set(file_nodes)),
            "edges": sorted(set(file_edges)),
            "ffi_patterns": sorted(set(file_ffi_patterns)),
            "module_hint": self._module_hint(rel_file),
        }

    def _synthetic_symbol(self, graph: dict[str, Any], rel_file: str, symbol: str) -> str:
        node_id = self._node_id(rel_file, symbol, "symbol", 0)
        if node_id not in graph["nodes"]:
            identity = entity_identity(
                self.repo_path,
                os.path.join(self.repo_path, rel_file),
                symbol,
                "symbol",
                0,
            )
            graph["nodes"][node_id] = {
                "id": node_id,
                "name": identity["display_name"],
                "qualified_name": None,
                "entity_id": identity["entity_id"],
                "type": "symbol",
                "file": rel_file,
                "line": 0,
                "end_line": 0,
            }
        return node_id

    def _rebuild_indices(self, graph: dict[str, Any]) -> None:
        symbol_index: dict[str, list[str]] = {}
        term_index: dict[str, list[str]] = {}
        entity_to_node: dict[str, str] = {}
        for node_id, node in graph.get("nodes", {}).items():
            file_path = str(node.get("file") or "")
            if not file_path:
                continue
            entity_id = str(node.get("entity_id") or "")
            if entity_id:
                entity_to_node[entity_id] = node_id
            for exact in {str(node.get("name") or ""), str(node.get("qualified_name") or "")}:
                normalized = exact.strip().lower()
                if not normalized:
                    continue
                symbol_index.setdefault(normalized, []).append(node_id)
            raw_terms = " ".join(
                [
                    str(node.get("name") or ""),
                    str(node.get("qualified_name") or ""),
                    file_path,
                ]
            )
            for term in _expanded_search_terms(raw_terms):
                term_index.setdefault(term, []).append(node_id)
        graph["symbol_index"] = {
            key: sorted(set(value)) for key, value in symbol_index.items()
        }
        graph["term_index"] = {key: sorted(set(value)) for key, value in term_index.items()}
        graph["entity_to_node"] = entity_to_node

    def _rebuild_bridge_edges(self, graph: dict[str, Any]) -> None:
        edges = graph.setdefault("edges", {})
        files = graph.setdefault("files", {})
        relation_families = self.disparate_relation_synthesizer.relation_families

        for edge_id, edge in list(edges.items()):
            relation = str(edge.get("relation") or "")
            if relation == "ffi_bridge" or relation in relation_families:
                edges.pop(edge_id, None)

        for entry in files.values():
            if isinstance(entry, dict) and isinstance(entry.get("edges"), list):
                entry["edges"] = [
                    edge_id
                    for edge_id in entry["edges"]
                    if str(edges.get(edge_id, {}).get("relation") or "") not in {"ffi_bridge", *relation_families}
                ]

        patterns = list(graph.setdefault("ffi_patterns", {}).values())
        bridge_edges = self.bridge_synthesizer.synthesize(patterns)
        for edge in bridge_edges:
            edge_id = str(edge.get("id") or "")
            if not edge_id:
                continue
            edges[edge_id] = edge
            rel_file = str(edge.get("file") or "")
            if rel_file and rel_file in files:
                file_edges = files[rel_file].setdefault("edges", [])
                if edge_id not in file_edges:
                    file_edges.append(edge_id)

        generation_id = str(graph.get("generated_at") or int(time.time()))
        for edge in self.disparate_relation_synthesizer.synthesize_graph_edges(
            graph,
            generation_id=generation_id,
        ):
            edge_id = str(edge.get("id") or "")
            if not edge_id:
                continue
            edges[edge_id] = edge
            for rel_file in {
                str(edge.get("file") or ""),
                str(edge.get("source_path") or ""),
                str(edge.get("target_path") or ""),
            }:
                if rel_file and rel_file in files:
                    file_edges = files[rel_file].setdefault("edges", [])
                    if edge_id not in file_edges:
                        file_edges.append(edge_id)

        for entry in files.values():
            if isinstance(entry, dict) and isinstance(entry.get("edges"), list):
                entry["edges"] = sorted(set(str(item) for item in entry["edges"]))

    def _refresh_stats(
        self,
        graph: dict[str, Any],
        *,
        parsed_files: int,
        parser_coverage_percent: float | None,
        total_files: int | None,
        incremental: bool | None,
    ) -> None:
        existing = dict(graph.get("stats", {}))
        edges = list(graph.get("edges", {}).values())
        files_count = len(graph.get("files", {}))
        stats = {
            **existing,
            "files": files_count,
            "parsed_files": int(parsed_files),
            "nodes": len(graph.get("nodes", {})),
            "edges": len(graph.get("edges", {})),
            "cfg_edges": sum(
                1
                for edge in edges
                if str(edge.get("relation") or "").startswith("cfg_")
            ),
            "dfg_edges": sum(
                1
                for edge in edges
                if str(edge.get("relation") or "").startswith("dfg_")
            ),
            "call_edges": sum(
                1 for edge in edges if str(edge.get("relation") or "") == "calls"
            ),
            "ffi_patterns": len(graph.get("ffi_patterns", {})),
            "bridge_edges": sum(
                1 for edge in edges if str(edge.get("relation") or "") == "ffi_bridge"
            ),
            "disparate_relation_edges": sum(
                1
                for edge in edges
                if str(edge.get("relation") or "") in self.disparate_relation_synthesizer.relation_families
            ),
        }
        if total_files is not None:
            stats["graph_coverage_percent"] = round(
                (files_count / max(int(total_files or 1), 1)) * 100,
                1,
            )
        elif "graph_coverage_percent" not in stats:
            stats["graph_coverage_percent"] = 0.0
        if parser_coverage_percent is not None:
            stats["parser_coverage_percent"] = float(parser_coverage_percent)
        elif "parser_coverage_percent" not in stats:
            stats["parser_coverage_percent"] = 0.0
        if incremental is not None:
            stats["incremental"] = bool(incremental)
        elif "incremental" not in stats:
            stats["incremental"] = False
        graph["stats"] = stats

    def _resolve_import_target(
        self, raw_import: str, module_index: dict[str, str]
    ) -> str | None:
        raw = str(raw_import or "").strip()
        include_match = re.search(r'#include\s*[<"]([^">]+)[">]', raw)
        if include_match:
            candidate = include_match.group(1).strip()
        else:
            candidate = (
                raw.replace("import ", "")
                .replace("from ", "")
                .split(" import ")[0]
                .strip()
                .strip("\"'")
                .strip("<>")
            )
        variants = [
            candidate,
            candidate.replace("/", "."),
            os.path.splitext(candidate)[0],
            os.path.basename(candidate),
            os.path.splitext(os.path.basename(candidate))[0],
        ]
        for variant in dict.fromkeys([item for item in variants if item]):
            rel_file = module_index.get(variant)
            if rel_file:
                return self._node_id(rel_file, os.path.basename(rel_file), "file", 1)
        return None

    @staticmethod
    def _node_id(rel_file: str, name: str, kind: str, line: int) -> str:
        return f"{rel_file}::{name}::{kind}::{int(line)}"

    @staticmethod
    def _edge_id(src: str, dst: str, relation: str, line: int) -> str:
        return f"{src}->{dst}::{relation}::{int(line)}"

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        except OSError:
            return ""


def edge_get_node(edge: dict[str, Any], key: str) -> str:
    """Handle edge get node."""
    return str(edge.get(key, ""))


class QueryService:
    """CPU-first query orchestration over vector, lexical, and graph signals."""

    _TASK_STRATEGIES = {
        "lexical": "lexical",
        "semantic": "semantic",
        "hybrid": "hybrid",
        "graph": "graph",
        "symbol": "lexical",
        "search-by-symbol": "lexical",
        "concept": "hybrid",
        "search-by-concept": "hybrid",
        "impact": "graph",
        "search-by-impact": "graph",
        "drift": "graph",
        "search-by-drift": "graph",
        "test-failure": "hybrid",
        "search-by-test-failure": "hybrid",
        "policy": "lexical",
        "search-by-policy": "lexical",
        "roadmap": "hybrid",
        "search-by-roadmap": "hybrid",
    }

    def __init__(
        self,
        repo_path: str,
        graph_service: GraphService,
        vectors_dir: str,
        load_stats: Callable[[], dict[str, Any]],
        load_index_stats: Callable[[], dict[str, Any]],
        check_store_compatibility: Callable[[int], dict[str, Any]],
        encode_text: Callable[[str, int, int], Any],
        hybrid_rerank: Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]],
        extract_terms: Callable[[str, int], list[str]],
        result_is_in_repo: Callable[[dict[str, Any]], bool],
        refresh_index: Callable[[list[str]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.graph_service = graph_service
        self.vectors_dir = vectors_dir
        self._load_stats = load_stats
        self._load_index_stats = load_index_stats
        self._check_store_compatibility = check_store_compatibility
        self._encode_text = encode_text
        self._hybrid_rerank = hybrid_rerank
        self._extract_terms = extract_terms
        self._result_is_in_repo = result_is_in_repo
        self._refresh_index = refresh_index
        self._pipeline = QueryPipeline(
            repo_path=self.repo_path,
            load_stats=self._load_stats,
            extract_terms=self._extract_terms,
            result_is_in_repo=self._result_is_in_repo,
        )
        self._store_cache_lock = threading.Lock()
        self._store_cache: tuple[tuple[float, float, int, int], Any] | None = None

    def _shared_store(self, *, active_dim: int, total_dim: int) -> Any:
        meta_path = os.path.join(self.vectors_dir, "index_meta.json")
        schema_path = os.path.join(os.path.dirname(self.vectors_dir), "index_schema.json")
        try:
            meta_mtime = os.path.getmtime(meta_path)
        except OSError:
            meta_mtime = 0.0
        try:
            schema_mtime = os.path.getmtime(schema_path)
        except OSError:
            schema_mtime = 0.0
        cache_key = (meta_mtime, schema_mtime, int(active_dim), int(total_dim))
        with self._store_cache_lock:
            if self._store_cache is not None and self._store_cache[0] == cache_key:
                return self._store_cache[1]
            if self._store_cache is not None:
                stale = self._store_cache[1]
                close = getattr(stale, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            store = VectorStore(
                storage_path=self.vectors_dir,
                dim=active_dim,
                active_dim=active_dim,
                total_dim=total_dim,
                read_only=True,
            )
            self._store_cache = (cache_key, store)
            return store

    def _prepare_query_context(self, *, strategies: set[str]) -> dict[str, Any]:
        stats = self._load_stats()
        active_dim = int(stats.get("active_dim", 4096))
        total_dim = int(stats.get("total_dim", 8192))
        graph: dict[str, Any] = {}
        if strategies & {"lexical", "graph", "hybrid"}:
            graph = self.graph_service.load_graph()
            if not graph.get("nodes"):
                try:
                    self.graph_service.build(path=".", incremental=True)
                except Exception:
                    pass
                graph = self.graph_service.load_graph()
        compatibility = self._check_store_compatibility(expected_dim=active_dim)
        store = None
        if strategies & {"semantic", "hybrid"} and not compatibility.get("incompatible"):
            store = self._shared_store(active_dim=active_dim, total_dim=total_dim)
        return {
            "stats": stats,
            "active_dim": active_dim,
            "total_dim": total_dim,
            "graph": graph,
            "compatibility": compatibility,
            "store": store,
        }

    def prime(self, *, strategies: set[str] | None = None) -> dict[str, Any]:
        selected_strategies = set(strategies or {"hybrid"})
        started = time.perf_counter()
        context = self._prepare_query_context(strategies=selected_strategies)
        return {
            "status": "ok",
            "strategies": sorted(selected_strategies),
            "prepared_in_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "graph_nodes": len(dict(context.get("graph") or {}).get("nodes", {})),
            "store_ready": context.get("store") is not None,
        }

    def _execute_query(
        self,
        *,
        raw_text: str,
        requested_strategy: str,
        resolved_strategy: str,
        k: int,
        explain: bool,
        aal: str,
        domain: str | list[str] | None,
        repo_role: str,
        auto_refresh: bool,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        expanded_text = self._expand_query_text(raw_text, requested_strategy)
        stale_files = self._stale_repo_files()
        auto_refreshed = False
        auto_refreshed_files: list[str] = []
        if auto_refresh:
            stale_files, auto_refreshed, auto_refreshed_files = self._maybe_refresh_stale_files(
                raw_text,
                stale_files,
            )

        active_dim = int(context.get("active_dim", 4096))
        total_dim = int(context.get("total_dim", 8192))
        graph = dict(context.get("graph") or {})
        compatibility = dict(context.get("compatibility") or {})
        store = context.get("store")

        semantic_rows: list[dict[str, Any]] = []
        exact_rows = self._exact_symbol_rows(raw_text, graph)
        lexical_rows = exact_rows + self._lexical_rows(raw_text, graph)
        graph_rows = self._graph_rows(raw_text, graph, seed_rows=lexical_rows)
        candidate_ids = self._candidate_entity_ids(lexical_rows, graph_rows, k)
        if self._should_bypass_candidate_prefilter(raw_text):
            candidate_ids = None
        if resolved_strategy in {"semantic", "hybrid"} and not compatibility.get(
            "incompatible"
        ):
            if store is None:
                store = self._shared_store(active_dim=active_dim, total_dim=total_dim)
            query_vec = self._encode_text(expanded_text, active_dim, total_dim)
            raw = store.query(
                query_vec,
                k=max(20, max(1, int(k)) * 12),
                query_text=expanded_text,
                candidate_ids=candidate_ids,
            )
            semantic_rows = self._hybrid_rerank(
                expanded_text,
                [item for item in raw if self._result_is_in_repo(item)],
                max(20, max(1, int(k)) * 4),
            )

        pipeline_output = self._pipeline.run(
            text=raw_text,
            strategy=resolved_strategy,
            semantic_rows=semantic_rows,
            lexical_rows=lexical_rows,
            graph_rows=graph_rows,
            k=max(1, int(k)),
            explain=explain,
            stale_files=stale_files,
            auto_refreshed=auto_refreshed,
            auto_refreshed_files=auto_refreshed_files,
        )
        merged = list(pipeline_output.get("results", []))
        if explain:
            for row in merged:
                if "explanation" not in row:
                    row["explanation"] = self._build_explanation(
                        row,
                        resolved_strategy,
                        requested_strategy=requested_strategy,
                    )
        promotable = bool(merged) and resolved_strategy in {"hybrid", "graph"}
        return {
            "results": merged[: max(1, int(k))],
            "strategy": requested_strategy,
            "execution_strategy": resolved_strategy,
            "semantic_candidates": len(semantic_rows),
            "lexical_candidates": len(lexical_rows),
            "graph_candidates": len(graph_rows),
            "seed_candidates": len(candidate_ids or []),
            "compatibility": compatibility,
            "stale_candidates": pipeline_output.get("stale_candidates", []),
            "auto_refreshed": bool(pipeline_output.get("auto_refreshed", False)),
            "auto_refreshed_files": pipeline_output.get("auto_refreshed_files", []),
            "index_age_seconds": pipeline_output.get("index_age_seconds"),
            "aes_envelope": _aes_envelope(
                action_type="repo_query",
                aal=aal,
                domain=domain,
                repo_role=repo_role,
                promotable=promotable,
            ),
        }

    def query(
        self,
        text: str,
        k: int = 5,
        strategy: str = "hybrid",
        explain: bool = False,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "analysis_local",
        auto_refresh: bool = False,
    ) -> dict[str, Any]:
        """Handle query."""
        raw_text = (text or "").strip()
        requested_strategy = (strategy or "hybrid").lower()
        resolved_strategy = self._TASK_STRATEGIES.get(requested_strategy)
        if resolved_strategy is None:
            raise ValueError(f"Unsupported query strategy: {strategy}")
        context = self._prepare_query_context(strategies={resolved_strategy})
        return self._execute_query(
            raw_text=raw_text,
            requested_strategy=requested_strategy,
            resolved_strategy=resolved_strategy,
            k=k,
            explain=explain,
            aal=aal,
            domain=domain,
            repo_role=repo_role,
            auto_refresh=auto_refresh,
            context=context,
        )

    def query_many(
        self,
        queries: list[str],
        *,
        k: int = 5,
        strategy: str = "hybrid",
        explain: bool = False,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "analysis_local",
        auto_refresh: bool = False,
    ) -> dict[str, dict[str, Any]]:
        normalized = [str(item or "").strip() for item in queries if str(item or "").strip()]
        if not normalized:
            return {}
        requested_strategy = (strategy or "hybrid").lower()
        resolved_strategy = self._TASK_STRATEGIES.get(requested_strategy)
        if resolved_strategy is None:
            raise ValueError(f"Unsupported query strategy: {strategy}")
        context = self._prepare_query_context(strategies={resolved_strategy})
        results: dict[str, dict[str, Any]] = {}
        for raw_text in normalized:
            results[raw_text] = self._execute_query(
                raw_text=raw_text,
                requested_strategy=requested_strategy,
                resolved_strategy=resolved_strategy,
                k=k,
                explain=explain,
                aal=aal,
                domain=domain,
                repo_role=repo_role,
                auto_refresh=auto_refresh,
                context=context,
            )
        return results

    def _stale_repo_files(self) -> set[str]:
        git_dir = os.path.join(self.repo_path, ".git")
        if not os.path.exists(git_dir):
            return set()
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return set()
        stale: set[str] = set()
        for line in (proc.stdout or "").splitlines():
            raw = line[3:].strip()
            if not raw:
                continue
            if " -> " in raw:
                raw = raw.split(" -> ", 1)[1].strip()
            rel = canonicalize_rel_path(raw, self.repo_path)
            if not rel or is_excluded_path(rel):
                continue
            stale.add(rel)
        return stale

    def _maybe_refresh_stale_files(
        self,
        text: str,
        stale_files: set[str],
    ) -> tuple[set[str], bool, list[str]]:
        if not stale_files or not self._refresh_index:
            return stale_files, False, []
        if os.getenv("SAGUARO_QUERY_AUTO_REFRESH", "1").lower() in {"0", "false", "no"}:
            return stale_files, False, []
        candidates = self._query_relevant_stale_files(
            text,
            stale_files,
            max_files=max(1, int(os.getenv("SAGUARO_QUERY_AUTO_REFRESH_MAX", "8"))),
        )
        if not candidates:
            return stale_files, False, []
        try:
            result = self._refresh_index(candidates)
        except Exception:
            return stale_files, False, []
        if result.get("status") != "ok":
            return stale_files, False, []
        return stale_files - set(candidates), True, candidates

    def _query_relevant_stale_files(
        self,
        text: str,
        stale_files: set[str],
        *,
        max_files: int,
    ) -> list[str]:
        query = (text or "").strip().lower()
        query_terms = set(self._extract_terms(text, limit=24))
        if not query_terms and not query:
            return []
        ranked: list[tuple[float, str]] = []
        for rel in stale_files:
            if is_excluded_path(rel):
                continue
            role = classify_file_role(rel)
            if role not in {"source", "bench", "test", "doc", "config"}:
                continue
            split_path_terms = {
                token.lower()
                for token in re.split(r"[/_.-]+", rel)
                if len(token) >= 2
            }
            path_terms = set(self._extract_terms(rel, limit=24)) | split_path_terms
            basename_terms = set(
                self._extract_terms(os.path.basename(rel), limit=12)
            ) | {
                token.lower()
                for token in re.split(r"[_.-]+", os.path.basename(rel))
                if len(token) >= 2
            }
            overlap = len(query_terms & path_terms)
            basename_overlap = len(query_terms & basename_terms)
            score = float(overlap) + (0.8 * float(basename_overlap))
            basename = os.path.basename(rel).lower()
            if rel.lower() in query:
                score += 5.0
            elif basename and basename in query:
                score += 3.0
            elif basename_overlap == 0 and overlap < 2:
                if not (role == "source" and overlap >= 1 and len(query_terms) >= 4):
                    continue
            if basename_overlap == 0 and overlap < 2 and score <= 0.0:
                continue
            if role == "source":
                score += 0.3
            elif role == "bench" and query_terms & {"benchmark", "audit", "runner"}:
                score += 0.25
            elif role == "test" and query_terms.isdisjoint({"test", "tests", "assert", "pytest"}):
                score -= 0.5
            elif role == "doc" and query_terms.isdisjoint({"doc", "docs", "readme", "roadmap"}):
                score -= 0.3
            if score > 0.0:
                ranked.append((score, rel))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [path for _, path in ranked[:max_files]]

    @staticmethod
    def _expand_query_text(text: str, requested_strategy: str) -> str:
        expansions = {
            "symbol": " exact symbol definition references identifier",
            "search-by-symbol": " exact symbol definition references identifier",
            "impact": " impact affected dependencies callers callees tests policies",
            "search-by-impact": " impact affected dependencies callers callees tests policies",
            "drift": " architecture drift semantic diff regression mismatch subsystem",
            "search-by-drift": " architecture drift semantic diff regression mismatch subsystem",
            "test-failure": " test failure assertion regression stack trace reproduction",
            "search-by-test-failure": " test failure assertion regression stack trace reproduction",
            "policy": " governance policy aes compliance traceability evidence",
            "search-by-policy": " governance policy aes compliance traceability evidence",
            "roadmap": " roadmap milestone architecture campaign objective backlog",
            "search-by-roadmap": " roadmap milestone architecture campaign objective backlog",
        }
        return f"{text}{expansions.get(requested_strategy, '')}".strip()

    def _exact_symbol_rows(
        self,
        text: str,
        graph: dict[str, Any],
    ) -> list[dict[str, Any]]:
        needle = (text or "").strip().lower()
        if not needle:
            return []
        node_map = graph.get("nodes", {})
        symbol_index = graph.get("symbol_index", {})
        rows = []
        for node_id in symbol_index.get(needle, []):
            node = node_map.get(node_id)
            if not node or not node.get("file"):
                continue
            rows.append(
                {
                    "name": node.get("name", ""),
                    "qualified_name": node.get("qualified_name"),
                    "entity_id": node.get("entity_id"),
                    "type": node.get("type", "symbol"),
                    "file": node.get("file", ""),
                    "line": node.get("line", 0),
                    "end_line": node.get("end_line", 0),
                    "lexical_score": 2.0,
                    "graph_score": 0.0,
                    "semantic_score": 0.0,
                    "matched_terms": [needle],
                    "reason": "Exact graph symbol match.",
                    "provenance": ["graph", "lexical"],
                    "node_id": node_id,
                }
            )
        return rows

    def _lexical_rows(self, text: str, graph: dict[str, Any]) -> list[dict[str, Any]]:
        terms = set(self._extract_terms(text, limit=24)) | _expanded_search_terms(text)
        query_lower = (text or "").lower()
        node_map = graph.get("nodes", {})
        term_index = graph.get("term_index", {})
        candidate_node_ids = {
            node_id
            for term in terms
            for node_id in term_index.get(term, [])
        }
        iterable = (
            (node_id, node_map[node_id])
            for node_id in candidate_node_ids
            if node_id in node_map
        )
        if not candidate_node_ids:
            iterable = graph.get("nodes", {}).items()
        rows: list[dict[str, Any]] = []
        for node_id, node in iterable:
            file_path = str(node.get("file") or "")
            if not file_path:
                continue
            name = str(node.get("name") or "")
            qualified_name = str(node.get("qualified_name") or "")
            haystack = f"{name} {qualified_name} {file_path}".lower()
            matched_terms = sorted(term for term in terms if term and term in haystack)
            overlap = len(matched_terms)
            exact_bonus = 0.0
            if name.lower() and name.lower() in query_lower:
                exact_bonus += 0.4
            if qualified_name.lower() and qualified_name.lower() in query_lower:
                exact_bonus += 0.5
            path_bonus = 0.25 if file_path.lower() in query_lower else 0.0
            score = (overlap / max(len(terms), 1)) + exact_bonus + path_bonus
            if score <= 0:
                continue
            rows.append(
                {
                    "name": name,
                    "qualified_name": qualified_name or None,
                    "entity_id": node.get("entity_id"),
                    "type": node.get("type", "symbol"),
                    "file": file_path,
                    "line": node.get("line", 0),
                    "end_line": node.get("end_line", 0),
                    "lexical_score": round(score, 6),
                    "graph_score": 0.0,
                    "semantic_score": 0.0,
                    "matched_terms": matched_terms,
                    "reason": "Lexical repository graph match.",
                    "provenance": ["lexical", "graph"],
                    "node_id": node_id,
                }
            )
        rows.sort(key=lambda item: item.get("lexical_score", 0.0), reverse=True)
        return rows[:80]

    def _graph_rows(
        self,
        text: str,
        graph: dict[str, Any],
        *,
        seed_rows: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        lexical_rows = (seed_rows or self._lexical_rows(text, graph))[:10]
        if not lexical_rows:
            return []
        node_map = graph.get("nodes", {})
        edge_map = graph.get("edges", {})
        adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edge_map.values():
            adjacency[edge["from"]].append(edge)
            adjacency[edge["to"]].append(edge)
        rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        for seed in lexical_rows:
            node_id = seed.get("node_id")
            if not node_id:
                continue
            queue = deque([(node_id, 0)])
            seen = {node_id}
            while queue:
                current, depth = queue.popleft()
                if depth > 1:
                    continue
                for edge in adjacency.get(current, []):
                    neighbor = edge["to"] if edge["from"] == current else edge["from"]
                    if neighbor in seen or neighbor not in node_map:
                        continue
                    seen.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    node = node_map[neighbor]
                    file_path = str(node.get("file") or "")
                    if not file_path:
                        continue
                    key = (file_path, str(node.get("name", "")), int(node.get("line", 0) or 0))
                    graph_score = seed.get("lexical_score", 0.0) * (0.45 / (depth + 1))
                    existing = rows.get(key)
                    if existing and existing.get("graph_score", 0.0) >= graph_score:
                        continue
                    rows[key] = {
                        "name": node.get("name", ""),
                        "qualified_name": node.get("qualified_name"),
                        "entity_id": node.get("entity_id"),
                        "type": node.get("type", "symbol"),
                        "file": file_path,
                        "line": node.get("line", 0),
                        "end_line": node.get("end_line", 0),
                        "lexical_score": 0.0,
                        "semantic_score": 0.0,
                        "graph_score": round(graph_score, 6),
                        "matched_terms": seed.get("matched_terms", []),
                        "graph_path": [
                            {
                                "from": current,
                                "to": neighbor,
                                "relation": edge.get("relation", "related"),
                            }
                        ],
                        "reason": "Graph-neighborhood expansion from a lexical seed.",
                        "provenance": ["graph"],
                        "node_id": neighbor,
                    }
        ordered = list(rows.values())
        ordered.sort(key=lambda item: item.get("graph_score", 0.0), reverse=True)
        return ordered[:80]

    @staticmethod
    def _candidate_entity_ids(
        lexical_rows: list[dict[str, Any]],
        graph_rows: list[dict[str, Any]],
        k: int,
    ) -> list[str] | None:
        candidate_ids = []
        for row in lexical_rows[: max(20, int(k) * 8)] + graph_rows[: max(20, int(k) * 8)]:
            entity_id = row.get("entity_id")
            if entity_id:
                candidate_ids.append(str(entity_id))
        if not candidate_ids:
            return None
        return sorted(set(candidate_ids))

    def _should_bypass_candidate_prefilter(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        if "/" in raw or "\\" in raw:
            return False
        if re.search(r"[A-Z][A-Za-z0-9_]+", raw) or "." in raw:
            return False
        terms = self._extract_terms(raw, limit=24)
        return len(terms) >= 3

    def _merge_rows(
        self,
        text: str,
        semantic_rows: list[dict[str, Any]],
        lexical_rows: list[dict[str, Any]],
        graph_rows: list[dict[str, Any]],
        strategy: str,
    ) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str, int], dict[str, Any]] = {}

        def ensure(row: dict[str, Any]) -> dict[str, Any]:
            key = (
                str(row.get("file", "")),
                str(row.get("name", "")),
                int(row.get("line", 0) or 0),
            )
            entry = merged.setdefault(
                key,
                {
                    "name": row.get("name", ""),
                    "qualified_name": row.get("qualified_name"),
                    "entity_id": row.get("entity_id"),
                    "type": row.get("type", "symbol"),
                    "file": row.get("file", ""),
                    "line": row.get("line", 0),
                    "end_line": row.get("end_line", 0),
                    "semantic_score": 0.0,
                    "lexical_score": 0.0,
                    "graph_score": 0.0,
                    "matched_terms": [],
                    "graph_path": [],
                    "candidate_pool": row.get("candidate_pool", 0),
                    "cpu_prefiltered": bool(row.get("cpu_prefiltered", False)),
                    "reason": row.get("reason", ""),
                    "provenance": [],
                },
            )
            return entry

        for row in semantic_rows:
            entry = ensure(row)
            entry["semantic_score"] = max(
                entry["semantic_score"], float(row.get("semantic_score", row.get("score", 0.0)))
            )
            entry["lexical_score"] = max(
                entry["lexical_score"], float(row.get("lexical_score", 0.0))
            )
            entry["reason"] = row.get("reason", entry["reason"])
            entry["provenance"] = sorted(set(entry["provenance"]) | {"semantic"})
            entry["candidate_pool"] = int(row.get("candidate_pool", entry["candidate_pool"]))
            entry["cpu_prefiltered"] = bool(
                row.get("cpu_prefiltered", entry["cpu_prefiltered"])
            )
            entry["qualified_name"] = row.get("qualified_name", entry.get("qualified_name"))
            entry["entity_id"] = row.get("entity_id", entry.get("entity_id"))

        for row in lexical_rows:
            entry = ensure(row)
            entry["lexical_score"] = max(
                entry["lexical_score"], float(row.get("lexical_score", 0.0))
            )
            entry["matched_terms"] = sorted(
                set(entry.get("matched_terms", [])) | set(row.get("matched_terms", []))
            )
            entry["provenance"] = sorted(set(entry["provenance"]) | {"lexical", "graph"})

        if strategy in {"graph", "hybrid"}:
            for row in graph_rows:
                entry = ensure(row)
                entry["graph_score"] = max(
                    entry["graph_score"], float(row.get("graph_score", 0.0))
                )
                entry["graph_path"] = row.get("graph_path", entry.get("graph_path", []))
                entry["matched_terms"] = sorted(
                    set(entry.get("matched_terms", [])) | set(row.get("matched_terms", []))
                )
                entry["provenance"] = sorted(set(entry["provenance"]) | {"graph"})

        weights = {
            "semantic": (0.8, 0.15, 0.05),
            "lexical": (0.0, 0.9, 0.1),
            "graph": (0.0, 0.45, 0.55),
            "hybrid": (0.55, 0.3, 0.15),
        }[strategy]
        for entry in merged.values():
            score = (
                entry["semantic_score"] * weights[0]
                + entry["lexical_score"] * weights[1]
                + entry["graph_score"] * weights[2]
            )
            if entry["name"] and str(entry["name"]).lower() in (text or "").lower():
                score += 0.1
            entry["score"] = round(score, 6)
            entry["explanation"] = self._build_explanation(
                entry,
                strategy,
                requested_strategy=strategy,
            )

        ordered = list(merged.values())
        ordered.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        for idx, row in enumerate(ordered, start=1):
            row["rank"] = idx
        return ordered

    @staticmethod
    def _build_explanation(
        row: dict[str, Any],
        strategy: str,
        *,
        requested_strategy: str,
    ) -> dict[str, Any]:
        return {
            "strategy": requested_strategy,
            "execution_strategy": strategy,
            "semantic_score": round(float(row.get("semantic_score", 0.0)), 6),
            "lexical_score": round(float(row.get("lexical_score", 0.0)), 6),
            "graph_score": round(float(row.get("graph_score", 0.0)), 6),
            "matched_terms": row.get("matched_terms", []),
            "graph_path": row.get("graph_path", []),
            "provenance": row.get("provenance", []),
            "candidate_pool": int(row.get("candidate_pool", 0) or 0),
            "cpu_prefiltered": bool(row.get("cpu_prefiltered", False)),
            "reason": row.get("reason", ""),
        }


class EvidenceService:
    """Persist evidence bundles for verification and app consumption."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.base_dir = os.path.join(self.repo_path, ".saguaro", "evidence")

    def write_bundle(self, category: str, payload: dict[str, Any]) -> str:
        """Write bundle."""
        target_dir = os.path.join(self.base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        path = os.path.join(target_dir, f"{int(time.time())}_{category}.json")
        if "aes_envelope" not in payload:
            payload = {
                **payload,
                "aes_envelope": _aes_envelope(
                    action_type="evidence_bundle",
                    repo_role="artifact_store",
                    promotable=True,
                ),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def list_bundles(self, limit: int = 20) -> list[dict[str, Any]]:
        """List bundles."""
        if not os.path.exists(self.base_dir):
            return []
        items: list[dict[str, Any]] = []
        for root, _dirs, files in os.walk(self.base_dir):
            for name in files:
                if not name.endswith(".json"):
                    continue
                path = os.path.join(root, name)
                items.append(
                    {
                        "path": _safe_relpath(path, self.repo_path),
                        "mtime": os.path.getmtime(path),
                        "size_bytes": os.path.getsize(path),
                    }
                )
        items.sort(key=lambda item: item["mtime"], reverse=True)
        return items[:limit]


class ResearchService:
    """Store external research entries separately from the code index."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.base_dir = os.path.join(self.repo_path, ".saguaro", "research")
        self.entries_path = os.path.join(self.base_dir, "entries.jsonl")

    def ingest(
        self,
        source: str,
        manifest_path: str | None = None,
        records: list[dict[str, Any]] | None = None,
        *,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Handle ingest."""
        os.makedirs(self.base_dir, exist_ok=True)
        payloads = list(records or [])
        if manifest_path:
            with open(manifest_path, encoding="utf-8") as f:
                if manifest_path.endswith(".jsonl"):
                    payloads.extend(json.loads(line) for line in f if line.strip())
                else:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        payloads.extend(loaded)
                    elif isinstance(loaded, dict):
                        payloads.append(loaded)
        normalized = []
        for item in payloads:
            normalized.append(
                {
                    "source": source,
                    "ingested_at": time.time(),
                    "record": item,
                    "aes_envelope": _aes_envelope(
                        action_type="research_ingest",
                        aal=aal,
                        domain=domain,
                        repo_role="analysis_external",
                        promotable=True,
                    ),
                }
            )
        with open(self.entries_path, "a", encoding="utf-8") as f:
            for item in normalized:
                f.write(json.dumps(item, sort_keys=True) + "\n")
        return {"status": "ok", "count": len(normalized), "entries_path": self.entries_path}

    def list_entries(self, limit: int = 50) -> list[dict[str, Any]]:
        """List entries."""
        if not os.path.exists(self.entries_path):
            return []
        entries: list[dict[str, Any]] = []
        with open(self.entries_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        entries.sort(key=lambda item: item.get("ingested_at", 0.0), reverse=True)
        return entries[:limit]


class MetricsService:
    """Persist benchmark and evaluation runs for app and CLI consumption."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.base_dir = os.path.join(self.repo_path, ".saguaro", "metrics")
        self.runs_path = os.path.join(self.base_dir, "runs.jsonl")

    def write_run(
        self,
        category: str,
        payload: dict[str, Any],
        *,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "artifact_store",
    ) -> dict[str, Any]:
        """Write run."""
        os.makedirs(self.base_dir, exist_ok=True)
        record = {
            "category": category,
            "recorded_at": time.time(),
            "payload": dict(payload),
            "aes_envelope": _aes_envelope(
                action_type="benchmark_run",
                aal=aal,
                domain=domain,
                repo_role=repo_role,
                promotable=True,
            ),
        }
        with open(self.runs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def list_runs(
        self,
        limit: int = 20,
        *,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """List runs."""
        if not os.path.exists(self.runs_path):
            return []
        runs: list[dict[str, Any]] = []
        with open(self.runs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if category and record.get("category") != category:
                    continue
                runs.append(record)
        runs.sort(key=lambda item: item.get("recorded_at", 0.0), reverse=True)
        return runs[:limit]

    def latest_run(self, *, category: str | None = None) -> dict[str, Any] | None:
        """Handle latest run."""
        runs = self.list_runs(limit=1, category=category)
        return runs[0] if runs else None


class VerifyService:
    """Augment verification results with coverage and evidence posture."""

    def __init__(
        self,
        repo_path: str,
        parse_service: ParseService,
        graph_service: GraphService,
        evidence_service: EvidenceService,
    ) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.parse_service = parse_service
        self.graph_service = graph_service
        self.evidence_service = evidence_service

    def _evidence_snapshot(self, result: dict[str, Any]) -> dict[str, Any]:
        graph = self.graph_service.load_graph()
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})
        files_of_interest = sorted(
            {
                str(item.get("file", "")).replace("\\", "/")
                for item in result.get("violations", [])
                if str(item.get("file", "")).strip() and str(item.get("file")) != "."
            }
        )
        entities = []
        for node in nodes.values():
            file_path = str(node.get("file", "") or "")
            if files_of_interest and file_path not in files_of_interest:
                continue
            entities.append(
                {
                    "file": file_path,
                    "line": node.get("line", 0),
                    "type": node.get("type", "symbol"),
                    "name": node.get("qualified_name") or node.get("name", ""),
                    "score": round(float(node.get("line", 0) or 0), 3),
                }
            )
            if len(entities) >= 50:
                break
        dependency_edges = []
        for edge in edges.values():
            src = nodes.get(edge.get("from", ""))
            dst = nodes.get(edge.get("to", ""))
            if not src or not dst:
                continue
            if files_of_interest:
                src_file = str(src.get("file", "") or "")
                dst_file = str(dst.get("file", "") or "")
                if src_file not in files_of_interest and dst_file not in files_of_interest:
                    continue
            dependency_edges.append(
                {
                    "from": src.get("qualified_name") or src.get("name", ""),
                    "to": dst.get("qualified_name") or dst.get("name", ""),
                    "relation": edge.get("relation", "related"),
                }
            )
            if len(dependency_edges) >= 50:
                break
        return {
            "primary_file": files_of_interest[0] if files_of_interest else None,
            "codebase_files": files_of_interest,
            "entities": entities,
            "dependency_graph": {
                "edges": dependency_edges,
                "reverse_edges": [],
            },
            "validation": {
                "parser_coverage_percent": result["confidence_posture"][
                    "parser_coverage_percent"
                ],
                "graph_status": result["confidence_posture"]["graph_status"],
                "graph_coverage_percent": result["confidence_posture"][
                    "graph_coverage_percent"
                ],
                "promotable": bool(result["aes_envelope"]["promotable"]),
            },
            "promotability": {
                "required_artifacts": ["evidence_bundle.json"],
                "status": (
                    "ready"
                    if result["status"] == "pass"
                    and not result["confidence_posture"]["low_confidence_reasons"]
                    else "guarded"
                ),
            },
            "traceability": {
                "review_signoffs": [],
                "signoff_token": None,
                "waiver_ids": [],
            },
        }

    def augment_result(
        self,
        result: dict[str, Any],
        *,
        evidence_bundle: bool = False,
        min_parser_coverage: float | None = None,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "analysis_local",
    ) -> dict[str, Any]:
        """Handle augment result."""
        coverage = self.parse_service.coverage_report()
        graph = self.graph_service.summary()
        parser_coverage = float(
            coverage.get(
                "dependency_quality_coverage_percent",
                coverage.get("coverage_percent", 0.0),
            )
        )
        low_confidence_reasons: list[str] = []
        if parser_coverage < 60.0:
            low_confidence_reasons.append("Parser coverage below 60%")
        if graph.get("status") != "ready":
            low_confidence_reasons.append("Repository graph not built")
        if min_parser_coverage is not None and parser_coverage < float(min_parser_coverage):
            result.setdefault("violations", []).append(
                {
                    "rule_id": "SAGUARO-PARSER-COVERAGE",
                    "message": (
                        f"Parser coverage {parser_coverage:.1f}% is below required "
                        f"threshold {float(min_parser_coverage):.1f}%."
                    ),
                    "severity": "P1",
                    "closure_level": "blocking",
                    "file": ".",
                    "line": 0,
                    "domain": ["universal"],
                    "engine": "saguaro",
                }
            )
        result["count"] = len(result.get("violations", []))
        result["status"] = "pass" if not result["violations"] else "fail"
        result["confidence_posture"] = {
            "confidence": "high" if not low_confidence_reasons else "guarded",
            "parser_coverage_percent": parser_coverage,
            "graph_status": graph.get("status", "missing"),
            "graph_coverage_percent": float(graph.get("graph_coverage_percent", 0.0) or 0.0),
            "low_confidence_reasons": low_confidence_reasons,
        }
        result["aes_envelope"] = _aes_envelope(
            action_type="repo_verify",
            aal=aal,
            domain=domain,
            repo_role=repo_role,
            promotable=result["status"] == "pass" and not low_confidence_reasons,
        )
        if evidence_bundle:
            snapshot = self._evidence_snapshot(result)
            bundle_payload = {
                "repo_path": self.repo_path,
                "generated_at": time.time(),
                "status": result["status"],
                "count": result["count"],
                "aes_envelope": result["aes_envelope"],
                "confidence_posture": result["confidence_posture"],
                "violations": result.get("violations", []),
                **snapshot,
            }
            result["evidence_bundle"] = self.evidence_service.write_bundle(
                "verification", bundle_payload
            )
        return result


class EvalService:
    """Run CPU-first local evaluation suites and persist their results."""

    def __init__(
        self,
        repo_path: str,
        graph_service: GraphService,
        metrics_service: MetricsService,
        query_runner: Callable[..., dict[str, Any]],
        verify_runner: Callable[..., dict[str, Any]],
        refresh_index: Callable[[list[str]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.graph_service = graph_service
        self.metrics_service = metrics_service
        self.query_runner = query_runner
        self.verify_runner = verify_runner
        self.refresh_index = refresh_index

    def run(
        self,
        suite: str,
        *,
        k: int = 5,
        limit: int = 8,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "analysis_local",
    ) -> dict[str, Any]:
        """Handle run."""
        normalized = (suite or "cpu_perf").lower()
        if normalized not in {
            "repoqa",
            "crosscodeeval",
            "local_live",
            "verify_regression",
            "cpu_perf",
            "retrieval_quality",
        }:
            raise ValueError(f"Unsupported evaluation suite: {suite}")

        started = time.perf_counter()
        if normalized == "verify_regression":
            payload = self._run_verify_regression()
        elif normalized == "retrieval_quality":
            freshness_sync = self._refresh_dirty_indexable_files()
            graph = self.graph_service.load_graph()
            if not graph.get("nodes"):
                self.graph_service.build(path=".", incremental=True, changed_files=None)
                graph = self.graph_service.load_graph()
            payload = self._run_retrieval_quality_suite(graph, k=k, limit=limit)
            payload["freshness_sync"] = freshness_sync
        else:
            graph = self.graph_service.load_graph()
            if not graph.get("nodes"):
                self.graph_service.build(path=".", incremental=True, changed_files=None)
                graph = self.graph_service.load_graph()
            payload = self._run_query_suite(normalized, graph, k=k, limit=limit)
        payload["wall_time_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        payload["cpu_count"] = os.cpu_count() or 1
        payload["execution_mode"] = "cpu-first"
        payload["aes_envelope"] = _aes_envelope(
            action_type="benchmark_run",
            aal=aal,
            domain=domain,
            repo_role=repo_role,
            promotable=payload.get("status") == "ok",
        )
        record = self.metrics_service.write_run(
            "eval",
            payload,
            aal=aal,
            domain=domain,
            repo_role=repo_role,
        )
        payload["metrics_record"] = record
        return payload

    def _refresh_dirty_indexable_files(self) -> dict[str, Any]:
        if not self.refresh_index:
            return {"status": "skipped", "reason": "no_refresh_callback", "files": []}
        git_dir = os.path.join(self.repo_path, ".git")
        if not os.path.exists(git_dir):
            return {"status": "skipped", "reason": "not_a_git_repo", "files": []}
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            return {"status": "skipped", "reason": "git_status_failed", "files": []}
        dirty_files: list[str] = []
        for line in (proc.stdout or "").splitlines():
            raw = line[3:].strip()
            if not raw:
                continue
            if " -> " in raw:
                raw = raw.split(" -> ", 1)[1].strip()
            rel = canonicalize_rel_path(raw, self.repo_path)
            if not rel or is_excluded_path(rel):
                continue
            dirty_files.append(rel)
        files = sorted(set(dirty_files))
        if not files:
            return {"status": "clean", "files": []}
        result = self.refresh_index(files)
        return {
            "status": result.get("status", "unknown"),
            "files": files,
            "updated_files": int(result.get("updated_files", 0) or 0),
            "indexed_files": int(result.get("indexed_files", 0) or 0),
            "removed_files": int(result.get("removed_files", 0) or 0),
        }

    def _run_query_suite(
        self,
        suite: str,
        graph: dict[str, Any],
        *,
        k: int,
        limit: int,
    ) -> dict[str, Any]:
        nodes = sorted(
            (
                node
                for node in graph.get("nodes", {}).values()
                if node.get("file")
                and node.get("type") not in {"dependency_graph", "file"}
            ),
            key=lambda item: (
                str(item.get("file", "")),
                int(item.get("line", 0) or 0),
                str(item.get("name", "")),
            ),
        )[: max(1, int(limit))]

        cases: list[dict[str, Any]] = []
        latencies: list[float] = []
        successes = 0
        for node in nodes:
            query_text = str(
                node.get("qualified_name")
                or node.get("name")
                or os.path.basename(str(node.get("file", "")))
            )
            case_started = time.perf_counter()
            result = self.query_runner(
                text=query_text,
                k=k,
                strategy="hybrid",
                explain=True,
                aal="AAL-3",
                domain=["universal"],
                repo_role="analysis_local",
                auto_refresh=False,
            )
            latency_ms = (time.perf_counter() - case_started) * 1000.0
            latencies.append(latency_ms)
            top = (result.get("results") or [{}])[0]
            success = bool(
                top
                and top.get("file") == node.get("file")
                and (
                    top.get("name") == node.get("name")
                    or top.get("qualified_name") == node.get("qualified_name")
                    or str(node.get("name", "")) in str(top.get("name", ""))
                )
            )
            successes += int(success)
            cases.append(
                {
                    "query": query_text,
                    "target_file": node.get("file"),
                    "target_name": node.get("name"),
                    "latency_ms": round(latency_ms, 3),
                    "success": success,
                    "top_result": {
                        "file": top.get("file"),
                        "name": top.get("name"),
                        "score": top.get("score"),
                    },
                }
            )

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        ordered_latencies = sorted(latencies)
        p95_index = min(
            len(ordered_latencies) - 1,
            int(max(len(ordered_latencies) - 1, 0) * 0.95),
        )
        p95_latency = ordered_latencies[p95_index] if ordered_latencies else 0.0
        thresholds = {
            "repoqa": {"avg_ms": 150.0, "p95_ms": 300.0},
            "crosscodeeval": {"avg_ms": 150.0, "p95_ms": 300.0},
            "local_live": {"avg_ms": 175.0, "p95_ms": 350.0},
            "cpu_perf": {"avg_ms": 120.0, "p95_ms": 250.0},
        }.get(suite, {"avg_ms": 150.0, "p95_ms": 300.0})

        return {
            "status": "ok",
            "suite": suite,
            "kind": "query_benchmark",
            "total_cases": len(cases),
            "successful_cases": successes,
            "accuracy": round(successes / len(cases), 4) if cases else 0.0,
            "avg_latency_ms": round(avg_latency, 3),
            "p95_latency_ms": round(p95_latency, 3),
            "max_latency_ms": round(max(latencies), 3) if latencies else 0.0,
            "slo": thresholds,
            "slo_status": (
                "pass"
                if avg_latency <= thresholds["avg_ms"]
                and p95_latency <= thresholds["p95_ms"]
                else "guarded"
            ),
            "cases": cases,
            "graph_summary": self.graph_service.summary(),
        }

    def _run_verify_regression(self) -> dict[str, Any]:
        result = self.verify_runner(
            path=".",
            engines="native",
            evidence_bundle=False,
            aal="AAL-3",
            domain=["universal"],
        )
        posture = result.get("confidence_posture", {})
        return {
            "status": "ok",
            "suite": "verify_regression",
            "kind": "verification_regression",
            "violation_count": int(result.get("count", 0) or 0),
            "verification_status": result.get("status", "unknown"),
            "confidence_posture": posture,
            "checks": {
                "parser_coverage_percent": posture.get("parser_coverage_percent", 0.0),
                "graph_status": posture.get("graph_status", "missing"),
                "low_confidence_reasons": posture.get("low_confidence_reasons", []),
            },
        }

    def _run_retrieval_quality_suite(
        self,
        graph: dict[str, Any],
        *,
        k: int,
        limit: int,
    ) -> dict[str, Any]:
        fixture_path = os.path.join(
            self.repo_path,
            "tests",
            "fixtures",
            "saguaro_accuracy",
            "anvil_query_benchmark.json",
        )
        if os.path.exists(fixture_path):
            cases = load_benchmark_cases(fixture_path)
            selected_cases = (
                cases
                if int(limit or 0) <= 0
                else cases[: max(1, int(limit))]
            )
            results_by_query: dict[str, list[dict[str, Any]]] = {}
            detailed_cases: list[dict[str, Any]] = []
            stale_hits = 0
            total_hits = 0
            top_k = max(5, int(k or 5))
            for case in selected_cases:
                query_text = str(case.get("query") or "").strip()
                expected = {
                    str(path).replace("\\", "/")
                    for path in case.get("expected_paths", []) or []
                    if path
                }
                if not query_text or not expected:
                    continue
                result = self.query_runner(
                    text=query_text,
                    k=top_k,
                    strategy="hybrid",
                    explain=True,
                    aal="AAL-3",
                    domain=["universal"],
                    repo_role="analysis_local",
                    scope="global",
                    dedupe_by="path",
                    auto_refresh=False,
                )
                rows = list(result.get("results", []))
                results_by_query[query_text] = rows
                hit_rank: int | None = None
                for idx, row in enumerate(rows, start=1):
                    row_file = str(row.get("file", "")).replace("\\", "/")
                    if row_file in expected:
                        hit_rank = idx
                        break
                for row in rows:
                    if row.get("file"):
                        total_hits += 1
                    if row.get("stale"):
                        stale_hits += 1
                detailed_cases.append(
                    {
                        "category": case.get("category", "uncategorized"),
                        "query": query_text,
                        "expected_paths": sorted(expected),
                        "hit_rank": hit_rank,
                        "hit_at_1": bool(hit_rank == 1),
                        "hit_at_3": bool(hit_rank is not None and hit_rank <= 3),
                        "top_result": rows[0] if rows else {},
                    }
                )

            score = score_benchmark_results(selected_cases, results_by_query)
            calibration = derive_query_calibration(selected_cases, results_by_query)
            persist_query_calibration(os.path.join(self.repo_path, ".saguaro"), calibration)
            critical_categories = {
                "native_qsg_runtime",
                "benchmark_audit",
                "query_engine",
                "symbol_lookup",
            }
            critical_total = 0
            critical_top1_hits = 0
            for category, bucket in score.get("categories", {}).items():
                if category not in critical_categories:
                    continue
                critical_total += int(bucket.get("total", 0) or 0)
                critical_top1_hits += int(bucket.get("top1_hits", 0) or 0)
            critical_top1 = round(
                critical_top1_hits / critical_total,
                3,
            ) if critical_total else 0.0
            high_precision = float(calibration.get("observed", {}).get("high_precision", 0.0) or 0.0)
            thresholds = {
                "critical_top1_min": 0.95,
                "broad_top1_min": 0.85,
                "broad_top3_min": 0.95,
                "high_precision_min": 0.95,
                "stale_hit_rate_max": 0.05,
            }
            stale_hit_rate = round((stale_hits / total_hits), 4) if total_hits else 0.0
            gate_pass = (
                critical_top1 >= thresholds["critical_top1_min"]
                and float(score.get("top1_precision", 0.0) or 0.0) >= thresholds["broad_top1_min"]
                and float(score.get("top3_recall", 0.0) or 0.0) >= thresholds["broad_top3_min"]
                and high_precision >= thresholds["high_precision_min"]
                and stale_hit_rate <= thresholds["stale_hit_rate_max"]
            )
            return {
                "status": "ok",
                "suite": "retrieval_quality",
                "kind": "retrieval_eval",
                "fixture_path": fixture_path,
                "total_available_cases": len(cases),
                "total_cases": int(score.get("total", 0) or 0),
                "case_limit": int(limit or 0),
                "critical_top1": critical_top1,
                "hit_at_1": score.get("top1_precision", 0.0),
                "hit_at_3": score.get("top3_recall", 0.0),
                "stale_hit_rate": stale_hit_rate,
                "confidence_calibration": calibration,
                "categories": score.get("categories", {}),
                "slo": thresholds,
                "slo_status": "pass" if gate_pass else "guarded",
                "cases": detailed_cases,
                "graph_summary": self.graph_service.summary(),
            }

        nodes = sorted(
            (
                node
                for node in graph.get("nodes", {}).values()
                if node.get("file")
                and node.get("type") not in {"dependency_graph", "file"}
                and not str(node.get("file", "")).startswith("Saguaro/")
            ),
            key=lambda item: (
                str(item.get("file", "")),
                int(item.get("line", 0) or 0),
                str(item.get("qualified_name") or item.get("name", "")),
            ),
        )[: max(1, int(limit))]

        cases: list[dict[str, Any]] = []
        hit1 = 0
        hit5 = 0
        mrr_sum = 0.0
        dedupe_ratios: list[float] = []
        stale_hits = 0
        total_hits = 0
        top_k = max(5, int(k or 5))

        for node in nodes:
            query_text = str(
                node.get("qualified_name")
                or node.get("name")
                or os.path.basename(str(node.get("file", "")))
            )
            result = self.query_runner(
                text=query_text,
                k=top_k,
                strategy="hybrid",
                explain=False,
                aal="AAL-3",
                domain=["universal"],
                repo_role="analysis_local",
                scope="global",
                dedupe_by="path",
                auto_refresh=False,
            )
            rows = list(result.get("results", []))
            target_file = str(node.get("file", ""))
            target_name = str(node.get("name", ""))
            hit_rank: int | None = None
            for idx, row in enumerate(rows, start=1):
                if str(row.get("file", "")) != target_file:
                    continue
                row_name = str(row.get("name", ""))
                row_qualified = str(row.get("qualified_name", ""))
                if (
                    row_name == target_name
                    or target_name in row_name
                    or target_name in row_qualified
                ):
                    hit_rank = idx
                    break

            if hit_rank == 1:
                hit1 += 1
            if hit_rank is not None and hit_rank <= 5:
                hit5 += 1
                mrr_sum += 1.0 / float(hit_rank)

            files = [str(item.get("file", "")) for item in rows if item.get("file")]
            unique_files = len(set(files))
            dedupe_ratios.append((unique_files / len(files)) if files else 1.0)
            for row in rows:
                row_file = str(row.get("file", ""))
                if not row_file:
                    continue
                total_hits += 1
                abs_path = (
                    row_file
                    if os.path.isabs(row_file)
                    else os.path.join(self.repo_path, row_file)
                )
                if (not os.path.exists(abs_path)) or row_file.startswith("Saguaro/"):
                    stale_hits += 1

            cases.append(
                {
                    "query": query_text,
                    "target_file": target_file,
                    "target_name": target_name,
                    "hit_rank": hit_rank,
                    "hit_at_1": bool(hit_rank == 1),
                    "hit_at_5": bool(hit_rank is not None and hit_rank <= 5),
                    "result_count": len(rows),
                }
            )

        count = len(cases)
        hit_at_1 = round((hit1 / count), 4) if count else 0.0
        hit_at_5 = round((hit5 / count), 4) if count else 0.0
        mrr = round((mrr_sum / count), 4) if count else 0.0
        dedupe_ratio = round(
            (sum(dedupe_ratios) / len(dedupe_ratios)) if dedupe_ratios else 1.0, 4
        )
        stale_hit_rate = round((stale_hits / total_hits), 4) if total_hits else 0.0
        thresholds = {
            "hit_at_5_min": 0.90,
            "mrr_min": 0.70,
            "dedupe_ratio_min": 0.80,
            "stale_hit_rate_max": 0.05,
        }
        gate_pass = (
            hit_at_5 >= thresholds["hit_at_5_min"]
            and mrr >= thresholds["mrr_min"]
            and dedupe_ratio >= thresholds["dedupe_ratio_min"]
            and stale_hit_rate <= thresholds["stale_hit_rate_max"]
        )
        return {
            "status": "ok",
            "suite": "retrieval_quality",
            "kind": "retrieval_eval",
            "total_cases": count,
            "hit_at_1": hit_at_1,
            "hit_at_5": hit_at_5,
            "mrr": mrr,
            "dedupe_ratio": dedupe_ratio,
            "stale_hit_rate": stale_hit_rate,
            "slo": thresholds,
            "slo_status": "pass" if gate_pass else "guarded",
            "cases": cases,
            "graph_summary": self.graph_service.summary(),
        }


class AppService:
    """Compose local-app payloads from graph, evidence, research, and health state."""

    def __init__(
        self,
        repo_path: str,
        health_provider: Callable[[], dict[str, Any]],
        graph_service: GraphService,
        evidence_service: EvidenceService,
        research_service: ResearchService,
        metrics_service: MetricsService,
    ) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.health_provider = health_provider
        self.graph_service = graph_service
        self.evidence_service = evidence_service
        self.research_service = research_service
        self.metrics_service = metrics_service

    def dashboard(self) -> dict[str, Any]:
        """Handle dashboard."""
        health = self.health_provider()
        latest_benchmark = self.metrics_service.latest_run(category="eval")
        graph_data = self.graph_service.load_graph()
        graph = self.graph_service.summary()
        evidence = self.evidence_service.list_bundles(limit=10)
        research = self.research_service.list_entries(limit=10)
        benchmarks = self.metrics_service.list_runs(limit=10, category="eval")
        aes_posture = self._aes_posture(health, latest_benchmark)
        return {
            "repo_path": self.repo_path,
            "health": health,
            "graph": graph,
            "evidence": evidence,
            "research": research,
            "benchmarks": benchmarks,
            "benchmark_status": latest_benchmark,
            "aes_posture": aes_posture,
            "repo_home": {
                "index_mode": health.get("runtime", {}).get("index_mode", "unknown"),
                "parser_coverage_percent": health.get("coverage", {}).get(
                    "coverage_percent", 0.0
                ),
                "graph_coverage_percent": graph.get("graph_coverage_percent", 0.0),
                "benchmark_status": latest_benchmark.get("payload", {}).get("slo_status")
                if latest_benchmark
                else "missing",
            },
            "verification_center": {
                "latest_evidence_bundle": evidence[0] if evidence else None,
                "confidence": aes_posture["promotable"],
                "blocked": aes_posture["blocked"],
            },
            "global_code_map": self._code_map(graph_data),
            "architecture_explorer": self._architecture_explorer(graph_data),
            "semantic_workspace": {
                "saved_queries": 0,
                "research_entries": len(research),
                "benchmark_runs": len(benchmarks),
            },
            "change_intelligence": self._change_intelligence(
                graph_data, evidence, latest_benchmark
            ),
            "research_center": self._research_center(research),
            "campaign": self.campaign_summary(),
        }

    @staticmethod
    def _aes_posture(
        health: dict[str, Any],
        latest_benchmark: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trusted = []
        guarded = []
        blocked = []
        coverage = health.get("coverage", {})
        graph = health.get("graph", {})
        if graph.get("status") == "ready":
            trusted.append("repository_graph")
        else:
            blocked.append("repository_graph")
        if float(coverage.get("coverage_percent", 0.0) or 0.0) >= 60.0:
            trusted.append("parser_coverage")
        else:
            guarded.append("parser_coverage")
        if latest_benchmark and latest_benchmark.get("payload", {}).get("slo_status") != "pass":
            guarded.append("cpu_benchmark_slo")
        return {
            "trusted": trusted,
            "guarded": guarded,
            "blocked": blocked,
            "promotable": not blocked,
        }

    def campaign_summary(self) -> dict[str, Any]:
        """Handle campaign summary."""
        anvil_dir = os.path.join(self.repo_path, ".anvil")
        if not os.path.exists(anvil_dir):
            return {"status": "missing", "artifacts": []}
        artifacts = []
        for name in sorted(os.listdir(anvil_dir))[:50]:
            path = os.path.join(anvil_dir, name)
            artifacts.append(
                {
                    "name": name,
                    "type": "directory" if os.path.isdir(path) else "file",
                    "size_bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
                }
            )
        return {"status": "ready", "artifacts": artifacts}

    @staticmethod
    def _code_map(graph: dict[str, Any]) -> dict[str, Any]:
        files = graph.get("files", {})
        nodes = graph.get("nodes", {})
        subsystem_counts: dict[str, int] = defaultdict(int)
        hot_files = []
        entry_points = []
        for rel_file, payload in files.items():
            subsystem = rel_file.split("/", 1)[0] if "/" in rel_file else "."
            subsystem_counts[subsystem] += 1
            node_ids = payload.get("nodes", [])
            hot_files.append(
                {
                    "file": rel_file,
                    "symbol_count": len(node_ids),
                    "module_hint": payload.get("module_hint"),
                }
            )
        for node in nodes.values():
            rel_file = str(node.get("file") or "")
            if not rel_file:
                continue
            basename = os.path.basename(rel_file)
            name = str(node.get("qualified_name") or node.get("name") or "")
            if basename in {"main.py", "server.py", "app.py", "cli.py"} or name in {
                "main",
                "run_server",
            }:
                entry_points.append(
                    {
                        "file": rel_file,
                        "name": name,
                        "type": node.get("type", "symbol"),
                        "line": int(node.get("line", 0) or 0),
                    }
                )
        hot_files.sort(key=lambda item: item["symbol_count"], reverse=True)
        subsystems = [
            {"name": name, "files": count}
            for name, count in sorted(
                subsystem_counts.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ]
        return {
            "subsystems": subsystems,
            "hot_files": hot_files[:10],
            "entry_points": entry_points[:10],
            "file_count": len(files),
            "symbol_count": len(nodes),
        }

    @staticmethod
    def _architecture_explorer(graph: dict[str, Any]) -> dict[str, Any]:
        edges = graph.get("edges", {})
        nodes = graph.get("nodes", {})
        relation_counts: dict[str, int] = defaultdict(int)
        imports: dict[str, set[str]] = defaultdict(set)
        reverse_imports: dict[str, set[str]] = defaultdict(set)
        for edge in edges.values():
            relation = str(edge.get("relation") or "related")
            relation_counts[relation] += 1
            if relation != "imports":
                continue
            src = str(edge.get("file") or "")
            dst_node = nodes.get(str(edge.get("to") or ""), {})
            dst = str(dst_node.get("file") or "")
            if not src or not dst:
                continue
            imports[src].add(dst)
            reverse_imports[dst].add(src)
        cycles = []
        for src, targets in imports.items():
            for dst in targets:
                if src in imports.get(dst, set()):
                    cycles.append({"from": src, "to": dst})
                    if len(cycles) >= 10:
                        break
            if len(cycles) >= 10:
                break
        return {
            "relation_counts": dict(
                sorted(relation_counts.items(), key=lambda item: item[1], reverse=True)
            ),
            "import_cycles": cycles,
            "top_import_hubs": [
                {"file": file_path, "imported_by": len(sources)}
                for file_path, sources in sorted(
                    reverse_imports.items(),
                    key=lambda item: len(item[1]),
                    reverse=True,
                )[:10]
            ],
        }

    @staticmethod
    def _change_intelligence(
        graph: dict[str, Any],
        evidence: list[dict[str, Any]],
        latest_benchmark: dict[str, Any] | None,
    ) -> dict[str, Any]:
        latest_bundle = evidence[0] if evidence else None
        payload = (latest_bundle or {}).get("payload", {})
        dependency_graph = payload.get("dependency_graph", [])
        return {
            "latest_primary_file": payload.get("primary_file"),
            "impacted_entities": payload.get("entities", [])[:10],
            "dependency_edges": dependency_graph[:10],
            "benchmark_guardrail": (
                latest_benchmark.get("payload", {}).get("slo_status")
                if latest_benchmark
                else "missing"
            ),
            "graph_nodes": len(graph.get("nodes", {})),
        }

    @staticmethod
    def _research_center(research: list[dict[str, Any]]) -> dict[str, Any]:
        sources: dict[str, int] = defaultdict(int)
        for item in research:
            payload = item.get("payload", {})
            sources[str(payload.get("source") or "unknown")] += 1
        return {
            "source_counts": dict(sorted(sources.items())),
            "latest_titles": [
                str(item.get("payload", {}).get("title") or item.get("payload", {}).get("url") or "")
                for item in research[:10]
            ],
        }
