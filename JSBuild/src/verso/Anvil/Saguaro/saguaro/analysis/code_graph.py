from __future__ import annotations

import json
import os
import time
from typing import Any

from saguaro.analysis.bridge_synthesizer import BridgeSynthesizer
from saguaro.analysis.call_graph_builder import CallGraphBuilder
from saguaro.analysis.cfg_builder import CFGBuilder
from saguaro.analysis.dfg_builder import DFGBuilder
from saguaro.analysis.disparate_relations import DisparateRelationSynthesizer
from saguaro.analysis.ffi_scanner import FFIScanner
from saguaro.utils.entity_ids import entity_identity


class CodeGraph:
    """Incremental analysis graph with CFG/DFG/call-graph + FFI bridge metadata."""

    GRAPH_SCHEMA_VERSION = 1

    def __init__(self, repo_path: str, graph_path: str | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        default_path = os.path.join(self.repo_path, ".saguaro", "graph", "code_graph.json")
        self.graph_path = os.path.abspath(graph_path or default_path)

        self.cfg_builder = CFGBuilder()
        self.dfg_builder = DFGBuilder()
        self.call_graph_builder = CallGraphBuilder()
        self.ffi_scanner = FFIScanner(repo_path=self.repo_path)
        self.bridge_synthesizer = BridgeSynthesizer()
        self.disparate_relation_synthesizer = DisparateRelationSynthesizer()

    def build_incremental(
        self,
        file_paths: list[str],
        parsed_entities_by_file: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        graph = self.load_graph()
        parsed_entities = parsed_entities_by_file or {}

        changed_files = sorted({os.path.abspath(path) for path in file_paths if path})
        changed_rel = {self._safe_relpath(path) for path in changed_files}
        self._drop_files(graph, changed_rel)

        processed_files: list[str] = []
        deleted_files: list[str] = []
        for abs_path in changed_files:
            rel_file = self._safe_relpath(abs_path)
            if not os.path.exists(abs_path):
                deleted_files.append(rel_file)
                continue

            file_entities = parsed_entities.get(abs_path)
            if file_entities is None:
                file_entities = parsed_entities.get(rel_file, [])

            payload = self._build_file_payload(abs_path, rel_file, file_entities or [])
            self._merge_file_payload(graph, rel_file, payload)
            processed_files.append(rel_file)

        self._rebuild_bridge_edges(graph)
        self._prune_dangling_edges(graph)
        self._refresh_stats(graph, processed_files=processed_files, deleted_files=deleted_files)

        graph["generated_at"] = time.time()
        graph["generated_fmt"] = time.ctime(graph["generated_at"])

        self._persist(graph)
        return {
            "status": "ok",
            "graph_path": self.graph_path,
            "updated_files": len(processed_files),
            "deleted_files": len(deleted_files),
            "files": int(graph.get("stats", {}).get("files", 0)),
            "nodes": int(graph.get("stats", {}).get("nodes", 0)),
            "edges": int(graph.get("stats", {}).get("edges", 0)),
            "ffi_patterns": int(graph.get("stats", {}).get("ffi_patterns", 0)),
            "bridge_edges": int(graph.get("stats", {}).get("bridge_edges", 0)),
            "disparate_relation_edges": int(
                graph.get("stats", {}).get("disparate_relation_edges", 0)
            ),
        }

    def load_graph(self) -> dict[str, Any]:
        if not os.path.exists(self.graph_path):
            return self._empty_graph()

        try:
            with open(self.graph_path, encoding="utf-8") as handle:
                raw = json.load(handle)
        except Exception:
            return self._empty_graph()

        if not isinstance(raw, dict):
            return self._empty_graph()

        graph = self._empty_graph()
        graph.update(raw)
        for key in ("nodes", "edges", "files", "ffi_patterns", "stats"):
            if not isinstance(graph.get(key), dict):
                graph[key] = {}
        graph["version"] = int(graph.get("version", self.GRAPH_SCHEMA_VERSION) or self.GRAPH_SCHEMA_VERSION)
        graph["repo_path"] = self.repo_path
        return graph

    def _build_file_payload(
        self,
        abs_path: str,
        rel_file: str,
        entities: list[Any],
    ) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        ffi_patterns: dict[str, dict[str, Any]] = {}

        source = self._read_text(abs_path)
        name_lookup: dict[str, str] = {}
        file_node_id: str | None = None

        dependency_payload: dict[str, Any] | None = None
        for entity in entities:
            entity_type = str(getattr(entity, "type", "") or "")
            if entity_type == "dependency_graph":
                try:
                    dependency_payload = json.loads(str(getattr(entity, "content", "") or ""))
                except Exception:
                    dependency_payload = None
                continue

            node = self._entity_node(rel_file, abs_path, entity)
            if node is None:
                continue

            nodes[node["id"]] = node
            if entity_type == "file":
                file_node_id = node["id"]
            for key in {
                str(getattr(entity, "name", "") or "").strip(),
                str(node.get("name") or "").strip(),
                str(node.get("qualified_name") or "").strip(),
            }:
                if key:
                    name_lookup.setdefault(key, node["id"])

        if file_node_id is None:
            file_node_id = self._node_id(rel_file, os.path.basename(rel_file), "file", 1)
            nodes[file_node_id] = {
                "id": file_node_id,
                "name": os.path.basename(rel_file),
                "qualified_name": os.path.splitext(rel_file)[0].replace("/", "."),
                "entity_id": f"{rel_file}:{os.path.basename(rel_file)}:1",
                "type": "file",
                "file": rel_file,
                "line": 1,
                "end_line": max(1, source.count("\n") + 1),
                "source": "code_graph",
            }

        if dependency_payload:
            self._ingest_dependency_payload(
                rel_file=rel_file,
                file_node_id=file_node_id,
                dep_payload=dependency_payload,
                nodes=nodes,
                edges=edges,
                name_lookup=name_lookup,
            )

        for builder in (self.cfg_builder, self.dfg_builder, self.call_graph_builder):
            payload = builder.build(rel_file, source)
            for node in payload.get("nodes", []):
                node_id = str(node.get("id") or "")
                if not node_id:
                    continue
                merged = dict(node)
                if "file" not in merged:
                    merged["file"] = rel_file
                nodes[node_id] = merged
            for edge in payload.get("edges", []):
                edge_id = str(edge.get("id") or "")
                if not edge_id:
                    continue
                merged = dict(edge)
                merged.setdefault("file", rel_file)
                edges[edge_id] = merged

        for pattern in self.ffi_scanner.scan_file(rel_file, source):
            pattern_id = str(pattern.get("id") or "")
            if not pattern_id:
                continue
            ffi_patterns[pattern_id] = dict(pattern)
            nodes[pattern_id] = {
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
            edge_line = int(pattern.get("line") or 0)
            edge_id = self._edge_id(file_node_id, pattern_id, "ffi_detected", edge_line)
            edges[edge_id] = {
                "id": edge_id,
                "from": file_node_id,
                "to": pattern_id,
                "relation": "ffi_detected",
                "line": edge_line,
                "file": rel_file,
                "confidence": float(pattern.get("confidence") or 0.0),
                "source": "ffi_scanner",
            }

        return {
            "nodes": nodes,
            "edges": edges,
            "ffi_patterns": ffi_patterns,
        }

    def _entity_node(self, rel_file: str, abs_path: str, entity: Any) -> dict[str, Any] | None:
        name = str(getattr(entity, "name", "") or "").strip()
        entity_type = str(getattr(entity, "type", "") or "").strip()
        start_line = int(getattr(entity, "start_line", 0) or 0)
        end_line = int(getattr(entity, "end_line", start_line) or start_line)
        if not name or not entity_type:
            return None

        node_id = self._node_id(rel_file, name, entity_type, start_line)
        identity = entity_identity(
            self.repo_path,
            abs_path,
            name,
            entity_type,
            start_line,
        )
        metadata = dict(getattr(entity, "metadata", {}) or {})
        return {
            "id": node_id,
            "name": identity.get("display_name", name),
            "qualified_name": identity.get("qualified_name", name),
            "entity_id": identity.get("entity_id", f"{rel_file}:{name}:{start_line}"),
            "type": entity_type,
            "file": rel_file,
            "line": start_line,
            "end_line": end_line,
            "language": str(metadata.get("language") or ""),
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
            "source": "parser",
        }

    def _ingest_dependency_payload(
        self,
        *,
        rel_file: str,
        file_node_id: str,
        dep_payload: dict[str, Any],
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
        name_lookup: dict[str, str],
    ) -> None:
        raw_imports = dep_payload.get("imports", [])
        if isinstance(raw_imports, list):
            for raw_import in sorted({str(item).strip() for item in raw_imports if str(item).strip()}):
                import_node_id = f"external::{raw_import}"
                nodes.setdefault(
                    import_node_id,
                    {
                        "id": import_node_id,
                        "name": raw_import,
                        "qualified_name": raw_import,
                        "type": "external",
                        "file": None,
                        "line": 0,
                        "end_line": 0,
                        "source": "parser",
                    },
                )
                edge_id = self._edge_id(file_node_id, import_node_id, "imports", 1)
                edges[edge_id] = {
                    "id": edge_id,
                    "from": file_node_id,
                    "to": import_node_id,
                    "relation": "imports",
                    "line": 1,
                    "file": rel_file,
                    "source": "parser",
                }

        raw_internal = dep_payload.get("internal_edges", [])
        if not isinstance(raw_internal, list):
            return

        for item in raw_internal:
            if not isinstance(item, dict):
                continue
            src_name = str(item.get("from") or "").strip()
            dst_name = str(item.get("to") or "").strip()
            if not src_name or not dst_name:
                continue

            src_id = name_lookup.get(src_name)
            if src_id is None:
                src_id = self._node_id(rel_file, src_name, "symbol", 0)
                nodes.setdefault(
                    src_id,
                    {
                        "id": src_id,
                        "name": src_name,
                        "qualified_name": src_name,
                        "entity_id": f"{rel_file}:{src_name}:0",
                        "type": "symbol",
                        "file": rel_file,
                        "line": 0,
                        "end_line": 0,
                        "source": "parser",
                    },
                )
                name_lookup[src_name] = src_id

            dst_id = name_lookup.get(dst_name)
            if dst_id is None:
                dst_id = self._node_id(rel_file, dst_name, "symbol", 0)
                nodes.setdefault(
                    dst_id,
                    {
                        "id": dst_id,
                        "name": dst_name,
                        "qualified_name": dst_name,
                        "entity_id": f"{rel_file}:{dst_name}:0",
                        "type": "symbol",
                        "file": rel_file,
                        "line": 0,
                        "end_line": 0,
                        "source": "parser",
                    },
                )
                name_lookup[dst_name] = dst_id

            line = int(item.get("line", 1) or 1)
            relation = str(item.get("relation", "related") or "related")
            edge_id = self._edge_id(src_id, dst_id, relation, line)
            edges[edge_id] = {
                "id": edge_id,
                "from": src_id,
                "to": dst_id,
                "relation": relation,
                "line": line,
                "file": rel_file,
                "source": "parser",
            }

    def _merge_file_payload(
        self,
        graph: dict[str, Any],
        rel_file: str,
        payload: dict[str, Any],
    ) -> None:
        graph.setdefault("nodes", {})
        graph.setdefault("edges", {})
        graph.setdefault("files", {})
        graph.setdefault("ffi_patterns", {})

        node_ids = sorted(payload.get("nodes", {}).keys())
        edge_ids = sorted(payload.get("edges", {}).keys())
        ffi_ids = sorted(payload.get("ffi_patterns", {}).keys())

        for node_id in node_ids:
            graph["nodes"][node_id] = payload["nodes"][node_id]
        for edge_id in edge_ids:
            graph["edges"][edge_id] = payload["edges"][edge_id]
        for ffi_id in ffi_ids:
            graph["ffi_patterns"][ffi_id] = payload["ffi_patterns"][ffi_id]

        graph["files"][rel_file] = {
            "nodes": node_ids,
            "edges": edge_ids,
            "ffi_patterns": ffi_ids,
        }

    def _drop_files(self, graph: dict[str, Any], rel_files: set[str]) -> None:
        if not rel_files:
            return

        nodes = graph.setdefault("nodes", {})
        edges = graph.setdefault("edges", {})
        files = graph.setdefault("files", {})
        ffi_patterns = graph.setdefault("ffi_patterns", {})

        removed_nodes: set[str] = set()
        removed_edges: set[str] = set()

        for rel_file in sorted(rel_files):
            entry = files.pop(rel_file, {})
            for node_id in entry.get("nodes", []):
                removed_nodes.add(str(node_id))
            for edge_id in entry.get("edges", []):
                removed_edges.add(str(edge_id))
            for ffi_id in entry.get("ffi_patterns", []):
                ffi_patterns.pop(str(ffi_id), None)

        for node_id in removed_nodes:
            nodes.pop(node_id, None)

        for edge_id in removed_edges:
            edges.pop(edge_id, None)

        for edge_id, edge in list(edges.items()):
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            edge_file = str(edge.get("file") or "")
            if src in removed_nodes or dst in removed_nodes or edge_file in rel_files:
                edges.pop(edge_id, None)

        for ffi_id, pattern in list(ffi_patterns.items()):
            if str(pattern.get("file") or "") in rel_files:
                ffi_patterns.pop(ffi_id, None)

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

    def _prune_dangling_edges(self, graph: dict[str, Any]) -> None:
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", {})
        if not isinstance(nodes, dict) or not isinstance(edges, dict):
            return

        for edge_id, edge in list(edges.items()):
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src and src.startswith("external::"):
                src_exists = True
            else:
                src_exists = src in nodes
            if dst and dst.startswith("external::"):
                dst_exists = True
            else:
                dst_exists = dst in nodes
            if not src_exists or not dst_exists:
                edges.pop(edge_id, None)

    def _refresh_stats(
        self,
        graph: dict[str, Any],
        *,
        processed_files: list[str],
        deleted_files: list[str],
    ) -> None:
        edges = list(graph.get("edges", {}).values())
        graph["stats"] = {
            "files": len(graph.get("files", {})),
            "nodes": len(graph.get("nodes", {})),
            "edges": len(graph.get("edges", {})),
            "cfg_edges": sum(1 for edge in edges if str(edge.get("relation") or "").startswith("cfg_")),
            "dfg_edges": sum(1 for edge in edges if str(edge.get("relation") or "").startswith("dfg_")),
            "call_edges": sum(1 for edge in edges if str(edge.get("relation") or "") == "calls"),
            "ffi_patterns": len(graph.get("ffi_patterns", {})),
            "bridge_edges": sum(1 for edge in edges if str(edge.get("relation") or "") == "ffi_bridge"),
            "disparate_relation_edges": sum(
                1
                for edge in edges
                if str(edge.get("relation") or "") in self.disparate_relation_synthesizer.relation_families
            ),
            "incremental": True,
            "updated_files": len(processed_files),
            "deleted_files": len(deleted_files),
        }

    def _persist(self, graph: dict[str, Any]) -> None:
        graph["version"] = self.GRAPH_SCHEMA_VERSION
        graph["repo_path"] = self.repo_path
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        with open(self.graph_path, "w", encoding="utf-8") as handle:
            json.dump(graph, handle, indent=2, sort_keys=True)

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
            "stats": {},
        }

    @staticmethod
    def _node_id(rel_file: str, name: str, kind: str, line: int) -> str:
        return f"{rel_file}::{name}::{kind}::{int(line)}"

    @staticmethod
    def _edge_id(src: str, dst: str, relation: str, line: int) -> str:
        return f"{src}->{dst}::{relation}::{int(line)}"

    def _safe_relpath(self, path: str) -> str:
        abs_path = os.path.abspath(path)
        try:
            rel = os.path.relpath(abs_path, self.repo_path)
        except ValueError:
            rel = os.path.basename(abs_path)
        rel = rel.replace("\\", "/")
        if rel.startswith("../"):
            return os.path.basename(abs_path)
        return rel

    @staticmethod
    def _read_text(path: str) -> str:
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        except Exception:
            return ""
