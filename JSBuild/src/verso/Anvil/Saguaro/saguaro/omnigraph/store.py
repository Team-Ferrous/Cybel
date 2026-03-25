"""Persist and query typed omni-graph state."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from saguaro.analysis.bridge_synthesizer import BridgeSynthesizer
from saguaro.analysis.disparate_relations import DisparateRelationSynthesizer
from saguaro.analysis.ffi_scanner import FFIScanner
from saguaro.omnigraph.model import OmniNode, OmniRelation

_TEMPLATE_LANGS = {"jinja", "django_template", "erb", "blade", "ejs", "handlebars", "pug", "html"}
_FRONTEND_LANGS = {"javascript", "typescript", "vue", "svelte", "mdx"}
_OMNIGRAPH_FILE_LANGS = _TEMPLATE_LANGS.union(_FRONTEND_LANGS).union(
    {
        "python",
        "c",
        "cpp",
        "rust",
        "go",
        "java",
        "csharp",
        "kotlin",
        "swift",
        "objective_c",
        "shell",
        "powershell",
        "hcl",
        "sql",
        "proto",
        "graphql",
        "solidity",
    }
)


class OmniGraphStore:
    """Build a typed cross-artifact graph from Saguaro state."""

    def __init__(self, repo_path: str, graph_service: Any | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.graph_service = graph_service
        self.base_dir = os.path.join(self.repo_path, ".saguaro", "omnigraph")
        self.graph_path = os.path.join(self.base_dir, "graph.json")
        self.disparate_relation_synthesizer = DisparateRelationSynthesizer()

    def build(self, traceability_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build and persist the omni-graph."""
        os.makedirs(self.base_dir, exist_ok=True)
        graph = self.graph_service.load_graph() if self.graph_service else {}
        generation = str(
            traceability_payload.get("generation_id")
            if traceability_payload
            else f"omnigraph-{int(time.time())}"
        )
        nodes: dict[str, OmniNode] = {}
        relations: dict[str, OmniRelation] = {}

        for item in list((graph.get("nodes") or {}).values()):
            node = OmniNode(
                id=str(item.get("id") or ""),
                type=str(item.get("type") or item.get("kind") or "symbol"),
                label=str(item.get("qualified_name") or item.get("name") or item.get("id") or ""),
                file=str(item.get("file") or ""),
                metadata={"line": item.get("line"), "language": item.get("language")},
            )
            nodes[node.id] = node

        for node in self._filesystem_nodes():
            nodes.setdefault(node.id, node)

        if traceability_payload:
            for requirement in traceability_payload.get("requirements", []):
                node = OmniNode(
                    id=requirement["id"],
                    type="requirement",
                    label=requirement.get("text_norm") or requirement.get("normalized_statement") or requirement.get("statement") or requirement["id"],
                    file=requirement["file"],
                    metadata={
                        "modality": requirement.get("modality"),
                        "strength": requirement.get("strength"),
                        "heading_path": requirement.get("heading_path"),
                    },
                )
                nodes[node.id] = node
            for record in traceability_payload.get("records", []):
                if "artifact_id" in record:
                    relation = OmniRelation(
                        id=record["id"],
                        src_type="requirement",
                        src_id=record["requirement_id"],
                        dst_type=record.get("artifact_type", "symbol"),
                        dst_id=record["artifact_id"],
                        relation_type=record.get("relation_type", "supports"),
                        evidence_types=list(record.get("evidence_types") or []),
                        confidence=float(record.get("confidence", 0.0) or 0.0),
                        verified=bool(record.get("verification_state") == "verified"),
                        drift_state="fresh",
                        generation_id=generation,
                        notes=list(record.get("notes") or []),
                        uncertainty=self._uncertainty_from_record(record),
                    )
                    relations[relation.id] = relation
                    continue
                relation_defs = self._relations_from_trace_record(record, generation)
                for relation in relation_defs:
                    relations[relation.id] = relation

        for boundary in self._ffi_boundaries():
            node = OmniNode(
                id=boundary["id"],
                type="ffi_boundary",
                label=boundary.get("boundary_type") or boundary.get("mechanism") or boundary["id"],
                file=str(boundary.get("file") or ""),
                metadata=dict(boundary),
            )
            nodes[node.id] = node

        for relation in self._template_relations(nodes.values(), generation):
            relations[relation.id] = relation
        for relation in self._bridge_relations(generation):
            relations[relation.id] = relation
        for edge in list((graph.get("edges") or {}).values()):
            relation_type = str(edge.get("relation") or "")
            if relation_type not in self.disparate_relation_synthesizer.relation_families:
                continue
            relation = OmniRelation(
                id=str(edge.get("id") or ""),
                src_type=str((graph.get("nodes") or {}).get(str(edge.get("from") or ""), {}).get("type") or "symbol"),
                src_id=str(edge.get("from") or ""),
                dst_type=str((graph.get("nodes") or {}).get(str(edge.get("to") or ""), {}).get("type") or "symbol"),
                dst_id=str(edge.get("to") or ""),
                relation_type=relation_type,
                evidence_types=list(edge.get("evidence_mix") or []),
                confidence=float(edge.get("confidence", 0.0) or 0.0),
                verified=float(edge.get("confidence", 0.0) or 0.0) >= 0.58,
                drift_state="fresh",
                generation_id=str(edge.get("generation_id") or generation),
                notes=[str(edge.get("reason") or "")],
                uncertainty={
                    "source": "disparate_relation_synthesizer",
                    "parser_uncertainty": str(edge.get("parser_uncertainty") or "unknown"),
                },
                evidence_spans=list(edge.get("evidence_spans") or []),
                evidence_mix=list(edge.get("evidence_mix") or []),
                confidence_components=dict(edge.get("confidence_components") or {}),
                parser_uncertainty=str(edge.get("parser_uncertainty") or "unknown"),
                counterevidence=list(edge.get("counterevidence") or []),
            )
            relations[relation.id] = relation

        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_id": generation,
            "nodes": {key: value.to_dict() for key, value in nodes.items()},
            "relations": {key: value.to_dict() for key, value in relations.items()},
            "summary": {
                "node_count": len(nodes),
                "relation_count": len(relations),
                "requirement_count": sum(1 for item in nodes.values() if item.type == "requirement"),
                "bridge_count": sum(1 for item in relations.values() if item.relation_type == "bridged_by"),
                "disparate_relation_count": sum(
                    1
                    for item in relations.values()
                    if item.relation_type in self.disparate_relation_synthesizer.relation_families
                ),
            },
        }
        with open(self.graph_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return payload

    def explain(self, requirement_id: str) -> dict[str, Any]:
        """Explain a requirement neighborhood from persisted state."""
        graph = self.load()
        if requirement_id not in graph.get("nodes", {}):
            return {"status": "missing", "requirement_id": requirement_id}
        relations = [
            item
            for item in graph.get("relations", {}).values()
            if item.get("src_id") == requirement_id or item.get("dst_id") == requirement_id
        ]
        related_ids = {requirement_id}
        for relation in relations:
            related_ids.add(str(relation.get("src_id") or ""))
            related_ids.add(str(relation.get("dst_id") or ""))
        nodes = [graph["nodes"][item] for item in related_ids if item in graph.get("nodes", {})]
        return {"status": "ok", "nodes": nodes, "relations": relations, "count": len(relations)}

    def find_equation(self, query: str) -> dict[str, Any]:
        """Find requirement/equation nodes related to a query."""
        needle = query.lower()
        graph = self.load()
        matches = [
            node
            for node in graph.get("nodes", {}).values()
            if needle in json.dumps(node, sort_keys=True).lower()
        ]
        return {"status": "ok", "query": query, "count": len(matches), "matches": matches[:20]}

    def diff(self) -> dict[str, Any]:
        """A placeholder diff over the latest persisted state."""
        graph = self.load()
        return {
            "status": "ok",
            "generation_id": graph.get("generation_id"),
            "summary": graph.get("summary", {}),
        }

    def gaps(self, modality: str | None = None) -> dict[str, Any]:
        """Return weak requirement relations."""
        graph = self.load()
        nodes = graph.get("nodes", {})
        relations = list(graph.get("relations", {}).values())
        weak = []
        for node in nodes.values():
            if node.get("type") != "requirement":
                continue
            if modality and node.get("metadata", {}).get("modality") != modality:
                continue
            node_relations = [item for item in relations if item.get("src_id") == node["id"]]
            if not node_relations or max(float(item.get("confidence", 0.0) or 0.0) for item in node_relations) < 0.58:
                weak.append({"requirement": node, "relations": node_relations})
        return {"status": "ok", "count": len(weak), "gaps": weak}

    def load(self) -> dict[str, Any]:
        """Load persisted state."""
        if not os.path.exists(self.graph_path):
            return {"nodes": {}, "relations": {}, "summary": {}}
        with open(self.graph_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _ffi_boundaries(self) -> list[dict[str, Any]]:
        scanner = FFIScanner(repo_path=self.repo_path)
        findings = []
        for root, files in self._walk_repo_files():
            for filename in files:
                rel = os.path.relpath(os.path.join(root, filename), self.repo_path).replace("\\", "/")
                try:
                    with open(os.path.join(root, filename), encoding="utf-8", errors="ignore") as handle:
                        source = handle.read()
                    findings.extend(scanner.scan_file(rel, source))
                except OSError:
                    continue
        return findings

    def _bridge_relations(self, generation: str) -> list[OmniRelation]:
        bridges = BridgeSynthesizer().synthesize(self._ffi_boundaries())
        relations = []
        for item in bridges:
            relations.append(
                OmniRelation(
                    id=str(item["id"]),
                    src_type="symbol",
                    src_id=str(item.get("from") or ""),
                    dst_type="ffi_boundary",
                    dst_id=str(item.get("to") or ""),
                    relation_type="bridged_by",
                    evidence_types=["trace", "graph"],
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    verified=float(item.get("confidence", 0.0) or 0.0) >= 0.7,
                    drift_state="fresh",
                    generation_id=generation,
                    notes=[str(item.get("reason") or "")],
                    uncertainty={"source": "ffi_bridge"},
                )
            )
        return relations

    def _template_relations(
        self, nodes: Any, generation: str
    ) -> list[OmniRelation]:
        file_nodes = [item for item in nodes if getattr(item, "file", None)]
        by_stem: dict[str, list[OmniNode]] = {}
        for node in file_nodes:
            stem = os.path.splitext(os.path.basename(node.file))[0]
            by_stem.setdefault(stem, []).append(node)
        relations = []
        for siblings in by_stem.values():
            if len(siblings) < 2:
                continue
            for left in siblings:
                for right in siblings:
                    if left.id == right.id:
                        continue
                    left_lang = self._infer_language(left.file)
                    right_lang = self._infer_language(right.file)
                    if {left_lang, right_lang}.intersection(_TEMPLATE_LANGS) and {left_lang, right_lang}.intersection(_FRONTEND_LANGS):
                        rel_id = f"{left.id}->{right.id}::bridged_by"
                        relations.append(
                            OmniRelation(
                                id=rel_id,
                                src_type=left.type,
                                src_id=left.id,
                                dst_type=right.type,
                                dst_id=right.id,
                                relation_type="bridged_by",
                                evidence_types=["lexical", "path"],
                                confidence=0.71,
                                verified=True,
                                drift_state="fresh",
                                generation_id=generation,
                                notes=["same_stem_template_frontend_pair"],
                                uncertainty={"parser": "heuristic"},
                            )
                        )
        unique: dict[str, OmniRelation] = {}
        for relation in relations:
            unique.setdefault(relation.id, relation)
        return list(unique.values())

    def _filesystem_nodes(self) -> list[OmniNode]:
        nodes: list[OmniNode] = []
        for root, files in self._walk_repo_files():
            for filename in files:
                rel = os.path.relpath(os.path.join(root, filename), self.repo_path).replace("\\", "/")
                lang = self._infer_language(rel)
                if lang not in _OMNIGRAPH_FILE_LANGS:
                    continue
                nodes.append(
                    OmniNode(
                        id=f"file::{rel}",
                        type="file",
                        label=rel,
                        file=rel,
                        metadata={"language": lang, "source": "filesystem_scan"},
                    )
                )
        return nodes

    def _walk_repo_files(self) -> list[tuple[str, list[str]]]:
        rows: list[tuple[str, list[str]]] = []
        for root, _, files in os.walk(self.repo_path):
            if any(
                part in {".git", ".saguaro", ".anvil", "venv", ".venv", "__pycache__", "Saguaro"}
                for part in root.split(os.sep)
            ):
                continue
            rows.append((root, files))
        return rows

    @staticmethod
    def _infer_language(file_path: str) -> str:
        lower = file_path.lower()
        if lower.endswith((".j2", ".jinja", ".jinja2")):
            return "jinja"
        if lower.endswith((".html", ".htm")):
            return "html"
        if lower.endswith((".jsx", ".js")):
            return "javascript"
        if lower.endswith((".tsx", ".ts")):
            return "typescript"
        if lower.endswith(".vue"):
            return "vue"
        if lower.endswith(".svelte"):
            return "svelte"
        if lower.endswith(".mdx"):
            return "mdx"
        return os.path.splitext(lower)[1].lstrip(".")

    @staticmethod
    def _uncertainty_from_record(record: dict[str, Any]) -> dict[str, Any]:
        confidence = float(record.get("confidence", 0.0) or 0.0)
        return {
            "parser_uncertainty": "low" if confidence >= 0.74 else "medium",
            "witness_scarcity": "present" if "test" not in list(record.get("evidence_types") or []) else "low",
            "backend_mismatch": False,
        }

    @staticmethod
    def _relations_from_trace_record(
        record: dict[str, Any], generation: str
    ) -> list[OmniRelation]:
        relations: list[OmniRelation] = []
        requirement_id = str(record.get("requirement_id") or "")
        for relation_type, refs, dst_type, base_confidence in (
            ("implements", list(record.get("code_refs") or []), "file", 0.76),
            ("witnessed_by", list(record.get("test_refs") or []), "test_case", 0.88),
            ("supports", list(record.get("graph_refs") or []), "symbol", 0.62),
            ("configured_by", list(record.get("verification_refs") or []), "config_key", 0.58),
        ):
            for ref in refs:
                dst_id = str(ref)
                relations.append(
                    OmniRelation(
                        id=f"{requirement_id}->{dst_id}::{relation_type}",
                        src_type="requirement",
                        src_id=requirement_id,
                        dst_type=dst_type,
                        dst_id=dst_id,
                        relation_type=relation_type,
                        evidence_types=["lexical", "graph"],
                        confidence=base_confidence,
                        verified=base_confidence >= 0.74,
                        drift_state="fresh",
                        generation_id=generation,
                        notes=[],
                        uncertainty={"parser": "heuristic"},
                    )
                )
        return relations
