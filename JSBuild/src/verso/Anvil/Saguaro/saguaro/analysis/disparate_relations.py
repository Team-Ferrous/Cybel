from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import os
import re
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
_NOISE_ROLE_TAGS = {"config_surface", "doc", "example_surface", "module_init", "test_harness"}
_RELATION_FAMILIES = {
    "adaptation_candidate",
    "analogous_to",
    "evaluation_analogue",
    "native_upgrade_path",
    "port_program_candidate",
    "subsystem_analogue",
}
_ROLE_HINTS = {
    "adapter": {"adapter", "service"},
    "adapters": {"adapter", "service"},
    "artifact": {"artifact", "report_surface"},
    "artifacts": {"artifact", "report_surface"},
    "cli": {"cli", "entrypoint", "orchestration"},
    "command": {"cli", "entrypoint", "orchestration"},
    "commands": {"cli", "entrypoint", "orchestration"},
    "context": {"state", "session"},
    "core": {"core_runtime", "service"},
    "engine": {"core_runtime", "service"},
    "framework": {"adapter", "framework"},
    "frameworks": {"adapter", "framework"},
    "output": {"artifact", "report_surface"},
    "outputs": {"artifact", "report_surface"},
    "pipeline": {"pipeline", "service"},
    "plugin": {"plugin", "registry"},
    "plugins": {"plugin", "registry"},
    "registry": {"plugin", "registry"},
    "registries": {"plugin", "registry"},
    "report": {"artifact", "report_surface"},
    "reporting": {"artifact", "report_surface"},
    "runtime": {"core_runtime", "service"},
    "service": {"core_runtime", "service"},
    "session": {"session", "state"},
    "state": {"session", "state"},
    "target": {"target"},
    "targets": {"target"},
}
_FEATURE_HINTS = {
    "adapter": "framework_adapter",
    "artifact": "artifact_output",
    "attack": "attack_orchestration",
    "coverage": "diagnostics",
    "dataflow": "dataflow",
    "diagnostic": "diagnostics",
    "diagnostics": "diagnostics",
    "evaluat": "evaluation_pipeline",
    "extractor": "extractor",
    "ffi": "native_integration",
    "framework": "framework_adapter",
    "graph": "dataflow",
    "index": "query_engine",
    "model": "evaluation_pipeline",
    "optim": "optimization",
    "pipeline": "evaluation_pipeline",
    "query": "query_engine",
    "registry": "target_registry",
    "report": "reporting",
    "security": "security_analysis",
    "session": "runtime_state",
    "state": "runtime_state",
    "target": "target_registry",
}


@dataclass(slots=True)
class DisparateRelationEvidence:
    source_path: str
    target_path: str
    source_line: int = 1
    target_line: int = 1
    evidence_mix: list[str] = field(default_factory=list)
    shared_role_tags: list[str] = field(default_factory=list)
    shared_feature_families: list[str] = field(default_factory=list)
    shared_tokens: list[str] = field(default_factory=list)
    shared_boundary_markers: list[str] = field(default_factory=list)
    confidence_components: dict[str, float] = field(default_factory=dict)
    parser_uncertainty: str = "unknown"
    counterevidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DisparateRelationSynthesizer:
    """Synthesize native disparate relations from typed graph or record facts."""

    relation_families = frozenset(_RELATION_FAMILIES)

    def synthesize_graph_edges(
        self,
        graph: dict[str, Any],
        *,
        generation_id: str,
        limit_per_record: int = 4,
        min_confidence: float = 0.36,
    ) -> list[dict[str, Any]]:
        records = self._graph_records(graph)
        relations: dict[str, dict[str, Any]] = {}
        for left_index, left in enumerate(records):
            ranked: list[dict[str, Any]] = []
            for right_index in range(left_index + 1, len(records)):
                relation = self._score_pair(left, records[right_index])
                if relation is None or float(relation["confidence"]) < float(min_confidence):
                    continue
                ranked.append(relation)
            ranked.sort(
                key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("target_path") or ""))
            )
            for relation in ranked[: max(1, int(limit_per_record))]:
                edge_id = str(relation["id"])
                relations.setdefault(edge_id, relation)
        return [relations[key] for key in sorted(relations)]

    def synthesize_cross_scope(
        self,
        *,
        source_records: list[dict[str, Any]],
        target_records: list[dict[str, Any]],
        generation_id: str,
        top_k: int = 32,
        min_confidence: float = 0.34,
    ) -> list[dict[str, Any]]:
        source_rows = [self._normalize_record(item, fallback_scope="source") for item in source_records]
        target_rows = [self._normalize_record(item, fallback_scope="target") for item in target_records]
        ranked: list[dict[str, Any]] = []
        for source_index, source_row in enumerate(source_rows):
            candidates: list[dict[str, Any]] = []
            for target_index, target_row in enumerate(target_rows):
                relation = self._score_pair(
                    source_row,
                    target_row,
                    generation_id=generation_id,
                    source_index=source_index,
                    target_index=target_index,
                    source_scope="candidate",
                    target_scope="target",
                )
                if relation is None or float(relation["confidence"]) < float(min_confidence):
                    continue
                candidates.append(relation)
            candidates.sort(
                key=lambda item: (-float(item.get("confidence") or 0.0), str(item.get("target_path") or ""))
            )
            ranked.extend(candidates[: max(1, min(8, int(top_k)))])
        ranked.sort(
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                str(item.get("source_path") or ""),
                str(item.get("target_path") or ""),
            )
        )
        return ranked[: max(1, int(top_k or 1))]

    def relation_summary(
        self,
        graph: dict[str, Any],
        *,
        relation: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        edges = list((graph.get("edges") or {}).values())
        selected = [
            edge
            for edge in edges
            if str(edge.get("relation") or "") in self.relation_families
            and (not relation or str(edge.get("relation") or "") == str(relation))
        ]
        selected.sort(
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                str(item.get("file") or ""),
                str(item.get("id") or ""),
            )
        )
        return {
            "status": "ok",
            "count": len(selected[: max(1, int(limit or 1))]),
            "total_count": len(selected),
            "relations": selected[: max(1, int(limit or 1))],
            "families": sorted({str(item.get("relation") or "") for item in selected}),
        }

    def _graph_records(self, graph: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = graph.get("nodes") or {}
        files = graph.get("files") or {}
        records: list[dict[str, Any]] = []
        for rel_file, entry in sorted(files.items()):
            if not isinstance(entry, dict):
                continue
            record_nodes = [
                nodes.get(node_id)
                for node_id in list(entry.get("nodes") or [])
                if isinstance(nodes.get(node_id), dict)
            ]
            if not record_nodes:
                continue
            file_node = next(
                (node for node in record_nodes if str(node.get("type") or "") == "file"),
                record_nodes[0],
            )
            role_tags: set[str] = set()
            feature_families: set[str] = set()
            boundary_markers: set[str] = set()
            terms: set[str] = set()
            symbol_kinds: set[str] = set()
            parser_uncertainty = "low"
            for node in record_nodes:
                role_tags.update(str(item) for item in list(node.get("role_tags") or []) if str(item))
                feature_families.update(
                    str(item) for item in list(node.get("feature_families") or []) if str(item)
                )
                boundary_markers.update(
                    str(item) for item in list(node.get("boundary_markers") or []) if str(item)
                )
                terms.update(str(item) for item in list(node.get("terms") or []) if str(item))
                symbol_kinds.add(str(node.get("type") or ""))
                node_uncertainty = str(node.get("parser_uncertainty") or "low")
                if node_uncertainty == "high":
                    parser_uncertainty = "high"
                elif node_uncertainty == "medium" and parser_uncertainty != "high":
                    parser_uncertainty = "medium"
            records.append(
                self._normalize_record(
                    {
                        "node_id": str(file_node.get("id") or ""),
                        "path": str(rel_file),
                        "file": str(rel_file),
                        "line": int(file_node.get("line") or 1),
                        "language": str(file_node.get("language") or ""),
                        "role_tags": sorted(role_tags),
                        "feature_families": sorted(feature_families),
                        "boundary_markers": sorted(boundary_markers),
                        "terms": sorted(terms),
                        "symbol_kinds": sorted(kind for kind in symbol_kinds if kind and kind != "file"),
                        "signature_fingerprint": str(file_node.get("signature_fingerprint") or ""),
                        "structural_fingerprint": str(file_node.get("structural_fingerprint") or ""),
                        "parser_uncertainty": parser_uncertainty,
                    },
                    fallback_scope="graph",
                )
            )
        return records

    def _normalize_record(self, raw: dict[str, Any], *, fallback_scope: str) -> dict[str, Any]:
        path = str(raw.get("path") or raw.get("file") or "").replace("\\", "/")
        tokens = set(str(item).lower() for item in list(raw.get("terms") or []) if str(item))
        tokens.update(self._path_tokens(path))
        role_tags = {
            str(item)
            for item in list(raw.get("role_tags") or raw.get("shared_role_tags") or [])
            if str(item)
        }
        if not role_tags:
            role_tags.update(self._role_tags_for_path(path))
        feature_families = {
            str(item)
            for item in list(raw.get("feature_families") or raw.get("shared_feature_families") or [])
            if str(item)
        }
        if not feature_families:
            feature_families.update(self._feature_families_for_terms(tokens | role_tags))
        boundary_markers = {
            str(item)
            for item in list(raw.get("boundary_markers") or raw.get("shared_boundary_markers") or [])
            if str(item)
        }
        symbol_kinds = {str(item) for item in list(raw.get("symbol_kinds") or []) if str(item)}
        signature_fingerprint = str(raw.get("signature_fingerprint") or "").strip()
        if not signature_fingerprint:
            signature_fingerprint = self._stable_fingerprint(
                "sig",
                [path, *sorted(symbol_kinds), *sorted(role_tags), *sorted(feature_families)],
            )
        structural_fingerprint = str(raw.get("structural_fingerprint") or "").strip()
        if not structural_fingerprint:
            structural_fingerprint = self._stable_fingerprint(
                "struct",
                [path, *sorted(feature_families), *sorted(boundary_markers)],
            )
        language = str(raw.get("language") or self._language_for_path(path))
        node_id = str(raw.get("node_id") or f"file::{path}")
        return {
            "node_id": node_id,
            "path": path,
            "line": int(raw.get("line") or 1),
            "language": language,
            "role_tags": sorted(role_tags),
            "feature_families": sorted(feature_families),
            "boundary_markers": sorted(boundary_markers),
            "terms": sorted(tokens),
            "symbol_kinds": sorted(symbol_kinds),
            "signature_fingerprint": signature_fingerprint,
            "structural_fingerprint": structural_fingerprint,
            "subsystem": self._subsystem_for_path(path, fallback=fallback_scope),
            "parser_uncertainty": str(raw.get("parser_uncertainty") or "low"),
            "tags": list(raw.get("tags") or []),
        }

    def _score_pair(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        generation_id: str = "generated",
        source_index: int | None = None,
        target_index: int | None = None,
        source_scope: str = "graph",
        target_scope: str = "graph",
    ) -> dict[str, Any] | None:
        if not left.get("path") or not right.get("path") or left.get("path") == right.get("path"):
            return None

        shared_roles = sorted(set(left.get("role_tags") or []) & set(right.get("role_tags") or []))
        shared_features = sorted(
            set(left.get("feature_families") or []) & set(right.get("feature_families") or [])
        )
        shared_tokens = sorted(set(left.get("terms") or []) & set(right.get("terms") or []))
        shared_boundary_markers = sorted(
            set(left.get("boundary_markers") or []) & set(right.get("boundary_markers") or [])
        )
        same_signature = (
            str(left.get("signature_fingerprint") or "")
            and str(left.get("signature_fingerprint") or "") == str(right.get("signature_fingerprint") or "")
        )
        same_structure = (
            str(left.get("structural_fingerprint") or "")
            and str(left.get("structural_fingerprint") or "")
            == str(right.get("structural_fingerprint") or "")
        )
        cross_subsystem = str(left.get("subsystem") or "") != str(right.get("subsystem") or "")
        evidence_mix: list[str] = []
        if shared_roles:
            evidence_mix.append("role_tags")
        if shared_features:
            evidence_mix.append("feature_families")
        if shared_tokens:
            evidence_mix.append("path_tokens")
        if shared_boundary_markers:
            evidence_mix.append("boundary_markers")
        if same_signature:
            evidence_mix.append("signature_fingerprint")
        if same_structure:
            evidence_mix.append("structural_fingerprint")
        if not evidence_mix:
            return None

        noise = 0.0
        counterevidence: list[str] = []
        combined_roles = set(left.get("role_tags") or []) | set(right.get("role_tags") or [])
        if combined_roles & _NOISE_ROLE_TAGS:
            noise += 0.12
            counterevidence.append("noise_surface_detected")
        if not cross_subsystem:
            noise += 0.08
            counterevidence.append("same_subsystem")

        confidence_components = {
            "role_overlap": round(min(0.28, 0.08 * len(shared_roles)), 3),
            "feature_overlap": round(min(0.3, 0.1 * len(shared_features)), 3),
            "token_overlap": round(min(0.14, 0.02 * len(shared_tokens)), 3),
            "boundary_overlap": round(min(0.14, 0.06 * len(shared_boundary_markers)), 3),
            "signature_match": 0.14 if same_signature else 0.0,
            "structural_match": 0.12 if same_structure else 0.0,
            "cross_subsystem": 0.08 if cross_subsystem else 0.0,
            "noise_penalty": round(-noise, 3),
        }
        confidence = round(
            max(
                0.0,
                min(0.98, 0.12 + sum(confidence_components.values())),
            ),
            3,
        )
        if confidence < 0.28:
            return None

        relation = self._relation_family(left=left, right=right, shared_roles=shared_roles, shared_features=shared_features)
        evidence = DisparateRelationEvidence(
            source_path=str(left.get("path") or ""),
            target_path=str(right.get("path") or ""),
            source_line=int(left.get("line") or 1),
            target_line=int(right.get("line") or 1),
            evidence_mix=evidence_mix,
            shared_role_tags=shared_roles,
            shared_feature_families=shared_features,
            shared_tokens=shared_tokens[:16],
            shared_boundary_markers=shared_boundary_markers[:8],
            confidence_components={key: float(value) for key, value in confidence_components.items()},
            parser_uncertainty=self._merge_uncertainty(
                str(left.get("parser_uncertainty") or "low"),
                str(right.get("parser_uncertainty") or "low"),
            ),
            counterevidence=counterevidence,
        )
        source_id = str(left.get("node_id") or f"file::{left.get('path')}")
        target_id = str(right.get("node_id") or f"file::{right.get('path')}")
        edge_id = f"{source_id}->{target_id}::{relation}"
        return {
            "id": edge_id,
            "from": source_id,
            "to": target_id,
            "relation": relation,
            "file": str(left.get("path") or ""),
            "line": int(left.get("line") or 1),
            "source_path": str(left.get("path") or ""),
            "target_path": str(right.get("path") or ""),
            "source_index": source_index,
            "target_index": target_index,
            "source_scope": source_scope,
            "target_scope": target_scope,
            "confidence": confidence,
            "reason": self._reason_text(relation, evidence),
            "generation_id": generation_id,
            "source": "disparate_relation_synthesizer",
            "evidence_spans": [
                {
                    "source_path": evidence.source_path,
                    "target_path": evidence.target_path,
                    "source_line": evidence.source_line,
                    "target_line": evidence.target_line,
                }
            ],
            "evidence_mix": list(evidence.evidence_mix),
            "shared_role_tags": list(evidence.shared_role_tags),
            "shared_feature_families": list(evidence.shared_feature_families),
            "shared_tokens": list(evidence.shared_tokens),
            "shared_boundary_markers": list(evidence.shared_boundary_markers),
            "confidence_components": dict(evidence.confidence_components),
            "parser_uncertainty": evidence.parser_uncertainty,
            "counterevidence": list(evidence.counterevidence),
        }

    @staticmethod
    def _merge_uncertainty(left: str, right: str) -> str:
        levels = {"low": 0, "medium": 1, "high": 2}
        left_level = levels.get(left, 1)
        right_level = levels.get(right, 1)
        return {0: "low", 1: "medium", 2: "high"}[max(left_level, right_level)]

    def _relation_family(
        self,
        *,
        left: dict[str, Any],
        right: dict[str, Any],
        shared_roles: list[str],
        shared_features: list[str],
    ) -> str:
        left_tags = set(left.get("tags") or []) | set(left.get("role_tags") or [])
        right_tags = set(right.get("tags") or []) | set(right.get("role_tags") or [])
        if "evaluation_pipeline" in shared_features or {"pipeline", "model"} & (left_tags | right_tags):
            return "evaluation_analogue"
        if {"native"} & left_tags.symmetric_difference(right_tags):
            return "native_upgrade_path"
        if str(left.get("language") or "") != str(right.get("language") or "") and shared_features:
            return "adaptation_candidate"
        if shared_roles and str(left.get("subsystem") or "") != str(right.get("subsystem") or ""):
            return "subsystem_analogue"
        if len(shared_roles) >= 2 or len(shared_features) >= 2:
            return "port_program_candidate"
        return "analogous_to"

    def _reason_text(self, relation: str, evidence: DisparateRelationEvidence) -> str:
        evidence_parts = []
        if evidence.shared_feature_families:
            evidence_parts.append(
                f"shared features {', '.join(evidence.shared_feature_families[:3])}"
            )
        if evidence.shared_role_tags:
            evidence_parts.append(f"shared roles {', '.join(evidence.shared_role_tags[:3])}")
        if not evidence_parts and evidence.evidence_mix:
            evidence_parts.append(f"evidence mix {', '.join(evidence.evidence_mix[:3])}")
        summary = "; ".join(evidence_parts) or "structural evidence"
        return f"{relation} promoted from {summary}."

    @staticmethod
    def _stable_fingerprint(prefix: str, parts: list[str]) -> str:
        text = "|".join(str(item).strip().lower() for item in parts if str(item).strip())
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}:{digest}"

    @staticmethod
    def _path_tokens(path: str) -> set[str]:
        stem = os.path.splitext(os.path.basename(path))[0]
        values = {stem, *path.replace("-", "_").replace("/", "_").split("_")}
        return {match.group(0).lower() for value in values for match in _TOKEN_RE.finditer(value)}

    def _role_tags_for_path(self, path: str) -> set[str]:
        parts = [part for part in path.lower().split("/") if part]
        tags: set[str] = set()
        for part in parts:
            tags.update(_ROLE_HINTS.get(part, set()))
        if "/tests/" in f"/{path.lower()}/" or path.lower().startswith("tests/"):
            tags.add("test_harness")
        if path.lower().endswith(".md"):
            tags.add("doc")
        return tags

    def _feature_families_for_terms(self, values: set[str]) -> set[str]:
        families: set[str] = set()
        for value in values:
            lowered = str(value).lower()
            for needle, family in _FEATURE_HINTS.items():
                if needle in lowered:
                    families.add(family)
        return families

    @staticmethod
    def _subsystem_for_path(path: str, *, fallback: str) -> str:
        lowered = path.lower()
        if lowered.startswith("core/qsg/"):
            return "qsg"
        if lowered.startswith("saguaro/") or "/saguaro/" in lowered:
            return "saguaro"
        if lowered.startswith("core/"):
            return "core"
        if lowered.startswith("cli/"):
            return "cli"
        return fallback

    @staticmethod
    def _language_for_path(path: str) -> str:
        ext = os.path.splitext(path.lower())[1]
        return {
            ".c": "c",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".go": "go",
            ".h": "c_header",
            ".hh": "cpp_header",
            ".hpp": "cpp_header",
            ".json": "json",
            ".md": "markdown",
            ".py": "python",
            ".rs": "rust",
            ".toml": "toml",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".yaml": "yaml",
            ".yml": "yaml",
        }.get(ext, "unknown")
