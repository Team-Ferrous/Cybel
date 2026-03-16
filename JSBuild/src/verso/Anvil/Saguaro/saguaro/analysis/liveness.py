"""Unified liveness and reachability analysis."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict, deque
from typing import Any

from saguaro.analysis.dead_code import DeadCodeAnalyzer
from saguaro.analysis.duplicates import DuplicateAnalyzer
from saguaro.analysis.entry_points import EntryPointDetector
from saguaro.analysis.unwired import UnwiredAnalyzer

_LIVE_RELATIONS = {"imports", "calls", "includes", "ffi_bridge", "ffi_detected", "references", "related"}
_STATIC_USAGE_RELATIONS = {"imports", "calls", "includes", "references", "related"}
_CANDIDATE_TYPES = {"function", "class", "method", "symbol"}
_AUXILIARY_SURFACE_PREFIXES = ("benchmarks/", "scripts/", "docs/")


class LivenessAnalyzer:
    """Aggregate graph, entrypoint, dead-code, and duplicate evidence."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self._duplicate_map = DuplicateAnalyzer(self.repo_path).file_cluster_map(path=".")

    def analyze(
        self,
        *,
        threshold: float = 0.5,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
        max_low_usage_refs: int = 1,
        path_prefix: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        graph_payload = self._load_graph()
        graph = graph_payload.get("graph") or {}
        nodes = self._graph_items(graph.get("nodes"))
        edges = self._graph_items(graph.get("edges"))
        files = self._graph_items(graph.get("files"))
        entry_points = EntryPointDetector(self.repo_path).detect()
        entrypoint_seed_ids = self._entrypoint_seed_ids(
            nodes=nodes,
            files=files,
            entry_points=entry_points,
            include_tests=include_tests,
        )
        reachable_ids = self._reachable_node_ids(
            nodes=nodes,
            edges=edges,
            seed_ids=entrypoint_seed_ids,
        )
        deadcode = DeadCodeAnalyzer(self.repo_path, include_tests=include_tests).analyze()
        dead_index = {
            (str(item.get("file") or "").replace("\\", "/"), str(item.get("symbol") or "")): item
            for item in deadcode
        }

        incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
        outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in edges.values():
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src:
                outgoing[src].append(edge)
            if dst:
                incoming[dst].append(edge)

        candidates: list[dict[str, Any]] = []
        for node_id, node in nodes.items():
            node_type = str(node.get("type") or "")
            if node_type not in _CANDIDATE_TYPES:
                continue
            rel_file = str(node.get("file") or "").replace("\\", "/")
            if not rel_file:
                continue
            if not include_tests and rel_file.startswith("tests/"):
                continue
            evidence = self._evidence(
                node_id=node_id,
                node=node,
                nodes=nodes,
                reachable_ids=reachable_ids,
                entrypoint_seed_ids=entrypoint_seed_ids,
                incoming=incoming,
                outgoing=outgoing,
            )
            duplicate_clusters = self._duplicate_map.get(rel_file, [])
            if duplicate_clusters:
                evidence["duplicate_of"] = [cluster["id"] for cluster in duplicate_clusters]
            classification = self._classify(evidence)
            dead_row = dead_index.get((rel_file, str(node.get("name") or "")))
            confidence = self._confidence(evidence=evidence, dead_row=dead_row)
            if classification.startswith("dead") and confidence < threshold:
                continue
            candidates.append(
                {
                    "symbol": str(node.get("qualified_name") or node.get("name") or node_id),
                    "name": str(node.get("name") or ""),
                    "file": rel_file,
                    "line": int(node.get("line", 0) or 0),
                    "type": node_type,
                    "classification": classification,
                    "confidence": confidence,
                    "evidence": evidence,
                    "reason": self._reason(classification, evidence),
                }
            )

        unwired = UnwiredAnalyzer(self.repo_path).analyze(
            graph_payload=graph_payload,
            entry_points=entry_points,
            threshold=threshold,
            include_tests=include_tests,
            include_fragments=include_fragments,
            max_clusters=max_clusters,
        )
        candidates.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                str(item.get("file") or ""),
                int(item.get("line", 0) or 0),
            ),
            reverse=True,
        )
        name_counts = Counter(
            self._symbol_key(str(item.get("name") or item.get("symbol") or ""))
            for item in candidates
            if str(item.get("name") or item.get("symbol") or "")
        )
        max_refs = max(0, int(max_low_usage_refs or 0))
        normalized_path = self._normalize_path_filter(path_prefix)
        enriched_low_usage = [
            self._enrich_low_usage_candidate(
                item,
                name_counts=name_counts,
            )
            for item in candidates
            if item.get("classification") == "live"
            and int((item.get("evidence") or {}).get("usage_count", 0) or 0) <= max_refs
            and self._matches_path_filter(str(item.get("file") or ""), normalized_path)
        ]
        enriched_low_usage.sort(
            key=lambda item: (
                0 if item.get("reuse_candidate") else 1,
                float(item.get("reuse_score", 0.0) or 0.0) * -1.0,
                int((item.get("evidence") or {}).get("usage_count", 0) or 0),
                str(item.get("file") or ""),
                int(item.get("line", 0) or 0),
            )
        )
        dry_candidates = [item for item in enriched_low_usage if item.get("reuse_candidate")]
        limited_candidates = self._limit_items(enriched_low_usage, limit)
        limited_dry_candidates = self._limit_items(dry_candidates, limit)
        return {
            "status": "ok",
            "graph_path": graph_payload.get("graph_path"),
            "count": len(candidates),
            "candidates": candidates,
            "low_usage": {
                "max_refs": max_refs,
                "count": len(enriched_low_usage),
                "returned_count": len(limited_candidates),
                "candidates": limited_candidates,
                "dry_count": len(dry_candidates),
                "dry_candidates": limited_dry_candidates,
                "areas": self._summarize_low_usage_areas(enriched_low_usage),
                "path_filter": normalized_path,
                "limit": None if limit is None else max(1, int(limit)),
            },
            "clusters": list(unwired.get("clusters", [])),
            "summary": {
                "dead_confident": sum(
                    1 for item in candidates if item.get("classification") == "dead_confident"
                ),
                "dead_probable": sum(
                    1 for item in candidates if item.get("classification") == "dead_probable"
                ),
                "registration_driven": sum(
                    1 for item in candidates if item.get("classification") == "registration_driven"
                ),
                "duplicate_logic": sum(
                    1 for item in candidates if item.get("classification") == "duplicate_logic"
                ),
                "unreachable_feature_clusters": int(
                    unwired.get("summary", {}).get("cluster_count", 0) or 0
                ),
                "low_usage": len(enriched_low_usage),
                "low_usage_reuse_candidates": len(dry_candidates),
            },
        }

    def explain(self, symbol: str) -> dict[str, Any]:
        report = self.analyze(threshold=0.0)
        needle = str(symbol or "").strip().lower()
        for item in report.get("candidates", []):
            haystack = " ".join(
                [str(item.get("symbol") or ""), str(item.get("name") or "")]
            ).lower()
            if needle and needle in haystack:
                return {"status": "ok", "item": item}
        return {"status": "missing", "symbol": symbol}

    def _load_graph(self) -> dict[str, Any]:
        candidates = [
            os.path.join(self.repo_path, ".saguaro", "graph", "graph.json"),
            os.path.join(self.repo_path, ".saguaro", "graph", "code_graph.json"),
            os.path.join(self.repo_path, ".saguaro", "code_graph.json"),
        ]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, encoding="utf-8") as handle:
                    payload = json.load(handle) or {}
            except Exception:
                continue
            graph = payload.get("graph") if isinstance(payload, dict) and "graph" in payload else payload
            if isinstance(graph, dict):
                return {"graph_path": candidate, "graph": graph}
        return {"graph_path": None, "graph": {}}

    @staticmethod
    def _graph_items(payload: Any) -> dict[str, dict[str, Any]]:
        if isinstance(payload, dict):
            return {str(key): dict(value) for key, value in payload.items() if isinstance(value, dict)}
        if isinstance(payload, list):
            out: dict[str, dict[str, Any]] = {}
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                out[str(item.get("id") or f"item_{index}")] = dict(item)
            return out
        return {}

    def _entrypoint_seed_ids(
        self,
        *,
        nodes: dict[str, dict[str, Any]],
        files: dict[str, dict[str, Any]],
        entry_points: list[dict[str, Any]],
        include_tests: bool,
    ) -> set[str]:
        file_entries = files if isinstance(files, dict) else {}
        seeds: set[str] = set()
        for entry in entry_points:
            rel_file = self._safe_relpath(str(entry.get("file") or ""))
            if not include_tests and rel_file.startswith("tests/"):
                continue
            file_payload = file_entries.get(rel_file, {})
            for node_id in file_payload.get("nodes", []) or []:
                if node_id in nodes:
                    seeds.add(str(node_id))
        return seeds

    def _reachable_node_ids(
        self,
        *,
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
        seed_ids: set[str],
    ) -> set[str]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges.values():
            relation = str(edge.get("relation") or "")
            if relation not in _LIVE_RELATIONS:
                continue
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src and dst and dst in nodes:
                adjacency[src].add(dst)
        queue = deque(sorted(seed_ids))
        seen = set(seed_ids)
        while queue:
            current = queue.popleft()
            for nxt in adjacency.get(current, set()):
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return seen

    def _evidence(
        self,
        *,
        node_id: str,
        node: dict[str, Any],
        nodes: dict[str, dict[str, Any]],
        reachable_ids: set[str],
        entrypoint_seed_ids: set[str],
        incoming: dict[str, list[dict[str, Any]]],
        outgoing: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        inc = incoming.get(node_id, [])
        out = outgoing.get(node_id, [])
        usage_edges = [
            edge
            for edge in inc
            if str(edge.get("relation") or "") in _STATIC_USAGE_RELATIONS
        ]
        referencing_files: set[str] = set()
        for edge in usage_edges:
            src_id = str(edge.get("from") or "")
            src_payload = nodes.get(src_id, {}) if src_id else {}
            source_file = str(src_payload.get("file") or edge.get("file") or "").replace(
                "\\", "/"
            )
            if source_file:
                referencing_files.add(source_file)
        rel_file = str(node.get("file") or "").replace("\\", "/")
        evidence = {
            "declared": True,
            "defined": int(node.get("line", 0) or 0) > 0,
            "import_referenced": any(str(edge.get("relation") or "") == "imports" for edge in inc),
            "call_referenced": any(str(edge.get("relation") or "") == "calls" for edge in inc),
            "include_referenced": any(str(edge.get("relation") or "") == "includes" for edge in inc),
            "usage_count": len(usage_edges),
            "referencing_files": sorted(referencing_files),
            "graph_reachable": node_id in reachable_ids,
            "entrypoint_reachable": node_id in reachable_ids,
            "entrypoint_seed": node_id in entrypoint_seed_ids,
            "self_referenced_only": bool(referencing_files) and referencing_files == {rel_file},
            "build_included": not rel_file.startswith("tests/"),
            "binary_exported": any(str(edge.get("relation") or "") == "ffi_bridge" for edge in out),
            "ffi_bound": any(
                str(edge.get("relation") or "") in {"ffi_bridge", "ffi_detected"}
                for edge in inc + out
            ),
            "runtime_observed": False,
            "duplicate_of": [],
        }
        return evidence

    @staticmethod
    def _classify(evidence: dict[str, Any]) -> str:
        if evidence.get("graph_reachable"):
            return "live"
        if evidence.get("duplicate_of"):
            return "duplicate_logic"
        if evidence.get("ffi_bound") or evidence.get("binary_exported"):
            return "registration_driven"
        if any(
            evidence.get(flag)
            for flag in ("import_referenced", "call_referenced", "include_referenced")
        ):
            return "dead_probable"
        return "dead_confident"

    @staticmethod
    def _confidence(
        *,
        evidence: dict[str, Any],
        dead_row: dict[str, Any] | None,
    ) -> float:
        if dead_row is not None:
            return float(dead_row.get("confidence", 0.0) or 0.0)
        if evidence.get("duplicate_of"):
            return 0.9
        if evidence.get("ffi_bound"):
            return 0.35
        if evidence.get("graph_reachable"):
            return 0.1
        if any(
            evidence.get(flag)
            for flag in ("import_referenced", "call_referenced", "include_referenced")
        ):
            return 0.58
        return 0.86

    @staticmethod
    def _reason(classification: str, evidence: dict[str, Any]) -> str:
        if classification == "duplicate_logic":
            return "File participates in a duplicate or structurally mirrored cluster."
        if classification == "registration_driven":
            return "Symbol is connected to an FFI/export registration path."
        if classification == "dead_probable":
            return "Symbol is not entrypoint-reachable and only weak reference evidence was found."
        if classification == "dead_confident":
            return "Symbol has no entrypoint reachability and no static reference evidence."
        return "Symbol remains reachable from entrypoints or graph roots."

    def _safe_relpath(self, path: str) -> str:
        rel = path
        if os.path.isabs(path):
            try:
                rel = os.path.relpath(path, self.repo_path)
            except ValueError:
                rel = os.path.basename(path)
        rel = str(rel).replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        return rel

    @staticmethod
    def _symbol_key(value: str) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _normalize_path_filter(path_prefix: str | None) -> str | None:
        if not path_prefix:
            return None
        normalized = str(path_prefix).strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        return normalized.rstrip("/") or None

    @staticmethod
    def _matches_path_filter(path: str, path_filter: str | None) -> bool:
        if not path_filter:
            return True
        normalized_path = str(path or "").replace("\\", "/")
        return normalized_path == path_filter or normalized_path.startswith(f"{path_filter}/")

    def _enrich_low_usage_candidate(
        self,
        item: dict[str, Any],
        *,
        name_counts: Counter[str],
    ) -> dict[str, Any]:
        enriched = dict(item)
        evidence = dict(item.get("evidence") or {})
        enriched["evidence"] = evidence
        file_path = str(enriched.get("file") or "")
        name_key = self._symbol_key(str(enriched.get("name") or enriched.get("symbol") or ""))
        usage_count = int(evidence.get("usage_count", 0) or 0)
        same_name_count = int(name_counts.get(name_key, 0) or 0)
        dry_signals: list[str] = []

        if evidence.get("entrypoint_seed"):
            dry_signals.append("entrypoint_seed")
        if usage_count == 0:
            dry_signals.append("zero_static_callers")
        elif usage_count == 1:
            dry_signals.append("single_static_caller")
        if evidence.get("self_referenced_only"):
            dry_signals.append("same_file_only")
        if list(evidence.get("duplicate_of") or []):
            dry_signals.append("duplicate_file_cluster")
        if same_name_count > 1:
            dry_signals.append("repeated_symbol_name")
        if self._is_auxiliary_surface(file_path):
            dry_signals.append("auxiliary_surface")

        reuse_candidate = self._is_reuse_candidate(
            evidence=evidence,
            dry_signals=dry_signals,
            file_path=file_path,
        )
        enriched["same_name_count"] = same_name_count
        enriched["dry_signals"] = dry_signals
        enriched["reuse_candidate"] = reuse_candidate
        enriched["reuse_score"] = self._reuse_score(
            evidence=evidence,
            dry_signals=dry_signals,
            same_name_count=same_name_count,
            file_path=file_path,
        )
        enriched["reuse_reason"] = self._reuse_reason(
            evidence=evidence,
            dry_signals=dry_signals,
            same_name_count=same_name_count,
        )
        return enriched

    def _is_reuse_candidate(
        self,
        *,
        evidence: dict[str, Any],
        dry_signals: list[str],
        file_path: str,
    ) -> bool:
        if evidence.get("entrypoint_seed"):
            return False
        strong_signals = {"same_file_only", "duplicate_file_cluster", "repeated_symbol_name"}
        if any(signal in strong_signals for signal in dry_signals):
            return True
        if (
            "zero_static_callers" in dry_signals
            and not self._is_auxiliary_surface(file_path)
            and not evidence.get("ffi_bound")
        ):
            return True
        return False

    def _reuse_score(
        self,
        *,
        evidence: dict[str, Any],
        dry_signals: list[str],
        same_name_count: int,
        file_path: str,
    ) -> float:
        score = 0.0
        if not evidence.get("entrypoint_seed"):
            score += 1.0
        if "zero_static_callers" in dry_signals:
            score += 1.3
        if "single_static_caller" in dry_signals:
            score += 0.8
        if "same_file_only" in dry_signals:
            score += 1.6
        if "duplicate_file_cluster" in dry_signals:
            score += 1.5
        if "repeated_symbol_name" in dry_signals:
            score += min(1.5, 0.5 * max(0, same_name_count - 1))
        if self._is_auxiliary_surface(file_path):
            score -= 0.5
        return round(max(score, 0.0), 2)

    @staticmethod
    def _reuse_reason(
        *,
        evidence: dict[str, Any],
        dry_signals: list[str],
        same_name_count: int,
    ) -> str:
        if "duplicate_file_cluster" in dry_signals:
            return "Lives in a duplicate file cluster and is only lightly reused."
        if "repeated_symbol_name" in dry_signals and same_name_count > 1:
            return "Symbol name appears in multiple places, suggesting a reuse/consolidation opportunity."
        if "same_file_only" in dry_signals:
            return "Only referenced from its own file, which is a common signal for extract-and-reuse cleanup."
        if "zero_static_callers" in dry_signals and not evidence.get("entrypoint_seed"):
            return "Reachable but has no static callers, which suggests an underutilized capability."
        if "single_static_caller" in dry_signals:
            return "Only one static caller was found."
        return "Low static usage."

    @staticmethod
    def _limit_items(items: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
        if limit is None:
            return list(items)
        return list(items[: max(1, int(limit))])

    def _summarize_low_usage_areas(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        areas: dict[str, dict[str, Any]] = {}
        for item in items:
            bucket = self._directory_bucket(str(item.get("file") or ""))
            entry = areas.setdefault(
                bucket,
                {
                    "path": bucket,
                    "count": 0,
                    "dry_count": 0,
                    "examples": [],
                },
            )
            entry["count"] += 1
            if item.get("reuse_candidate"):
                entry["dry_count"] += 1
            examples = entry["examples"]
            if len(examples) < 3:
                examples.append(str(item.get("symbol") or item.get("name") or ""))
        return sorted(
            areas.values(),
            key=lambda area: (
                int(area.get("dry_count", 0) or 0),
                int(area.get("count", 0) or 0),
                str(area.get("path") or ""),
            ),
            reverse=True,
        )[:10]

    @staticmethod
    def _directory_bucket(path: str) -> str:
        parts = [part for part in str(path or "").split("/") if part]
        if not parts:
            return "."
        if len(parts) == 1:
            return parts[0]
        return "/".join(parts[:2])

    @staticmethod
    def _is_auxiliary_surface(path: str) -> bool:
        normalized = str(path or "").replace("\\", "/")
        return any(normalized.startswith(prefix) for prefix in _AUXILIARY_SURFACE_PREFIXES)
