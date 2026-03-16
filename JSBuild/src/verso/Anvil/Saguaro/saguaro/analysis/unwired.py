"""Unwired feature analysis based on graph reachability from runtime entrypoints."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict, deque
from typing import Any

_IGNORE_PREFIXES = (
    "venv/",
    ".venv/",
    ".git/",
    ".saguaro/",
    "node_modules/",
    "build/",
    "dist/",
    "saguaro/mcp/",
    "Saguaro/",
    ".anvil/toolchains/",
)


class UnwiredAnalyzer:
    """Detect isolated unreachable feature islands from entrypoint roots."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = os.path.abspath(repo_path)

    def analyze(
        self,
        graph_payload: dict[str, Any] | None = None,
        entry_points: list[dict[str, Any]] | None = None,
        *,
        threshold: float = 0.55,
        min_nodes: int = 4,
        min_files: int = 2,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
        external_warnings: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved = self._resolve_graph_payload(graph_payload)
        graph_path = resolved.get("graph_path")
        graph = resolved.get("graph") or {}

        nodes = self._graph_items(graph.get("nodes"))
        edges = self._graph_items(graph.get("edges"))
        files = self._graph_items(graph.get("files"))
        warnings: list[str] = []
        roots_input = list(entry_points or [])

        if external_warnings:
            warnings.extend(
                [str(item) for item in external_warnings if str(item).strip()]
            )
        if resolved.get("source") == "persisted_code_graph":
            warnings.append("Using persisted code_graph snapshot for unwired analysis.")
        if not roots_input:
            roots_input = self._infer_entry_points_from_graph(files)
            if roots_input:
                warnings.append(
                    "No entrypoints supplied; inferred roots from graph file hints."
                )

        in_scope_nodes = {
            node_id: node
            for node_id, node in nodes.items()
            if self._node_in_scope(node, include_tests=include_tests)
        }
        in_scope_ids = set(in_scope_nodes.keys())

        if any(
            self._normalize_rel_path(str(node.get("file") or "")).startswith("Saguaro/")
            for node in in_scope_nodes.values()
        ):
            warnings.append(
                "Legacy 'Saguaro/' nodes were detected and filtered from unwired analysis."
            )

        out_neighbors: dict[str, set[str]] = defaultdict(set)
        in_neighbors: dict[str, set[str]] = defaultdict(set)
        internal_edge_count: Counter[tuple[str, str]] = Counter()
        module_file_nodes: dict[str, set[str]] = defaultdict(set)
        package_by_node_id: dict[str, str] = {}
        module_by_node_id: dict[str, str] = {}
        for node_id, node in in_scope_nodes.items():
            rel_file = self._normalize_rel_path(str(node.get("file") or ""))
            module_name = self._module_name_from_file(rel_file)
            if module_name:
                module_by_node_id[node_id] = module_name
                package_by_node_id[node_id] = self._module_package(
                    module_name=module_name,
                    rel_file=rel_file,
                )
                if str(node.get("type") or "") == "file":
                    module_file_nodes[module_name].add(node_id)

        for edge in edges.values():
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src not in in_scope_ids:
                continue
            if dst in in_scope_ids:
                out_neighbors[src].add(dst)
                in_neighbors[dst].add(src)
                internal_edge_count[(src, dst)] += 1
                continue
            if not dst.startswith("external::"):
                continue

            raw_external = dst[len("external::") :].strip()
            for resolved_dst in self._resolve_external_targets(
                raw_external,
                module_file_nodes,
                source_package=package_by_node_id.get(src, ""),
                source_module=module_by_node_id.get(src, ""),
            ):
                if resolved_dst not in in_scope_ids:
                    continue
                out_neighbors[src].add(resolved_dst)
                in_neighbors[resolved_dst].add(src)
                internal_edge_count[(src, resolved_dst)] += 1

        file_node_ids_by_path: dict[str, set[str]] = defaultdict(set)
        member_node_ids_by_path: dict[str, set[str]] = defaultdict(set)
        for node_id, node in in_scope_nodes.items():
            rel_file = self._normalize_rel_path(str(node.get("file") or ""))
            if not rel_file:
                continue
            if str(node.get("type") or "") == "file":
                file_node_ids_by_path[rel_file].add(node_id)
            else:
                member_node_ids_by_path[rel_file].add(node_id)

        # Graph payloads may not include file->symbol edges. Add lightweight structural
        # edges so import-reachable files propagate reachability to contained symbols.
        for rel_file, file_node_ids in file_node_ids_by_path.items():
            members = member_node_ids_by_path.get(rel_file, set())
            if not members:
                continue
            for file_node_id in file_node_ids:
                for member_node_id in members:
                    out_neighbors[file_node_id].add(member_node_id)
                    in_neighbors[member_node_id].add(file_node_id)
                    internal_edge_count[(file_node_id, member_node_id)] += 1

        roots = self._resolve_roots(
            entry_points=roots_input,
            files=files,
            in_scope_ids=in_scope_ids,
            include_tests=include_tests,
        )

        root_node_ids = set(roots["node_ids"])
        if not root_node_ids:
            warnings.append(
                "No in-scope entrypoint roots were resolved; unwired analysis returned no clusters."
            )
            return {
                "status": "no_roots",
                "threshold": float(threshold),
                "summary": {
                    "cluster_count": 0,
                    "unreachable_node_count": 0,
                    "unreachable_file_count": 0,
                },
                "clusters": [],
                "roots": roots["meta"],
                "warnings": sorted(set(warnings)),
                "graph": {
                    "path": graph_path,
                    "generated_at": graph.get("generated_at"),
                    "stats": dict(graph.get("stats") or {}),
                },
            }

        reachable = self._reachable_nodes(root_node_ids, out_neighbors)
        candidate_ids = {
            node_id
            for node_id in in_scope_ids
            if node_id not in reachable
            and str(in_scope_nodes[node_id].get("type") or "") != "external"
        }

        components = self._components(candidate_ids, out_neighbors, in_neighbors)

        clusters: list[dict[str, Any]] = []
        for component in components:
            cluster = self._cluster_payload(
                component=component,
                in_scope_nodes=in_scope_nodes,
                out_neighbors=out_neighbors,
                in_neighbors=in_neighbors,
                reachable_ids=reachable,
                edge_counter=internal_edge_count,
                min_nodes=min_nodes,
                min_files=min_files,
            )
            if (
                cluster["classification"] == "unreachable_fragment"
                and not include_fragments
            ):
                continue
            if float(cluster["confidence"]) < float(threshold):
                continue
            clusters.append(cluster)

        clusters.sort(
            key=lambda item: (
                float(item.get("confidence", 0.0)),
                int(item.get("node_count", 0)),
                int(item.get("file_count", 0)),
            ),
            reverse=True,
        )
        if max_clusters > 0:
            clusters = clusters[:max_clusters]

        unreachable_files = {
            self._normalize_rel_path(str(in_scope_nodes[node_id].get("file") or ""))
            for node_id in candidate_ids
            if self._normalize_rel_path(str(in_scope_nodes[node_id].get("file") or ""))
        }

        return {
            "status": "ok",
            "threshold": float(threshold),
            "summary": {
                "cluster_count": len(clusters),
                "unreachable_node_count": len(candidate_ids),
                "unreachable_file_count": len(unreachable_files),
            },
            "clusters": clusters,
            "roots": roots["meta"],
            "warnings": sorted(set(warnings)),
            "graph": {
                "path": graph_path,
                "generated_at": graph.get("generated_at"),
                "stats": dict(graph.get("stats") or {}),
            },
        }

    def _resolve_roots(
        self,
        *,
        entry_points: list[dict[str, Any]],
        files: dict[str, Any],
        in_scope_ids: set[str],
        include_tests: bool,
    ) -> dict[str, Any]:
        node_ids: set[str] = set()
        kept_entries: list[dict[str, Any]] = []
        filtered_out = 0

        for entry in entry_points:
            raw_file = str(entry.get("file") or "")
            rel_file = self._normalize_rel_path(self._safe_relpath(raw_file))
            if not rel_file:
                continue
            if not self._file_in_scope(rel_file, include_tests=include_tests):
                filtered_out += 1
                continue

            file_nodes = set((files.get(rel_file) or {}).get("nodes") or [])
            file_nodes = {node_id for node_id in file_nodes if node_id in in_scope_ids}
            if not file_nodes:
                continue

            node_ids.update(file_nodes)
            kept_entries.append(
                {
                    "type": str(entry.get("type") or "unknown"),
                    "file": rel_file,
                    "line": int(entry.get("line") or 0),
                    "name": str(entry.get("name") or "").strip() or None,
                    "seeded_nodes": len(file_nodes),
                }
            )

        return {
            "node_ids": sorted(node_ids),
            "meta": {
                "count": len(kept_entries),
                "seeded_nodes": len(node_ids),
                "filtered_out": filtered_out,
                "entrypoints": kept_entries,
            },
        }

    def _reachable_nodes(
        self,
        seeds: set[str],
        out_neighbors: dict[str, set[str]],
    ) -> set[str]:
        seen = set(seeds)
        queue = deque(seeds)
        while queue:
            current = queue.popleft()
            for neighbor in out_neighbors.get(current, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        return seen

    def _components(
        self,
        candidate_ids: set[str],
        out_neighbors: dict[str, set[str]],
        in_neighbors: dict[str, set[str]],
    ) -> list[set[str]]:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for node_id in candidate_ids:
            for neighbor in out_neighbors.get(node_id, set()):
                if neighbor in candidate_ids:
                    adjacency[node_id].add(neighbor)
                    adjacency[neighbor].add(node_id)
            for neighbor in in_neighbors.get(node_id, set()):
                if neighbor in candidate_ids:
                    adjacency[node_id].add(neighbor)
                    adjacency[neighbor].add(node_id)

        components: list[set[str]] = []
        remaining = set(candidate_ids)
        while remaining:
            start = remaining.pop()
            stack = [start]
            component = {start}
            while stack:
                current = stack.pop()
                for neighbor in adjacency.get(current, set()):
                    if neighbor in component:
                        continue
                    component.add(neighbor)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                    stack.append(neighbor)
            components.append(component)
        return components

    def _cluster_payload(
        self,
        *,
        component: set[str],
        in_scope_nodes: dict[str, dict[str, Any]],
        out_neighbors: dict[str, set[str]],
        in_neighbors: dict[str, set[str]],
        reachable_ids: set[str],
        edge_counter: Counter[tuple[str, str]],
        min_nodes: int,
        min_files: int,
    ) -> dict[str, Any]:
        component_ids = set(component)
        node_count = len(component_ids)

        file_counts: Counter[str] = Counter()
        symbol_degree: Counter[str] = Counter()
        sample_nodes: list[dict[str, Any]] = []

        for node_id in sorted(component_ids):
            node = in_scope_nodes.get(node_id, {})
            rel_file = self._normalize_rel_path(str(node.get("file") or ""))
            if rel_file:
                file_counts[rel_file] += 1
            name = str(node.get("name") or "").strip()
            if name:
                symbol_degree[name] += len(out_neighbors.get(node_id, set()))
                symbol_degree[name] += len(in_neighbors.get(node_id, set()))
            if len(sample_nodes) < 8:
                sample_nodes.append(
                    {
                        "id": node_id,
                        "name": name,
                        "type": str(node.get("type") or "unknown"),
                        "file": rel_file,
                        "line": int(node.get("line") or 0),
                    }
                )

        file_count = len(file_counts)

        edge_count = 0
        inbound_from_reachable = 0
        outbound_to_reachable = 0
        for src in component_ids:
            for dst in out_neighbors.get(src, set()):
                if dst in component_ids:
                    edge_count += int(edge_counter.get((src, dst), 0) or 1)
                elif dst in reachable_ids:
                    outbound_to_reachable += 1
            for src_neighbor in in_neighbors.get(src, set()):
                if src_neighbor in reachable_ids:
                    inbound_from_reachable += 1

        if node_count > 1:
            density = edge_count / float(node_count * (node_count - 1))
        else:
            density = 0.0

        inbound_limit = max(0, min(2, int(node_count * 0.02)))
        is_unwired_feature = (
            node_count >= int(min_nodes)
            and file_count >= int(min_files)
            and inbound_from_reachable <= inbound_limit
        )
        classification = (
            "unwired_feature" if is_unwired_feature else "unreachable_fragment"
        )

        size_score = min(1.0, node_count / float(max(1, min_nodes * 2)))
        file_score = min(1.0, file_count / float(max(1, min_files * 2)))
        density_score = min(1.0, density * 3.0)
        isolation_score = max(
            0.0, 1.0 - (inbound_from_reachable / float(node_count + 1))
        )
        confidence = (
            0.35 * size_score
            + 0.2 * file_score
            + 0.2 * density_score
            + 0.25 * isolation_score
        )
        if classification == "unreachable_fragment":
            confidence *= 0.8

        top_files = [path for path, _ in file_counts.most_common(5)]
        top_symbols = [name for name, _ in symbol_degree.most_common(5)]
        label = self._cluster_label(
            classification=classification,
            files=top_files,
            symbols=top_symbols,
        )

        return {
            "id": f"cluster_{abs(hash(tuple(sorted(component_ids)))) % 1_000_000:06d}",
            "label": label,
            "classification": classification,
            "confidence": round(float(max(0.0, min(1.0, confidence))), 3),
            "node_count": node_count,
            "file_count": file_count,
            "edge_count": edge_count,
            "density": round(float(density), 4),
            "inbound_from_reachable": inbound_from_reachable,
            "outbound_to_reachable": outbound_to_reachable,
            "top_symbols": top_symbols,
            "top_files": top_files,
            "sample_nodes": sample_nodes,
        }

    def _cluster_label(
        self,
        *,
        classification: str,
        files: list[str],
        symbols: list[str],
    ) -> str:
        prefix = "unknown"
        if files:
            dirs = [os.path.dirname(path) for path in files if path]
            dirs = [d for d in dirs if d and d != "."]
            if dirs:
                common = os.path.commonpath(dirs)
                common = self._normalize_rel_path(common)
                if common and common != ".":
                    prefix = common
                else:
                    first_segment = dirs[0].split("/", 1)[0]
                    if first_segment:
                        prefix = first_segment
            else:
                prefix = files[0].split("/", 1)[0]

        symbol = symbols[0] if symbols else "entry"
        if classification == "unwired_feature":
            return f"Unwired Feature: {prefix}::{symbol}"
        return f"Unreachable Fragment: {prefix}::{symbol}"

    def _node_in_scope(self, node: dict[str, Any], *, include_tests: bool) -> bool:
        rel_file = self._normalize_rel_path(str(node.get("file") or ""))
        if not rel_file:
            return False
        if str(node.get("type") or "") == "external":
            return False
        if not rel_file.endswith(".py"):
            return False
        return self._file_in_scope(rel_file, include_tests=include_tests)

    def _file_in_scope(self, rel_file: str, *, include_tests: bool) -> bool:
        normalized = self._normalize_rel_path(rel_file)
        if not normalized:
            return False
        if normalized.startswith(".."):
            return False
        if any(normalized.startswith(prefix) for prefix in _IGNORE_PREFIXES):
            return False
        if not include_tests and (
            normalized.startswith("tests/")
            or "/tests/" in normalized
            or normalized.endswith("_test.py")
            or normalized.startswith("test_")
        ):
            return False
        return True

    def _safe_relpath(self, value: str) -> str:
        if not value:
            return ""
        absolute = value
        if not os.path.isabs(absolute):
            absolute = os.path.abspath(os.path.join(self.repo_path, absolute))
        try:
            rel = os.path.relpath(absolute, self.repo_path)
        except Exception:
            return ""
        return rel

    @staticmethod
    def _normalize_rel_path(value: str) -> str:
        return str(value or "").replace("\\", "/").strip()

    def _resolve_graph_payload(
        self,
        graph_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if isinstance(graph_payload, dict):
            graph_path = graph_payload.get("graph_path")
            graph = graph_payload.get("graph")
            if isinstance(graph, dict):
                return {
                    "graph_path": graph_path,
                    "graph": graph,
                    "source": "payload",
                }
            if all(key in graph_payload for key in ("nodes", "edges", "files")):
                return {
                    "graph_path": graph_path,
                    "graph": graph_payload,
                    "source": "payload",
                }
            code_graph = graph_payload.get("code_graph")
            if isinstance(code_graph, dict):
                return {
                    "graph_path": graph_path,
                    "graph": code_graph,
                    "source": "payload_code_graph",
                }

        candidates = [
            os.path.join(self.repo_path, ".saguaro", "graph", "code_graph.json"),
            os.path.join(self.repo_path, ".saguaro", "code_graph.json"),
            os.path.join(self.repo_path, ".saguaro", "graph", "graph.json"),
        ]
        for candidate in candidates:
            if not os.path.exists(candidate):
                continue
            try:
                with open(candidate, encoding="utf-8") as f:
                    raw = json.load(f) or {}
            except Exception:
                continue
            graph = (
                raw.get("graph") if isinstance(raw, dict) and "graph" in raw else raw
            )
            if isinstance(graph, dict):
                return {
                    "graph_path": candidate,
                    "graph": graph,
                    "source": "persisted_code_graph",
                }

        return {"graph_path": None, "graph": {}, "source": "none"}

    @staticmethod
    def _graph_items(payload: Any) -> dict[str, dict[str, Any]]:
        if isinstance(payload, dict):
            return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
        if isinstance(payload, list):
            out: dict[str, dict[str, Any]] = {}
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("id") or f"item_{idx}")
                out[node_id] = dict(item)
            return out
        return {}

    def _infer_entry_points_from_graph(
        self,
        files: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        preferred = ("main.py", "__main__.py", "app.py", "cli.py")
        inferred: list[dict[str, Any]] = []
        for rel_file in sorted(files.keys()):
            if not rel_file:
                continue
            basename = os.path.basename(rel_file)
            if basename not in preferred:
                continue
            inferred.append(
                {
                    "type": "inferred",
                    "file": rel_file,
                    "line": 1,
                    "name": basename,
                }
            )
        if inferred:
            return inferred

        for rel_file in sorted(files.keys()):
            if rel_file.endswith(".py"):
                inferred.append(
                    {
                        "type": "inferred",
                        "file": rel_file,
                        "line": 1,
                        "name": os.path.basename(rel_file),
                    }
                )
                if len(inferred) >= 3:
                    break
        return inferred

    @staticmethod
    def _module_name_from_file(rel_file: str) -> str:
        if not rel_file.endswith(".py"):
            return ""
        module = rel_file[: -len(".py")].replace("/", ".")
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        return module

    @staticmethod
    def _module_package(module_name: str, rel_file: str) -> str:
        if not module_name:
            return ""
        if rel_file.endswith("/__init__.py"):
            return module_name
        if "." in module_name:
            return module_name.rsplit(".", 1)[0]
        return ""

    @staticmethod
    def _resolve_external_targets(
        external_ref: str,
        module_file_nodes: dict[str, set[str]],
        *,
        source_package: str = "",
        source_module: str = "",
    ) -> set[str]:
        target = external_ref.strip().lstrip(".")
        if not target:
            return set()

        candidates = [target]
        if source_package:
            candidates.append(f"{source_package}.{target}")
        if source_module:
            candidates.append(f"{source_module}.{target}")

        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            parts = [part for part in candidate.split(".") if part]
            while parts:
                module = ".".join(parts)
                node_ids = module_file_nodes.get(module)
                if node_ids:
                    return set(node_ids)
                parts = parts[:-1]
        return set()
