"""Impact Analyzer
Determines the downstream impact of code changes on tests, interfaces, and build targets.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from typing import Any

from saguaro.refactor.planner import RefactorPlanner

logger = logging.getLogger(__name__)


class ImpactAnalyzer:
    """Provide ImpactAnalyzer support."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.planner = RefactorPlanner(repo_path)

    def analyze_change(self, file_path: str, symbol: str = None) -> dict:
        """Analyze impact of changing a specific file or symbol."""
        target = (
            file_path
            if os.path.isabs(file_path)
            else os.path.join(self.repo_path, file_path)
        )
        target = os.path.abspath(target)

        graph_result = self._analyze_with_code_graph(target, symbol=symbol)
        if graph_result:
            return graph_result

        target_module = self._file_to_module(target)
        dependents = self._find_importers(target_module)

        tests = [f for f in dependents if "test" in f or "tests" in f.split(os.sep)]
        interfaces = [f for f in dependents if f not in tests]
        build_targets = self._find_build_targets(target)

        return {
            "target": target,
            "module": target_module,
            "impact_score": len(dependents),
            "tests_impacted": tests,
            "interfaces_impacted": interfaces,
            "build_targets": build_targets,
            "analysis_mode": "heuristic_scan",
        }

    def _file_to_module(self, path: str) -> str:
        rel = os.path.relpath(path, self.repo_path)
        if rel.endswith(".py"):
            rel = rel[:-3]
        return rel.replace(os.sep, ".")

    def _find_importers(self, module_name: str) -> list[str]:
        importers = []
        # Naive scan
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root or "venv" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    fpath = os.path.join(root, file)
                    try:
                        with open(fpath) as f:
                            content = f.read()
                            # Text search for import
                            if (
                                f"import {module_name}" in content
                                or f"from {module_name}" in content
                            ):
                                importers.append(fpath)
                    except Exception:
                        pass
        return importers

    def _find_build_targets(self, file_path: str) -> list[str]:
        """Finds build configuration files in the directory hierarchy."""
        targets = []
        current = os.path.dirname(file_path)
        while current.startswith(self.repo_path):
            # Check for common build files
            for bf in [
                "setup.py",
                "CMakeLists.txt",
                "package.json",
                "Makefile",
                "BUILD",
                "pyproject.toml",
            ]:
                p = os.path.join(current, bf)
                if os.path.exists(p):
                    targets.append(p)
            current = os.path.dirname(current)
        return targets

    def _analyze_with_code_graph(
        self,
        target_file: str,
        *,
        symbol: str | None = None,
    ) -> dict[str, Any] | None:
        payload = self._load_code_graph()
        graph = payload.get("graph") or {}
        if not isinstance(graph, dict):
            return None

        nodes = self._graph_items(graph.get("nodes"))
        edges = self._graph_items(graph.get("edges"))
        files = self._graph_items(graph.get("files"))
        if not nodes or not edges:
            return None

        target_rel = os.path.relpath(target_file, self.repo_path).replace("\\", "/")
        target_ids = set((files.get(target_rel) or {}).get("nodes") or [])
        if not target_ids:
            for node_id, node in nodes.items():
                rel_file = str(node.get("file") or "").replace("\\", "/")
                if rel_file == target_rel:
                    target_ids.add(node_id)

        if symbol:
            needle = symbol.lower()
            filtered = {
                node_id
                for node_id in target_ids
                if needle
                in " ".join(
                    [
                        str(nodes.get(node_id, {}).get("name") or ""),
                        str(nodes.get(node_id, {}).get("qualified_name") or ""),
                    ]
                ).lower()
            }
            if filtered:
                target_ids = filtered

        if not target_ids:
            return None

        incoming: dict[str, set[str]] = defaultdict(set)
        for edge in edges.values():
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if not src or not dst:
                continue
            incoming[dst].add(src)

        impacted_nodes: set[str] = set()
        queue = deque([(node_id, 0) for node_id in target_ids])
        visited = set(target_ids)
        while queue:
            node_id, depth = queue.popleft()
            if depth >= 8:
                continue
            for src in incoming.get(node_id, set()):
                if src in visited:
                    continue
                visited.add(src)
                impacted_nodes.add(src)
                queue.append((src, depth + 1))

        impacted_files = set()
        for node_id in impacted_nodes:
            node = nodes.get(node_id, {})
            rel_file = str(node.get("file") or "").replace("\\", "/")
            if not rel_file or rel_file == target_rel:
                continue
            impacted_files.add(os.path.join(self.repo_path, rel_file))

        tests = sorted(
            file_path
            for file_path in impacted_files
            if "test" in file_path.lower() or f"{os.sep}tests{os.sep}" in file_path
        )
        interfaces = sorted(impacted_files - set(tests))
        build_targets = self._find_build_targets(target_file)

        return {
            "target": target_file,
            "module": self._file_to_module(target_file),
            "impact_score": len(impacted_files),
            "tests_impacted": tests,
            "interfaces_impacted": interfaces,
            "build_targets": build_targets,
            "analysis_mode": "code_graph",
            "graph_path": payload.get("graph_path"),
            "impacted_nodes": len(impacted_nodes),
        }

    def _load_code_graph(self) -> dict[str, Any]:
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
                return {"graph_path": candidate, "graph": graph}
        return {"graph_path": None, "graph": {}}

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
