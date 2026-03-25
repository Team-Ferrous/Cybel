"""Refactor Planner
Generates execution plans for refactoring tasks by analyzing dependencies and impact.
"""

import ast
import logging
import os
from typing import Any

from saguaro.indexing.engine import IndexEngine

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Provide DependencyGraph support."""
    def __init__(self) -> None:
        """Initialize the instance."""
        self.edges = {}  # file -> list of imported files
        self.reverse_edges = {}  # file -> list of importers

    def add_dependency(self, source: str, target: str) -> None:
        """Handle add dependency."""
        if source not in self.edges:
            self.edges[source] = set()
        self.edges[source].add(target)

        if target not in self.reverse_edges:
            self.reverse_edges[target] = set()
        self.reverse_edges[target].add(source)


class RefactorPlanner:
    """Provide RefactorPlanner support."""
    def __init__(self, repo_path: str, engine: IndexEngine = None) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.engine = engine

    def plan_symbol_modification(self, symbol_name: str) -> dict[str, Any]:
        """Analyze the impact of modifying a symbol (e.g. rename/change signature)."""
        # 1. Find candidates via Index (or global grep if no index)
        candidates = self._find_candidates(symbol_name)

        # 2. Verify usage via AST
        verified_usages = self._verify_usages(symbol_name, candidates)

        # 3. Build Dependency Graph of impacted files
        # We need the graph of ALL impacted files to sort them.
        graph = self._build_dependency_graph(list(verified_usages.keys()))

        # 4. Analyze API Risk
        api_risk = self._analyze_api_risk(symbol_name, verified_usages, candidates)

        # 5. Generate Phased Plan
        ordered_files = self._schedule_changes(graph, verified_usages)

        return {
            "symbol": symbol_name,
            "impact_score": len(verified_usages),
            "api_surface_risk": api_risk,
            "files_impacted": ordered_files,
            "modules": self._group_by_module(verified_usages.keys()),
            "phases": [
                {"order": i + 1, "file": f, "reason": "Dependency chain"}
                for i, f in enumerate(ordered_files)
            ],
        }


    def _find_candidates(self, symbol: str) -> list[str]:
        # Fast text search
        matches = []
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root or "venv" in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, errors="ignore") as f:
                            if symbol in f.read():
                                matches.append(path)
                    except Exception:
                        pass
        return matches

    def _verify_usages(self, symbol: str, files: list[str]) -> dict[str, list[int]]:
        """Parse files to confirm symbol is used as an identifier."""
        usage_map = {}
        for file_path in files:
            try:
                with open(file_path) as f:
                    content = f.read()

                tree = ast.parse(content)
                lines = []
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Name)
                        and node.id == symbol
                        or isinstance(node, ast.FunctionDef)
                        and node.name == symbol
                        or isinstance(node, ast.ClassDef)
                        and node.name == symbol
                        or isinstance(node, ast.Attribute)
                        and node.attr == symbol
                    ):
                        lines.append(node.lineno)

                if lines:
                    usage_map[file_path] = sorted(list(set(lines)))
            except Exception:
                pass

        return usage_map

    def _analyze_api_risk(
        self, symbol: str, usages: dict[str, list[int]], files: list[str]
    ) -> str:
        """Determines if the symbol is part of the public API surface."""
        risk = "Low (Internal)"

        # 1. Check naming convention
        if not symbol.startswith("_"):
            risk = "Medium (Public Name)"

        # 2. Check if exported in __init__.py
        for f in files:
            if f.endswith("__init__.py") and f in usages:
                risk = "High (Exported in __init__)"
                break

        # 3. Check for external usage (outside own module)
        modules = self._group_by_module(usages.keys())
        if len(modules) > 2:
            risk = f"{risk} - Highly Coupled ({len(modules)} modules)"

        return risk

    def _build_dependency_graph(self, files: list[str]) -> DependencyGraph:
        graph = DependencyGraph()

        # 1. Map all files in repo for resolution
        file_map = {}  # module_name -> file_path
        for root, _, repo_files in os.walk(self.repo_path):
            if ".git" in root or "venv" in root:
                continue
            for f in repo_files:
                if f.endswith(".py"):
                    full_path = os.path.join(root, f)
                    mod_name = self._file_to_module(full_path)
                    file_map[mod_name] = full_path

        # 2. Parse imports for each relevant file
        # We scan ALL provided files + their likely dependencies if we want a full graph.
        # For this prototype, we scan the input files and resolve their immediate imports.
        queue = list(files)
        visited = set()

        while queue:
            current_file = queue.pop(0)
            if current_file in visited:
                continue
            visited.add(current_file)

            try:
                with open(current_file) as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    imported_modules = []
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_modules.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imported_modules.append(node.module)

                    for mod in imported_modules:
                        # Attempt to resolve to a local file
                        # Exact match
                        if mod in file_map:
                            target = file_map[mod]
                            graph.add_dependency(current_file, target)
                            if target not in visited:
                                queue.append(target)
                        else:
                            # Try prefix matching (e.g. saguaro.refactor.planner -> saguaro.refactor)
                            parts = mod.split(".")
                            for i in range(len(parts), 0, -1):
                                sub = ".".join(parts[:i])
                                if sub in file_map:
                                    target = file_map[sub]
                                    graph.add_dependency(current_file, target)
                                    if target not in visited:
                                        queue.append(target)
                                    break
            except Exception:
                pass

        return graph

    def _file_to_module(self, path: str) -> str:
        rel = os.path.relpath(path, self.repo_path)
        if rel.endswith(".py"):
            rel = rel[:-3]
        return rel.replace(os.sep, ".")

    def detect_cycles(self, graph: DependencyGraph) -> list[list[str]]:
        """Detect circular dependencies in the graph."""
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(u: str) -> None:
            visited.add(u)
            path.append(u)
            path_set.add(u)

            if u in graph.edges:
                for v in graph.edges[u]:
                    if v in path_set:
                        # Cycle found
                        cycle_start_index = path.index(v)
                        cycles.append(path[cycle_start_index:])
                    elif v not in visited:
                        dfs(v)

            path.pop()
            path_set.remove(u)

        for node in list(graph.edges.keys()):
            if node not in visited:
                dfs(node)

        return cycles

    def _group_by_module(self, files: list[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for file_path in files:
            module = self._file_to_module(file_path)
            root = module.split(".")[0] if module else "<unknown>"
            grouped.setdefault(root, []).append(file_path)
        return grouped

    def _schedule_changes(
        self, graph: DependencyGraph, usages: dict[str, list[int]]
    ) -> list[str]:
        impacted = list(usages.keys())
        if not impacted:
            return []

        impacted_set = set(impacted)
        indegree = {path: 0 for path in impacted}

        for src in impacted:
            for dst in graph.edges.get(src, set()):
                if dst in impacted_set:
                    indegree[dst] += 1

        ready = sorted([p for p, degree in indegree.items() if degree == 0])
        ordered: list[str] = []
        remaining = dict(indegree)

        while ready:
            current = ready.pop(0)
            if current not in remaining:
                continue
            ordered.append(current)
            remaining.pop(current, None)

            for dst in graph.edges.get(current, set()):
                if dst not in remaining:
                    continue
                remaining[dst] -= 1
                if remaining[dst] == 0:
                    ready.append(dst)
            ready.sort()

        # If cycles remain, append deterministically.
        if remaining:
            ordered.extend(sorted(remaining.keys()))

        return ordered
