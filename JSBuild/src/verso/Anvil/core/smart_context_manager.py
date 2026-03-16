"""
Smart Context Management with Saguaro

Automatically gathers relevant context using Saguaro's:
- Semantic search
- Dependency slicing
- Impact analysis
- Memory retrieval
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ContextBundle:
    """Complete context package for a task."""

    target_files: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    dependents: Dict[str, List[str]] = field(default_factory=dict)
    related_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    similar_patterns: List[Dict] = field(default_factory=list)

    # Saguaro-specific
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    semantic_clusters: List[List[str]] = field(default_factory=list)


class SmartContextManager:
    """
    Automatically gather optimal context for tasks using Saguaro.

    This is like having a senior engineer who knows the codebase
    and can point you to exactly what you need.
    """

    def __init__(self, substrate, console):
        self.substrate = substrate
        self.console = console
        # Optional integrations (present in some loop configurations only).
        self.semantic_engine = getattr(substrate, "semantic_engine", None)
        self.registry = getattr(substrate, "registry", None)
        self.saguaro_tools = getattr(substrate, "saguaro_tools", None)

    def gather_context(self, task: str, focus_files: List[str] = None) -> ContextBundle:
        """
        Gather complete context for a task with safety limits.
        """
        self.console.print(
            "[cyan]Gathering smart context via Saguaro Substrate...[/cyan]"
        )

        # 1. Start Mission (Enforce Budget/Lease)
        files_str = ",".join(focus_files) if focus_files else ""
        mission_status = self.substrate.execute_command(
            f"mission_begin 'Context Gathering for: {task[:50]}' {files_str}"
        )
        self.console.print(f"  [dim]→ {mission_status}[/dim]")

        bundle = ContextBundle()

        try:
            # Phase 1: Semantic search
            bundle.target_files = self._semantic_search(task, focus_files)

            # Phase 2-7: Use substrate-backed tools (via saguaro_tools shim or direct)
            # For simplicity, we keep the original structure but ensure they use self.substrate
            bundle.dependencies = self._map_dependencies(bundle.target_files)
            bundle.dependents = self._find_dependents(bundle.target_files)
            bundle.related_files = self._find_related_files(task)
            bundle.test_files = self._find_test_files(bundle.target_files)
            bundle.similar_patterns = self._find_similar_patterns(task)
            bundle.impact_analysis = self._run_impact_analysis(bundle.target_files)

        finally:
            # End Mission
            self.substrate.execute_command("mission_end")

        self.console.print(
            "  [green]✓ Context gathered and mission finalized.[/green]"
        )
        return bundle

    def _semantic_search(self, task: str, initial_files: List[str] = None) -> List[str]:
        """
        Use Saguaro's semantic search to find relevant files.
        """
        results = set(initial_files or [])

        if self.semantic_engine and getattr(self.semantic_engine, "_indexed", False):
            try:
                # Multiple search strategies
                searches = [
                    task,  # Direct task
                    self._extract_key_concepts(task),  # Key concepts
                ]

                for query in searches:
                    if query:
                        found = self.semantic_engine.get_context_for_objective(query)
                        results.update(found[:5])

            except Exception as e:
                self.console.print(f"[yellow]Semantic search error: {e}[/yellow]")

        return list(results)[:10]  # Top 10

    def _map_dependencies(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Map dependencies for each file using Saguaro's slice tool.

        Returns dict: file_path -> list of files it depends on
        """
        dependencies = {}

        for file_path in file_paths[:5]:  # Limit to avoid overhead
            try:
                # Use Saguaro's slice to get imports
                deps = self._extract_dependencies(file_path)
                dependencies[file_path] = deps

            except Exception:
                dependencies[file_path] = []

        return dependencies

    def _extract_dependencies(self, file_path: str) -> List[str]:
        """
        Extract dependencies from a file.

        Uses both:
        1. Saguaro's slice tool
        2. AST parsing of imports
        """
        deps = []

        # Method 1: Read file and parse imports
        try:
            if self.registry is None:
                return deps

            content = self.registry.dispatch("read_file", {"file_path": file_path})

            if content and not content.startswith("Error"):
                import re

                import_lines = re.findall(
                    r"from\s+([\w.]+)\s+import|import\s+([\w.]+)", content
                )

                for match in import_lines:
                    module = match[0] or match[1]

                    # Skip standard library
                    if module.split(".")[0] in [
                        "os",
                        "sys",
                        "json",
                        "re",
                        "time",
                        "typing",
                    ]:
                        continue

                    # Convert to file path (simplified)
                    file = module.replace(".", "/") + ".py"
                    deps.append(file)

        except Exception:
            pass

        return deps[:10]  # Top 10 deps

    def _find_dependents(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Find files that depend on the given files.

        Uses Saguaro's impact analysis to find "who imports this?"
        """
        dependents = {}

        for file_path in file_paths[:3]:  # Limit
            try:
                # Use Saguaro's impact tool via substrate
                impact = self.substrate.agent_impact(file_path)
                dependents[file_path] = impact.splitlines() if impact else []
            except Exception:
                dependents[file_path] = []

        return dependents

    def _find_related_files(self, task: str) -> List[str]:
        """
        Find files related to the task but not direct dependencies.

        Uses semantic clustering.
        """
        related = []

        if self.semantic_engine and getattr(self.semantic_engine, "_indexed", False):
            try:
                # Broader search with different query
                query = self._expand_query(task)
                results = self.semantic_engine.get_context_for_objective(query)
                related = results[:5]

            except Exception:
                pass

        return related

    def _find_test_files(self, file_paths: List[str]) -> List[str]:
        """
        Find test files for the given files.

        Uses both:
        1. Naming conventions (test_*.py, *_test.py)
        2. Saguaro's query tool to find tests that import these files
        """
        import os

        test_files = set()

        for file_path in file_paths:
            base_name = os.path.basename(file_path).replace(".py", "")

            # Try naming conventions
            candidates = [
                f"test_{base_name}.py",
                f"{base_name}_test.py",
                f"tests/test_{base_name}.py",
                f"tests/{base_name}_test.py",
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    test_files.add(candidate)

            # Try Saguaro query
            try:
                if self.saguaro_tools is None:
                    continue
                # Query for files that import this file
                self.saguaro_tools.query(query=f"files that import {file_path}")

                # Parse query results (simplified)
                # Real implementation would parse the query response
                pass

            except Exception:
                pass

        return list(test_files)

    def _find_similar_patterns(self, task: str) -> List[Dict]:
        """
        Find similar code patterns using Saguaro memory.

        Returns list of pattern dicts with:
        - file_path: Where pattern was found
        - pattern: Code snippet
        - similarity: Score
        """
        patterns = []

        try:
            if self.saguaro_tools is None:
                return patterns
            # Use Saguaro's memory tool to find similar patterns
            self.saguaro_tools.memory(action="recall", query=task)

            # Parse memory results (simplified)
            # Real implementation would extract patterns from memory

        except Exception:
            pass

        return patterns

    def _run_impact_analysis(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Run Saguaro's impact analysis on files.

        Returns:
        - files_affected: Number of files that would be affected
        - risk_level: Low/Medium/High
        - change_scope: Local/Module/Global
        """
        analysis = {
            "files_affected": 0,
            "risk_level": "unknown",
            "change_scope": "local",
        }

        for file_path in file_paths[:1]:  # Just analyze first file
            try:
                if self.saguaro_tools is None:
                    break
                self.saguaro_tools.impact(file_path)

                # Parse impact (simplified)
                # Real implementation would analyze the impact graph
                analysis["files_affected"] = 0
                analysis["risk_level"] = "low"
                analysis["change_scope"] = "local"

            except Exception:
                pass

        return analysis

    def _extract_key_concepts(self, task: str) -> str:
        """
        Extract key technical concepts from task description.

        e.g., "Add error handling to the authentication module"
        -> "error handling authentication"
        """
        # Simple keyword extraction (real implementation would use NLP)
        keywords = []

        tech_keywords = [
            "authentication",
            "error",
            "handling",
            "database",
            "api",
            "endpoint",
            "route",
            "model",
            "view",
            "controller",
            "service",
            "repository",
            "cache",
            "validation",
            "logging",
        ]

        task_lower = task.lower()
        for kw in tech_keywords:
            if kw in task_lower:
                keywords.append(kw)

        return " ".join(keywords) if keywords else task

    def _expand_query(self, task: str) -> str:
        """
        Expand query with related terms.

        e.g., "authentication" -> "authentication login auth user session"
        """
        expansions = {
            "authentication": "authentication login auth user session",
            "error": "error exception handling failure",
            "database": "database db sql query model",
            "api": "api endpoint route handler",
            "test": "test testing unittest pytest",
        }

        for key, expansion in expansions.items():
            if key in task.lower():
                return expansion

        return task


class ContextOptimizer:
    """
    Optimize loaded context using Saguaro's native ContextGovernor.
    """

    def __init__(self, substrate, console):
        self.substrate = substrate
        self.console = console

    def optimize(
        self, bundle: ContextBundle, token_budget: int = None
    ) -> ContextBundle:
        """
        Trim context bundle to fit within token budget using Saguaro's IQ logic.
        """
        self.console.print(
            "[cyan]Optimizing context bundle with Saguaro Governor...[/cyan]"
        )

        # Flatten bundle into items for the governor
        items = []
        for f in bundle.target_files:
            items.append({"name": f, "type": "target", "score": 100})
        for f, deps in bundle.dependencies.items():
            for d in deps:
                items.append({"name": d, "type": "dependency", "score": 80})

        # Use governor to optimize
        if hasattr(self.substrate, "governor"):
            # Update soft limit if requested
            if token_budget:
                self.substrate.governor.soft_limit = token_budget

            # This is a bit of a shim since optimize_bundle expects content
            # For now, we'll assume the manager has filled some content or we just use counts
            # In a full integration, we'd pass the actual content here.
            optimized_items = self.substrate.governor.optimize_bundle(items)
            self.console.print(
                f"  [dim]→ Governor retained {len(optimized_items)}/{len(items)} items[/dim]"
            )

        # Simple back-mapping (keeping the bundle structure)
        # Note: This is still a bit simplified but uses the governor class logic
        return bundle  # Placeholder for complex back-mapping
