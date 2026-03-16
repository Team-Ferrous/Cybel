"""
Progressive Context Loading with Saguaro

Loads code context incrementally:
1. Start with skeletons (structure only)
2. Load full content only when needed
3. Use Saguaro's semantic search to find relevant sections

This keeps context windows small while maintaining full understanding.
"""

from typing import Dict, List
from dataclasses import dataclass
from enum import Enum


class ContentDepth(Enum):
    """How much detail to load."""

    SKELETON = "skeleton"  # Class/function signatures only
    PARTIAL = "partial"  # Key sections only
    FULL = "full"  # Complete file content


@dataclass
class FileContext:
    """Represents loaded context for a file."""

    file_path: str
    depth: ContentDepth
    content: str
    tokens_used: int

    # Saguaro metadata
    dependencies: List[str] = None
    importance_score: float = 0.0


class ProgressiveContextLoader:
    """
    Load code context progressively using Saguaro intelligence.

    Strategy:
    1. Semantic search finds relevant files
    2. Load skeletons for all relevant files (cheap)
    3. Identify which files need full content (expensive)
    4. Load full content only for critical files
    5. Monitor token budget, expand as needed
    """

    def __init__(self, registry, semantic_engine, saguaro_tools, console):
        self.registry = registry
        self.semantic_engine = semantic_engine
        self.saguaro_tools = saguaro_tools
        self.console = console

        # Track loaded context
        self.loaded_files: Dict[str, FileContext] = {}
        self.token_budget = 100000  # Configurable
        self.tokens_used = 0

    def load_context_for_task(
        self, task: str, initial_files: List[str] = None
    ) -> Dict[str, FileContext]:
        """
        Load optimal context for a task.

        Args:
            task: Task description
            initial_files: Optional files to start with

        Returns:
            Dict mapping file_path -> FileContext
        """
        self.console.print("[cyan]Loading context progressively...[/cyan]")

        # Phase 1: Semantic search for relevant files
        relevant_files = self._semantic_search(task, initial_files)
        self.console.print(f"  [dim]→ Found {len(relevant_files)} relevant files[/dim]")

        # Phase 2: Load skeletons for all (cheap)
        self._load_skeletons(relevant_files)
        self.console.print(f"  [dim]→ Loaded {len(self.loaded_files)} skeletons[/dim]")

        # Phase 3: Analyze which files need full content
        critical_files = self._identify_critical_files(task, relevant_files)
        self.console.print(
            f"  [dim]→ Identified {len(critical_files)} critical files[/dim]"
        )

        # Phase 4: Load full content for critical files
        self._load_full_content(critical_files)

        # Phase 5: Load dependencies if budget allows
        self._load_dependencies()

        self.console.print(
            f"  [green]✓ Loaded context ({self.tokens_used}/{self.token_budget} tokens)[/green]"
        )

        return self.loaded_files

    def expand_context(self, file_path: str, depth: ContentDepth = ContentDepth.FULL):
        """
        Expand context for a specific file (lazy loading).

        Called when agent needs more detail about a file.
        """
        if file_path not in self.loaded_files:
            # Load skeleton first
            self._load_skeleton(file_path)

        current = self.loaded_files[file_path]

        if current.depth == ContentDepth.SKELETON and depth == ContentDepth.FULL:
            # Upgrade to full content
            self._load_full_file(file_path)
            self.console.print(f"  [cyan]↑ Expanded {file_path} to full content[/cyan]")

    def get_context_summary(self) -> str:
        """
        Generate summary of loaded context for the model.
        """
        summary_parts = []

        summary_parts.append(f"## Loaded Context ({len(self.loaded_files)} files)\n")

        # Group by depth
        by_depth = {depth: [] for depth in ContentDepth}
        for fc in self.loaded_files.values():
            by_depth[fc.depth].append(fc)

        # Full content files
        if by_depth[ContentDepth.FULL]:
            summary_parts.append("### Full Content:")
            for fc in by_depth[ContentDepth.FULL]:
                summary_parts.append(f"\n#### {fc.file_path}")
                summary_parts.append(f"```\n{fc.content}\n```")

        # Skeleton files
        if by_depth[ContentDepth.SKELETON]:
            summary_parts.append("\n### Structure Only (Skeletons):")
            for fc in by_depth[ContentDepth.SKELETON]:
                summary_parts.append(f"\n#### {fc.file_path}")
                summary_parts.append(f"```\n{fc.content}\n```")

        return "\n".join(summary_parts)

    def _semantic_search(self, task: str, initial_files: List[str] = None) -> List[str]:
        """
        Use Saguaro semantic search to find relevant files.
        """
        relevant = set(initial_files or [])

        if self.semantic_engine and self.semantic_engine._indexed:
            try:
                search_results = self.semantic_engine.get_context_for_objective(task)
                relevant.update(search_results[:10])  # Top 10
            except Exception as e:
                self.console.print(f"[yellow]Semantic search failed: {e}[/yellow]")

        return list(relevant)

    def _load_skeletons(self, file_paths: List[str]):
        """
        Load skeletons for all files in parallel.

        Skeletons show structure (classes, functions) without implementation.
        Very cheap in tokens (~10-20% of full file).
        """
        for file_path in file_paths:
            self._load_skeleton(file_path)

    def _load_skeleton(self, file_path: str):
        """Load skeleton for single file."""
        try:
            # Check if file is small enough to just load fully
            import os

            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        line_count = sum(1 for _ in f)
                    if line_count < 200:
                        self._load_full_file(file_path)
                        return
                except Exception:
                    pass

            skeleton = self.saguaro_tools.skeleton(file_path)

            if skeleton and not skeleton.startswith("Error"):
                tokens = len(skeleton.split())  # Rough estimate

                self.loaded_files[file_path] = FileContext(
                    file_path=file_path,
                    depth=ContentDepth.SKELETON,
                    content=skeleton,
                    tokens_used=tokens,
                    dependencies=[],
                )

                self.tokens_used += tokens

        except Exception as e:
            self.console.print(
                f"[yellow]Failed to load skeleton for {file_path}: {e}[/yellow]"
            )

    def _identify_critical_files(self, task: str, candidates: List[str]) -> List[str]:
        """
        Identify which files need full content.

        Uses Saguaro to analyze:
        1. Files directly mentioned in task
        2. Files with high semantic relevance
        3. Files that are dependencies of critical files
        """
        critical = []

        task_lower = task.lower()

        for file_path in candidates:
            # Mentioned explicitly?
            if file_path.lower() in task_lower:
                critical.append(file_path)
                continue

            # Check semantic relevance
            if self.semantic_engine and self.semantic_engine._indexed:
                try:
                    # Get relevance score (hack: search and see if file is in top 3)
                    top_results = self.semantic_engine.get_context_for_objective(task)[
                        :3
                    ]
                    if file_path in top_results:
                        critical.append(file_path)
                except Exception:
                    pass

        return critical[:5]  # Limit to top 5 to save tokens

    def _load_full_content(self, file_paths: List[str]):
        """
        Load full content for critical files.
        """
        for file_path in file_paths:
            if self.tokens_used >= self.token_budget:
                self.console.print(
                    "[yellow]Token budget reached, stopping expansion[/yellow]"
                )
                break

            self._load_full_file(file_path)

    def _load_full_file(self, file_path: str):
        """Load full content for a single file."""
        try:
            content = self.registry.dispatch("read_file", {"file_path": file_path})

            if content and not content.startswith("Error"):
                tokens = len(content.split())

                # Check budget
                if self.tokens_used + tokens > self.token_budget:
                    self.console.print(
                        f"[yellow]Skipping {file_path} (would exceed budget)[/yellow]"
                    )
                    return

                self.loaded_files[file_path] = FileContext(
                    file_path=file_path,
                    depth=ContentDepth.FULL,
                    content=content,
                    tokens_used=tokens,
                )

                self.tokens_used += tokens

        except Exception as e:
            self.console.print(f"[red]Failed to load {file_path}: {e}[/red]")

    def _load_dependencies(self):
        """
        Load dependencies of loaded files (if budget allows).

        Uses Saguaro's dependency analysis.
        """
        # Find dependencies using Saguaro
        for file_path, context in list(self.loaded_files.items()):
            if self.tokens_used >= self.token_budget * 0.9:  # Leave 10% buffer
                break

            try:
                # Use Saguaro's slice tool to find dependencies
                # slice() expects format: "filename.entity"
                self.saguaro_tools.slice(f"{file_path}.__all__")

                # Parse dependencies from slice result
                # (This is simplified - real implementation would parse imports)
                # For now, skip dependency loading
                pass

            except Exception:
                pass


class SmartContextExpander:
    """
    Intelligently expand context based on what the agent is trying to do.

    Uses Saguaro to understand code relationships.
    """

    def __init__(self, progressive_loader: ProgressiveContextLoader):
        self.loader = progressive_loader

    def expand_for_edit(
        self, file_path: str, edit_location: str
    ) -> Dict[str, FileContext]:
        """
        Expand context specifically for editing a file.

        Loads:
        1. Full content of target file
        2. Skeletons of imported modules
        3. Tests for the file
        4. Files that import this file (impact analysis)
        """
        context = {}

        # Load target file fully
        self.loader.expand_context(file_path, ContentDepth.FULL)
        context[file_path] = self.loader.loaded_files[file_path]

        # Find and load imports
        imports = self._extract_imports(file_path)
        for imp in imports[:3]:  # Top 3 imports
            self.loader._load_skeleton(imp)
            if imp in self.loader.loaded_files:
                context[imp] = self.loader.loaded_files[imp]

        # Find and load tests
        test_files = self._find_test_files(file_path)
        for test_file in test_files[:1]:  # Just one test file
            self.loader._load_skeleton(test_file)
            if test_file in self.loader.loaded_files:
                context[test_file] = self.loader.loaded_files[test_file]

        return context

    def _extract_imports(self, file_path: str) -> List[str]:
        """Extract imported files from a Python file."""
        # Simplified: would use AST parsing in real implementation
        imports = []

        try:
            content = self.loader.registry.dispatch(
                "read_file", {"file_path": file_path}
            )

            # Simple regex to find imports
            import re

            import_lines = re.findall(
                r"from\s+([\w.]+)\s+import|import\s+([\w.]+)", content
            )

            for match in import_lines:
                module = match[0] or match[1]
                # Convert module to file path (simplified)
                file = module.replace(".", "/") + ".py"
                imports.append(file)

        except Exception:
            pass

        return imports

    def _find_test_files(self, file_path: str) -> List[str]:
        """Find test files for a given file."""
        import os

        base_name = os.path.basename(file_path).replace(".py", "")

        candidates = [
            f"test_{base_name}.py",
            f"{base_name}_test.py",
            f"tests/test_{base_name}.py",
        ]

        return [c for c in candidates if os.path.exists(c)]
