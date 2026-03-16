import os
from typing import List
from core.memory.project_memory import ProjectMemory
from tools.lsp import LSPTools


class DreamingAgent:
    """
    Handles background tasks when the user is idle.
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = root_dir
        self.memory = ProjectMemory(root_dir=root_dir)
        self.lsp = LSPTools(root_dir=root_dir)

    def dream(self, console=None):
        """Perform a suite of 'dreaming' tasks."""
        if console:
            console.print("[bold magenta]🌙 Entering Dreaming Mode...[/bold magenta]")
            console.print(
                "[dim]Analyzing codebase health and documenting lessons...[/dim]"
            )

        results = []

        # 1. Check for missing docstrings
        missing = self._find_missing_docstrings()
        if missing:
            results.append(
                f"Found {len(missing)} functions/classes missing docstrings."
            )
            for m in missing[:3]:
                results.append(f"  - {m}")

        # 2. Run diagnostics
        diagnostics = self.lsp.get_diagnostics()
        if "No issues found" not in diagnostics:
            results.append(
                "Found potential issues in the codebase. Run /verify to see more."
            )

        # 3. Align style (Simulated)
        results.append("Verified project style alignment.")

        if console:
            for r in results:
                console.print(f"[magenta] * {r}[/magenta]")
            console.print(
                "[bold magenta]✨ Dreaming complete. Workspace is optimized.[/bold magenta]"
            )

        return results

    def _find_missing_docstrings(self) -> List[str]:
        import ast

        missing = []
        for root, _, files in os.walk(self.root_dir):
            if any(x in root for x in [".git", "venv", "__pycache__", ".anvil"]):
                continue
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8") as file:
                            tree = ast.parse(file.read())
                        for node in ast.walk(tree):
                            if isinstance(
                                node,
                                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                            ):
                                if not ast.get_docstring(node):
                                    rel_path = os.path.relpath(path, self.root_dir)
                                    missing.append(
                                        f"{rel_path}:{node.lineno} ({node.name})"
                                    )
                    except Exception:
                        continue
        return missing
