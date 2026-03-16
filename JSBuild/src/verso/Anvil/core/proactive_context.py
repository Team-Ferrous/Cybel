import os
from pathlib import Path
from typing import List, Dict
from rich.progress import Progress, SpinnerColumn, TextColumn


class ProactiveContextManager:
    """
    Scans the workspace proactively to build an initial mental model for the agent.
    """

    def __init__(self, root_dir: str = ".", max_files: int = 100):
        self.root_dir = Path(root_dir).resolve()
        self.max_files = max_files
        self.ignore_patterns = [
            ".git",
            "__pycache__",
            "venv",
            ".pytest_cache",
            ".ruff_cache",
            ".anvil",
        ]
        self.context_summary = ""

    def scan(self, console=None) -> str:
        """
        Performs a recursive scan of the workspace.
        """
        files_found = []

        if console:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    description="Scanning workspace structure...", total=None
                )
                files_found = self._gather_files()
                progress.update(task, completed=True, description="Scan complete.")
        else:
            files_found = self._gather_files()

        # Build summary
        summary = f"WORKSPACE SCAN REPORT (Root: {self.root_dir})\n"
        summary += f"Total relevant files indexed: {len(files_found)}\n\n"

        # Identify key files
        key_files = self._identify_key_files(files_found)
        summary += "KEY FILES IDENTIFIED:\n"
        for k, v in key_files.items():
            summary += f"- {k}: {v}\n"

        summary += "\nDIRECTORY STRUCTURE (Partial):\n"
        summary += self._render_tree(self.root_dir)

        self.context_summary = summary
        return summary

    def _gather_files(self) -> List[Path]:
        relevant_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Prune ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]

            for f in files:
                if len(relevant_files) >= self.max_files:
                    break
                relevant_files.append(Path(root) / f)
        return relevant_files

    def _identify_key_files(self, files: List[Path]) -> Dict[str, str]:
        keys = {}
        for p in files:
            name = p.name.lower()
            rel_path = p.relative_to(self.root_dir)
            if name == "anvil.md":
                keys["Memory Bank"] = str(rel_path)
            elif name == "readme.md":
                keys["README"] = str(rel_path)
            elif name in ["main.py", "app.py", "index.js", "init.py", "cli.py"]:
                keys["Entry Point"] = str(rel_path)
            elif name in ["requirements.txt", "package.json", "go.mod", "cargo.toml"]:
                keys["Dependencies"] = str(rel_path)
            elif name in [
                "setup.py",
                "pyproject.toml",
                "webpack.config.js",
                "tsconfig.json",
            ]:
                keys["Project Config"] = str(rel_path)
            elif name in ["dockerfile", "docker-compose.yml", "docker-compose.yaml"]:
                keys["Containerization"] = str(rel_path)
            elif name in [".env", "config.py", "settings.py"]:
                keys["Environment/Settings"] = str(rel_path)
        return keys

    def _render_tree(self, path: Path, prefix: str = "", depth: int = 0) -> str:
        if depth > 2:  # Keep it shallow for prompt context
            return ""

        tree = ""
        try:
            items = sorted(
                [d for d in path.iterdir() if d.name not in self.ignore_patterns]
            )
        except (PermissionError, FileNotFoundError):
            return f"{prefix}[Access Denied/Not Found]\n"

        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "

            # Add semantic tag for directories
            tag = ""
            if item.is_dir():
                tag = self._get_directory_tag(item)

            tree += f"{prefix}{connector}{item.name}{tag}\n"

            if item.is_dir():
                new_prefix = prefix + ("    " if is_last else "│   ")
                tree += self._render_tree(item, new_prefix, depth + 1)

        return tree

    def _get_directory_tag(self, path: Path) -> str:
        """Heuristic-based semantic tagging of directories."""
        name = path.name.lower()
        if name in ["src", "lib", "core"]:
            return " [Source Core]"
        if name in ["tests", "testing"]:
            return " [Test Suite]"
        if name in ["docs", "documentation"]:
            return " [Documentation]"
        if name in ["tools", "scripts", "utils"]:
            return " [Utilities/Infrastructure]"
        if name in ["cli", "cmd"]:
            return " [Entry Points/CLI Interface]"
        if name in ["config", "settings"]:
            return " [Configuration]"
        if name in ["data", "db", "migrations"]:
            return " [Data/Persistence]"
        return ""

    def get_memory_bank_content(self) -> str:
        """Reads both the Root ANVIL.md (Public) and Internal .anvil/memory.md."""
        from core.memory.project_memory import ProjectMemory

        pm = ProjectMemory(root_dir=str(self.root_dir))
        return pm.read_combined()

    def get_context_prompt(self) -> str:
        """Returns the summary formatted for a system prompt."""
        if not self.context_summary:
            return ""

        memory_bank = self.get_memory_bank_content()
        memory_section = ""
        if memory_bank:
            memory_section = f"\n### PROJECT MEMORY BANK (ANVIL.md)\n{memory_bank}\n"

        return f"\n### PROACTIVE WORKSPACE CONTEXT\nThis is a proactive scan of the workspace to orient you. Use this to understand where to look for code.\n\n{self.context_summary}\n{memory_section}"
