"""Entry Point Detector
Identifies runtime entry points like CLI commands, API routes, and main blocks.
"""

import ast
import os
from typing import Any

from saguaro.query.corpus_rules import canonicalize_rel_path
from saguaro.utils.file_utils import build_corpus_manifest


class EntryPointDetector:
    """Provide EntryPointDetector support."""
    def __init__(self, root_dir: str) -> None:
        """Initialize the instance."""
        self.root_dir = os.path.abspath(root_dir)

    def detect(self) -> list[dict[str, Any]]:
        """Handle detect."""
        entry_points: list[dict[str, Any]] = []
        seen: set[tuple[str, str, int, str | None]] = set()
        manifest = build_corpus_manifest(self.root_dir)
        for path in manifest.files:
            rel_path = canonicalize_rel_path(path, repo_path=self.root_dir)
            if not path.endswith(".py"):
                continue
            if rel_path.lower().startswith(("tests/", "test/")) or "/tests/" in rel_path.lower():
                continue
            for entry in self._scan_file(path):
                key = (
                    str(entry.get("type") or ""),
                    str(entry.get("file") or ""),
                    int(entry.get("line") or 0),
                    str(entry.get("name") or "") or None,
                )
                if key in seen:
                    continue
                seen.add(key)
                entry_points.append(entry)
        return entry_points

    def _scan_file(self, path: str) -> list[dict]:
        found = []
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            tree = ast.parse(content, filename=path)

            for node in ast.walk(tree):
                # 1. Main block
                if isinstance(node, ast.If) and (
                    isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == "__name__"
                    and isinstance(node.test.comparators[0], ast.Constant)
                    and node.test.comparators[0].value == "__main__"
                ):
                    found.append(
                        {"type": "main_block", "file": path, "line": node.lineno}
                    )

                # 2. Flask/FastAPI Routes (Decorator heuristics)
                if isinstance(node, ast.FunctionDef):
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Call):
                            # @app.route() or @router.get()
                            if isinstance(dec.func, ast.Attribute):
                                if dec.func.attr in [
                                    "route",
                                    "get",
                                    "post",
                                    "put",
                                    "delete",
                                ]:
                                    found.append(
                                        {
                                            "type": "api_route",
                                            "file": path,
                                            "line": node.lineno,
                                            "name": node.name,
                                        }
                                    )

                # 3. Click/Argparse (Heuristic)
                # Checking for click.command()
                if isinstance(node, ast.FunctionDef):
                    for dec in node.decorator_list:
                        if (
                            isinstance(dec, ast.Attribute)
                            and dec.attr == "command"
                            or (
                                isinstance(dec, ast.Call)
                                and isinstance(dec.func, ast.Attribute)
                                and dec.func.attr == "command"
                            )
                        ):
                            found.append(
                                {
                                    "type": "cli_command",
                                    "file": path,
                                    "line": node.lineno,
                                    "name": node.name,
                                }
                            )

        except Exception:
            pass
        return found

    def _normalize_relpath(self, path: str) -> str:
        rel = os.path.relpath(path, self.root_dir)
        rel = rel.replace("\\", "/")
        if rel.startswith("./"):
            rel = rel[2:]
        return rel

    def _is_ignored_path(self, rel_path: str) -> bool:
        return False
