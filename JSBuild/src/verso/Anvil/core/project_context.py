import os
from pathlib import Path
from typing import List


class ProjectContextManager:
    CONTEXT_FILENAME = "GRANITE.md"

    MAX_DEPTH = 5

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.ignore_patterns = self._load_ignore_patterns()

    def _load_ignore_patterns(self) -> List[str]:
        patterns = []
        try:
            ignore_file = Path(os.getcwd()) / ".anvilignore"
            if ignore_file.exists():
                with open(ignore_file, "r") as f:
                    patterns = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.startswith("#")
                    ]
        except Exception:
            pass
        return patterns

    def is_ignored(self, filename: str) -> bool:
        import fnmatch

        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False

    def get_context(self) -> str:
        """
        Gathers context from GRANITE.md files in the hierarchy.
        Strategy: ~/.anvil/GRANITE.md -> Parent dirs -> Current dir
        """
        if not self.enabled:
            return ""

        context_parts = []

        # 1. Global context
        global_path = Path(os.path.expanduser(f"~/.anvil/{self.CONTEXT_FILENAME}"))
        if global_path.exists():
            context_parts.append(
                f"--- Global Context ({global_path}) ---\n{global_path.read_text()}"
            )

        # 2. Project hierarchy (walking up from CWD)
        try:
            cwd = Path(os.getcwd()).resolve()
            found_contexts = []

            # Walk up from CWD
            curr = cwd
            for _ in range(self.MAX_DEPTH):
                local_path = curr / self.CONTEXT_FILENAME
                if local_path.exists():
                    found_contexts.append(local_path)

                if curr.parent == curr:  # Root
                    break
                curr = curr.parent

            # Reverse so we have Top-most -> Bottom-most (CWD)
            # Actually, standard override usually means Child overrides Parent.
            # But for context, we usually want to append/merge. Top first allows General -> Specific.
            # Let's verify: Global -> (Root -> Subdir).
            # We collected CWD -> Parent. So reversing gives Parent -> CWD.
            for ctx_path in reversed(found_contexts):
                context_parts.append(
                    f"--- Project Context ({ctx_path}) ---\n{ctx_path.read_text()}"
                )

        except Exception as e:
            print(f"[WARN] Failed to load project context: {e}")

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)
