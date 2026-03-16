from pathlib import Path


class ProjectMemory:
    """
    Manages the persistent project memory (e.g., .anvil/memory.md).
    This is a human-readable/agent-writable markdown file that persists across sessions.
    """

    DEFAULT_PATH = ".anvil/memory.md"
    TEMPLATE = """# Project Memory Bank

## Project Manifesto
*Architecture patterns, banned libraries, style guide.*
- 

## Active Tasks
*What is currently being worked on.*
- 

## Learned Lessons
*"Don't use subprocess.run here, use os.system because of X."*
- 
"""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir).resolve()
        self.internal_path = self.root_dir / ".anvil/memory.md"
        self.public_path = self.root_dir / "GRANITE.md"
        self._ensure_exists()

    def _ensure_exists(self):
        """Ensures the internal memory file exists."""
        if not self.internal_path.exists():
            self.internal_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.internal_path, "w", encoding="utf-8") as f:
                f.write(self.TEMPLATE)

    def read_internal(self) -> str:
        """Reads the internal memory bank."""
        try:
            return self.internal_path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def read_public(self) -> str:
        """Reads the public GRANITE.md if it exists."""
        if self.public_path.exists():
            return self.public_path.read_text(encoding="utf-8")
        return ""

    def read_combined(self) -> str:
        """Reads both and combines them for the agent context."""
        internal = self.read_internal()
        public = self.read_public()

        parts = []
        if public:
            parts.append(f"--- PUBLIC MEMORY BANK (GRANITE.md) ---\n{public}")
        if internal:
            parts.append(
                f"--- INTERNAL MEMORY BANK (.anvil/memory.md) ---\n{internal}"
            )

        return "\n\n".join(parts)

    def update_internal(self, content: str):
        """Overwrites the internal memory bank."""
        self.internal_path.write_text(content, encoding="utf-8")

    def append_lesson(self, lesson: str):
        """Appends a lesson to the 'Learned Lessons' section of the internal memory."""
        content = self.read_internal()
        marker = "## Learned Lessons"
        if marker in content:
            parts = content.split(marker)
            # Find the bullet list or end of section
            parts[1] = f"\n- {lesson}\n" + parts[1].lstrip("- ").lstrip()
            new_content = marker.join(parts)
            self.update_internal(new_content)
        else:
            self.update_internal(content + f"\n## Learned Lessons\n- {lesson}\n")
