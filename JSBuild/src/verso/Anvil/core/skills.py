import os

# yaml imported locally in SkillManager
from typing import Dict, List, Optional
from pathlib import Path


class Skill:
    def __init__(self, path: Path, metadata: Dict, content: str):
        self.path = path
        self.name = metadata.get("name")
        self.description = metadata.get("description")
        self.triggers = metadata.get("triggers", [])
        self.content = content


class SkillManager:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.discover_skills()

    def discover_skills(self):
        """Scans skill directories."""
        self.skills = {}
        paths = [
            Path(os.path.expanduser("~/.anvil/skills")),
            Path(os.getcwd()) / ".anvil/skills",
        ]

        for p in paths:
            if not p.exists():
                continue
            for item in p.iterdir():
                if item.is_dir():
                    skill_file = item / "SKILL.md"
                    if skill_file.exists():
                        self._load_skill(skill_file)

    def _load_skill(self, path: Path):
        try:
            content = path.read_text()
            # Split YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    metadata_str = parts[1]
                    body = parts[2]

                    metadata = {}
                    try:
                        import yaml

                        metadata = yaml.safe_load(metadata_str)
                    except ImportError:
                        # Simple regex fallback for basic key-value pairs
                        import re

                        for line in metadata_str.splitlines():
                            match = re.match(r"^(\w+):\s*(.*)$", line)
                            if match:
                                key, val = match.groups()
                                metadata[key.strip()] = val.strip()

                    if metadata and "name" in metadata:
                        self.skills[metadata["name"]] = Skill(path, metadata, body)
        except Exception as e:
            print(f"Error loading skill {path}: {e}")

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self) -> List[Dict]:
        return [
            {"name": s.name, "description": s.description, "triggers": s.triggers}
            for s in self.skills.values()
        ]
