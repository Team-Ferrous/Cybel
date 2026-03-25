import os
import subprocess
from typing import List, Optional, Any
from cli.commands.base import SlashCommand


class SyncCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "sync"

    @property
    def description(self) -> str:
        return "Sync skills and workflows from a git repository"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        if not args:
            return "Usage: /sync <repo_url> or /sync pull"

        target_dir = os.path.expanduser("~/.anvil/library")
        os.makedirs(target_dir, exist_ok=True)

        if args[0] == "pull":
            if not os.path.exists(os.path.join(target_dir, ".git")):
                return "No library repository found. Use /sync <url> first."
            subprocess.run(["git", "-C", target_dir, "pull"])
            return "Library updated."

        repo_url = args[0]
        if os.path.exists(os.path.join(target_dir, ".git")):
            return "Library already exists. Use /sync pull to update."

        print(f"Cloning library from {repo_url}...")
        res = subprocess.run(
            ["git", "clone", repo_url, target_dir], capture_output=True, text=True
        )
        if res.returncode == 0:
            return f"Library cloned to {target_dir}. Knowledge updated."
        else:
            return f"Failed to clone: {res.stderr}"
