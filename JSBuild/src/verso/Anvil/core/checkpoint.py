import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class CheckpointManager:
    def __init__(self, project_path: str = None):
        self.project_path = project_path or os.getcwd()
        self.project_hash = hashlib.md5(self.project_path.encode()).hexdigest()
        self.checkpoint_dir = Path(
            os.path.expanduser(f"~/.anvil/checkpoints/{self.project_hash}")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, name: str = None, state: Dict[str, Any] = None) -> str:
        """
        Saves a checkpoint.
        state: Dict containing 'history', 'config', etc.
        Returns the checkpoint ID/name.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not name:
            name = f"ckpt_{timestamp}"

        ckpt_path = self.checkpoint_dir / f"{name}.json"

        data = {
            "id": name,
            "timestamp": timestamp,
            "project_path": self.project_path,
            "state": state or {},
        }

        with open(ckpt_path, "w") as f:
            json.dump(data, f, indent=2)

        return name

    def load_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        ckpt_path = self.checkpoint_dir / f"{name}.json"
        if not ckpt_path.exists():
            return None

        with open(ckpt_path, "r") as f:
            return json.load(f)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        checkpoints = []
        for f in self.checkpoint_dir.glob("*.json"):
            try:
                with open(f, "r") as cf:
                    data = json.load(cf)
                    checkpoints.append(
                        {
                            "id": data.get("id"),
                            "timestamp": data.get("timestamp"),
                            "path": str(f),
                        }
                    )
            except Exception:
                pass
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)

    def create_git_snapshot(self) -> Optional[str]:
        """
        If in a git repo, creates a stash-like snapshot without modifying working tree?
        Or just commit hash.
        For now, returns None as placeholder for 'git stash create' logic.
        """
        # TODO: Implement git integration
        return None
