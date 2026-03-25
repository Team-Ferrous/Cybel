# saguaro/agents/state.py
"""Utilities for state."""

import json
import os
import time
from typing import Any


class AgentStateManager:
    """Manages persistence of agent states across sessions.
    Allows for recovery and resume of complex multi-step missions.
    """

    def __init__(self, persistence_dir: str = ".saguaro/agents") -> None:
        """Initialize the instance."""
        self.persistence_dir = persistence_dir
        os.makedirs(self.persistence_dir, exist_ok=True)

    def save_checkpoint(self, agent_id: str, state: dict[str, Any]) -> str:
        """Saves the current state of an agent."""
        path = os.path.join(self.persistence_dir, f"{agent_id}.json")
        checkpoint = {"agent_id": agent_id, "timestamp": time.time(), "state": state}
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        return path

    def load_checkpoint(self, agent_id: str) -> dict:
        """Loads the previous state of an agent."""
        path = os.path.join(self.persistence_dir, f"{agent_id}.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}

    def list_checkpoints(self) -> list[str]:
        """Lists all stored agent checkpoints."""
        return [
            f.replace(".json", "")
            for f in os.listdir(self.persistence_dir)
            if f.endswith(".json")
        ]
