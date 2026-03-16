from core.ollama_client import DeterministicOllama
import os
from typing import List, Dict, Any, Optional
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate


class SemanticEngine:
    """
    Deep code understanding engine.
    Wraps SaguaroSubstrate and adds higher-level semantic analysis.
    """

    def __init__(
        self, root_dir: str = ".", brain: Optional[DeterministicOllama] = None
    ):
        self.substrate = SaguaroSubstrate(root_dir=root_dir)
        self.root_dir = root_dir
        self.brain = brain
        self._indexed = False

    def analyze_workspace(self, force: bool = False) -> Dict[str, Any]:
        """
        Checks if Saguaro indexing is complete.

        NOTE: Actual Saguaro indexing is handled by EnvironmentManager.ensure_ready()
        at REPL startup. This method just verifies the index exists.
        """
        if self._indexed and not force:
            return {"status": "already_indexed"}

        saguaro_dir = os.path.join(self.root_dir, ".saguaro")

        # Check if Saguaro index exists (created by EnvironmentManager)
        if not os.path.exists(saguaro_dir):
            return {
                "status": "not_initialized",
                "message": "Saguaro not initialized. Run /saguaro init or restart REPL.",
            }

        # Get file count from tracking.json
        indexed_files = 0
        try:
            import json

            tracking_path = os.path.join(saguaro_dir, "tracking.json")
            if os.path.exists(tracking_path):
                with open(tracking_path, "r") as f:
                    tracking = json.load(f)
                    indexed_files = len(tracking)
        except Exception:
            pass

        self._indexed = True

        return {
            "root": self.root_dir,
            "status": "success",
            "indexed_via": "Saguaro",
            "indexed_files": indexed_files,
        }

    def get_context_for_objective(self, objective: str) -> List[str]:
        """Identifies relevant files using Saguaro semantic search."""
        output = self.substrate.agent_query(objective, k=5)

        # Parse Saguaro query output to extract file paths
        # Format: [rank] [score] name (type) \n Path: file:line
        relevant_files = []
        import re

        path_matches = re.findall(r"Path: (.*?):", output)
        for path in path_matches:
            if path not in relevant_files:
                relevant_files.append(path)

        return relevant_files

    def get_impact_graph(self, file_path: str) -> str:
        """Traces dependencies to predict side effects."""
        return self.substrate.agent_impact(file_path)

    def get_skeleton(self, file_path: str) -> str:
        """Returns structural overview of a file."""
        return self.substrate.agent_skeleton(file_path)

    def get_slice(self, file_path: str, entity_name: str) -> str:
        """Returns specific logic from a file."""
        return self.substrate.agent_slice(file_path, entity_name)
