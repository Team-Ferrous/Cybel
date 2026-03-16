"""Utilities for graph."""

import logging
import os
from typing import Any

from saguaro.refactor.planner import RefactorPlanner

from .base import BaseEngine

logger = logging.getLogger(__name__)


class CodeGraphEngine(BaseEngine):
    """Validates code structure using dependency graphs.
    Checks for:
    - Circular dependencies
    - Invalid imports (files not found).
    """

    def run(self, path_arg: str = ".") -> list[dict[str, Any]]:
        """Handle run."""
        _ = path_arg
        violations = []

        # Initialize planner to use its graph capabilities
        planner = RefactorPlanner(self.repo_path)

        # 1. Build Full Graph (or relevant subgraph based on policy)
        # For Sentinel, we might want to scan everything relative to modify/create actions
        # But here we do a full scan of the repo's python files
        all_files = []
        for root, _, files in os.walk(self.repo_path):
            if ".git" in root or "venv" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    all_files.append(os.path.join(root, f))

        graph = planner._build_dependency_graph(all_files)

        # 2. Check for Cycles
        cycles = planner.detect_cycles(graph)
        for cycle in cycles:
            # Format cycle for display
            cycle_str = " -> ".join([os.path.basename(p) for p in cycle])
            violations.append(
                {
                    "file": cycle[0],  # Blame the start of the cycle
                    "line": 1,  # Whole file issue
                    "rule_id": "GRAPH_001_CYCLE",
                    "message": f"Circular dependency detected: {cycle_str}",
                    "severity": "error",
                    "aal": "AAL-1",
                    "domain": ["universal"],
                    "closure_level": "blocking",
                    "evidence_refs": [],
                }
            )

        return violations
