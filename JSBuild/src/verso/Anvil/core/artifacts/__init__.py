"""
Artifact System - Task artifacts for planning and documentation.

Implements:
- TaskArtifact: Living checklist (task.md)
- PlanArtifact: Implementation plan (implementation_plan.md)
- WalkthroughArtifact: Proof of work (walkthrough.md)
"""

from .task_artifact import TaskArtifact
from .compiled_mission_plan import CompiledMissionPlan
from .plan_artifact import PlanArtifact
from .walkthrough_artifact import WalkthroughArtifact
from .manager import ArtifactManager

__all__ = [
    "TaskArtifact",
    "CompiledMissionPlan",
    "PlanArtifact",
    "WalkthroughArtifact",
    "ArtifactManager",
]
