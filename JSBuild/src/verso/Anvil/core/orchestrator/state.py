from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from core.orchestrator.graph import TaskGraph


@dataclass
class OrchestratorState:
    """Persistent state for the AgentOrchestrator."""

    objective: str
    graph: TaskGraph = field(default_factory=TaskGraph)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "IDLE"
    global_context: Dict[str, Any] = field(default_factory=dict)
    ownership_state: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "graph": self.graph.to_json(),
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "status": self.status,
            "global_context": self.global_context,
            "ownership_state": self.ownership_state,
        }


class OrchestratorStateManager:
    """Manages persistence of OrchestratorState."""

    def __init__(self, persistence_dir: str = ".agent/orchestrator"):
        self.persistence_dir = persistence_dir
        import os

        os.makedirs(persistence_dir, exist_ok=True)

    def save_state(self, state: OrchestratorState):
        import json
        import os

        # Simplistic filename for now
        filename = "state.json"
        path = os.path.join(self.persistence_dir, filename)
        with open(path, "w") as f:
            json.dump(state.to_json(), f, indent=2)

    def load_state(self) -> Optional[OrchestratorState]:
        import json
        import os
        from core.orchestrator.graph import TaskUnit, TaskType

        path = os.path.join(self.persistence_dir, "state.json")
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            data = json.load(f)

        state = OrchestratorState(
            objective=data["objective"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data["completed_at"]
                else None
            ),
            status=data["status"],
            global_context=data["global_context"],
            ownership_state=data.get("ownership_state", {}),
        )

        # Hydrate graph
        for tid, tdata in data["graph"]["tasks"].items():
            task = TaskUnit(
                id=tdata["id"],
                parent_id=tdata["parent_id"],
                type=TaskType(tdata["type"]),
                instruction=tdata["instruction"],
                context_files=tdata.get("context_files", []),
                output_schema=tdata.get("output_schema", {}),
                dependencies=tdata.get("dependencies", []),
                semantic_context=tdata.get("semantic_context", {}),
                status=tdata.get("status", "PENDING"),
                result=tdata.get("result"),
                assigned_agent_id=tdata.get("assigned_agent_id"),
                owned_files=tdata.get("owned_files", []),
                read_files=tdata.get("read_files", []),
                phase_id=tdata.get("phase_id"),
                ownership_workset_id=tdata.get("ownership_workset_id"),
            )
            state.graph.add_task(task)

        return state
