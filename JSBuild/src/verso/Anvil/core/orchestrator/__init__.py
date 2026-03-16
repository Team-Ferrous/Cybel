import json
import re
from typing import List, Dict, Any, Optional
from core.ollama_client import DeterministicOllama
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
from config.settings import MASTER_MODEL


class MissionDecomposer:
    """
    Handles mission decomposition, subagent delegation, and task tracking.
    """

    def __init__(self, brain: Optional[DeterministicOllama] = None):
        self.brain = brain or DeterministicOllama(MASTER_MODEL)
        self.saguaro = SaguaroSubstrate()
        self.task_history = []
        self.active_tasks = []

    def plan_mission(self, user_objective: str) -> List[Dict[str, Any]]:
        """Decomposes a user objective into a sequence of atomic tasks."""
        plan_prompt = f"""
        OBJECTIVE: {user_objective}
        
        You are the Master Architect. Break this into atomic, sequential tasks.
        Each task must specify:
        1. 'id': sequential integer
        2. 'role': researcher, architect, implementer, or validator
        3. 'task': specific instructions
        
        Return ONLY a JSON list of objects.
        """
        plan_raw = self.brain.generate(plan_prompt)
        return self._sanitize_json(plan_raw)

    def dispatch(
        self, role: str, task: str, context: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Dispatches a task to a specialized subagent."""
        # This will eventually use the SubAgent class or a similar mechanism
        from core.delegation import SubAgent  # Assuming this exists or will be unified

        sub_agent = SubAgent(role=role, task=task, context=context or self.task_history)
        return sub_agent.execute()

    def _sanitize_json(self, raw_text: str) -> List[Dict[str, Any]]:
        """Extracts JSON from model output."""
        try:
            # Simple extraction heuristic
            match = re.search(r"\[.*\]", raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(raw_text)
        except Exception:
            return []

    def run_mission(self, objective: str):
        """Main execution flow for a mission."""
        tasks = self.plan_mission(objective)
        for task_obj in tasks:
            result = self.dispatch(task_obj["role"], task_obj["task"])
            self.task_history.append({"task": task_obj["task"], "result": result})
            # Add validation logic here...
