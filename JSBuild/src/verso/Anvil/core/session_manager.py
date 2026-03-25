import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class SessionManager:
    def __init__(self):
        self.session_dir = Path(os.path.expanduser("~/.anvil/sessions"))
        self.agent_states_dir = self.session_dir / "agent_states"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.agent_states_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, name: str, history: List[Dict], config: Dict = None):
        data = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "history": history,
            "config": config or {},
        }

        filepath = self.session_dir / f"{name}.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_session(self, name: str) -> Optional[Dict[str, Any]]:
        filepath = self.session_dir / f"{name}.json"

        if not filepath.exists():
            return None

        with open(filepath, "r") as f:
            return json.load(f)

    def list_sessions(self) -> List[str]:
        return [f.stem for f in self.session_dir.glob("*.json")]

    def save_agent_state(self, agent_instance: Any, session_name: str):
        """
        Saves the full state of an agent instance.
        'agent_instance' is expected to have a to_dict() method.
        """
        from core.agent import BaseAgent  # Imported here to avoid circular dependency

        if not isinstance(agent_instance, BaseAgent):
            raise TypeError(
                "agent_instance must be an instance of BaseAgent or a subclass."
            )

        state_data = agent_instance.to_dict()
        filepath = self.agent_states_dir / f"{session_name}.json"

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2)
        print(f"[bold green]Agent state saved to: {filepath}[/bold green]")

    def load_agent_state(self, session_name: str, console: Any = None) -> Optional[Any]:
        """
        Loads and reconstructs an agent instance from a saved state file.
        """
        from core.agent import BaseAgent  # Imported here to avoid circular dependency

        filepath = self.agent_states_dir / f"{session_name}.json"

        if not filepath.exists():
            print(
                f"[bold yellow]No agent state found for session: {session_name}[/bold yellow]"
            )
            return None

        with open(filepath, "r") as f:
            state_data = json.load(f)

        # Reconstruct the BaseAgent from the dictionary
        agent = BaseAgent.from_dict(state_data, console=console)
        print(f"[bold green]Agent state loaded from: {filepath}[/bold green]")
        return agent

    def list_agent_states(self) -> List[str]:
        """Lists available saved agent states."""
        return [f.stem for f in self.agent_states_dir.glob("*.json")]
