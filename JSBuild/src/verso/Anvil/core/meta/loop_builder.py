import os
from core.agent import BaseAgent


class LoopBuilder:
    """
    Meta-agent capability to construct custom execution loops.
    Allows Anvil to build specialized mini-agents for specific recurring tasks.
    """

    TEMPLATE = """
import time
from typing import Any
from core.agent import BaseAgent
from rich.panel import Panel

class {class_name}:
    \"\"\"{docstring}\"\"\"
    
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.console = agent.console
        self.brain = agent.brain
        
    def run(self, input_data: Any) -> Any:
        self.console.print(Panel(f"Starting {class_name}", style="cyan"))
        
        # Generated Logic
        {logic}
        
        return "Done"
"""

    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.console = agent.console

    def generate_loop_code(self, goal: str, name: str = "CustomLoop") -> str:
        """Generate Python code for a custom loop based on the goal."""

        prompt = f"""
        You are a meta-programmer.
        Task: Create a Python class that implements a specific execution loop for: "{goal}"
        
        Requirements:
        1. Class Name: {name}
        2. Methods: __init__(self, agent), run(self, input_data)
        3. Use 'self.agent.registry.dispatch("tool_name", {{args}})' to call tools.
        4. Use 'self.agent.brain.chat([...])' for inference.
        5. Keep it simple and robust.
        
        Output ONLY the Python code for the class.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a Python expert specializing in agent architectures.",
            },
            {"role": "user", "content": prompt},
        ]

        code = ""
        for chunk in self.agent.brain.stream_chat(messages, temperature=0.2):
            code += chunk

        # Strip markdown if present
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

    def save_loop(self, code: str, filename: str) -> str:
        """Save the generated loop to core/loops/"""
        path = os.path.join("core", "loops", filename)
        with open(path, "w") as f:
            f.write(code)
        return path
