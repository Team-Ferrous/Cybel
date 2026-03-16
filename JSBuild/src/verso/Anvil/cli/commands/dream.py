from typing import List, Optional, Any
from cli.commands.base import SlashCommand
from core.dream.dreamer import DreamingAgent


class DreamCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "dream"

    @property
    def description(self) -> str:
        return "Manually trigger 'Dreaming Mode' to optimize codebase"

    def execute(self, args: List[str], context: Any) -> Optional[str]:
        # context is AgentREPL
        dreamer = DreamingAgent(root_dir=".")
        dreamer.dream(console=context.console)
        return "You feel refreshed."
