from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from rich.console import Console
from core.prompts import PromptManager


class BasePhase(ABC):
    """
    Abstract base class for UnifiedChatLoop phases.
    """

    def __init__(self, loop: Any, prompt_manager: PromptManager, console: Console):
        self.loop = loop
        self.agent = loop.agent
        self.prompt_manager = prompt_manager
        self.console = console
        self.brain = loop.brain
        self.registry = loop.agent.registry

    @abstractmethod
    def execute(
        self, user_input: str, context: Dict[str, Any], dashboard: Optional[Any] = None
    ) -> Any:
        """Execute the phase logic."""
        pass
