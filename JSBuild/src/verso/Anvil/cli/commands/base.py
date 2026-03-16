from abc import ABC, abstractmethod
from typing import List, Optional, Any


class SlashCommand(ABC):
    """Abstract base class for all slash commands."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Primary command name (e.g., 'help')."""
        pass

    @property
    def aliases(self) -> List[str]:
        """Alternative names for the command."""
        return []

    @property
    def category(self) -> str:
        """High-level operator-facing category for help grouping."""
        return "general"

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description for help menu."""
        pass

    @abstractmethod
    def execute(self, args: List[str], context: Any) -> Optional[str]:
        """
        Execute the command.

        Args:
            args: List of string arguments.
            context: The REPL/Agent context (repl instance).

        Returns:
            Optional string result to display, or None.
        """
        pass
