from abc import ABC, abstractmethod
from typing import Dict, Any


class Hook(ABC):
    """Base class for lifecycle hooks."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Hook name for identification."""
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute hook logic, return modified context.
        Note: We are using synchronous execution for now to simplify
        integration with the current synchronous agent loops,
        but implementation can be updated to async if needed.
        """
        pass
