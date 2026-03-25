"""Utilities for base."""

from abc import ABC, abstractmethod
from typing import Any

from saguaro.client import SAGUAROClient
from saguaro.context import Context


class Agent(ABC):
    """Base class for SAGUARO specialized agents."""

    def __init__(self, name: str, role: str) -> None:
        """Initialize the instance."""
        self.name = name
        self.role = role
        self.client = SAGUAROClient()  # Access to SAGUARO Core

    @abstractmethod
    def run(self, context: Context, **kwargs) -> dict[str, Any]:
        """Execute the agent's main logic."""
        pass

    def log_activity(self, message: str) -> None:
        """Handle log activity."""
        print(f"[{self.role.upper()}] {self.name}: {message}")
