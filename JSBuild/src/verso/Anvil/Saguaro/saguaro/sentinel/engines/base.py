"""Utilities for base."""

from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Abstract base class for analysis engines."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self.repo_path = repo_path
        self.policy_config = {}

    def set_policy(self, config: dict[str, Any]) -> None:
        """Set policy."""
        self.policy_config = config

    @abstractmethod
    def run(self, path_arg: str = ".") -> list[dict[str, Any]]:
        """Run the engine and return a list of violations.
        Each violation is a dict with:
        - file: str
        - line: int
        - rule_id: str
        - message: str
        - severity: str ('P0'..'P3' or legacy values)
        - aal: str (optional)
        - domain: list[str] (optional)
        - closure_level: str (optional, advisory|guarded|blocking)
        - evidence_refs: list[str] (optional)
        - context: str (optional).
        """
        pass

    def fix(self, violation: dict[str, Any]) -> bool:
        """Attempt to fix a specific violation.
        Returns True if fixed, False otherwise.
        """
        return False
