"""
Agent Mode Enum - Defines operational modes for the agentic loop.

Modes determine what tools are prioritized and what actions are permitted.
"""

from enum import Enum, auto


class AgentMode(Enum):
    """Operational mode for the agent's task execution."""

    PLANNING = auto()  # Research, design, create plans - blocks file writes
    EXECUTION = auto()  # Write code, make changes - full tool access
    VERIFICATION = (
        auto()
    )  # Test, validate, document - read-only for source, can modify tests
    IDLE = auto()  # Waiting for user input

    @property
    def color(self) -> str:
        """Rich-compatible color for UI rendering."""
        return {
            AgentMode.PLANNING: "blue",
            AgentMode.EXECUTION: "green",
            AgentMode.VERIFICATION: "yellow",
            AgentMode.IDLE: "dim white",
        }[self]

    @property
    def description(self) -> str:
        """Human-readable description of the mode."""
        return {
            AgentMode.PLANNING: "Researching and designing approach",
            AgentMode.EXECUTION: "Implementing changes",
            AgentMode.VERIFICATION: "Testing and validating",
            AgentMode.IDLE: "Waiting for input",
        }[self]

    @property
    def allowed_write_tools(self) -> list:
        """Tools that can modify files, restricted by mode."""
        if self == AgentMode.PLANNING:
            return []  # No file modifications during planning
        elif self == AgentMode.EXECUTION:
            return ["write_file", "edit_file", "write_files", "apply_patch"]
        elif self == AgentMode.VERIFICATION:
            return []  # Read-only during verification (tests are run, not modified)
        return []

    @property
    def prioritized_tools(self) -> list:
        """Tools that should be prioritized in this mode."""
        if self == AgentMode.PLANNING:
            return [
                "saguaro_query",
                "skeleton",
                "slice",
                "read_file",
                "web_search",
                "think",
            ]
        elif self == AgentMode.EXECUTION:
            return ["write_file", "edit_file", "run_command", "think"]
        elif self == AgentMode.VERIFICATION:
            return ["run_command", "verify", "think"]
        return []
