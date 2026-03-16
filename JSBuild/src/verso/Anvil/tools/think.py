"""
Think Tool - Allows the agent to pause and reflect during tool execution.

Use this tool when:
- Complex multi-step operations require mid-execution reflection
- Need to verify progress before continuing
- Uncertain about the next action
- After unexpected tool results
"""

from typing import Optional


class ThinkTool:
    """
    Allows the agent to explicitly pause and think during multi-tool execution.

    This implements the "interleaved thinking" pattern from Claude Code,
    enabling self-reflection between tool calls.
    """

    schema = {
        "name": "think",
        "description": "Pause to think and reflect on the current task progress. Use when uncertain, after unexpected results, or before complex decisions.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "What the agent is thinking about or reasoning through",
                },
                "question": {
                    "type": "string",
                    "description": "Optional question to answer through this thinking",
                },
                "type": {
                    "type": "string",
                    "enum": [
                        "understanding",
                        "planning",
                        "reasoning",
                        "reflection",
                        "correction",
                    ],
                    "default": "reasoning",
                    "description": "Type of thinking being performed",
                },
            },
            "required": ["thought"],
        },
    }

    def __init__(self, thinking_system=None):
        """
        Initialize the think tool.

        Args:
            thinking_system: Optional EnhancedThinkingSystem for recording thoughts
        """
        self.thinking_system = thinking_system
        self.thought_count = 0

    def execute(
        self, thought: str, question: Optional[str] = None, type: str = "reasoning"
    ) -> str:
        """
        Execute the think tool - records thought and provides acknowledgment.

        Args:
            thought: The content of the thinking
            question: Optional question being pondered
            type: Type of thinking (understanding, planning, reasoning, reflection, correction)

        Returns:
            Acknowledgment message with any guidance
        """
        self.thought_count += 1

        # Record in thinking system if available
        if self.thinking_system:
            try:
                from core.thinking import ThinkingType

                thinking_type = ThinkingType(type)
                self.thinking_system.think(thinking_type, thought)
            except (ImportError, ValueError):
                pass  # Continue even if thinking system isn't set up

        # Build response
        response_parts = [f"[Thought #{self.thought_count} recorded]"]

        if question:
            response_parts.append(f"Question being considered: {question}")

        # Provide guidance based on thinking type
        guidance = {
            "understanding": "Continue gathering context before acting.",
            "planning": "Finalize your approach, then proceed to execution.",
            "reasoning": "Work through the logic step-by-step.",
            "reflection": "Evaluate what's working and what needs adjustment.",
            "correction": "Identify the specific issue and how to fix it.",
        }

        if type in guidance:
            response_parts.append(f"Guidance: {guidance[type]}")

        response_parts.append("Continue with your next action.")

        return "\n".join(response_parts)

    def reset(self) -> None:
        """Reset thought counter (typically at task start)."""
        self.thought_count = 0


def think(thought: str, question: Optional[str] = None, type: str = "reasoning") -> str:
    """
    Standalone function for think tool execution.

    This is the function registered in the tool registry.
    """
    # Use a module-level tool instance
    global _think_tool_instance
    if "_think_tool_instance" not in globals():
        _think_tool_instance = ThinkTool()

    return _think_tool_instance.execute(thought, question, type)


# Module-level instance for registry
_think_tool_instance = ThinkTool()
