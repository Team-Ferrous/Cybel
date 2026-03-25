"""
Notify User Tool - Explicit communication with the user during tasks.

This is the only way to communicate with the user during active task execution.
Supports artifact review, blocking for user input, and auto-proceed flags.
"""

from typing import Optional, List, Dict, Any


class NotifyUserTool:
    """
    Explicit tool for communicating with the user during active tasks.

    This tool:
    - Exits task mode when called (returns control to user)
    - Supports artifact review with file paths
    - Can block until user responds (blocked_on_user)
    - Can auto-proceed for non-critical notifications
    """

    schema = {
        "name": "notify_user",
        "description": "Send a message to the user. Use for updates, requesting review, or asking questions. This is the only way to communicate during active tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the user",
                },
                "paths_to_review": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths that the user should review",
                },
                "blocked_on_user": {
                    "type": "boolean",
                    "default": False,
                    "description": "If true, agent will wait for user response before continuing",
                },
                "should_auto_proceed": {
                    "type": "boolean",
                    "default": True,
                    "description": "If true and not blocked, agent can continue automatically",
                },
                "notification_type": {
                    "type": "string",
                    "enum": ["info", "warning", "error", "success", "question"],
                    "default": "info",
                    "description": "Type of notification for UI styling",
                },
            },
            "required": ["message"],
        },
    }

    def __init__(self, task_state_manager=None):
        """
        Initialize the notify_user tool.

        Args:
            task_state_manager: Optional TaskStateManager for updating blocked status
        """
        self.task_state_manager = task_state_manager
        self.pending_notifications: List[Dict[str, Any]] = []

    def execute(
        self,
        message: str,
        paths_to_review: Optional[List[str]] = None,
        blocked_on_user: bool = False,
        should_auto_proceed: bool = True,
        notification_type: str = "info",
    ) -> str:
        """
        Execute the notify_user tool.

        Args:
            message: The message to display
            paths_to_review: Optional list of file paths requiring review
            blocked_on_user: If True, wait for user response
            should_auto_proceed: If True and not blocked, continue automatically
            notification_type: Type of notification (info, warning, error, success, question)

        Returns:
            Formatted notification result
        """
        # Record notification
        notification = {
            "message": message,
            "paths_to_review": paths_to_review or [],
            "blocked_on_user": blocked_on_user,
            "should_auto_proceed": should_auto_proceed,
            "notification_type": notification_type,
        }
        self.pending_notifications.append(notification)

        # Update task state if manager is available
        if self.task_state_manager and self.task_state_manager.current_task:
            self.task_state_manager.current_task.blocked_on_user = blocked_on_user

        # Build response
        type_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "success": "✅",
            "question": "❓",
        }.get(notification_type, "📢")

        response_parts = [f"{type_emoji} NOTIFICATION TO USER:"]
        response_parts.append(f"\n{message}\n")

        if paths_to_review:
            response_parts.append("Files for review:")
            for path in paths_to_review:
                response_parts.append(f"  • {path}")

        if blocked_on_user:
            response_parts.append("\n[BLOCKED - Waiting for user response]")
        elif should_auto_proceed:
            response_parts.append("\n[Agent will continue automatically]")

        return "\n".join(response_parts)

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get pending notifications."""
        return self.pending_notifications

    def clear_pending(self) -> None:
        """Clear pending notifications."""
        self.pending_notifications = []

    def is_blocked(self) -> bool:
        """Check if there's a blocking notification."""
        return any(n.get("blocked_on_user", False) for n in self.pending_notifications)


def notify_user(
    message: str,
    paths_to_review: Optional[List[str]] = None,
    blocked_on_user: bool = False,
    should_auto_proceed: bool = True,
    notification_type: str = "info",
) -> str:
    """
    Standalone function for notify_user tool execution.

    This is the function registered in the tool registry.
    """
    global _notify_tool_instance
    if "_notify_tool_instance" not in globals():
        _notify_tool_instance = NotifyUserTool()

    return _notify_tool_instance.execute(
        message,
        paths_to_review,
        blocked_on_user,
        should_auto_proceed,
        notification_type,
    )


# Module-level instance
_notify_tool_instance = NotifyUserTool()
