"""Utilities for governor."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextBudgetExceeded(Exception):
    """Provide ContextBudgetExceeded support."""
    pass


class ContextGovernor:
    """Manages the 'Context Budget' for agent retrieval operations.
    Prevents agents from accidentally consuming massive amounts of tokens.
    """

    def __init__(self, soft_limit_tokens: int = 8000, hard_limit_tokens: int = 32000) -> None:
        """Initialize the instance."""
        self.soft_limit = soft_limit_tokens
        self.hard_limit = hard_limit_tokens

        # Approximate tokens per character (conservative 4 chars/token)
        self.CHARS_PER_TOKEN = 4
        # Metadata overhead for each context item
        self.BASE_ITEM_OVERHEAD = 50

    def estimate_tokens(self, text: str) -> int:
        """Handle estimate tokens."""
        return len(text) // self.CHARS_PER_TOKEN

    def check_budget(
        self, proposed_context_items: list[dict[str, Any]]
    ) -> tuple[bool, int, str]:
        """Checks if a proposed context bundle fits within the budget.
        Returns: (is_safe, estimated_tokens, message).
        """
        total_tokens = 0
        for item in proposed_context_items:
            # Assume item has 'content' or we estimate based on 'file' size if loaded
            # For now, let's assume 'content' key exists or we just count metadata overhead
            content = item.get("content", "")
            total_tokens += self.estimate_tokens(content)

            # Add some overhead for metadata
            total_tokens += self.BASE_ITEM_OVERHEAD

        if total_tokens > self.hard_limit:
            return (
                False,
                total_tokens,
                f"EXCEEDS HARD LIMIT ({total_tokens} > {self.hard_limit})",
            )

        if total_tokens > self.soft_limit:
            return (
                True,
                total_tokens,
                f"WARNING: Exceeds soft limit ({total_tokens} > {self.soft_limit})",
            )

        return True, total_tokens, "OK"

    def optimize_bundle(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trims a list of context items to fit within the Soft Limit.
        Prioritizes by 'score' if present, otherwise preserves order.
        """
        # Sort by score descending (higher priority items first)
        sorted_items = sorted(items, key=lambda x: x.get("score", 0), reverse=True)

        keep_items = []
        current_tokens = 0

        for item in sorted_items:
            # Calculate token cost based on character length and constant overhead
            content_len = len(item.get("content", ""))
            cost = (content_len // self.CHARS_PER_TOKEN) + self.BASE_ITEM_OVERHEAD

            if current_tokens + cost <= self.soft_limit:
                keep_items.append(item)
                current_tokens += cost
            else:
                logger.debug(
                    f"Governor: Dropping item '{item.get('name', 'unknown')}' "
                    f"({cost} tokens) to satisfy budget ({self.soft_limit})."
                )

        return keep_items

    def escalate(self, current_limit: int) -> int:
        """Explicit escalation API.
        Returns new limit if allowed, or raises if at hard ceiling.
        """
        if current_limit >= self.hard_limit:
            raise ContextBudgetExceeded("Cannot escalate past hard limit.")

        # Double, but cap at hard limit
        return min(current_limit * 2, self.hard_limit)
