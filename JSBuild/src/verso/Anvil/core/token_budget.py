"""
Token Budget Manager

Centralized token counting and budget allocation for context management.
Provides accurate token estimation and enforces budgets across the pipeline.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence, Tuple


class TokenBudgetManager:
    """
    Manages token budgets for evidence formatting and context loading.

    Uses a 4-char/token heuristic (standard approximation) with
    adjustments for code/markdown formatting overhead.
    """

    def __init__(self, total_budget: int):
        """
        Initialize budget manager.

        Args:
            total_budget: Total tokens available for this context
        """
        self.total_budget = total_budget
        self.allocated = 0
        self.items: List[Dict[str, Any]] = []

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses 4-char/token heuristic with adjustments:
        - Code blocks: +10% overhead (syntax highlighting tokens)
        - Markdown: +5% overhead (formatting tokens)
        - JSON: +8% overhead (structure tokens)

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        base_tokens = len(text) // 4

        # Detect content type and apply overhead
        if "```" in text:
            # Code block overhead
            base_tokens = int(base_tokens * 1.10)
        elif text.strip().startswith("{") or text.strip().startswith("["):
            # JSON overhead
            base_tokens = int(base_tokens * 1.08)
        elif any(marker in text for marker in ["#", "*", "-", "|"]):
            # Markdown overhead
            base_tokens = int(base_tokens * 1.05)

        return base_tokens

    def fits_in_budget(self, tokens: int) -> bool:
        """
        Check if tokens fit in remaining budget.

        Args:
            tokens: Number of tokens to check

        Returns:
            True if tokens fit, False otherwise
        """
        return (self.allocated + tokens) <= self.total_budget

    def allocate(self, tokens: int, item_name: Optional[str] = None) -> bool:
        """
        Allocate tokens from budget.

        Args:
            tokens: Number of tokens to allocate
            item_name: Optional name for tracking

        Returns:
            True if allocation succeeded, False if budget exceeded
        """
        if not self.fits_in_budget(tokens):
            return False

        self.allocated += tokens
        self.items.append(
            {"name": item_name or f"item_{len(self.items)}", "tokens": tokens}
        )
        return True

    def remaining(self) -> int:
        """Get remaining budget."""
        return self.total_budget - self.allocated

    def utilization(self) -> float:
        """Get budget utilization as fraction (0.0 to 1.0)."""
        if self.total_budget == 0:
            return 0.0
        return self.allocated / self.total_budget

    def reset(self):
        """Reset budget allocation."""
        self.allocated = 0
        self.items = []

    def get_stats(self) -> Dict[str, Any]:
        """
        Get budget statistics.

        Returns:
            Dict with total, allocated, remaining, utilization, items
        """
        return {
            "total_budget": self.total_budget,
            "allocated": self.allocated,
            "remaining": self.remaining(),
            "utilization": f"{self.utilization():.1%}",
            "items_count": len(self.items),
            "items": self.items,
        }


@dataclass(frozen=True)
class PromptContextBudgetPolicy:
    """Hard context allocation policy for prompt assembly."""

    master_prompt_tokens: int = 2200
    subagent_prompt_tokens: int = 8000
    verification_prompt_tokens: int = 1600
    response_reserve_tokens: int = 300


def assemble_prompt_with_budget(
    sections: Sequence[Tuple[str, str]],
    token_budget: int,
    reserve_tokens: int = 0,
) -> Dict[str, Any]:
    """
    Assemble prompt sections within a token budget.

    Args:
        sections: Sequence of (section_name, section_text)
        token_budget: Maximum tokens available for all sections
        reserve_tokens: Tokens reserved for additional runtime prompt layers

    Returns:
        Dict containing assembled prompt text and inclusion metadata
    """
    available_budget = max(0, int(token_budget) - int(reserve_tokens))
    manager = TokenBudgetManager(total_budget=available_budget)
    included_sections: List[str] = []
    dropped_sections: List[str] = []
    assembled_chunks: List[str] = []

    for section_name, section_text in sections:
        normalized_text = (section_text or "").strip()
        if not normalized_text:
            continue
        tokens = manager.count_tokens(normalized_text)
        if manager.allocate(tokens, item_name=section_name):
            included_sections.append(section_name)
            assembled_chunks.append(normalized_text)
        else:
            dropped_sections.append(section_name)

    return {
        "text": "\n\n".join(assembled_chunks).strip(),
        "included_sections": included_sections,
        "dropped_sections": dropped_sections,
        "token_budget": available_budget,
        "token_usage": manager.allocated,
    }


def estimate_context_budget(
    max_context: int = 400000,
    system_prompt_tokens: int = 2000,
    user_input_tokens: int = 500,
    history_tokens: int = 5000,
    response_reserve: int = 15000,
) -> int:
    """
    Calculate available token budget for evidence formatting.

    Accounts for:
    - System prompt overhead
    - User input
    - Conversation history
    - Reserved tokens for model response

    Args:
        max_context: Maximum context window
        system_prompt_tokens: Estimated system prompt tokens
        user_input_tokens: Estimated user input tokens
        history_tokens: Estimated conversation history tokens
        response_reserve: Tokens reserved for model response

    Returns:
        Available tokens for evidence
    """
    used = system_prompt_tokens + user_input_tokens + history_tokens + response_reserve
    available = max_context - used

    # Safety margin: use 90% of available to prevent edge cases
    safe_budget = int(available * 0.9)

    return max(0, safe_budget)


def rank_by_relevance(
    items: List[tuple], scores: Optional[List[float]] = None, reverse: bool = True
) -> List[tuple]:
    """
    Rank items by relevance scores.

    Args:
        items: List of (key, value) tuples
        scores: Optional relevance scores (same length as items)
        reverse: If True, sort descending (highest score first)

    Returns:
        Sorted list of items
    """
    if scores is None:
        # No scores provided, return as-is
        return items

    if len(items) != len(scores):
        raise ValueError(
            f"Items ({len(items)}) and scores ({len(scores)}) length mismatch"
        )

    # Combine items with scores
    scored_items = list(zip(items, scores))

    # Sort by score
    scored_items.sort(key=lambda x: x[1], reverse=reverse)

    # Return just the items
    return [item for item, score in scored_items]


def smart_truncate(text: str, max_tokens: int, preserve_structure: bool = True) -> str:
    """
    Intelligently truncate text to fit token budget.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        preserve_structure: If True, try to preserve code blocks/markdown

    Returns:
        Truncated text
    """
    current_tokens = len(text) // 4

    if current_tokens <= max_tokens:
        return text

    # Calculate target character length
    target_chars = max_tokens * 4

    if not preserve_structure:
        # Simple truncation
        return text[:target_chars] + "\n... (truncated)"

    # Try to truncate at natural boundaries
    truncated = text[:target_chars]

    # Find last complete line
    last_newline = truncated.rfind("\n")
    if last_newline > target_chars * 0.8:  # At least 80% of target
        truncated = truncated[:last_newline]

    # If inside code block, try to close it
    code_blocks = truncated.count("```")
    if code_blocks % 2 == 1:  # Odd number = unclosed block
        truncated += "\n```"

    truncated += "\n\n... (truncated to fit token budget)"

    return truncated
