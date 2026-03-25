from enum import Enum


class MemoryTier(Enum):
    """
    Tiered Memory Levels for Anvil.
    """

    WORKING = "working"  # Immediate context, active changes (Volatile)
    EPISODIC = "episodic"  # Recent sessions/conversations (Compressed)
    SEMANTIC = "semantic"  # Long-term facts, codebase knowledge (Indexed)
    PREFERENCE = "preference"  # User preferences, style guides (Permanent)
