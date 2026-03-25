"""
Backward-compatible shim for MemoryTier.
"""

import warnings
from domains.memory_management.memory_types import MemoryTier

warnings.warn(
    "core.memory.memory_types is deprecated. Use domains.memory_management.memory_types instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MemoryTier"]
