"""
Backward-compatible shim for MemoryPersistence.
"""

import warnings
from domains.memory_management.persistence import MemoryPersistence

warnings.warn(
    "core.memory.persistence is deprecated. Use domains.memory_management.persistence instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MemoryPersistence"]
