"""
Backward-compatible shim for HolographicBundle.
"""

import warnings
from domains.memory_management.hd.bundle import HolographicBundle

warnings.warn(
    "core.hd.bundle is deprecated. Use domains.memory_management.hd.bundle instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HolographicBundle"]
