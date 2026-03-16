"""
Backward-compatible shim for Hook.
"""

import warnings
from infrastructure.hooks.base import Hook

warnings.warn(
    "core.hooks.base is deprecated. Use infrastructure.hooks.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Hook"]
