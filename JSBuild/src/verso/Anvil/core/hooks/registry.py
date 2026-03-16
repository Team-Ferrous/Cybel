"""
Backward-compatible shim for HookRegistry.
"""

import warnings
from infrastructure.hooks.registry import HookRegistry

warnings.warn(
    "core.hooks.registry is deprecated. Use infrastructure.hooks.registry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HookRegistry"]
