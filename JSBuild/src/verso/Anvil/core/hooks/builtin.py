"""
Backward-compatible shim for builtin hooks.
"""

import warnings
from infrastructure.hooks.builtin import (
    AALClassifyHook,
    ChronicleHook,
    ToolAuditHook,
    PrivacySafetyHook,
    TimingHook,
    SaguaroSyncHook,
)

warnings.warn(
    "core.hooks.builtin is deprecated. Use infrastructure.hooks.builtin instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ToolAuditHook",
    "PrivacySafetyHook",
    "TimingHook",
    "SaguaroSyncHook",
    "AALClassifyHook",
    "ChronicleHook",
]
