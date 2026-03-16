"""Backward-compatible shim for AESPreVerifyHook."""

import warnings

from infrastructure.hooks.aes_pre_verify import AESPreVerifyHook

warnings.warn(
    "core.hooks.aes_pre_verify is deprecated. "
    "Use infrastructure.hooks.aes_pre_verify instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AESPreVerifyHook"]
