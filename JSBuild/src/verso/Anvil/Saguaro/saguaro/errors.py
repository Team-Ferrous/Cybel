"""Typed errors for Saguaro state, durability, and locking failures."""

from __future__ import annotations


class SaguaroError(Exception):
    """Base class for Saguaro-specific exceptions."""


class SaguaroBusyError(SaguaroError):
    """Raised when a required repo-scoped lock cannot be acquired in time."""


class SaguaroStateError(SaguaroError):
    """Base class for on-disk index state failures."""


class SaguaroStateCorruptionError(SaguaroStateError):
    """Raised when persisted state is unreadable or structurally invalid."""


class SaguaroStateMismatchError(SaguaroStateError):
    """Raised when committed artifacts disagree with the committed manifest."""
