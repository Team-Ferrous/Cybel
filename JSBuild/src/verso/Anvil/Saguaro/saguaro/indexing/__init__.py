"""Indexing package exports."""

from __future__ import annotations

from .auto_scaler import (
    allocate_dark_space,
    calculate_ideal_dim,
    get_repo_stats_and_config,
)

__all__ = [
    "calculate_ideal_dim",
    "allocate_dark_space",
    "get_repo_stats_and_config",
    "IndexCoordinator",
]


def __getattr__(name: str):
    if name == "IndexCoordinator":
        # Lazy import avoids a cycle with saguaro.api during package init.
        from .coordinator import IndexCoordinator

        return IndexCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
