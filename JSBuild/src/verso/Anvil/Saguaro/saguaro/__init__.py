# SAGUARO Python Package
"""Package initialization for saguaro."""

from __future__ import annotations

from typing import Any

__version__ = "5.0.0"
__all__ = ["SaguaroAPI"]


def __getattr__(name: str) -> Any:
    if name == "SaguaroAPI":
        from .api import SaguaroAPI

        return SaguaroAPI
    raise AttributeError(name)


# Note: Compatibility for _native imports is handled via a symbolic link
# in the filesystem (saguaro.native -> native) to allow submodules
# to be correctly recognized as packages.
