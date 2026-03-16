"""COCONUT backend factory for CPU-only execution."""

from .backend_interface import CoconutBackend
from .native_backend import NativeBackend


def get_backend(backend_type: str = "auto", **config) -> CoconutBackend:
    """Return the native CPU backend for auto/native requests."""
    if backend_type in ("auto", "native"):
        return NativeBackend(**config)
    raise ValueError(
        f"Unknown backend '{backend_type}'. Only 'native' is supported."
    )
