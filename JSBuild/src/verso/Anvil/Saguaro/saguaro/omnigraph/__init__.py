"""Typed omni-graph support."""

from .model import OmniNode, OmniRelation
from .store import OmniGraphStore

__all__ = ["OmniGraphStore", "OmniNode", "OmniRelation"]
