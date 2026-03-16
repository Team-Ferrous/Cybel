"""
Backward-compatible shim for SemanticEngine.

This module has been moved to domains.code_intelligence.semantic_engine
as part of the DDD restructuring (Phase 1 of Technical Roadmap).

All imports from this module are deprecated but will continue to work.
"""

import warnings

# Import from new location
from domains.code_intelligence.semantic_engine import SemanticEngine

# Emit deprecation warning
warnings.warn(
    "core.analysis.semantic is deprecated. "
    "Use domains.code_intelligence.semantic_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Export for backward compatibility
__all__ = ["SemanticEngine"]
