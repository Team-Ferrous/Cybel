from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AdaptiveExplorationMetrics:
    """Runtime metrics captured during adaptive COCONUT exploration."""

    actual_steps: int
    max_steps: int
    min_steps: int
    final_entropy: Optional[float]
    final_confidence: Optional[float]
    path_amplitudes: Optional[List[float]] = None
    step_entropies: List[float] = field(default_factory=list)
    step_confidences: List[float] = field(default_factory=list)
    termination_reason: str = "max_steps_reached"

    @property
    def depth_utilization(self) -> float:
        if self.max_steps <= 0:
            return 0.0
        return float(self.actual_steps) / float(self.max_steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actual_steps": int(self.actual_steps),
            "max_steps": int(self.max_steps),
            "min_steps": int(self.min_steps),
            "final_entropy": (
                float(self.final_entropy) if self.final_entropy is not None else None
            ),
            "final_confidence": (
                float(self.final_confidence)
                if self.final_confidence is not None
                else None
            ),
            "path_amplitudes": self.path_amplitudes or [],
            "step_entropies": [float(v) for v in self.step_entropies],
            "step_confidences": [float(v) for v in self.step_confidences],
            "termination_reason": str(self.termination_reason),
            "depth_utilization": float(self.depth_utilization),
        }


def amplitude_entropy(amplitudes: Optional[np.ndarray]) -> Optional[float]:
    """Compute Shannon entropy for amplitude distribution."""
    if amplitudes is None:
        return None
    arr = np.asarray(amplitudes, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    probs = np.abs(arr)
    denom = float(np.sum(probs))
    if denom <= 1e-12:
        return None
    probs = probs / denom
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def amplitude_confidence(amplitudes: Optional[np.ndarray]) -> Optional[float]:
    """Compute confidence as max normalized amplitude."""
    if amplitudes is None:
        return None
    arr = np.asarray(amplitudes, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    probs = np.abs(arr)
    denom = float(np.sum(probs))
    if denom <= 1e-12:
        return None
    probs = probs / denom
    return float(np.max(probs))
