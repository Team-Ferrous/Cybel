from enum import Enum
from typing import Dict, Any


class TuningPhase(Enum):
    COARSE = 0  # Global scaling, high exploration
    REFINE = 1  # Group-level tuning, medium temperature
    FINE = 2  # Precise tuning, greedy/low temperature
    EXPLORE = 3  # Random perturbations to escape plateaus


class AdaptivePhaseController:
    """
    Port of HighNoon's C++ AdaptivePhaseController to Python.
    Manages generation parameters (Temperature, Draft Count) based on
    confidence/convergence metrics.
    """

    def __init__(self, config: Any):
        self.config = config
        self.phase = TuningPhase.COARSE
        self.batches_in_phase = 0

        # Thresholds (can be moved to config)
        self.coarse_min_batches = 10
        self.refine_min_batches = 20
        self.fine_min_batches = 40
        self.explore_duration = 5

        # Stability thresholds
        self.confidence_threshold_coarse = 0.7
        self.confidence_threshold_refine = 0.9
        self.plateau_threshold = 0.98  # High confidence plateau
        self.instability_threshold = 0.4  # Sudden drop in confidence

    def update_phase(self, confidence: float):
        """
        Update current phase based on generation confidence score.
        confidence: Average probability of selected tokens.
        """
        self.batches_in_phase += 1

        if self.phase == TuningPhase.COARSE:
            if (
                confidence > self.confidence_threshold_coarse
                and self.batches_in_phase > self.coarse_min_batches
            ):
                self.phase = TuningPhase.REFINE
                self.batches_in_phase = 0

        elif self.phase == TuningPhase.REFINE:
            if (
                confidence > self.confidence_threshold_refine
                and self.batches_in_phase > self.refine_min_batches
            ):
                self.phase = TuningPhase.FINE
                self.batches_in_phase = 0

        elif self.phase == TuningPhase.FINE:
            if (
                confidence > self.plateau_threshold
                and self.batches_in_phase > self.fine_min_batches
            ):
                self.phase = TuningPhase.EXPLORE
                self.batches_in_phase = 0
            elif confidence < self.instability_threshold:
                self.phase = TuningPhase.COARSE
                self.batches_in_phase = 0

        elif self.phase == TuningPhase.EXPLORE:
            if self.batches_in_phase > self.explore_duration:
                self.phase = TuningPhase.FINE
                self.batches_in_phase = 0
            elif confidence < self.instability_threshold:
                self.phase = TuningPhase.COARSE
                self.batches_in_phase = 0

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get dynamic parameters based on current phase.
        """
        if self.phase == TuningPhase.COARSE:
            return {
                "temperature": 1.2,
                "draft_count": (
                    self.config.speculative_drafts * 2
                    if hasattr(self.config, "speculative_drafts")
                    else 16
                ),
                "phase": "COARSE",
            }
        elif self.phase == TuningPhase.REFINE:
            return {
                "temperature": 0.8,
                "draft_count": (
                    self.config.speculative_drafts
                    if hasattr(self.config, "speculative_drafts")
                    else 8
                ),
                "phase": "REFINE",
            }
        elif self.phase == TuningPhase.FINE:
            return {
                "temperature": 0.1,  # Near greedy
                "draft_count": max(
                    1,
                    (
                        (self.config.speculative_drafts // 2)
                        if hasattr(self.config, "speculative_drafts")
                        else 4
                    ),
                ),
                "phase": "FINE",
            }
        elif self.phase == TuningPhase.EXPLORE:
            return {
                "temperature": 1.5,  # Boost for exploration
                "draft_count": (
                    self.config.speculative_drafts
                    if hasattr(self.config, "speculative_drafts")
                    else 8
                ),
                "phase": "EXPLORE",
            }
        return {"temperature": 1.0, "draft_count": 8, "phase": "UNKNOWN"}
