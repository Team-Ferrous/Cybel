"""AES domain templates for machine-learning code generation."""

from domains.ml.aes_ml_templates import (
    EVIDENCE_METADATA_TEMPLATE,
    GRADIENT_HEALTH_GATE,
    STABLE_SOFTMAX,
    TRAINING_LOOP_SKELETON,
)

__all__ = [
    "GRADIENT_HEALTH_GATE",
    "STABLE_SOFTMAX",
    "TRAINING_LOOP_SKELETON",
    "EVIDENCE_METADATA_TEMPLATE",
]
