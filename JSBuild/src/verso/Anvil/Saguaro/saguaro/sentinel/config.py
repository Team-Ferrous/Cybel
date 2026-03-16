# saguaro/sentinel/config.py

"""Utilities for config."""

DRIFT_THRESHOLDS = {
    "minor": 0.1,  # Acceptable drift: no action needed
    "major": 0.25,  # Warning level: notify agent/user
    "critical": 0.4,  # Block level: reject commit/snapshot
}


def get_drift_level(drift_score: float) -> str:
    """Classifies a semantic drift score into a level."""
    if drift_score >= DRIFT_THRESHOLDS["critical"]:
        return "critical"
    if drift_score >= DRIFT_THRESHOLDS["major"]:
        return "major"
    if drift_score >= DRIFT_THRESHOLDS["minor"]:
        return "minor"
    return "nominal"
