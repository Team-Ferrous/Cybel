# saguaro/config/grounding.py

"""Utilities for grounding."""

GRANITE_4_DETERMINISTIC = {
    "granite-4.0-h-small": {
        "temperature": 1e-14,
        "top_p": 1e-14,
        "top_k": 1,
        "seed": 720720,
        "repeat_penalty": 1.0,
        "num_ctx": 200000,
        "keep_alive": -1,
    },
    "granite-4.0-h-tiny": {
        "temperature": 1e-14,
        "top_p": 1e-14,
        "top_k": 1,
        "seed": 720720,
        "repeat_penalty": 1.0,
        "num_ctx": 200000,  # Match master context
        "keep_alive": 0,  # Release after use
    },
}


def get_deterministic_params(model_name: str) -> dict:
    """Returns deterministic parameters for a given model.
    Matches the 'infinitesimal parameter tuning' strategy.
    """
    # Try exact match, then base name
    if model_name in GRANITE_4_DETERMINISTIC:
        return GRANITE_4_DETERMINISTIC[model_name]

    # Check if it's a variant of granite-4
    if "granite-4" in model_name:
        if "tiny" in model_name:
            return GRANITE_4_DETERMINISTIC["granite-4.0-h-tiny"]
        return GRANITE_4_DETERMINISTIC["granite-4.0-h-small"]

    return {}
