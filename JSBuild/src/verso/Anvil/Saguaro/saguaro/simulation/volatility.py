"""Utilities for volatility."""

class VolatilityMapper:
    """Visualizes which files churn most and cause downstream breakage."""

    def generate_map(self, repo_path: str) -> dict[str, float]:
        """Returns a map of file_path -> volatility_score (0-1)."""
        # Ideally this reads git history.
        # For now, we stub it.

        return {"saguaro/core.py": 0.1, "saguaro/utils/common.py": 0.8}
