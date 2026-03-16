import json
from typing import List, Dict, Any
from pathlib import Path


class AdaptiveRouter:
    """
    Learns to route tasks to the optimal execution loop (simple vs enhanced)
    based on historical success and task characteristics.
    """

    def __init__(self, history_file: str = ".anvil/routing_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text())
            except Exception:
                return []
        return []

    def _save_history(self):
        self.history_file.write_text(json.dumps(self.history, indent=2))

    def collect_feedback(
        self, objective: str, loop_type: str, success: bool, metrics: Dict[str, Any]
    ):
        """Records the outcome of a routing decision."""
        entry = {
            "objective": objective,
            "loop_type": loop_type,
            "success": success,
            "metrics": metrics,
            "features": self._extract_features(objective),
        }
        self.history.append(entry)
        # Keep only last 1000 entries
        if len(self.history) > 1000:
            self.history.pop(0)
        self._save_history()

    def _extract_features(self, objective: str) -> Dict[str, Any]:
        """Extracts numerical features from the objective text."""
        obj_lower = objective.lower()
        return {
            "length": len(objective),
            "has_fix": "fix" in obj_lower or "bug" in obj_lower,
            "has_add": "add" in obj_lower
            or "create" in obj_lower
            or "implement" in obj_lower,
            "has_refactor": "refactor" in obj_lower,
            "has_question": "?" in objective
            or objective.startswith(("what", "how", "why")),
        }

    def predict_loop(self, objective: str) -> str:
        """
        Predicts the best loop type.
        Uses historical success rates if enough data exists, else defaults to heuristic.
        """
        if len(self.history) < 20:
            return "auto"  # Fallback to existing LoopSelector logic

        features = self._extract_features(objective)

        # Simple K-Nearest Neighbors like heuristic based on history
        # (This avoids heavy ML dependencies while being adaptive)
        enhanced_success = 0
        simple_success = 0
        enhanced_count = 0
        simple_count = 0

        for h in self.history[-100:]:
            # Simple similarity check: shared keywords
            if (
                h["features"]["has_fix"] == features["has_fix"]
                and h["features"]["has_add"] == features["has_add"]
            ):
                if h["loop_type"] == "enhanced":
                    enhanced_count += 1
                    if h["success"]:
                        enhanced_success += 1
                else:
                    simple_count += 1
                    if h["success"]:
                        simple_success += 1

        if enhanced_count > 0 and simple_count > 0:
            enhanced_rate = enhanced_success / enhanced_count
            simple_rate = simple_success / simple_count

            if enhanced_rate > simple_rate:
                return "enhanced"
            else:
                return "simple"

        return "auto"
