import os
import json
from datetime import datetime


class CorrectionManager:
    """
    Tracks user corrections and feedback to improve future performance.
    """

    def __init__(self, storage_path="~/.anvil/corrections.json"):
        self.path = os.path.expanduser(storage_path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.corrections = self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return []

    def add_correction(self, prompt: str, error: str, correction: str):
        self.corrections.append(
            {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "error": error,
                "correction": correction,
            }
        )
        self._save()

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.corrections, f, indent=2)

    def get_context_summary(self, max_items=5) -> str:
        """
        Returns a summary of recent corrections to be included in the system prompt.
        """
        if not self.corrections:
            return ""

        recent = self.corrections[-max_items:]
        summary = "--- USER CORRECTIONS & STYLE PREFERENCES ---\n"
        for c in recent:
            summary += (
                f"- When you did: {c['error']}\n  The user said: {c['correction']}\n"
            )
        return summary
