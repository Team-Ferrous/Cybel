from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


SQALE_DIMENSIONS = (
    "testability",
    "reliability",
    "security",
    "efficiency",
    "changeability",
)


@dataclass
class DebtRecord:
    dimension: str
    minutes: float


class TechnicalDebtTracker:
    """Lightweight SQALE-style debt tracker for AES compliance reporting."""

    def __init__(self) -> None:
        self._debt_minutes: Dict[str, float] = {name: 0.0 for name in SQALE_DIMENSIONS}

    def add_debt(self, dimension: str, minutes: float) -> None:
        if dimension not in self._debt_minutes:
            raise ValueError(f"Unknown debt dimension: {dimension}")
        if minutes < 0:
            raise ValueError("Debt minutes must be non-negative")
        self._debt_minutes[dimension] += minutes

    def get_dimension_debt(self, dimension: str) -> float:
        if dimension not in self._debt_minutes:
            raise ValueError(f"Unknown debt dimension: {dimension}")
        return self._debt_minutes[dimension]

    def total_debt_minutes(self) -> float:
        return sum(self._debt_minutes.values())

    def debt_ratio(self, development_minutes: float) -> float:
        if development_minutes <= 0:
            return 0.0
        return self.total_debt_minutes() / development_minutes

    def snapshot(self) -> dict[str, float]:
        payload = dict(self._debt_minutes)
        payload["total"] = self.total_debt_minutes()
        return payload
