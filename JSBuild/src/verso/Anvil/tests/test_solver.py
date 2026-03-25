from __future__ import annotations

from saguaro.synthesis.solver import DeterministicSolver
from saguaro.synthesis.spec import SpecLowerer


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def test_solver_verifies_bounded_numeric_helper() -> None:
    spec = SpecLowerer().lower_objective("Implement bounded clamp helper in generated/clamp.py")
    samples = [
        {"value": -1.0, "lower": 0.0, "upper": 1.0},
        {"value": 0.5, "lower": 0.0, "upper": 1.0},
        {"value": 2.0, "lower": 0.0, "upper": 1.0},
    ]

    result = DeterministicSolver().verify_spec_constraints(spec, _clamp, samples=samples)

    assert result.passed is True
    assert "range_safety_constraint_satisfied" in result.proofs

