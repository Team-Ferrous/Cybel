from __future__ import annotations

from saguaro.synthesis.eqsat_runner import BoundedEqsatRunner


def test_eqsat_runner_simplifies_small_expression_trees() -> None:
    result = BoundedEqsatRunner().optimize_expression("1 * (value + 0)")

    assert result.optimized == "value"
    assert result.telemetry["rewrite_fire_count"] >= 1

