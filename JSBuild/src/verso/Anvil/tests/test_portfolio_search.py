from __future__ import annotations

from saguaro.synthesis.portfolio_search import PortfolioSearch, SearchBudget


def test_portfolio_search_finds_bounded_candidate() -> None:
    def _validator(expression: str, cases: list[dict[str, float]]) -> bool:
        safe_globals = {"min": min, "max": max}
        for case in cases:
            observed = eval(expression, safe_globals, dict(case))
            if not (case["lower"] <= observed <= case["upper"]):
                return False
        return True

    candidate = PortfolioSearch().search(
        cases=[
            {"value": -1.0, "lower": 0.0, "upper": 1.0},
            {"value": 2.0, "lower": 0.0, "upper": 1.0},
        ],
        validator=_validator,
        budget=SearchBudget(max_expansions=4, max_depth=3),
    )

    assert candidate.score == 1.0
    assert "min" in candidate.expression or "max" in candidate.expression

