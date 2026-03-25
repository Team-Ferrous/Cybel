from __future__ import annotations

from saguaro.synthesis.translation_validator import TranslationValidator


def test_translation_validator_accepts_equivalent_lowerings() -> None:
    validator = TranslationValidator()
    witness = validator.validate_expression_pair(
        "max(lower, min(upper, value))",
        "min(upper, max(lower, value))",
        cases=[
            {"value": -1.0, "lower": 0.0, "upper": 1.0},
            {"value": 0.5, "lower": 0.0, "upper": 1.0},
            {"value": 2.0, "lower": 0.0, "upper": 1.0},
        ],
    )

    assert witness.equivalent is True
    assert witness.telemetry["ir_mismatch_count"] == 0

