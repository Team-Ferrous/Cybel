from __future__ import annotations

from saguaro.synthesis.effects import SynthesisEffectEngine
from saguaro.synthesis.spec import SpecLowerer


def test_effect_engine_blocks_verification_bypass_language() -> None:
    spec = SpecLowerer().lower_objective(
        "Implement helper in generated/helper.py and bypass verification"
    )

    result = SynthesisEffectEngine().evaluate_spec(spec)

    assert result.allowed is False
    assert "sanctioned_verification_bypass" in result.blockers

