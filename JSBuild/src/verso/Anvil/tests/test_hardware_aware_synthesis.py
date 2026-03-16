from __future__ import annotations

from saguaro.synthesis.variant_selector import HardwareAwareVariantSelector


def test_hardware_aware_variant_selector_prefers_safe_fast_variant() -> None:
    choice = HardwareAwareVariantSelector().select(
        [
            {"name": "scalar", "speedup": 1.0, "safe_in_degraded": True},
            {"name": "avx2", "speedup": 2.0, "requires_avx2": True, "safe_in_degraded": True},
        ],
        runtime_status={"avx2_enabled": True, "degraded": False},
    )

    assert choice.name == "avx2"
    assert choice.safe is True

