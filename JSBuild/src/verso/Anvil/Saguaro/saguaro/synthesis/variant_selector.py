from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class VariantChoice:
    name: str
    rationale: str
    expected_speedup: float
    safe: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HardwareAwareVariantSelector:
    def select(
        self,
        variants: list[dict[str, Any]],
        *,
        runtime_status: dict[str, Any],
    ) -> VariantChoice:
        avx2 = bool(runtime_status.get("avx2_enabled"))
        degraded = bool(runtime_status.get("degraded") or runtime_status.get("degraded_capabilities"))
        ordered = sorted(
            variants,
            key=lambda item: (
                not bool(item.get("safe_in_degraded", True)),
                -float(item.get("speedup", 0.0)),
                str(item.get("name") or ""),
            ),
        )
        for variant in ordered:
            if bool(variant.get("requires_avx2")) and not avx2:
                continue
            if degraded and not bool(variant.get("safe_in_degraded", True)):
                continue
            return VariantChoice(
                name=str(variant.get("name") or "fallback"),
                rationale="selected_from_runtime_capabilities",
                expected_speedup=float(variant.get("speedup", 0.0)),
                safe=True,
            )
        return VariantChoice(
            name="fallback",
            rationale="no_hardware_specific_variant_available",
            expected_speedup=0.0,
            safe=True,
        )

