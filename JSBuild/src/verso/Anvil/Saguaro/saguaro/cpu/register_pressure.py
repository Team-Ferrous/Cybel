"""Static register-pressure surrogate model."""

from __future__ import annotations

from saguaro.cpu.topology import ArchitecturePack
from saguaro.math.ir import MathIRRecord

_BAND_NAMES = {0: "low", 1: "medium", 2: "high"}


def estimate_register_pressure(
    record: MathIRRecord,
    pack: ArchitecturePack,
    *,
    native_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Estimate whether a hotspot is likely to spill."""

    native_pressure = dict((native_report or {}).get("register_pressure") or {})
    if native_pressure:
        return {
            "score": int(native_pressure.get("score", 0) or 0),
            "band": _BAND_NAMES.get(int(native_pressure.get("band", 0) or 0), "low"),
            "spill_risk": bool(native_pressure.get("spill_risk", False)),
            "analysis_engine": str((native_report or {}).get("engine") or "native"),
        }

    complexity = record.complexity
    symbol_count = int(complexity.symbol_count if complexity is not None else 0)
    nesting = int(complexity.max_nesting_depth if complexity is not None else 0)
    calls = int(complexity.function_call_count if complexity is not None else 0)
    live_accesses = len(record.access_signatures)
    score = symbol_count + (2 * nesting) + (3 * calls) + live_accesses
    if score >= 18:
        band = "high"
    elif score >= 10:
        band = "medium"
    else:
        band = "low"
    spill_risk = score > max(8, pack.vector_bits // max(pack.lane_width_bits, 1) + 4)
    return {
        "score": score,
        "band": band,
        "spill_risk": spill_risk,
        "analysis_engine": "python_fallback",
    }
