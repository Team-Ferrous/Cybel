"""Static roofline-style estimation."""

from __future__ import annotations

from saguaro.cpu.topology import ArchitecturePack
from saguaro.math.ir import MathIRRecord

_BOUND_NAMES = {0: "memory_bound", 1: "balanced", 2: "compute_bound"}


def _hierarchical_bounds(intensity: float) -> dict[str, str]:
    if intensity < 0.2:
        return {"l1": "memory_bound", "l2": "memory_bound", "l3": "memory_bound", "dram": "memory_bound"}
    if intensity < 0.8:
        return {"l1": "balanced", "l2": "balanced", "l3": "memory_bound", "dram": "memory_bound"}
    if intensity < 1.5:
        return {"l1": "compute_bound", "l2": "balanced", "l3": "balanced", "dram": "memory_bound"}
    return {"l1": "compute_bound", "l2": "compute_bound", "l3": "balanced", "dram": "balanced"}


def estimate_roofline(
    record: MathIRRecord,
    pack: ArchitecturePack,
    *,
    native_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Estimate a coarse operational-intensity position."""

    native_roofline = dict((native_report or {}).get("roofline") or {})
    runtime_witness = dict((native_report or {}).get("roofline_witness") or {})
    if native_roofline:
        intensity = round(float(native_roofline.get("operational_intensity", 0.0) or 0.0), 3)
        bound = _BOUND_NAMES.get(int(native_roofline.get("bound_class", 0) or 0), "memory_bound")
        observed_bound = str(runtime_witness.get("observed_bound") or "").strip()
        return {
            "estimated_ops": int(native_roofline.get("estimated_ops", 0) or 0),
            "estimated_bytes": int(native_roofline.get("estimated_bytes", 0) or 0),
            "operational_intensity": intensity,
            "bound": bound,
            "hierarchical_bounds": _hierarchical_bounds(intensity),
            "observed_operational_intensity": float(
                runtime_witness.get("observed_operational_intensity", 0.0) or 0.0
            ),
            "observed_bound": observed_bound,
            "bound_agreement": (observed_bound == bound) if observed_bound else None,
            "analysis_engine": str((native_report or {}).get("engine") or "native"),
        }

    complexity = record.complexity
    ops = int(complexity.operator_count if complexity is not None else 0)
    ops += int(complexity.function_call_count if complexity is not None else 0) * 2
    bytes_touched = max(1, len(record.access_signatures)) * (pack.lane_width_bits // 8)
    intensity = round(float(ops) / float(bytes_touched), 3)
    if intensity < 0.25:
        bound = "memory_bound"
    elif intensity < 1.0:
        bound = "balanced"
    else:
        bound = "compute_bound"
    return {
        "estimated_ops": ops,
        "estimated_bytes": bytes_touched,
        "operational_intensity": intensity,
        "bound": bound,
        "hierarchical_bounds": _hierarchical_bounds(intensity),
        "observed_operational_intensity": 0.0,
        "observed_bound": "",
        "bound_agreement": None,
        "analysis_engine": "python_fallback",
    }
