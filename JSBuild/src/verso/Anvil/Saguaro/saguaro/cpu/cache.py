"""Static cache-risk shaping for CPU hotspot reports."""

from __future__ import annotations

from saguaro.cpu.topology import ArchitecturePack
from saguaro.math.ir import MathIRRecord

_RISK_NAMES = {0: "low", 1: "medium", 2: "high"}
_REUSE_NAMES = {
    0: "register_resident",
    1: "contiguous_reuse",
    2: "streaming",
    3: "strided",
    4: "indirect",
}


def analyze_cache(
    record: MathIRRecord,
    pack: ArchitecturePack,
    *,
    native_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Estimate cache-line pressure from access signatures."""

    native_cache = dict((native_report or {}).get("cache") or {})
    if native_cache:
        return {
            "estimated_cache_lines_touched": int(
                native_cache.get("estimated_cache_lines_touched", 0) or 0
            ),
            "reuse_distance_class": _REUSE_NAMES.get(
                int(native_cache.get("reuse_distance_class", 0) or 0),
                "register_resident",
            ),
            "l1_risk": _RISK_NAMES.get(int(native_cache.get("l1_risk", 0) or 0), "low"),
            "l2_risk": _RISK_NAMES.get(int(native_cache.get("l2_risk", 0) or 0), "low"),
            "l3_risk": _RISK_NAMES.get(int(native_cache.get("l3_risk", 0) or 0), "low"),
            "memory_pressure_score": round(
                float(native_cache.get("memory_pressure_score", 0.0) or 0.0), 3
            ),
            "cache_line_bytes": int(pack.cache_line_bytes),
        }

    access_count = len(record.access_signatures)
    indirect = sum(1 for item in record.access_signatures if item.stride_class == "indirect")
    strided = sum(1 for item in record.access_signatures if item.stride_class == "strided")
    streaming = sum(1 for item in record.access_signatures if item.reuse_hint == "streaming")
    estimated_lines = max(1, access_count + (2 * strided) + (3 * indirect))
    reuse_class = (
        "indirect"
        if indirect
        else "strided"
        if strided
        else "streaming"
        if streaming
        else "contiguous_reuse"
    )
    return {
        "estimated_cache_lines_touched": estimated_lines,
        "reuse_distance_class": reuse_class,
        "l1_risk": "high" if estimated_lines >= 10 else "medium" if estimated_lines >= 4 else "low",
        "l2_risk": "high" if estimated_lines >= 14 else "medium" if estimated_lines >= 6 else "low",
        "l3_risk": "high" if estimated_lines >= 18 else "medium" if estimated_lines >= 8 else "low",
        "memory_pressure_score": round(float(estimated_lines + streaming + indirect), 3),
        "cache_line_bytes": int(pack.cache_line_bytes),
    }
