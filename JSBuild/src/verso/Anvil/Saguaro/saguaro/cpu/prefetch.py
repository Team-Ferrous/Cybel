"""Static prefetch opportunity heuristics."""

from __future__ import annotations

from saguaro.cpu.topology import ArchitecturePack
from saguaro.math.ir import MathIRRecord


def analyze_prefetch(
    record: MathIRRecord,
    pack: ArchitecturePack,
    *,
    native_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Recommend or reject prefetch strategies from static access shapes."""

    native_prefetch = dict((native_report or {}).get("prefetch") or {})
    if native_prefetch:
        recommendations: list[str] = []
        anti_recommendations: list[str] = []
        distance = int(native_prefetch.get("distance_lines", 0) or 0)
        if bool(native_prefetch.get("streaming", False)):
            recommendations.append(f"Prefetch read streams about {distance} cache lines ahead.")
        if bool(native_prefetch.get("conservative", False)):
            recommendations.append("Use conservative prefetch distance because the loop is strided.")
        if bool(native_prefetch.get("avoid_indirect", False)):
            anti_recommendations.append("Avoid blind prefetch on indirect accesses.")
        if bool(native_prefetch.get("no_runway", False)):
            anti_recommendations.append("No loop context means there is no stable prefetch runway.")
        return {
            "recommended": bool(native_prefetch.get("recommended", False)),
            "recommendations": recommendations,
            "anti_recommendations": anti_recommendations,
            "analysis_engine": str((native_report or {}).get("engine") or "native"),
        }

    recommendations: list[str] = []
    anti_recommendations: list[str] = []
    streaming_reads = [
        item
        for item in record.access_signatures
        if item.access_kind == "read" and item.reuse_hint == "streaming"
    ]
    if streaming_reads:
        recommendations.append(
            f"Prefetch read streams about {pack.prefetch_distance} cache lines ahead."
        )
    if any(item.stride_class == "strided" for item in record.access_signatures):
        recommendations.append("Use conservative prefetch distance because the loop is strided.")
    if any(item.stride_class == "indirect" for item in record.access_signatures):
        anti_recommendations.append("Avoid blind prefetch on indirect accesses.")
    if record.loop_context is None:
        anti_recommendations.append("No loop context means there is no stable prefetch runway.")
    return {
        "recommended": not anti_recommendations and bool(recommendations),
        "recommendations": recommendations,
        "anti_recommendations": anti_recommendations,
        "analysis_engine": "python_fallback",
    }
