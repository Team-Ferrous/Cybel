"""Static SIMD legality and profitability heuristics."""

from __future__ import annotations

from saguaro.cpu.topology import ArchitecturePack
from saguaro.math.ir import MathIRRecord

_BLOCKER_BITS = {
    1: ("no_loop_context", "No loop context was found around the statement."),
    2: ("indirect_access", "Indirect indexing makes lane-wise legality uncertain."),
    4: (
        "loop_carried_recurrence",
        "Loop-carried recurrence prevents straightforward vectorization.",
    ),
}


def analyze_vectorization(
    record: MathIRRecord,
    pack: ArchitecturePack,
    *,
    native_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Predict whether the record can use SIMD profitably."""

    native_vector = dict((native_report or {}).get("vectorization") or {})
    if native_vector:
        blocker_mask = int(native_vector.get("blocker_mask", 0) or 0)
        blockers: list[str] = []
        reasons: list[str] = []
        for bit, (blocker, reason) in sorted(_BLOCKER_BITS.items()):
            if blocker_mask & bit:
                blockers.append(blocker)
                reasons.append(reason)
        if record.loop_context is not None and record.loop_context.reduction:
            reasons.append("Reduction shape is a legal SIMD candidate with a horizontal combine.")
        if record.complexity is not None and record.complexity.function_call_count:
            reasons.append("Function calls reduce profitability unless they inline cleanly.")
        legal = bool(native_vector.get("legal", False))
        profitable = bool(native_vector.get("profitable", False))
        lane_count = int(native_vector.get("recommended_lane_count", 1) or 1)
        if legal and not profitable:
            reasons.append("The loop appears legal, but the arithmetic density is marginal.")
        if legal and profitable:
            reasons.append(
                f"{pack.isa_family.upper()} can amortize the contiguous work across roughly {lane_count} lanes."
            )
        return {
            "legal": legal,
            "profitable": profitable,
            "recommended_lane_count": lane_count if profitable else 1,
            "blockers": blockers,
            "reasons": reasons,
            "alignment_bytes": int(native_vector.get("alignment_bytes", pack.preferred_alignment) or pack.preferred_alignment),
            "analysis_engine": str((native_report or {}).get("engine") or "native"),
            "score": round(float(native_vector.get("vector_score", 0.0) or 0.0), 3),
        }

    blockers: list[str] = []
    reasons: list[str] = []
    loop = record.loop_context
    accesses = record.access_signatures
    if loop is None:
        blockers.append("no_loop_context")
        reasons.append("No loop context was found around the statement.")
    if any(item.stride_class == "indirect" for item in accesses):
        blockers.append("indirect_access")
        reasons.append("Indirect indexing makes lane-wise legality uncertain.")
    if loop is not None and loop.recurrence and not loop.reduction:
        blockers.append("loop_carried_recurrence")
        reasons.append("Loop-carried recurrence prevents straightforward vectorization.")
    legal = not blockers
    contiguous_reads = sum(
        1 for item in accesses if item.stride_class in {"contiguous", "contiguous_offset"}
    )
    score = int(record.complexity.structural_score if record.complexity is not None else 0)
    score += contiguous_reads * 2
    score -= 3 * sum(1 for item in accesses if item.stride_class == "strided")
    score -= 5 * sum(1 for item in accesses if item.stride_class == "indirect")
    if loop is not None and loop.reduction:
        reasons.append("Reduction shape is a legal SIMD candidate with a horizontal combine.")
        score += 2
    if record.complexity is not None and record.complexity.function_call_count:
        reasons.append("Function calls reduce profitability unless they inline cleanly.")
        score -= record.complexity.function_call_count * 2
    profitable = legal and score >= 10
    lane_count = max(1, pack.vector_bits // max(pack.lane_width_bits, 1))
    if legal and not profitable:
        reasons.append("The loop appears legal, but the arithmetic density is marginal.")
    if legal and profitable:
        reasons.append(
            f"{pack.isa_family.upper()} can amortize the contiguous work across roughly {lane_count} lanes."
        )
    return {
        "legal": legal,
        "profitable": profitable,
        "recommended_lane_count": lane_count if profitable else 1,
        "blockers": blockers,
        "reasons": reasons,
        "alignment_bytes": pack.preferred_alignment,
        "analysis_engine": "python_fallback",
        "score": round(float(score), 3),
    }
