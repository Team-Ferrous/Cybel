"""Counterfactual schedule suggestions for hotspot records."""

from __future__ import annotations

from saguaro.cpu.topology import ArchitecturePack
from saguaro.math.ir import MathIRRecord

_SCHEDULE_LIBRARY = {
    "vectorize": {
        "kind": 1,
        "reason": "Exploit ISA lane parallelism.",
        "transforms": ["widen_simd_lane", "align_loads", "hoist_tail_mask"],
        "prerequisites": ["vectorization_legal"],
    },
    "cache_block": {
        "kind": 2,
        "reason": "Reduce cache-line churn on streaming accesses.",
        "transforms": ["tile_outer_loop", "pack_temporal_window", "prefetch_stream"],
        "prerequisites": ["memory_bound_or_streaming"],
    },
    "tree_reduce": {
        "kind": 3,
        "reason": "Use a partial reduction tree before the scalar combine.",
        "transforms": ["split_reduction_tree", "vector_partial_accumulate", "scalar_epilogue"],
        "prerequisites": ["reduction_detected"],
    },
    "scalar_stabilize": {
        "kind": 4,
        "reason": "Stabilize the recurrence before wider SIMD rewrites.",
        "transforms": ["isolate_recurrence", "hoist_invariants", "narrow_scalar_core"],
        "prerequisites": ["recurrence_detected"],
    },
}
_SCHEDULE_NAMES = {
    item["kind"]: (name, str(item["reason"]))
    for name, item in _SCHEDULE_LIBRARY.items()
}


def propose_schedule_twin(
    record: MathIRRecord,
    pack: ArchitecturePack,
    *,
    vectorization: dict[str, object],
    roofline: dict[str, object],
    native_report: dict[str, object] | None = None,
) -> dict[str, object]:
    """Rank a constrained set of static schedule alternatives."""

    native_schedule = dict((native_report or {}).get("schedule_twin") or {})
    if native_schedule:
        candidates: list[dict[str, object]] = []
        for item in list(native_schedule.get("candidates") or []):
            kind = int(item.get("kind", 0) or 0)
            name, reason = _SCHEDULE_NAMES.get(kind, ("unknown", "Native runtime did not classify this candidate."))
            candidates.append(
                _candidate_payload(
                    name=name,
                    score=float(item.get("score", 0.0) or 0.0),
                    reason=(
                        f"Exploit {pack.isa_family.upper()} lane parallelism."
                        if name == "vectorize"
                        else reason
                    ),
                    vectorization=vectorization,
                    roofline=roofline,
                    has_reduction=bool(record.loop_context and record.loop_context.reduction),
                    has_recurrence=bool(record.loop_context and record.loop_context.recurrence),
                )
            )
        return _schedule_payload(
            candidates[:3],
            analysis_engine=str((native_report or {}).get("engine") or "native"),
        )

    candidates: list[dict[str, object]] = []
    if vectorization.get("legal"):
        candidates.append(
            _candidate_payload(
                name="vectorize",
                score=0.9 if vectorization.get("profitable") else 0.55,
                reason=f"Exploit {pack.isa_family.upper()} lane parallelism.",
                vectorization=vectorization,
                roofline=roofline,
                has_reduction=bool(record.loop_context and record.loop_context.reduction),
                has_recurrence=bool(record.loop_context and record.loop_context.recurrence),
            )
        )
    if any(item.reuse_hint == "streaming" for item in record.access_signatures):
        candidates.append(
            _candidate_payload(
                name="cache_block",
                score=0.75 if roofline.get("bound") == "memory_bound" else 0.5,
                reason="Reduce cache-line churn on streaming accesses.",
                vectorization=vectorization,
                roofline=roofline,
                has_reduction=bool(record.loop_context and record.loop_context.reduction),
                has_recurrence=bool(record.loop_context and record.loop_context.recurrence),
            )
        )
    if record.loop_context is not None and record.loop_context.reduction:
        candidates.append(
            _candidate_payload(
                name="tree_reduce",
                score=0.7,
                reason="Use a partial reduction tree before the scalar combine.",
                vectorization=vectorization,
                roofline=roofline,
                has_reduction=True,
                has_recurrence=bool(record.loop_context.recurrence),
            )
        )
    if record.loop_context is not None and record.loop_context.recurrence:
        candidates.append(
            _candidate_payload(
                name="scalar_stabilize",
                score=0.64,
                reason="Stabilize the recurrence before wider SIMD rewrites.",
                vectorization=vectorization,
                roofline=roofline,
                has_reduction=bool(record.loop_context.reduction),
                has_recurrence=True,
            )
        )
    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return _schedule_payload(candidates[:3], analysis_engine="python_fallback")


def _candidate_payload(
    *,
    name: str,
    score: float,
    reason: str,
    vectorization: dict[str, object],
    roofline: dict[str, object],
    has_reduction: bool,
    has_recurrence: bool,
) -> dict[str, object]:
    spec = dict(_SCHEDULE_LIBRARY.get(name) or {})
    blockers: list[str] = []
    if name == "vectorize" and not vectorization.get("legal"):
        blockers.append("vectorization_illegal")
    if name == "cache_block" and roofline.get("bound") != "memory_bound":
        blockers.append("not_memory_bound")
    if name == "tree_reduce" and not has_reduction:
        blockers.append("reduction_not_detected")
    if name == "scalar_stabilize" and not has_recurrence:
        blockers.append("recurrence_not_detected")
    recipe = {
        "recipe_id": f"schedule::{name}",
        "transforms": list(spec.get("transforms") or []),
        "prerequisites": list(spec.get("prerequisites") or []),
        "rationale": reason,
    }
    return {
        "name": name,
        "score": round(float(score), 3),
        "reason": reason,
        "blockers": blockers,
        "recipe": recipe,
        "transforms": list(recipe["transforms"]),
    }


def _schedule_payload(
    candidates: list[dict[str, object]],
    *,
    analysis_engine: str,
) -> dict[str, object]:
    selected = dict(candidates[0]) if candidates else {}
    return {
        "candidates": candidates,
        "selected": selected,
        "analysis_engine": analysis_engine,
    }
