"""ctypes bridge for the native CPU math advisory runtime."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any


class _CpuMathFeatures(ctypes.Structure):
    _fields_ = [
        ("structural_score", ctypes.c_int32),
        ("operator_count", ctypes.c_int32),
        ("symbol_count", ctypes.c_int32),
        ("function_call_count", ctypes.c_int32),
        ("max_nesting_depth", ctypes.c_int32),
        ("access_count", ctypes.c_int32),
        ("contiguous_reads", ctypes.c_int32),
        ("contiguous_writes", ctypes.c_int32),
        ("strided_accesses", ctypes.c_int32),
        ("indirect_accesses", ctypes.c_int32),
        ("streaming_accesses", ctypes.c_int32),
        ("temporal_accesses", ctypes.c_int32),
        ("accumulate_writes", ctypes.c_int32),
        ("has_loop", ctypes.c_int32),
        ("has_recurrence", ctypes.c_int32),
        ("has_reduction", ctypes.c_int32),
        ("native_execution_domain", ctypes.c_int32),
        ("vector_bits", ctypes.c_int32),
        ("lane_width_bits", ctypes.c_int32),
        ("preferred_alignment", ctypes.c_int32),
        ("cache_line_bytes", ctypes.c_int32),
        ("prefetch_distance", ctypes.c_int32),
        ("gather_penalty", ctypes.c_double),
    ]


class _CpuMathScheduleCandidate(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("score", ctypes.c_double),
    ]


class _CpuMathReport(ctypes.Structure):
    _fields_ = [
        ("engine_version", ctypes.c_int32),
        ("vector_legal", ctypes.c_int32),
        ("vector_profitable", ctypes.c_int32),
        ("recommended_lane_count", ctypes.c_int32),
        ("blocker_mask", ctypes.c_uint32),
        ("vector_score", ctypes.c_double),
        ("alignment_bytes", ctypes.c_int32),
        ("prefetch_recommended", ctypes.c_int32),
        ("prefetch_distance_lines", ctypes.c_int32),
        ("prefetch_streaming", ctypes.c_int32),
        ("prefetch_conservative", ctypes.c_int32),
        ("prefetch_avoid_indirect", ctypes.c_int32),
        ("prefetch_no_runway", ctypes.c_int32),
        ("estimated_ops", ctypes.c_int32),
        ("estimated_bytes", ctypes.c_int32),
        ("operational_intensity", ctypes.c_double),
        ("bound_class", ctypes.c_int32),
        ("estimated_cache_lines_touched", ctypes.c_int32),
        ("reuse_distance_class", ctypes.c_int32),
        ("l1_risk", ctypes.c_int32),
        ("l2_risk", ctypes.c_int32),
        ("l3_risk", ctypes.c_int32),
        ("memory_pressure_score", ctypes.c_double),
        ("register_pressure_score", ctypes.c_int32),
        ("register_pressure_band", ctypes.c_int32),
        ("spill_risk", ctypes.c_int32),
        ("benchmark_priority", ctypes.c_double),
        ("schedule_count", ctypes.c_int32),
        ("schedules", _CpuMathScheduleCandidate * 3),
    ]


_LIB: ctypes.CDLL | None = None
_LOAD_FAILED = False


def _native_library_candidates() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[3]
    native_root = repo_root / "core" / "native"
    return [
        native_root / "libanvil_native_ops.so",
        native_root / "build" / "libanvil_native_ops.so",
    ]


def _lib() -> ctypes.CDLL | None:
    global _LIB
    global _LOAD_FAILED
    if _LOAD_FAILED:
        return None
    if _LIB is not None:
        return _LIB
    for candidate in _native_library_candidates():
        if not candidate.exists():
            continue
        try:
            lib = ctypes.CDLL(str(candidate))
            lib.anvil_cpu_math_analyze.argtypes = [
                ctypes.POINTER(_CpuMathFeatures),
                ctypes.POINTER(_CpuMathReport),
            ]
            lib.anvil_cpu_math_analyze.restype = None
            lib.anvil_cpu_math_runtime_version.argtypes = []
            lib.anvil_cpu_math_runtime_version.restype = ctypes.c_char_p
            _LIB = lib
            return lib
        except Exception:
            continue
    _LOAD_FAILED = True
    return None


def analyze_cpu_math(features: dict[str, Any]) -> dict[str, Any] | None:
    """Run the native CPU math runtime when the shared library exports it."""

    lib = _lib()
    if lib is None:
        return None
    payload = _CpuMathFeatures(
        structural_score=int(features.get("structural_score", 0) or 0),
        operator_count=int(features.get("operator_count", 0) or 0),
        symbol_count=int(features.get("symbol_count", 0) or 0),
        function_call_count=int(features.get("function_call_count", 0) or 0),
        max_nesting_depth=int(features.get("max_nesting_depth", 0) or 0),
        access_count=int(features.get("access_count", 0) or 0),
        contiguous_reads=int(features.get("contiguous_reads", 0) or 0),
        contiguous_writes=int(features.get("contiguous_writes", 0) or 0),
        strided_accesses=int(features.get("strided_accesses", 0) or 0),
        indirect_accesses=int(features.get("indirect_accesses", 0) or 0),
        streaming_accesses=int(features.get("streaming_accesses", 0) or 0),
        temporal_accesses=int(features.get("temporal_accesses", 0) or 0),
        accumulate_writes=int(features.get("accumulate_writes", 0) or 0),
        has_loop=int(bool(features.get("has_loop", False))),
        has_recurrence=int(bool(features.get("has_recurrence", False))),
        has_reduction=int(bool(features.get("has_reduction", False))),
        native_execution_domain=int(bool(features.get("native_execution_domain", False))),
        vector_bits=int(features.get("vector_bits", 0) or 0),
        lane_width_bits=int(features.get("lane_width_bits", 0) or 0),
        preferred_alignment=int(features.get("preferred_alignment", 0) or 0),
        cache_line_bytes=int(features.get("cache_line_bytes", 0) or 0),
        prefetch_distance=int(features.get("prefetch_distance", 0) or 0),
        gather_penalty=float(features.get("gather_penalty", 1.0) or 1.0),
    )
    report = _CpuMathReport()
    lib.anvil_cpu_math_analyze(ctypes.byref(payload), ctypes.byref(report))
    version = lib.anvil_cpu_math_runtime_version()
    runtime_version = (
        version.decode("utf-8", errors="ignore")
        if isinstance(version, bytes)
        else str(version or "")
    )
    return {
        "engine": "native",
        "runtime_version": runtime_version,
        "engine_version": int(report.engine_version),
        "vectorization": {
            "legal": bool(report.vector_legal),
            "profitable": bool(report.vector_profitable),
            "recommended_lane_count": int(report.recommended_lane_count),
            "blocker_mask": int(report.blocker_mask),
            "vector_score": float(report.vector_score),
            "alignment_bytes": int(report.alignment_bytes),
        },
        "prefetch": {
            "recommended": bool(report.prefetch_recommended),
            "distance_lines": int(report.prefetch_distance_lines),
            "streaming": bool(report.prefetch_streaming),
            "conservative": bool(report.prefetch_conservative),
            "avoid_indirect": bool(report.prefetch_avoid_indirect),
            "no_runway": bool(report.prefetch_no_runway),
        },
        "roofline": {
            "estimated_ops": int(report.estimated_ops),
            "estimated_bytes": int(report.estimated_bytes),
            "operational_intensity": float(report.operational_intensity),
            "bound_class": int(report.bound_class),
        },
        "cache": {
            "estimated_cache_lines_touched": int(report.estimated_cache_lines_touched),
            "reuse_distance_class": int(report.reuse_distance_class),
            "l1_risk": int(report.l1_risk),
            "l2_risk": int(report.l2_risk),
            "l3_risk": int(report.l3_risk),
            "memory_pressure_score": float(report.memory_pressure_score),
        },
        "register_pressure": {
            "score": int(report.register_pressure_score),
            "band": int(report.register_pressure_band),
            "spill_risk": bool(report.spill_risk),
        },
        "schedule_twin": {
            "candidates": [
                {
                    "kind": int(report.schedules[index].kind),
                    "score": float(report.schedules[index].score),
                }
                for index in range(max(0, int(report.schedule_count)))
                if int(report.schedules[index].kind) > 0
            ]
        },
        "benchmark_priority": float(report.benchmark_priority),
    }
