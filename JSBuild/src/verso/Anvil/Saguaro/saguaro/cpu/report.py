"""CPU scan report shaping helpers."""

from __future__ import annotations

from typing import Any


def summarize_hotspots(hotspots: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact summary for CLI and benchmark preflight consumers."""

    return {
        "count": len(hotspots),
        "native_analyzed": sum(1 for item in hotspots if item.get("analysis_engine") == "native"),
        "proof_packets_complete": sum(
            1
            for item in hotspots
            if (item.get("proof_packet") or {}).get("completeness", {}).get("complete")
        ),
        "proof_packets_with_contradictions": sum(
            1
            for item in hotspots
            if (item.get("proof_packet") or {})
            .get("completeness", {})
            .get("contradictions")
        ),
        "vectorizable": sum(1 for item in hotspots if item["vectorization"]["legal"]),
        "profitable_vectorization": sum(
            1 for item in hotspots if item["vectorization"]["profitable"]
        ),
        "memory_bound": sum(1 for item in hotspots if item["roofline"]["bound"] == "memory_bound"),
        "high_l3_risk": sum(1 for item in hotspots if item["cache"]["l3_risk"] == "high"),
        "spill_risk": sum(
            1 for item in hotspots if item["register_pressure"]["spill_risk"]
        ),
        "benchmark_candidates": [
            {
                "id": item["id"],
                "file": item["file"],
                "line_start": item["line_start"],
                "benchmark_priority": item["benchmark_priority"],
                "analysis_engine": item.get("analysis_engine", "python_fallback"),
                "proof_packet": str(item.get("proof_packet", {}).get("capsule_id") or ""),
            }
            for item in hotspots[:5]
        ],
    }
