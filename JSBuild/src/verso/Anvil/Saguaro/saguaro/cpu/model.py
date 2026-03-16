"""End-to-end static CPU scan built on top of Saguaro math extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from saguaro.cpu.cache import analyze_cache
from saguaro.cpu.native_runtime import analyze_cpu_math
from saguaro.cpu.prefetch import analyze_prefetch
from saguaro.cpu.register_pressure import estimate_register_pressure
from saguaro.cpu.report import summarize_hotspots
from saguaro.cpu.roofline import estimate_roofline
from saguaro.cpu.schedule_twin import propose_schedule_twin
from saguaro.cpu.topology import get_architecture_pack
from saguaro.cpu.vectorization import analyze_vectorization
from saguaro.math import MathEngine
from saguaro.validation.hotspot_capsules import persist_hotspot_capsules


class CPUScanner:
    """Static CPU advisory surface for hotspot triage before benchmarking."""

    def __init__(self, repo_path: str) -> None:
        self.repo_path = Path(repo_path).resolve()

    def scan(
        self,
        *,
        path: str = ".",
        arch: str = "x86_64-avx2",
        limit: int = 20,
    ) -> dict[str, Any]:
        pack = get_architecture_pack(arch)
        math_payload = MathEngine(str(self.repo_path)).parse(path)
        record_payloads = list(math_payload.get("records") or math_payload.get("equations") or [])
        hotspots: list[dict[str, Any]] = []
        for record in record_payloads:
            if record.get("source_kind") != "code_expression":
                continue
            if (record.get("provenance") or {}).get("execution_domain") != "native_kernel":
                continue
            record_proxy = _RecordProxy(record)
            native_report = analyze_cpu_math(self._feature_vector(record_proxy, pack))
            vectorization = analyze_vectorization(
                record_proxy,
                pack,
                native_report=native_report,
            )
            prefetch = analyze_prefetch(
                record_proxy,
                pack,
                native_report=native_report,
            )
            roofline = estimate_roofline(
                record_proxy,
                pack,
                native_report=native_report,
            )
            cache = analyze_cache(
                record_proxy,
                pack,
                native_report=native_report,
            )
            register_pressure = estimate_register_pressure(
                record_proxy,
                pack,
                native_report=native_report,
            )
            schedule_twin = propose_schedule_twin(
                record_proxy,
                pack,
                vectorization=vectorization,
                roofline=roofline,
                native_report=native_report,
            )
            benchmark_priority = self._benchmark_priority(
                record=record,
                vectorization=vectorization,
                roofline=roofline,
                register_pressure=register_pressure,
                cache=cache,
                native_report=native_report,
            )
            hotspots.append(
                {
                    "id": record["id"],
                    "file": record["file"],
                    "line_start": record["line_start"],
                    "expression": record["expression"],
                    "language": record["language"],
                    "complexity": record.get("complexity") or {},
                    "loop_context": record.get("loop_context"),
                    "access_signatures": record.get("access_signatures") or [],
                    "layout_states": record.get("layout_states") or [],
                    "complexity_reduction_hints": record.get("complexity_reduction_hints")
                    or [],
                    "provenance": record.get("provenance") or {},
                    "analysis_engine": (
                        "native_cpp"
                        if native_report is not None
                        else "python_fallback"
                    ),
                    "native_runtime_version": str((native_report or {}).get("runtime_version") or ""),
                    "cache": cache,
                    "vectorization": vectorization,
                    "prefetch": prefetch,
                    "roofline": roofline,
                    "register_pressure": register_pressure,
                    "schedule_twin": schedule_twin,
                    "optimization_packet": {
                        "schedule_recipe": dict(schedule_twin.get("selected") or {}),
                        "reduction_hints": list(
                            record.get("complexity_reduction_hints") or []
                        ),
                    },
                    "benchmark_priority": benchmark_priority,
                }
            )
        hotspots.sort(key=lambda item: item["benchmark_priority"], reverse=True)
        limited = hotspots[: max(1, int(limit or 20))]
        capsule_manifest = persist_hotspot_capsules(
            str(self.repo_path),
            limited,
            arch=pack.arch,
            scan_path=str(math_payload.get("path", path) or path),
            math_cache_path=str(math_payload.get("cache_path", "") or ""),
        )
        capsule_by_source = {
            (
                str(capsule.get("source", {}).get("file") or ""),
                int(capsule.get("source", {}).get("line_start", 0) or 0),
                str(capsule.get("source", {}).get("id") or ""),
            ): capsule
            for capsule in list(capsule_manifest.get("capsules") or [])
        }
        for hotspot in limited:
            proof_packet = capsule_by_source.get(
                (hotspot["file"], int(hotspot["line_start"]), hotspot["id"])
            )
            hotspot["proof_packet"] = proof_packet or {}
            hotspot["proof_packet_path"] = str(
                (proof_packet or {}).get("artifact_paths", {}).get("capsule_path") or ""
            )
        return {
            "status": "ok",
            "path": math_payload.get("path", path),
            "arch": pack.arch,
            "architecture_pack": pack.to_dict(),
            "files_scanned": math_payload.get("files_scanned", 0),
            "math_record_count": math_payload.get("count", 0),
            "hotspot_count": len(limited),
            "hotspots": limited,
            "summary": summarize_hotspots(limited),
            "math_cache_path": math_payload.get("cache_path", ""),
            "capsule_manifest": capsule_manifest,
        }

    @staticmethod
    def _benchmark_priority(
        *,
        record: dict[str, Any],
        vectorization: dict[str, Any],
        roofline: dict[str, Any],
        register_pressure: dict[str, Any],
        cache: dict[str, Any],
        native_report: dict[str, Any] | None,
    ) -> float:
        if native_report is not None:
            return round(float(native_report.get("benchmark_priority", 0.0) or 0.0), 2)
        complexity = float((record.get("complexity") or {}).get("structural_score", 0.0) or 0.0)
        score = complexity
        if vectorization.get("profitable"):
            score += 6.0
        elif vectorization.get("legal"):
            score += 2.0
        if roofline.get("bound") == "memory_bound":
            score += 4.0
        if register_pressure.get("spill_risk"):
            score += 3.0
        if record.get("provenance", {}).get("execution_domain") == "native_kernel":
            score += 5.0
        if cache.get("l3_risk") == "high":
            score += 2.0
        elif cache.get("l3_risk") == "medium":
            score += 1.0
        return round(score, 2)

    @staticmethod
    def _feature_vector(record: "_RecordProxy", pack: Any) -> dict[str, Any]:
        accesses = list(record.access_signatures)
        return {
            "structural_score": int(record.complexity.structural_score),
            "operator_count": int(record.complexity.operator_count),
            "symbol_count": int(record.complexity.symbol_count),
            "function_call_count": int(record.complexity.function_call_count),
            "max_nesting_depth": int(record.complexity.max_nesting_depth),
            "access_count": len(accesses),
            "contiguous_reads": sum(
                1
                for item in accesses
                if item.access_kind == "read"
                and item.stride_class in {"contiguous", "contiguous_offset"}
            ),
            "contiguous_writes": sum(
                1
                for item in accesses
                if item.access_kind == "write"
                and item.stride_class in {"contiguous", "contiguous_offset", "scalar"}
            ),
            "strided_accesses": sum(1 for item in accesses if item.stride_class == "strided"),
            "indirect_accesses": sum(1 for item in accesses if item.stride_class == "indirect"),
            "streaming_accesses": sum(1 for item in accesses if item.reuse_hint == "streaming"),
            "temporal_accesses": sum(
                1 for item in accesses if item.reuse_hint == "temporal_reuse"
            ),
            "accumulate_writes": sum(1 for item in accesses if item.write_mode == "accumulate"),
            "has_loop": bool(record.loop_context),
            "has_recurrence": bool(record.loop_context and record.loop_context.recurrence),
            "has_reduction": bool(record.loop_context and record.loop_context.reduction),
            "native_execution_domain": record.provenance.get("execution_domain") == "native_kernel",
            "vector_bits": int(pack.vector_bits),
            "lane_width_bits": int(pack.lane_width_bits),
            "preferred_alignment": int(pack.preferred_alignment),
            "cache_line_bytes": int(pack.cache_line_bytes),
            "prefetch_distance": int(pack.prefetch_distance),
            "gather_penalty": float(pack.gather_penalty),
        }


class _RecordProxy:
    """Read-only adapter so CPU heuristics can consume dict-backed records."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.id = str(payload.get("id") or "")
        self.file = str(payload.get("file") or "")
        self.expression = str(payload.get("expression") or "")
        self.line_start = int(payload.get("line_start") or 0)
        self.language = str(payload.get("language") or "")
        self.access_signatures = [
            _AccessProxy(item) for item in list(payload.get("access_signatures") or [])
        ]
        self.layout_states = list(payload.get("layout_states") or [])
        self.provenance = dict(payload.get("provenance") or {})
        loop_payload = payload.get("loop_context") or None
        self.loop_context = _LoopProxy(loop_payload) if loop_payload else None
        self.complexity = _ComplexityProxy(payload.get("complexity") or {})


class _LoopProxy:
    def __init__(self, payload: dict[str, Any] | None) -> None:
        self._payload = dict(payload or {})
        self.loop_variables = list(self._payload.get("loop_variables") or [])
        self.recurrence = bool(self._payload.get("recurrence", False))
        self.reduction = bool(self._payload.get("reduction", False))

    def __bool__(self) -> bool:
        return bool(self._payload)


class _AccessProxy:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.base_symbol = str(payload.get("base_symbol") or "")
        self.access_kind = str(payload.get("access_kind") or "")
        self.index_expression = str(payload.get("index_expression") or "")
        self.index_affinity = str(payload.get("index_affinity") or "")
        self.stride_class = str(payload.get("stride_class") or "")
        self.reuse_hint = str(payload.get("reuse_hint") or "")
        self.write_mode = str(payload.get("write_mode") or "")


class _ComplexityProxy:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.operator_count = int(payload.get("operator_count", 0) or 0)
        self.symbol_count = int(payload.get("symbol_count", 0) or 0)
        self.function_call_count = int(payload.get("function_call_count", 0) or 0)
        self.max_nesting_depth = int(payload.get("max_nesting_depth", 0) or 0)
        self.structural_score = int(payload.get("structural_score", 0) or 0)
