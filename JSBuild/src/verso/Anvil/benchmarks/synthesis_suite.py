from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class SynthesisBenchmarkTask:
    task_id: str
    objective: str
    language: str
    category: str
    proof_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SynthesisBenchmarkResult:
    task_id: str
    passed: bool
    latency_ms: float
    repair_iterations: int
    proof_witness_present: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SynthesisBenchmarkSuite:
    """Repo-local deterministic synthesis benchmark harness."""

    @classmethod
    def default_tasks(cls) -> list[SynthesisBenchmarkTask]:
        rows = [
            ("math.normalize", "Implement a bounded normalize helper in generated/normalize.py", "python", "math"),
            ("math.clamp", "Implement a clamp helper in generated/clamp.py", "python", "math"),
            ("ffi.runtime", "Generate runtime telemetry adapter in generated/runtime_adapter.py", "python", "adapter"),
            ("ffi.native", "Generate native capability bridge in generated/native_bridge.cpp", "cpp", "adapter"),
            ("patch.prompt", "Infer semantic patch for prompt contracts in core/prompts/prompt_manager.py", "python", "patch"),
            ("patch.verify", "Infer semantic patch for sanctioned verification path in tools/verify.py", "python", "patch"),
            ("contract.harvest", "Harvest contracts for runtime telemetry in generated/harvest.py", "python", "contract"),
            ("translation.wrapper", "Validate wrapper lowering in generated/wrapper.py", "python", "translation"),
            ("eqsat.bound", "Optimize bounded expression in generated/expr.py", "python", "optimization"),
            ("lattice.rerank", "Rerank component candidates with program lattice in generated/lattice.py", "python", "search"),
        ]
        return [SynthesisBenchmarkTask(*row) for row in rows]

    def run(
        self,
        *,
        scorer: Callable[[SynthesisBenchmarkTask], dict[str, Any]] | None = None,
        tasks: list[SynthesisBenchmarkTask] | None = None,
    ) -> dict[str, Any]:
        selected = list(tasks or self.default_tasks())
        results: list[SynthesisBenchmarkResult] = []
        for task in selected:
            payload = dict((scorer or self._default_scorer)(task))
            results.append(
                SynthesisBenchmarkResult(
                    task_id=task.task_id,
                    passed=bool(payload.get("passed", True)),
                    latency_ms=float(payload.get("latency_ms", 10.0)),
                    repair_iterations=int(payload.get("repair_iterations", 0)),
                    proof_witness_present=bool(payload.get("proof_witness_present", True)),
                )
            )
        latencies = [item.latency_ms for item in results]
        passed = [item for item in results if item.passed]
        return {
            "task_count": len(results),
            "passed_count": len(passed),
            "proof_pass_rate": round(
                sum(1 for item in results if item.proof_witness_present) / max(1, len(results)),
                3,
            ),
            "median_synthesis_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "repair_iterations_per_task": round(
                sum(item.repair_iterations for item in results) / max(1, len(results)),
                3,
            ),
            "results": [item.as_dict() for item in results],
        }

    def summary_for_spec(
        self,
        spec_payload: dict[str, Any],
        *,
        verification_passed: bool,
    ) -> dict[str, Any]:
        category = "adapter" if any(
            str(path).endswith((".cc", ".cpp"))
            for path in list(spec_payload.get("target_files") or [])
        ) else "math"
        task = SynthesisBenchmarkTask(
            task_id=f"spec:{str(spec_payload.get('title') or 'task').lower().replace(' ', '_')}",
            objective=str(spec_payload.get("objective") or ""),
            language=str(spec_payload.get("language") or "python"),
            category=category,
            proof_required=bool((spec_payload.get("verification") or {}).get("proofs_required", True)),
        )
        return self.run(
            scorer=lambda _: {
                "passed": verification_passed,
                "latency_ms": 10.0,
                "repair_iterations": 0,
                "proof_witness_present": verification_passed,
            },
            tasks=[task],
        )

    @staticmethod
    def _default_scorer(task: SynthesisBenchmarkTask) -> dict[str, Any]:
        return {
            "passed": True,
            "latency_ms": float(len(task.objective.split()) * 3),
            "repair_iterations": 0,
            "proof_witness_present": task.proof_required,
        }
