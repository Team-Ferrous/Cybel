from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ENABLED_LANES = [
    "canonical_all_on",
    "continuous_scheduler",
    "kernel_microbench",
    "quality_eval",
]


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    env: dict[str, str] = field(default_factory=dict)
    measured_runs: int = 0
    warmup_runs: int = 0


@dataclass(frozen=True)
class KernelMicrobenchSpec:
    warmups: int
    iterations: int
    synthetic_lengths: list[int]
    target_kernels: list[str]


@dataclass(frozen=True)
class QualityEvalSpec:
    perplexity_corpus: str
    confidence_corpus: str
    rubric_corpus: str
    accuracy_corpus: str = ""
    max_samples_per_family: int = 0
    adaptive_top_k: int = 0
    shadow_mode: bool = True


@dataclass(frozen=True)
class MemoryReplaySpec:
    db_path: str = ""
    campaign_id: str = ""
    storage_root: str = ""
    cases_path: str = ""
    case_limit: int = 0
    gate_thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioPack:
    canonical_prompt: str
    canonical_max_new_tokens: int
    canonical_context_length: int
    warmup_runs: int
    measured_runs: int
    thread_matrix_ubatch: list[int]
    continuous_concurrency: list[int]
    continuous_prompt_classes: list[str]
    continuous_scheduler_policies: list[str]


@dataclass(frozen=True)
class CalibrationSearchSpec:
    bootstrap_profile: str = "probe"
    search_profile: str = "search"
    deep_search_profile: str = "deep_search"
    stage1_top_k: int = 3
    stage2_top_k: int = 4
    scheduler_top_k: int = 2
    max_scheduler_candidates: int = 12
    fairness_floor: float = 0.85
    queue_wait_p95_ceiling_ms: float = 250.0
    decode_tps_regression_floor_pct: float = 0.92
    continuous_max_active_requests: list[int] = field(default_factory=list)
    continuous_batch_wait_timeout_ms: list[int] = field(
        default_factory=lambda: [1, 2, 4]
    )
    continuous_state_page_rows: list[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    continuous_max_prefill_rows_per_iteration: list[int] = field(
        default_factory=lambda: [512, 1024]
    )
    continuous_interleaved_streams: list[bool] = field(
        default_factory=lambda: [False, True]
    )
    probe_kernel_iterations: int = 0
    search_kernel_iterations: int = 24
    deep_search_kernel_iterations: int = 64


@dataclass(frozen=True)
class BenchmarkSuiteSpec:
    schema_version: str
    profile_name: str
    models: list[str]
    max_parallel_models: int
    strict_host: bool
    require_saguaro: bool
    require_perf: bool
    host_contract_id: str | None
    affinity_policy: str
    tuning_contract_policy: str
    preflight_strictness: str
    assurance_level: str
    evidence_class: str
    force_parallel_decode: bool
    forbid_autoregressive_fallback: bool
    canonical_decode_threads: list[int | None]
    canonical_batch_threads: list[int | None]
    scenario_pack: ScenarioPack
    calibration_search: CalibrationSearchSpec
    kernel_microbench: KernelMicrobenchSpec
    quality_eval: QualityEvalSpec
    ablations: list[AblationSpec] = field(default_factory=list)
    memory_replay: MemoryReplaySpec = field(default_factory=MemoryReplaySpec)
    enabled_lanes: list[str] = field(
        default_factory=lambda: list(DEFAULT_ENABLED_LANES)
    )
    host_requirements: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Suite profile must decode to a mapping: {path}")
    return payload


def _ablation_specs(items: list[dict[str, Any]]) -> list[AblationSpec]:
    results: list[AblationSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            AblationSpec(
                ablation_id=str(item.get("ablation_id") or "").strip(),
                env={
                    str(key): str(value)
                    for key, value in dict(item.get("env") or {}).items()
                },
                measured_runs=int(item.get("measured_runs", 0) or 0),
                warmup_runs=int(item.get("warmup_runs", 0) or 0),
            )
        )
    return [item for item in results if item.ablation_id]


def load_suite_profile(path: Path) -> BenchmarkSuiteSpec:
    payload = _load_yaml(path)
    scenario_raw = dict(payload.get("scenario_pack") or {})
    calibration_raw = dict(payload.get("calibration_search") or {})
    kernel_raw = dict(payload.get("kernel_microbench") or {})
    quality_raw = dict(payload.get("quality_eval") or {})
    memory_raw = dict(payload.get("memory_replay") or {})
    return BenchmarkSuiteSpec(
        schema_version=str(payload.get("schema_version") or "native_qsg_suite.v1"),
        profile_name=str(payload.get("profile_name") or path.stem),
        models=[
            str(item) for item in list(payload.get("models") or []) if str(item).strip()
        ],
        max_parallel_models=int(payload.get("max_parallel_models", 1) or 1),
        strict_host=bool(payload.get("strict_host", True)),
        require_saguaro=bool(payload.get("require_saguaro", True)),
        require_perf=bool(payload.get("require_perf", True)),
        host_contract_id=str(payload.get("host_contract_id") or "").strip() or None,
        affinity_policy=str(payload.get("affinity_policy") or "repair_allowed").strip(),
        tuning_contract_policy=str(
            payload.get("tuning_contract_policy") or "optional"
        ).strip(),
        preflight_strictness=str(
            payload.get("preflight_strictness") or "certify"
        ).strip(),
        assurance_level=str(payload.get("assurance_level") or "AAL-2").strip().upper(),
        evidence_class=str(payload.get("evidence_class") or "exploratory").strip(),
        force_parallel_decode=bool(payload.get("force_parallel_decode", False)),
        forbid_autoregressive_fallback=bool(
            payload.get("forbid_autoregressive_fallback", False)
        ),
        canonical_decode_threads=[
            (None if item is None else int(item))
            for item in list(payload.get("canonical_decode_threads") or [None])
        ],
        canonical_batch_threads=[
            (None if item is None else int(item))
            for item in list(payload.get("canonical_batch_threads") or [None])
        ],
        scenario_pack=ScenarioPack(
            canonical_prompt=str(scenario_raw.get("canonical_prompt") or ""),
            canonical_max_new_tokens=int(
                scenario_raw.get("canonical_max_new_tokens", 32) or 32
            ),
            canonical_context_length=int(
                scenario_raw.get("canonical_context_length", 2048) or 2048
            ),
            warmup_runs=int(scenario_raw.get("warmup_runs", 1) or 1),
            measured_runs=int(scenario_raw.get("measured_runs", 3) or 3),
            thread_matrix_ubatch=[
                int(item)
                for item in list(
                    scenario_raw.get("thread_matrix_ubatch") or [16, 32, 64]
                )
            ],
            continuous_concurrency=[
                int(item)
                for item in list(
                    scenario_raw.get("continuous_concurrency") or [1, 2, 4, 8]
                )
            ],
            continuous_prompt_classes=[
                str(item)
                for item in list(
                    scenario_raw.get("continuous_prompt_classes")
                    or ["short", "medium", "long"]
                )
            ],
            continuous_scheduler_policies=[
                str(item)
                for item in list(
                    scenario_raw.get("continuous_scheduler_policies")
                    or ["fcfs", "priority"]
                )
            ],
        ),
        calibration_search=CalibrationSearchSpec(
            bootstrap_profile=str(
                calibration_raw.get("bootstrap_profile") or "probe"
            ).strip(),
            search_profile=str(
                calibration_raw.get("search_profile") or "search"
            ).strip(),
            deep_search_profile=str(
                calibration_raw.get("deep_search_profile") or "deep_search"
            ).strip(),
            stage1_top_k=int(calibration_raw.get("stage1_top_k", 3) or 3),
            stage2_top_k=int(calibration_raw.get("stage2_top_k", 4) or 4),
            scheduler_top_k=int(calibration_raw.get("scheduler_top_k", 2) or 2),
            max_scheduler_candidates=int(
                calibration_raw.get("max_scheduler_candidates", 12) or 12
            ),
            fairness_floor=float(calibration_raw.get("fairness_floor", 0.85) or 0.85),
            queue_wait_p95_ceiling_ms=float(
                calibration_raw.get("queue_wait_p95_ceiling_ms", 250.0) or 250.0
            ),
            decode_tps_regression_floor_pct=float(
                calibration_raw.get("decode_tps_regression_floor_pct", 0.92) or 0.92
            ),
            continuous_max_active_requests=[
                int(item)
                for item in list(
                    calibration_raw.get("continuous_max_active_requests") or []
                )
            ],
            continuous_batch_wait_timeout_ms=[
                int(item)
                for item in list(
                    calibration_raw.get("continuous_batch_wait_timeout_ms") or [1, 2, 4]
                )
            ],
            continuous_state_page_rows=[
                int(item)
                for item in list(
                    calibration_raw.get("continuous_state_page_rows") or [64, 128, 256]
                )
            ],
            continuous_max_prefill_rows_per_iteration=[
                int(item)
                for item in list(
                    calibration_raw.get("continuous_max_prefill_rows_per_iteration")
                    or [512, 1024]
                )
            ],
            continuous_interleaved_streams=[
                bool(item)
                for item in list(
                    calibration_raw.get("continuous_interleaved_streams")
                    or [False, True]
                )
            ],
            probe_kernel_iterations=int(
                calibration_raw.get("probe_kernel_iterations", 0) or 0
            ),
            search_kernel_iterations=int(
                calibration_raw.get("search_kernel_iterations", 24) or 24
            ),
            deep_search_kernel_iterations=int(
                calibration_raw.get("deep_search_kernel_iterations", 64) or 64
            ),
        ),
        kernel_microbench=KernelMicrobenchSpec(
            warmups=int(kernel_raw.get("warmups", 10) or 10),
            iterations=int(kernel_raw.get("iterations", 200) or 200),
            synthetic_lengths=[
                int(item)
                for item in list(
                    kernel_raw.get("synthetic_lengths") or [32, 64, 128, 256]
                )
            ],
            target_kernels=[
                str(item)
                for item in list(kernel_raw.get("target_kernels") or [])
                if str(item).strip()
            ],
        ),
        quality_eval=QualityEvalSpec(
            perplexity_corpus=str(quality_raw.get("perplexity_corpus") or ""),
            confidence_corpus=str(quality_raw.get("confidence_corpus") or ""),
            rubric_corpus=str(quality_raw.get("rubric_corpus") or ""),
            accuracy_corpus=str(quality_raw.get("accuracy_corpus") or ""),
            max_samples_per_family=int(
                quality_raw.get(
                    "max_samples_per_family",
                    quality_raw.get("ablation_sample_limit", 0),
                )
                or 0
            ),
            adaptive_top_k=int(quality_raw.get("adaptive_top_k", 0) or 0),
            shadow_mode=bool(quality_raw.get("shadow_mode", True)),
        ),
        memory_replay=MemoryReplaySpec(
            db_path=str(memory_raw.get("db_path") or ""),
            campaign_id=str(memory_raw.get("campaign_id") or ""),
            storage_root=str(memory_raw.get("storage_root") or ""),
            cases_path=str(memory_raw.get("cases_path") or ""),
            case_limit=int(memory_raw.get("case_limit", 0) or 0),
            gate_thresholds={
                str(key): float(value)
                for key, value in dict(memory_raw.get("gate_thresholds") or {}).items()
            },
        ),
        ablations=_ablation_specs(list(payload.get("ablations") or [])),
        enabled_lanes=[
            str(item)
            for item in list(payload.get("enabled_lanes") or DEFAULT_ENABLED_LANES)
            if str(item).strip()
        ],
        host_requirements=dict(payload.get("host_requirements") or {}),
    )


def compile_suite_profile(spec: BenchmarkSuiteSpec) -> dict[str, Any]:
    return {
        "schema_version": str(spec.schema_version),
        "profile_name": str(spec.profile_name),
        "enabled_lanes": list(spec.enabled_lanes),
        "preflight_strictness": str(spec.preflight_strictness),
        "quality_policy": {
            "accuracy_corpus": str(spec.quality_eval.accuracy_corpus),
            "max_samples_per_family": int(spec.quality_eval.max_samples_per_family),
            "adaptive_top_k": int(spec.quality_eval.adaptive_top_k),
            "shadow_mode": bool(spec.quality_eval.shadow_mode),
        },
        "calibration_search": {
            "bootstrap_profile": str(spec.calibration_search.bootstrap_profile),
            "search_profile": str(spec.calibration_search.search_profile),
            "deep_search_profile": str(spec.calibration_search.deep_search_profile),
            "stage1_top_k": int(spec.calibration_search.stage1_top_k),
            "stage2_top_k": int(spec.calibration_search.stage2_top_k),
            "scheduler_top_k": int(spec.calibration_search.scheduler_top_k),
            "max_scheduler_candidates": int(
                spec.calibration_search.max_scheduler_candidates
            ),
            "fairness_floor": float(spec.calibration_search.fairness_floor),
            "queue_wait_p95_ceiling_ms": float(
                spec.calibration_search.queue_wait_p95_ceiling_ms
            ),
            "decode_tps_regression_floor_pct": float(
                spec.calibration_search.decode_tps_regression_floor_pct
            ),
            "continuous_max_active_requests": list(
                spec.calibration_search.continuous_max_active_requests
            ),
            "continuous_batch_wait_timeout_ms": list(
                spec.calibration_search.continuous_batch_wait_timeout_ms
            ),
            "continuous_state_page_rows": list(
                spec.calibration_search.continuous_state_page_rows
            ),
            "continuous_max_prefill_rows_per_iteration": list(
                spec.calibration_search.continuous_max_prefill_rows_per_iteration
            ),
            "continuous_interleaved_streams": list(
                spec.calibration_search.continuous_interleaved_streams
            ),
            "probe_kernel_iterations": int(
                spec.calibration_search.probe_kernel_iterations
            ),
            "search_kernel_iterations": int(
                spec.calibration_search.search_kernel_iterations
            ),
            "deep_search_kernel_iterations": int(
                spec.calibration_search.deep_search_kernel_iterations
            ),
        },
        "memory_replay": {
            "db_path": str(spec.memory_replay.db_path),
            "campaign_id": str(spec.memory_replay.campaign_id),
            "storage_root": str(spec.memory_replay.storage_root),
            "cases_path": str(spec.memory_replay.cases_path),
            "case_limit": int(spec.memory_replay.case_limit),
            "gate_thresholds": dict(spec.memory_replay.gate_thresholds),
        },
        "ablations": [
            {
                "ablation_id": item.ablation_id,
                "env": dict(item.env),
                "warmup_runs": item.warmup_runs,
                "measured_runs": item.measured_runs,
            }
            for item in spec.ablations
        ],
        "host_requirements": dict(spec.host_requirements),
    }
