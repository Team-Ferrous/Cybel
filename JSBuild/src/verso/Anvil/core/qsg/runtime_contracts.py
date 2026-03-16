from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from typing import Any


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _digest_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DeltaWatermark:
    delta_id: str = ""
    parent_delta_id: str = ""
    logical_clock: int = 0
    workspace_id: str = ""
    created_at: float = 0.0
    git_head: str = ""
    git_status: str = ""
    changed_paths: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "delta_id": str(self.delta_id),
            "parent_delta_id": str(self.parent_delta_id),
            "logical_clock": int(self.logical_clock),
            "workspace_id": str(self.workspace_id),
            "created_at": float(self.created_at),
            "git_head": str(self.git_head),
            "git_status": str(self.git_status),
            "changed_paths": list(self.changed_paths),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DeltaWatermark":
        data = dict(payload or {})
        changed_paths = [str(path) for path in list(data.get("changed_paths") or [])]
        return cls(
            delta_id=str(data.get("delta_id") or data.get("event_id") or ""),
            parent_delta_id=str(data.get("parent_delta_id") or ""),
            logical_clock=int(data.get("logical_clock") or 0),
            workspace_id=str(data.get("workspace_id") or ""),
            created_at=float(data.get("created_at") or 0.0),
            git_head=str(data.get("git_head") or ""),
            git_status=str(data.get("git_status") or ""),
            changed_paths=changed_paths,
        )

    def stable_digest(self) -> str:
        return _digest_payload(self.as_dict())


@dataclass(slots=True)
class RuntimeCapabilityVector:
    model: str = ""
    digest: str = ""
    architecture: str = ""
    native_isa_baseline: str = ""
    optional_isa_leaves: list[str] = field(default_factory=list)
    abi_tags: list[str] = field(default_factory=list)
    backend_module: str = ""
    backend_module_loaded: bool = False
    native_backend_abi_match: bool = False
    strict_path_stable: bool = False
    hot_path_numpy_detected: bool = False
    affinity_policy: str = ""
    l3_domain_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": str(self.model),
            "digest": str(self.digest),
            "architecture": str(self.architecture),
            "native_isa_baseline": str(self.native_isa_baseline),
            "optional_isa_leaves": list(self.optional_isa_leaves),
            "abi_tags": list(self.abi_tags),
            "backend_module": str(self.backend_module),
            "backend_module_loaded": bool(self.backend_module_loaded),
            "native_backend_abi_match": bool(self.native_backend_abi_match),
            "strict_path_stable": bool(self.strict_path_stable),
            "hot_path_numpy_detected": bool(self.hot_path_numpy_detected),
            "affinity_policy": str(self.affinity_policy),
            "l3_domain_count": int(self.l3_domain_count),
        }

    @classmethod
    def from_status(cls, status: dict[str, Any] | None) -> "RuntimeCapabilityVector":
        payload = dict(status or {})
        optional_isa_leaves: list[str] = []
        configured_leaves = payload.get("native_optional_isa_leaves")
        if isinstance(configured_leaves, (list, tuple)):
            optional_isa_leaves.extend(
                str(item).strip() for item in configured_leaves if str(item).strip()
            )
        else:
            csv_value = str(payload.get("native_optional_isa_leaves_csv") or "").strip()
            if csv_value:
                optional_isa_leaves.extend(
                    item.strip() for item in csv_value.split(",") if item.strip()
                )
        if bool(payload.get("avx2_enabled")):
            optional_isa_leaves.append("avx2")
        if bool(payload.get("avx512_enabled")):
            optional_isa_leaves.append("avx512")
        if bool(payload.get("amx_enabled")):
            optional_isa_leaves.append("amx")
        abi_tags = [
            f"split_abi:{int(payload.get('native_split_abi_version') or 0)}",
            f"backend_abi:{int(payload.get('backend_module_abi_version') or 0)}",
        ]
        return cls(
            model=str(payload.get("model") or payload.get("adapter_model") or ""),
            digest=str(payload.get("digest") or ""),
            architecture=str(payload.get("architecture") or ""),
            native_isa_baseline=str(payload.get("native_isa_baseline") or ""),
            optional_isa_leaves=list(dict.fromkeys(optional_isa_leaves)),
            abi_tags=abi_tags,
            backend_module=str(
                payload.get("backend_module") or payload.get("backend") or ""
            ),
            backend_module_loaded=bool(payload.get("backend_module_loaded")),
            native_backend_abi_match=bool(payload.get("native_backend_abi_match")),
            strict_path_stable=bool(payload.get("strict_path_stable")),
            hot_path_numpy_detected=bool(payload.get("hot_path_numpy_detected")),
            affinity_policy=str(payload.get("affinity_policy") or ""),
            l3_domain_count=int(payload.get("l3_domain_count") or 0),
        )

    def stable_digest(self) -> str:
        return _digest_payload(self.as_dict())


@dataclass(slots=True)
class ControllerDecisionRecord:
    controller: str
    selected_mode: str
    reason: str
    telemetry: dict[str, Any] = field(default_factory=dict)
    decided_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "controller": str(self.controller),
            "selected_mode": str(self.selected_mode),
            "reason": str(self.reason),
            "telemetry": dict(self.telemetry),
            "decided_at": float(self.decided_at),
        }


@dataclass(slots=True)
class BranchCoherenceMatrix:
    matrix: list[list[float]] = field(default_factory=list)
    source: str = "jacobi_neighbor_agreement"

    def as_dict(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.matrix],
            "source": str(self.source),
        }


@dataclass(slots=True)
class JacobiFrontierState:
    frontier_width: int = 0
    branch_survival_rate: float = 0.0
    verify_cost_ms: float = 0.0
    branch_entropy: float = 0.0
    branch_scores: list[list[float]] = field(default_factory=list)
    surviving_tokens: list[list[int]] = field(default_factory=list)
    coherence: BranchCoherenceMatrix = field(default_factory=BranchCoherenceMatrix)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frontier_width": int(self.frontier_width),
            "branch_survival_rate": float(self.branch_survival_rate),
            "verify_cost_ms": float(self.verify_cost_ms),
            "branch_entropy": float(self.branch_entropy),
            "branch_scores": [list(row) for row in self.branch_scores],
            "surviving_tokens": [list(row) for row in self.surviving_tokens],
            "coherence": self.coherence.as_dict(),
        }


@dataclass(slots=True)
class SpeculativeFrontierPolicy:
    mode: str = "adaptive"
    min_acceptance_ratio: float = 0.35
    max_frontier_width: int = 8
    prompt_lookup_window: int = 6
    verify_depth_cap: int = 8
    temperature_ceiling: float = 0.85
    structured_frontier_bias: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_lookup_limit(self, max_new_tokens: int) -> int:
        return max(
            0,
            min(
                int(self.max_frontier_width),
                int(self.prompt_lookup_window),
                int(max_new_tokens),
            ),
        )

    def decision(
        self,
        *,
        selected_mode: str,
        reason: str,
        prompt_tokens: int,
        max_new_tokens: int,
        temperature: float,
        accepted_parallel_tokens: int = 0,
        rejected_parallel_tokens: int = 0,
        draft_frontier_width: int = 0,
        verify_depth: int = 0,
    ) -> ControllerDecisionRecord:
        return ControllerDecisionRecord(
            controller="frontier",
            selected_mode=str(selected_mode),
            reason=str(reason),
            telemetry={
                "prompt_tokens": int(prompt_tokens),
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "accepted_parallel_tokens": int(accepted_parallel_tokens),
                "rejected_parallel_tokens": int(rejected_parallel_tokens),
                "draft_frontier_width": int(draft_frontier_width),
                "verify_depth": int(verify_depth),
                "policy": self.as_dict(),
            },
        )

    def decide(self, status: dict[str, Any] | None) -> ControllerDecisionRecord:
        payload = dict(status or {})
        selected_mode = str(payload.get("generation_mode") or "ar_verify")
        prompt_category = str(payload.get("prompt_category") or "")
        observed_frontier = int(
            payload.get("jacobi_frontier_width")
            or payload.get("draft_frontier_width")
            or 0
        )
        proposed = int(payload.get("proposed_parallel_tokens") or 0)
        accepted = int(payload.get("accepted_parallel_tokens") or 0)
        acceptance_ratio = float(accepted) / float(proposed) if proposed > 0 else 0.0
        branch_survival_rate = float(payload.get("jacobi_branch_survival_rate") or 0.0)
        branch_entropy = float(payload.get("jacobi_branch_entropy") or 0.0)
        target_frontier = min(
            max(0, int(self.max_frontier_width)),
            max(
                int(observed_frontier),
                1 if selected_mode not in {"ar_baseline", "ar_verify"} else 0,
            ),
        )
        reason = "steady_state_native_generation"
        if str(self.mode).strip().lower() != "adaptive":
            reason = f"frontier_mode:{self.mode}"
        elif prompt_category == "structured" and selected_mode in {
            "ar_baseline",
            "ar_verify",
        }:
            selected_mode = "prompt_lookup"
            target_frontier = max(
                target_frontier, 1 + int(self.structured_frontier_bias)
            )
            reason = "structured_prefix_reuse"
        elif branch_survival_rate > 0.0 and branch_survival_rate < float(
            self.min_acceptance_ratio
        ):
            selected_mode = "ar_verify"
            target_frontier = 0
            reason = "jacobi_frontier_guard_fallback"
        elif acceptance_ratio >= float(self.min_acceptance_ratio):
            if selected_mode in {"ar_baseline", "ar_verify"}:
                selected_mode = "parallel_hybrid"
            target_frontier = max(target_frontier, 2)
            reason = "acceptance_supports_speculation"
        elif selected_mode not in {"ar_baseline", "ar_verify"}:
            selected_mode = "ar_verify"
            target_frontier = 0
            reason = "acceptance_guard_fallback"
        if branch_entropy > 1.5 and target_frontier > 2:
            target_frontier = 2
            reason = "jacobi_entropy_guard"
        return ControllerDecisionRecord(
            controller="frontier",
            selected_mode=selected_mode,
            reason=reason,
            telemetry={
                "prompt_category": prompt_category,
                "observed_frontier_width": int(observed_frontier),
                "target_frontier_width": int(target_frontier),
                "draft_acceptance_ratio": float(acceptance_ratio),
                "jacobi_branch_survival_rate": float(branch_survival_rate),
                "jacobi_branch_entropy": float(branch_entropy),
                "verify_depth": int(payload.get("verify_depth") or 0),
                "policy": self.as_dict(),
            },
        )


@dataclass(slots=True)
class DriftController:
    mode: str = "adaptive"
    overhead_target_pct: float = 15.0
    overhead_max_pct: float = 20.0
    control_interval_tokens: int = 64
    recovery_interval_tokens: int = 256
    hysteresis: float = 0.05
    preserve_head_tokens: int = 256
    preserve_recent_tokens: int = 8192
    observation_window_tokens: int = 1024
    min_active_tokens: int = 16384
    damping_strength: float = 1.2
    min_recent_tokens: int = 1024

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def build_config(
        self,
        *,
        mode: int,
        prompt_category: str = "",
        acceptance_ratio: float = 0.0,
        current_overhead_pct: float = 0.0,
    ) -> dict[str, int | float]:
        retention = self.retention_policy(
            prompt_category=prompt_category,
            acceptance_ratio=acceptance_ratio,
        )
        preserve_recent = int(retention["preserve_recent_tokens"])
        if float(current_overhead_pct) > float(self.overhead_target_pct):
            preserve_recent = max(
                int(self.min_recent_tokens),
                preserve_recent - max(128, int(self.observation_window_tokens // 2)),
            )
        return {
            "mode": max(0, min(2, int(mode))),
            "preserve_head_tokens": int(retention["preserve_head_tokens"]),
            "preserve_recent_tokens": int(preserve_recent),
            "min_active_tokens": int(self.min_active_tokens),
            "damping_strength": float(self.damping_strength),
            "hysteresis": float(self.hysteresis),
            "observation_window_tokens": int(self.observation_window_tokens),
        }

    def retention_policy(
        self,
        *,
        prompt_category: str = "",
        acceptance_ratio: float = 0.0,
    ) -> dict[str, int]:
        preserve_head = int(self.preserve_head_tokens)
        preserve_recent = int(self.preserve_recent_tokens)
        category = str(prompt_category or "").strip().lower()
        if category == "structured":
            preserve_head = max(preserve_head, 512)
            preserve_recent = max(
                int(preserve_recent * 0.5), int(self.min_recent_tokens)
            )
        elif acceptance_ratio >= 0.75:
            preserve_recent = max(
                int(preserve_recent * 0.75), int(self.min_recent_tokens)
            )
        elif acceptance_ratio <= 0.25:
            preserve_recent = max(preserve_recent, int(self.preserve_recent_tokens))
        return {
            "preserve_head_tokens": int(max(0, preserve_head)),
            "preserve_recent_tokens": int(max(self.min_recent_tokens, preserve_recent)),
        }

    def transition(
        self,
        *,
        current_mode_native: int | None = None,
        current_mode: int | None = None,
        overhead_percent: float | None = None,
        current_overhead_pct: float | None = None,
        recovery_steps: int,
        control_interval_tokens: int | None = None,
        recovery_interval_tokens: int | None = None,
    ) -> tuple[int, int, ControllerDecisionRecord]:
        target_mode_native = int(
            current_mode_native
            if current_mode_native is not None
            else current_mode or 0
        )
        overhead_value = float(
            overhead_percent
            if overhead_percent is not None
            else current_overhead_pct or 0.0
        )
        control_window = int(
            control_interval_tokens
            if control_interval_tokens is not None
            else self.control_interval_tokens
        )
        recovery_window = int(
            recovery_interval_tokens
            if recovery_interval_tokens is not None
            else self.recovery_interval_tokens
        )
        next_recovery_steps = int(recovery_steps)
        reason = "steady_state"
        if int(target_mode_native) >= 2:
            if overhead_value > float(self.overhead_max_pct):
                target_mode_native = 1
                next_recovery_steps = 0
                reason = "overhead_above_max"
        elif overhead_value < max(
            0.0, float(self.overhead_target_pct) - float(self.hysteresis)
        ):
            next_recovery_steps += int(control_window)
            reason = "recovery_window_accumulating"
            if next_recovery_steps >= int(recovery_window):
                target_mode_native = 2
                next_recovery_steps = 0
                reason = "recovery_complete"
        else:
            next_recovery_steps = 0
            reason = "overhead_above_recovery_target"
        return (
            int(target_mode_native),
            int(next_recovery_steps),
            ControllerDecisionRecord(
                controller="drift",
                selected_mode=str(target_mode_native),
                reason=reason,
                telemetry={
                    "current_mode_native": int(
                        current_mode_native
                        if current_mode_native is not None
                        else current_mode or 0
                    ),
                    "overhead_percent": float(overhead_value),
                    "recovery_steps": int(recovery_steps),
                    "policy": self.as_dict(),
                },
            ),
        )

    def decide(
        self,
        status: dict[str, Any] | None,
        *,
        prompt_category: str = "",
        acceptance_ratio: float = 0.0,
    ) -> ControllerDecisionRecord:
        payload = dict(status or {})
        enabled = bool(payload.get("context_stabilizer_enabled"))
        overhead_percent = float(
            payload.get("drift_overhead_percent")
            or payload.get("qsg_drift_overhead_percent")
            or 0.0
        )
        drift_mode = str(payload.get("context_stabilizer_mode") or "disabled")
        reason = (
            "timecrystal_stabilizer_enabled"
            if enabled
            else "timecrystal_stabilizer_disabled"
        )
        if enabled and overhead_percent > float(self.overhead_max_pct):
            drift_mode = "conservative"
            reason = "overhead_guard"
        retention = self.retention_policy(
            prompt_category=prompt_category,
            acceptance_ratio=acceptance_ratio,
        )
        return ControllerDecisionRecord(
            controller="drift",
            selected_mode=drift_mode,
            reason=reason,
            telemetry={
                "drift_overhead_percent": float(overhead_percent),
                "drift_mean": float(payload.get("drift_mean") or 0.0),
                "stabilizer_calls": int(payload.get("stabilizer_calls") or 0),
                **retention,
                "policy": self.as_dict(),
            },
        )


@dataclass(slots=True)
class PerformanceEnvelope:
    capability_digest: str
    delta_watermark: dict[str, Any] = field(default_factory=dict)
    ttft_ms: float = 0.0
    tpot_ms_p95: float = 0.0
    end_to_end_throughput_tps: float = 0.0
    effective_prefill_throughput_tps: float = 0.0
    prompt_cache_hit_ratio: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    drift_overhead_percent: float = 0.0
    draft_acceptance_ratio: float = 0.0
    scheduler_queue_wait_ms: float = 0.0
    scheduler_iteration_ms: float = 0.0
    python_hot_path_calls: int = 0
    numpy_hot_path_calls: int = 0
    measurement_valid: bool = True
    collected_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_runtime_status(
        cls,
        status: dict[str, Any] | None,
        *,
        capability_digest: str,
        delta_watermark: dict[str, Any] | None = None,
    ) -> "PerformanceEnvelope":
        payload = dict(status or {})
        proposed = int(payload.get("proposed_parallel_tokens") or 0)
        accepted = int(payload.get("accepted_parallel_tokens") or 0)
        acceptance_ratio = 0.0
        if proposed > 0:
            acceptance_ratio = float(accepted) / float(proposed)
        return cls(
            capability_digest=str(capability_digest),
            delta_watermark=dict(delta_watermark or {}),
            ttft_ms=float(payload.get("ttft_ms") or 0.0),
            tpot_ms_p95=float(
                payload.get("per_token_latency_p95_ms")
                or payload.get("qsg_tpot_ms_p95")
                or 0.0
            ),
            end_to_end_throughput_tps=float(
                payload.get("end_to_end_throughput_tps") or 0.0
            ),
            effective_prefill_throughput_tps=float(
                payload.get("effective_prefill_throughput_tps") or 0.0
            ),
            prompt_cache_hit_ratio=float(payload.get("prompt_cache_hit_ratio") or 0.0),
            prefix_cache_hit_rate=float(payload.get("prefix_cache_hit_rate") or 0.0),
            drift_overhead_percent=float(
                payload.get("drift_overhead_percent")
                or payload.get("qsg_drift_overhead_percent")
                or 0.0
            ),
            draft_acceptance_ratio=acceptance_ratio,
            scheduler_queue_wait_ms=float(
                payload.get("scheduler_queue_wait_ms")
                or payload.get("qsg_queue_wait_ms_p95")
                or 0.0
            ),
            scheduler_iteration_ms=float(
                payload.get("scheduler_iteration_ms")
                or payload.get("qsg_scheduler_iteration_ms_p95")
                or 0.0
            ),
            python_hot_path_calls=int(
                payload.get("python_hot_path_calls")
                or payload.get("qsg_python_hot_path_calls")
                or 0
            ),
            numpy_hot_path_calls=int(
                payload.get("numpy_hot_path_calls")
                or payload.get("qsg_numpy_hot_path_calls")
                or 0
            ),
            measurement_valid=bool(payload.get("measurement_valid", True)),
        )


@dataclass(slots=True)
class MemoryTierPolicy:
    mode: str = "adaptive"
    prompt_cache_hit_threshold: float = 0.40
    latent_replay_threshold: float = 0.60
    repo_delta_window: int = 8

    def decide(
        self,
        status: dict[str, Any] | None,
        *,
        latent_available: bool = False,
        delta_watermark: dict[str, Any] | None = None,
    ) -> ControllerDecisionRecord:
        payload = dict(status or {})
        watermark = DeltaWatermark.from_dict(
            delta_watermark or payload.get("delta_watermark")
        )
        prompt_cache_hit_ratio = float(payload.get("prompt_cache_hit_ratio") or 0.0)
        prefix_cache_hit_rate = float(payload.get("prefix_cache_hit_rate") or 0.0)
        changed_paths = list(watermark.changed_paths)
        selected_mode = "recent_kv"
        reason = "default_recent_kv"
        if latent_available and not changed_paths:
            selected_mode = "latent_replay"
            reason = "compatible_latent_capsule_available"
        elif max(prompt_cache_hit_ratio, prefix_cache_hit_rate) >= float(
            self.prompt_cache_hit_threshold
        ):
            selected_mode = "prompt_cache"
            reason = "prompt_prefix_cache_hot"
        elif changed_paths and len(changed_paths) <= int(self.repo_delta_window):
            selected_mode = "repo_delta_memory"
            reason = "recent_repo_delta"
        elif latent_available and len(changed_paths) <= int(self.repo_delta_window):
            selected_mode = "latent_replay"
            reason = "latent_replay_with_small_delta"
        return ControllerDecisionRecord(
            controller="memory_tier",
            selected_mode=selected_mode,
            reason=reason,
            telemetry={
                "prompt_cache_hit_ratio": float(prompt_cache_hit_ratio),
                "prefix_cache_hit_rate": float(prefix_cache_hit_rate),
                "latent_available": bool(latent_available),
                "changed_path_count": len(changed_paths),
                "delta_id": str(watermark.delta_id),
            },
        )


@dataclass(slots=True)
class RepoDeltaMemoryRecord:
    delta_id: str = ""
    workspace_id: str = ""
    logical_clock: int = 0
    changed_paths: list[str] = field(default_factory=list)
    capability_digest: str = ""
    summary_text: str = ""
    semantic_impact_hint: str = ""
    created_at: float = field(default_factory=time.time)

    @classmethod
    def from_watermark(
        cls,
        watermark: dict[str, Any] | None,
        *,
        capability_digest: str = "",
    ) -> "RepoDeltaMemoryRecord":
        delta = DeltaWatermark.from_dict(watermark)
        changed_paths = list(delta.changed_paths)
        summary = "No repo delta available."
        if changed_paths:
            preview = ", ".join(changed_paths[:3])
            if len(changed_paths) > 3:
                preview = f"{preview}, ..."
            summary = (
                f"Delta {delta.delta_id or 'unknown'} touched "
                f"{len(changed_paths)} path(s): {preview}"
            )
        return cls(
            delta_id=str(delta.delta_id),
            workspace_id=str(delta.workspace_id),
            logical_clock=int(delta.logical_clock),
            changed_paths=changed_paths,
            capability_digest=str(capability_digest),
            summary_text=summary,
            semantic_impact_hint="repo_delta_memory",
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformanceTwinPrediction:
    capability_digest: str
    predicted_regime: str
    risk_level: str
    risk_score: float
    issues: list[str] = field(default_factory=list)
    suggested_actions: list[str] = field(default_factory=list)
    observed_controller_modes: dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MissionReplayDescriptor:
    hypothesis_id: str = ""
    request_id: str = ""
    memory_id: str = ""
    latent_package_id: str = ""
    capsule_id: str = ""
    capability_digest: str = ""
    delta_watermark: dict[str, Any] = field(default_factory=dict)
    supporting_memory_ids: list[str] = field(default_factory=list)
    repo_delta_memory_id: str = ""
    memory_tier_decision: dict[str, Any] = field(default_factory=dict)
    replay_tape_path: str = ""
    replay_run_id: str = ""
    mode: str = ""
    restored: bool = False
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PerformanceTwinModel:
    ttft_medium_ms: float = 150.0
    ttft_high_ms: float = 400.0
    tpot_medium_ms: float = 40.0
    tpot_high_ms: float = 120.0
    drift_medium_pct: float = 8.0
    drift_high_pct: float = 15.0
    queue_wait_medium_ms: float = 15.0
    queue_wait_high_ms: float = 40.0

    def predict(
        self,
        *,
        envelope: PerformanceEnvelope,
        capability_vector: RuntimeCapabilityVector | None = None,
        controller_state: dict[str, Any] | None = None,
    ) -> PerformanceTwinPrediction:
        issues: list[str] = []
        actions: list[str] = []
        score = 0.0
        regime = "stable"
        controllers = dict(controller_state or {})
        frontier_mode = str(
            (controllers.get("frontier") or {}).get("selected_mode") or ""
        )
        drift_mode = str((controllers.get("drift") or {}).get("selected_mode") or "")

        if envelope.python_hot_path_calls or envelope.numpy_hot_path_calls:
            score += 2.5
            regime = "python_regressed"
            issues.append("python_or_numpy_hot_path_detected")
            actions.append("restore native-only sampling and decode path")
        if envelope.ttft_ms >= self.ttft_high_ms:
            score += 2.0
            regime = "prefill_bound"
            issues.append("ttft_high")
            actions.append("reduce prefill contention or batch width")
        elif envelope.ttft_ms >= self.ttft_medium_ms:
            score += 1.0
            issues.append("ttft_elevated")
        if envelope.tpot_ms_p95 >= self.tpot_high_ms:
            score += 2.0
            regime = "decode_bound"
            issues.append("tpot_high")
            actions.append("tighten decode affinity or fused sampler path")
        elif envelope.tpot_ms_p95 >= self.tpot_medium_ms:
            score += 1.0
            issues.append("tpot_elevated")
        if envelope.scheduler_queue_wait_ms >= self.queue_wait_high_ms:
            score += 2.0
            regime = "queue_bound"
            issues.append("queue_wait_high")
            actions.append("lower queue depth or rebalance scheduler threads")
        elif envelope.scheduler_queue_wait_ms >= self.queue_wait_medium_ms:
            score += 1.0
            issues.append("queue_wait_elevated")
        if envelope.drift_overhead_percent >= self.drift_high_pct:
            score += 2.0
            regime = "drift_bound"
            issues.append("drift_overhead_high")
            actions.append("downgrade drift aggressiveness or shrink retention window")
        elif envelope.drift_overhead_percent >= self.drift_medium_pct:
            score += 1.0
            issues.append("drift_overhead_elevated")
        if frontier_mode and frontier_mode != "ar_verify":
            if envelope.draft_acceptance_ratio <= 0.15:
                score += 1.5
                issues.append("frontier_acceptance_low")
                actions.append(
                    "narrow speculative frontier or increase verifier strictness"
                )
        if capability_vector is not None and not capability_vector.strict_path_stable:
            score += 1.0
            issues.append("strict_native_path_unstable")
            actions.append("inspect backend ABI and hot-path proof")
        if capability_vector is not None and capability_vector.hot_path_numpy_detected:
            score += 1.5
            issues.append("numpy_detected_in_strict_path")
            actions.append("remove numpy from strict native path")

        if score >= 5.0:
            risk_level = "high"
        elif score >= 2.0:
            risk_level = "medium"
        else:
            risk_level = "low"
        if not actions:
            actions.append("keep current frontier and drift controller settings")
        return PerformanceTwinPrediction(
            capability_digest=str(envelope.capability_digest),
            predicted_regime=regime,
            risk_level=risk_level,
            risk_score=float(score),
            issues=issues,
            suggested_actions=actions,
            observed_controller_modes={
                "frontier": frontier_mode,
                "drift": drift_mode,
            },
        )


@dataclass(slots=True)
class ExecutionCapsule:
    capsule_id: str
    request_id: str
    version: int = 2
    model_family: str = ""
    capability_digest: str = ""
    delta_watermark: dict[str, Any] = field(default_factory=dict)
    generated_tokens: int = 0
    phase_state: float = 0.0
    hidden_dim: int = 0
    latent_packet_abi_version: int = 2
    segment_count: int = 0
    segment_kinds: list[str] = field(default_factory=list)
    segment_index: list[dict[str, Any]] = field(default_factory=list)
    controller_decisions: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "segment_kinds": list(self.segment_kinds),
            "segment_index": list(self.segment_index),
        }


@dataclass(slots=True)
class TypedLatentSegment:
    segment_id: str
    segment_kind: str
    row_start: int
    row_count: int
    hidden_dim: int
    tensor_format: str = "float32"
    codec: str = "float32"
    importance: float = 0.5
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": str(self.segment_id),
            "segment_kind": str(self.segment_kind),
            "row_start": int(self.row_start),
            "row_count": int(self.row_count),
            "hidden_dim": int(self.hidden_dim),
            "tensor_format": str(self.tensor_format),
            "codec": str(self.codec),
            "importance": float(self.importance),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TypedLatentSegment":
        data = dict(payload or {})
        return cls(
            segment_id=str(data.get("segment_id") or ""),
            segment_kind=str(data.get("segment_kind") or "branch_state"),
            row_start=int(data.get("row_start") or 0),
            row_count=int(data.get("row_count") or 0),
            hidden_dim=int(data.get("hidden_dim") or 0),
            tensor_format=str(data.get("tensor_format") or "float32"),
            codec=str(data.get("codec") or data.get("tensor_format") or "float32"),
            importance=float(data.get("importance") or 0.5),
            provenance=dict(data.get("provenance") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class CapsuleSegmentIndex:
    segments: list[TypedLatentSegment] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        segments = [segment.as_dict() for segment in self.segments]
        return {
            "segments": segments,
            "segment_count": len(segments),
            "segment_kinds": [segment["segment_kind"] for segment in segments],
            "total_rows": sum(int(segment["row_count"]) for segment in segments),
            "total_bytes_estimate": sum(
                int(segment["row_count"]) * int(segment["hidden_dim"]) * 4
                for segment in segments
            ),
        }


@dataclass(slots=True)
class LatentPacketABI:
    abi_version: int = 2
    tensor: list[list[float]] = field(default_factory=list)
    tensor_format: str = "float32"
    tensor_codec: str = "float32"
    hidden_dim: int = 0
    generated_tokens: int = 0
    phase_state: float = 0.0
    capability_digest: str = ""
    delta_watermark: dict[str, Any] = field(default_factory=dict)
    execution_capsule_id: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)
    segment_count: int = 0
    compatibility_score: float = 1.0
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "segments": list(self.segments),
            "segment_count": int(self.segment_count or len(self.segments)),
        }


def evaluate_qsg_runtime_invariants(
    *,
    runtime_status: dict[str, Any] | None = None,
    latent_packet: dict[str, Any] | None = None,
    execution_capsule: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    runtime = dict(runtime_status or {})
    packet = dict(latent_packet or {})
    capsule = dict(execution_capsule or {})

    def _add(code: str, message: str, *, severity: str = "error") -> None:
        violations.append({"code": code, "message": message, "severity": severity})

    if capsule:
        if int(capsule.get("version") or 0) < 2:
            _add(
                "execution_capsule_version",
                "Execution capsule ABI version must be >= 2.",
            )
        if not str(capsule.get("capability_digest") or "").strip():
            _add(
                "execution_capsule_capability_digest",
                "Execution capsule must include a capability digest.",
            )
        if not isinstance(capsule.get("delta_watermark"), dict):
            _add(
                "execution_capsule_delta_watermark",
                "Execution capsule delta watermark must be a mapping.",
            )
        segment_count = int(capsule.get("segment_count") or 0)
        segment_index = list(capsule.get("segment_index") or [])
        if segment_count <= 0 and segment_index:
            segment_count = len(segment_index)
        if segment_count and not segment_index:
            _add(
                "execution_capsule_segment_index",
                "Execution capsule with segments must include a segment index.",
            )

    if packet:
        if int(packet.get("abi_version") or 0) < 2:
            _add("latent_packet_abi_version", "Latent packet ABI version must be >= 2.")
        if not str(packet.get("execution_capsule_id") or "").strip():
            _add(
                "latent_packet_execution_capsule_id",
                "Latent packet must reference an execution capsule id.",
            )
        if not str(packet.get("capability_digest") or "").strip():
            _add(
                "latent_packet_capability_digest",
                "Latent packet must include a capability digest.",
            )
        if not isinstance(packet.get("delta_watermark"), dict):
            _add(
                "latent_packet_delta_watermark",
                "Latent packet delta watermark must be a mapping.",
            )
        segments = list(packet.get("segments") or [])
        if int(packet.get("abi_version") or 0) >= 3 and not segments:
            _add(
                "latent_packet_segments",
                "Latent packet ABI v3 must include typed segments.",
            )
        for idx, segment in enumerate(segments):
            typed = TypedLatentSegment.from_dict(segment)
            if typed.row_count <= 0:
                _add(
                    "latent_packet_segment_rows",
                    f"Latent packet segment {idx} must have a positive row_count.",
                )
            if typed.hidden_dim <= 0:
                _add(
                    "latent_packet_segment_hidden_dim",
                    f"Latent packet segment {idx} must declare hidden_dim.",
                )

    if packet and capsule:
        packet_capsule_id = str(packet.get("execution_capsule_id") or "")
        capsule_id = str(capsule.get("capsule_id") or "")
        if packet_capsule_id and capsule_id and packet_capsule_id != capsule_id:
            _add(
                "packet_capsule_mismatch",
                "Latent packet execution capsule id must match the execution capsule id.",
            )
        packet_digest = str(packet.get("capability_digest") or "")
        capsule_digest = str(capsule.get("capability_digest") or "")
        if packet_digest and capsule_digest and packet_digest != capsule_digest:
            _add(
                "capability_digest_mismatch",
                "Latent packet and execution capsule capability digests must match.",
            )
        packet_delta = dict(packet.get("delta_watermark") or {})
        capsule_delta = dict(capsule.get("delta_watermark") or {})
        packet_delta_id = str(packet_delta.get("delta_id") or "")
        capsule_delta_id = str(capsule_delta.get("delta_id") or "")
        if packet_delta_id and capsule_delta_id and packet_delta_id != capsule_delta_id:
            _add(
                "delta_watermark_mismatch",
                "Latent packet and execution capsule delta ids must match.",
            )
        packet_segments = list(packet.get("segments") or [])
        capsule_segments = list(capsule.get("segment_index") or [])
        if (
            packet_segments
            and capsule_segments
            and len(packet_segments) != len(capsule_segments)
        ):
            _add(
                "segment_index_mismatch",
                "Latent packet segments must match the execution capsule segment index.",
            )

    if runtime:
        authority = bool(
            runtime.get("qsg_native_runtime_authority")
            if "qsg_native_runtime_authority" in runtime
            else runtime.get("native_runtime_authority", False)
        )
        capability_digest = str(
            runtime.get("qsg_capability_digest")
            or runtime.get("capability_digest")
            or ""
        ).strip()
        delta_watermark = runtime.get(
            "qsg_delta_watermark", runtime.get("delta_watermark")
        )
        if authority and not capability_digest:
            _add(
                "runtime_capability_digest",
                "Runtime authority requires a non-empty capability digest.",
            )
        if delta_watermark is not None and not isinstance(delta_watermark, dict):
            _add(
                "runtime_delta_watermark",
                "Runtime delta watermark must be a mapping when present.",
            )

    return violations
