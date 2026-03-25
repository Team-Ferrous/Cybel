"""Governance policies for ALMF."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any, Dict


class RetentionPolicy:
    """Retention helpers for memory objects and latent packages."""

    DEFAULTS_BY_KIND = {
        "conversation_turn": "durable",
        "task_memory": "durable",
        "research_chunk": "durable",
        "research_claim": "durable",
        "research_document": "durable",
        "research_source": "durable",
        "experiment_result": "session",
        "experiment_stdout": "session",
        "latent_branch": "session",
        "latent_package": "ephemeral",
        "repo_delta_memory": "session",
        "telemetry_snapshot": "durable",
        "roadmap_evidence": "archival",
    }
    RETENTION_WINDOWS = {
        "ephemeral": 60.0 * 60.0,
        "session": 60.0 * 60.0 * 24.0,
        "durable": None,
        "archival": None,
    }

    @classmethod
    def expires_at(
        cls,
        retention_class: str,
        *,
        created_at: float | None = None,
    ) -> float | None:
        window = cls.RETENTION_WINDOWS.get(retention_class)
        if window is None:
            return None
        return float(created_at if created_at is not None else time.time()) + float(
            window
        )

    @classmethod
    def default_for_kind(cls, memory_kind: str) -> str:
        return str(cls.DEFAULTS_BY_KIND.get(str(memory_kind or ""), "durable"))

    @staticmethod
    def is_expired(expires_at: float | None, *, now: float | None = None) -> bool:
        if expires_at is None:
            return False
        return float(expires_at) <= float(now if now is not None else time.time())

    @classmethod
    def validate_memory(
        cls,
        memory_kind: str,
        *,
        retention_class: str,
        sensitivity_class: str,
        provenance_present: bool,
    ) -> Dict[str, Any]:
        issues = []
        if retention_class not in cls.RETENTION_WINDOWS:
            issues.append(f"unknown retention_class `{retention_class}`")
        if sensitivity_class not in {"public", "internal", "restricted", "secret"}:
            issues.append(f"unknown sensitivity_class `{sensitivity_class}`")
        if not provenance_present:
            issues.append("missing provenance")
        return {
            "ok": not issues,
            "expected_retention_class": cls.default_for_kind(memory_kind),
            "issues": issues,
        }


class LatentCompatibilityPolicy:
    """Compatibility gate for latent replay."""

    REQUIRED_KEYS = (
        "model_family",
        "hidden_dim",
        "tokenizer_hash",
        "prompt_protocol_hash",
        "qsg_runtime_version",
        "latent_packet_abi_version",
    )

    @classmethod
    def evaluate(
        cls,
        package_row: Dict[str, Any],
        *,
        model_family: str,
        hidden_dim: int,
        tokenizer_hash: str,
        prompt_protocol_hash: str,
        qsg_runtime_version: str,
        quantization_profile: str = "",
        capability_digest: str = "",
        delta_watermark: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        mismatches = []
        compatibility = dict(package_row.get("compatibility_json") or {})
        expected = {
            "model_family": model_family,
            "hidden_dim": int(hidden_dim),
            "tokenizer_hash": tokenizer_hash,
            "prompt_protocol_hash": prompt_protocol_hash,
            "qsg_runtime_version": qsg_runtime_version,
            "latent_packet_abi_version": int(
                compatibility.get("latent_packet_abi_version") or 2
            ),
        }
        if quantization_profile:
            expected["quantization_profile"] = quantization_profile
        if capability_digest:
            expected["capability_digest"] = capability_digest
        if delta_watermark:
            expected["delta_watermark"] = dict(delta_watermark)
        for key, value in expected.items():
            current = package_row.get(key)
            if current is None and key in compatibility:
                current = compatibility.get(key)
            if current != value:
                mismatches.append({"field": key, "expected": value, "actual": current})
        return {
            "compatible": not mismatches,
            "mode": "exact" if not mismatches else "degraded",
            "mismatches": mismatches,
        }


@dataclass(slots=True)
class MemoryTierDecision:
    purpose: str
    selected_tier: str
    reason: str
    fallback_tier: str = ""
    replay_allowed: bool = False
    telemetry: Dict[str, Any] = field(default_factory=dict)
    candidate_tiers: list[str] = field(default_factory=list)
    decided_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MemoryTierPolicy:
    prompt_cache_hit_floor: float = 0.35
    prefix_cache_hit_floor: float = 0.20
    queue_wait_budget_ms: float = 20.0
    drift_overhead_ceiling_pct: float = 10.0
    repo_delta_path_soft_limit: int = 16

    def choose(
        self,
        *,
        purpose: str,
        runtime_status: Dict[str, Any] | None = None,
        latent_package: Dict[str, Any] | None = None,
        compatibility: Dict[str, Any] | None = None,
        repo_delta_memory: Dict[str, Any] | None = None,
    ) -> MemoryTierDecision:
        runtime = dict(runtime_status or {})
        package = dict(latent_package or {})
        compatibility_payload = dict(compatibility or {})
        repo_delta = dict(repo_delta_memory or {})
        candidates = [
            "recent_kv",
            "latent_replay",
            "semantic_summary",
            "repo_delta_memory",
        ]
        prompt_cache_hit = float(
            runtime.get("prompt_cache_hit_ratio")
            or runtime.get("qsg_prompt_cache_hit_ratio")
            or 0.0
        )
        prefix_cache_hit = float(
            runtime.get("prefix_cache_hit_rate")
            or runtime.get("qsg_prefix_cache_hit_rate")
            or 0.0
        )
        queue_wait_ms = float(
            runtime.get("qsg_queue_wait_ms_p95")
            or runtime.get("scheduler_queue_wait_ms")
            or 0.0
        )
        drift_overhead_percent = float(
            runtime.get("qsg_drift_overhead_percent")
            or runtime.get("drift_overhead_percent")
            or 0.0
        )
        path_count = int(
            repo_delta.get("path_count")
            or len(list(repo_delta.get("changed_paths") or []))
        )
        compatible = compatibility_payload.get("compatible", True)

        if purpose == "mission_replay":
            if not package:
                return MemoryTierDecision(
                    purpose=purpose,
                    selected_tier="semantic_summary",
                    reason="missing_latent_package",
                    telemetry={
                        "path_count": path_count,
                        "prompt_cache_hit_ratio": prompt_cache_hit,
                    },
                    candidate_tiers=candidates,
                )
            if compatible:
                return MemoryTierDecision(
                    purpose=purpose,
                    selected_tier="latent_replay",
                    reason="replay_safe_latent_package_available",
                    fallback_tier="repo_delta_memory" if path_count else "semantic_summary",
                    replay_allowed=True,
                    telemetry={
                        "path_count": path_count,
                        "prompt_cache_hit_ratio": prompt_cache_hit,
                        "prefix_cache_hit_rate": prefix_cache_hit,
                        "queue_wait_ms_p95": queue_wait_ms,
                        "drift_overhead_percent": drift_overhead_percent,
                    },
                    candidate_tiers=candidates,
                )
            selected = "repo_delta_memory" if path_count else "semantic_summary"
            reason = (
                "delta_scoped_recovery_required"
                if path_count
                else "latent_package_incompatible"
            )
            return MemoryTierDecision(
                purpose=purpose,
                selected_tier=selected,
                reason=reason,
                telemetry={
                    "path_count": path_count,
                    "compatibility_mismatches": list(
                        compatibility_payload.get("mismatches") or []
                    ),
                },
                candidate_tiers=candidates,
            )

        if (
            prompt_cache_hit >= self.prompt_cache_hit_floor
            or prefix_cache_hit >= self.prefix_cache_hit_floor
        ):
            return MemoryTierDecision(
                purpose=purpose,
                selected_tier="recent_kv",
                reason="cache_hit_ratio_above_floor",
                fallback_tier="latent_replay" if package else "semantic_summary",
                replay_allowed=bool(package and compatible),
                telemetry={
                    "prompt_cache_hit_ratio": prompt_cache_hit,
                    "prefix_cache_hit_rate": prefix_cache_hit,
                },
                candidate_tiers=candidates,
            )
        if package and compatible and (
            queue_wait_ms >= self.queue_wait_budget_ms
            or drift_overhead_percent >= self.drift_overhead_ceiling_pct
        ):
            return MemoryTierDecision(
                purpose=purpose,
                selected_tier="latent_replay",
                reason="runtime_pressure_favors_latent_replay",
                fallback_tier="recent_kv",
                replay_allowed=True,
                telemetry={
                    "queue_wait_ms_p95": queue_wait_ms,
                    "drift_overhead_percent": drift_overhead_percent,
                    "path_count": path_count,
                },
                candidate_tiers=candidates,
            )
        if path_count and path_count <= self.repo_delta_path_soft_limit:
            return MemoryTierDecision(
                purpose=purpose,
                selected_tier="repo_delta_memory",
                reason="delta_scoped_context_available",
                fallback_tier="semantic_summary",
                telemetry={"path_count": path_count},
                candidate_tiers=candidates,
            )
        return MemoryTierDecision(
            purpose=purpose,
            selected_tier="semantic_summary",
            reason="summary_memory_is_safest_available_tier",
            fallback_tier="repo_delta_memory" if path_count else "",
            telemetry={
                "path_count": path_count,
                "prompt_cache_hit_ratio": prompt_cache_hit,
                "prefix_cache_hit_rate": prefix_cache_hit,
            },
            candidate_tiers=candidates,
        )

    def decide(
        self,
        runtime_status: Dict[str, Any] | None = None,
        *,
        delta_watermark: Dict[str, Any] | None = None,
        latent_package: Dict[str, Any] | None = None,
        compatibility: Dict[str, Any] | None = None,
        purpose: str = "runtime",
    ) -> MemoryTierDecision:
        repo_delta_memory = dict(delta_watermark or {})
        return self.choose(
            purpose=purpose,
            runtime_status=runtime_status,
            latent_package=latent_package,
            compatibility=compatibility,
            repo_delta_memory=repo_delta_memory,
        )
