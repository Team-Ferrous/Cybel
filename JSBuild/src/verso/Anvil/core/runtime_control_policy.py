"""Runtime control policy for adapting mission execution from live telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeControlDecision:
    """Concrete execution posture derived from native runtime telemetry."""

    posture: str = "unknown"
    planning_depth: str = "standard"
    verification_max_attempts: int = 2
    verification_breadth: str = "standard"
    degraded: bool = False
    reasons: list[str] = field(default_factory=list)
    telemetry_digest: str = ""
    backend_proof: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "posture": self.posture,
            "planning_depth": self.planning_depth,
            "verification_max_attempts": int(self.verification_max_attempts),
            "verification_breadth": self.verification_breadth,
            "degraded": bool(self.degraded),
            "reasons": list(self.reasons),
            "telemetry_digest": self.telemetry_digest,
            "backend_proof": self.backend_proof,
        }


@dataclass
class GenerationOptimizationPolicy:
    """Model- and prompt-aware runtime policy for native generation."""

    model_family: str = "generic"
    prompt_band: str = "medium"
    prompt_tokens_estimate: int = 0
    task_hint: str = "general"
    policy_id: str = "generic-medium-general"
    cache_mode: str = "balanced"
    shortlist_size: int = 32
    reason: str = ""
    controller: dict[str, Any] = field(default_factory=dict)
    sampling_overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "prompt_band": self.prompt_band,
            "prompt_tokens_estimate": int(self.prompt_tokens_estimate),
            "task_hint": self.task_hint,
            "policy_id": self.policy_id,
            "cache_mode": self.cache_mode,
            "shortlist_size": int(self.shortlist_size),
            "reason": self.reason,
            "controller": dict(self.controller),
            "sampling_overrides": dict(self.sampling_overrides),
        }


class RuntimeControlPolicy:
    """Turn runtime telemetry into explicit mission-control guidance."""

    def __init__(
        self,
        *,
        drift_overhead_warn_pct: float = 15.0,
        drift_overhead_fail_pct: float = 25.0,
    ) -> None:
        self.drift_overhead_warn_pct = float(drift_overhead_warn_pct)
        self.drift_overhead_fail_pct = float(drift_overhead_fail_pct)

    def decide(self, runtime_status: dict[str, Any] | None) -> RuntimeControlDecision:
        status = dict(runtime_status or {})
        digest = str(status.get("capability_digest") or "")
        hot_path = dict(status.get("hot_path_proof") or {})
        full_qsg = str(hot_path.get("full_qsg") or "").lower()
        python_hot_path_calls = int(status.get("python_hot_path_calls") or 0)
        drift_overhead = float(status.get("drift_overhead_percent") or 0.0)
        backend_loaded = bool(status.get("backend_module_loaded", False))
        abi_match = bool(status.get("native_backend_abi_match", False))
        native_fast_path = bool(status.get("native_fast_path", False))

        reasons: list[str] = []
        posture = "native_ready"
        planning_depth = "deep"
        verification_max_attempts = 2
        verification_breadth = "standard"
        degraded = False

        if not status:
            return RuntimeControlDecision(
                posture="unknown",
                planning_depth="standard",
                verification_max_attempts=2,
                verification_breadth="standard",
                degraded=False,
                reasons=["runtime_status_unavailable"],
                telemetry_digest="",
                backend_proof="unknown",
            )

        if not backend_loaded or not abi_match:
            posture = "backend_unverified"
            planning_depth = "compact"
            verification_breadth = "targeted"
            verification_max_attempts = 1
            degraded = True
            reasons.append("backend_module_unverified")

        if python_hot_path_calls > 0 or not native_fast_path or full_qsg == "disabled":
            posture = "degraded_python_fallback"
            planning_depth = "compact"
            verification_breadth = "targeted"
            verification_max_attempts = 1
            degraded = True
            reasons.append("python_hot_path_detected")

        if drift_overhead >= self.drift_overhead_fail_pct:
            posture = "high_drift_overhead"
            planning_depth = "lean"
            verification_breadth = "targeted"
            verification_max_attempts = 1
            degraded = True
            reasons.append("drift_overhead_critical")
        elif drift_overhead >= self.drift_overhead_warn_pct:
            planning_depth = "compact"
            verification_breadth = "targeted"
            verification_max_attempts = min(verification_max_attempts, 1)
            degraded = True
            reasons.append("drift_overhead_elevated")

        if not reasons:
            reasons.append("native_runtime_ready")

        backend_proof = "verified" if backend_loaded and abi_match and native_fast_path else "degraded"
        return RuntimeControlDecision(
            posture=posture,
            planning_depth=planning_depth,
            verification_max_attempts=verification_max_attempts,
            verification_breadth=verification_breadth,
            degraded=degraded,
            reasons=reasons,
            telemetry_digest=digest,
            backend_proof=backend_proof,
        )

    def decide_generation(
        self,
        *,
        model_name: str,
        prompt: str,
        options: dict[str, Any] | None = None,
    ) -> GenerationOptimizationPolicy:
        """Choose a model-family runtime policy for the current prompt."""

        resolved = dict(options or {})
        model_lower = str(model_name or "").strip().lower()
        family = (
            "qwen"
            if "qwen" in model_lower
            else "granite"
            if "granite" in model_lower
            else "generic"
        )
        prompt_tokens_estimate = max(
            1,
            int(
                resolved.get("prompt_tokens_estimate")
                or resolved.get("prompt_tokens")
                or (len(str(prompt or "")) + 3) // 4
            ),
        )
        if prompt_tokens_estimate <= 512:
            prompt_band = "short"
        elif prompt_tokens_estimate <= 4096:
            prompt_band = "medium"
        else:
            prompt_band = "long"

        text = str(prompt or "").lower()
        if any(token in text for token in ("code", "bug", "test", "refactor", "python", "api")):
            task_hint = "coding"
        elif any(token in text for token in ("reason", "research", "analy", "proof", "math")):
            task_hint = "reasoning"
        else:
            task_hint = "general"

        temperature = float(resolved.get("temperature", 0.0) or 0.0)
        shortlist_size = 32
        cache_mode = "balanced"
        sampling_overrides: dict[str, Any] = {}

        if family == "granite":
            if prompt_band == "long":
                cache_mode = "decode_resident"
            elif task_hint == "reasoning":
                cache_mode = "balanced_hybrid"
            else:
                cache_mode = "prompt_lookup"
            sampling_overrides["granite_hybrid_cache_mode"] = cache_mode
        elif family == "qwen":
            shortlist_size = 48 if prompt_band == "short" else 32 if prompt_band == "medium" else 16
            if task_hint == "reasoning":
                shortlist_size = max(12, shortlist_size - 8)
            if temperature >= 0.6:
                shortlist_size = max(8, shortlist_size - 8)
            sampling_overrides["qwen_dynamic_shortlist"] = shortlist_size
            sampling_overrides.setdefault("top_k", shortlist_size)

        policy_id = f"{family}-{prompt_band}-{task_hint}"
        reason = f"{family} {prompt_band}-context {task_hint} policy"
        return GenerationOptimizationPolicy(
            model_family=family,
            prompt_band=prompt_band,
            prompt_tokens_estimate=prompt_tokens_estimate,
            task_hint=task_hint,
            policy_id=policy_id,
            cache_mode=cache_mode,
            shortlist_size=shortlist_size,
            reason=reason,
            controller={
                "controller": "generation_policy",
                "selected_mode": cache_mode,
                "prompt_band": prompt_band,
                "task_hint": task_hint,
            },
            sampling_overrides=sampling_overrides,
        )
