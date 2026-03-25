import json
import hashlib
from typing import Any, Dict, List

from core.model.chat_templates import (
    format_chat_messages,
    resolve_chat_template_name,
    resolve_prompt_contract,
)
from core.model.model_contract import canonicalize_model_name
from core.runtime_control_policy import RuntimeControlPolicy
from core.utils.logger import get_logger
from config.settings import (
    CONTINUOUS_BATCHING_CONFIG,
    GRANITE4_SAMPLING_PROFILES,
    GENERATION_PARAMS,
    QSG_CONFIG,
    QWEN35_SAMPLING_PROFILES,
)

logger = get_logger(__name__)


class DeterministicOllama:
    _loader_cache = {}

    def __init__(self, model_name):
        self.model_name = canonicalize_model_name(model_name)
        self.runtime_control_policy = RuntimeControlPolicy()
        cache_key = self._loader_cache_key(self.model_name)
        if cache_key not in DeterministicOllama._loader_cache:
            DeterministicOllama._loader_cache[cache_key] = self._get_loader()
        self.loader = DeterministicOllama._loader_cache[cache_key]

    @staticmethod
    def _loader_cache_key(model_name: str) -> tuple[str, str, int, int, int]:
        lower = str(model_name or "").lower()
        if "qwen3.5" in lower or "qwen35" in lower:
            profile = str(GENERATION_PARAMS.get("qwen35_sampling_profile", ""))
            native_default = int(GENERATION_PARAMS.get("qwen35_native_ctx_default", 0))
            native_cap = int(GENERATION_PARAMS.get("qwen35_native_ctx_cap", 0))
        elif "granite4" in lower:
            profile = str(GENERATION_PARAMS.get("granite4_sampling_profile", ""))
            native_default = int(
                GENERATION_PARAMS.get("granite4_native_ctx_default", 0)
            )
            native_cap = int(GENERATION_PARAMS.get("granite4_native_ctx_cap", 0))
        else:
            profile = ""
            native_default = int(GENERATION_PARAMS.get("native_ctx_default", 0))
            native_cap = int(GENERATION_PARAMS.get("native_ctx_cap", 0))
        num_ctx = int(GENERATION_PARAMS.get("num_ctx", 400000))
        return (str(model_name), profile, num_ctx, native_default, native_cap)

    @classmethod
    def clear_loader_cache(cls, model_name: str | None = None) -> None:
        if model_name is None:
            cls._loader_cache.clear()
            return
        prefix = str(model_name)
        stale = [
            key
            for key in cls._loader_cache
            if (isinstance(key, tuple) and key and key[0] == prefix) or key == prefix
        ]
        for key in stale:
            cls._loader_cache.pop(key, None)

    def _loader_label(self) -> str:
        if self.loader is None:
            return "unavailable"
        if hasattr(self.loader, "native_engine"):
            return "qsg"
        return type(self.loader).__name__.lower()

    def _require_loader(self, operation: str):
        if self.loader is None:
            raise RuntimeError(
                f"Unified native QSG-only path is required for '{operation}' "
                f"(model='{self.model_name}'). HTTP Ollama API fallback is disabled."
            )
        return self.loader

    def _params_hash(self, params: Dict[str, Any]) -> str:
        try:
            payload = json.dumps(params, sort_keys=True, default=str)
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        except Exception:
            return "unhashable"

    def _log_stream_envelope(self, event: str, metrics: Dict[str, Any]) -> None:
        payload = {
            "component": "ollama_client",
            "event": event,
            "model": self.model_name,
            "metrics": metrics,
        }
        logger.debug(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        )

    def _get_loader(self):
        from core.qsg.ollama_adapter import OllamaQSGAdapter
        from core.qsg.config import QSGConfig

        config = QSGConfig()
        config.use_grover = bool(QSG_CONFIG.get("use_grover", True))
        config.use_coconut = bool(QSG_CONFIG.get("use_coconut_reasoning", True))
        config.use_coconut_reasoning = config.use_coconut

        # Keep model-profile calibration authoritative for numeric knobs.
        # The adapter treats 0/None as "use profile defaults".
        config.grover_iterations = 0
        config.coconut_paths = 0
        config.acceptance_threshold = None
        config.continuous_batching_enabled = bool(
            CONTINUOUS_BATCHING_CONFIG.get("enabled", False)
        )
        config.max_active_requests = max(
            1,
            int(CONTINUOUS_BATCHING_CONFIG.get("max_active_requests", 4)),
        )
        config.max_pending_requests = max(
            config.max_active_requests,
            int(CONTINUOUS_BATCHING_CONFIG.get("max_pending_requests", 4096)),
        )
        config.scheduler_policy = (
            str(CONTINUOUS_BATCHING_CONFIG.get("scheduler_policy", "fcfs")).strip()
            or "fcfs"
        )
        config.batch_wait_timeout_ms = max(
            1,
            int(CONTINUOUS_BATCHING_CONFIG.get("batch_wait_timeout_ms", 2)),
        )
        config.semantic_resonance_timeout_ms = max(
            1,
            int(CONTINUOUS_BATCHING_CONFIG.get("semantic_poll_timeout_ms", 4)),
        )

        return OllamaQSGAdapter(self.model_name, config=config, parent_ollama=self)

    def _chat_template_name(self) -> str:
        return resolve_chat_template_name(
            self.model_name,
            profile=getattr(self.loader, "profile", None),
            strict=True,
        )

    def _apply_model_sampling_profile(
        self,
        params: Dict[str, Any],
        explicit_keys: set[str] | None = None,
    ) -> Dict[str, Any]:
        """Apply model-specific sampler defaults while preserving explicit overrides."""
        explicit = set(explicit_keys or set())
        model_lower = self.model_name.lower()
        if "qwen3.5" in model_lower or "qwen35" in model_lower:
            configured_default = str(
                GENERATION_PARAMS.get(
                    "qwen35_sampling_profile", "instruct_deterministic"
                )
            )
            profile_name = str(
                params.get("qwen35_sampling_profile")
                or params.get("sampling_profile")
                or configured_default
            )
            profile = QWEN35_SAMPLING_PROFILES.get(profile_name)
            if profile is None:
                profile_name = (
                    configured_default
                    if configured_default in QWEN35_SAMPLING_PROFILES
                    else "instruct_deterministic"
                )
                profile = QWEN35_SAMPLING_PROFILES[profile_name]
            for key, value in profile.items():
                if key not in explicit:
                    params[key] = value
            params["qwen35_sampling_profile"] = profile_name
            return params

        if "granite4" in model_lower:
            profile_name = str(
                params.get("granite4_sampling_profile")
                or params.get("sampling_profile")
                or GENERATION_PARAMS.get("granite4_sampling_profile", "coding_balanced")
            )
            profile = GRANITE4_SAMPLING_PROFILES.get(profile_name)
            if profile is None:
                profile_name = "coding_balanced"
                profile = GRANITE4_SAMPLING_PROFILES[profile_name]
            for key, value in profile.items():
                if key not in explicit:
                    params[key] = value
            params["granite4_sampling_profile"] = profile_name
        return params

    def _apply_runtime_generation_policy(
        self,
        prompt: str,
        params: Dict[str, Any],
        explicit_keys: set[str] | None = None,
    ) -> Dict[str, Any]:
        explicit = set(explicit_keys or set())
        policy = self.runtime_control_policy.decide_generation(
            model_name=self.model_name,
            prompt=prompt,
            options=params,
        )
        for key, value in policy.sampling_overrides.items():
            if key not in explicit:
                params[key] = value
        params["runtime_context_band"] = policy.prompt_band
        params["runtime_policy_id"] = policy.policy_id
        params["runtime_model_family"] = policy.model_family
        params["runtime_prompt_tokens_estimate"] = policy.prompt_tokens_estimate
        params["runtime_task_hint"] = policy.task_hint
        params["runtime_cache_mode"] = policy.cache_mode
        params["runtime_shortlist_size"] = policy.shortlist_size
        return params

    def _format_chat_prompt(self, messages, assistant_prefix=None) -> str:
        contract = resolve_prompt_contract(
            self.model_name,
            profile=getattr(self.loader, "profile", None),
            strict=True,
        )
        return format_chat_messages(
            messages,
            contract.template_name,
            assistant_prefix=(
                assistant_prefix
                if assistant_prefix is not None
                else contract.assistant_prefix
            ),
            system_prompt=contract.system_prompt,
            inject_system_prompt=bool(contract.inject_system_prompt),
        )

    def generate(
        self,
        prompt,
        system_prompt=None,
        generation_params: Dict[str, Any] | None = None,
        **custom_params,
    ):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        loader = self._require_loader("generate")
        if not hasattr(loader, "generate"):
            raise RuntimeError(
                "Unified native QSG loader does not implement 'generate'. "
                "HTTP Ollama API fallback is disabled."
            )
        gen_params = (
            generation_params.copy() if generation_params else GENERATION_PARAMS.copy()
        )
        if "max_tokens" in custom_params:
            gen_params["num_predict"] = custom_params.pop("max_tokens")
        explicit_keys = set(custom_params.keys())
        gen_params.update(custom_params)
        gen_params = self._apply_model_sampling_profile(gen_params)
        gen_params = self._apply_runtime_generation_policy(
            prompt,
            gen_params,
            explicit_keys,
        )
        return loader.generate(prompt, gen_params)

    def stream_generate(
        self,
        prompt,
        system_prompt=None,
        generation_params: Dict[str, Any] = None,
        log_envelope: bool = True,
    ):
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"

        gen_params = (
            generation_params.copy() if generation_params else GENERATION_PARAMS.copy()
        )
        gen_params = self._apply_model_sampling_profile(gen_params)
        gen_params = self._apply_runtime_generation_policy(prompt, gen_params)
        if log_envelope:
            self._log_stream_envelope(
                "ollama.stream_generate.start",
                {
                    "loader": self._loader_label(),
                    "prompt_chars": len(prompt or ""),
                    "assistant_prefix_chars": 0,
                    "params_hash": self._params_hash(gen_params),
                    "override_keys": sorted(
                        k
                        for k, v in gen_params.items()
                        if GENERATION_PARAMS.get(k) != v or k not in GENERATION_PARAMS
                    ),
                },
            )

        loader = self._require_loader("stream_generate")
        if not hasattr(loader, "stream_generate"):
            raise RuntimeError(
                "Unified native QSG loader does not implement 'stream_generate'. "
                "HTTP Ollama API fallback is disabled."
            )
        yield from loader.stream_generate(prompt, gen_params)

    def chat(self, messages, assistant_prefix=None, **custom_params):
        """
        Chat completion with optional custom generation parameters.

        Args:
            messages: List of message dicts with 'role' and 'content'
            assistant_prefix: Optional prefix for assistant response
            **custom_params: Custom generation parameters (e.g., max_tokens, temperature)
                            These override GENERATION_PARAMS for this call only
        """
        # Merge custom params with defaults
        gen_params = GENERATION_PARAMS.copy()
        explicit_keys = set(custom_params.keys())

        # Handle max_tokens -> num_predict mapping
        if "max_tokens" in custom_params:
            gen_params["num_predict"] = custom_params.pop("max_tokens")

        # Merge remaining custom params
        gen_params.update(custom_params)
        gen_params = self._apply_model_sampling_profile(gen_params, explicit_keys)

        loader = self._require_loader("chat")
        if not hasattr(loader, "generate"):
            raise RuntimeError(
                "Unified native QSG loader does not implement 'generate' for chat. "
                "HTTP Ollama API fallback is disabled."
            )

        prompt = self._format_chat_prompt(messages, assistant_prefix=assistant_prefix)
        gen_params = self._apply_runtime_generation_policy(
            prompt,
            gen_params,
            explicit_keys,
        )
        return loader.generate(prompt, gen_params)

    def stream_chat(self, messages, assistant_prefix=None, **custom_params):
        """
        Streaming chat completion with optional custom generation parameters.

        Args:
            messages: List of message dicts with 'role' and 'content'
            assistant_prefix: Optional prefix for assistant response
            **custom_params: Custom generation parameters (e.g., max_tokens, temperature)
                            These override GENERATION_PARAMS for this call only
        """
        # Merge custom params with defaults
        gen_params = GENERATION_PARAMS.copy()
        provided_params = dict(custom_params)
        explicit_keys = set(custom_params.keys())

        # Handle max_tokens -> num_predict mapping
        if "max_tokens" in custom_params:
            gen_params["num_predict"] = custom_params.pop("max_tokens")

        # Merge remaining custom params
        gen_params.update(custom_params)
        gen_params = self._apply_model_sampling_profile(gen_params, explicit_keys)
        role_counts: Dict[str, int] = {}
        total_prompt_chars = 0
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
            total_prompt_chars += len(str(msg.get("content", "")))

        self._log_stream_envelope(
            "ollama.stream_chat.start",
            {
                "loader": self._loader_label(),
                "role_counts": role_counts,
                "messages": len(messages),
                "prompt_chars": total_prompt_chars,
                "assistant_prefix_chars": len(assistant_prefix or ""),
                "params_hash": self._params_hash(gen_params),
                "override_keys": sorted(provided_params.keys()),
            },
        )

        self._require_loader("stream_chat")
        prompt = self._format_chat_prompt(messages, assistant_prefix=assistant_prefix)
        gen_params = self._apply_runtime_generation_policy(
            prompt,
            gen_params,
            explicit_keys,
        )
        yield from self.stream_generate(
            prompt,
            generation_params=gen_params,
            log_envelope=False,
        )

    def embeddings(self, prompt):
        return self.batch_embeddings([prompt])[0]

    def runtime_status(self) -> Dict[str, Any]:
        loader = self._require_loader("runtime_status")
        if hasattr(loader, "runtime_status"):
            return loader.runtime_status()
        status_fn = getattr(loader, "get_runtime_status", None)
        if callable(status_fn):
            return dict(status_fn())
        if hasattr(loader, "native_engine") and hasattr(
            loader.native_engine, "get_runtime_status"
        ):
            return loader.native_engine.get_runtime_status()
        return {"backend": self._loader_label(), "model": self.model_name}

    def runtime_capability_banner(self) -> str:
        status = self.runtime_status()
        return (
            f"QSG {status.get('backend', 'native_qsg')} "
            f"model={status.get('model', self.model_name)} "
            f"digest={str(status.get('digest', 'unknown'))[:18]} "
            f"threads={status.get('decode_threads', 'n/a')}/{status.get('batch_threads', 'n/a')} "
            f"OpenMP={status.get('openmp_enabled', False)} "
            f"AVX2={status.get('avx2_enabled', False)}"
        )

    def batch_embeddings(self, prompts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of prompts."""
        loader = self._require_loader("embeddings")

        if hasattr(loader, "batch_embeddings"):
            try:
                embs = loader.batch_embeddings(prompts)
                return (
                    embs.tolist()
                    if hasattr(embs, "tolist")
                    else [list(e) for e in embs]
                )
            except Exception as exc:
                raise RuntimeError(
                    "Unified native QSG loader failed in 'batch_embeddings'. "
                    "HTTP Ollama API fallback is disabled."
                ) from exc

        if hasattr(loader, "get_embeddings"):
            results = []
            for prompt_text in prompts:
                emb = loader.get_embeddings(prompt_text)
                results.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
            return results

        if hasattr(loader, "native_engine") and loader.native_engine:
            if hasattr(loader.native_engine, "embed"):
                return [
                    loader.native_engine.embed(prompt_text) for prompt_text in prompts
                ]

        raise RuntimeError(
            "Unified native QSG loader does not provide embeddings "
            "(expected one of: batch_embeddings, get_embeddings, native_engine.embed). "
            "HTTP Ollama API fallback is disabled."
        )
