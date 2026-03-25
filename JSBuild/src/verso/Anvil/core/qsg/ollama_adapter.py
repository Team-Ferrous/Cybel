import logging
import math
import inspect
import os
import re
import threading
import time
import importlib
from typing import Any, List

from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGInferenceEngine, QSGRequest
from core.qsg.holographic_encoder import HolographicEncoder
from core.qsg.grover import GroverAmplifier
from core.model.chat_templates import format_prompt_for_model
from core.model.gguf_loader import get_loader
from core.model.model_profile import ModelProfile
from core.runtime_control_policy import RuntimeControlPolicy
from core.native.native_tokenizer import NativeTokenizer
from core.native import engine as native_engine_module
from core.native import simd_ops_wrapper as simd_ops

_DEFAULT_NATIVE_ENGINE = native_engine_module.NativeInferenceEngine
NativeInferenceEngine = _DEFAULT_NATIVE_ENGINE
np = importlib.import_module("numpy")
LogitsProcessorList = list

from config.settings import (
    GENERATION_PARAMS,
    GRANITE4_SAMPLING_PROFILES,
    PERFORMANCE_CONFIG,
    QWEN35_SAMPLING_PROFILES,
)

logger = logging.getLogger(__name__)


def _sanitize_logits(scores: np.ndarray) -> np.ndarray:
    """Return finite float32 logits suitable for downstream sampling."""
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    return simd_ops.sanitize_logits(arr)


def _normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    """Normalize non-negative scores into a valid probability simplex."""
    probs = np.asarray(values, dtype=np.float32).reshape(-1)
    if probs.size == 0:
        return probs

    cleaned = np.empty_like(probs)
    total = 0.0
    for idx, value in enumerate(probs.tolist()):
        if not math.isfinite(value) or value < 0.0:
            cleaned[idx] = 0.0
            continue
        cleaned[idx] = float(value)
        total += float(value)

    if total <= 0.0 or not math.isfinite(total):
        return np.full(probs.shape, 1.0 / float(probs.size), dtype=np.float32)

    inv_total = 1.0 / total
    cleaned *= inv_total
    return cleaned


def _top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    if k <= 0:
        return []
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return []
    k = min(int(k), int(flat.size))
    return sorted(range(int(flat.size)), key=lambda i: float(flat[i]), reverse=True)[:k]


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except Exception:
        return None


def _native_context_profile_keys(profile: ModelProfile) -> tuple[str, str]:
    family = str(getattr(profile, "family", "") or "").strip().lower()
    arch = str(getattr(profile, "architecture", "") or "").strip().lower()
    if family == "qwen" or "qwen" in arch:
        return ("qwen35_native_ctx_default", "qwen35_native_ctx_cap")
    if family == "granite" or "granite" in arch:
        return ("granite4_native_ctx_default", "granite4_native_ctx_cap")
    return ("native_ctx_default", "native_ctx_cap")


def _resolve_native_context_length(
    *,
    configured_ctx: int,
    loader,
    profile: ModelProfile,
) -> tuple[int, int, int, int]:
    default_key, cap_key = _native_context_profile_keys(profile)
    global_default = int(GENERATION_PARAMS.get("native_ctx_default", 400000))
    profile_default = int(GENERATION_PARAMS.get(default_key, global_default))
    requested = int(configured_ctx)
    if requested <= 0:
        requested = max(1, profile_default)

    model_ctx_limit = requested
    get_ctx = getattr(loader, "get_context_length", None)
    if callable(get_ctx):
        try:
            model_ctx_limit = int(get_ctx())
        except Exception:
            model_ctx_limit = requested

    env_cap = _env_int("ANVIL_NATIVE_CTX_CAP")
    if env_cap is not None:
        configured_cap = int(env_cap)
    else:
        global_cap = int(GENERATION_PARAMS.get("native_ctx_cap", 400000))
        configured_cap = int(GENERATION_PARAMS.get(cap_key, global_cap))

    resolved = int(requested)
    if model_ctx_limit > 0:
        resolved = min(resolved, int(model_ctx_limit))
    if configured_cap > 0:
        resolved = min(resolved, int(configured_cap))
    resolved = max(1, int(resolved))
    return resolved, requested, int(model_ctx_limit), int(configured_cap)


class QSGStrictModeError(RuntimeError):
    """Raised when strict native QSG invariants are violated."""


class QSGLogitsProcessorError(QSGStrictModeError):
    """Raised when strict mode encounters logits processor failures."""


class RepetitionPenaltyProcessor:
    """Lightweight repetition penalty for QSG/COCONUT logits."""

    def __init__(self, penalty: float = 1.2, window: int = 128):
        self.penalty = penalty
        self.window = window

    def apply(self, input_ids: list, scores: np.ndarray) -> np.ndarray:
        if input_ids is None:
            return scores

        if isinstance(input_ids, np.ndarray):
            ids = input_ids.tolist()
        else:
            ids = list(input_ids)

        if len(ids) == 0:
            return scores

        recent = ids[-self.window :]
        counts = {}
        for tid in recent:
            counts[tid] = counts.get(tid, 0) + 1

        for tid, count in counts.items():
            if tid < len(scores):
                # Apply exponential penalty based on occurrence count
                p = self.penalty**count
                if scores[tid] > 0:
                    scores[tid] /= p
                else:
                    scores[tid] *= p
        return scores


class OllamaQSGAdapter:
    """
    Bridge between Ollama API and QSG pipeline.

    1. Extracts embeddings from Ollama for context.
    2. Runs QSG generation using real model weights via NativeInferenceEngine.
    3. (Optionally) Verifies with Ollama.
    """

    def __init__(self, model_name: str, config: QSGConfig = None, parent_ollama=None):
        # Determine model name from config or argument
        self.model_name = model_name
        self._config_provided = config is not None
        self.config = config or QSGConfig()
        self.strict_native_qsg = bool(
            PERFORMANCE_CONFIG.get("strict_native_qsg", False)
        )
        self.strict_coconut_bridge = (
            bool(PERFORMANCE_CONFIG.get("strict_coconut_bridge", False))
            or self.strict_native_qsg
        )
        self.strict_speculative_decode = (
            bool(PERFORMANCE_CONFIG.get("strict_speculative_decode", False))
            or self.strict_native_qsg
        )
        self.strict_logits_processor = (
            bool(PERFORMANCE_CONFIG.get("strict_logits_processor", False))
            or self.strict_native_qsg
        )

        # Compatibility with speculative verifier
        # Use parent if provided to avoid circular recursion
        self.ollama = parent_ollama

        # Load GGUF model directly
        self.start_total = time.time()
        logger.debug(f"Loading GGUF model: {model_name}...")
        self.loader = get_loader(model_name)
        self.profile = ModelProfile.from_loader(
            model_name=model_name, loader=self.loader
        )
        self._apply_profile_defaults()

        # Initialize Tokenizer
        logger.debug("Initializing tokenizer...")
        start_tok = time.time()
        vocab = self.loader.get_vocab_tokens()
        special_tokens = self.loader.get_special_tokens()
        metadata = self.loader.get_metadata()
        get_merges = getattr(self.loader, "get_tokenizer_merges", None)
        self.tokenizer = AnvilTokenizer(
            vocab,
            special_tokens,
            model_path=str(self.loader.model_path),
            bpe_merges=get_merges() if callable(get_merges) else None,
            tokenizer_model=str(metadata.get("tokenizer.ggml.model", "") or ""),
            tokenizer_pre=str(metadata.get("tokenizer.ggml.pre", "") or ""),
        )
        logger.debug(f"Tokenizer initialized in {time.time() - start_tok:.2f}s")

        # Initialize Native Engine
        logger.debug("Initializing Native Inference Engine...")
        start_engine = time.time()
        try:
            configured_ctx = int(GENERATION_PARAMS.get("num_ctx", 400000))
            ctx_len, requested_ctx, model_ctx_limit, cap_ctx = (
                _resolve_native_context_length(
                    configured_ctx=configured_ctx,
                    loader=self.loader,
                    profile=self.profile,
                )
            )
            logger.debug(
                "Info: Initializing QSG adapter with %s context tokens "
                "(requested=%s, model_limit=%s, cap=%s)",
                ctx_len,
                requested_ctx,
                model_ctx_limit,
                cap_ctx,
            )
            engine_factory = NativeInferenceEngine
            if engine_factory is _DEFAULT_NATIVE_ENGINE:
                engine_factory = native_engine_module.NativeInferenceEngine

            self.native_engine = engine_factory(
                self.model_name, context_length=ctx_len, embedding=False
            )
            if not getattr(self.native_engine, "supports_token_api", False):
                raise QSGStrictModeError(
                    "Unified native QSG requires token-level llama_cpp APIs."
                )
            logger.debug(
                f"Native Engine initialized in {time.time() - start_engine:.2f}s"
            )
        except Exception as e:
            metadata = {}
            try:
                metadata = self.loader.get_metadata()
            except Exception:
                metadata = {}
            architecture = str(metadata.get("general.architecture", "unknown"))
            raise RuntimeError(
                "Failed to initialize NativeInferenceEngine for "
                f"model='{model_name}', arch='{architecture}', "
                f"path='{self.loader.model_path}', "
                f"context={GENERATION_PARAMS.get('num_ctx', 400000)} "
                f"(native cap={GENERATION_PARAMS.get('native_ctx_cap', 16384)}). Error: {e}"
            ) from e

        # Lazy initialization of QSG components to save VRAM/RAM during startup
        self._vocab_embeddings = None
        self._lm_head_weight = None
        self._propagator = None
        self._encoder = None

        # Initialize placeholders for extra components
        self.coconut_bridge = None
        self.grover_amplifier = None
        self.speculative_decoder = None
        self.steering_engine = None
        self.hybrid_ssm = None
        self._active_latent_prior: list[float] | None = None
        self._active_repo_delta: dict[str, Any] = {}
        self._active_invariant_terms: list[str] = []
        self._active_context_text: str = ""
        self._last_grover_telemetry: dict[str, float] = {}
        self._last_generation_policy: dict[str, Any] = {}
        self._runtime_control_policy = RuntimeControlPolicy()
        self._spec_rejection_count = 0
        self._spec_disable_after_rejections = int(
            PERFORMANCE_CONFIG.get("spec_disable_after_rejections", 2)
        )

        # Initialize extra components (Coconut, Grover, Speculative, Steering, Hybrid SSM)
        try:
            self._init_extra_components()
        except Exception as e:
            raise QSGStrictModeError(
                "Unified native QSG requires successful component initialization."
            ) from e

        self._continuous_engine = None
        self._continuous_engine_thread = None
        if bool(getattr(self.config, "continuous_batching_enabled", False)):
            self._init_continuous_engine()

    @property
    def vocab_embeddings(self):
        if self._vocab_embeddings is None:
            logger.debug("Lazy-loading vocab embeddings...")
            self._vocab_embeddings = self.loader.get_token_embeddings()
        return self._vocab_embeddings

    @property
    def lm_head_weight(self):
        if self._lm_head_weight is None:
            logger.debug("Lazy-loading LM head weight...")
            self._lm_head_weight = self.loader.get_lm_head_weight()
        return self._lm_head_weight

    @property
    def propagator(self):
        if self._propagator is None:
            logger.debug("Extracting spectral propagator...")
            self._propagator = self.loader.extract_propagator(
                strategy="auto", profile=self.profile
            )
        return self._propagator

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = HolographicEncoder(dim=self.vocab_embeddings.shape[1])
        return self._encoder

    def _apply_profile_defaults(self):
        """Apply model-specific defaults to runtime QSG config."""
        # Respect explicit caller config while using model defaults when
        # parameters are not provided.
        grover_iterations = int(
            getattr(self.config, "grover_iterations", 0)
            if self._config_provided
            else getattr(self.profile, "grover_iterations", 2)
        )
        if grover_iterations <= 0:
            grover_iterations = int(getattr(self.profile, "grover_iterations", 2))
        self.config.grover_iterations = max(1, grover_iterations)
        if (
            bool(getattr(self.profile, "gqa", False))
            and self.config.grover_iterations > 2
        ):
            self.config.grover_iterations = 2
        coconut_paths = int(
            getattr(self.config, "coconut_paths", 0)
            if self._config_provided
            else getattr(self.profile, "coconut_paths", 2)
        )
        if coconut_paths <= 0:
            coconut_paths = int(getattr(self.profile, "coconut_paths", 2))
        self.config.coconut_paths = max(2, coconut_paths)
        explicit_coconut = getattr(self.config, "use_coconut", None)
        if explicit_coconut is None:
            explicit_coconut = getattr(self.config, "use_coconut_reasoning", True)
        self.config.use_coconut = bool(explicit_coconut) or self.strict_native_qsg
        configured_threshold = getattr(self.config, "acceptance_threshold", None)
        if configured_threshold is None or not self._config_provided:
            configured_threshold = getattr(
                self.profile, "speculative_acceptance_threshold", 0.7
            )
        self.config.acceptance_threshold = float(configured_threshold)

    @staticmethod
    def _infer_task_hint(prompt: str, resolved: dict) -> str:
        explicit = (
            str(
                resolved.get("task_type")
                or resolved.get("task")
                or resolved.get("intent")
                or ""
            )
            .strip()
            .lower()
        )
        if explicit:
            return explicit

        text = str(prompt or "").strip().lower()
        if not text:
            return ""

        coding_markers = (
            "code",
            "coding",
            "python",
            "typescript",
            "javascript",
            "rust",
            "golang",
            "c++",
            "bug",
            "fix",
            "test",
            "refactor",
            "implement",
            "compile",
            "webdev",
            "api",
        )

        def _contains_marker(marker: str) -> bool:
            if marker == "c++":
                return "c++" in text
            if len(marker) <= 3:
                return bool(
                    re.search(
                        rf"(?<![a-z0-9_]){re.escape(marker)}(?![a-z0-9_])",
                        text,
                    )
                )
            return marker in text

        if any(_contains_marker(marker) for marker in coding_markers):
            return "coding"

        reasoning_markers = (
            "reason",
            "analy",
            "compare",
            "tradeoff",
            "prove",
            "math",
            "explain why",
            "research",
        )
        if any(marker in text for marker in reasoning_markers):
            return "reasoning"
        return "general"

    @staticmethod
    def _resolve_qwen_profile_name(
        prompt: str,
        resolved: dict,
    ) -> str:
        explicit_profile = resolved.get("qwen35_sampling_profile") or resolved.get(
            "sampling_profile"
        )
        if explicit_profile:
            return str(explicit_profile)

        # If no explicit runtime profile is provided, allow config default to win
        # unless the caller gave a clear task/mode hint.
        configured_default = str(
            GENERATION_PARAMS.get("qwen35_sampling_profile", "instruct_deterministic")
        )
        task_hint = OllamaQSGAdapter._infer_task_hint(prompt, resolved)
        thinking_mode = bool(
            resolved.get("thinking")
            or resolved.get("use_thinking")
            or str(resolved.get("mode", "")).strip().lower() == "thinking"
        )

        if not task_hint:
            return configured_default

        if thinking_mode:
            return (
                "thinking_coding"
                if "cod" in task_hint or "web" in task_hint
                else "thinking_general"
            )

        if any(token in task_hint for token in ("reason", "research", "anal")):
            return "instruct_reasoning"
        if any(token in task_hint for token in ("cod", "web", "dev", "bug", "test")):
            return "thinking_coding"
        if task_hint == "general":
            return "instruct_general"
        return configured_default

    @staticmethod
    def _resolve_granite_profile_name(
        prompt: str,
        resolved: dict,
    ) -> str:
        explicit_profile = resolved.get("granite4_sampling_profile") or resolved.get(
            "sampling_profile"
        )
        if explicit_profile:
            return str(explicit_profile)

        configured_default = str(
            GENERATION_PARAMS.get("granite4_sampling_profile", "coding_balanced")
        )
        task_hint = OllamaQSGAdapter._infer_task_hint(prompt, resolved)
        if not task_hint:
            return configured_default
        if any(token in task_hint for token in ("reason", "research", "anal")):
            return "research_balanced"
        if any(token in task_hint for token in ("cod", "web", "dev", "bug", "test")):
            return "coding_balanced"
        if task_hint == "general":
            return "coding_deterministic"
        return configured_default

    def _apply_sampling_profile(self, options: dict | None, prompt: str = "") -> dict:
        """Resolve model-aware sampling defaults while honoring explicit overrides."""
        resolved = dict(options or {})
        if "num_predict" not in resolved:
            for alias in ("max_tokens", "max_new_tokens"):
                if alias not in resolved:
                    continue
                try:
                    resolved["num_predict"] = max(0, int(resolved[alias]))
                except Exception:
                    continue
                break
        model_lower = self.model_name.lower()

        if "qwen3.5" in model_lower or "qwen35" in model_lower:
            explicit_profile = resolved.get("qwen35_sampling_profile") or resolved.get(
                "sampling_profile"
            )
            if self.strict_native_qsg and not explicit_profile:
                profile_name = str(
                    GENERATION_PARAMS.get(
                        "qwen35_sampling_profile", "instruct_deterministic"
                    )
                )
            else:
                profile_name = self._resolve_qwen_profile_name(
                    prompt=prompt, resolved=resolved
                )
            profile = QWEN35_SAMPLING_PROFILES.get(profile_name)
            if profile is None:
                profile_name = str(
                    GENERATION_PARAMS.get(
                        "qwen35_sampling_profile", "instruct_deterministic"
                    )
                )
                profile = QWEN35_SAMPLING_PROFILES.get(profile_name)
            if profile is None:
                profile_name = "instruct_deterministic"
                profile = QWEN35_SAMPLING_PROFILES[profile_name]
            for key, value in profile.items():
                resolved.setdefault(key, value)
            resolved["qwen35_sampling_profile"] = profile_name
            return resolved

        if "granite4" in model_lower:
            profile_name = self._resolve_granite_profile_name(
                prompt=prompt, resolved=resolved
            )
            profile = GRANITE4_SAMPLING_PROFILES.get(profile_name)
            if profile is None:
                profile_name = str(
                    GENERATION_PARAMS.get(
                        "granite4_sampling_profile", "coding_balanced"
                    )
                )
                profile = GRANITE4_SAMPLING_PROFILES.get(profile_name)
            if profile is None:
                profile_name = "coding_balanced"
                profile = GRANITE4_SAMPLING_PROFILES[profile_name]
            for key, value in profile.items():
                resolved.setdefault(key, value)
            resolved["granite4_sampling_profile"] = profile_name

        runtime_policy = getattr(self, "_runtime_control_policy", None)
        if runtime_policy is None:
            runtime_policy = RuntimeControlPolicy()
            self._runtime_control_policy = runtime_policy

        policy = runtime_policy.decide_generation(
            model_name=self.model_name,
            prompt=prompt,
            options=resolved,
        )
        for key, value in policy.sampling_overrides.items():
            resolved.setdefault(key, value)
        resolved["runtime_context_band"] = policy.prompt_band
        resolved["runtime_policy_id"] = policy.policy_id
        resolved["runtime_model_family"] = policy.model_family
        resolved["runtime_prompt_tokens_estimate"] = policy.prompt_tokens_estimate
        resolved["runtime_task_hint"] = policy.task_hint
        resolved["runtime_cache_mode"] = policy.cache_mode
        resolved["runtime_shortlist_size"] = policy.shortlist_size
        self._last_generation_policy = policy.to_dict()
        return resolved

    @staticmethod
    def _filter_supported_kwargs(method, kwargs: dict) -> dict:
        try:
            params = inspect.signature(method).parameters
        except Exception:
            return dict(kwargs)

        for p in params.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return dict(kwargs)
        return {key: value for key, value in kwargs.items() if key in params}

    @staticmethod
    def _coerce_resonance_vector(payload: Any) -> list[float] | None:
        if payload is None:
            return None
        try:
            values = [float(value) for value in list(payload)]
        except Exception:
            return None
        if not values or not all(math.isfinite(value) for value in values):
            return None
        return values

    def _activate_resonance_context(
        self, prompt: str, resolved: dict[str, Any]
    ) -> None:
        self._active_context_text = str(
            resolved.get("context_text") or prompt or self._active_context_text or ""
        )
        latent_prior = (
            resolved.get("latent_prior")
            or resolved.get("subagent_latent_merged")
            or getattr(self.config, "latent_prior", None)
        )
        self._active_latent_prior = self._coerce_resonance_vector(latent_prior)
        self._active_repo_delta = dict(
            resolved.get("repo_delta")
            or resolved.get("delta_watermark")
            or self.config.delta_watermark
            or {}
        )
        self._active_invariant_terms = [
            str(term)
            for term in list(resolved.get("invariant_terms") or [])
            if str(term).strip()
        ]
        self._last_grover_telemetry = {}

    def _init_extra_components(self):
        """Initialize remaining non-critical components."""
        # Initialize Coconut Bridge
        try:
            from core.native.coconut_bridge import CoconutNativeBridge

            self.coconut_bridge = CoconutNativeBridge(
                embedding_dim=self.vocab_embeddings.shape[1],
                strict_native=self.strict_coconut_bridge,
            )
            logger.debug("COCONUT Native Bridge initialized.")
        except Exception as e:
            raise QSGStrictModeError(
                "Unified native QSG requires an available native COCONUT bridge."
            ) from e

        # Initialize Grover Amplifier with Semantic Engine
        self.semantic_engine = (
            self.ollama.semantic_engine
            if hasattr(self.ollama, "semantic_engine")
            else None
        )
        self.grover_amplifier = GroverAmplifier(
            semantic_engine=self.semantic_engine,
            resonance_timeout_ms=int(
                getattr(self.config, "semantic_resonance_timeout_ms", 4)
            ),
        )

        # Extract weights for COCONUT evolution
        self.evolution_weights = {
            "norm_gamma": np.ones(self.vocab_embeddings.shape[1], dtype=np.float32),
            "norm_beta": np.zeros(self.vocab_embeddings.shape[1], dtype=np.float32),
            "w1": self.propagator,
            "b1": np.zeros(self.propagator.shape[1], dtype=np.float32),
            "w2": np.eye(self.propagator.shape[0], dtype=np.float32),
            "b2": np.zeros(self.propagator.shape[0], dtype=np.float32),
            "hidden_dim": self.propagator.shape[1],
        }

        # Initialize CPU-optimized speculative decoder
        self.speculative_decoder = None
        if (
            PERFORMANCE_CONFIG.get("cpu_speculative_decode", False)
            and bool(getattr(self.profile, "speculative_enabled", True))
            and self.native_engine
        ):
            try:
                from core.native.cpu_speculative_decode import (
                    CPUSpeculativeDecoder,
                    SpeculativeConfig,
                )

                # Keep speculative decode parameters model-aware by default.
                # (granite4:tiny-h and qwen3.5 variants have different calibrated profiles)
                spec_config = SpeculativeConfig(
                    num_candidates=int(self.profile.spec_num_candidates),
                    max_draft_length=int(self.profile.spec_max_draft_length),
                    acceptance_threshold=float(self.profile.spec_acceptance_threshold),
                    use_coconut_drafts=True,
                    fallback_to_top_k=False,
                    fallback_to_standard_generation=False,
                    strict_native_only=True,
                )
                self.speculative_decoder = CPUSpeculativeDecoder(
                    self.native_engine, self.coconut_bridge, spec_config
                )
                logger.debug(
                    "CPU Speculative Decoder initialized (COCONUT-guided drafting)"
                )
            except Exception as e:
                raise QSGStrictModeError(
                    "Unified native QSG requires speculative decoder initialization without fallbacks."
                ) from e
        elif PERFORMANCE_CONFIG.get("cpu_speculative_decode", False):
            logger.info(
                "CPU speculative decode disabled for model profile '%s'.",
                self.model_name,
            )

        # Initialize latent steering engine
        self.steering_engine = None
        if PERFORMANCE_CONFIG.get("latent_steering", False):
            try:
                from core.native.latent_steering import LatentSteeringEngine

                self.steering_engine = LatentSteeringEngine(
                    self.vocab_embeddings,
                    self.tokenizer,
                    embedding_dim=self.vocab_embeddings.shape[1],
                    steering_dim=512,
                )
                logger.debug("Latent Steering Engine initialized")
            except Exception as e:
                logger.warning(f"Latent steering init failed: {e}")

        # Hybrid Python SSM path is disabled in strict native-only execution.
        self.hybrid_ssm = None

        logger.info(
            f"QSG Pipeline Ready (Total Init: {time.time() - self.start_total:.2f}s)"
        )

    def get_embeddings(self, text: str) -> np.ndarray:
        """Extract high-dimensional embedding for text using tokenizer + embedding matrix."""
        # Use batch_embeddings for a single string for consistency
        return self.batch_embeddings([text])[0].reshape(1, -1)

    def batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a batch of strings in parallel.
        Leverages 3D SIMD kernels for massive throughput.
        """
        if not texts:
            return np.zeros((0, self.vocab_embeddings.shape[1]), dtype=np.float32)

        # 1. Batch Tokenize (truncate to 2048 to save memory/time)
        padded_ids, _ = self.tokenizer.batch_encode(
            texts, padding=True, max_length=2048
        )

        # 2. Batch Lookup
        # Shape: [batch, seq, dim]
        batch_emb = self.vocab_embeddings[padded_ids]

        # 3. Batch Holographic Pooling (C++ SIMD)
        # encoder.encode handles 3D tensors by calling _encode_batch_cpp
        return self.encoder.encode(batch_emb)

    def generate(self, prompt: str, options: dict = None) -> str:
        """
        Generate text using the internal QSG pipeline.
        """
        if bool(getattr(self.config, "continuous_batching_enabled", False)):
            if self._continuous_engine is None:
                self._init_continuous_engine()
            self._ensure_continuous_engine_running()
            request = self._build_continuous_request(prompt, options)
            request_id = self._continuous_engine.submit(request)
            chunks: list[str] = []
            poll_interval = self._continuous_poll_interval_s()
            try:
                while True:
                    chunk = self._continuous_engine.poll(request_id)
                    if chunk is None:
                        time.sleep(poll_interval)
                        continue
                    if chunk.error:
                        raise RuntimeError(
                            f"Continuous QSG request failed: {chunk.error}"
                        )
                    if chunk.text:
                        chunks.append(chunk.text)
                    if chunk.done:
                        break
            except BaseException:
                self._continuous_engine.cancel(request_id)
                raise
            return "".join(chunks)
        return self._generate_qsg(prompt, options)

    def _init_continuous_engine(self) -> None:
        if self.native_engine is not None and hasattr(
            self.native_engine, "get_runtime_status"
        ):
            try:
                runtime_status = dict(self.native_engine.get_runtime_status())
            except Exception:
                runtime_status = {}
            self.config.capability_digest = str(
                runtime_status.get("capability_digest") or self.config.capability_digest
            )
            self.config.delta_watermark = dict(
                runtime_status.get("delta_watermark")
                or self.config.delta_watermark
                or {}
            )
        native_builder = getattr(
            self.native_engine,
            "build_parallel_generation_engine",
            None,
        )
        if callable(native_builder):
            self._continuous_engine = native_builder(
                config=self.config,
                stream_producer=self._continuous_stream_producer,
            )
        else:
            self._continuous_engine = QSGInferenceEngine(
                config=self.config,
                stream_producer=self._continuous_stream_producer,
            )
        self._ensure_continuous_engine_running()

    def _ensure_continuous_engine_running(self) -> None:
        if self._continuous_engine is None:
            return
        if (
            self._continuous_engine_thread is not None
            and self._continuous_engine_thread.is_alive()
        ):
            return
        self._continuous_engine_thread = threading.Thread(
            target=self._continuous_engine.run_forever,
            name=f"qsg-continuous-{self.model_name}",
            daemon=True,
        )
        self._continuous_engine_thread.start()

    def _continuous_stream_producer(self, request: QSGRequest):
        return self._stream_generate_qsg(
            request.prompt,
            options=request.options or {},
            prompt_token_ids=request.prompt_tokens,
            resolved_options=request.sampling,
        )

    def _build_continuous_request(
        self,
        prompt: str,
        options: dict | None = None,
    ) -> QSGRequest:
        resolved = self._apply_sampling_profile(options, prompt=prompt)
        formatted_prompt = self._format_prompt_for_model(prompt)
        prompt_tokens = self.native_engine.tokenize(formatted_prompt)
        sampling = {
            "num_predict": int(resolved.get("num_predict", 100)),
            "temperature": float(resolved.get("temperature", 0.8)),
            "top_p": float(resolved.get("top_p", 0.9)),
            "top_k": int(resolved.get("top_k", 40)),
            "min_p": float(resolved.get("min_p", 0.0)),
            "presence_penalty": float(resolved.get("presence_penalty", 0.0)),
            "repetition_penalty": float(
                resolved.get("repetition_penalty", resolved.get("repeat_penalty", 1.1))
            ),
        }
        if resolved.get("seed") is not None:
            sampling["seed"] = int(resolved["seed"])
        return QSGRequest(
            prompt=prompt,
            options=dict(options or {}),
            prompt_tokens=[int(token) for token in prompt_tokens],
            max_new_tokens=int(sampling["num_predict"]),
            sampling=sampling,
        )

    def _continuous_poll_interval_s(self) -> float:
        timeout_ms = int(getattr(self.config, "semantic_resonance_timeout_ms", 4))
        return max(0.001, float(timeout_ms) / 1000.0)

    def runtime_status(self) -> dict[str, Any]:
        return self.get_runtime_status()

    def _resolve_processor_mode(self, prompt_token_count: int) -> tuple[int, bool]:
        """Resolve COCONUT processor policy for generation/streaming."""
        prompt_token_count = int(max(0, prompt_token_count))
        context_usage = prompt_token_count / float(
            max(1, int(self.native_engine.context_length))
        )
        original_num_paths = max(1, int(getattr(self.config, "coconut_paths", 4)))

        if self.strict_native_qsg:
            # The sanctioned strict-native path keeps QSG logic inside the native
            # engine. Passing Python logits processors here forces a fallback the
            # native engine rejects on purpose.
            return original_num_paths, False

        if prompt_token_count < 60000:
            return original_num_paths, context_usage < 0.7
        if prompt_token_count < 100000:
            logger.info(
                "Info: Using lightweight COCONUT (2 paths) for %s tokens",
                prompt_token_count,
            )
            return 2, context_usage < 0.7
        return 0, False

    def _create_logits_processor(self) -> LogitsProcessorList:
        """Create a logits processor chain for QSG enhancements (Grover + CoCoNut)."""
        processors = []

        def _safe_processor(processor):
            def _wrapped(input_ids, scores):
                base_scores = _sanitize_logits(scores)
                out = processor(input_ids, base_scores.copy())
                if out is None:
                    if self.strict_logits_processor:
                        raise QSGLogitsProcessorError(
                            "Logits processor returned None in strict native mode."
                        )
                    return base_scores
                cleaned = _sanitize_logits(out)
                if cleaned.shape != base_scores.shape:
                    if self.strict_logits_processor:
                        raise QSGLogitsProcessorError(
                            "Logits processor changed tensor shape in strict native mode."
                        )
                    return base_scores
                return cleaned

            return _wrapped

        # 1. CoCoNut: Continuous Contextual Nuance Theory
        # High-Fidelity version: Uses C++ BFS kernels to evolve hidden state proxy
        if (
            hasattr(self.config, "use_coconut")
            and self.config.use_coconut
            and self.coconut_bridge
            and getattr(self.profile, "coconut_mode", "logits_proxy") != "disabled"
        ):
            # Define num_paths from config
            num_paths = getattr(self.config, "coconut_paths", 4)
            # Initialize repetitive penalty for COCONUT specifically
            rep_penalty = RepetitionPenaltyProcessor(penalty=1.1, window=128)

            def coconut_processor(input_ids, scores):
                try:
                    # scores is [vocab_size]. Build a logits-proxy hidden state by
                    # weighting top-k token embeddings with local softmax weights.
                    scores_arr = _sanitize_logits(scores)
                    top_k = min(
                        max(1, int(self.profile.logits_proxy_top_k)),
                        scores_arr.shape[0],
                    )
                    if top_k <= 0:
                        if self.strict_logits_processor:
                            raise QSGLogitsProcessorError(
                                "Invalid top-k derived for COCONUT logits proxy."
                            )
                        return scores

                    top_k_idx = _top_k_indices(scores_arr, top_k)
                    top_k_logits = [float(scores_arr[idx]) for idx in top_k_idx]
                    max_logit = max(top_k_logits) if top_k_logits else 0.0
                    weights = _normalize_nonnegative(
                        np.asarray(
                            [
                                math.exp(
                                    max(
                                        -80.0,
                                        min(0.0, float(logit) - float(max_logit)),
                                    )
                                )
                                for logit in top_k_logits
                            ],
                            dtype=np.float32,
                        )
                    )
                    if weights.size <= 0:
                        if self.strict_logits_processor:
                            raise QSGLogitsProcessorError(
                                "COCONUT logits proxy produced empty weights."
                            )
                        return scores

                    dim = int(self.vocab_embeddings.shape[1])
                    hidden_vec = np.zeros((dim,), dtype=np.float32)
                    for idx, w in zip(top_k_idx, weights.tolist()):
                        hidden_vec += self.vocab_embeddings[int(idx)] * float(w)
                    hidden_state = hidden_vec.reshape(1, -1)

                    # 1. Expand
                    paths = self.coconut_bridge.expand_paths(
                        hidden_state, num_paths, noise_scale=0.01
                    )

                    # 2. Evolve (using propagator as transition matrix)
                    w = self.evolution_weights
                    self.coconut_bridge.evolve_paths(
                        paths,
                        w["norm_gamma"],
                        w["norm_beta"],
                        w["w1"],
                        w["b1"],
                        w["w2"],
                        w["b2"],
                        w["hidden_dim"],
                    )

                    # 3. Score & Aggregate
                    # Use the logits-proxy hidden state as context for resonance.
                    amplitudes = self.coconut_bridge.score_paths(paths, hidden_state)

                    # OPTIMIZATION: Self-consistency verification
                    # Uses DeepSeek-R1 style multi-path agreement checking
                    # Force on for subagent reasoning stability
                    use_self_consistency = getattr(
                        self.config, "use_self_consistency", True
                    )
                    if use_self_consistency and num_paths >= 2:
                        # Verify paths against each other for consistency
                        aggregated_state, sorted_confidence = (
                            self.coconut_bridge.verify_paths_with_consistency(
                                paths, amplitudes, verification_threshold=0.7
                            )
                        )
                        aggregated_state = aggregated_state.reshape(1, -1)  # [1, dim]

                        # Alpha is dynamic based on resonance confidence
                        # If paths disagree (low confidence), reduce COCONUT influence to avoid hallucinated loops
                        confidence = sorted_confidence[0]
                        alpha = max(
                            0.0,
                            min(
                                1.0,
                                float(self.profile.coconut_alpha) * float(confidence),
                            ),
                        )
                    else:
                        # Standard aggregation
                        aggregated_state = self.coconut_bridge.aggregate_paths(
                            paths, amplitudes
                        )  # [1, dim]
                        alpha = float(self.profile.coconut_alpha)
                    if not math.isfinite(alpha):
                        alpha = 0.0
                    alpha = max(0.0, min(1.0, float(alpha)))

                    # 4. Project back to logit space using LM head
                    coconut_logits = _sanitize_logits(
                        simd_ops.matvec(
                            np.asarray(aggregated_state, dtype=np.float32).reshape(-1),
                            np.asarray(self.lm_head_weight.T, dtype=np.float32),
                        )
                    )

                    # 5. Apply local repetition penalty to COCONUT logits to discourage looping
                    coconut_logits = rep_penalty.apply(input_ids, coconut_logits)
                    coconut_logits = _sanitize_logits(coconut_logits)

                    # 6. Blend with original scores
                    blended = (1.0 - alpha) * scores_arr + alpha * coconut_logits
                    return _sanitize_logits(blended)
                except (
                    AttributeError,
                    FloatingPointError,
                    IndexError,
                    KeyError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                ) as exc:
                    logger.error(
                        "COCONUT logits processor failed: %s",
                        exc,
                        exc_info=True,
                    )
                    raise QSGLogitsProcessorError(
                        "Unified native QSG rejected COCONUT logits processor failure."
                    ) from exc

            processors.append(_safe_processor(coconut_processor))

        # 2. Grover: Amplitude Amplification
        if hasattr(self.config, "use_grover") and self.config.use_grover:

            def grover_processor(input_ids, scores):
                scores_arr = _sanitize_logits(scores)
                # Detokenize recent context for semantic engine
                recent_ids = input_ids[-64:] if len(input_ids) > 64 else input_ids
                context_text = (
                    self.native_engine.detokenize(recent_ids)
                    if self.native_engine
                    else ""
                )

                # Get vocabs for resonance scoring
                tokens = self.loader.get_vocab_tokens()

                iterations = getattr(self.config, "grover_iterations", 1)

                # Reshape for amplifier
                logits_reshaped = scores_arr.reshape(1, -1)

                # Use Semantic Resonance Oracle
                grover_telemetry: dict[str, float] = {}
                grover_kwargs = {
                    "iterations": iterations,
                    "model_profile": self.profile,
                    "top_k_oracle": self.profile.grover_top_k,
                    "damping": self.profile.grover_damping,
                    "token_embeddings": self.vocab_embeddings,
                    "latent_prior": self._active_latent_prior,
                    "repo_delta": self._active_repo_delta,
                    "invariant_terms": self._active_invariant_terms,
                    "telemetry_sink": grover_telemetry,
                }
                try:
                    grover_sig = inspect.signature(
                        self.grover_amplifier.amplify_with_resonance
                    )
                    grover_kwargs = {
                        key: value
                        for key, value in grover_kwargs.items()
                        if key in grover_sig.parameters
                    }
                    if "timeout_ms" in grover_sig.parameters:
                        grover_kwargs["timeout_ms"] = int(
                            getattr(self.config, "semantic_resonance_timeout_ms", 4)
                        )
                except (TypeError, ValueError):
                    pass
                amplified_probs = self.grover_amplifier.amplify_with_resonance(
                    logits_reshaped,
                    tokens,
                    context_text or self._active_context_text,
                    **grover_kwargs,
                )
                self._last_grover_telemetry = dict(grover_telemetry)

                # Convert back to logits
                amp = np.asarray(amplified_probs, dtype=np.float32)
                if amp.ndim == 0:
                    if self.strict_logits_processor:
                        raise QSGLogitsProcessorError(
                            "Grover amplification returned a scalar."
                        )
                    return scores_arr
                if amp.ndim == 1:
                    amp = amp.reshape(1, -1)
                if amp.shape[-1] != scores_arr.shape[0]:
                    if self.strict_logits_processor:
                        raise QSGLogitsProcessorError(
                            "Grover amplification returned mismatched vocabulary shape."
                        )
                    return scores_arr

                probs = _normalize_nonnegative(amp[0])
                if probs.size == 0:
                    if self.strict_logits_processor:
                        raise QSGLogitsProcessorError(
                            "Grover amplification produced empty probability vector."
                        )
                    return scores_arr
                new_logits = np.asarray(
                    [math.log(max(float(p), 1.0e-15)) for p in probs.tolist()],
                    dtype=np.float32,
                )
                return _sanitize_logits(new_logits)

            processors.append(_safe_processor(grover_processor))

        # 3. Latent Steering: HD gradient projection for intent alignment
        if self.steering_engine is not None and PERFORMANCE_CONFIG.get(
            "latent_steering", False
        ):
            steering_processor = self.steering_engine.create_steering_processor(
                self.lm_head_weight, adaptive_strength=True
            )
            processors.append(_safe_processor(steering_processor))

        return LogitsProcessorList(processors)

    def get_runtime_status(self) -> dict:
        if self.native_engine is None:
            status = {
                "backend": "unavailable",
                "model": self.model_name,
                "strict_native_qsg": bool(self.strict_native_qsg),
            }
            if self._last_grover_telemetry:
                status["grover_resonance"] = dict(self._last_grover_telemetry)
            return status
        get_status = getattr(self.native_engine, "get_runtime_status", None)
        status = dict(get_status()) if callable(get_status) else {}
        status.setdefault("adapter_model", self.model_name)
        get_contract = getattr(self.loader, "get_model_contract", None)
        if callable(get_contract):
            status.setdefault("model_contract", get_contract())
        status.setdefault("strict_native_qsg", bool(self.strict_native_qsg))
        status.setdefault(
            "model_profile",
            {
                "family": getattr(self.profile, "family", ""),
                "architecture": getattr(self.profile, "architecture", ""),
                "chat_template": getattr(self.profile, "chat_template", ""),
            },
        )
        if self._last_grover_telemetry:
            status.setdefault("grover_resonance", {})
            status["grover_resonance"].update(self._last_grover_telemetry)
        if self._last_generation_policy:
            status["runtime_policy"] = dict(self._last_generation_policy)
            status.setdefault("controller_state", {})
            status["controller_state"].setdefault(
                "generation_policy",
                dict(self._last_generation_policy.get("controller") or {}),
            )
        if self._continuous_engine is not None:
            status.setdefault("continuous_batching", {})
            status["continuous_batching"].update(
                {
                    "enabled": True,
                    **self._continuous_engine.metrics_snapshot(),
                }
            )
        return status

    def get_last_run_metrics(self) -> dict:
        if self.native_engine is None:
            return {}
        get_metrics = getattr(self.native_engine, "get_last_run_metrics", None)
        if not callable(get_metrics):
            return {}
        metrics = dict(get_metrics())
        if self._last_generation_policy:
            metrics["runtime_policy"] = dict(self._last_generation_policy)
            metrics.setdefault("controller_state", {})
            metrics["controller_state"].setdefault(
                "generation_policy",
                dict(self._last_generation_policy.get("controller") or {}),
            )
        return metrics

    def _format_prompt_for_model(self, prompt: str) -> str:
        return format_prompt_for_model(
            prompt,
            self.model_name,
            profile=self.profile,
        )

    def _decode_generated_tokens(self, token_ids: list[int]) -> str:
        if self.native_engine is None:
            return ""
        decode_generated_tokens = getattr(
            self.native_engine, "decode_generated_tokens", None
        )
        if callable(decode_generated_tokens):
            return str(decode_generated_tokens(token_ids))
        return str(self.native_engine.detokenize(token_ids))

    def _generate_qsg(self, prompt: str, options: dict = None) -> str:
        """QSG generation implementation."""
        logger.debug("Starting QSG generation...")

        resolved = self._apply_sampling_profile(options, prompt=prompt)
        self._last_generation_policy = {
            **dict(self._last_generation_policy),
            "sampling_profile": str(
                resolved.get("qwen35_sampling_profile")
                or resolved.get("granite4_sampling_profile")
                or ""
            ),
        }
        self._activate_resonance_context(prompt, resolved)
        max_tokens = resolved.get("num_predict", 100)
        temperature = resolved.get("temperature", 0.8)
        top_p = resolved.get("top_p", 0.9)
        top_k = resolved.get("top_k", 40)
        min_p = resolved.get("min_p", 0.0)
        presence_penalty = resolved.get("presence_penalty", 0.0)
        repetition_penalty = resolved.get(
            "repetition_penalty", resolved.get("repeat_penalty", 1.1)
        )
        seed = resolved.get("seed")
        _ = seed

        if not self.native_engine:
            raise RuntimeError(
                "Unified QSG requires NativeInferenceEngine; no legacy fallback path is available."
            )

        if not getattr(self.native_engine, "supports_token_api", False):
            raise QSGStrictModeError(
                "Unified native QSG requires token-level native generation support."
            )

        logger.debug(f"Using Native Engine for generation (max {max_tokens} tokens)...")

        # Use native tokenizer on a model-formatted prompt.
        formatted_prompt = self._format_prompt_for_model(prompt)
        token_ids = self.native_engine.tokenize(formatted_prompt)

        num_paths, use_processors = self._resolve_processor_mode(len(token_ids))
        original_num_paths = int(getattr(self.config, "coconut_paths", 4))

        logits_proc = None
        if use_processors:
            # Temporarily set num_paths for this generation
            self.config.coconut_paths = num_paths
            try:
                logits_proc = self._create_logits_processor()
            except Exception as e:
                raise QSGLogitsProcessorError(
                    "Unified native QSG disallows logits processor fallback."
                ) from e
            finally:
                # Restore original
                self.config.coconut_paths = original_num_paths
        else:
            logger.info("Info: COCONUT/Grover processors disabled for this prompt.")

        # Generate tokens using speculative decoder if enabled
        if self.speculative_decoder is not None:
            # Use CPU-optimized speculative decoding
            coconut_resources = {
                "vocab_embeddings": self.vocab_embeddings,
                "lm_head_weight": self.lm_head_weight,
                "evolution_weights": self.evolution_weights,
            }
            try:
                new_tokens = []
                for token in self.speculative_decoder.generate_speculative(
                    token_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    coconut_resources=coconut_resources,
                    logits_processor=logits_proc,
                ):
                    new_tokens.append(token)

                # Decode
                text = self._decode_generated_tokens(new_tokens)
                logger.debug(
                    f"Speculative generation complete ({len(new_tokens)} tokens)."
                )
                return text
            except Exception as exc:
                if exc.__class__.__name__ != "SpeculativeFallbackDisabledError":
                    raise
                raise QSGStrictModeError(
                    "Unified native QSG disallows speculative decode fallback."
                ) from exc

        # Standard generation
        generate_kwargs = self._filter_supported_kwargs(
            self.native_engine.generate,
            {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
                "logits_processor": logits_proc,
            },
        )
        full_sequence = self.native_engine.generate(token_ids, **generate_kwargs)
        new_tokens = full_sequence[len(token_ids) :]

        # Decode
        text = self._decode_generated_tokens(new_tokens)
        logger.debug(f"Native generation complete ({len(new_tokens)} tokens).")
        return text

    def stream_generate(self, prompt: str, options: dict = None) -> str:
        """
        Stream generated text using the internal QSG pipeline.
        """
        if bool(getattr(self.config, "continuous_batching_enabled", False)):
            if self._continuous_engine is None:
                self._init_continuous_engine()
            self._ensure_continuous_engine_running()
            request = self._build_continuous_request(prompt, options)
            request_id = self._continuous_engine.submit(request)
            poll_interval = self._continuous_poll_interval_s()
            try:
                while True:
                    chunk = self._continuous_engine.poll(request_id)
                    if chunk is None:
                        time.sleep(poll_interval)
                        continue
                    if chunk.error:
                        raise RuntimeError(
                            f"Continuous QSG request failed: {chunk.error}"
                        )
                    if chunk.text:
                        yield chunk.text
                    if chunk.done:
                        break
            except BaseException:
                self._continuous_engine.cancel(request_id)
                raise
            return
        yield from self._stream_generate_qsg(prompt, options)

    def _stream_generate_qsg(
        self,
        prompt: str,
        options: dict = None,
        *,
        prompt_token_ids: list[int] | None = None,
        resolved_options: dict[str, Any] | None = None,
    ) -> str:
        """QSG streaming implementation."""
        resolved = dict(
            resolved_options or self._apply_sampling_profile(options, prompt=prompt)
        )
        self._activate_resonance_context(prompt, resolved)
        max_tokens = resolved.get("num_predict", 100)
        temperature = resolved.get("temperature", 0.8)
        top_p = resolved.get("top_p", 0.9)
        top_k = resolved.get("top_k", 40)
        min_p = resolved.get("min_p", 0.0)
        presence_penalty = resolved.get("presence_penalty", 0.0)
        repetition_penalty = resolved.get(
            "repetition_penalty", resolved.get("repeat_penalty", 1.1)
        )
        seed = resolved.get("seed")
        _ = seed

        if not self.native_engine:
            raise RuntimeError(
                "Unified QSG requires NativeInferenceEngine; no legacy fallback path is available."
            )

        if not getattr(self.native_engine, "supports_token_api", False):
            raise QSGStrictModeError(
                "Unified native QSG requires token-level native streaming support."
            )

        logger.debug(f"Using Native Engine for streaming (max {max_tokens} tokens)...")

        native_token_ids = (
            [int(token) for token in prompt_token_ids]
            if prompt_token_ids is not None
            else self.native_engine.tokenize(self._format_prompt_for_model(prompt))
        )

        num_paths, use_processors = self._resolve_processor_mode(len(native_token_ids))
        original_num_paths = int(getattr(self.config, "coconut_paths", 4))

        logits_proc = None
        if use_processors:
            # Temporarily set num_paths for this generation
            self.config.coconut_paths = num_paths
            try:
                logits_proc = self._create_logits_processor()
            except Exception as e:
                raise QSGLogitsProcessorError(
                    "Unified native QSG disallows logits processor fallback."
                ) from e
            finally:
                # Restore original
                self.config.coconut_paths = original_num_paths
        else:
            logger.info("Info: COCONUT/Grover processors disabled for this prompt.")

        # Use speculative decoder if enabled
        if self.speculative_decoder is not None:
            # CPU-optimized speculative decoding (streaming)
            coconut_resources = {
                "vocab_embeddings": self.vocab_embeddings,
                "lm_head_weight": self.lm_head_weight,
                "evolution_weights": self.evolution_weights,
            }
            try:
                emitted_text = ""
                generated_tokens: list[int] = []
                for next_token in self.speculative_decoder.generate_speculative(
                    native_token_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    coconut_resources=coconut_resources,
                    logits_processor=logits_proc,
                ):
                    generated_tokens.append(int(next_token))
                    text = self._decode_generated_tokens(generated_tokens)
                    if text.startswith(emitted_text):
                        delta = text[len(emitted_text) :]
                    else:
                        delta = text
                    emitted_text = text
                    if delta:
                        yield delta
                return
            except Exception as exc:
                if exc.__class__.__name__ != "SpeculativeFallbackDisabledError":
                    raise
                raise QSGStrictModeError(
                    "Unified native QSG disallows speculative streaming fallback."
                ) from exc

        # Standard streaming generation
        stream_kwargs = self._filter_supported_kwargs(
            self.native_engine.generate_stream,
            {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "presence_penalty": presence_penalty,
                "repetition_penalty": repetition_penalty,
                "logits_processor": logits_proc,
            },
        )
        emitted_text = ""
        generated_tokens: list[int] = []
        for next_token in self.native_engine.generate_stream(
            native_token_ids, **stream_kwargs
        ):
            generated_tokens.append(int(next_token))
            text = self._decode_generated_tokens(generated_tokens)
            if text.startswith(emitted_text):
                delta = text[len(emitted_text) :]
            else:
                delta = text
            emitted_text = text
            if delta:
                yield delta


AnvilTokenizer = NativeTokenizer
