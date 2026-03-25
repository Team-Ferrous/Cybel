import os
import sys
import subprocess
import json
import time
from typing import List, Optional, Any
from core.model.gguf_loader import GGUFModelLoader

_LLAMA_PROBE_CACHE: dict[str, tuple[bool, str]] = {}
_LLAMA_CLASS = None

# Performance optimizations
try:
    from core.native.incremental_kv_cache import IncrementalKVCache

    INCREMENTAL_KV_AVAILABLE = True
except Exception:
    INCREMENTAL_KV_AVAILABLE = False

try:
    from core.native.semantic_kv_cache import SemanticKVCache

    SEMANTIC_KV_AVAILABLE = True
except Exception:
    SEMANTIC_KV_AVAILABLE = False

try:
    import importlib.util

    PAGED_KV_AVAILABLE = (
        importlib.util.find_spec("core.native.paged_kv_cache") is not None
    )
except Exception:
    PAGED_KV_AVAILABLE = False

try:
    from config.settings import PERFORMANCE_CONFIG, GPU_CONFIG
except ImportError:
    PERFORMANCE_CONFIG = {
        "incremental_kv_cache": False,
        "semantic_kv_cache": False,
        "paged_kv_cache": False,
    }
    GPU_CONFIG = {"enabled": False, "n_gpu_layers": 0}


def detect_gpu():
    """Auto-detect GPU availability."""
    # Priority: CUDA -> ROCm -> Metal -> Vulkan
    try:
        # Check for CUDA (NVIDIA)
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            return {"type": "cuda", "available": True, "name": "NVIDIA GPU"}
    except Exception:
        pass

    try:
        # Check for ROCm (AMD)
        result = subprocess.run(["rocm-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            return {"type": "rocm", "available": True, "name": "AMD GPU (ROCm)"}
    except Exception:
        pass

    # Check for Metal (Apple Silicon)
    if sys.platform == "darwin":
        try:
            # Simple way to check for Metal on macOS
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            if "Metal" in result.stdout:
                return {"type": "metal", "available": True, "name": "Apple Metal"}
        except Exception:
            pass

    try:
        # Check for Vulkan (Cross-platform, good for AMD RX 6600)
        # Check for vulkaninfo or look for discrete GPUs in vulkaninfo summary
        result = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_name = "Vulkan GPU"
            for line in result.stdout.split("\n"):
                if "deviceName" in line:
                    gpu_name = line.split("=")[1].strip()
                    break
            return {"type": "vulkan", "available": True, "name": gpu_name}
    except Exception:
        pass

    return {"type": None, "available": False, "name": "None"}


def _load_llama_class():
    global _LLAMA_CLASS
    if _LLAMA_CLASS is not None:
        return _LLAMA_CLASS
    from llama_cpp import Llama

    _LLAMA_CLASS = Llama
    return _LLAMA_CLASS


def _try_read_model_architecture(model_path: str) -> Optional[str]:
    """Best-effort architecture probe for loader diagnostics."""
    if not model_path or not os.path.exists(model_path):
        return None

    loader = None
    try:
        loader = GGUFModelLoader(model_path)
        metadata = loader.get_metadata()
        architecture = str(metadata.get("general.architecture", "")).strip().lower()
        return architecture or None
    except Exception:
        return None
    finally:
        if loader is not None:
            try:
                loader.close()
            except Exception:
                pass


def _build_unsupported_architecture_message(
    model_path: str, error_msg: str
) -> Optional[str]:
    """Build a targeted guidance message for unsupported GGUF architectures."""
    msg = (error_msg or "").lower()
    architecture = _try_read_model_architecture(model_path)

    unsupported_markers = (
        "unknown model architecture",
        "unsupported model architecture",
        "unknown architecture",
        "model architecture not supported",
        "unsupported architecture",
    )
    looks_unsupported = any(marker in msg for marker in unsupported_markers)

    if not looks_unsupported:
        return None

    architecture_label = architecture or "unknown"
    specific_hint = (
        "Your current llama-cpp-python runtime likely does not support this GGUF "
        "architecture tag yet."
    )

    return (
        f"Unsupported GGUF architecture '{architecture_label}' for model '{model_path}'. "
        f"{specific_hint} "
        f"Original loader error: {error_msg}. "
        "Action: upgrade llama-cpp-python to a build that supports this architecture, "
        "or re-export/re-pull the model with compatible metadata in "
        "'general.architecture'. Unified QSG has no legacy generation fallback."
    )


def _probe_llama_runtime_support(model_path: str) -> tuple[bool, str]:
    """Probe model load in a subprocess so crashes don't kill the parent process."""
    cached = _LLAMA_PROBE_CACHE.get(model_path)
    if cached is not None:
        return cached

    timeout_s = int(os.getenv("ANVIL_LLAMA_PROBE_TIMEOUT", "45") or "45")
    probe_script = r"""
import json
import sys
try:
    from llama_cpp import Llama
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"llama_cpp import failed: {exc}"}))
    raise SystemExit(0)

model_path = sys.argv[1]
kwargs = {
    "model_path": model_path,
    "n_ctx": 256,
    "n_threads": 1,
    "n_threads_batch": 1,
    "embedding": False,
    "verbose": False,
    "use_mmap": True,
}
try:
    try:
        Llama(vocab_only=True, **kwargs)
    except TypeError:
        Llama(**kwargs)
    print(json.dumps({"ok": True}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_script, str(model_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        outcome = (False, f"probe timeout after {timeout_s}s")
        _LLAMA_PROBE_CACHE[model_path] = outcome
        return outcome
    except Exception as exc:
        outcome = (False, f"probe launcher failed: {exc}")
        _LLAMA_PROBE_CACHE[model_path] = outcome
        return outcome

    stdout_lines = (result.stdout or "").strip().splitlines()
    stderr_text = (result.stderr or "").strip()
    probe_error = ""
    if stdout_lines:
        try:
            parsed = json.loads(stdout_lines[-1])
            if isinstance(parsed, dict):
                if bool(parsed.get("ok")):
                    outcome = (True, "")
                    _LLAMA_PROBE_CACHE[model_path] = outcome
                    return outcome
                probe_error = str(parsed.get("error") or "").strip()
        except Exception:
            probe_error = ""

    if not probe_error:
        if result.returncode < 0:
            probe_error = f"probe process terminated by signal {-result.returncode}"
        elif result.returncode > 0:
            probe_error = f"probe process exited with code {result.returncode}"
        else:
            probe_error = "probe reported unsupported runtime"
    if stderr_text:
        probe_error = f"{probe_error}; stderr={stderr_text[-500:]}"

    outcome = (False, probe_error)
    _LLAMA_PROBE_CACHE[model_path] = outcome
    return outcome


def _build_preflight_architecture_guard_message(model_path: str) -> Optional[str]:
    """
    Fail fast for architectures known to be unstable with the current runtime.

    This prevents hard crashes (segfaults) during Llama(...) construction when
    the runtime cannot safely parse a GGUF architecture.
    """
    if os.getenv("ANVIL_ALLOW_UNSAFE_ARCH_LOAD", "0") == "1":
        return None

    architecture = _try_read_model_architecture(model_path)
    supported, reason = _probe_llama_runtime_support(str(model_path))
    if not supported:
        architecture_label = architecture or "unknown"
        return (
            f"Preflight blocked model load for architecture '{architecture_label}' "
            f"(model='{model_path}'). Runtime probe failed: {reason}. "
            "Action: upgrade llama-cpp-python/llama.cpp to a build with explicit "
            f"support for '{architecture_label}', then retry. "
            "Set ANVIL_ALLOW_UNSAFE_ARCH_LOAD=1 only if you explicitly want to risk "
            "a native crash during model init."
        )
    return None


def _validate_native_backend_setting() -> None:
    """Enforce single-backend native QSG runtime."""
    forced = str(os.getenv("ANVIL_NATIVE_ENGINE_BACKEND", "native_qsg")).strip().lower()
    if forced in {"", "auto", "native_qsg"}:
        return
    raise RuntimeError(
        "Supported native backends: native_qsg (default). "
        "llama_cpp fallback is disabled in strict native mode."
    )


class RepetitionTracker:
    """Tracks token history to detect semantic/infinite loops."""

    def __init__(self, window_size: int = 32):
        self.tokens = []
        self.window_size = window_size

    def add(self, token: int) -> bool:
        self.tokens.append(token)
        if len(self.tokens) > self.window_size:
            self.tokens.pop(0)
        return self.is_looping()

    def is_looping(self) -> bool:
        if len(self.tokens) < 6:
            return False

        # 1. Broad Single token repetition (e.g. "static static static...")
        # Higher threshold for single tokens as they might be common (e.g. spaces)
        last_token = self.tokens[-1]
        if all(t == last_token for t in self.tokens[-10:]):
            return True

        # 2. Sequence repetition (length 2-8)
        # Check if a sequence repeats 3 times consecutively
        for length in range(2, 9):
            if len(self.tokens) < length * 3:
                continue

            # Extract last 3 segments of this length
            seg1 = self.tokens[-length:]
            seg2 = self.tokens[-2 * length : -length]
            seg3 = self.tokens[-3 * length : -2 * length]

            if seg1 == seg2 == seg3:
                # Special case: if the sequence is just one repeating token,
                # let the single-token check handle it with its own threshold
                if len(set(seg1)) > 1:
                    return True
                # If it's a single token, we only trigger if it's long enough
                # (handled by the single-token check above)
        return False


class LlamaCppInferenceEngine:
    """
    Wrapper around llama-cpp-python for robust CPU inference.
    Replaces the experimental Native/SIMD engine for production reliability.
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 400000,
        use_mmap: bool = True,
        embedding: bool = False,
    ):
        strict_native = bool(PERFORMANCE_CONFIG.get("strict_native_qsg", False))
        allow_legacy = (
            str(os.getenv("ANVIL_ALLOW_LLAMA_CPP_ENGINE", "0")).strip() == "1"
        )
        if strict_native and not allow_legacy:
            raise RuntimeError(
                "LlamaCppInferenceEngine is disabled in strict native mode. "
                "Set ANVIL_ALLOW_LLAMA_CPP_ENGINE=1 only for legacy debugging."
            )

        self.model_path = model_path
        self.context_length = context_length
        self.embedding_enabled = embedding
        self.architecture = _try_read_model_architecture(str(model_path))
        self.backend = "llama_cpp"
        self.supports_token_api = True
        self.llm = None

        _validate_native_backend_setting()

        try:
            llama_cls = _load_llama_class()
        except Exception as exc:
            raise ImportError(
                "llama-cpp-python is required for NativeInferenceEngine. "
                "Please pip install llama-cpp-python."
            ) from exc

        preflight_msg = _build_preflight_architecture_guard_message(str(model_path))
        if preflight_msg:
            raise RuntimeError(preflight_msg)

        print(f"Loading model via llama-cpp: {model_path}...")
        # Llama class handles GGUF loading directly
        try:
            # Autodetect CPU cores
            threads = os.cpu_count() or 4

            # CPU-first configuration. GPU probing is intentionally disabled
            # during engine initialization to avoid startup subprocess overhead.
            if GPU_CONFIG.get("force_cpu", False):
                n_gpu_layers = 0
                gpu_backend_type = None
                print("Native Engine: force_cpu=True, using CPU.")
            elif GPU_CONFIG.get("enabled", False):
                n_gpu_layers = GPU_CONFIG.get("n_gpu_layers", 0)
                gpu_backend_type = None
                print(
                    "Native Engine: GPU enabled in config, using configured offload "
                    f"layers={n_gpu_layers}."
                )
            else:
                n_gpu_layers = 0
                gpu_backend_type = None
                print("Native Engine: GPU disabled in config, using CPU.")

            # --- OPTIMIZED ALLOCATION FOR LONG CONTEXT (200K+) ---
            # 1. flash_attn=True reduces KV cache memory usage by ~50%
            # 2. use_mlock=False prevents redundant RAM pinning
            # 3. use_mmap=True allows efficient lazy loading
            # 4. n_ctx is applied carefully to avoid pre-allocation overflow

            self.llm = llama_cls(
                model_path=str(model_path),
                n_ctx=context_length,
                verbose=False,
                use_mmap=use_mmap,
                use_mlock=GPU_CONFIG.get(
                    "use_mlock", False
                ),  # Disabled by default for OOM safety
                n_gpu_layers=n_gpu_layers,
                main_gpu=GPU_CONFIG.get("main_gpu", 0),
                tensor_split=GPU_CONFIG.get("tensor_split"),
                n_threads=threads,
                n_threads_batch=threads,
                embedding=self.embedding_enabled,
                flash_attn=True,  # CRITICAL: Enabled for long context efficiency
                # Explicitly pass the detected GPU backend type
                gpu_backend=gpu_backend_type if n_gpu_layers != 0 else None,
            )
        except Exception as e:
            error_msg = str(e)
            if (
                "OutOfDeviceMemory" in error_msg
                or "failed to allocate" in error_msg.lower()
            ):
                print("\n[bold red]FATAL: GPU Out of Memory (OOM)[/bold red]")
                print(
                    f"Native Engine failed to allocate VRAM for model + {context_length} context tokens."
                )
                print("Suggestions:")
                print("1. Reduce 'num_ctx' in config/settings.py (try 32768 or 65536)")
                print(
                    "2. Reduce 'n_gpu_layers' in config/settings.py (e.g. to 20 instead of -1) to offload less to VRAM"
                )
                print("3. Ensure no other large applications are using the GPU VRAM\n")
            else:
                unsupported_arch_msg = _build_unsupported_architecture_message(
                    str(model_path), error_msg
                )
                if unsupported_arch_msg:
                    print(f"Error loading model: {unsupported_arch_msg}")
                    raise RuntimeError(unsupported_arch_msg) from e
                print(f"Error loading model: {e}")
            raise

        # Metadata access via loader for standard interface if needed,
        # but Llama object has it too.
        # Keeping self.loader as None or minimal if adapter relies on it?
        # Adapter used self.native_engine.generate()

        # We can keep a loader instance just for metadata if strictly needed by clients
        # accessing engine.loader.*
        try:
            self.loader = GGUFModelLoader(model_path)
        except Exception:
            self.loader = None

        # Expose metadata for compatibility
        if self.loader:
            self.n_layer = self.loader.get_layer_count()
            self.dim = self.loader.get_embedding_dim()
        else:
            self.n_layer = 0
            self.dim = 0

        # Initialize KV cache with best available strategy
        self.kv_cache_manager = None
        use_semantic = (
            PERFORMANCE_CONFIG.get("semantic_kv_cache", False) and SEMANTIC_KV_AVAILABLE
        )
        use_paged = (
            PERFORMANCE_CONFIG.get("paged_kv_cache", False) and PAGED_KV_AVAILABLE
        )
        use_incremental = (
            PERFORMANCE_CONFIG.get("incremental_kv_cache", False)
            and INCREMENTAL_KV_AVAILABLE
        )

        # Priority order: Semantic > Paged > Incremental > Standard
        if use_semantic:
            try:
                # Semantic KV cache for massive contexts (1M+ tokens)
                self.kv_cache_manager = SemanticKVCache(
                    self.llm._ctx,
                    max_seq_len=context_length,
                    llm_obj=self.llm,
                    recent_window=8192,  # Keep 8K tokens uncompressed
                    max_crystals=64,  # 64 semantic crystals
                    compression_interval=16384,  # Compress every 16K tokens
                )
                print(
                    "Native Engine (llama-cpp) Initialized with Semantic KV Cache (compression enabled)."
                )
            except Exception as e:
                print(
                    f"Warning: Could not enable semantic KV cache: {e}. Falling back."
                )
                use_semantic = False

        if not use_semantic and use_paged:
            try:
                # PagedAttention KV cache for CPU cache efficiency
                # Use HybridPagedCache for single-sequence semantics
                from core.native.paged_kv_cache import HybridPagedCache

                self.kv_cache_manager = HybridPagedCache(
                    self.llm._ctx,
                    llm_obj=self.llm,
                    page_size=64,  # CPU cache line aligned
                    num_pages=512,  # Adjust based on available memory
                )
                print(
                    "Native Engine (llama-cpp) Initialized with Paged KV Cache (64-token pages)."
                )
            except Exception as e:
                print(f"Warning: Could not enable paged KV cache: {e}. Falling back.")
                use_paged = False

        if not use_semantic and not use_paged and use_incremental:
            try:
                # Standard incremental KV cache with prefix matching
                self.kv_cache_manager = IncrementalKVCache(
                    self.llm._ctx, self.context_length, llm_obj=self.llm
                )
                print(
                    "Native Engine (llama-cpp) Initialized with Incremental KV Cache."
                )
            except Exception as e:
                print(f"Warning: Could not enable incremental KV cache: {e}")
                print("Native Engine (llama-cpp) Initialized (standard mode).")

        if self.kv_cache_manager is None:
            print("Native Engine (llama-cpp) Initialized (standard mode).")

    def reset_kv_cache(self):
        """Reset the KV cache and token counter to a clean state. Critical for preventing llama_decode errors."""
        cache_reset = False
        if getattr(self, "kv_cache_manager", None) is not None and hasattr(
            self.kv_cache_manager, "reset"
        ):
            try:
                self.kv_cache_manager.reset()
                cache_reset = True
            except Exception as e:
                print(f"Warning: Could not reset KV cache manager: {e}")

        if not cache_reset:
            try:
                self.llm.reset()
            except Exception as e:
                print(f"Warning: Could not reset model state: {e}")

    def tokenize(self, text: str) -> List[int]:
        if isinstance(text, bytes):
            return self.llm.tokenize(text, add_bos=True)
        return self.llm.tokenize(text.encode("utf-8"), add_bos=True)

    def detokenize(self, tokens: List[int]) -> str:
        return self.llm.detokenize(tokens).decode("utf-8", errors="ignore")

    def embed(self, text: str) -> List[float]:
        """Compute embeddings locally using llama-cpp."""
        if not self.embedding_enabled:
            return []
        return self.llm.embed(text)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ) -> str:
        prompt_tokens = self.tokenize(prompt)
        full_sequence = self.generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return self.detokenize(full_sequence[len(prompt_tokens) :])

    def generate_stream_text(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        seed: Optional[int] = None,
    ):
        prompt_tokens = self.tokenize(prompt)
        for token in self.generate_stream(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            yield self.detokenize([token])

    def generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        logits_processor: Optional[List[Any]] = None,
    ) -> List[int]:
        """
        Generate tokens using llama-cpp.
        """
        # Use incremental KV cache if available, otherwise reset
        start_pos = 0
        if self.kv_cache_manager:
            start_pos = self.kv_cache_manager.prepare_for_generation(
                prompt_tokens, allow_reuse=True
            )
        else:
            self.reset_kv_cache()

        # Llama.generate expects tokens
        output_tokens = list(prompt_tokens)

        # Only generating new tokens
        if len(prompt_tokens) >= self.context_length:
            return prompt_tokens

        count = 0

        # Determine if we can reuse KV cache
        # If start_pos > 0, we set n_past and use reset=False
        if start_pos > 0:
            self.llm.n_past = start_pos
            tokens_to_process = prompt_tokens[start_pos:]
        else:
            tokens_to_process = prompt_tokens

        for token in self.llm.generate(
            tokens_to_process,
            temp=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            logits_processor=logits_processor,
            reset=(start_pos == 0),
        ):
            if max_new_tokens > 0 and count >= max_new_tokens:
                break

            output_tokens.append(token)
            count += 1

            # Update KV cache position if using incremental cache
            if self.kv_cache_manager:
                self.kv_cache_manager.advance_position(1, token_ids=[token])

            # Stop token check? Llama usually handles eos?
            if token == self.llm.token_eos():
                break

        return output_tokens

    def generate_stream(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        logits_processor: Optional[List[Any]] = None,
    ):
        """
        Stream tokens using llama-cpp with robust error recovery.
        """
        # CRITICAL FIX: Validate prompt length and ensure room for generation
        # Cap safety_margin to 30% of context window to prevent negative values
        max_safety_margin = int(self.context_length * 0.3)
        safety_margin = min(max(max_new_tokens, 128), max_safety_margin)
        max_prompt_tokens = self.context_length - safety_margin

        if max_prompt_tokens <= 0:
            # Emergency fallback: use 70/30 split
            max_prompt_tokens = int(self.context_length * 0.7)
            print(
                f"Warning: Invalid token budget. Using {max_prompt_tokens} for prompt."
            )

        if len(prompt_tokens) > max_prompt_tokens:
            original_length = len(prompt_tokens)
            # Truncate from the start (keep recent context)
            prompt_tokens = prompt_tokens[-max_prompt_tokens:]
            print(
                f"Info: Truncated prompt to {len(prompt_tokens)} tokens (was {original_length}) to fit context window"
            )

        # Double-check we're not at the very edge
        if len(prompt_tokens) >= self.context_length:
            print(
                f"Warning: Prompt length ({len(prompt_tokens)}) still exceeds context window ({self.context_length}). This should not happen."
            )
            # Emergency truncation
            prompt_tokens = prompt_tokens[-(self.context_length - 256) :]

        start_pos = 0
        # Use incremental KV cache if available, otherwise reset
        if self.kv_cache_manager:
            try:
                start_pos = self.kv_cache_manager.prepare_for_generation(
                    prompt_tokens, allow_reuse=True
                )
            except Exception as e:
                print(f"Warning: KV cache preparation failed: {e}. Resetting.")
                self.reset_kv_cache()
        else:
            # Always reset KV cache before generation to prevent stale state if not using incremental
            self.reset_kv_cache()

        count = 0
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries:
            try:
                # Determine tokens to decode based on cache reuse
                if start_pos > 0:
                    self.llm.n_past = start_pos
                    tokens_to_decode = prompt_tokens[start_pos:]
                    should_reset = False
                else:
                    tokens_to_decode = prompt_tokens
                    should_reset = True

                tracker = RepetitionTracker()
                for token in self.llm.generate(
                    tokens_to_decode,
                    temp=temperature,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                    logits_processor=logits_processor if retry_count == 0 else None,
                    reset=should_reset,
                ):
                    # Check for repetitions/loops
                    if tracker.add(token):
                        raise RuntimeError(
                            f"Semantic loop detected in output stream after {count} tokens"
                        )

                    # Allow negative max_new_tokens to mean "unlimited" (up to context limit)
                    if max_new_tokens > 0 and count >= max_new_tokens:
                        return

                    yield token
                    count += 1

                    # Update KV cache position if using incremental cache
                    if self.kv_cache_manager:
                        try:
                            self.kv_cache_manager.advance_position(1, token_ids=[token])
                        except Exception:
                            pass  # Ignore cache update errors during generation

                    if token == self.llm.token_eos():
                        return

                # If we got here, generation completed successfully
                return

            except RuntimeError as e:
                # Catch both direct RuntimeErrors and those wrapped from C++
                error_msg = str(e)
                is_decode_error = (
                    "llama_decode" in error_msg or "failed to decode" in error_msg
                )
                is_loop_error = "Semantic loop" in error_msg

                if (is_decode_error or is_loop_error) and retry_count < max_retries - 1:
                    print(
                        f"Warning: Native Engine error (attempt {retry_count + 1}/{max_retries}): {e}"
                    )
                    print(
                        "Critical: Performing hard state reset and retrying without logits processor..."
                    )

                    # Full Hard Reset - RECREATE THE CONTEXT IF POSSIBLE OR JUST RESET
                    self.reset_kv_cache()
                    start_pos = 0

                    # Progressive trimming strategy: trim more on each retry
                    # Retry 1: trim 10% from start
                    # Retry 2: trim 25% from start
                    # This is more aggressive than the fixed 256 token approach
                    trim_ratios = [0.10, 0.25, 0.40]
                    trim_ratio = (
                        trim_ratios[retry_count]
                        if retry_count < len(trim_ratios)
                        else 0.5
                    )

                    tokens_to_remove = int(len(prompt_tokens) * trim_ratio)
                    # Ensure we remove at least 512 tokens on first retry, more on subsequent
                    min_removal = 512 * (retry_count + 1)
                    tokens_to_remove = max(tokens_to_remove, min_removal)

                    if len(prompt_tokens) > tokens_to_remove:
                        original_len = len(prompt_tokens)
                        prompt_tokens = prompt_tokens[tokens_to_remove:]
                        print(
                            f"Info: Trimmed {tokens_to_remove} tokens ({trim_ratio:.0%}) from prompt ({original_len} → {len(prompt_tokens)}) for retry stability."
                        )
                    else:
                        # Emergency fallback: keep last 50% of prompt (less aggressive than 20%)
                        emergency_keep = int(len(prompt_tokens) * 0.5)
                        prompt_tokens = prompt_tokens[-emergency_keep:]
                        print(
                            f"Warning: Emergency trim - keeping last {emergency_keep} tokens (50%)."
                        )
                    retry_count += 1
                    # Retry with simpler parameters and no logits processor
                    logits_processor = None
                    continue
                else:
                    print(
                        f"Error: Generation failed after {retry_count + 1} attempts: {e}"
                    )
                    return  # Give up gracefully instead of crashing
            except Exception as e:
                print(f"Error during generation: {e}")
                return

    def close(self):
        """Release model weights, KV cache, and loader to free memory."""
        if hasattr(self, "kv_cache_manager") and self.kv_cache_manager is not None:
            try:
                if hasattr(self.kv_cache_manager, "reset"):
                    self.kv_cache_manager.reset()
            except Exception:
                pass
            self.kv_cache_manager = None

        if hasattr(self, "llm") and self.llm is not None:
            try:
                del self.llm
            except Exception:
                pass
            self.llm = None

        if hasattr(self, "loader") and self.loader is not None:
            try:
                self.loader.close()
            except Exception:
                pass
            self.loader = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _get_backend_name() -> str:
    forced = str(os.getenv("ANVIL_NATIVE_ENGINE_BACKEND", "native_qsg")).strip().lower()
    if forced in {"", "auto"}:
        return "native_qsg"
    return forced


class NativeInferenceEngine:
    """Backend selector preserving the adapter-facing engine symbol."""

    def __new__(cls, model_path: str, *args, **kwargs):
        if cls is not NativeInferenceEngine:
            return super().__new__(cls)

        _validate_native_backend_setting()
        backend = _get_backend_name()
        if backend != "native_qsg":
            raise RuntimeError(
                "NativeInferenceEngine only supports backend='native_qsg'. "
                f"Received '{backend}'."
            )

        from core.native.native_qsg_engine import NativeQSGEngine

        return NativeQSGEngine(model_path, *args, **kwargs)
