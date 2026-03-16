"""Native graph-backed QSG engine built directly on GGUF weights."""

from __future__ import annotations

import inspect
import json
import math
import os
import pathlib
import platform
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from array import array

import numpy as np


def _bootstrap_openmp_env_defaults() -> None:
    if os.getenv("ANVIL_DISABLE_OMP_AFFINITY_DEFAULTS", "0") == "1":
        return
    if os.getenv("OMP_PROC_BIND") is None:
        os.environ["OMP_PROC_BIND"] = str(os.getenv("ANVIL_OMP_PROC_BIND", "close"))
    if os.getenv("OMP_PLACES") is None:
        os.environ["OMP_PLACES"] = str(os.getenv("ANVIL_OMP_PLACES", "cores"))


_bootstrap_openmp_env_defaults()

from config.settings import GENERATION_PARAMS, PERFORMANCE_CONFIG
from core.model.chat_templates import (
    format_strict_native_prompt,
    get_strict_prompt_contract,
    postprocess_strict_native_response,
    resolve_chat_template_name,
)
from core.model.gguf_loader import GGUFModelLoader
from core.model.model_contract import model_contract_snapshot, resolve_model_contract
from core.model.model_profile import ModelProfile
from core.native.native_tokenizer import NativeTokenizer
from core.native.native_ops import (
    get_native_backend_info,
    get_native_library_info,
)
from core.native.parallel_generation import (
    _native_verify_draft_tokens,
    benchmark_label_for_mode,
    DraftCandidateBundle,
    GenerationEvidence,
    GenerationMode,
    NativeParallelGenerationEngine,
    ParallelDecodePlanner,
    supported_benchmark_labels,
)
from core.native.parallel_decode import JacobiDecoder
from core.native.qsg_parallel_kernels_wrapper import (
    NativeQSGRuntime,
    native_parallel_kernels_available,
    qsg_autoregressive_generate,
    qsg_block_diffusion_draft,
    qsg_hydra_head_draft,
    qsg_masked_diffusion_draft,
    qsg_medusa_head_draft,
)
from core.native.qsg_forward import QSGForwardPass
from core.native.runtime_telemetry import NativeGenerationTelemetry
from core.native.runtime_telemetry import build_runtime_capability_ledger
from core.native import simd_ops_wrapper as simd_ops
from core.native.weight_store import WeightStore
from core.qsg.runtime_contracts import (
    DeltaWatermark,
    DriftController,
    MemoryTierPolicy,
    PerformanceEnvelope,
    PerformanceTwinModel,
    RuntimeCapabilityVector,
    SpeculativeFrontierPolicy,
)

AnvilTokenizer = NativeTokenizer
SANCTIONED_BACKEND_PATH = (
    "prompt -> native tokenizer -> NativeQSGEngine -> NativeModelGraph -> "
    "C++ graph -> C++ QSG postprocess/sampling -> tokens"
)


def _kv_cache_quantization_mode() -> str:
    raw = str(os.getenv("ANVIL_KV_QUANT") or "").strip().lower()
    if raw in {"1", "true", "on", "q8", "int8"}:
        return "q8"
    return "fp32"


try:
    from core.native.mmap_weight_store import MMapWeightStore
except Exception:
    MMapWeightStore = None

try:
    from core.native.model_graph_wrapper import NativeModelGraph
except Exception:
    NativeModelGraph = None

try:
    from core.native.native_kv_cache_wrapper import NativeKVCacheWrapper
except Exception:
    NativeKVCacheWrapper = None

try:
    from core.native.cpu_speculative_decode import SSMSelfSpeculativeDecoder
except Exception:
    SSMSelfSpeculativeDecoder = None


def _env_flag_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    normalized = str(raw).strip().lower()
    return normalized not in {"0", "false", "no", "off"}


def _metadata_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "off", "no"}:
            return False
        if normalized in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def _classify_prompt_category(prompt: str) -> str:
    normalized = str(prompt or "").strip().lower()
    if not normalized:
        return "unknown"
    if any(
        marker in normalized
        for marker in (
            "```",
            "def ",
            "class ",
            "function",
            "refactor",
            "stack trace",
            "bug",
            "python",
            "javascript",
            "typescript",
            "c++",
            "code",
        )
    ):
        return "code"
    if any(
        marker in normalized
        for marker in ("json", "yaml", "schema", "table", "csv", "xml")
    ):
        return "structured"
    if any(
        marker in normalized
        for marker in ("explain", "analyze", "compare", "reason", "why")
    ):
        return "analysis"
    if "summarize" in normalized or "summary" in normalized:
        return "summarization"
    return "general"


def _temperature_band(value: float) -> str:
    temperature = float(value)
    if temperature <= 0.2:
        return "deterministic"
    if temperature <= 0.7:
        return "low"
    if temperature <= 1.0:
        return "medium"
    return "high"


@dataclass(slots=True)
class _DraftHeadConfig:
    kind: str
    num_heads: int = 0
    hidden_dim: int = 0
    vocab_size: int = 0
    weights: Any = None
    bias: Any = None

    @property
    def ready(self) -> bool:
        return (
            self.weights is not None
            and self.num_heads > 0
            and self.hidden_dim > 0
            and self.vocab_size > 1
        )


def _draft_head_index(name: str) -> int:
    matches = re.findall(r"(?:^|[._])(\d+)(?=[._]|$)", str(name).lower())
    if not matches:
        return -1
    return int(matches[-1])


def _normalize_draft_head_weight(
    weight: Any,
    *,
    hidden_dim: int,
    vocab_size: int,
) -> Any:
    arr = np.asarray(weight, dtype=np.float32)
    if arr.ndim != 2:
        return None
    if arr.shape == (vocab_size, hidden_dim):
        normalized = arr
    elif arr.shape == (hidden_dim, vocab_size):
        normalized = arr.T
    else:
        return None
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _normalize_draft_head_bias(bias: Any, *, vocab_size: int) -> Any:
    arr = np.asarray(bias, dtype=np.float32).reshape(-1)
    if arr.size < vocab_size:
        return None
    return np.ascontiguousarray(arr[:vocab_size], dtype=np.float32)


def _discover_draft_head_config(
    loader: GGUFModelLoader,
    *,
    kind: str,
    hidden_dim: int,
    vocab_size: int,
) -> _DraftHeadConfig:
    kind_lower = str(kind).strip().lower()
    weight_names: list[str] = []
    bias_names: list[str] = []
    try:
        tensors = list(getattr(loader.reader, "tensors", ()) or ())
    except Exception:
        tensors = []
    for tensor in tensors:
        name = str(getattr(tensor, "name", "") or "")
        lowered = name.lower()
        if kind_lower not in lowered:
            continue
        if lowered.endswith(".weight"):
            weight_names.append(name)
        elif lowered.endswith(".bias"):
            bias_names.append(name)
    if not weight_names:
        return _DraftHeadConfig(kind=kind_lower)

    weight_names = sorted(
        weight_names,
        key=lambda item: (_draft_head_index(item) < 0, _draft_head_index(item), item),
    )
    bias_by_index = {_draft_head_index(name): name for name in bias_names}

    weights: list[Any] = []
    biases: list[Any] = []
    any_bias = False
    for order, name in enumerate(weight_names):
        index = _draft_head_index(name)
        if index < 0:
            index = order
        try:
            weight = loader.get_tensor(name)
        except Exception:
            weight = None
        normalized_weight = _normalize_draft_head_weight(
            weight,
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )
        if normalized_weight is None:
            continue
        weights.append(normalized_weight)
        bias_name = bias_by_index.get(index)
        if bias_name is None:
            biases.append(np.zeros((vocab_size,), dtype=np.float32))
            continue
        try:
            bias = loader.get_tensor(bias_name)
        except Exception:
            bias = None
        normalized_bias = _normalize_draft_head_bias(bias, vocab_size=vocab_size)
        if normalized_bias is None:
            biases.append(np.zeros((vocab_size,), dtype=np.float32))
            continue
        any_bias = True
        biases.append(normalized_bias)

    if not weights:
        return _DraftHeadConfig(kind=kind_lower)

    weight_blob = np.ascontiguousarray(np.stack(weights, axis=0), dtype=np.float32)
    bias_blob = None
    if biases:
        bias_blob = np.ascontiguousarray(np.stack(biases, axis=0), dtype=np.float32)
        if not any_bias and not np.any(bias_blob):
            bias_blob = None
    return _DraftHeadConfig(
        kind=kind_lower,
        num_heads=int(weight_blob.shape[0]),
        hidden_dim=int(hidden_dim),
        vocab_size=int(vocab_size),
        weights=weight_blob.reshape(-1),
        bias=(None if bias_blob is None else bias_blob.reshape(-1)),
    )


def _build_weight_store(
    loader: GGUFModelLoader, profile: ModelProfile
) -> tuple[WeightStore, bool]:
    use_mmap_weights = _env_flag_enabled("ANVIL_NATIVE_USE_MMAP_WEIGHTS", default=True)
    if not use_mmap_weights:
        return WeightStore(loader, profile), False
    if MMapWeightStore is None:
        raise RuntimeError(
            "ANVIL_NATIVE_USE_MMAP_WEIGHTS is enabled but core.native.mmap_weight_store "
            "could not be imported."
        )
    return MMapWeightStore(loader, profile), True


def _allow_full_graph_for_architecture(architecture: str) -> bool:
    """Unified runtime policy: always prefer full C++ graph execution."""
    _ = architecture
    return True


def _allow_parallel_decode_for_architecture(architecture: str) -> bool:
    """Parallel decode is a native-wide policy, not an architecture allowlist."""
    _ = architecture
    return True


def _visible_logical_threads() -> int:
    logical = 0
    try:
        affinity = os.sched_getaffinity(0)
        if affinity:
            logical = int(len(affinity))
    except Exception:
        logical = 0
    if logical <= 0:
        logical = int(os.cpu_count() or 1)
    return max(1, logical)


def _physical_core_thread_hint(logical: int) -> int:
    logical = max(1, int(logical))
    if logical >= 8 and logical % 2 == 0:
        return max(1, logical // 2)
    return logical


def _granite_decode_thread_hint(logical: int) -> int:
    logical = max(1, int(logical))
    minimum = _auto_min_threads(logical)
    physical = _physical_core_thread_hint(logical)
    default_target = physical
    if logical > physical:
        smt_extra = max(0, logical - physical)
        default_target = max(1, min(logical, physical + max(1, smt_extra // 2)))
    default_reserve = max(0, logical - default_target)
    reserve = _decode_thread_headroom_reserve(logical, default_reserve=default_reserve)
    return _threads_from_reserve(logical, reserve=reserve, minimum=minimum)


def _auto_min_threads(logical: int) -> int:
    logical = max(1, int(logical))
    configured = _env_int("ANVIL_AUTO_MIN_THREADS")
    if configured is None:
        configured = 4
    return max(1, min(logical, int(configured)))


def _decode_thread_headroom_reserve(logical: int, *, default_reserve: int) -> int:
    logical = max(1, int(logical))
    configured = _env_int("ANVIL_NUM_THREADS_HEADROOM")
    reserve = default_reserve if configured is None else configured
    return max(0, min(logical, int(reserve)))


def _threads_from_reserve(logical: int, *, reserve: int, minimum: int) -> int:
    logical = max(1, int(logical))
    minimum = max(1, min(logical, int(minimum)))
    reserve = max(0, int(reserve))
    return max(minimum, min(logical, logical - reserve))


def _qwen_decode_thread_hint(
    logical: int,
    *,
    n_layers: int = 0,
    embedding_dim: int = 0,
) -> int:
    logical = max(1, int(logical))
    minimum = _auto_min_threads(logical)
    workload = max(0, int(n_layers)) * max(0, int(embedding_dim))
    default_reserve = 0
    if logical > 8:
        default_reserve = 1
        if logical >= 16:
            default_reserve = 4
        elif logical >= 12:
            default_reserve = 2
        # On very large hosts, let heavier Qwen decode consume more threads.
        if workload >= 130_000 and logical >= 24:
            default_reserve = min(default_reserve, 2)
    reserve = _decode_thread_headroom_reserve(logical, default_reserve=default_reserve)
    return _threads_from_reserve(logical, reserve=reserve, minimum=minimum)


def _auto_num_threads(
    architecture: str = "",
    *,
    n_layers: int = 0,
    embedding_dim: int = 0,
) -> int:
    """Detect a sane default decode thread count for inference."""
    logical = _visible_logical_threads()
    minimum = _auto_min_threads(logical)
    if _env_flag_enabled("ANVIL_NATIVE_USE_LOGICAL_THREADS", default=False):
        return max(minimum, logical)
    if _env_flag_enabled("ANVIL_NATIVE_USE_PHYSICAL_THREADS", default=False):
        return max(minimum, _physical_core_thread_hint(logical))
    arch = str(architecture or "").strip().lower()
    if arch == "granitehybrid":
        # Granite decode benefits from some SMT headroom on Ryzen-class CPUs,
        # but regresses badly when all logical threads are used.
        return _granite_decode_thread_hint(logical)
    if arch == "qwen35":
        return _qwen_decode_thread_hint(
            logical,
            n_layers=n_layers,
            embedding_dim=embedding_dim,
        )
    reserve = _decode_thread_headroom_reserve(logical, default_reserve=0)
    return _threads_from_reserve(logical, reserve=reserve, minimum=minimum)


def _auto_batch_threads(decode_threads: int, architecture: str = "") -> int:
    logical = _visible_logical_threads()
    minimum = _auto_min_threads(logical)
    decode = max(1, int(decode_threads))
    return max(minimum, min(logical, decode))


def _auto_num_ubatch(batch_threads: int, architecture: str = "") -> int:
    batch_threads = max(1, int(batch_threads))
    arch = str(architecture or "").strip().lower()
    if arch == "granitehybrid":
        if batch_threads >= 8:
            return 32
        if batch_threads >= 6:
            return 16
        return 8
    if arch == "qwen35":
        if batch_threads >= 12:
            return 32
        if batch_threads >= 8:
            return 16
        if batch_threads >= 4:
            return 8
        return 4
    if batch_threads >= 16:
        return 32
    if batch_threads >= 8:
        return 16
    return 8


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except Exception:
        return None


def _env_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return float(str(raw).strip())
    except Exception:
        return None


def _read_text_file(path: str) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_selected_kernel_mode(path: str) -> str:
    raw = _read_text_file(path)
    if not raw:
        return ""
    for token in raw.split("["):
        if "]" in token:
            selected, _, _ = token.partition("]")
            if selected.strip():
                return selected.strip()
    return raw.splitlines()[0].strip()


def _cpu_governor() -> str:
    governor = _read_text_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor:
        return governor
    return _read_selected_kernel_mode(
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
    )


def _perf_event_probe() -> tuple[bool, str]:
    paranoid = _read_text_file("/proc/sys/kernel/perf_event_paranoid")
    if paranoid:
        try:
            value = int(paranoid)
        except Exception:
            value = None
        else:
            if value >= 3:
                return False, f"blocked:perf_event_paranoid={value}"
            return True, f"available:perf_event_paranoid={value}"
    if shutil.which("perf") is None:
        return False, "blocked:perf_missing"
    return True, "available:unknown"


def _host_fingerprint() -> str:
    try:
        visible_threads = len(os.sched_getaffinity(0))
    except Exception:
        visible_threads = int(os.cpu_count() or 1)
    fields = (
        platform.node(),
        platform.machine(),
        platform.platform(),
        _cpu_governor(),
        str(visible_threads),
    )
    return "|".join(fields)


def _autotune_cache_path(model_name: str) -> pathlib.Path:
    safe_model = "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(model_name)
    )
    safe_host = "".join(
        ch if ch.isalnum() or ch in {"_", "-", "."} else "_"
        for ch in _host_fingerprint()
    )
    return (
        pathlib.Path(__file__).resolve().parents[2]
        / ".anvil"
        / "benchmarks"
        / "autotune"
        / safe_host
        / f"{safe_model}.json"
    )


def _load_autotune_profile(model_name: str) -> dict[str, Any]:
    path = _autotune_cache_path(model_name)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _cpu_mask_string(cpus: Sequence[int]) -> str:
    ordered = sorted({int(cpu) for cpu in cpus if int(cpu) >= 0})
    return ",".join(str(cpu) for cpu in ordered)


def _parse_json_object(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        payload = json.loads(str(raw))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _active_l3_domain_ids(
    topology: dict[str, Any], active_cpus: Sequence[int]
) -> list[int]:
    active = {int(cpu) for cpu in active_cpus if int(cpu) >= 0}
    if not active:
        return []
    matches: list[int] = []
    for domain in topology.get("l3_domains", []) or []:
        if not isinstance(domain, dict):
            continue
        cpus = {
            int(cpu)
            for cpu in (domain.get("cpus", []) or [])
            if isinstance(cpu, (int, float))
        }
        if active.intersection(cpus):
            try:
                matches.append(int(domain.get("id", len(matches))))
            except Exception:
                matches.append(len(matches))
    return sorted(set(matches))


def _timecrystal_mode_to_native(mode: object) -> int:
    normalized = str(mode or "").strip().lower()
    if normalized in {"telemetry", "observe", "off", "disabled"}:
        return 0
    if normalized in {"conservative", "damp_only", "damp", "safe"}:
        return 1
    return 2


def _timecrystal_native_mode_label(mode: int) -> str:
    if int(mode) <= 0:
        return "telemetry"
    if int(mode) == 1:
        return "conservative"
    return "aggressive"


def _configure_openmp_affinity(architecture: str) -> None:
    arch = str(architecture or "").strip().lower()
    default_proc_bind = "close"
    default_places = "cores"
    strict_numa = _env_flag_enabled("ANVIL_NUMA_STRICT", default=False)
    if arch == "granitehybrid":
        # Granite regresses badly on some Ryzen systems when libgomp binds the
        # process too tightly to a subset of cores.
        default_proc_bind = "false"
        default_places = "threads"
    if strict_numa:
        # Strict mode prioritizes deterministic CPU residency.
        default_proc_bind = "true"
        default_places = "cores"
        if os.getenv("ANVIL_NUMA_AFFINITY_MODE") is None:
            os.environ["ANVIL_NUMA_AFFINITY_MODE"] = "compact"
        if os.getenv("ANVIL_NUMA_FIRST_TOUCH") is None:
            os.environ["ANVIL_NUMA_FIRST_TOUCH"] = "1"
        if os.getenv("ANVIL_NUMA_MIGRATION_WATCHDOG") is None:
            os.environ["ANVIL_NUMA_MIGRATION_WATCHDOG"] = "1"
    if os.getenv("OMP_PROC_BIND") is None:
        os.environ["OMP_PROC_BIND"] = str(
            os.getenv("ANVIL_OMP_PROC_BIND", default_proc_bind)
        )
    if os.getenv("OMP_PLACES") is None:
        os.environ["OMP_PLACES"] = str(os.getenv("ANVIL_OMP_PLACES", default_places))
    if os.getenv("OMP_DYNAMIC") is None:
        os.environ["OMP_DYNAMIC"] = str(os.getenv("ANVIL_OMP_DYNAMIC", "FALSE"))
    if os.getenv("OMP_MAX_ACTIVE_LEVELS") is None:
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = str(
            os.getenv("ANVIL_OMP_MAX_ACTIVE_LEVELS", "1")
        )
    if strict_numa and os.getenv("ANVIL_NATIVE_PIN_THREADS") is None:
        os.environ["ANVIL_NATIVE_PIN_THREADS"] = "1"
    if not _env_flag_enabled("ANVIL_NATIVE_PIN_THREADS", default=False):
        return
    try:
        simd_ops.set_thread_affinity(use_p_cores_only=(arch == "granitehybrid"))
    except Exception:
        return


def _default_min_new_tokens_before_eos(architecture: str) -> int:
    arch = str(architecture or "").strip().lower()
    if "qwen" in arch:
        return 8
    return 0


def _native_context_profile_keys(profile: ModelProfile) -> tuple[str, str]:
    family = str(getattr(profile, "family", "") or "").strip().lower()
    arch = str(getattr(profile, "architecture", "") or "").strip().lower()
    if family == "qwen" or "qwen" in arch:
        return ("qwen35_native_ctx_default", "qwen35_native_ctx_cap")
    if family == "granite" or "granite" in arch:
        return ("granite4_native_ctx_default", "granite4_native_ctx_cap")
    return ("native_ctx_default", "native_ctx_cap")


def _resolve_native_context_length(
    requested_ctx: int,
    profile: ModelProfile,
    loader: GGUFModelLoader,
) -> tuple[int, int, int, int]:
    default_key, cap_key = _native_context_profile_keys(profile)
    global_default = int(GENERATION_PARAMS.get("native_ctx_default", 400000))
    profile_default = int(GENERATION_PARAMS.get(default_key, global_default))
    requested = int(requested_ctx)
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


def _matrix_output_dim(weight: Any, embedding_dim: int) -> int:
    if weight is None:
        return 0
    output_dim = int(getattr(weight, "output_dim", 0) or 0)
    if output_dim > 0:
        return output_dim
    shape = getattr(weight, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        return 0
    rows = int(shape[0] or 0)
    cols = int(shape[1] or 0)
    if rows <= 0 or cols <= 0:
        return 0
    if rows == embedding_dim and cols != embedding_dim:
        return cols
    if cols == embedding_dim and rows != embedding_dim:
        return rows
    return max(rows, cols)


def _infer_attention_layout(
    profile: ModelProfile,
    weight_store: WeightStore,
) -> tuple[int, int, int]:
    """Infer graph attention dimensions from loaded tensors."""
    n_heads = max(1, int(profile.n_heads or 1))
    embedding_dim = max(1, int(profile.embedding_dim))
    head_dim = max(1, embedding_dim // n_heads)
    n_kv_heads = max(1, int(profile.n_kv_heads or n_heads))

    get_layer_weights = getattr(weight_store, "get_layer_weights", None)
    if not callable(get_layer_weights):
        return n_heads, n_kv_heads, head_dim

    # Prefer concrete per-layer K projection dimensions when present.
    for layer_idx in range(int(profile.n_layers or 0)):
        try:
            weights = get_layer_weights(layer_idx)
        except Exception:
            continue
        if not isinstance(weights, dict):
            continue
        wk_out = _matrix_output_dim(weights.get("attn_k"), embedding_dim)
        if wk_out > 0 and wk_out % head_dim == 0:
            n_kv_heads = max(1, wk_out // head_dim)
            break

        # Fallback for fused QKV layouts: q + k + v.
        qkv_out = _matrix_output_dim(weights.get("attn_qkv"), embedding_dim)
        q_out = n_heads * head_dim
        residual = qkv_out - q_out
        if residual > 0 and residual % 2 == 0:
            kv_out = residual // 2
            if kv_out % head_dim == 0:
                n_kv_heads = max(1, kv_out // head_dim)
                break

    return n_heads, n_kv_heads, head_dim


def _sanitize_logits(logits: Sequence[float] | Any):
    if isinstance(logits, (array,)):
        return simd_ops.sanitize_logits_inplace(logits)
    if hasattr(logits, "ctypes"):
        return simd_ops.sanitize_logits_inplace(logits)
    return simd_ops.sanitize_logits(logits)


def _float_list(values: Sequence[float] | Any) -> list[float]:
    if isinstance(values, list):
        return [float(value) for value in values]
    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, list):
            return [float(value) for value in converted]
    return [float(value) for value in values]


def _zero_logits(vocab_size: int) -> list[float]:
    return [0.0] * max(0, int(vocab_size))


def _callable_supports_keyword_arg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _full_qsg_enabled(
    *,
    full_graph_enabled: bool,
    qsg_processors_native_enabled: bool,
    batched_prefill_native_enabled: bool,
    tokenizer_backend: str,
    sanctioned_backend_path: str,
) -> bool:
    return bool(
        full_graph_enabled
        and qsg_processors_native_enabled
        and batched_prefill_native_enabled
        and tokenizer_backend == "native"
        and sanctioned_backend_path == SANCTIONED_BACKEND_PATH
    )


class _NativeLLMCompat:
    def __init__(self, eos_token: int):
        self._eos_token = int(eos_token)

    def token_eos(self) -> int:
        return self._eos_token


class NativeQSGEngine:
    """GGUF-native engine with full vocabulary projection.

    Supports granite4 (granitehybrid) and qwen3.5 (qwen35) architectures natively.
    No llama-cpp dependency required.
    """

    def __init__(
        self,
        model_name: str,
        context_length: int = 8192,
        embedding: bool = False,
        **_: Any,
    ):
        init_started = time.perf_counter()
        resolved_contract = resolve_model_contract(model_name)
        canonical_model_name = resolved_contract.canonical_name
        self.contract = model_contract_snapshot(resolved_contract)
        self.contract["template"] = resolved_contract.template_name
        self.contract["strict_native_supported"] = True
        self.loader = GGUFModelLoader(canonical_model_name)
        self.profile = ModelProfile.from_loader(
            model_name=canonical_model_name, loader=self.loader
        )
        self.architecture = self.profile.architecture
        self.n_layer = self.profile.n_layers
        self.dim = self.profile.embedding_dim
        self._autotune_profile = _load_autotune_profile(canonical_model_name)
        self._autotune_profile_id = str(
            self._autotune_profile.get("profile_id", "")
            or self._autotune_profile.get("host_fingerprint", "")
        )
        self._autotune_source = "cache" if self._autotune_profile else "heuristic"
        self._autotune_score = float(self._autotune_profile.get("score", 0.0) or 0.0)
        self._autotune_exploration_count = int(
            self._autotune_profile.get("exploration_count", 0) or 0
        )
        env_threads = int(os.getenv("ANVIL_NUM_THREADS", "0") or "0")
        env_decode_threads = int(os.getenv("ANVIL_NUM_THREADS_DECODE", "0") or "0")
        env_batch_threads = int(os.getenv("ANVIL_NUM_THREADS_BATCH", "0") or "0")
        auto_decode_threads = _auto_num_threads(
            self.architecture,
            n_layers=self.n_layer,
            embedding_dim=self.dim,
        )
        auto_batch_threads = _auto_batch_threads(auto_decode_threads, self.architecture)
        if env_threads <= 0 and env_decode_threads <= 0:
            cached_decode_threads = int(
                self._autotune_profile.get("decode_threads", 0) or 0
            )
            if cached_decode_threads > 0:
                auto_decode_threads = cached_decode_threads
        if env_threads <= 0 and env_batch_threads <= 0:
            cached_batch_threads = int(
                self._autotune_profile.get("batch_threads", 0) or 0
            )
            if cached_batch_threads > 0:
                auto_batch_threads = cached_batch_threads
        base_threads = env_threads if env_threads > 0 else auto_decode_threads
        self.num_threads_decode = max(
            1, env_decode_threads if env_decode_threads > 0 else base_threads
        )
        self.num_threads_batch = max(
            1,
            (
                env_batch_threads
                if env_batch_threads > 0
                else (env_threads if env_threads > 0 else auto_batch_threads)
            ),
        )
        default_ubatch = _auto_num_ubatch(self.num_threads_batch, self.architecture)
        if os.getenv("ANVIL_NUM_UBATCH") is None:
            cached_ubatch = int(self._autotune_profile.get("ubatch", 0) or 0)
            if cached_ubatch > 0:
                default_ubatch = cached_ubatch
        self.num_ubatch = max(
            1,
            int(
                os.getenv("ANVIL_NUM_UBATCH", str(default_ubatch))
                or str(default_ubatch)
            ),
        )
        if os.getenv("ANVIL_NUM_THREADS_DECODE") is None:
            os.environ["ANVIL_NUM_THREADS_DECODE"] = str(self.num_threads_decode)
        if os.getenv("ANVIL_NUM_THREADS_BATCH") is None:
            os.environ["ANVIL_NUM_THREADS_BATCH"] = str(self.num_threads_batch)
        _configure_openmp_affinity(self.architecture)
        self._logical_core_count = _visible_logical_threads()
        try:
            self._physical_core_count = int(simd_ops.detect_physical_cores())
        except Exception:
            self._physical_core_count = 0
        if self._physical_core_count <= 1 and self._logical_core_count >= 4:
            self._physical_core_count = _physical_core_thread_hint(
                self._logical_core_count
            )
        elif self._physical_core_count <= 0:
            self._physical_core_count = _physical_core_thread_hint(
                self._logical_core_count
            )
        try:
            self._p_core_count = int(simd_ops.get_p_core_count())
        except Exception:
            self._p_core_count = 0
        try:
            simd_ops.refresh_topology()
        except Exception:
            pass
        try:
            self._affinity_mode = int(simd_ops.get_affinity_mode())
        except Exception:
            self._affinity_mode = 1
        try:
            self._l3_domain_count = int(simd_ops.get_l3_domain_count())
        except Exception:
            self._l3_domain_count = 0
        try:
            self._topology_json = str(simd_ops.export_topology_json() or "")
        except Exception:
            self._topology_json = ""
        try:
            self._affinity_plan_json = str(simd_ops.export_affinity_plan_json() or "")
        except Exception:
            self._affinity_plan_json = ""
        self._numa_strict = bool(_env_flag_enabled("ANVIL_NUMA_STRICT", default=False))
        self._numa_affinity_mode = str(os.getenv("ANVIL_NUMA_AFFINITY_MODE", "legacy"))
        self._numa_hugepage = str(os.getenv("ANVIL_NUMA_HUGEPAGE", "off"))
        self._numa_bind_policy = str(os.getenv("ANVIL_NUMA_BIND_POLICY", "none"))
        self._numa_first_touch = bool(
            _env_flag_enabled("ANVIL_NUMA_FIRST_TOUCH", default=False)
        )
        self._os_thread_migrations = 0
        self._os_last_cpu = -1
        self._refresh_os_thread_telemetry()
        self._affinity_policy = str(os.getenv("ANVIL_OMP_PROC_BIND", "close"))
        self._omp_places = str(os.getenv("OMP_PLACES", ""))
        self._omp_proc_bind = str(os.getenv("OMP_PROC_BIND", ""))
        try:
            self._omp_max_threads = int(simd_ops.get_omp_max_threads())
        except Exception:
            self._omp_max_threads = 0
        try:
            self._omp_dynamic = bool(simd_ops.get_omp_dynamic())
        except Exception:
            self._omp_dynamic = False
        try:
            self._omp_active_levels = int(simd_ops.get_omp_active_levels())
        except Exception:
            self._omp_active_levels = 0
        self._perf_event_access, self._perf_event_access_reason = _perf_event_probe()
        self._cpu_governor = _cpu_governor()
        self._thp_mode = _read_selected_kernel_mode(
            "/sys/kernel/mm/transparent_hugepage/enabled"
        )
        self._perf_counter_source = (
            "hw_pmu" if self._perf_event_access else "telemetry_only"
        )
        self._sanctioned_backend_path = SANCTIONED_BACKEND_PATH
        self._native_backend_requested = str(
            os.getenv("ANVIL_NATIVE_BACKEND_MODULE", "") or ""
        ).strip()
        self._native_backend_required = _env_flag_enabled(
            "ANVIL_REQUIRE_NATIVE_BACKEND_MODULE",
            default=True,
        )
        self._native_backend_info = get_native_backend_info(
            backend_name=self._native_backend_requested,
            model_name=canonical_model_name,
            architecture=self.architecture,
            family=str(getattr(self.profile, "family", "") or ""),
        )
        self._native_backend_module = str(
            self._native_backend_info.get("backend_module", "")
        )
        if self._native_backend_required and not bool(
            self._native_backend_info.get("backend_module_loaded", False)
        ):
            raise RuntimeError(
                "Strict native backend module wiring failed for "
                f"'{canonical_model_name}' "
                f"(backend='{self._native_backend_module or 'unknown'}'): "
                f"{self._native_backend_info.get('backend_module_error', 'unknown error')} "
                f"[selection={self._native_backend_info.get('backend_selection_source', 'unknown')}]"
            )
        self._granite_moe_mode = "disabled"
        if self.profile.architecture == "granitehybrid":
            routed_disabled = os.getenv("ANVIL_DISABLE_MOE_ROUTED", "0") == "1"
            shared_disabled = os.getenv("ANVIL_DISABLE_MOE_SHARED", "0") == "1"
            if routed_disabled and shared_disabled:
                self._granite_moe_mode = "disabled"
            elif routed_disabled:
                self._granite_moe_mode = "shared_only"
            elif shared_disabled:
                self._granite_moe_mode = "routed_only"
            else:
                self._granite_moe_mode = "shared+routed"
        (
            self.context_length,
            self._requested_context_length,
            self._model_context_limit,
            self._native_context_cap,
        ) = _resolve_native_context_length(
            int(context_length),
            self.profile,
            self.loader,
        )
        if self.context_length != int(context_length):
            print(
                "  [Native Context] Clamped context length "
                f"from {int(context_length)} to {self.context_length} "
                f"(model_limit={self._model_context_limit}, cap={self._native_context_cap})."
            )
        self.weight_store, self._use_mmap_weights = _build_weight_store(
            self.loader, self.profile
        )
        self.embedding_enabled = bool(embedding)
        if self.embedding_enabled:
            raise RuntimeError(
                "Strict native production mode is generation-only. "
                "Embedding and hidden-state APIs are disabled unless reimplemented natively."
            )

        # Keep the Python forward pass unavailable in strict production mode.
        self.forward_pass: Optional[QSGForwardPass] = None
        self.supports_token_api = True
        self.backend = "native_qsg"
        self.model_path = str(self.loader.model_path)
        self._strict_prompt_contract = get_strict_prompt_contract(
            str(self.contract.get("model") or canonical_model_name),
            strict=True,
        )
        expected_template = (
            str(
                self.contract.get("template_name")
                or self.contract.get("template")
                or ""
            )
            .strip()
            .lower()
        )
        actual_template = (
            str(getattr(self.profile, "chat_template", "") or "").strip().lower()
        )
        if (
            expected_template
            and expected_template != self._strict_prompt_contract.template_name
        ):
            raise RuntimeError(
                "Strict native prompt contract mismatch for "
                f"{self.contract.get('model')}: expected '{expected_template}', "
                f"resolved '{self._strict_prompt_contract.template_name}'."
            )
        if (
            actual_template
            and actual_template != self._strict_prompt_contract.template_name
        ):
            raise RuntimeError(
                "Model profile chat-template mismatch for strict native mode: "
                f"profile='{actual_template}', contract='{self._strict_prompt_contract.template_name}'."
            )
        self._special_tokens = self.loader.get_special_tokens()
        metadata = self.loader.get_metadata()
        add_bos_meta = metadata.get("tokenizer.ggml.add_bos_token")
        has_bos = "bos" in self._special_tokens
        if add_bos_meta is None:
            self._add_bos = has_bos
        else:
            self._add_bos = _metadata_bool(add_bos_meta, has_bos) and has_bos
        tokenizer_model = str(metadata.get("tokenizer.ggml.model", "") or "")
        tokenizer_pre = str(metadata.get("tokenizer.ggml.pre", "") or "")
        tokenizer_merges = None
        merges_getter = getattr(self.loader, "get_tokenizer_merges", None)
        if callable(merges_getter):
            tokenizer_merges = merges_getter()

        tokenizer_cls = globals().get("AnvilTokenizer")
        native_tokenizer_cls = globals().get("NativeTokenizer", NativeTokenizer)
        if tokenizer_cls is None or tokenizer_cls is AnvilTokenizer:
            tokenizer_cls = native_tokenizer_cls
        self.tokenizer = tokenizer_cls(
            self.loader.get_vocab_tokens(),
            self._special_tokens,
            model_path=self.model_path,
            bpe_merges=tokenizer_merges,
            tokenizer_model=tokenizer_model,
            tokenizer_pre=tokenizer_pre,
        )
        self._tokenizer_backend = "native"
        self._eos_token = int(
            self._special_tokens.get("eos", getattr(self.tokenizer, "eos_id", 2))
        )
        self._suppressed_token_ids = {
            int(token_id)
            for token_id in getattr(self.tokenizer, "decode_skip_ids", set())
            if 0 <= int(token_id) < int(self.profile.vocab_size)
        }
        self._suppressed_token_ids.discard(self._eos_token)
        disallowed_prefix_token_ids: set[int] = set()
        vocab_tokens = list(getattr(self.tokenizer, "vocab_tokens", ()) or ())
        vocab_lookup = {str(token): idx for idx, token in enumerate(vocab_tokens)}
        for prefix in getattr(
            self._strict_prompt_contract, "disallowed_output_prefixes", ()
        ):
            prefix_text = str(prefix or "")
            if not prefix_text:
                continue
            exact_vocab_id = vocab_lookup.get(prefix_text)
            if exact_vocab_id is not None and 0 <= int(exact_vocab_id) < int(
                self.profile.vocab_size
            ):
                disallowed_prefix_token_ids.add(int(exact_vocab_id))
            encoded_prefix = self.tokenizer.encode(
                prefix_text,
                add_bos=False,
                add_eos=False,
            )
            if len(encoded_prefix) == 1:
                disallowed_prefix_token_ids.add(int(encoded_prefix[0]))
        disallowed_prefix_token_ids.discard(self._eos_token)
        self._leading_disallowed_token_ids = tuple(sorted(disallowed_prefix_token_ids))
        self._suppressed_token_ids_sorted = tuple(sorted(self._suppressed_token_ids))
        self._suppressed_token_ids_buffer = array(
            "i", self._suppressed_token_ids_sorted
        )
        self._leading_suppressed_token_ids_sorted = tuple(
            sorted(self._suppressed_token_ids | disallowed_prefix_token_ids)
        )
        self._leading_suppressed_token_ids_buffer = array(
            "i", self._leading_suppressed_token_ids_sorted
        )
        self._json_fastlane_open_ids = array("i")
        self._json_fastlane_object_follow_ids = array("i")
        try:
            json_open_ids = self.tokenizer.encode("{", add_bos=False, add_eos=False)
            json_close_ids = self.tokenizer.encode("}", add_bos=False, add_eos=False)
            json_quote_ids = self.tokenizer.encode('"', add_bos=False, add_eos=False)
            if len(json_open_ids) == 1:
                self._json_fastlane_open_ids = array("i", [int(json_open_ids[0])])
            object_follow = {
                int(token_id)
                for token_id in (
                    list(json_close_ids if len(json_close_ids) == 1 else [])
                    + list(json_quote_ids if len(json_quote_ids) == 1 else [])
                )
                if 0 <= int(token_id) < int(self.profile.vocab_size)
            }
            self._json_fastlane_object_follow_ids = array("i", sorted(object_follow))
        except Exception:
            self._json_fastlane_open_ids = array("i")
            self._json_fastlane_object_follow_ids = array("i")
        self.llm = _NativeLLMCompat(self._eos_token)
        self._jacobi = JacobiDecoder(
            width=int(PERFORMANCE_CONFIG.get("parallel_width", 4))
        )
        self._force_parallel_decode = (
            os.getenv("ANVIL_FORCE_PARALLEL_DECODE", "0") == "1"
        )
        self._forbid_autoregressive_fallback = _env_flag_enabled(
            "ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK",
            default=False,
        )
        self._parallel_decode_allowed = _allow_parallel_decode_for_architecture(
            self.architecture
        )
        self._parallel_decode_min_new_tokens = int(
            os.getenv("ANVIL_PARALLEL_DECODE_MIN_NEW_TOKENS", "32") or "32"
        )
        self._parallel_decode_min_prompt_tokens = int(
            os.getenv("ANVIL_PARALLEL_DECODE_MIN_PROMPT_TOKENS", "64") or "64"
        )
        self._parallel_prompt_lookup_enabled = _env_flag_enabled(
            "ANVIL_PARALLEL_PROMPT_LOOKUP_ENABLED",
            default=bool(
                PERFORMANCE_CONFIG.get("parallel_prompt_lookup_enabled", True)
            ),
        )
        self._parallel_jacobi_lookahead_enabled = _env_flag_enabled(
            "ANVIL_PARALLEL_JACOBI_LOOKAHEAD_ENABLED",
            default=bool(
                PERFORMANCE_CONFIG.get("parallel_jacobi_lookahead_enabled", True)
            ),
        )
        self._parallel_ssd_bridge_enabled = _env_flag_enabled(
            "ANVIL_PARALLEL_SSD_BRIDGE_ENABLED",
            default=bool(PERFORMANCE_CONFIG.get("parallel_ssd_bridge_enabled", True)),
        )
        self._self_spec_exit_layer_min = max(
            1,
            int(
                os.getenv(
                    "ANVIL_SELF_SPEC_EXIT_LAYER_MIN",
                    str(max(1, self.n_layer // 4)),
                )
                or max(1, self.n_layer // 4)
            ),
        )
        self._self_spec_exit_layer_max = max(
            self._self_spec_exit_layer_min,
            int(
                os.getenv(
                    "ANVIL_SELF_SPEC_EXIT_LAYER_MAX",
                    str(max(1, self.n_layer - 1)),
                )
                or max(1, self.n_layer - 1)
            ),
        )
        forced_exit_layer = _env_int("ANVIL_SELF_SPEC_FORCE_EXIT_LAYER")
        self._self_spec_force_exit_layer = (
            None if forced_exit_layer is None else int(forced_exit_layer)
        )
        self._self_spec_native_supported = False
        self._parallel_replacement_enabled = _env_flag_enabled(
            "ANVIL_PARALLEL_REPLACEMENT_ENABLED",
            default=bool(PERFORMANCE_CONFIG.get("parallel_replacement_enabled", True)),
        )
        self._parallel_ar_recovery_enabled = _env_flag_enabled(
            "ANVIL_PARALLEL_AR_RECOVERY_ENABLED",
            default=bool(PERFORMANCE_CONFIG.get("parallel_ar_recovery_enabled", True)),
        )
        if self._forbid_autoregressive_fallback:
            self._parallel_ar_recovery_enabled = False
        self._parallel_prompt_lookup_window = int(
            os.getenv("ANVIL_PARALLEL_PROMPT_LOOKUP_WINDOW", "6") or "6"
        )
        self._parallel_prompt_lookup_min_ngram = int(
            os.getenv("ANVIL_PARALLEL_PROMPT_LOOKUP_MIN_NGRAM", "2") or "2"
        )
        self._parallel_prompt_lookup_max_ngram = int(
            os.getenv("ANVIL_PARALLEL_PROMPT_LOOKUP_MAX_NGRAM", "24") or "24"
        )
        self._parallel_prompt_lookup_accept_prob = float(
            os.getenv("ANVIL_PARALLEL_PROMPT_LOOKUP_ACCEPT_PROB", "0.18") or "0.18"
        )
        self._parallel_replacement_max_tree_width = int(
            os.getenv(
                "ANVIL_PARALLEL_REPLACEMENT_MAX_TREE_WIDTH",
                str(
                    int(
                        PERFORMANCE_CONFIG.get("parallel_replacement_max_tree_width", 4)
                    )
                ),
            )
            or "4"
        )
        self._parallel_replacement_acceptance_floor = float(
            os.getenv(
                "ANVIL_PARALLEL_REPLACEMENT_ACCEPTANCE_FLOOR",
                str(
                    float(
                        PERFORMANCE_CONFIG.get(
                            "parallel_replacement_acceptance_floor", 0.20
                        )
                    )
                ),
            )
            or "0.20"
        )
        self._parallel_replacement_max_draft_tokens = int(
            os.getenv(
                "ANVIL_PARALLEL_REPLACEMENT_MAX_DRAFT_TOKENS",
                str(
                    int(
                        PERFORMANCE_CONFIG.get(
                            "parallel_replacement_max_draft_tokens", 6
                        )
                    )
                ),
            )
            or "6"
        )
        self._parallel_replacement_top_k = int(
            os.getenv(
                "ANVIL_PARALLEL_REPLACEMENT_TOP_K",
                str(int(PERFORMANCE_CONFIG.get("parallel_replacement_top_k", 8))),
            )
            or "8"
        )
        self._medusa_head_enabled = _env_flag_enabled(
            "ANVIL_MEDUSA_HEAD_ENABLED",
            default=bool(PERFORMANCE_CONFIG.get("medusa_head_enabled", True)),
        )
        self._hydra_head_enabled = _env_flag_enabled(
            "ANVIL_HYDRA_HEAD_ENABLED",
            default=bool(PERFORMANCE_CONFIG.get("hydra_head_enabled", True)),
        )
        self._medusa_head_max_draft_tokens = int(
            os.getenv(
                "ANVIL_MEDUSA_HEAD_MAX_DRAFT_TOKENS",
                str(int(PERFORMANCE_CONFIG.get("medusa_head_max_draft_tokens", 4))),
            )
            or "4"
        )
        self._hydra_head_max_draft_tokens = int(
            os.getenv(
                "ANVIL_HYDRA_HEAD_MAX_DRAFT_TOKENS",
                str(int(PERFORMANCE_CONFIG.get("hydra_head_max_draft_tokens", 4))),
            )
            or "4"
        )
        self._medusa_head_top_k = int(
            os.getenv(
                "ANVIL_MEDUSA_HEAD_TOP_K",
                str(int(PERFORMANCE_CONFIG.get("medusa_head_top_k", 8))),
            )
            or "8"
        )
        self._hydra_head_top_k = int(
            os.getenv(
                "ANVIL_HYDRA_HEAD_TOP_K",
                str(int(PERFORMANCE_CONFIG.get("hydra_head_top_k", 8))),
            )
            or "8"
        )
        self._medusa_head_acceptance_floor = float(
            os.getenv(
                "ANVIL_MEDUSA_HEAD_ACCEPTANCE_FLOOR",
                str(
                    float(PERFORMANCE_CONFIG.get("medusa_head_acceptance_floor", 0.20))
                ),
            )
            or "0.20"
        )
        self._hydra_head_acceptance_floor = float(
            os.getenv(
                "ANVIL_HYDRA_HEAD_ACCEPTANCE_FLOOR",
                str(float(PERFORMANCE_CONFIG.get("hydra_head_acceptance_floor", 0.22))),
            )
            or "0.22"
        )
        self._hydra_head_blend_alpha = float(
            os.getenv(
                "ANVIL_HYDRA_HEAD_BLEND_ALPHA",
                str(float(PERFORMANCE_CONFIG.get("hydra_head_blend_alpha", 0.55))),
            )
            or "0.55"
        )
        self._native_parallel_kernels_ready = bool(native_parallel_kernels_available())
        if not self._native_parallel_kernels_ready:
            raise RuntimeError(
                "Strict native QSG requires qsg_parallel_kernels symbols but they are unavailable."
            )
        self._medusa_head_config = _discover_draft_head_config(
            self.loader,
            kind="medusa",
            hidden_dim=int(self.profile.embedding_dim),
            vocab_size=int(self.profile.vocab_size),
        )
        self._hydra_head_config = _discover_draft_head_config(
            self.loader,
            kind="hydra",
            hidden_dim=int(self.profile.embedding_dim),
            vocab_size=int(self.profile.vocab_size),
        )
        self._medusa_head_ready = bool(
            self._native_parallel_kernels_ready
            and self._medusa_head_enabled
            and self._medusa_head_config.ready
        )
        self._hydra_head_ready = bool(
            self._native_parallel_kernels_ready
            and self._hydra_head_enabled
            and self._hydra_head_config.ready
        )
        self._block_diffusion_enabled = _env_flag_enabled(
            "ANVIL_BLOCK_DIFFUSION_ENABLED",
            default=bool(PERFORMANCE_CONFIG.get("block_diffusion_enabled", True)),
        )
        self._block_diffusion_force = _env_flag_enabled(
            "ANVIL_BLOCK_DIFFUSION_FORCE",
            default=bool(PERFORMANCE_CONFIG.get("block_diffusion_force", False)),
        )
        self._block_diffusion_min_new_tokens = int(
            os.getenv(
                "ANVIL_BLOCK_DIFFUSION_MIN_NEW_TOKENS",
                str(int(PERFORMANCE_CONFIG.get("block_diffusion_min_new_tokens", 96))),
            )
            or "96"
        )
        self._block_diffusion_min_prompt_tokens = int(
            os.getenv(
                "ANVIL_BLOCK_DIFFUSION_MIN_PROMPT_TOKENS",
                str(
                    int(
                        PERFORMANCE_CONFIG.get("block_diffusion_min_prompt_tokens", 256)
                    )
                ),
            )
            or "256"
        )
        self._block_diffusion_block_size_tokens = int(
            os.getenv(
                "ANVIL_BLOCK_DIFFUSION_BLOCK_SIZE",
                str(
                    int(PERFORMANCE_CONFIG.get("block_diffusion_block_size_tokens", 16))
                ),
            )
            or "16"
        )
        self._block_diffusion_denoise_iterations = int(
            os.getenv(
                "ANVIL_BLOCK_DIFFUSION_DENOISE_ITERATIONS",
                str(
                    int(PERFORMANCE_CONFIG.get("block_diffusion_denoise_iterations", 2))
                ),
            )
            or "2"
        )
        self._block_diffusion_acceptance_floor = float(
            os.getenv(
                "ANVIL_BLOCK_DIFFUSION_ACCEPTANCE_FLOOR",
                str(
                    float(
                        PERFORMANCE_CONFIG.get("block_diffusion_acceptance_floor", 0.18)
                    )
                ),
            )
            or "0.18"
        )
        self._block_diffusion_native_ready = bool(
            self._native_parallel_kernels_ready and self._block_diffusion_enabled
        )
        self._masked_diffusion_enabled = _env_flag_enabled(
            "ANVIL_MASKED_DIFFUSION_ENABLED",
            default=bool(PERFORMANCE_CONFIG.get("masked_diffusion_enabled", False)),
        )
        self._masked_diffusion_force = _env_flag_enabled(
            "ANVIL_MASKED_DIFFUSION_FORCE",
            default=bool(PERFORMANCE_CONFIG.get("masked_diffusion_force", False)),
        )
        self._masked_diffusion_block_size_tokens = int(
            os.getenv(
                "ANVIL_MASKED_DIFFUSION_BLOCK_SIZE",
                str(
                    int(
                        PERFORMANCE_CONFIG.get("masked_diffusion_block_size_tokens", 16)
                    )
                ),
            )
            or "16"
        )
        self._masked_diffusion_mask_stride = int(
            os.getenv(
                "ANVIL_MASKED_DIFFUSION_MASK_STRIDE",
                str(int(PERFORMANCE_CONFIG.get("masked_diffusion_mask_stride", 2))),
            )
            or "2"
        )
        self._masked_diffusion_denoise_iterations = int(
            os.getenv(
                "ANVIL_MASKED_DIFFUSION_DENOISE_ITERATIONS",
                str(
                    int(
                        PERFORMANCE_CONFIG.get(
                            "masked_diffusion_denoise_iterations",
                            2,
                        )
                    )
                ),
            )
            or "2"
        )
        self._masked_diffusion_acceptance_floor = float(
            os.getenv(
                "ANVIL_MASKED_DIFFUSION_ACCEPTANCE_FLOOR",
                str(
                    float(
                        PERFORMANCE_CONFIG.get(
                            "masked_diffusion_acceptance_floor",
                            0.18,
                        )
                    )
                ),
            )
            or "0.18"
        )
        self._masked_diffusion_native_ready = bool(
            self._native_parallel_kernels_ready and self._masked_diffusion_enabled
        )
        self._scheduler_queue_wait_ms = 0.0
        self._scheduler_iteration_ms = 0.0
        self._scheduler_kv_fragmentation_ratio = 0.0
        self._scheduler_kv_pages_total = 0
        self._scheduler_kv_pages_in_use = 0
        self._scheduler_kv_active_page_slots = 0
        self._scheduler_kv_shared_page_slots = 0
        self._scheduler_kv_snapshot_count = 0
        self._scheduler_kv_cow_events = 0
        self._scheduler_kv_prefix_share_events = 0
        self._scheduler_kv_active_tokens = 0
        self._scheduler_kv_committed_token_capacity = 0
        self._scheduler_kv_page_tokens = 0
        self._parallel_planner = ParallelDecodePlanner(self)
        min_new_tokens = _env_int("ANVIL_MIN_NEW_TOKENS_BEFORE_EOS")
        if min_new_tokens is None:
            min_new_tokens = _default_min_new_tokens_before_eos(self.architecture)
        self._min_new_tokens_before_eos = max(0, int(min_new_tokens))
        self._native_fast_path = _env_flag_enabled(
            "ANVIL_NATIVE_FAST_PATH", default=True
        )
        disable_logits_processors = bool(
            _env_flag_enabled(
                "ANVIL_DISABLE_LOGITS_PROCESSORS",
                default=False,
            )
        )
        disable_token_penalties = bool(
            _env_flag_enabled(
                "ANVIL_DISABLE_TOKEN_PENALTIES",
                default=False,
            )
        )
        if disable_logits_processors:
            raise RuntimeError(
                "Strict native QSG no longer allows disabling logits/QSG processors on the sanctioned path."
            )
        if disable_token_penalties:
            raise RuntimeError(
                "Strict native QSG no longer allows disabling native token penalties on the sanctioned path."
            )
        self._disable_logits_processors = False
        self._disable_token_penalties = False
        self._qsg_processors_native_enabled = True
        default_use_coconut = bool(
            getattr(self.profile, "coconut_mode", "logits_proxy") != "disabled"
        )
        self._native_qsg_use_coconut = _env_flag_enabled(
            "ANVIL_NATIVE_QSG_USE_COCONUT",
            default=default_use_coconut,
        )
        self._native_qsg_use_grover = _env_flag_enabled(
            "ANVIL_NATIVE_QSG_USE_GROVER",
            default=True,
        )
        self._native_qsg_coconut_alpha = float(
            getattr(self.profile, "coconut_alpha", 0.0) or 0.0
        )
        self._native_qsg_coconut_paths = int(
            max(1, int(getattr(self.profile, "coconut_paths", 1) or 1))
        )
        self._native_qsg_grover_top_k = int(
            max(0, int(getattr(self.profile, "grover_top_k", 0) or 0))
        )
        self._native_qsg_grover_damping = float(
            getattr(self.profile, "grover_damping", 0.0) or 0.0
        )
        if self.architecture == "qwen35":
            # Keep full native QSG enabled for Qwen, but cap deterministic
            # amplification so the recurrent branch does not collapse into loops.
            self._native_qsg_coconut_alpha = min(
                self._native_qsg_coconut_alpha,
                0.10,
            )
            self._native_qsg_grover_top_k = min(
                max(1, self._native_qsg_grover_top_k),
                4,
            )
            self._native_qsg_grover_damping = min(
                self._native_qsg_grover_damping,
                0.20,
            )
        native_no_repeat_ngram = _env_int("ANVIL_NATIVE_NO_REPEAT_NGRAM_SIZE")
        if native_no_repeat_ngram is None:
            native_no_repeat_ngram = 4 if self.architecture == "qwen35" else 0
        self._native_no_repeat_ngram_size = max(0, int(native_no_repeat_ngram))
        self._parallel_decode_disable_reason = ""
        self._active_thread_count = 0
        self._active_thread_mode = "decode"
        self._runtime_thread_switches = 0
        self._batch_token_fallback_count = 0
        self._coherence_guard_events = 0
        self._set_runtime_thread_mode("decode")
        self.num_threads = max(1, simd_ops.get_num_threads())
        self._cached_logits_tokens: list[int] = []
        self._cached_logits: Optional[list[float]] = None
        self._prefix_logits_cache: dict[tuple[int, ...], list[float]] = {}
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._prompt_cache_reused_tokens = 0
        self._prefill_chunk_count = 0
        self._last_prompt_format_seconds = 0.0
        self._last_tokenize_seconds = 0.0
        self._last_embedding_lookup_seconds = 0.0
        self._last_graph_prefill_seconds = 0.0
        self._last_graph_decode_seconds = 0.0
        self._last_sample_seconds = 0.0
        self._last_logits_processor_seconds = 0.0
        self._last_penalty_seconds = 0.0
        self._last_suppression_seconds = 0.0
        self._last_graph_prefill_calls = 0
        self._last_graph_decode_calls = 0
        self._last_sample_calls = 0
        self._last_logits_processor_calls = 0
        self._last_penalty_calls = 0
        self._last_suppression_calls = 0
        self._graph_token_id_enabled = True
        self._graph_batch_token_id_enabled = True
        self._graph_mode = _env_flag_enabled("ANVIL_NATIVE_GRAPH_MODE", default=True)
        self._strict_cpp_only = _env_flag_enabled(
            "ANVIL_NATIVE_STRICT_CPP_ONLY",
            default=True,
        )
        tc_enabled_default = bool(
            PERFORMANCE_CONFIG.get("timecrystal_context_stabilizer", True)
        )
        tc_mode_default = str(
            PERFORMANCE_CONFIG.get("timecrystal_mode", "aggressive_staged")
        )
        self._tc_enabled = _env_flag_enabled(
            "ANVIL_TC_STABILIZER",
            default=tc_enabled_default,
        )
        self._tc_base_mode_native = _timecrystal_mode_to_native(
            os.getenv("ANVIL_TC_MODE", tc_mode_default)
        )
        if not self._tc_enabled:
            self._tc_base_mode_native = 0
        self._tc_current_mode_native = int(self._tc_base_mode_native)
        self._tc_current_mode = _timecrystal_native_mode_label(
            self._tc_current_mode_native
        )
        self._tc_block_size_tokens = max(
            1,
            int(
                _env_int("ANVIL_TC_BLOCK_SIZE")
                or PERFORMANCE_CONFIG.get("timecrystal_block_size_tokens", 128)
            ),
        )
        self._tc_update_interval_tokens = max(
            1,
            int(
                _env_int("ANVIL_TC_UPDATE_INTERVAL")
                or PERFORMANCE_CONFIG.get("timecrystal_update_interval_tokens", 64)
            ),
        )
        self._tc_prune_interval_tokens = max(
            1,
            int(PERFORMANCE_CONFIG.get("timecrystal_prune_interval_tokens", 128)),
        )
        self._tc_preserve_head_tokens = max(
            0,
            int(PERFORMANCE_CONFIG.get("timecrystal_preserve_head_tokens", 256)),
        )
        self._tc_preserve_recent_tokens = max(
            0,
            int(
                _env_int("ANVIL_TC_PRESERVE_RECENT")
                or PERFORMANCE_CONFIG.get("timecrystal_preserve_recent_tokens", 8192)
            ),
        )
        self._tc_min_active_tokens = max(
            0,
            int(PERFORMANCE_CONFIG.get("timecrystal_min_active_tokens", 16384)),
        )
        tc_damp_threshold_env = _env_float("ANVIL_TC_DAMP_THRESHOLD")
        tc_prune_threshold_env = _env_float("ANVIL_TC_PRUNE_THRESHOLD")
        tc_overhead_target_pct_env = _env_float("ANVIL_TC_OVERHEAD_TARGET_PCT")
        tc_overhead_max_pct_env = _env_float("ANVIL_TC_OVERHEAD_MAX_PCT")
        self._tc_damp_threshold = float(
            tc_damp_threshold_env
            if tc_damp_threshold_env is not None
            else PERFORMANCE_CONFIG.get("timecrystal_damp_threshold", 0.35)
        )
        self._tc_prune_threshold = float(
            tc_prune_threshold_env
            if tc_prune_threshold_env is not None
            else PERFORMANCE_CONFIG.get("timecrystal_prune_threshold", 0.72)
        )
        self._tc_damping_strength = float(
            PERFORMANCE_CONFIG.get("timecrystal_damping_strength", 1.2)
        )
        self._tc_hysteresis = float(
            PERFORMANCE_CONFIG.get("timecrystal_hysteresis", 0.05)
        )
        self._tc_overhead_target_pct = max(
            0.0,
            float(
                tc_overhead_target_pct_env
                if tc_overhead_target_pct_env is not None
                else PERFORMANCE_CONFIG.get("timecrystal_overhead_target_pct", 15.0)
            ),
        )
        self._tc_overhead_max_pct = max(
            self._tc_overhead_target_pct,
            float(
                tc_overhead_max_pct_env
                if tc_overhead_max_pct_env is not None
                else PERFORMANCE_CONFIG.get("timecrystal_overhead_max_pct", 20.0)
            ),
        )
        self._tc_control_interval_tokens = max(
            1,
            int(PERFORMANCE_CONFIG.get("timecrystal_control_interval_tokens", 64)),
        )
        self._tc_overhead_window = max(
            self._tc_control_interval_tokens,
            int(PERFORMANCE_CONFIG.get("timecrystal_overhead_window_tokens", 128)),
        )
        self._tc_recovery_interval_tokens = max(
            self._tc_control_interval_tokens,
            int(PERFORMANCE_CONFIG.get("timecrystal_recovery_tokens", 256)),
        )
        self._tc_decode_steps = 0
        self._tc_recovery_steps = 0
        self._tc_overhead_samples: list[float] = []
        self._tc_last_overhead_percent = 0.0
        self._tc_prev_stabilizer_seconds = 0.0
        self._tc_prev_stabilizer_calls = 0
        self._tc_stabilizer_seconds_total = 0.0
        self._tc_stabilizer_calls_total = 0
        self._tc_last_snapshot: dict[str, float | int] = {}
        self._tc_last_snapshot_valid = False
        self._tc_auto_downgrade_events = 0
        self._tc_graph_drift_available = False
        # Kept for compatibility with older tests and introspection; runtime no
        # longer dispatches through Python hybrid execution.
        self._hybrid_mode = False
        self._model_graph: Optional[NativeModelGraph] = None
        self._hybrid_cpp_layer_mask: list[bool] = [False] * self.n_layer
        self._hybrid_cpp_full_layer_mask: list[bool] = [False] * self.n_layer
        self._hybrid_python_ffn_after_cpp_mask: list[bool] = [False] * self.n_layer
        self._hybrid_layer_weights_cache: dict[int, dict[str, Any]] = {}
        graph_max_seq = int(
            os.getenv("ANVIL_GRAPH_MAX_SEQ", str(self.context_length))
            or str(self.context_length)
        )
        graph_n_heads, graph_n_kv_heads, graph_head_dim = _infer_attention_layout(
            self.profile, self.weight_store
        )
        if graph_n_kv_heads != int(self.profile.n_kv_heads or self.profile.n_heads):
            print(
                "  [C++ Graph Mode] Adjusted KV-head layout from profile metadata "
                f"to tensor-backed values (n_heads={graph_n_heads}, "
                f"n_kv_heads={graph_n_kv_heads}, head_dim={graph_head_dim})."
            )
        metadata = self.loader.get_metadata()
        rms_eps = float(
            metadata.get(f"{self.architecture}.attention.layer_norm_rms_epsilon", 1e-5)
            or 1e-5
        )
        rope_theta = float(
            metadata.get(f"{self.architecture}.rope.freq_base", 10000.0) or 10000.0
        )

        graph_init_error: Optional[Exception] = None
        if self._graph_mode and NativeModelGraph is not None:
            try:
                graph = NativeModelGraph(
                    n_layers=self.profile.n_layers,
                    embedding_dim=self.profile.embedding_dim,
                    vocab_size=self.profile.vocab_size,
                    n_heads=graph_n_heads,
                    n_kv_heads=graph_n_kv_heads,
                    head_dim=graph_head_dim,
                    max_seq=graph_max_seq,
                    rms_eps=rms_eps,
                    rope_theta=rope_theta,
                    weight_store=self.weight_store,
                    profile=self.profile,
                )
                if graph.has_full_graph:
                    self._model_graph = graph
                    self._tc_graph_drift_available = (
                        self._apply_timecrystal_drift_config()
                    )
                    non_cpp_layers: list[int] = []
                    can_cpp_full_fn = getattr(graph, "can_layer_cpp_full", None)
                    can_cpp_fn = getattr(graph, "can_layer_cpp", None)
                    for i in range(self.n_layer):
                        if callable(can_cpp_full_fn):
                            can_cpp_full = bool(can_cpp_full_fn(i))
                        elif callable(can_cpp_fn):
                            can_cpp_full = bool(can_cpp_fn(i))
                        else:
                            can_cpp_full = True
                        self._hybrid_cpp_layer_mask[i] = can_cpp_full
                        self._hybrid_cpp_full_layer_mask[i] = can_cpp_full
                        self._hybrid_python_ffn_after_cpp_mask[i] = False
                        if not can_cpp_full:
                            non_cpp_layers.append(i)

                    if self._strict_cpp_only and non_cpp_layers:
                        preview = ",".join(str(idx) for idx in non_cpp_layers[:12])
                        if len(non_cpp_layers) > 12:
                            preview += ",..."
                        raise RuntimeError(
                            "Strict native C++ mode requires full graph coverage for "
                            f"all layers. Missing C++ coverage in layers [{preview}] "
                            f"for architecture '{self.architecture}'."
                        )
                    print(
                        f"  [C++ Graph Mode] Full graph execution enabled "
                        f"(arch={self.architecture}, max_seq={graph_max_seq}, "
                        f"strict_cpp_only={self._strict_cpp_only})"
                    )
                else:
                    graph.close()
            except Exception as exc:
                if "graph" in locals():
                    try:
                        graph.close()
                    except Exception:
                        pass
                self._model_graph = None
                graph_init_error = exc

        if self._model_graph is None:
            if graph_init_error is not None:
                raise RuntimeError(
                    "Native-only QSG requires full NativeModelGraph initialization; "
                    f"failed for architecture '{self.architecture}'."
                ) from graph_init_error
            raise RuntimeError(
                "Native-only QSG requires NativeModelGraph full-graph execution, "
                "but no graph backend is available."
            )
        self._self_spec_native_supported = bool(
            self._model_graph is not None
            and getattr(self._model_graph, "supports_exit_continuation", False)
        )

        # Native KV cache with Flash Attention
        # Skip when graph mode is active — graph has its own paged KV cache
        self._native_kv_cache = None
        if NativeKVCacheWrapper is not None and not (
            self._model_graph and self._model_graph.has_full_graph
        ):
            try:
                nkv = NativeKVCacheWrapper(
                    profile=self.profile, max_seq_len=graph_max_seq
                )
                if nkv.available:
                    self._native_kv_cache = nkv
                    has_flash = "+ Flash Attention" if nkv.has_flash_attention else ""
                    print(f"  [Native KV Cache] Enabled {has_flash}")
            except Exception:
                self._native_kv_cache = None

        self._ssm_spec_decoder = None
        self._self_spec_max_draft_length = int(
            os.getenv("ANVIL_SSM_SPEC_DRAFT_LEN", "4") or "4"
        )
        self._self_spec_acceptance_threshold = float(
            os.getenv("ANVIL_SSM_SPEC_ACCEPTANCE", "0.7") or "0.7"
        )
        # Speculative decode is disabled by default. Enable with ANVIL_ENABLE_SSM_SELF_SPEC=1.
        # It requires a working logits path first; enabling it on top of NaN-producing paths
        # causes cascading failures.
        enable_self_spec = _env_flag_enabled(
            "ANVIL_ENABLE_SSM_SELF_SPEC",
            default=False,
        )
        if enable_self_spec and SSMSelfSpeculativeDecoder is not None:
            try:
                self._ssm_spec_decoder = SSMSelfSpeculativeDecoder(
                    native_engine=self,
                    max_draft_length=self._self_spec_max_draft_length,
                    acceptance_threshold=self._self_spec_acceptance_threshold,
                )
            except Exception:
                self._ssm_spec_decoder = None
        if self._self_spec_native_supported:
            self._ssm_spec_decoder = None

        print(
            f"NativeQSGEngine loaded: {model_name} "
            f"(arch={self.architecture}, layers={self.n_layer}, dim={self.dim}, "
            f"vocab={self.profile.vocab_size}, threads={self.num_threads}, "
            f"decode_threads={self.num_threads_decode}, batch_threads={self.num_threads_batch}, "
            f"ubatch={self.num_ubatch}, backend_module={self._native_backend_module})"
        )
        self.load_seconds = time.perf_counter() - init_started
        self._last_generation = NativeGenerationTelemetry()
        self._last_generation.generation_mode = GenerationMode.AR_VERIFY.value
        self._last_generation.benchmark_label = benchmark_label_for_mode(
            GenerationMode.AR_VERIFY
        ).value
        self._supported_benchmark_labels = supported_benchmark_labels()
        try:
            self._native_library_info = get_native_library_info()
        except Exception:
            self._native_library_info = {
                "loaded_native_library": "",
                "native_build_id": "",
                "native_build_sha256": "",
            }
        self._enforce_native_split_backend_abi()
        self._runtime_capabilities = self._build_runtime_capabilities()

    def _clear_logits_cache(self) -> None:
        self._cached_logits_tokens = []
        self._cached_logits = None
        self._prefix_logits_cache.clear()
        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._prompt_cache_reused_tokens = 0
        self._prefill_chunk_count = 0

    def _reset_generation_counters(self) -> None:
        try:
            simd_ops.reset_qsg_sampling_stats()
        except Exception:
            pass
        self._coherence_guard_events = 0
        self._last_prompt_format_seconds = 0.0
        self._last_tokenize_seconds = 0.0
        self._last_embedding_lookup_seconds = 0.0
        self._last_graph_prefill_seconds = 0.0
        self._last_graph_decode_seconds = 0.0
        self._last_sample_seconds = 0.0
        self._last_logits_processor_seconds = 0.0
        self._last_penalty_seconds = 0.0
        self._last_suppression_seconds = 0.0
        self._last_graph_prefill_calls = 0
        self._last_graph_decode_calls = 0
        self._last_sample_calls = 0
        self._last_logits_processor_calls = 0
        self._last_penalty_calls = 0
        self._last_suppression_calls = 0
        self._last_grammar_fastlane_calls = 0
        self._prefill_chunk_count = 0
        self._last_speculative_accept_count = 0
        self._last_speculative_reject_count = 0
        self._last_self_spec_exit_layer = 0
        self._last_self_spec_exit_fraction = 0.0
        self._last_self_spec_policy = ""
        self._last_self_spec_native_path = False
        self._last_self_spec_draft_tokens = 0
        self._batch_token_fallback_count = 0
        self._python_hot_path_calls = 0
        self._numpy_hot_path_calls = 0
        self._python_qsg_forward_calls = 0
        self._python_attention_fallback_calls = 0
        self._python_ssm_fallback_calls = 0
        self._python_moe_fallback_calls = 0
        self._llama_cpp_hot_path_calls = 0
        self._tc_decode_steps = 0
        self._tc_recovery_steps = 0
        self._tc_overhead_samples = []
        self._tc_last_overhead_percent = 0.0
        self._tc_prev_stabilizer_seconds = 0.0
        self._tc_prev_stabilizer_calls = 0
        self._tc_stabilizer_seconds_total = 0.0
        self._tc_stabilizer_calls_total = 0
        self._tc_last_snapshot = {}
        self._tc_last_snapshot_valid = False
        self._tc_auto_downgrade_events = 0
        if int(getattr(self, "_tc_current_mode_native", 0)) != int(
            getattr(self, "_tc_base_mode_native", 0)
        ):
            self._tc_current_mode_native = int(getattr(self, "_tc_base_mode_native", 0))
            self._tc_current_mode = _timecrystal_native_mode_label(
                self._tc_current_mode_native
            )
            self._apply_timecrystal_drift_config()

    def _reset_graph_perf_stats_for_decode_window(self) -> bool:
        graph = getattr(self, "_model_graph", None)
        reset_perf_stats = getattr(graph, "reset_perf_stats", None)
        if not callable(reset_perf_stats):
            return False
        try:
            return bool(reset_perf_stats())
        except Exception:
            return False

    def _build_timecrystal_drift_config(
        self,
        *,
        mode_override: Optional[int] = None,
    ) -> dict[str, int | float]:
        mode = int(
            self._tc_current_mode_native
            if mode_override is None
            else int(mode_override)
        )
        if not bool(getattr(self, "_tc_enabled", False)):
            mode = 0
        drift_controller = DriftController(
            mode=str(getattr(self, "_controller_drift_mode", "adaptive") or "adaptive"),
            overhead_target_pct=float(getattr(self, "_tc_overhead_target_pct", 15.0)),
            overhead_max_pct=float(getattr(self, "_tc_overhead_max_pct", 20.0)),
            hysteresis=float(getattr(self, "_tc_hysteresis", 0.05)),
            preserve_head_tokens=int(getattr(self, "_tc_preserve_head_tokens", 256)),
            preserve_recent_tokens=int(
                getattr(self, "_tc_preserve_recent_tokens", 8192)
            ),
        )
        acceptance_ratio = 0.0
        proposed = int(getattr(self, "_last_parallel_proposed_tokens", 0) or 0)
        accepted = int(getattr(self, "_last_parallel_accepted_tokens", 0) or 0)
        if proposed > 0:
            acceptance_ratio = float(accepted) / float(proposed)
        drift_config = drift_controller.build_config(
            mode=max(0, min(2, int(mode))),
            prompt_category=str(getattr(self, "_last_prompt_category", "") or ""),
            acceptance_ratio=acceptance_ratio,
            current_overhead_pct=float(getattr(self, "_tc_last_overhead_percent", 0.0)),
        )
        return {
            "enabled": 1 if bool(getattr(self, "_tc_enabled", False)) else 0,
            "mode": max(0, min(2, int(mode))),
            "block_size_tokens": int(getattr(self, "_tc_block_size_tokens", 128)),
            "update_interval_tokens": int(
                getattr(self, "_tc_update_interval_tokens", 64)
            ),
            "prune_interval_tokens": int(
                getattr(self, "_tc_prune_interval_tokens", 128)
            ),
            "preserve_head_tokens": int(drift_config["preserve_head_tokens"]),
            "preserve_recent_tokens": int(drift_config["preserve_recent_tokens"]),
            "min_active_tokens": int(drift_config["min_active_tokens"]),
            "damp_threshold": float(getattr(self, "_tc_damp_threshold", 0.35)),
            "prune_threshold": float(getattr(self, "_tc_prune_threshold", 0.72)),
            "damping_strength": float(drift_config["damping_strength"]),
            "hysteresis": float(drift_config["hysteresis"]),
        }

    def _apply_timecrystal_drift_config(
        self,
        *,
        mode_override: Optional[int] = None,
    ) -> bool:
        graph = getattr(self, "_model_graph", None)
        setter = getattr(graph, "set_drift_config", None)
        if not callable(setter):
            self._tc_graph_drift_available = False
            return False
        try:
            ok = bool(
                setter(
                    self._build_timecrystal_drift_config(mode_override=mode_override)
                )
            )
        except Exception:
            ok = False
        self._tc_graph_drift_available = ok
        if ok and mode_override is not None:
            self._tc_current_mode_native = int(mode_override)
            self._tc_current_mode = _timecrystal_native_mode_label(
                self._tc_current_mode_native
            )
        return ok

    def _pull_timecrystal_snapshot(
        self,
        *,
        decode_step: bool,
        graph_elapsed_seconds: float,
    ) -> None:
        graph = getattr(self, "_model_graph", None)
        getter = getattr(graph, "get_last_drift_snapshot", None)
        if not callable(getter):
            return
        try:
            snapshot_raw = getter()
        except Exception:
            return
        if not snapshot_raw:
            return
        snapshot = {
            "latest_drift": float(snapshot_raw.get("latest_drift", 0.0)),
            "mean_drift": float(snapshot_raw.get("mean_drift", 0.0)),
            "max_drift": float(snapshot_raw.get("max_drift", 0.0)),
            "decay_ratio": float(snapshot_raw.get("decay_ratio", 1.0)),
            "active_token_count": int(snapshot_raw.get("active_token_count", 0)),
            "damped_block_count": int(snapshot_raw.get("damped_block_count", 0)),
            "pruned_block_count": int(snapshot_raw.get("pruned_block_count", 0)),
            "stabilizer_seconds": float(snapshot_raw.get("stabilizer_seconds", 0.0)),
            "stabilizer_calls": int(snapshot_raw.get("stabilizer_calls", 0)),
            "mode": int(snapshot_raw.get("mode", self._tc_current_mode_native)),
        }
        self._tc_last_snapshot = snapshot
        self._tc_last_snapshot_valid = True

        total_stabilizer_seconds = max(0.0, float(snapshot["stabilizer_seconds"]))
        total_stabilizer_calls = max(0, int(snapshot["stabilizer_calls"]))
        stabilizer_seconds_delta = total_stabilizer_seconds - float(
            self._tc_prev_stabilizer_seconds
        )
        stabilizer_calls_delta = total_stabilizer_calls - int(
            self._tc_prev_stabilizer_calls
        )
        if stabilizer_seconds_delta < 0.0:
            stabilizer_seconds_delta = total_stabilizer_seconds
        if stabilizer_calls_delta < 0:
            stabilizer_calls_delta = total_stabilizer_calls
        self._tc_prev_stabilizer_seconds = total_stabilizer_seconds
        self._tc_prev_stabilizer_calls = total_stabilizer_calls
        self._tc_stabilizer_seconds_total += float(max(0.0, stabilizer_seconds_delta))
        self._tc_stabilizer_calls_total += int(max(0, stabilizer_calls_delta))

        if not decode_step:
            return
        self._tc_decode_steps = int(getattr(self, "_tc_decode_steps", 0)) + 1
        if graph_elapsed_seconds > 0.0:
            step_overhead_percent = (
                float(max(0.0, stabilizer_seconds_delta)) / float(graph_elapsed_seconds)
            ) * 100.0
        else:
            step_overhead_percent = 0.0
        self._tc_overhead_samples.append(float(max(0.0, step_overhead_percent)))
        max_samples = max(1, int(getattr(self, "_tc_overhead_window", 128)))
        if len(self._tc_overhead_samples) > max_samples:
            self._tc_overhead_samples = self._tc_overhead_samples[-max_samples:]
        if self._tc_overhead_samples:
            self._tc_last_overhead_percent = float(
                sum(self._tc_overhead_samples) / len(self._tc_overhead_samples)
            )
        if (
            self._tc_decode_steps
            % max(1, int(getattr(self, "_tc_control_interval_tokens", 64)))
            == 0
        ):
            self._evaluate_timecrystal_mode()

    def _evaluate_timecrystal_mode(self) -> None:
        if not bool(getattr(self, "_tc_enabled", False)):
            return
        if not bool(getattr(self, "_tc_graph_drift_available", False)):
            return
        if int(getattr(self, "_tc_base_mode_native", 0)) < 2:
            return
        overhead = float(getattr(self, "_tc_last_overhead_percent", 0.0))
        drift_controller = DriftController(
            mode=str(getattr(self, "_controller_drift_mode", "adaptive") or "adaptive"),
            overhead_target_pct=float(getattr(self, "_tc_overhead_target_pct", 15.0)),
            overhead_max_pct=float(getattr(self, "_tc_overhead_max_pct", 20.0)),
            hysteresis=float(getattr(self, "_tc_hysteresis", 0.05)),
            preserve_head_tokens=int(getattr(self, "_tc_preserve_head_tokens", 256)),
            preserve_recent_tokens=int(
                getattr(self, "_tc_preserve_recent_tokens", 8192)
            ),
        )
        next_mode_native, next_recovery_steps, controller_decision = (
            drift_controller.transition(
                current_mode=int(getattr(self, "_tc_current_mode_native", 0)),
                current_overhead_pct=overhead,
                recovery_steps=int(getattr(self, "_tc_recovery_steps", 0)),
            )
        )
        self._tc_recovery_steps = int(next_recovery_steps)
        self._last_drift_decision = controller_decision.as_dict()
        if int(next_mode_native) != int(getattr(self, "_tc_current_mode_native", 0)):
            if str(controller_decision.reason) == "overhead_above_max":
                self._tc_auto_downgrade_events = (
                    int(getattr(self, "_tc_auto_downgrade_events", 0)) + 1
                )
            self._apply_timecrystal_drift_config(mode_override=int(next_mode_native))

    def _should_block_eos(self, generated_tokens: int) -> bool:
        return int(generated_tokens) < int(
            getattr(self, "_min_new_tokens_before_eos", 0)
        )

    def _annotate_telemetry(
        self,
        telemetry: NativeGenerationTelemetry,
        *,
        parallel_decode: bool = False,
        speculative_decode: bool = False,
    ) -> NativeGenerationTelemetry:
        model_name = str(
            getattr(self, "contract", {}).get("model")
            or getattr(getattr(self, "profile", None), "model_name", "")
            or ""
        )
        telemetry.template_name = str(
            resolve_chat_template_name(
                model_name,
                profile=getattr(self, "profile", None),
            )
        )
        self._refresh_os_thread_telemetry()
        telemetry.granite_moe_mode = str(getattr(self, "_granite_moe_mode", ""))
        telemetry.active_thread_mode = str(getattr(self, "_active_thread_mode", ""))
        telemetry.prefill_chunk_count = int(getattr(self, "_prefill_chunk_count", 0))
        telemetry.runtime_thread_switches = int(
            getattr(self, "_runtime_thread_switches", 0)
        )
        telemetry.parallel_decode = bool(parallel_decode)
        telemetry.speculative_decode = bool(speculative_decode)
        telemetry.generation_mode = str(
            telemetry.generation_mode or GenerationMode.AR_VERIFY.value
        )
        telemetry.benchmark_label = str(telemetry.benchmark_label or "").strip() or (
            benchmark_label_for_mode(telemetry.generation_mode).value
        )
        telemetry.prompt_category = str(
            getattr(self, "_last_prompt_category", telemetry.prompt_category)
        ).strip()
        telemetry.temperature_band = str(
            getattr(self, "_last_temperature_band", telemetry.temperature_band)
        ).strip()
        telemetry.prompt_cache_hits = int(getattr(self, "_prefix_cache_hits", 0))
        telemetry.prompt_cache_misses = int(getattr(self, "_prefix_cache_misses", 0))
        telemetry.prompt_cache_hit = telemetry.prompt_cache_hits > 0
        telemetry.prompt_cache_reused_tokens = int(
            getattr(self, "_prompt_cache_reused_tokens", 0)
        )
        telemetry.scheduler_queue_wait_ms = float(
            getattr(self, "_scheduler_queue_wait_ms", telemetry.scheduler_queue_wait_ms)
        )
        telemetry.scheduler_iteration_ms = float(
            getattr(self, "_scheduler_iteration_ms", telemetry.scheduler_iteration_ms)
        )
        if float(telemetry.kv_fragmentation_ratio) <= 0.0:
            telemetry.kv_fragmentation_ratio = float(
                getattr(
                    self,
                    "_scheduler_kv_fragmentation_ratio",
                    telemetry.kv_fragmentation_ratio,
                )
            )
        telemetry.prompt_format_seconds = float(
            getattr(self, "_last_prompt_format_seconds", 0.0)
        )
        telemetry.tokenize_seconds = float(getattr(self, "_last_tokenize_seconds", 0.0))
        telemetry.embedding_lookup_seconds = float(
            getattr(self, "_last_embedding_lookup_seconds", 0.0)
        )
        telemetry.graph_prefill_seconds = float(
            getattr(self, "_last_graph_prefill_seconds", 0.0)
        )
        telemetry.graph_decode_seconds = float(
            getattr(self, "_last_graph_decode_seconds", 0.0)
        )
        telemetry.sample_seconds = float(getattr(self, "_last_sample_seconds", 0.0))
        telemetry.logits_processor_seconds = float(
            getattr(self, "_last_logits_processor_seconds", 0.0)
        )
        telemetry.penalty_seconds = float(getattr(self, "_last_penalty_seconds", 0.0))
        telemetry.suppression_seconds = float(
            getattr(self, "_last_suppression_seconds", 0.0)
        )
        telemetry.graph_prefill_calls = int(
            getattr(self, "_last_graph_prefill_calls", 0)
        )
        telemetry.graph_decode_calls = int(getattr(self, "_last_graph_decode_calls", 0))
        telemetry.sample_calls = int(getattr(self, "_last_sample_calls", 0))
        telemetry.logits_processor_calls = int(
            getattr(self, "_last_logits_processor_calls", 0)
        )
        telemetry.penalty_calls = int(getattr(self, "_last_penalty_calls", 0))
        telemetry.suppression_calls = int(getattr(self, "_last_suppression_calls", 0))
        telemetry.context_stabilizer_enabled = bool(
            getattr(self, "_tc_enabled", False)
            and getattr(self, "_tc_graph_drift_available", False)
        )
        telemetry.context_stabilizer_mode = str(
            getattr(self, "_tc_current_mode", "telemetry")
        )
        snapshot = (
            dict(getattr(self, "_tc_last_snapshot", {}))
            if getattr(self, "_tc_last_snapshot_valid", False)
            else {}
        )
        telemetry.drift_latest = float(snapshot.get("latest_drift", 0.0))
        telemetry.drift_mean = float(snapshot.get("mean_drift", 0.0))
        telemetry.drift_max = float(snapshot.get("max_drift", 0.0))
        telemetry.drift_decay_ratio = float(snapshot.get("decay_ratio", 1.0))
        telemetry.drift_active_tokens = int(snapshot.get("active_token_count", 0))
        telemetry.drift_damped_blocks = int(snapshot.get("damped_block_count", 0))
        telemetry.drift_pruned_blocks = int(snapshot.get("pruned_block_count", 0))
        telemetry.stabilizer_seconds = float(
            getattr(self, "_tc_stabilizer_seconds_total", 0.0)
        )
        telemetry.stabilizer_calls = int(getattr(self, "_tc_stabilizer_calls_total", 0))
        telemetry.drift_auto_downgrade_events = int(
            getattr(self, "_tc_auto_downgrade_events", 0)
        )
        telemetry.drift_overhead_percent = float(
            getattr(self, "_tc_last_overhead_percent", 0.0)
        )
        try:
            qsg_sampling_stats = simd_ops.get_qsg_sampling_stats()
        except Exception:
            qsg_sampling_stats = {}
        affinity_plan = _parse_json_object(getattr(self, "_affinity_plan_json", ""))
        topology = _parse_json_object(getattr(self, "_topology_json", ""))
        worker_cpus = [
            int(cpu)
            for cpu in affinity_plan.get("decode_worker_cpus", []) or []
            if isinstance(cpu, (int, float))
        ]
        orchestrator_cpus = [
            int(cpu)
            for cpu in affinity_plan.get("orchestrator_cpus", []) or []
            if isinstance(cpu, (int, float))
        ]
        telemetry.coconut_enabled = bool(
            getattr(self, "_native_qsg_use_coconut", False)
        )
        telemetry.coconut_paths = int(getattr(self, "_native_qsg_coconut_paths", 0))
        telemetry.coconut_alpha = float(getattr(self, "_native_qsg_coconut_alpha", 0.0))
        telemetry.coconut_seconds = float(
            qsg_sampling_stats.get("coconut_seconds", 0.0)
        )
        telemetry.coconut_candidate_count = int(
            qsg_sampling_stats.get("coconut_candidate_count", 0)
        )
        telemetry.coconut_entropy_mean = float(
            qsg_sampling_stats.get("coconut_entropy_mean", 0.0)
        )
        telemetry.coconut_amplitude_mean = float(
            qsg_sampling_stats.get("coconut_amplitude_mean", 0.0)
        )
        telemetry.coconut_consistency_rejects = int(
            qsg_sampling_stats.get("coconut_consistency_rejects", 0)
        )
        telemetry.grammar_fastlane_calls = int(
            qsg_sampling_stats.get("grammar_fastlane_calls", 0)
        )
        telemetry.grover_enabled = bool(getattr(self, "_native_qsg_use_grover", False))
        telemetry.grover_top_k = int(getattr(self, "_native_qsg_grover_top_k", 0))
        telemetry.grover_damping = float(
            getattr(self, "_native_qsg_grover_damping", 0.0)
        )
        telemetry.grover_calls = int(qsg_sampling_stats.get("grover_calls", 0))
        telemetry.grover_seconds = float(qsg_sampling_stats.get("grover_seconds", 0.0))
        telemetry.grover_candidate_count = int(
            qsg_sampling_stats.get("grover_candidate_count", 0)
        )
        telemetry.grover_rescore_delta_mean = float(
            qsg_sampling_stats.get("grover_rescore_delta_mean", 0.0)
        )
        telemetry.grover_timeout_events = int(
            qsg_sampling_stats.get("grover_timeout_events", 0)
        )
        telemetry.delta_watermark = DeltaWatermark.from_dict(
            getattr(self, "_delta_watermark", None)
        ).as_dict()
        telemetry.controller_state = {
            "drift": dict(getattr(self, "_last_drift_decision", {}) or {}),
        }
        telemetry.execution_capsule_id = str(
            getattr(self, "_last_execution_capsule_id", "") or ""
        )
        telemetry.strict_cpp_only = bool(getattr(self, "_strict_cpp_only", False))
        telemetry.strict_native_qsg = bool(
            PERFORMANCE_CONFIG.get("strict_native_qsg", False)
        )
        telemetry.coherence_guard_events = int(
            getattr(self, "_coherence_guard_events", 0)
        )
        telemetry.native_fast_path = bool(getattr(self, "_native_fast_path", False))
        telemetry.parallel_decode_disable_reason = str(
            getattr(self, "_parallel_decode_disable_reason", "")
        )
        telemetry.speculative_accept_count = int(
            getattr(self, "_last_speculative_accept_count", 0)
        )
        telemetry.speculative_reject_count = int(
            getattr(self, "_last_speculative_reject_count", 0)
        )
        telemetry.self_spec_native_path = bool(
            getattr(self, "_last_self_spec_native_path", False)
        )
        telemetry.self_spec_policy = str(getattr(self, "_last_self_spec_policy", ""))
        telemetry.self_spec_exit_layer = int(
            getattr(self, "_last_self_spec_exit_layer", 0)
        )
        telemetry.self_spec_exit_fraction = float(
            getattr(self, "_last_self_spec_exit_fraction", 0.0)
        )
        telemetry.self_spec_draft_tokens = int(
            getattr(self, "_last_self_spec_draft_tokens", 0)
        )
        telemetry.native_build_id = str(
            getattr(self, "_native_library_info", {}).get("native_build_id", "")
        )
        telemetry.native_build_sha256 = str(
            getattr(self, "_native_library_info", {}).get("native_build_sha256", "")
        )
        telemetry.loaded_native_library = str(
            getattr(self, "_native_library_info", {}).get("loaded_native_library", "")
        )
        telemetry.native_split_layout = str(
            getattr(self, "_native_library_info", {}).get("native_split_layout", "")
        )
        telemetry.native_public_load_target = str(
            getattr(self, "_native_library_info", {}).get(
                "native_public_load_target", ""
            )
        )
        telemetry.native_runtime_core_target = str(
            getattr(self, "_native_library_info", {}).get(
                "native_runtime_core_target", ""
            )
        )
        telemetry.native_split_abi_version = int(
            getattr(self, "_native_library_info", {}).get("native_split_abi_version", 0)
            or 0
        )
        telemetry.native_isa_baseline = str(
            getattr(self, "_native_library_info", {}).get("native_isa_baseline", "")
        )
        telemetry.native_compat_aliases = [
            str(alias).strip()
            for alias in getattr(self, "_native_library_info", {}).get(
                "native_compat_aliases", []
            )
            if str(alias).strip()
        ]
        telemetry.sanctioned_backend_path = str(
            getattr(self, "_sanctioned_backend_path", SANCTIONED_BACKEND_PATH)
        )
        telemetry.tokenizer_backend = str(getattr(self, "_tokenizer_backend", "native"))
        telemetry.backend_module = str(
            getattr(self, "_native_backend_info", {}).get("backend_module", "")
        )
        telemetry.backend_module_requested = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_module_requested", ""
            )
        )
        telemetry.backend_module_library = str(
            getattr(self, "_native_backend_info", {}).get("backend_module_library", "")
        )
        telemetry.backend_module_loaded = bool(
            getattr(self, "_native_backend_info", {}).get(
                "backend_module_loaded", False
            )
        )
        telemetry.backend_module_candidates = [
            str(candidate).strip()
            for candidate in getattr(self, "_native_backend_info", {}).get(
                "backend_module_candidates", []
            )
            if str(candidate).strip()
        ]
        telemetry.backend_selection_source = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_selection_source", ""
            )
        )
        telemetry.backend_selection_reason = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_selection_reason", ""
            )
        )
        telemetry.backend_selection_model_name = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_selection_model_name", ""
            )
        )
        telemetry.backend_selection_architecture = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_selection_architecture", ""
            )
        )
        telemetry.backend_selection_family = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_selection_family", ""
            )
        )
        telemetry.backend_module_marker_symbol = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_module_marker_symbol", ""
            )
        )
        telemetry.backend_module_marker = int(
            getattr(self, "_native_backend_info", {}).get("backend_module_marker", 0)
            or 0
        )
        telemetry.backend_module_name_symbol = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_module_name_symbol", ""
            )
        )
        telemetry.backend_module_name = str(
            getattr(self, "_native_backend_info", {}).get("backend_module_name", "")
        )
        telemetry.backend_module_build_id_symbol = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_module_build_id_symbol", ""
            )
        )
        telemetry.backend_module_build_id = str(
            getattr(self, "_native_backend_info", {}).get("backend_module_build_id", "")
        )
        telemetry.backend_module_abi_symbol = str(
            getattr(self, "_native_backend_info", {}).get(
                "backend_module_abi_symbol", ""
            )
        )
        telemetry.backend_module_abi_version = int(
            getattr(self, "_native_backend_info", {}).get(
                "backend_module_abi_version", 0
            )
            or 0
        )
        telemetry.backend_module_required = bool(
            getattr(self, "_native_backend_required", False)
        )
        telemetry.native_backend_abi_match = bool(
            telemetry.backend_module_loaded
            and telemetry.native_split_abi_version > 0
            and telemetry.backend_module_abi_version > 0
            and telemetry.native_split_abi_version
            == telemetry.backend_module_abi_version
        )
        telemetry.python_hot_path_calls = int(
            getattr(self, "_python_hot_path_calls", 0)
        )
        telemetry.numpy_hot_path_calls = int(getattr(self, "_numpy_hot_path_calls", 0))
        telemetry.python_qsg_forward_calls = int(
            getattr(self, "_python_qsg_forward_calls", 0)
        )
        telemetry.python_attention_fallback_calls = int(
            getattr(self, "_python_attention_fallback_calls", 0)
        )
        telemetry.python_ssm_fallback_calls = int(
            getattr(self, "_python_ssm_fallback_calls", 0)
        )
        telemetry.python_moe_fallback_calls = int(
            getattr(self, "_python_moe_fallback_calls", 0)
        )
        telemetry.llama_cpp_hot_path_calls = int(
            getattr(self, "_llama_cpp_hot_path_calls", 0)
        )
        telemetry.logical_core_count = int(
            getattr(self, "_logical_core_count", _visible_logical_threads())
        )
        telemetry.physical_core_count = int(
            getattr(
                self,
                "_physical_core_count",
                _physical_core_thread_hint(telemetry.logical_core_count),
            )
        )
        telemetry.p_core_count = int(getattr(self, "_p_core_count", 0))
        telemetry.affinity_policy = str(
            getattr(self, "_affinity_policy", os.getenv("ANVIL_OMP_PROC_BIND", "close"))
        )
        telemetry.affinity_mode = int(getattr(self, "_affinity_mode", 1))
        telemetry.l3_domain_count = int(getattr(self, "_l3_domain_count", 0))
        telemetry.numa_strict = bool(getattr(self, "_numa_strict", False))
        telemetry.numa_affinity_mode = str(
            getattr(
                self, "_numa_affinity_mode", os.getenv("ANVIL_NUMA_AFFINITY_MODE", "")
            )
        )
        telemetry.numa_hugepage = str(
            getattr(self, "_numa_hugepage", os.getenv("ANVIL_NUMA_HUGEPAGE", ""))
        )
        telemetry.numa_bind_policy = str(
            getattr(self, "_numa_bind_policy", os.getenv("ANVIL_NUMA_BIND_POLICY", ""))
        )
        telemetry.numa_first_touch = bool(getattr(self, "_numa_first_touch", False))
        telemetry.topology_json = str(getattr(self, "_topology_json", ""))
        telemetry.os_thread_migrations = int(getattr(self, "_os_thread_migrations", 0))
        telemetry.os_last_cpu = int(getattr(self, "_os_last_cpu", -1))
        telemetry.omp_places = str(
            getattr(self, "_omp_places", os.getenv("OMP_PLACES", ""))
        )
        telemetry.omp_proc_bind = str(
            getattr(self, "_omp_proc_bind", os.getenv("OMP_PROC_BIND", ""))
        )
        telemetry.perf_event_access = bool(getattr(self, "_perf_event_access", False))
        telemetry.perf_event_access_reason = str(
            getattr(self, "_perf_event_access_reason", "")
        )
        telemetry.cpu_governor = str(getattr(self, "_cpu_governor", ""))
        telemetry.thp_mode = str(getattr(self, "_thp_mode", ""))
        telemetry.perf_counter_source = str(getattr(self, "_perf_counter_source", ""))
        telemetry.worker_cpu_mask = _cpu_mask_string(worker_cpus)
        telemetry.orchestrator_cpu_mask = _cpu_mask_string(orchestrator_cpus)
        telemetry.l3_domain_ids_active = _active_l3_domain_ids(
            topology,
            [*worker_cpus, *orchestrator_cpus],
        )
        telemetry.autotune_profile_id = str(getattr(self, "_autotune_profile_id", ""))
        telemetry.autotune_source = str(getattr(self, "_autotune_source", ""))
        telemetry.autotune_score = float(getattr(self, "_autotune_score", 0.0))
        telemetry.autotune_exploration_count = int(
            getattr(self, "_autotune_exploration_count", 0)
        )
        omp_max_threads = getattr(self, "_omp_max_threads", None)
        if omp_max_threads is None:
            try:
                omp_max_threads = simd_ops.get_omp_max_threads()
            except Exception:
                omp_max_threads = 0
        omp_dynamic = getattr(self, "_omp_dynamic", None)
        if omp_dynamic is None:
            try:
                omp_dynamic = simd_ops.get_omp_dynamic()
            except Exception:
                omp_dynamic = False
        omp_active_levels = getattr(self, "_omp_active_levels", None)
        if omp_active_levels is None:
            try:
                omp_active_levels = simd_ops.get_omp_active_levels()
            except Exception:
                omp_active_levels = 0
        telemetry.omp_max_threads = int(omp_max_threads)
        telemetry.omp_dynamic = bool(omp_dynamic)
        telemetry.omp_active_levels = int(omp_active_levels)
        telemetry.batch_token_fallback_count = int(
            getattr(self, "_batch_token_fallback_count", 0)
        )
        telemetry.prefill_phase_gap_ms = float(telemetry.prefill_seconds) * 1000.0 - (
            float(telemetry.prompt_format_seconds) * 1000.0
            + float(telemetry.tokenize_seconds) * 1000.0
            + float(telemetry.embedding_lookup_seconds) * 1000.0
            + float(telemetry.graph_prefill_seconds) * 1000.0
        )
        telemetry.decode_phase_gap_ms = float(telemetry.decode_seconds) * 1000.0 - (
            float(telemetry.graph_decode_seconds) * 1000.0
            + float(telemetry.sample_seconds) * 1000.0
            + float(telemetry.logits_processor_seconds) * 1000.0
            + float(telemetry.penalty_seconds) * 1000.0
            + float(telemetry.suppression_seconds) * 1000.0
        )
        graph = getattr(self, "_model_graph", None)
        telemetry.sampling_backend = "native_simd"
        telemetry.penalties_backend = "native_simd"
        telemetry.suppression_backend = (
            "native_simd"
            if getattr(self, "_suppressed_token_ids_sorted", ())
            else "none"
        )
        telemetry.logits_backend = "native_graph_token_id"
        telemetry.full_graph_enabled = bool(
            graph is not None and getattr(graph, "has_full_graph", False)
        )
        telemetry.batched_prefill_native_enabled = bool(
            graph is not None and getattr(graph, "_has_batch_token_id_api", False)
        )
        telemetry.qsg_processors_native_enabled = bool(
            getattr(self, "_qsg_processors_native_enabled", False)
        )
        telemetry.full_qsg_enabled = _full_qsg_enabled(
            full_graph_enabled=telemetry.full_graph_enabled,
            qsg_processors_native_enabled=telemetry.qsg_processors_native_enabled,
            batched_prefill_native_enabled=telemetry.batched_prefill_native_enabled,
            tokenizer_backend=telemetry.tokenizer_backend,
            sanctioned_backend_path=telemetry.sanctioned_backend_path,
        )
        telemetry.hot_path_numpy_detected = bool(
            telemetry.python_hot_path_calls > 0
            or telemetry.numpy_hot_path_calls > 0
            or telemetry.python_qsg_forward_calls > 0
            or telemetry.python_attention_fallback_calls > 0
            or telemetry.python_ssm_fallback_calls > 0
            or telemetry.python_moe_fallback_calls > 0
            or telemetry.llama_cpp_hot_path_calls > 0
        )
        telemetry.strict_path_stable = bool(
            telemetry.full_qsg_enabled
            and not telemetry.hot_path_numpy_detected
            and (telemetry.strict_cpp_only or telemetry.strict_native_qsg)
        )
        telemetry.hot_path_proof = {
            "sanctioned_backend_path": telemetry.sanctioned_backend_path,
            "tokenizer_backend": telemetry.tokenizer_backend,
            "prompt_tokenization": "C++ AVX2/OpenMP",
            "graph_prefill_decode": "C++ AVX2/OpenMP",
            "sampling": "C++ AVX2/OpenMP",
            "qsg_processors": (
                "native_simd" if telemetry.qsg_processors_native_enabled else "disabled"
            ),
            "penalties": telemetry.penalties_backend,
            "suppression": telemetry.suppression_backend,
            "coconut_enabled": "true" if telemetry.coconut_enabled else "false",
            "grover_enabled": "true" if telemetry.grover_enabled else "false",
            "native_isa_baseline": telemetry.native_isa_baseline,
            "native_backend_abi_match": (
                "true" if telemetry.native_backend_abi_match else "false"
            ),
            "perf_event_access": ("true" if telemetry.perf_event_access else "false"),
            "perf_event_access_reason": telemetry.perf_event_access_reason,
            "cpu_governor": telemetry.cpu_governor,
            "thp_mode": telemetry.thp_mode,
            "worker_cpu_mask": telemetry.worker_cpu_mask,
            "orchestrator_cpu_mask": telemetry.orchestrator_cpu_mask,
            "l3_domain_ids_active": ",".join(
                str(value) for value in telemetry.l3_domain_ids_active
            ),
            "autotune_profile_id": telemetry.autotune_profile_id,
            "autotune_source": telemetry.autotune_source,
            "strict_cpp_only": "true" if telemetry.strict_cpp_only else "false",
            "strict_native_qsg": "true" if telemetry.strict_native_qsg else "false",
            "strict_path_stable": "true" if telemetry.strict_path_stable else "false",
            "context_stabilizer_enabled": (
                "true" if telemetry.context_stabilizer_enabled else "false"
            ),
            "context_stabilizer_mode": telemetry.context_stabilizer_mode,
            "drift_overhead_percent": f"{telemetry.drift_overhead_percent:.3f}",
            "python_or_numpy_hot_path": (
                "detected" if telemetry.hot_path_numpy_detected else "not_detected"
            ),
            "executed_cpp_only": (
                "true" if not telemetry.hot_path_numpy_detected else "false"
            ),
            "python_hot_path_calls": str(telemetry.python_hot_path_calls),
            "numpy_hot_path_calls": str(telemetry.numpy_hot_path_calls),
            "python_qsg_forward_calls": str(telemetry.python_qsg_forward_calls),
            "python_attention_fallback_calls": str(
                telemetry.python_attention_fallback_calls
            ),
            "python_ssm_fallback_calls": str(telemetry.python_ssm_fallback_calls),
            "python_moe_fallback_calls": str(telemetry.python_moe_fallback_calls),
            "llama_cpp_hot_path_calls": str(telemetry.llama_cpp_hot_path_calls),
            "full_qsg": ("enabled" if telemetry.full_qsg_enabled else "disabled"),
        }
        if graph is not None:
            telemetry.hot_path_proof["lm_head_layout"] = str(
                getattr(graph, "_lm_head_layout", "unknown")
            )
            telemetry.hot_path_proof["graph_batched_token_id_api"] = (
                "enabled"
                if getattr(graph, "_has_batch_token_id_api", False)
                else "disabled"
            )
        if graph is not None and hasattr(graph, "get_perf_stats"):
            perf_stats = graph.get_perf_stats()
            if perf_stats:
                telemetry.graph_stage_seconds = {
                    "embedding_lookup": float(
                        perf_stats.get("embedding_lookup_seconds", 0.0)
                    ),
                    "attention_proj": float(
                        perf_stats.get("attention_proj_seconds", 0.0)
                    ),
                    "attention_rope_kv": float(
                        perf_stats.get("attention_rope_kv_seconds", 0.0)
                    ),
                    "attention_decode": float(
                        perf_stats.get("attention_decode_seconds", 0.0)
                    ),
                    "attention_out_proj": float(
                        perf_stats.get("attention_out_proj_seconds", 0.0)
                    ),
                    "ffn_norm": float(perf_stats.get("ffn_norm_seconds", 0.0)),
                    "ffn_gate_up": float(perf_stats.get("ffn_gate_up_seconds", 0.0)),
                    "ffn_down": float(perf_stats.get("ffn_down_seconds", 0.0)),
                    "ssm_projection": float(
                        perf_stats.get("ssm_projection_seconds", 0.0)
                    ),
                    "ssm_conv": float(perf_stats.get("ssm_conv_seconds", 0.0)),
                    "ssm_recurrent": float(
                        perf_stats.get("ssm_recurrent_seconds", 0.0)
                    ),
                    "ssm_output": float(perf_stats.get("ssm_output_seconds", 0.0)),
                    "moe": float(perf_stats.get("moe_seconds", 0.0)),
                    "final_norm": float(perf_stats.get("final_norm_seconds", 0.0)),
                    "lm_head": float(perf_stats.get("lm_head_seconds", 0.0)),
                    "sanitize": float(perf_stats.get("sanitize_seconds", 0.0)),
                }
                forward_token_calls = int(perf_stats.get("forward_token_calls", 0))
                forward_token_id_calls = int(
                    perf_stats.get("forward_token_id_calls", 0)
                )
                forward_token_ids_calls = int(
                    perf_stats.get("forward_token_ids_calls", 0)
                )
                forward_token_ids_token_count = int(
                    perf_stats.get("forward_token_ids_token_count", 0)
                )
                attention_calls = int(perf_stats.get("attention_calls", 0))
                ffn_calls = int(perf_stats.get("ffn_calls", 0))
                ssm_calls = int(perf_stats.get("ssm_calls", 0))
                moe_calls = int(perf_stats.get("moe_calls", 0))
                telemetry.batched_prefill_token_id_calls = forward_token_ids_calls
                telemetry.batched_prefill_token_id_tokens = (
                    forward_token_ids_token_count
                )
                telemetry.packed_lm_head_calls = int(
                    perf_stats.get("packed_lm_head_calls", 0)
                )
                telemetry.graph_stage_calls = {
                    "embedding_lookup": forward_token_id_calls,
                    "attention_proj": attention_calls,
                    "attention_rope_kv": attention_calls,
                    "attention_decode": attention_calls,
                    "attention_out_proj": attention_calls,
                    "ffn_norm": ffn_calls,
                    "ffn_gate_up": ffn_calls,
                    "ffn_down": ffn_calls,
                    "ssm_projection": ssm_calls,
                    "ssm_conv": ssm_calls,
                    "ssm_recurrent": ssm_calls,
                    "ssm_output": ssm_calls,
                    "moe": moe_calls,
                    "final_norm": forward_token_calls,
                    "lm_head": forward_token_calls,
                    "sanitize": forward_token_calls,
                    "forward_token": forward_token_calls,
                    "forward_token_id": forward_token_id_calls,
                    "forward_token_ids": forward_token_ids_calls,
                    "forward_token_ids_token_count": forward_token_ids_token_count,
                    "attention": attention_calls,
                    "ffn": ffn_calls,
                    "ssm_ops": ssm_calls,
                    "moe_ops": moe_calls,
                    "packed_lm_head": telemetry.packed_lm_head_calls,
                }
        return telemetry

    @staticmethod
    def _apply_generation_evidence(
        telemetry: NativeGenerationTelemetry,
        evidence: Optional[GenerationEvidence],
    ) -> NativeGenerationTelemetry:
        if evidence is None:
            return telemetry
        telemetry.generation_mode = str(evidence.generation_mode)
        telemetry.benchmark_label = str(
            evidence.benchmark_label
            or benchmark_label_for_mode(telemetry.generation_mode).value
        )
        telemetry.accepted_parallel_tokens = int(evidence.accepted_parallel_tokens)
        telemetry.rejected_parallel_tokens = int(evidence.rejected_parallel_tokens)
        telemetry.proposed_parallel_tokens = int(evidence.proposed_parallel_tokens)
        telemetry.draft_frontier_width = int(evidence.draft_frontier_width)
        telemetry.verify_depth = int(evidence.verify_depth)
        telemetry.jacobi_frontier_width = int(
            evidence.jacobi_frontier_width or evidence.draft_frontier_width
        )
        telemetry.jacobi_branch_survival_rate = float(
            evidence.jacobi_branch_survival_rate
        )
        telemetry.jacobi_verify_cost_ms = float(evidence.jacobi_verify_cost_ms)
        telemetry.jacobi_branch_entropy = float(evidence.jacobi_branch_entropy)
        telemetry.parallel_step_latency_ms = float(evidence.parallel_step_latency_ms)
        telemetry.draft_confidence_mean = float(evidence.draft_confidence_mean)
        telemetry.draft_confidence_min = float(evidence.draft_confidence_min)
        telemetry.draft_source = str(evidence.draft_source)
        telemetry.blockwise_blocks = int(evidence.blockwise_blocks)
        telemetry.blockwise_denoise_steps = int(evidence.blockwise_denoise_steps)
        telemetry.blockwise_convergence_rate = float(
            evidence.blockwise_convergence_rate
        )
        prompt_cache_lookups = max(
            0,
            int(telemetry.prompt_cache_hits) + int(telemetry.prompt_cache_misses),
        )
        telemetry.prefix_cache_hit_rate = (
            float(evidence.prefix_cache_hit_rate)
            if evidence.prefix_cache_hit_rate > 0.0
            else (
                float(telemetry.prompt_cache_hits) / float(prompt_cache_lookups)
                if prompt_cache_lookups > 0
                else 0.0
            )
        )
        telemetry.scheduler_queue_wait_ms = float(evidence.scheduler_queue_wait_ms)
        telemetry.scheduler_iteration_ms = float(evidence.scheduler_iteration_ms)
        telemetry.kv_fragmentation_ratio = float(
            evidence.kv_fragmentation_ratio
            if evidence.kv_fragmentation_ratio > 0.0
            else telemetry.kv_fragmentation_ratio
        )
        telemetry.quality_guard_triggered = bool(
            evidence.quality_guard_triggered or telemetry.coherence_guard_events > 0
        )
        return telemetry

    def _update_scheduler_metrics_snapshot(self, metrics: dict[str, Any]) -> None:
        self._scheduler_queue_wait_ms = float(
            metrics.get(
                "scheduler_queue_wait_ms",
                metrics.get("qsg_queue_wait_ms_p95", 0.0),
            )
        )
        self._scheduler_iteration_ms = float(
            metrics.get(
                "scheduler_iteration_ms",
                metrics.get("qsg_scheduler_iteration_ms_p95", 0.0),
            )
        )
        self._scheduler_kv_fragmentation_ratio = float(
            metrics.get(
                "kv_fragmentation_ratio",
                metrics.get("qsg_state_fragmentation_ratio", 0.0),
            )
        )
        self._scheduler_kv_pages_total = int(metrics.get("qsg_state_pages_total", 0))
        self._scheduler_kv_pages_in_use = int(metrics.get("qsg_state_pages_in_use", 0))
        self._scheduler_kv_active_page_slots = int(
            metrics.get("qsg_state_active_page_slots", 0)
        )
        self._scheduler_kv_shared_page_slots = int(
            metrics.get("qsg_state_shared_page_slots", 0)
        )
        self._scheduler_kv_snapshot_count = int(
            metrics.get("qsg_state_snapshot_count", 0)
        )
        self._scheduler_kv_cow_events = int(metrics.get("qsg_state_cow_events", 0))
        self._scheduler_kv_prefix_share_events = int(
            metrics.get("qsg_state_prefix_share_events", 0)
        )
        self._scheduler_kv_active_tokens = int(
            metrics.get("qsg_state_active_tokens", 0)
        )
        self._scheduler_kv_committed_token_capacity = int(
            metrics.get("qsg_state_committed_token_capacity", 0)
        )
        self._scheduler_kv_page_tokens = int(metrics.get("qsg_state_page_tokens", 0))
        self._scheduler_sequence_state_counts = dict(
            metrics.get("qsg_sequence_state_counts", {})
        )
        self._scheduler_sequence_mode_counts = dict(
            metrics.get("qsg_sequence_mode_counts", {})
        )
        self._scheduler_sequence_checkpoint_count = int(
            metrics.get("qsg_sequence_checkpoint_count", 0)
        )
        self._scheduler_tool_wait_requests = int(
            metrics.get("qsg_tool_wait_requests", 0)
        )
        self._scheduler_latent_packet_count = int(
            metrics.get("qsg_latent_packet_count", 0)
        )
        self._scheduler_evidence_capsule_count = int(
            metrics.get("qsg_evidence_capsule_count", 0)
        )
        self._scheduler_suspend_events = int(metrics.get("qsg_suspend_events", 0))
        self._scheduler_resume_events = int(metrics.get("qsg_resume_events", 0))
        runtime_caps = getattr(self, "_runtime_capabilities", None)
        if isinstance(runtime_caps, dict):
            runtime_caps["scheduler_queue_wait_ms"] = float(
                self._scheduler_queue_wait_ms
            )
            runtime_caps["scheduler_iteration_ms"] = float(self._scheduler_iteration_ms)
            runtime_caps["kv_fragmentation_ratio"] = float(
                self._scheduler_kv_fragmentation_ratio
            )
            runtime_caps["qsg_state_pages_total"] = int(self._scheduler_kv_pages_total)
            runtime_caps["qsg_state_pages_in_use"] = int(
                self._scheduler_kv_pages_in_use
            )
            runtime_caps["qsg_state_active_page_slots"] = int(
                self._scheduler_kv_active_page_slots
            )
            runtime_caps["qsg_state_shared_page_slots"] = int(
                self._scheduler_kv_shared_page_slots
            )
            runtime_caps["qsg_state_snapshot_count"] = int(
                self._scheduler_kv_snapshot_count
            )
            runtime_caps["qsg_state_cow_events"] = int(self._scheduler_kv_cow_events)
            runtime_caps["qsg_state_prefix_share_events"] = int(
                self._scheduler_kv_prefix_share_events
            )
            runtime_caps["qsg_state_active_tokens"] = int(
                self._scheduler_kv_active_tokens
            )
            runtime_caps["qsg_state_committed_token_capacity"] = int(
                self._scheduler_kv_committed_token_capacity
            )
            runtime_caps["qsg_state_page_tokens"] = int(self._scheduler_kv_page_tokens)
            runtime_caps["qsg_sequence_state_counts"] = dict(
                self._scheduler_sequence_state_counts
            )
            runtime_caps["qsg_sequence_mode_counts"] = dict(
                self._scheduler_sequence_mode_counts
            )
            runtime_caps["qsg_sequence_checkpoint_count"] = int(
                self._scheduler_sequence_checkpoint_count
            )
            runtime_caps["qsg_tool_wait_requests"] = int(
                self._scheduler_tool_wait_requests
            )
            runtime_caps["qsg_latent_packet_count"] = int(
                self._scheduler_latent_packet_count
            )
            runtime_caps["qsg_evidence_capsule_count"] = int(
                self._scheduler_evidence_capsule_count
            )
            runtime_caps["qsg_suspend_events"] = int(self._scheduler_suspend_events)
            runtime_caps["qsg_resume_events"] = int(self._scheduler_resume_events)

    def build_parallel_generation_engine(
        self,
        *,
        config: Any,
        stream_producer: Any,
    ) -> NativeParallelGenerationEngine:
        return NativeParallelGenerationEngine(
            native_engine=self,
            config=config,
            stream_producer=stream_producer,
        )

    def _build_decode_runtime(self) -> NativeQSGRuntime | None:
        model_graph = getattr(self, "_model_graph", None)
        graph_handle = int(getattr(model_graph, "_handle", 0) or 0)
        if graph_handle <= 0:
            return None
        try:
            return NativeQSGRuntime(
                model_graph_handle=graph_handle,
                vocab_size=int(self.profile.vocab_size),
                eos_token=int(self.token_eos()),
                ubatch=int(max(1, getattr(self, "num_ubatch", 1))),
                max_active_requests=1,
                max_pending_requests=4,
                priority_policy=False,
                interleaved_streams=False,
            )
        except Exception:
            return None

    def _build_runtime_capabilities(self) -> dict[str, Any]:
        metadata = self.loader.get_metadata()
        get_quantization = getattr(self.loader, "get_quantization_label", None)
        contract = getattr(self, "contract", {}) or {}
        model_name = str(
            contract.get("model")
            or getattr(getattr(self, "profile", None), "model_name", "")
            or ""
        )
        self._refresh_os_thread_telemetry()
        try:
            qsg_sampling_stats = simd_ops.get_qsg_sampling_stats()
        except Exception:
            qsg_sampling_stats = {}
        omp_max_threads = getattr(self, "_omp_max_threads", None)
        if omp_max_threads is None:
            try:
                omp_max_threads = simd_ops.get_omp_max_threads()
            except Exception:
                omp_max_threads = 0
        omp_dynamic = getattr(self, "_omp_dynamic", None)
        if omp_dynamic is None:
            try:
                omp_dynamic = simd_ops.get_omp_dynamic()
            except Exception:
                omp_dynamic = False
        omp_active_levels = getattr(self, "_omp_active_levels", None)
        if omp_active_levels is None:
            try:
                omp_active_levels = simd_ops.get_omp_active_levels()
            except Exception:
                omp_active_levels = 0
        affinity_plan = _parse_json_object(getattr(self, "_affinity_plan_json", ""))
        topology = _parse_json_object(getattr(self, "_topology_json", ""))
        worker_cpus = [
            int(cpu)
            for cpu in affinity_plan.get("decode_worker_cpus", []) or []
            if isinstance(cpu, (int, float))
        ]
        orchestrator_cpus = [
            int(cpu)
            for cpu in affinity_plan.get("orchestrator_cpus", []) or []
            if isinstance(cpu, (int, float))
        ]
        visible_l3_domains = [
            int(domain)
            for domain in affinity_plan.get("visible_l3_domains", []) or []
            if isinstance(domain, (int, float))
        ]
        batch_l3_domains = [
            int(domain)
            for domain in affinity_plan.get("batch_l3_domains", []) or []
            if isinstance(domain, (int, float))
        ]
        decode_primary_l3_domain_value = affinity_plan.get(
            "decode_primary_l3_domain", -1
        )
        preferred_decode_l3_domain_value = affinity_plan.get(
            "preferred_decode_l3_domain", -1
        )
        decode_primary_l3_domain = int(
            decode_primary_l3_domain_value
            if isinstance(decode_primary_l3_domain_value, (int, float))
            else -1
        )
        preferred_decode_l3_domain = int(
            preferred_decode_l3_domain_value
            if isinstance(preferred_decode_l3_domain_value, (int, float))
            else -1
        )
        capabilities = {
            "backend": self.backend,
            "sanctioned_backend_path": str(
                getattr(self, "_sanctioned_backend_path", SANCTIONED_BACKEND_PATH)
            ),
            "model": model_name,
            "digest": str(contract.get("digest", "")),
            "quantization": (
                get_quantization() if callable(get_quantization) else "unknown"
            ),
            "kv_cache_quantization": _kv_cache_quantization_mode(),
            "architecture": self.architecture,
            "context_length": int(self.context_length),
            "load_seconds": float(getattr(self, "load_seconds", 0.0)),
            "decode_threads": int(self.num_threads_decode),
            "batch_threads": int(self.num_threads_batch),
            "ubatch": int(self.num_ubatch),
            "mmap_enabled": bool(self._use_mmap_weights),
            "mapped_model_bytes": int(contract.get("blob_size", 0)),
            "loader_cache_residency_bytes": int(contract.get("blob_size", 0)),
            "embedding_materialization_bytes": 0,
            "openmp_enabled": bool(simd_ops.openmp_enabled()),
            "avx2_enabled": bool(simd_ops.compiled_with_avx2()),
            "avx512_enabled": bool(simd_ops.compiled_with_avx512()),
            "amx_compiled": bool(simd_ops.compiled_with_amx()),
            "amx_runtime_available": bool(simd_ops.runtime_amx_available()),
            "affinity_visible_threads": int(_visible_logical_threads()),
            "openmp_visible_threads": int(simd_ops.get_num_procs()),
            "logical_core_count": int(
                getattr(self, "_logical_core_count", _visible_logical_threads())
            ),
            "physical_core_count": int(
                getattr(
                    self,
                    "_physical_core_count",
                    _physical_core_thread_hint(_visible_logical_threads()),
                )
            ),
            "p_core_count": int(getattr(self, "_p_core_count", 0)),
            "affinity_policy": str(getattr(self, "_affinity_policy", "")),
            "affinity_mode": int(getattr(self, "_affinity_mode", 1)),
            "l3_domain_count": int(getattr(self, "_l3_domain_count", 0)),
            "visible_l3_domains": visible_l3_domains,
            "batch_l3_domains": batch_l3_domains,
            "decode_primary_l3_domain": decode_primary_l3_domain,
            "preferred_decode_l3_domain": preferred_decode_l3_domain,
            "decode_domain_reserved": bool(
                affinity_plan.get("decode_domain_reserved", False)
            ),
            "numa_strict": bool(getattr(self, "_numa_strict", False)),
            "numa_affinity_mode": str(
                getattr(
                    self,
                    "_numa_affinity_mode",
                    os.getenv("ANVIL_NUMA_AFFINITY_MODE", ""),
                )
            ),
            "numa_hugepage": str(
                getattr(self, "_numa_hugepage", os.getenv("ANVIL_NUMA_HUGEPAGE", ""))
            ),
            "numa_bind_policy": str(
                getattr(
                    self, "_numa_bind_policy", os.getenv("ANVIL_NUMA_BIND_POLICY", "")
                )
            ),
            "numa_first_touch": bool(getattr(self, "_numa_first_touch", False)),
            "topology_json": str(getattr(self, "_topology_json", "")),
            "os_thread_migrations": int(getattr(self, "_os_thread_migrations", 0)),
            "os_last_cpu": int(getattr(self, "_os_last_cpu", -1)),
            "omp_places": str(
                getattr(self, "_omp_places", os.getenv("OMP_PLACES", ""))
            ),
            "omp_proc_bind": str(
                getattr(self, "_omp_proc_bind", os.getenv("OMP_PROC_BIND", ""))
            ),
            "omp_max_threads": int(omp_max_threads),
            "omp_dynamic": bool(omp_dynamic),
            "omp_active_levels": int(omp_active_levels),
            "batch_token_fallback_count": int(
                getattr(self, "_batch_token_fallback_count", 0)
            ),
            "tokenizer_model": str(metadata.get("tokenizer.ggml.model", "") or ""),
            "tokenizer_backend": str(getattr(self, "_tokenizer_backend", "native")),
            "backend_module": str(
                getattr(self, "_native_backend_info", {}).get("backend_module", "")
            ),
            "backend_module_requested": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_requested", ""
                )
            ),
            "backend_module_library": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_library", ""
                )
            ),
            "backend_module_loaded": bool(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_loaded", False
                )
            ),
            "backend_module_candidates": [
                str(candidate).strip()
                for candidate in getattr(self, "_native_backend_info", {}).get(
                    "backend_module_candidates", []
                )
                if str(candidate).strip()
            ],
            "backend_selection_source": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_selection_source", ""
                )
            ),
            "backend_selection_reason": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_selection_reason", ""
                )
            ),
            "backend_selection_model_name": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_selection_model_name", ""
                )
            ),
            "backend_selection_architecture": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_selection_architecture", ""
                )
            ),
            "backend_selection_family": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_selection_family", ""
                )
            ),
            "backend_module_marker_symbol": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_marker_symbol", ""
                )
            ),
            "backend_module_marker": int(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_marker", 0
                )
                or 0
            ),
            "backend_module_name_symbol": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_name_symbol", ""
                )
            ),
            "backend_module_name": str(
                getattr(self, "_native_backend_info", {}).get("backend_module_name", "")
            ),
            "backend_module_build_id_symbol": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_build_id_symbol", ""
                )
            ),
            "backend_module_build_id": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_build_id", ""
                )
            ),
            "backend_module_abi_symbol": str(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_abi_symbol", ""
                )
            ),
            "backend_module_abi_version": int(
                getattr(self, "_native_backend_info", {}).get(
                    "backend_module_abi_version", 0
                )
                or 0
            ),
            "chat_template": str(
                getattr(self._strict_prompt_contract, "template_name", "")
                or resolve_chat_template_name(
                    model_name,
                    profile=getattr(self, "profile", None),
                    strict=True,
                )
            ),
            "granite_moe_mode": str(getattr(self, "_granite_moe_mode", "")),
            "active_thread_mode": str(getattr(self, "_active_thread_mode", "")),
            "thread_mode_switches": int(getattr(self, "_runtime_thread_switches", 0)),
            "min_new_tokens_before_eos": int(
                getattr(self, "_min_new_tokens_before_eos", 0)
            ),
            "coherence_guard_events": int(getattr(self, "_coherence_guard_events", 0)),
            "sampling_backend": "native_simd",
            "native_fast_path": bool(getattr(self, "_native_fast_path", False)),
            "disable_logits_processors": bool(
                getattr(self, "_disable_logits_processors", False)
            ),
            "disable_token_penalties": bool(
                getattr(self, "_disable_token_penalties", False)
            ),
            "strict_cpp_only": bool(getattr(self, "_strict_cpp_only", False)),
            "graph_mode": bool(getattr(self, "_graph_mode", False)),
            "context_stabilizer_enabled": bool(
                getattr(self, "_tc_enabled", False)
                and getattr(self, "_tc_graph_drift_available", False)
            ),
            "context_stabilizer_mode": str(
                getattr(self, "_tc_current_mode", "telemetry")
            ),
            "drift_auto_downgrade_events": int(
                getattr(self, "_tc_auto_downgrade_events", 0)
            ),
            "drift_overhead_percent": float(
                getattr(self, "_tc_last_overhead_percent", 0.0)
            ),
            "coconut_enabled": bool(getattr(self, "_native_qsg_use_coconut", False)),
            "coconut_paths": int(getattr(self, "_native_qsg_coconut_paths", 0)),
            "coconut_alpha": float(getattr(self, "_native_qsg_coconut_alpha", 0.0)),
            "coconut_seconds": float(qsg_sampling_stats.get("coconut_seconds", 0.0)),
            "coconut_candidate_count": int(
                qsg_sampling_stats.get("coconut_candidate_count", 0)
            ),
            "coconut_entropy_mean": float(
                qsg_sampling_stats.get("coconut_entropy_mean", 0.0)
            ),
            "coconut_amplitude_mean": float(
                qsg_sampling_stats.get("coconut_amplitude_mean", 0.0)
            ),
            "coconut_consistency_rejects": int(
                qsg_sampling_stats.get("coconut_consistency_rejects", 0)
            ),
            "grammar_fastlane_calls": int(
                qsg_sampling_stats.get("grammar_fastlane_calls", 0)
            ),
            "grover_enabled": bool(getattr(self, "_native_qsg_use_grover", False)),
            "grover_top_k": int(getattr(self, "_native_qsg_grover_top_k", 0)),
            "grover_damping": float(getattr(self, "_native_qsg_grover_damping", 0.0)),
            "grover_calls": int(qsg_sampling_stats.get("grover_calls", 0)),
            "grover_seconds": float(qsg_sampling_stats.get("grover_seconds", 0.0)),
            "grover_candidate_count": int(
                qsg_sampling_stats.get("grover_candidate_count", 0)
            ),
            "grover_rescore_delta_mean": float(
                qsg_sampling_stats.get("grover_rescore_delta_mean", 0.0)
            ),
            "grover_timeout_events": int(
                qsg_sampling_stats.get("grover_timeout_events", 0)
            ),
            "penalties_backend": "native_simd",
            "suppression_backend": (
                "native_simd" if self._suppressed_token_ids_sorted else "none"
            ),
            "logits_backend": "native_graph_token_id",
            "strict_native_qsg": bool(
                PERFORMANCE_CONFIG.get("strict_native_qsg", False)
            ),
            "strict_path_stable": False,
            "parallel_decode_allowed": bool(
                getattr(self, "_parallel_decode_allowed", False)
            ),
            "native_parallel_kernels_ready": bool(
                getattr(self, "_native_parallel_kernels_ready", False)
            ),
            "block_diffusion_native_ready": bool(
                getattr(self, "_block_diffusion_native_ready", False)
            ),
            "masked_diffusion_native_ready": bool(
                getattr(self, "_masked_diffusion_native_ready", False)
            ),
            "masked_generation_ready": bool(
                getattr(self, "_masked_diffusion_native_ready", False)
            ),
            "medusa_head_ready": bool(getattr(self, "_medusa_head_ready", False)),
            "hydra_head_ready": bool(getattr(self, "_hydra_head_ready", False)),
            "medusa_head_enabled": bool(getattr(self, "_medusa_head_enabled", False)),
            "hydra_head_enabled": bool(getattr(self, "_hydra_head_enabled", False)),
            "masked_diffusion_enabled": bool(
                getattr(self, "_masked_diffusion_enabled", False)
            ),
            "medusa_head_count": int(
                getattr(getattr(self, "_medusa_head_config", None), "num_heads", 0)
            ),
            "hydra_head_count": int(
                getattr(getattr(self, "_hydra_head_config", None), "num_heads", 0)
            ),
            "parallel_replacement_enabled": bool(
                getattr(self, "_parallel_replacement_enabled", False)
            ),
            "generation_mode": GenerationMode.AR_VERIFY.value,
            "benchmark_label": benchmark_label_for_mode(GenerationMode.AR_VERIFY).value,
            "supported_benchmark_labels": list(
                getattr(
                    self, "_supported_benchmark_labels", supported_benchmark_labels()
                )
            ),
            "scheduler_queue_wait_ms": float(
                getattr(self, "_scheduler_queue_wait_ms", 0.0)
            ),
            "scheduler_iteration_ms": float(
                getattr(self, "_scheduler_iteration_ms", 0.0)
            ),
            "kv_fragmentation_ratio": float(
                getattr(self, "_scheduler_kv_fragmentation_ratio", 0.0)
            ),
            "qsg_state_pages_total": int(getattr(self, "_scheduler_kv_pages_total", 0)),
            "qsg_state_pages_in_use": int(
                getattr(self, "_scheduler_kv_pages_in_use", 0)
            ),
            "qsg_state_active_page_slots": int(
                getattr(self, "_scheduler_kv_active_page_slots", 0)
            ),
            "qsg_state_shared_page_slots": int(
                getattr(self, "_scheduler_kv_shared_page_slots", 0)
            ),
            "qsg_state_snapshot_count": int(
                getattr(self, "_scheduler_kv_snapshot_count", 0)
            ),
            "qsg_state_cow_events": int(getattr(self, "_scheduler_kv_cow_events", 0)),
            "qsg_state_prefix_share_events": int(
                getattr(self, "_scheduler_kv_prefix_share_events", 0)
            ),
            "qsg_state_active_tokens": int(
                getattr(self, "_scheduler_kv_active_tokens", 0)
            ),
            "qsg_state_committed_token_capacity": int(
                getattr(self, "_scheduler_kv_committed_token_capacity", 0)
            ),
            "qsg_state_page_tokens": int(getattr(self, "_scheduler_kv_page_tokens", 0)),
            "qsg_sequence_state_counts": dict(
                getattr(self, "_scheduler_sequence_state_counts", {})
            ),
            "qsg_sequence_mode_counts": dict(
                getattr(self, "_scheduler_sequence_mode_counts", {})
            ),
            "qsg_sequence_checkpoint_count": int(
                getattr(self, "_scheduler_sequence_checkpoint_count", 0)
            ),
            "qsg_tool_wait_requests": int(
                getattr(self, "_scheduler_tool_wait_requests", 0)
            ),
            "qsg_latent_packet_count": int(
                getattr(self, "_scheduler_latent_packet_count", 0)
            ),
            "qsg_evidence_capsule_count": int(
                getattr(self, "_scheduler_evidence_capsule_count", 0)
            ),
            "qsg_suspend_events": int(getattr(self, "_scheduler_suspend_events", 0)),
            "qsg_resume_events": int(getattr(self, "_scheduler_resume_events", 0)),
            "qsg_processors_native_enabled": bool(
                getattr(self, "_qsg_processors_native_enabled", False)
            ),
            "python_hot_path_calls": int(getattr(self, "_python_hot_path_calls", 0)),
            "numpy_hot_path_calls": int(getattr(self, "_numpy_hot_path_calls", 0)),
            "python_qsg_forward_calls": int(
                getattr(self, "_python_qsg_forward_calls", 0)
            ),
            "python_attention_fallback_calls": int(
                getattr(self, "_python_attention_fallback_calls", 0)
            ),
            "python_ssm_fallback_calls": int(
                getattr(self, "_python_ssm_fallback_calls", 0)
            ),
            "python_moe_fallback_calls": int(
                getattr(self, "_python_moe_fallback_calls", 0)
            ),
            "llama_cpp_hot_path_calls": int(
                getattr(self, "_llama_cpp_hot_path_calls", 0)
            ),
            "perf_event_access": bool(getattr(self, "_perf_event_access", False)),
            "perf_event_access_reason": str(
                getattr(self, "_perf_event_access_reason", "")
            ),
            "cpu_governor": str(getattr(self, "_cpu_governor", "")),
            "thp_mode": str(getattr(self, "_thp_mode", "")),
            "perf_counter_source": str(getattr(self, "_perf_counter_source", "")),
            "worker_cpu_mask": _cpu_mask_string(worker_cpus),
            "orchestrator_cpu_mask": _cpu_mask_string(orchestrator_cpus),
            "l3_domain_ids_active": _active_l3_domain_ids(
                topology,
                [*worker_cpus, *orchestrator_cpus],
            ),
            "autotune_profile_id": str(getattr(self, "_autotune_profile_id", "")),
            "autotune_source": str(getattr(self, "_autotune_source", "")),
            "autotune_score": float(getattr(self, "_autotune_score", 0.0)),
            "autotune_exploration_count": int(
                getattr(self, "_autotune_exploration_count", 0)
            ),
            "hot_path_proof": {
                "full_qsg": "disabled",
                "python_or_numpy_hot_path": "not_detected",
            },
        }
        capabilities["hot_path_numpy_detected"] = bool(
            capabilities["python_hot_path_calls"] > 0
            or capabilities["numpy_hot_path_calls"] > 0
            or capabilities["python_qsg_forward_calls"] > 0
            or capabilities["python_attention_fallback_calls"] > 0
            or capabilities["python_ssm_fallback_calls"] > 0
            or capabilities["python_moe_fallback_calls"] > 0
            or capabilities["llama_cpp_hot_path_calls"] > 0
        )
        capabilities["hot_path_proof"].update(
            {
                "sanctioned_backend_path": SANCTIONED_BACKEND_PATH,
                "tokenizer_backend": capabilities["tokenizer_backend"],
                "backend_module": str(capabilities["backend_module"]),
                "backend_module_library": str(capabilities["backend_module_library"]),
                "backend_module_loaded": (
                    "true" if bool(capabilities["backend_module_loaded"]) else "false"
                ),
                "backend_selection_source": str(
                    capabilities["backend_selection_source"]
                ),
                "backend_selection_reason": str(
                    capabilities["backend_selection_reason"]
                ),
                "coconut_enabled": (
                    "true" if bool(capabilities["coconut_enabled"]) else "false"
                ),
                "grover_enabled": (
                    "true" if bool(capabilities["grover_enabled"]) else "false"
                ),
                "perf_event_access": (
                    "true" if bool(capabilities["perf_event_access"]) else "false"
                ),
                "perf_event_access_reason": str(
                    capabilities["perf_event_access_reason"]
                ),
                "cpu_governor": str(capabilities["cpu_governor"]),
                "thp_mode": str(capabilities["thp_mode"]),
                "worker_cpu_mask": str(capabilities["worker_cpu_mask"]),
                "orchestrator_cpu_mask": str(capabilities["orchestrator_cpu_mask"]),
                "l3_domain_ids_active": ",".join(
                    str(value) for value in capabilities["l3_domain_ids_active"]
                ),
                "autotune_profile_id": str(capabilities["autotune_profile_id"]),
                "autotune_source": str(capabilities["autotune_source"]),
                "strict_cpp_only": (
                    "true" if bool(capabilities["strict_cpp_only"]) else "false"
                ),
                "strict_native_qsg": (
                    "true" if bool(capabilities["strict_native_qsg"]) else "false"
                ),
                "context_stabilizer_enabled": (
                    "true"
                    if bool(capabilities["context_stabilizer_enabled"])
                    else "false"
                ),
                "context_stabilizer_mode": str(capabilities["context_stabilizer_mode"]),
                "drift_overhead_percent": (
                    f"{float(capabilities['drift_overhead_percent']):.3f}"
                ),
                "python_or_numpy_hot_path": (
                    "detected"
                    if capabilities["hot_path_numpy_detected"]
                    else "not_detected"
                ),
                "executed_cpp_only": (
                    "true" if not capabilities["hot_path_numpy_detected"] else "false"
                ),
                "python_hot_path_calls": str(capabilities["python_hot_path_calls"]),
                "numpy_hot_path_calls": str(capabilities["numpy_hot_path_calls"]),
                "python_qsg_forward_calls": str(
                    capabilities["python_qsg_forward_calls"]
                ),
                "python_attention_fallback_calls": str(
                    capabilities["python_attention_fallback_calls"]
                ),
                "python_ssm_fallback_calls": str(
                    capabilities["python_ssm_fallback_calls"]
                ),
                "python_moe_fallback_calls": str(
                    capabilities["python_moe_fallback_calls"]
                ),
                "llama_cpp_hot_path_calls": str(
                    capabilities["llama_cpp_hot_path_calls"]
                ),
            }
        )
        graph = getattr(self, "_model_graph", None)
        if graph is not None:
            capabilities["lm_head_layout"] = str(
                getattr(graph, "_lm_head_layout", "unknown")
            )
            capabilities["lm_head_qtype"] = int(getattr(graph, "_lm_head_qtype", 0))
            capabilities["graph_batched_token_id_api"] = bool(
                getattr(graph, "_has_batch_token_id_api", False)
            )
            capabilities["full_graph_enabled"] = bool(graph.has_full_graph)
            capabilities["batched_prefill_native_enabled"] = bool(
                getattr(graph, "_has_batch_token_id_api", False)
            )
            capabilities["full_qsg_enabled"] = _full_qsg_enabled(
                full_graph_enabled=bool(graph.has_full_graph),
                qsg_processors_native_enabled=bool(
                    getattr(self, "_qsg_processors_native_enabled", False)
                ),
                batched_prefill_native_enabled=bool(
                    getattr(graph, "_has_batch_token_id_api", False)
                ),
                tokenizer_backend=str(capabilities["tokenizer_backend"]),
                sanctioned_backend_path=str(capabilities["sanctioned_backend_path"]),
            )
            capabilities["hot_path_proof"].update(
                {
                    "full_qsg": (
                        "enabled" if capabilities["full_qsg_enabled"] else "disabled"
                    ),
                    "graph_batched_token_id_api": (
                        "enabled"
                        if capabilities["graph_batched_token_id_api"]
                        else "disabled"
                    ),
                    "qsg_processors": (
                        "enabled"
                        if capabilities["qsg_processors_native_enabled"]
                        else "disabled"
                    ),
                }
            )
        else:
            capabilities["graph_batched_token_id_api"] = False
            capabilities["full_graph_enabled"] = False
            capabilities["batched_prefill_native_enabled"] = False
            capabilities["full_qsg_enabled"] = False
        capabilities["strict_path_stable"] = bool(
            capabilities.get("full_qsg_enabled", False)
            and not capabilities.get("hot_path_numpy_detected", False)
            and (
                bool(capabilities.get("strict_cpp_only", False))
                or bool(capabilities.get("strict_native_qsg", False))
            )
        )
        capabilities["hot_path_proof"]["strict_path_stable"] = (
            "true" if capabilities["strict_path_stable"] else "false"
        )
        capabilities.update(getattr(self, "_native_library_info", {}))
        capabilities.update(getattr(self, "_native_backend_info", {}))
        optional_isa_leaves = [
            str(item).strip()
            for item in capabilities.get("native_optional_isa_leaves", []) or []
            if str(item).strip()
        ]
        if not optional_isa_leaves:
            optional_isa_leaves = [
                item.strip()
                for item in str(
                    capabilities.get("native_optional_isa_leaves_csv", "")
                ).split(",")
                if item.strip()
            ]
        capabilities["native_optional_isa_leaves"] = list(
            dict.fromkeys(optional_isa_leaves)
        )
        capabilities["amx_compiled"] = bool(
            capabilities.get("amx_compiled")
            or capabilities.get("native_compiled_with_amx")
            or "amx" in capabilities["native_optional_isa_leaves"]
        )
        capabilities["amx_runtime_available"] = bool(
            capabilities.get("amx_runtime_available")
            or capabilities.get("native_runtime_amx_available")
        )
        capabilities["amx_enabled"] = bool(capabilities["amx_runtime_available"])
        capabilities["amx_kernel_enabled"] = bool(capabilities["amx_compiled"])
        split_abi = int(capabilities.get("native_split_abi_version", 0) or 0)
        backend_abi = int(capabilities.get("backend_module_abi_version", 0) or 0)
        capabilities["native_backend_abi_match"] = bool(
            bool(capabilities.get("backend_module_loaded", False))
            and split_abi > 0
            and backend_abi > 0
            and split_abi == backend_abi
        )
        model_graph_handle = int(
            getattr(getattr(self, "_model_graph", None), "_handle", 0) or 0
        )
        capabilities["native_runtime_abi_ready"] = bool(
            capabilities["native_backend_abi_match"]
            and bool(native_parallel_kernels_available())
            and model_graph_handle > 0
        )
        capabilities["hot_path_proof"]["native_isa_baseline"] = str(
            capabilities.get("native_isa_baseline", "")
        )
        capabilities["hot_path_proof"]["native_optional_isa_leaves"] = ",".join(
            capabilities.get("native_optional_isa_leaves", [])
        )
        capabilities["hot_path_proof"]["amx_leaf"] = (
            "enabled"
            if capabilities["amx_enabled"]
            else ("compiled" if capabilities["amx_kernel_enabled"] else "disabled")
        )
        capabilities["hot_path_proof"]["decode_primary_l3_domain"] = str(
            capabilities.get("decode_primary_l3_domain", -1)
        )
        capabilities["hot_path_proof"]["batch_l3_domains"] = ",".join(
            str(value) for value in capabilities.get("batch_l3_domains", [])
        )
        capabilities["hot_path_proof"]["native_backend_abi_match"] = (
            "true" if capabilities["native_backend_abi_match"] else "false"
        )
        return capabilities

    def _enforce_native_split_backend_abi(self) -> None:
        backend_info = getattr(self, "_native_backend_info", {}) or {}
        if not bool(backend_info.get("backend_module_loaded", False)):
            return
        split_abi = int(
            (getattr(self, "_native_library_info", {}) or {}).get(
                "native_split_abi_version", 0
            )
            or 0
        )
        backend_abi = int(backend_info.get("backend_module_abi_version", 0) or 0)
        backend_name = str(backend_info.get("backend_module", "") or "unknown")
        if split_abi <= 0:
            raise RuntimeError(
                "Strict native split ABI metadata is missing from "
                f"'{self.model_name}' (backend='{backend_name}')."
            )
        if backend_abi <= 0:
            raise RuntimeError(
                "Strict native backend ABI metadata is missing for "
                f"'{self.model_name}' (backend='{backend_name}')."
            )
        if split_abi != backend_abi:
            raise RuntimeError(
                "Strict native split/backend ABI mismatch for "
                f"'{self.model_name}' (backend='{backend_name}', "
                f"split_abi={split_abi}, backend_abi={backend_abi})."
            )

    def get_runtime_status(self) -> dict[str, Any]:
        status = dict(self._runtime_capabilities)
        telemetry = self._last_generation.as_dict()
        for key, value in telemetry.items():
            if key not in status:
                status[key] = value
                continue
            existing = status.get(key)
            if (
                key
                in {
                    "full_qsg_enabled",
                    "full_graph_enabled",
                    "qsg_processors_native_enabled",
                    "batched_prefill_native_enabled",
                }
                and isinstance(existing, bool)
                and existing
                and value is False
            ):
                continue
            if isinstance(existing, str) and value == "":
                continue
            if isinstance(existing, dict) and isinstance(value, dict):
                if not value:
                    continue
                merged = dict(existing)
                merged.update(value)
                if key == "hot_path_proof":
                    if (
                        existing.get("full_qsg") == "enabled"
                        and value.get("full_qsg") == "disabled"
                    ):
                        merged["full_qsg"] = "enabled"
                status[key] = merged
                continue
            if isinstance(existing, dict) and not value:
                continue
            status[key] = value
        capability_vector = RuntimeCapabilityVector.from_status(status)
        delta_watermark = DeltaWatermark.from_dict(
            getattr(self, "_delta_watermark", None)
        ).as_dict()
        status["capability_vector"] = capability_vector.as_dict()
        status["capability_digest"] = capability_vector.stable_digest()
        status["delta_watermark"] = delta_watermark
        performance_envelope = PerformanceEnvelope.from_runtime_status(
            status,
            capability_digest=capability_vector.stable_digest(),
            delta_watermark=delta_watermark,
        )
        frontier_decision = SpeculativeFrontierPolicy(
            mode=str(
                getattr(self, "_controller_frontier_mode", "adaptive") or "adaptive"
            )
        ).decide(status)
        drift_decision = DriftController(
            mode=str(getattr(self, "_controller_drift_mode", "adaptive") or "adaptive"),
            overhead_target_pct=float(getattr(self, "_tc_overhead_target_pct", 15.0)),
            overhead_max_pct=float(getattr(self, "_tc_overhead_max_pct", 20.0)),
            hysteresis=float(getattr(self, "_tc_hysteresis", 0.05)),
            preserve_head_tokens=int(getattr(self, "_tc_preserve_head_tokens", 256)),
            preserve_recent_tokens=int(
                getattr(self, "_tc_preserve_recent_tokens", 8192)
            ),
        ).decide(
            status,
            prompt_category=str(status.get("prompt_category") or ""),
            acceptance_ratio=float(performance_envelope.draft_acceptance_ratio),
        )
        memory_tier = MemoryTierPolicy(
            mode=str(
                getattr(self, "_memory_tier_policy_mode", "adaptive") or "adaptive"
            ),
            prompt_cache_hit_threshold=float(
                getattr(self, "_memory_prompt_cache_hit_threshold", 0.40)
            ),
            latent_replay_threshold=float(
                getattr(self, "_memory_latent_replay_threshold", 0.60)
            ),
            repo_delta_window=int(getattr(self, "_repo_delta_memory_window", 8)),
        ).decide(status, delta_watermark=delta_watermark)
        status["controller_state"] = {
            "frontier": frontier_decision.as_dict(),
            "drift": drift_decision.as_dict(),
            "memory_tier": memory_tier.as_dict(),
        }
        status["performance_envelope"] = performance_envelope.as_dict()
        status["performance_twin"] = (
            PerformanceTwinModel()
            .predict(
                envelope=performance_envelope,
                capability_vector=capability_vector,
                controller_state=status["controller_state"],
            )
            .as_dict()
        )
        config = getattr(self, "config", None)
        status["repo_coupled_runtime"] = {
            "capability_digest": capability_vector.stable_digest(),
            "delta_watermark": dict(performance_envelope.delta_watermark),
            "controller_state": dict(status["controller_state"]),
            "performance_twin": dict(status["performance_twin"]),
            "execution_capsule_version": int(
                getattr(config, "execution_capsule_version", 2)
            ),
            "latent_packet_abi_version": int(
                getattr(config, "latent_packet_abi_version", 2)
            ),
            "delta_authority": "state_ledger",
        }
        self._last_generation.controller_state = dict(status["controller_state"])
        self._last_generation.delta_watermark = dict(delta_watermark)
        self._last_generation.performance_twin = dict(status["performance_twin"])
        self._last_generation.repo_coupled_runtime = dict(
            status["repo_coupled_runtime"]
        )
        return status

    def get_last_run_metrics(self) -> dict[str, Any]:
        return self._last_generation.as_dict()

    def format_capability_banner(self) -> str:
        caps = self.get_runtime_status()
        vector = dict(caps.get("capability_vector") or {})
        frontier = dict((caps.get("controller_state") or {}).get("frontier") or {})
        drift = dict((caps.get("controller_state") or {}).get("drift") or {})
        return (
            "Native QSG"
            f" | model={caps['model']}"
            f" | digest={caps['digest'][:20]}"
            f" | arch={caps['architecture']}"
            f" | threads={caps['decode_threads']}/{caps['batch_threads']}"
            f" | ubatch={caps['ubatch']}"
            f" | mmap={'on' if caps['mmap_enabled'] else 'off'}"
            f" | openmp={'on' if caps['openmp_enabled'] else 'off'}"
            f" | avx2={'on' if caps['avx2_enabled'] else 'off'}"
            f" | isa={vector.get('native_isa_baseline', '')}"
            f" | frontier={frontier.get('selected_mode', '')}"
            f" | drift={drift.get('selected_mode', '')}"
        )

    def get_runtime_capability_ledger(self) -> dict[str, Any]:
        """Return a stable capability contract snapshot for governance surfaces."""
        return build_runtime_capability_ledger(
            self.get_runtime_status(),
            host_fingerprint=str(getattr(self, "_host_fingerprint", "")),
            source="native_qsg_engine",
        )

    def _refresh_os_thread_telemetry(self) -> None:
        try:
            self._os_thread_migrations = int(simd_ops.get_thread_migration_count())
        except Exception:
            self._os_thread_migrations = int(getattr(self, "_os_thread_migrations", 0))
        try:
            self._os_last_cpu = int(simd_ops.get_last_cpu())
        except Exception:
            self._os_last_cpu = int(getattr(self, "_os_last_cpu", -1))

    def _sample_os_thread_cpu(self) -> None:
        try:
            sampled_cpu = int(simd_ops.sample_thread_cpu())
        except Exception:
            return
        self._os_last_cpu = sampled_cpu

    def _set_runtime_thread_mode(self, mode: str) -> None:
        decode_threads = max(
            1,
            int(getattr(self, "num_threads_decode", getattr(self, "num_threads", 1))),
        )
        batch_threads = max(
            1,
            int(getattr(self, "num_threads_batch", decode_threads)),
        )
        target = decode_threads if mode == "decode" else batch_threads
        if int(getattr(self, "_active_thread_count", 0)) == target:
            self._active_thread_mode = mode
            return
        decode_path = mode == "decode"
        os.environ["ANVIL_NUM_THREADS"] = str(target)
        try:
            simd_ops.set_thread_mode(decode_path=decode_path)
        except Exception:
            pass
        try:
            simd_ops.set_num_threads(target)
        except Exception:
            pass
        self._active_thread_count = target
        self._active_thread_mode = mode
        self._runtime_thread_switches = (
            int(getattr(self, "_runtime_thread_switches", 0)) + 1
        )

    def tokenize(self, text: str) -> list[int]:
        started = time.perf_counter()
        tokens = self.tokenizer.encode(text, add_bos=self._add_bos)
        self._last_tokenize_seconds = time.perf_counter() - started
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special=True)

    def finalize_response_text(self, text: str) -> str:
        return postprocess_strict_native_response(
            text,
            model_name=str(
                self.contract.get("model")
                or getattr(self.profile, "model_name", "")
                or ""
            ),
            template_name=str(
                getattr(self._strict_prompt_contract, "template_name", "")
                or getattr(self.profile, "chat_template", "")
                or ""
            ),
        )

    def decode_generated_tokens(self, tokens: list[int]) -> str:
        return self.finalize_response_text(self.detokenize(tokens))

    def format_prompt(self, prompt: str) -> str:
        started = time.perf_counter()
        model_name = str(
            self.contract.get("model") or getattr(self.profile, "model_name", "") or ""
        )
        formatted = format_strict_native_prompt(
            prompt,
            self._strict_prompt_contract.template_name,
            model_name=model_name,
            system_prompt=getattr(self._strict_prompt_contract, "system_prompt", None),
            assistant_prefix=getattr(
                self._strict_prompt_contract, "assistant_prefix", None
            ),
        )
        self._last_prompt_format_seconds = time.perf_counter() - started
        return formatted

    def prepare_prompt_tokens(self, prompt: str) -> list[int]:
        self._last_prompt_category = _classify_prompt_category(prompt)
        return self.tokenize(self.format_prompt(prompt))

    def token_eos(self) -> int:
        return self._eos_token

    def _ensure_forward_pass(self) -> QSGForwardPass:
        self._python_hot_path_calls = (
            int(getattr(self, "_python_hot_path_calls", 0)) + 1
        )
        self._python_qsg_forward_calls = (
            int(getattr(self, "_python_qsg_forward_calls", 0)) + 1
        )
        raise RuntimeError(
            "Strict native production mode disables Python forward-pass helpers. "
            "Use token generation APIs only."
        )

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("Strict native production mode disables embedding helpers.")

    def reset_kv_cache(self) -> None:
        if self._model_graph is not None:
            self._model_graph.reset()
        if self.forward_pass is not None:
            self.forward_pass.reset()
        self._clear_logits_cache()

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    def _get_logits(self, token_ids: list[int], start_pos: int = 0) -> Any:
        """Get logits for the last token position.

        Unified strict mode:
        - all supported architectures use full C++ graph execution
        - Python layer fallbacks are disabled
        """
        if not token_ids:
            return _zero_logits(self.profile.vocab_size)

        self._set_runtime_thread_mode("decode" if len(token_ids) == 1 else "batch")
        graph = self._model_graph
        if graph is None:
            raise RuntimeError(
                "NativeModelGraph is required for native-only logits execution."
            )

        if graph.has_full_graph:
            last_logits: Optional[Any] = None
            graph_started = time.perf_counter()
            batch_token_forward = getattr(graph, "forward_token_ids", None)
            token_forward = getattr(graph, "forward_token_id", None)
            if (
                len(token_ids) > 1
                and bool(getattr(self, "_graph_batch_token_id_enabled", True))
                and callable(batch_token_forward)
            ):
                logits = batch_token_forward(token_ids, start_pos)
                if logits is not None:
                    last_logits = logits
                else:
                    self._graph_batch_token_id_enabled = False
                    self._batch_token_fallback_count = (
                        int(getattr(self, "_batch_token_fallback_count", 0)) + 1
                    )
            if (
                last_logits is None
                and bool(getattr(self, "_graph_token_id_enabled", True))
                and callable(token_forward)
            ):
                for i, token_id in enumerate(token_ids):
                    logits = token_forward(int(token_id), start_pos + i)
                    if logits is None:
                        self._graph_token_id_enabled = False
                        last_logits = None
                        break
                    last_logits = logits
            if last_logits is None:
                raise RuntimeError(
                    "Strict native generation requires a valid token-id graph execution path. "
                    "Python/Numpy embedding lookup fallback is disabled."
                )
            elapsed = time.perf_counter() - graph_started
            if len(token_ids) == 1:
                self._last_graph_decode_seconds = (
                    float(getattr(self, "_last_graph_decode_seconds", 0.0)) + elapsed
                )
                self._last_graph_decode_calls = (
                    int(getattr(self, "_last_graph_decode_calls", 0)) + 1
                )
            else:
                self._last_graph_prefill_seconds = (
                    float(getattr(self, "_last_graph_prefill_seconds", 0.0)) + elapsed
                )
                self._last_graph_prefill_calls = (
                    int(getattr(self, "_last_graph_prefill_calls", 0)) + 1
                )
            self._pull_timecrystal_snapshot(
                decode_step=(len(token_ids) == 1),
                graph_elapsed_seconds=elapsed,
            )
            if last_logits is not None:
                if (
                    bool(getattr(self, "_native_fast_path", False))
                    and bool(getattr(self, "_disable_logits_processors", False))
                    and bool(getattr(self, "_disable_token_penalties", False))
                ):
                    return last_logits
                return _sanitize_logits(last_logits)

        raise RuntimeError(
            "Native-only logits execution has no valid graph path for "
            f"architecture '{self.architecture}'."
        )

    def _get_logits_hybrid(
        self,
        token_ids: list[int],
        start_pos: int,
    ) -> list[float]:
        _ = token_ids
        _ = start_pos
        raise RuntimeError(
            "Hybrid Python dispatch is disabled in strict native C++ mode."
        )

    def _get_logits_for_tokens(self, token_ids: list[int]) -> list[float]:
        """Compatibility hook used by JacobiDecoder verification passes."""
        tokens = [int(t) for t in token_ids]
        if not tokens:
            return _zero_logits(self.profile.vocab_size)
        key = tuple(tokens)
        cached_logits = self._prefix_logits_cache.get(key)
        if cached_logits is not None:
            self._prefix_cache_hits += 1
            self._prompt_cache_reused_tokens = max(
                self._prompt_cache_reused_tokens,
                len(tokens),
            )
            return list(cached_logits)

        cached = self._cached_logits_tokens
        if cached and len(tokens) >= len(cached) and tokens[: len(cached)] == cached:
            if len(tokens) == len(cached) and self._cached_logits is not None:
                self._prefix_cache_hits += 1
                self._prompt_cache_reused_tokens = max(
                    self._prompt_cache_reused_tokens,
                    len(cached),
                )
                return list(self._cached_logits)

            delta = tokens[len(cached) :]
            logits = self._get_logits(delta, start_pos=len(cached))
            self._cached_logits_tokens = list(tokens)
            self._cached_logits = [float(value) for value in logits]
            self._prefix_logits_cache[key] = list(self._cached_logits)
            self._prefix_cache_hits += 1
            self._prompt_cache_reused_tokens = max(
                self._prompt_cache_reused_tokens,
                len(cached),
            )
            return list(self._cached_logits)

        if self._model_graph is not None:
            self._model_graph.reset()
        self._prefix_cache_misses += 1
        logits = self._get_logits(tokens, start_pos=0)
        self._cached_logits_tokens = list(tokens)
        self._cached_logits = [float(value) for value in logits]
        self._prefix_logits_cache[key] = list(self._cached_logits)
        return list(self._cached_logits)

    def _copy_last_hidden_from_graph(self) -> Any:
        graph = getattr(self, "_model_graph", None)
        if graph is None:
            return None
        copy_hidden = getattr(graph, "copy_last_hidden", None)
        if not callable(copy_hidden):
            return None
        try:
            return copy_hidden()
        except Exception:
            return None

    def _get_hidden_and_logits_for_tokens(
        self, token_ids: list[int]
    ) -> tuple[Any, Any]:
        tokens = [int(token) for token in token_ids]
        if not tokens:
            return None, np.asarray(
                _zero_logits(self.profile.vocab_size), dtype=np.float32
            )

        if self._model_graph is not None:
            self._model_graph.reset()
        logits = np.asarray(self._get_logits(tokens, start_pos=0), dtype=np.float32)
        hidden = self._copy_last_hidden_from_graph()
        self._cached_logits_tokens = list(tokens)
        self._cached_logits = [float(value) for value in logits]
        self._prefix_logits_cache[tuple(tokens)] = list(self._cached_logits)
        if hidden is None:
            return None, logits
        return np.asarray(hidden, dtype=np.float32), logits

    def _supports_native_self_spec(self) -> bool:
        graph = getattr(self, "_model_graph", None)
        return bool(
            getattr(self, "_self_spec_native_supported", False)
            and graph is not None
            and getattr(graph, "supports_exit_continuation", True)
        )

    def _select_self_spec_exit_layer(
        self,
        *,
        prompt_tokens: list[int],
        temperature: float,
    ) -> int:
        forced = getattr(self, "_self_spec_force_exit_layer", None)
        max_exit = max(
            1, min(int(self.n_layer) - 1, int(self._self_spec_exit_layer_max))
        )
        min_exit = max(1, min(int(self._self_spec_exit_layer_min), max_exit))
        if forced is not None:
            layer = max(min_exit, min(int(forced), max_exit))
            self._last_self_spec_policy = "forced"
            self._last_self_spec_exit_layer = layer
            self._last_self_spec_exit_fraction = float(layer) / float(
                max(1, self.n_layer)
            )
            return layer

        attempts = int(getattr(self, "_last_speculative_accept_count", 0)) + int(
            getattr(self, "_last_speculative_reject_count", 0)
        )
        recent_acceptance = (
            float(getattr(self, "_last_speculative_accept_count", 0)) / float(attempts)
            if attempts > 0
            else 0.5
        )
        temp = max(0.0, min(1.5, float(temperature)))
        temp_factor = 1.0 - (temp / 1.5)
        prompt_factor = 0.15 if len(prompt_tokens) >= 256 else 0.0
        acceptance_factor = (recent_acceptance - 0.5) * 0.5
        normalized = max(
            0.0,
            min(
                1.0,
                0.55 + temp_factor * 0.20 + prompt_factor + acceptance_factor,
            ),
        )
        span = max(0, max_exit - min_exit)
        layer = min_exit + int(round(span * normalized))
        layer = max(min_exit, min(layer, max_exit))
        self._last_self_spec_policy = "heuristic"
        self._last_self_spec_exit_layer = layer
        self._last_self_spec_exit_fraction = float(layer) / float(max(1, self.n_layer))
        return layer

    def _prefill_native_self_spec_context(
        self,
        *,
        prompt_tokens: list[int],
        exit_layer: int,
    ) -> tuple[Any, Any, int]:
        graph = getattr(self, "_model_graph", None)
        if graph is None or not prompt_tokens:
            return None, None, 0

        position = 0
        if len(prompt_tokens) > 1:
            prefix = [int(token) for token in prompt_tokens[:-1]]
            _ = self._get_logits(prefix, start_pos=0)
            position = len(prefix)

        exit_hidden = graph.forward_token_id_to_exit(
            int(prompt_tokens[-1]),
            int(exit_layer),
            int(position),
        )
        if exit_hidden is None:
            return None, None, position
        logits = graph.continue_from_hidden(
            exit_hidden,
            start_layer=int(exit_layer),
            position=int(position),
        )
        if logits is None:
            return None, None, position
        self._cached_logits_tokens = list(prompt_tokens)
        self._cached_logits = [float(value) for value in logits]
        self._prefix_logits_cache[tuple(prompt_tokens)] = list(self._cached_logits)
        return (
            np.asarray(exit_hidden, dtype=np.float32),
            np.asarray(logits, dtype=np.float32),
            position + 1,
        )

    def _advance_native_self_spec_token(
        self,
        *,
        token_id: int,
        exit_layer: int,
        position: int,
    ) -> tuple[Any, Any]:
        graph = getattr(self, "_model_graph", None)
        if graph is None:
            return None, None
        exit_hidden = graph.forward_token_id_to_exit(
            int(token_id),
            int(exit_layer),
            int(position),
        )
        if exit_hidden is None:
            return None, None
        logits = graph.continue_from_hidden(
            exit_hidden,
            start_layer=int(exit_layer),
            position=int(position),
        )
        if logits is None:
            return None, None
        return np.asarray(exit_hidden, dtype=np.float32), np.asarray(
            logits, dtype=np.float32
        )

    def _draft_model_head_bundle(
        self,
        *,
        head_type: str,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> DraftCandidateBundle:
        override_candidates = self.__dict__.get("_draft_model_head_candidates")

        def _legacy_override_bundle() -> DraftCandidateBundle:
            if not callable(override_candidates):
                return DraftCandidateBundle()
            tokens = override_candidates(
                head_type=head_type,
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
            )
            return DraftCandidateBundle(
                tokens=[int(token) for token in list(tokens or ())],
                source=f"{str(head_type or '').strip().lower()}_head_native",
            )

        kind = str(head_type or "").strip().lower()
        if kind == "medusa":
            config = getattr(self, "_medusa_head_config", None)
            if not bool(getattr(self, "_medusa_head_ready", False)) or config is None:
                return _legacy_override_bundle()
        elif kind == "hydra":
            config = getattr(self, "_hydra_head_config", None)
            if not bool(getattr(self, "_hydra_head_ready", False)) or config is None:
                return _legacy_override_bundle()
        else:
            return DraftCandidateBundle()

        hidden, base_logits = self._get_hidden_and_logits_for_tokens(prompt_tokens)
        if hidden is None or base_logits is None:
            return DraftCandidateBundle()

        limit = min(max(0, int(max_new_tokens)), int(config.num_heads))
        if limit <= 0:
            return DraftCandidateBundle()
        seed = (
            (len(prompt_tokens) << 32)
            ^ max(0, int(max_new_tokens))
            ^ (int(time.time_ns()) & 0xFFFFFFFF)
        )
        if kind == "medusa":
            tokens, probabilities = qsg_medusa_head_draft(
                hidden,
                config.weights,
                config.bias,
                num_heads=int(config.num_heads),
                hidden_dim=int(config.hidden_dim),
                vocab_size=int(config.vocab_size),
                draft_tokens=limit,
                temperature=float(max(temperature, 1.0e-6)),
                top_k=max(
                    1, int(top_k) if int(top_k) > 0 else int(self._medusa_head_top_k)
                ),
                min_probability=float(
                    getattr(self, "_medusa_head_acceptance_floor", 0.20)
                ),
                seed=int(seed),
            )
            return DraftCandidateBundle(
                tokens=[int(token) for token in tokens],
                probabilities=[float(probability) for probability in probabilities],
                source="medusa_head_native",
            )

        tokens, probabilities = qsg_hydra_head_draft(
            hidden,
            base_logits,
            config.weights,
            config.bias,
            num_heads=int(config.num_heads),
            hidden_dim=int(config.hidden_dim),
            vocab_size=int(config.vocab_size),
            draft_tokens=limit,
            temperature=float(max(temperature, 1.0e-6)),
            top_k=max(1, int(top_k) if int(top_k) > 0 else int(self._hydra_head_top_k)),
            blend_alpha=float(getattr(self, "_hydra_head_blend_alpha", 0.55)),
            min_probability=float(getattr(self, "_hydra_head_acceptance_floor", 0.22)),
            seed=int(seed),
        )
        return DraftCandidateBundle(
            tokens=[int(token) for token in tokens],
            probabilities=[float(probability) for probability in probabilities],
            source="hydra_head_native",
        )

    def _draft_model_head_candidates(
        self,
        *,
        head_type: str,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[int]:
        bundle = self._draft_model_head_bundle(
            head_type=head_type,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return [int(token) for token in bundle.tokens]

    @staticmethod
    def _apply_logits_processors(
        input_ids: list[int],
        logits: Sequence[float] | Any,
        logits_processor,
    ) -> Any:
        scores = _sanitize_logits(logits)
        for processor in logits_processor:
            next_scores = processor(input_ids, scores)
            if next_scores is None:
                continue
            scores = _sanitize_logits(next_scores)
        return scores

    @staticmethod
    def _apply_token_penalties(
        logits: Sequence[float] | Any,
        token_history: list[int],
        presence_penalty: float,
        repetition_penalty: float,
    ) -> list[float]:
        adjusted = list(_float_list(logits))
        if not adjusted or not token_history:
            return adjusted

        vocab_size = len(adjusted)
        presence = float(presence_penalty)
        repetition = float(repetition_penalty)

        if presence != 0.0:
            seen = set(int(tid) for tid in token_history if 0 <= int(tid) < vocab_size)
            for tid in seen:
                adjusted[tid] -= presence

        if repetition > 0.0 and abs(repetition - 1.0) > 1.0e-6:
            counts: dict[int, int] = {}
            for tid in token_history:
                tid_int = int(tid)
                if 0 <= tid_int < vocab_size:
                    counts[tid_int] = counts.get(tid_int, 0) + 1
            for tid, count in counts.items():
                scale = repetition ** max(1, count)
                if adjusted[tid] > 0.0:
                    adjusted[tid] /= scale
                else:
                    adjusted[tid] *= scale

        return _sanitize_logits(adjusted)

    @staticmethod
    def _apply_token_penalties_native(
        logits: Sequence[float] | Any,
        token_history,
        presence_penalty: float,
        repetition_penalty: float,
    ) -> Any:
        presence = float(presence_penalty)
        repetition = float(repetition_penalty)
        if abs(presence) <= 1.0e-12 and abs(repetition - 1.0) <= 1.0e-6:
            return logits
        return simd_ops.apply_token_penalties_inplace(
            logits,
            token_history,
            presence_penalty=presence,
            repetition_penalty=repetition,
        )

    def _grammar_fastlane_allowed_ids(self, token_history) -> array:
        if str(getattr(self, "_last_prompt_category", "") or "") != "structured":
            return array("i")
        open_ids = getattr(self, "_json_fastlane_open_ids", array("i"))
        follow_ids = getattr(self, "_json_fastlane_object_follow_ids", array("i"))
        if len(open_ids) == 0 or len(follow_ids) == 0:
            return array("i")
        if not token_history:
            return array("i", open_ids)
        try:
            current_text = self.finalize_response_text(
                self.detokenize(list(token_history))
            ).lstrip()
        except Exception:
            return array("i")
        if not current_text:
            return array("i", open_ids)
        if current_text.rstrip() == "{":
            return array("i", follow_ids)
        return array("i")

    def _sample(
        self,
        logits: Sequence[float] | Any,
        token_history=(),
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        *,
        leading_response: bool = False,
        disallow_eos: bool = False,
    ) -> int:
        scores = logits
        effective_presence_penalty = float(presence_penalty)
        effective_repetition_penalty = float(repetition_penalty)
        architecture = str(getattr(self, "architecture", "") or "")
        if architecture == "qwen35":
            effective_presence_penalty = max(0.10, effective_presence_penalty)
            effective_repetition_penalty = max(1.10, effective_repetition_penalty)
        if leading_response:
            suppressed_ids = getattr(self, "_leading_suppressed_token_ids_buffer", None)
        else:
            suppressed_ids = getattr(self, "_suppressed_token_ids_buffer", None)
        if suppressed_ids is None:
            fallback = {
                int(token_id) for token_id in getattr(self, "_suppressed_token_ids", ())
            }
            if leading_response:
                fallback.update(
                    int(token_id)
                    for token_id in getattr(self, "_leading_disallowed_token_ids", ())
                )
                fallback.update(
                    int(token_id)
                    for token_id in getattr(self, "_leading_suppressed_token_ids", ())
                )
            fallback_ids = sorted(fallback)
            suppressed_ids = array("i", fallback_ids)
        eos_token = int(self.token_eos())
        if disallow_eos:
            self._coherence_guard_events = (
                int(getattr(self, "_coherence_guard_events", 0)) + 1
            )
            if eos_token >= 0:
                if suppressed_ids is None:
                    suppressed_ids = array("i", [])
                suppressed_set = {int(token_id) for token_id in suppressed_ids}
                suppressed_set.add(eos_token)
                suppressed_ids = array("i", sorted(suppressed_set))
        grammar_allowed_ids = self._grammar_fastlane_allowed_ids(token_history)
        sample_started = time.perf_counter()
        if temperature <= 0.0:
            token = int(
                simd_ops.qsg_postprocess_and_sample(
                    scores,
                    suppressed_ids=suppressed_ids or (),
                    token_history=token_history,
                    grammar_allowed_ids=grammar_allowed_ids or (),
                    use_coconut=False,
                    coconut_paths=1,
                    coconut_alpha=0.0,
                    use_grover=False,
                    grover_top_k=0,
                    grover_damping=0.0,
                    presence_penalty=effective_presence_penalty,
                    repetition_penalty=effective_repetition_penalty,
                    no_repeat_ngram_size=0,
                    temperature=0.0,
                    eos_token=eos_token,
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                )
            )
            elapsed = time.perf_counter() - sample_started
            self._last_sample_seconds = (
                float(getattr(self, "_last_sample_seconds", 0.0)) + elapsed
            )
            self._last_sample_calls = int(getattr(self, "_last_sample_calls", 0)) + 1
            self._last_logits_processor_calls = (
                int(getattr(self, "_last_logits_processor_calls", 0)) + 1
            )
            self._last_penalty_calls = int(getattr(self, "_last_penalty_calls", 0)) + 1
            if suppressed_ids is not None and len(suppressed_ids) > 0:
                self._last_suppression_calls = (
                    int(getattr(self, "_last_suppression_calls", 0)) + 1
                )
            if grammar_allowed_ids:
                self._last_grammar_fastlane_calls = (
                    int(getattr(self, "_last_grammar_fastlane_calls", 0)) + 1
                )
            return token
        token = int(
            simd_ops.qsg_postprocess_and_sample(
                scores,
                suppressed_ids=suppressed_ids or (),
                token_history=token_history,
                grammar_allowed_ids=grammar_allowed_ids or (),
                use_coconut=bool(getattr(self, "_native_qsg_use_coconut", False)),
                coconut_paths=int(getattr(self, "_native_qsg_coconut_paths", 1)),
                coconut_alpha=float(getattr(self, "_native_qsg_coconut_alpha", 0.0)),
                use_grover=bool(getattr(self, "_native_qsg_use_grover", False)),
                grover_top_k=int(getattr(self, "_native_qsg_grover_top_k", 0)),
                grover_damping=float(getattr(self, "_native_qsg_grover_damping", 0.0)),
                presence_penalty=effective_presence_penalty,
                repetition_penalty=effective_repetition_penalty,
                no_repeat_ngram_size=int(
                    getattr(self, "_native_no_repeat_ngram_size", 0)
                ),
                temperature=float(temperature),
                eos_token=eos_token,
                top_p=float(top_p),
                top_k=int(top_k),
                min_p=float(min_p),
            )
        )
        elapsed = time.perf_counter() - sample_started
        self._last_sample_seconds = (
            float(getattr(self, "_last_sample_seconds", 0.0)) + elapsed
        )
        self._last_sample_calls = int(getattr(self, "_last_sample_calls", 0)) + 1
        self._last_logits_processor_calls = (
            int(getattr(self, "_last_logits_processor_calls", 0)) + 1
        )
        self._last_penalty_calls = int(getattr(self, "_last_penalty_calls", 0)) + 1
        if suppressed_ids is not None and len(suppressed_ids) > 0:
            self._last_suppression_calls = (
                int(getattr(self, "_last_suppression_calls", 0)) + 1
            )
        if grammar_allowed_ids:
            self._last_grammar_fastlane_calls = (
                int(getattr(self, "_last_grammar_fastlane_calls", 0)) + 1
            )
        return token

    def _generate_ssd_bridge(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        logits_processor: Optional[list[Any]],
        generation_started: float,
    ) -> list[int]:
        telemetry = NativeGenerationTelemetry(
            prompt_tokens=len(prompt_tokens),
            kv_used_cells=len(prompt_tokens),
            stop_reason="max_new_tokens" if max_new_tokens > 0 else "prompt_only",
        )
        active_logits_processor = (
            None
            if (self._native_fast_path and self._disable_logits_processors)
            else logits_processor
        )
        self._last_self_spec_native_path = False
        self._last_self_spec_draft_tokens = 0
        self._last_self_spec_policy = ""
        self._last_self_spec_exit_layer = 0
        self._last_self_spec_exit_fraction = 0.0
        self._last_speculative_accept_count = 0
        self._last_speculative_reject_count = 0
        generated: list[int] = []
        if (
            self._supports_native_self_spec()
            and active_logits_processor is None
            and prompt_tokens
        ):
            if self._model_graph is not None:
                self._model_graph.reset()
            self._clear_logits_cache()
            self._reset_generation_counters()
            exit_layer = self._select_self_spec_exit_layer(
                prompt_tokens=list(prompt_tokens),
                temperature=float(temperature),
            )
            exit_hidden, verifier_logits, position = (
                self._prefill_native_self_spec_context(
                    prompt_tokens=list(prompt_tokens),
                    exit_layer=int(exit_layer),
                )
            )
            if exit_hidden is None or verifier_logits is None:
                raise RuntimeError(
                    "Native self-spec prefill could not capture exit hidden state."
                )
            graph = getattr(self, "_model_graph", None)
            if graph is None:
                raise RuntimeError("Native self-spec requires an active model graph.")
            architecture = str(getattr(self, "architecture", "") or "")
            effective_presence_penalty = float(0.0)
            effective_repetition_penalty = float(1.0)
            if architecture == "qwen35":
                effective_presence_penalty = max(0.10, effective_presence_penalty)
                effective_repetition_penalty = max(1.10, effective_repetition_penalty)
            use_coconut = bool(getattr(self, "_native_qsg_use_coconut", False))
            coconut_paths = int(getattr(self, "_native_qsg_coconut_paths", 1))
            coconut_alpha = float(getattr(self, "_native_qsg_coconut_alpha", 0.0))
            use_grover = bool(getattr(self, "_native_qsg_use_grover", False))
            grover_top_k = int(getattr(self, "_native_qsg_grover_top_k", 0))
            grover_damping = float(getattr(self, "_native_qsg_grover_damping", 0.0))
            no_repeat_ngram_size = int(getattr(self, "_native_no_repeat_ngram_size", 0))
            token_history_buf = array("i", (int(token) for token in prompt_tokens))
            accepted = 0
            rejected = 0
            drafted = 0
            for token_index in range(int(max_new_tokens)):
                draft_logits = graph.forward_head(exit_hidden)
                if draft_logits is None:
                    raise RuntimeError(
                        "Native self-spec could not project exit hidden state."
                    )
                disallow_eos = self._should_block_eos(len(generated))
                draft_token = self._sample(
                    draft_logits,
                    token_history_buf,
                    temperature=float(temperature),
                    leading_response=(token_index == 0),
                    disallow_eos=disallow_eos,
                )
                drafted += 1
                suppressed_ids = (
                    getattr(self, "_leading_suppressed_token_ids_buffer", None)
                    if token_index == 0
                    else getattr(self, "_suppressed_token_ids_buffer", None)
                )
                if suppressed_ids is None:
                    suppressed_ids = sorted(
                        {
                            int(token_id)
                            for token_id in (
                                getattr(
                                    self,
                                    "_suppressed_token_ids",
                                    (),
                                )
                                or ()
                            )
                            if int(token_id) >= 0
                        }
                    )
                    if token_index == 0:
                        suppressed_ids = sorted(
                            {
                                int(token_id)
                                for token_id in (
                                    tuple(suppressed_ids)
                                    + tuple(
                                        getattr(
                                            self,
                                            "_leading_disallowed_token_ids",
                                            (),
                                        )
                                        or ()
                                    )
                                    + tuple(
                                        getattr(
                                            self,
                                            "_leading_suppressed_token_ids",
                                            (),
                                        )
                                        or ()
                                    )
                                )
                            }
                        )
                    suppressed_ids = array("i", suppressed_ids)
                if disallow_eos:
                    eos_token = int(self.token_eos())
                    if eos_token >= 0:
                        suppressed_ids = array("i", (*list(suppressed_ids), eos_token))

                _, draft_prob = simd_ops.qsg_postprocess_and_score(
                    draft_logits,
                    suppressed_ids=suppressed_ids,
                    token_history=token_history_buf,
                    use_coconut=use_coconut,
                    coconut_paths=coconut_paths,
                    coconut_alpha=coconut_alpha,
                    use_grover=use_grover,
                    grover_top_k=grover_top_k,
                    grover_damping=grover_damping,
                    presence_penalty=effective_presence_penalty,
                    repetition_penalty=effective_repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    temperature=max(float(temperature), 1.0e-6),
                    eos_token=int(self.token_eos()),
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    token_id=int(draft_token),
                )
                greedy_token, verifier_prob = simd_ops.qsg_postprocess_and_score(
                    verifier_logits,
                    suppressed_ids=suppressed_ids,
                    token_history=token_history_buf,
                    use_coconut=use_coconut,
                    coconut_paths=coconut_paths,
                    coconut_alpha=coconut_alpha,
                    use_grover=use_grover,
                    grover_top_k=grover_top_k,
                    grover_damping=grover_damping,
                    presence_penalty=effective_presence_penalty,
                    repetition_penalty=effective_repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    temperature=max(float(temperature), 1.0e-6),
                    eos_token=int(self.token_eos()),
                    top_p=1.0,
                    top_k=0,
                    min_p=0.0,
                    token_id=int(draft_token),
                )
                accept_ratio = min(1.0, verifier_prob / max(draft_prob, 1.0e-8))
                if int(greedy_token) == int(draft_token) or accept_ratio >= float(
                    self._self_spec_acceptance_threshold
                ):
                    chosen_token = int(draft_token)
                    accepted += 1
                else:
                    chosen_token = int(
                        self._sample(
                            verifier_logits,
                            token_history_buf,
                            temperature=float(temperature),
                            leading_response=(token_index == 0),
                            disallow_eos=disallow_eos,
                        )
                    )
                    rejected += 1
                generated.append(chosen_token)
                token_history_buf.append(int(chosen_token))
                if token_index == 0:
                    telemetry.first_token_latency_seconds = (
                        time.perf_counter() - generation_started
                    )
                if chosen_token == self.token_eos():
                    break
                exit_hidden, verifier_logits = self._advance_native_self_spec_token(
                    token_id=int(chosen_token),
                    exit_layer=int(exit_layer),
                    position=int(position),
                )
                if exit_hidden is None or verifier_logits is None:
                    raise RuntimeError(
                        "Native self-spec could not continue verifier tail."
                    )
                position += 1
                context_tokens = list(prompt_tokens) + list(generated)
                self._cached_logits_tokens = list(context_tokens)
                self._cached_logits = [float(value) for value in verifier_logits]
                self._prefix_logits_cache[tuple(context_tokens)] = list(
                    self._cached_logits
                )
            self._last_speculative_accept_count = int(accepted)
            self._last_speculative_reject_count = int(rejected)
            self._last_self_spec_native_path = True
            self._last_self_spec_draft_tokens = int(drafted)
            telemetry.generation_mode = GenerationMode.SSD_BRIDGE.value
            telemetry.benchmark_label = benchmark_label_for_mode(
                GenerationMode.SSD_BRIDGE
            ).value
        else:
            if self._model_graph is not None:
                self._model_graph.reset()
            self._clear_logits_cache()
            if self._ssm_spec_decoder is None:
                raise RuntimeError(
                    "SSD bridge requested, but neither native exit continuation "
                    "nor SSM speculative fallback is available."
                )
            generated = list(
                self._ssm_spec_decoder.generate(
                    prompt_tokens=list(prompt_tokens),
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    logits_processor=active_logits_processor,
                )
            )
            self._last_self_spec_native_path = False
            self._last_self_spec_draft_tokens = int(len(generated))
            telemetry.generation_mode = GenerationMode.SSD_BRIDGE.value
            telemetry.benchmark_label = benchmark_label_for_mode(
                GenerationMode.SSD_BRIDGE
            ).value
        telemetry.generated_tokens = len(generated)
        telemetry.total_seconds = time.perf_counter() - generation_started
        telemetry.decode_seconds = telemetry.total_seconds
        telemetry.stop_reason = (
            "eos"
            if generated and int(generated[-1]) == self.token_eos()
            else "speculative_complete"
        )
        telemetry.kv_used_cells = len(prompt_tokens) + len(generated)
        self._last_generation = self._annotate_telemetry(
            telemetry,
            speculative_decode=True,
        )
        return list(prompt_tokens) + generated

    def _verify_native_draft_bundle(
        self,
        *,
        prompt_tokens: list[int],
        draft_tokens: list[int],
        generated_prefix_count: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        acceptance_floor: float,
        logits_processor: Optional[list[Any]],
    ):
        return _native_verify_draft_tokens(
            self,
            prompt_tokens=list(prompt_tokens),
            draft_tokens=list(draft_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            min_p=float(min_p),
            min_accept_prob=float(acceptance_floor),
            logits_processor=logits_processor,
            generated_prefix_count=int(max(0, generated_prefix_count)),
            presence_penalty=float(presence_penalty),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(getattr(self, "_no_repeat_ngram_size", 0)),
            min_new_tokens_before_eos=int(
                getattr(self, "_min_new_tokens_before_eos", 0)
            ),
            sample_recovery_token=True,
        )

    def _generate_block_diffusion(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        logits_processor: Optional[list[Any]],
        generation_started: float,
        evidence: GenerationEvidence,
    ) -> list[int]:
        if not bool(getattr(self, "_block_diffusion_native_ready", False)):
            raise RuntimeError(
                "Block diffusion mode requested but native block diffusion kernels are unavailable."
            )
        context_tokens = [int(token) for token in prompt_tokens]
        generated: list[int] = []
        accepted_tokens = 0
        rejected_tokens = 0
        verify_depth = 0
        step_latency_ms = 0.0
        blockwise_blocks = 0
        blockwise_denoise_steps = 0
        blockwise_converged_blocks = 0
        python_hot_path_calls = 0
        use_processors = bool(logits_processor) and not bool(
            self._native_fast_path and self._disable_logits_processors
        )
        block_size = max(
            1, int(getattr(self, "_block_diffusion_block_size_tokens", 16))
        )
        denoise_iters = max(
            1, int(getattr(self, "_block_diffusion_denoise_iterations", 2))
        )
        acceptance_floor = float(
            getattr(self, "_block_diffusion_acceptance_floor", 0.18)
        )

        while len(generated) < int(max_new_tokens):
            remaining = int(max_new_tokens) - len(generated)
            block_limit = min(block_size, remaining)
            drafted: list[int] = []
            draft_confidences: list[float] = []
            blockwise_blocks += 1
            for _ in range(denoise_iters):
                blockwise_denoise_steps += 1
                draft_started = time.perf_counter()
                logits = np.asarray(
                    self._get_logits_for_tokens(list(context_tokens)),
                    dtype=np.float32,
                )
                if use_processors:
                    logits = np.asarray(
                        self._apply_logits_processors(
                            list(context_tokens),
                            logits,
                            logits_processor,
                        ),
                        dtype=np.float32,
                    )
                draft_top_k = int(top_k)
                if draft_top_k <= 0:
                    draft_top_k = int(getattr(self, "_parallel_replacement_top_k", 8))
                seed = (
                    (len(context_tokens) << 32)
                    ^ (len(generated) << 8)
                    ^ (int(time.time_ns()) & 0xFFFFFFFF)
                )
                drafted, draft_confidences = qsg_block_diffusion_draft(
                    logits,
                    draft_tokens=block_limit,
                    temperature=max(float(temperature), 1.0e-6),
                    top_k=max(1, draft_top_k),
                    min_probability=max(0.0, acceptance_floor),
                    seed=int(seed),
                )
                step_latency_ms = max(
                    step_latency_ms,
                    (time.perf_counter() - draft_started) * 1000.0,
                )
                if drafted:
                    break

            if not drafted:
                break

            accepted_in_block = 0
            native_verified = self._verify_native_draft_bundle(
                prompt_tokens=context_tokens,
                draft_tokens=drafted,
                generated_prefix_count=len(generated),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                acceptance_floor=acceptance_floor,
                logits_processor=logits_processor,
            )
            if native_verified is not None:
                accepted_in_block = int(native_verified.accepted_count)
                verify_depth += len(native_verified.probabilities or [])
                for token in drafted[:accepted_in_block]:
                    context_tokens.append(int(token))
                    generated.append(int(token))
                    accepted_tokens += 1
                    if int(token) == int(self.token_eos()):
                        break
                    if len(generated) >= int(max_new_tokens):
                        break
            else:
                for token in drafted:
                    logits = self._get_logits_for_tokens(list(context_tokens))
                    if use_processors:
                        logits = self._apply_logits_processors(
                            list(context_tokens),
                            logits,
                            logits_processor,
                        )
                    verify_depth += 1
                    python_hot_path_calls += 1
                    greedy_token, draft_prob = simd_ops.score_token(
                        logits,
                        token_id=int(token),
                        temperature=max(float(temperature), 1.0e-6),
                    )
                    if greedy_token != int(token) and draft_prob < acceptance_floor:
                        rejected_tokens += 1
                        break

                    context_tokens.append(int(token))
                    generated.append(int(token))
                    accepted_tokens += 1
                    accepted_in_block += 1
                    if int(token) == int(self.token_eos()):
                        break
                    if len(generated) >= int(max_new_tokens):
                        break
            if native_verified is not None and accepted_in_block < len(drafted):
                rejected_tokens += 1
            if drafted and accepted_in_block >= len(drafted):
                blockwise_converged_blocks += 1
            if accepted_in_block <= 0 and len(generated) < int(max_new_tokens):
                recovery_token = (
                    int(native_verified.recovery_token)
                    if native_verified is not None
                    and native_verified.recovery_token is not None
                    else None
                )
                if recovery_token is None:
                    recovery_logits = self._get_logits_for_tokens(list(context_tokens))
                    recovery_token = int(
                        self._sample(
                            recovery_logits,
                            array("i", (int(token) for token in context_tokens)),
                            presence_penalty,
                            repetition_penalty,
                            temperature,
                            top_p,
                            top_k,
                            min_p,
                            disallow_eos=self._should_block_eos(len(generated)),
                            leading_response=(len(generated) == 0),
                        )
                    )
                    python_hot_path_calls += 1
                rejected_tokens += 1
                context_tokens.append(int(recovery_token))
                generated.append(int(recovery_token))
                if int(recovery_token) == int(self.token_eos()):
                    break

            if generated and int(generated[-1]) == int(self.token_eos()):
                break

        evidence.generation_mode = GenerationMode.BLOCK_DIFFUSION.value
        evidence.benchmark_label = benchmark_label_for_mode(
            GenerationMode.BLOCK_DIFFUSION
        ).value
        evidence.accepted_parallel_tokens = int(
            max(evidence.accepted_parallel_tokens, int(accepted_tokens))
        )
        evidence.rejected_parallel_tokens = int(
            max(evidence.rejected_parallel_tokens, int(rejected_tokens))
        )
        evidence.proposed_parallel_tokens = int(
            max(
                evidence.proposed_parallel_tokens,
                int(accepted_tokens + rejected_tokens),
            )
        )
        evidence.verify_depth = int(max(evidence.verify_depth, int(verify_depth)))
        evidence.draft_frontier_width = int(
            max(evidence.draft_frontier_width, int(accepted_tokens))
        )
        evidence.parallel_step_latency_ms = float(
            max(evidence.parallel_step_latency_ms, float(step_latency_ms))
        )
        if draft_confidences:
            evidence.draft_confidence_mean = float(
                max(
                    evidence.draft_confidence_mean,
                    sum(float(value) for value in draft_confidences)
                    / float(len(draft_confidences)),
                )
            )
            confidence_floor = min(float(value) for value in draft_confidences)
            evidence.draft_confidence_min = float(
                confidence_floor
                if evidence.draft_confidence_min <= 0.0
                else min(evidence.draft_confidence_min, confidence_floor)
            )
        evidence.draft_source = str(evidence.draft_source or "block_diffusion_native")
        evidence.blockwise_blocks = int(
            max(evidence.blockwise_blocks, int(blockwise_blocks))
        )
        evidence.blockwise_denoise_steps = int(
            max(evidence.blockwise_denoise_steps, int(blockwise_denoise_steps))
        )
        if blockwise_blocks > 0:
            evidence.blockwise_convergence_rate = float(
                max(
                    evidence.blockwise_convergence_rate,
                    blockwise_converged_blocks / float(blockwise_blocks),
                )
            )
        evidence.quality_guard_triggered = bool(
            evidence.quality_guard_triggered or int(rejected_tokens) > 0
        )

        telemetry = NativeGenerationTelemetry(
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(generated),
            kv_used_cells=len(prompt_tokens) + len(generated),
            stop_reason=(
                "eos"
                if generated and int(generated[-1]) == self.token_eos()
                else "block_diffusion_complete"
            ),
            total_seconds=time.perf_counter() - generation_started,
            decode_seconds=time.perf_counter() - generation_started,
            first_token_latency_seconds=(
                (time.perf_counter() - generation_started) if generated else 0.0
            ),
        )
        telemetry.python_hot_path_calls = int(max(0, python_hot_path_calls))
        self._last_generation = self._apply_generation_evidence(
            self._annotate_telemetry(
                telemetry,
                parallel_decode=True,
                speculative_decode=False,
            ),
            evidence,
        )
        return list(prompt_tokens) + generated

    def _generate_model_head(
        self,
        *,
        head_type: str,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        logits_processor: Optional[list[Any]],
        generation_started: float,
        evidence: GenerationEvidence,
    ) -> list[int]:
        kind = str(head_type or "").strip().lower()
        ready = bool(
            getattr(
                self,
                "_medusa_head_ready" if kind == "medusa" else "_hydra_head_ready",
                False,
            )
        )
        if not ready:
            raise RuntimeError(
                f"{kind} head mode requested but native {kind} head weights are unavailable."
            )
        context_tokens = [int(token) for token in prompt_tokens]
        generated: list[int] = []
        accepted_tokens = 0
        rejected_tokens = 0
        proposed_tokens = 0
        verify_depth = 0
        step_latency_ms = 0.0
        observed_frontier_width = 0
        draft_probability_total = 0.0
        draft_probability_count = 0
        draft_probability_min = 1.0
        draft_source = f"{kind}_head_native"
        python_hot_path_calls = 0
        max_draft_tokens = int(
            getattr(
                self,
                (
                    "_medusa_head_max_draft_tokens"
                    if kind == "medusa"
                    else "_hydra_head_max_draft_tokens"
                ),
                4,
            )
        )
        acceptance_floor = float(
            getattr(
                self,
                (
                    "_medusa_head_acceptance_floor"
                    if kind == "medusa"
                    else "_hydra_head_acceptance_floor"
                ),
                0.20 if kind == "medusa" else 0.22,
            )
        )
        use_processors = bool(logits_processor) and not bool(
            self._native_fast_path and self._disable_logits_processors
        )

        while len(generated) < int(max_new_tokens):
            remaining = int(max_new_tokens) - len(generated)
            draft_started = time.perf_counter()
            draft_bundle = self._draft_model_head_bundle(
                head_type=kind,
                prompt_tokens=list(context_tokens),
                max_new_tokens=min(remaining, max_draft_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
            )
            drafted = list(draft_bundle.tokens)
            step_latency_ms = max(
                step_latency_ms,
                (time.perf_counter() - draft_started) * 1000.0,
            )
            if not drafted:
                break
            proposed_tokens += len(drafted)
            observed_frontier_width = max(observed_frontier_width, len(drafted))
            if draft_bundle.probabilities:
                draft_probability_total += sum(
                    float(probability) for probability in draft_bundle.probabilities
                )
                draft_probability_count += len(draft_bundle.probabilities)
                draft_probability_min = min(
                    draft_probability_min,
                    min(
                        float(probability) for probability in draft_bundle.probabilities
                    ),
                )
            if str(draft_bundle.source).strip():
                draft_source = str(draft_bundle.source)

            accepted_in_step = 0
            native_verified = self._verify_native_draft_bundle(
                prompt_tokens=context_tokens,
                draft_tokens=drafted,
                generated_prefix_count=len(generated),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                acceptance_floor=acceptance_floor,
                logits_processor=logits_processor,
            )
            if native_verified is not None:
                accepted_in_step = int(native_verified.accepted_count)
                verify_depth += len(native_verified.probabilities or [])
                for token in drafted[:accepted_in_step]:
                    context_tokens.append(int(token))
                    generated.append(int(token))
                    accepted_tokens += 1
                    if int(token) == int(self.token_eos()):
                        break
                    if len(generated) >= int(max_new_tokens):
                        break
            else:
                for token in drafted:
                    logits = self._get_logits_for_tokens(list(context_tokens))
                    if use_processors:
                        logits = self._apply_logits_processors(
                            list(context_tokens),
                            logits,
                            logits_processor,
                        )
                    verify_depth += 1
                    python_hot_path_calls += 1
                    greedy_token, draft_prob = simd_ops.score_token(
                        logits,
                        token_id=int(token),
                        temperature=max(float(temperature), 1.0e-6),
                    )
                    if greedy_token != int(token) and draft_prob < acceptance_floor:
                        rejected_tokens += 1
                        break
                    context_tokens.append(int(token))
                    generated.append(int(token))
                    accepted_tokens += 1
                    accepted_in_step += 1
                    if int(token) == int(self.token_eos()):
                        break
                    if len(generated) >= int(max_new_tokens):
                        break
            if native_verified is not None and accepted_in_step < len(drafted):
                rejected_tokens += 1
            if accepted_in_step <= 0 and len(generated) < int(max_new_tokens):
                recovery_token = (
                    int(native_verified.recovery_token)
                    if native_verified is not None
                    and native_verified.recovery_token is not None
                    else None
                )
                if recovery_token is None:
                    recovery_logits = self._get_logits_for_tokens(list(context_tokens))
                    recovery_token = int(
                        self._sample(
                            recovery_logits,
                            array("i", (int(token) for token in context_tokens)),
                            presence_penalty,
                            repetition_penalty,
                            temperature,
                            top_p,
                            top_k,
                            min_p,
                            disallow_eos=self._should_block_eos(len(generated)),
                            leading_response=(len(generated) == 0),
                        )
                    )
                    python_hot_path_calls += 1
                rejected_tokens += 1
                context_tokens.append(int(recovery_token))
                generated.append(int(recovery_token))
                if int(recovery_token) == int(self.token_eos()):
                    break

            if generated and int(generated[-1]) == int(self.token_eos()):
                break

        mode = (
            GenerationMode.MEDUSA_HEAD
            if kind == "medusa"
            else GenerationMode.HYDRA_HEAD
        )
        evidence.generation_mode = mode.value
        evidence.benchmark_label = benchmark_label_for_mode(mode).value
        evidence.accepted_parallel_tokens = int(
            max(evidence.accepted_parallel_tokens, int(accepted_tokens))
        )
        evidence.rejected_parallel_tokens = int(
            max(evidence.rejected_parallel_tokens, int(rejected_tokens))
        )
        evidence.proposed_parallel_tokens = int(
            max(evidence.proposed_parallel_tokens, int(proposed_tokens))
        )
        evidence.verify_depth = int(max(evidence.verify_depth, int(verify_depth)))
        evidence.draft_frontier_width = int(
            max(evidence.draft_frontier_width, int(observed_frontier_width))
        )
        evidence.parallel_step_latency_ms = float(
            max(evidence.parallel_step_latency_ms, float(step_latency_ms))
        )
        if draft_probability_count > 0:
            evidence.draft_confidence_mean = float(
                max(
                    evidence.draft_confidence_mean,
                    draft_probability_total / float(draft_probability_count),
                )
            )
            evidence.draft_confidence_min = float(
                min(
                    (
                        evidence.draft_confidence_min
                        if evidence.draft_confidence_min > 0.0
                        else draft_probability_min
                    ),
                    draft_probability_min,
                )
            )
        evidence.draft_source = str(evidence.draft_source or draft_source)
        evidence.quality_guard_triggered = bool(
            evidence.quality_guard_triggered or int(rejected_tokens) > 0
        )

        telemetry = NativeGenerationTelemetry(
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(generated),
            kv_used_cells=len(prompt_tokens) + len(generated),
            stop_reason=(
                "eos"
                if generated and int(generated[-1]) == self.token_eos()
                else f"{kind}_head_complete"
            ),
            total_seconds=time.perf_counter() - generation_started,
            decode_seconds=time.perf_counter() - generation_started,
            first_token_latency_seconds=(
                (time.perf_counter() - generation_started) if generated else 0.0
            ),
        )
        telemetry.python_hot_path_calls = int(max(0, python_hot_path_calls))
        telemetry.numpy_hot_path_calls = 0
        self._last_generation = self._annotate_telemetry(
            telemetry,
            parallel_decode=True,
            speculative_decode=False,
        )
        return list(prompt_tokens) + generated

    def _generate_masked_diffusion(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        logits_processor: Optional[list[Any]],
        generation_started: float,
        evidence: GenerationEvidence,
    ) -> list[int]:
        if not bool(getattr(self, "_masked_diffusion_native_ready", False)):
            raise RuntimeError(
                "Masked diffusion mode requested but native masked diffusion kernels are unavailable."
            )
        context_tokens = [int(token) for token in prompt_tokens]
        generated: list[int] = []
        accepted_tokens = 0
        rejected_tokens = 0
        verify_depth = 0
        step_latency_ms = 0.0
        masked_steps = 0
        masked_positions_total = 0
        python_hot_path_calls = 0
        use_processors = bool(logits_processor) and not bool(
            self._native_fast_path and self._disable_logits_processors
        )
        block_size = max(
            1, int(getattr(self, "_masked_diffusion_block_size_tokens", 16))
        )
        denoise_iters = max(
            1, int(getattr(self, "_masked_diffusion_denoise_iterations", 2))
        )
        mask_stride = max(1, int(getattr(self, "_masked_diffusion_mask_stride", 2)))
        acceptance_floor = float(
            getattr(self, "_masked_diffusion_acceptance_floor", 0.18)
        )

        while len(generated) < int(max_new_tokens):
            remaining = int(max_new_tokens) - len(generated)
            block_limit = min(block_size, remaining)
            drafted: list[int] = []
            draft_confidences: list[float] = []
            masked_positions: list[int] = []
            for _ in range(denoise_iters):
                masked_steps += 1
                draft_started = time.perf_counter()
                logits = np.asarray(
                    self._get_logits_for_tokens(list(context_tokens)),
                    dtype=np.float32,
                )
                if use_processors:
                    logits = np.asarray(
                        self._apply_logits_processors(
                            list(context_tokens),
                            logits,
                            logits_processor,
                        ),
                        dtype=np.float32,
                    )
                draft_top_k = int(top_k)
                if draft_top_k <= 0:
                    draft_top_k = int(getattr(self, "_parallel_replacement_top_k", 8))
                seed = (
                    (len(context_tokens) << 32)
                    ^ (len(generated) << 8)
                    ^ (int(time.time_ns()) & 0xFFFFFFFF)
                )
                drafted, draft_confidences, masked_positions = (
                    qsg_masked_diffusion_draft(
                        logits,
                        draft_tokens=block_limit,
                        mask_stride=mask_stride,
                        temperature=max(float(temperature), 1.0e-6),
                        top_k=max(1, draft_top_k),
                        min_probability=max(0.0, acceptance_floor),
                        seed=int(seed),
                    )
                )
                step_latency_ms = max(
                    step_latency_ms,
                    (time.perf_counter() - draft_started) * 1000.0,
                )
                if drafted:
                    break

            if not drafted:
                break

            masked_positions_total += len(masked_positions)
            accepted_in_step = 0
            native_verified = self._verify_native_draft_bundle(
                prompt_tokens=context_tokens,
                draft_tokens=drafted,
                generated_prefix_count=len(generated),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                acceptance_floor=acceptance_floor,
                logits_processor=logits_processor,
            )
            if native_verified is not None:
                accepted_in_step = int(native_verified.accepted_count)
                verify_depth += len(native_verified.probabilities or [])
                for token in drafted[:accepted_in_step]:
                    context_tokens.append(int(token))
                    generated.append(int(token))
                    accepted_tokens += 1
                    if int(token) == int(self.token_eos()):
                        break
                    if len(generated) >= int(max_new_tokens):
                        break
            else:
                for token in drafted:
                    logits = self._get_logits_for_tokens(list(context_tokens))
                    if use_processors:
                        logits = self._apply_logits_processors(
                            list(context_tokens),
                            logits,
                            logits_processor,
                        )
                    verify_depth += 1
                    python_hot_path_calls += 1
                    greedy_token, draft_prob = simd_ops.score_token(
                        logits,
                        token_id=int(token),
                        temperature=max(float(temperature), 1.0e-6),
                    )
                    if greedy_token != int(token) and draft_prob < acceptance_floor:
                        rejected_tokens += 1
                        break
                    context_tokens.append(int(token))
                    generated.append(int(token))
                    accepted_tokens += 1
                    accepted_in_step += 1
                    if int(token) == int(self.token_eos()):
                        break
                    if len(generated) >= int(max_new_tokens):
                        break
            if native_verified is not None and accepted_in_step < len(drafted):
                rejected_tokens += 1
            if accepted_in_step <= 0 and len(generated) < int(max_new_tokens):
                recovery_token = (
                    int(native_verified.recovery_token)
                    if native_verified is not None
                    and native_verified.recovery_token is not None
                    else None
                )
                if recovery_token is None:
                    recovery_logits = self._get_logits_for_tokens(list(context_tokens))
                    recovery_token = int(
                        self._sample(
                            recovery_logits,
                            array("i", (int(token) for token in context_tokens)),
                            presence_penalty,
                            repetition_penalty,
                            temperature,
                            top_p,
                            top_k,
                            min_p,
                            disallow_eos=self._should_block_eos(len(generated)),
                            leading_response=(len(generated) == 0),
                        )
                    )
                    python_hot_path_calls += 1
                rejected_tokens += 1
                context_tokens.append(int(recovery_token))
                generated.append(int(recovery_token))
                if int(recovery_token) == int(self.token_eos()):
                    break

            if generated and int(generated[-1]) == int(self.token_eos()):
                break

        evidence.generation_mode = GenerationMode.MASKED_DIFFUSION.value
        evidence.benchmark_label = benchmark_label_for_mode(
            GenerationMode.MASKED_DIFFUSION
        ).value
        evidence.accepted_parallel_tokens = int(
            max(evidence.accepted_parallel_tokens, int(accepted_tokens))
        )
        evidence.rejected_parallel_tokens = int(
            max(evidence.rejected_parallel_tokens, int(rejected_tokens))
        )
        evidence.proposed_parallel_tokens = int(
            max(
                evidence.proposed_parallel_tokens,
                int(accepted_tokens + rejected_tokens),
            )
        )
        evidence.verify_depth = int(max(evidence.verify_depth, int(verify_depth)))
        evidence.parallel_step_latency_ms = float(
            max(evidence.parallel_step_latency_ms, float(step_latency_ms))
        )
        evidence.draft_frontier_width = int(
            max(evidence.draft_frontier_width, int(masked_positions_total))
        )
        evidence.draft_source = str(evidence.draft_source or "masked_diffusion_native")
        evidence.quality_guard_triggered = bool(
            evidence.quality_guard_triggered or int(rejected_tokens) > 0
        )

        telemetry = NativeGenerationTelemetry(
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(generated),
            kv_used_cells=len(prompt_tokens) + len(generated),
            stop_reason=(
                "eos"
                if generated and int(generated[-1]) == self.token_eos()
                else "masked_diffusion_complete"
            ),
            total_seconds=time.perf_counter() - generation_started,
            decode_seconds=time.perf_counter() - generation_started,
            first_token_latency_seconds=(
                (time.perf_counter() - generation_started) if generated else 0.0
            ),
            masked_generation_ready=True,
            masked_generation_steps=int(masked_steps),
            masked_generation_proposed_tokens=int(masked_positions_total),
            masked_generation_accepted_tokens=int(accepted_tokens),
            masked_generation_density=(
                float(masked_positions_total)
                / float(max(1, accepted_tokens + rejected_tokens))
            ),
        )
        telemetry.python_hot_path_calls = int(max(0, python_hot_path_calls))
        self._last_generation = self._apply_generation_evidence(
            self._annotate_telemetry(
                telemetry,
                parallel_decode=True,
                speculative_decode=False,
            ),
            evidence,
        )
        return list(prompt_tokens) + generated

    def _generate_autoregressive(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        generation_started: float,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        logits_processor: Optional[list[Any]],
    ) -> tuple[list[int], NativeGenerationTelemetry]:
        runtime = self._build_decode_runtime()
        if runtime is not None:
            telemetry = NativeGenerationTelemetry(
                prompt_tokens=len(prompt_tokens),
                kv_used_cells=len(prompt_tokens),
                stop_reason="max_new_tokens" if max_new_tokens > 0 else "prompt_only",
                native_fast_path=True,
            )
            if logits_processor and not (
                self._native_fast_path and self._disable_logits_processors
            ):
                runtime.close()
                raise RuntimeError(
                    "Strict native generation does not allow Python logits_processor "
                    "callables on the active hot path."
                )
            output_tokens = list(prompt_tokens)
            request_id = f"decode-{time.time_ns()}"
            runtime.submit(
                request_id,
                priority=0,
                arrival_ts_ns=time.time_ns(),
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                min_p=float(min_p),
                presence_penalty=float(presence_penalty),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(getattr(self, "_no_repeat_ngram_size", 0)),
                min_new_tokens_before_eos=int(
                    getattr(self, "_min_new_tokens_before_eos", 0)
                ),
            )
            first_emit_at: float | None = None
            last_emit_at: float | None = None
            try:
                while True:
                    event = runtime.poll(request_id)
                    if event is None:
                        time.sleep(0.0005)
                        continue
                    now = time.perf_counter()
                    if event.token_id is not None:
                        token = int(event.token_id)
                        output_tokens.append(token)
                        telemetry.generated_tokens += 1
                        telemetry.kv_used_cells = len(output_tokens)
                        if first_emit_at is None:
                            first_emit_at = now
                            telemetry.first_token_latency_seconds = (
                                now - generation_started
                            )
                        if last_emit_at is not None:
                            telemetry.per_token_latencies_seconds.append(
                                max(0.0, now - last_emit_at)
                            )
                        last_emit_at = now
                        if token == self.token_eos():
                            telemetry.stop_reason = "eos"
                    if event.done:
                        if event.error:
                            telemetry.stop_reason = "error"
                        elif (
                            telemetry.stop_reason != "eos"
                            and telemetry.generated_tokens >= max_new_tokens
                        ):
                            telemetry.stop_reason = "max_new_tokens"
                        telemetry.total_seconds = (
                            time.perf_counter() - generation_started
                        )
                        telemetry.prefill_seconds = (
                            telemetry.first_token_latency_seconds
                            if telemetry.first_token_latency_seconds > 0.0
                            else telemetry.total_seconds
                        )
                        telemetry.decode_seconds = max(
                            0.0, telemetry.total_seconds - telemetry.prefill_seconds
                        )
                        runtime_metrics = runtime.metrics()
                        telemetry.prefill_chunk_count = int(
                            runtime_metrics.prefill_batches
                        )
                        telemetry.graph_prefill_calls = int(
                            runtime_metrics.prefill_batches
                        )
                        telemetry.graph_decode_calls = int(
                            runtime_metrics.runtime_decode_steps
                        )
                        telemetry.sample_calls = int(
                            runtime_metrics.runtime_decode_steps
                        )
                        telemetry.batched_prefill_token_id_calls = int(
                            runtime_metrics.prefill_batches
                        )
                        telemetry.batched_prefill_token_id_tokens = int(
                            runtime_metrics.runtime_prefill_tokens
                        )
                        return output_tokens, telemetry
            finally:
                runtime.close()

        telemetry = NativeGenerationTelemetry(
            prompt_tokens=len(prompt_tokens),
            kv_used_cells=len(prompt_tokens),
            stop_reason="max_new_tokens" if max_new_tokens > 0 else "prompt_only",
        )
        self._clear_logits_cache()
        self._reset_generation_counters()
        if self._model_graph is not None:
            self._model_graph.reset()
        output_tokens = list(prompt_tokens)

        logits: list[float]
        pos = 0
        if logits_processor and not (
            self._native_fast_path and self._disable_logits_processors
        ):
            raise RuntimeError(
                "Strict native generation does not allow Python logits_processor "
                "callables on the active hot path."
            )
        model_graph = getattr(self, "_model_graph", None)
        graph_handle = int(getattr(model_graph, "_handle", 0) or 0)
        if graph_handle > 0:
            try:
                generated_only, stop_reason = qsg_autoregressive_generate(
                    model_graph_handle=graph_handle,
                    prompt_tokens=list(prompt_tokens),
                    max_new_tokens=int(max_new_tokens),
                    vocab_size=int(self.profile.vocab_size),
                    eos_token=int(self.token_eos()),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    min_p=float(min_p),
                    presence_penalty=float(presence_penalty),
                    repetition_penalty=float(repetition_penalty),
                    no_repeat_ngram_size=int(getattr(self, "_no_repeat_ngram_size", 0)),
                    min_new_tokens_before_eos=int(
                        getattr(self, "_min_new_tokens_before_eos", 0)
                    ),
                )
                telemetry.generated_tokens = len(generated_only)
                telemetry.kv_used_cells = len(prompt_tokens) + len(generated_only)
                telemetry.total_seconds = time.perf_counter() - generation_started
                telemetry.decode_seconds = telemetry.total_seconds
                telemetry.first_token_latency_seconds = (
                    telemetry.total_seconds if generated_only else 0.0
                )
                telemetry.stop_reason = (
                    "eos"
                    if stop_reason == 1
                    else (
                        "max_new_tokens"
                        if generated_only or max_new_tokens > 0
                        else "prompt_only"
                    )
                )
                telemetry.python_hot_path_calls = 0
                telemetry.numpy_hot_path_calls = 0
                return list(prompt_tokens) + list(generated_only), telemetry
            except Exception:
                pass
        ubatch = max(1, int(getattr(self, "num_ubatch", 1)))
        prefill_started = time.perf_counter()
        if len(prompt_tokens) == 0:
            logits = _zero_logits(self.profile.vocab_size)
        elif ubatch > 1 and len(prompt_tokens) > ubatch:
            logits = _zero_logits(self.profile.vocab_size)
            for i in range(0, len(prompt_tokens), ubatch):
                chunk = prompt_tokens[i : i + ubatch]
                logits = self._get_logits(chunk, start_pos=pos)
                pos += len(chunk)
                self._prefill_chunk_count += 1
        else:
            logits = self._get_logits(prompt_tokens, start_pos=0)
            pos = len(prompt_tokens)
            if prompt_tokens:
                self._prefill_chunk_count = 1
        telemetry.prefill_seconds = time.perf_counter() - prefill_started
        self._reset_graph_perf_stats_for_decode_window()
        token_history_buf = array("i", (int(token) for token in output_tokens))

        for token_index in range(max_new_tokens):
            self._sample_os_thread_cpu()
            step_started = time.perf_counter()
            telemetry.python_hot_path_calls += 1
            generated_tokens = len(output_tokens) - len(prompt_tokens)
            token = self._sample(
                logits,
                token_history_buf,
                presence_penalty,
                repetition_penalty,
                temperature,
                top_p,
                top_k,
                min_p,
                disallow_eos=self._should_block_eos(generated_tokens),
                leading_response=(token_index == 0),
            )
            output_tokens.append(token)
            token_history_buf.append(int(token))
            telemetry.generated_tokens += 1
            telemetry.kv_used_cells = len(output_tokens)
            if token_index == 0:
                telemetry.first_token_latency_seconds = (
                    time.perf_counter() - generation_started
                )
            step_elapsed = time.perf_counter() - step_started
            telemetry.per_token_latencies_seconds.append(step_elapsed)

            if token == self.token_eos():
                telemetry.stop_reason = "eos"
                break

            logits = self._get_logits([token], start_pos=pos)
            pos += 1

        telemetry.total_seconds = time.perf_counter() - generation_started
        telemetry.decode_seconds = max(
            0.0, telemetry.total_seconds - telemetry.prefill_seconds
        )
        return output_tokens, telemetry

    def generate(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        logits_processor: Optional[list[Any]] = None,
        seed: Optional[int] = None,
    ) -> list[int]:
        """Generate tokens through the parallel planner, then verify/recover as needed."""
        if seed is not None:
            simd_ops.seed_rng(int(seed))
        generation_started = time.perf_counter()
        self._parallel_decode_disable_reason = ""
        self._last_temperature_band = _temperature_band(float(temperature))
        planner = getattr(self, "_parallel_planner", None)
        if planner is None:
            planner = ParallelDecodePlanner(self)
            self._parallel_planner = planner
        plan = planner.plan(
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            logits_processor=logits_processor,
        )
        planned_prompt = list(prompt_tokens) + list(plan.accepted_prefix_tokens)
        remaining_tokens = max(
            0, int(max_new_tokens) - len(plan.accepted_prefix_tokens)
        )
        if planned_prompt and int(planned_prompt[-1]) == int(self.token_eos()):
            remaining_tokens = 0

        if (
            plan.mode == GenerationMode.MASKED_DIFFUSION
            and bool(getattr(self, "_masked_diffusion_native_ready", False))
            and remaining_tokens > 0
        ):
            output_tokens = self._generate_masked_diffusion(
                prompt_tokens=planned_prompt,
                max_new_tokens=remaining_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                logits_processor=logits_processor,
                generation_started=generation_started,
                evidence=plan.evidence,
            )
            if len(output_tokens) <= len(planned_prompt):
                self._raise_if_autoregressive_fallback_forbidden(
                    "masked_diffusion_native_rejected_or_empty",
                    plan_mode=plan.mode,
                )
                plan.evidence.generation_mode = GenerationMode.AR_RECOVERY.value
                plan.evidence.benchmark_label = benchmark_label_for_mode(
                    GenerationMode.AR_RECOVERY
                ).value
                plan.evidence.quality_guard_triggered = True
                output_tokens, telemetry = self._generate_autoregressive(
                    prompt_tokens=planned_prompt,
                    max_new_tokens=remaining_tokens,
                    generation_started=generation_started,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    logits_processor=logits_processor,
                )
                self._last_generation = self._annotate_telemetry(
                    telemetry,
                    parallel_decode=False,
                    speculative_decode=False,
                )
            self._last_generation = self._apply_generation_evidence(
                self._last_generation,
                plan.evidence,
            )
            return output_tokens

        if (
            plan.mode == GenerationMode.BLOCK_DIFFUSION
            and bool(getattr(self, "_block_diffusion_native_ready", False))
            and remaining_tokens > 0
        ):
            output_tokens = self._generate_block_diffusion(
                prompt_tokens=planned_prompt,
                max_new_tokens=remaining_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                logits_processor=logits_processor,
                generation_started=generation_started,
                evidence=plan.evidence,
            )
            if len(output_tokens) <= len(planned_prompt):
                self._raise_if_autoregressive_fallback_forbidden(
                    "block_diffusion_native_rejected_or_empty",
                    plan_mode=plan.mode,
                )
                plan.evidence.generation_mode = GenerationMode.AR_RECOVERY.value
                plan.evidence.benchmark_label = benchmark_label_for_mode(
                    GenerationMode.AR_RECOVERY
                ).value
                plan.evidence.quality_guard_triggered = True
                output_tokens, telemetry = self._generate_autoregressive(
                    prompt_tokens=planned_prompt,
                    max_new_tokens=remaining_tokens,
                    generation_started=generation_started,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    logits_processor=logits_processor,
                )
                self._last_generation = self._annotate_telemetry(
                    telemetry,
                    parallel_decode=False,
                    speculative_decode=False,
                )
            self._last_generation = self._apply_generation_evidence(
                self._last_generation,
                plan.evidence,
            )
            return output_tokens

        if (
            plan.mode in {GenerationMode.MEDUSA_HEAD, GenerationMode.HYDRA_HEAD}
            and remaining_tokens > 0
        ):
            output_tokens = self._generate_model_head(
                head_type=(
                    "medusa" if plan.mode == GenerationMode.MEDUSA_HEAD else "hydra"
                ),
                prompt_tokens=planned_prompt,
                max_new_tokens=remaining_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                logits_processor=logits_processor,
                generation_started=generation_started,
                evidence=plan.evidence,
            )
            if len(output_tokens) <= len(planned_prompt):
                self._raise_if_autoregressive_fallback_forbidden(
                    "draft_head_native_rejected_or_empty",
                    plan_mode=plan.mode,
                )
                plan.evidence.generation_mode = GenerationMode.AR_RECOVERY.value
                plan.evidence.benchmark_label = benchmark_label_for_mode(
                    GenerationMode.AR_RECOVERY
                ).value
                plan.evidence.quality_guard_triggered = True
                output_tokens, telemetry = self._generate_autoregressive(
                    prompt_tokens=planned_prompt,
                    max_new_tokens=remaining_tokens,
                    generation_started=generation_started,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    logits_processor=logits_processor,
                )
                self._last_generation = self._annotate_telemetry(
                    telemetry,
                    parallel_decode=False,
                    speculative_decode=False,
                )
            self._last_generation = self._apply_generation_evidence(
                self._last_generation,
                plan.evidence,
            )
            return output_tokens

        if plan.mode in {
            GenerationMode.PROMPT_LOOKUP,
            GenerationMode.PARALLEL_HYBRID,
            GenerationMode.REPLACEMENT,
        } and remaining_tokens > 0:
            parallel_decode_ready = self._should_parallel_decode(
                planned_prompt,
                remaining_tokens,
                temperature,
            )
            if (
                not parallel_decode_ready
                and plan.accepted_prefix_tokens
                and bool(getattr(self, "_forbid_autoregressive_fallback", False))
            ):
                parallel_decode_ready = True
                self._parallel_decode_disable_reason = ""
            if parallel_decode_ready:
                output_tokens = self.generate_parallel(
                    prompt_tokens=planned_prompt,
                    max_new_tokens=remaining_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    logits_processor=None,
                )
                self._last_generation = self._apply_generation_evidence(
                    self._last_generation,
                    plan.evidence,
                )
                return output_tokens

        if (
            plan.mode == GenerationMode.SSD_BRIDGE
            and (
                self._supports_native_self_spec() or self._ssm_spec_decoder is not None
            )
            and remaining_tokens > 0
        ):
            output_tokens = self._generate_ssd_bridge(
                prompt_tokens=planned_prompt,
                max_new_tokens=remaining_tokens,
                temperature=temperature,
                logits_processor=logits_processor,
                generation_started=generation_started,
            )
            self._last_generation = self._apply_generation_evidence(
                self._last_generation,
                plan.evidence,
            )
            return output_tokens

        if remaining_tokens <= 0:
            if not plan.accepted_prefix_tokens:
                self._raise_if_autoregressive_fallback_forbidden(
                    "planner_returned_no_non_ar_prefix",
                    plan_mode=plan.mode,
                )
                output_tokens, telemetry = self._generate_autoregressive(
                    prompt_tokens=planned_prompt,
                    max_new_tokens=0,
                    generation_started=generation_started,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    presence_penalty=presence_penalty,
                    repetition_penalty=repetition_penalty,
                    logits_processor=logits_processor,
                )
                self._last_generation = self._apply_generation_evidence(
                    self._annotate_telemetry(
                        telemetry,
                        parallel_decode=bool(
                            plan.mode == GenerationMode.PARALLEL_HYBRID
                        ),
                        speculative_decode=(plan.mode == GenerationMode.SSD_BRIDGE),
                    ),
                    plan.evidence,
                )
                return output_tokens
            telemetry = NativeGenerationTelemetry(
                prompt_tokens=len(prompt_tokens),
                generated_tokens=len(plan.accepted_prefix_tokens),
                total_seconds=time.perf_counter() - generation_started,
                decode_seconds=time.perf_counter() - generation_started,
                stop_reason="parallel_prefix_complete",
                kv_used_cells=len(planned_prompt),
            )
            self._last_generation = self._apply_generation_evidence(
                self._annotate_telemetry(
                    telemetry,
                    parallel_decode=True,
                ),
                plan.evidence,
            )
            return planned_prompt

        self._raise_if_autoregressive_fallback_forbidden(
            "planner_selected_autoregressive_completion",
            plan_mode=plan.mode,
        )
        output_tokens, telemetry = self._generate_autoregressive(
            prompt_tokens=planned_prompt,
            max_new_tokens=remaining_tokens,
            generation_started=generation_started,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            logits_processor=logits_processor,
        )
        self._last_generation = self._apply_generation_evidence(
            self._annotate_telemetry(
                telemetry,
                parallel_decode=bool(
                    plan.mode == GenerationMode.PARALLEL_HYBRID
                    or plan.accepted_prefix_tokens
                ),
                speculative_decode=(plan.mode == GenerationMode.SSD_BRIDGE),
            ),
            plan.evidence,
        )
        return output_tokens

    def _raise_if_autoregressive_fallback_forbidden(
        self,
        reason: str,
        *,
        plan_mode: GenerationMode,
    ) -> None:
        if not bool(getattr(self, "_forbid_autoregressive_fallback", False)):
            return
        raise RuntimeError(
            "Strict native non-autoregressive mode forbids autoregressive fallback: "
            f"{reason} (plan_mode={plan_mode.value})"
        )

    def _should_parallel_decode(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> bool:
        min_new_tokens_before_eos = int(getattr(self, "_min_new_tokens_before_eos", 0))
        if not PERFORMANCE_CONFIG.get("parallel_decode", False):
            self._parallel_decode_disable_reason = "parallel_decode_config_disabled"
            return False
        if max_new_tokens <= 1 or len(prompt_tokens) == 0:
            self._parallel_decode_disable_reason = (
                "insufficient_input_for_parallel_decode"
            )
            return False
        if max_new_tokens <= min_new_tokens_before_eos:
            self._parallel_decode_disable_reason = (
                "max_new_tokens_conflicts_with_eos_guard"
            )
            return False
        if temperature <= 0.0:
            self._parallel_decode_disable_reason = "temperature_zero_or_negative"
            return False
        if max_new_tokens < self._parallel_decode_min_new_tokens:
            self._parallel_decode_disable_reason = "max_new_tokens_below_threshold"
            return False
        if len(prompt_tokens) < self._parallel_decode_min_prompt_tokens:
            self._parallel_decode_disable_reason = "prompt_tokens_below_threshold"
            return False
        if self._force_parallel_decode:
            self._parallel_decode_disable_reason = ""
            return True
        self._parallel_decode_disable_reason = ""
        return True

    def generate_parallel(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        logits_processor: Optional[list[Any]] = None,
    ) -> list[int]:
        """Jacobi parallel decode in strict native mode (no fallback ladder)."""
        started = time.perf_counter()
        if self._model_graph is not None:
            self._model_graph.reset()
        self._reset_generation_counters()
        self._clear_logits_cache()
        output_tokens = list(prompt_tokens)
        remaining = int(max_new_tokens)
        while remaining > 0:
            result = self._jacobi.decode(
                engine=self,
                prompt_tokens=output_tokens,
                max_tokens=remaining,
                temperature=temperature,
                logits_processor=(
                    None
                    if self._native_fast_path and self._disable_logits_processors
                    else logits_processor
                ),
            )

            drafted = list(result.tokens)
            if result.accepted <= 0 or not drafted:
                raise RuntimeError(
                    "Jacobi parallel decode produced no accepted tokens. "
                    "Single-step fallback is disabled in strict native mode."
                )

            for token in drafted:
                self._sample_os_thread_cpu()
                generated_tokens = len(output_tokens) - len(prompt_tokens)
                if token == self.token_eos() and self._should_block_eos(
                    generated_tokens
                ):
                    logits = self._get_logits_for_tokens(output_tokens)
                    token = self._sample(
                        logits,
                        array("i", (int(item) for item in output_tokens)),
                        presence_penalty,
                        repetition_penalty,
                        temperature,
                        top_p,
                        top_k,
                        min_p,
                        leading_response=(generated_tokens == 0),
                        disallow_eos=True,
                    )
                output_tokens.append(int(token))
                remaining -= 1
                if token == self.token_eos():
                    self._last_generation = NativeGenerationTelemetry(
                        prompt_tokens=len(prompt_tokens),
                        generated_tokens=len(output_tokens) - len(prompt_tokens),
                        total_seconds=time.perf_counter() - started,
                        decode_seconds=time.perf_counter() - started,
                        first_token_latency_seconds=0.0,
                        stop_reason="eos",
                        kv_used_cells=len(output_tokens),
                        python_hot_path_calls=len(output_tokens) - len(prompt_tokens),
                    )
                    self._last_generation = self._annotate_telemetry(
                        self._last_generation,
                        parallel_decode=True,
                    )
                    return output_tokens
                if remaining <= 0:
                    break

        self._last_generation = NativeGenerationTelemetry(
            prompt_tokens=len(prompt_tokens),
            generated_tokens=len(output_tokens) - len(prompt_tokens),
            total_seconds=time.perf_counter() - started,
            decode_seconds=time.perf_counter() - started,
            first_token_latency_seconds=0.0,
            stop_reason="max_new_tokens",
            kv_used_cells=len(output_tokens),
            python_hot_path_calls=len(output_tokens) - len(prompt_tokens),
        )
        self._last_generation = self._annotate_telemetry(
            self._last_generation,
            parallel_decode=True,
        )
        return output_tokens

    def generate_stream(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        logits_processor: Optional[list[Any]] = None,
    ):
        """Stream tokens one at a time."""
        runtime = self._build_decode_runtime()
        if runtime is not None:
            generation_started = time.perf_counter()
            telemetry = NativeGenerationTelemetry(
                prompt_tokens=len(prompt_tokens),
                kv_used_cells=len(prompt_tokens),
                stop_reason="max_new_tokens" if max_new_tokens > 0 else "prompt_only",
                native_fast_path=True,
            )
            if logits_processor and not (
                self._native_fast_path and self._disable_logits_processors
            ):
                runtime.close()
                raise RuntimeError(
                    "Strict native generation does not allow Python logits_processor "
                    "callables on the active hot path."
                )
            request_id = f"stream-{time.time_ns()}"
            runtime.submit(
                request_id,
                priority=0,
                arrival_ts_ns=time.time_ns(),
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                min_p=float(min_p),
                presence_penalty=float(presence_penalty),
                repetition_penalty=float(repetition_penalty),
                no_repeat_ngram_size=int(getattr(self, "_no_repeat_ngram_size", 0)),
                min_new_tokens_before_eos=int(
                    getattr(self, "_min_new_tokens_before_eos", 0)
                ),
            )
            first_emit_at: float | None = None
            last_emit_at: float | None = None
            try:
                while True:
                    event = runtime.poll(request_id)
                    if event is None:
                        time.sleep(0.0005)
                        continue
                    now = time.perf_counter()
                    if event.token_id is not None:
                        token = int(event.token_id)
                        telemetry.generated_tokens += 1
                        telemetry.kv_used_cells = len(prompt_tokens) + int(
                            telemetry.generated_tokens
                        )
                        if first_emit_at is None:
                            first_emit_at = now
                            telemetry.first_token_latency_seconds = (
                                now - generation_started
                            )
                        if last_emit_at is not None:
                            telemetry.per_token_latencies_seconds.append(
                                max(0.0, now - last_emit_at)
                            )
                        last_emit_at = now
                        if token == self.token_eos():
                            telemetry.stop_reason = "eos"
                        else:
                            yield token
                    if event.done:
                        if event.error:
                            telemetry.stop_reason = "error"
                        elif (
                            telemetry.stop_reason != "eos"
                            and telemetry.generated_tokens >= max_new_tokens
                        ):
                            telemetry.stop_reason = "max_new_tokens"
                        telemetry.total_seconds = (
                            time.perf_counter() - generation_started
                        )
                        telemetry.prefill_seconds = (
                            telemetry.first_token_latency_seconds
                            if telemetry.first_token_latency_seconds > 0.0
                            else telemetry.total_seconds
                        )
                        telemetry.decode_seconds = max(
                            0.0, telemetry.total_seconds - telemetry.prefill_seconds
                        )
                        runtime_metrics = runtime.metrics()
                        telemetry.prefill_chunk_count = int(
                            runtime_metrics.prefill_batches
                        )
                        telemetry.graph_prefill_calls = int(
                            runtime_metrics.prefill_batches
                        )
                        telemetry.graph_decode_calls = int(
                            runtime_metrics.runtime_decode_steps
                        )
                        telemetry.sample_calls = int(
                            runtime_metrics.runtime_decode_steps
                        )
                        telemetry.batched_prefill_token_id_calls = int(
                            runtime_metrics.prefill_batches
                        )
                        telemetry.batched_prefill_token_id_tokens = int(
                            runtime_metrics.runtime_prefill_tokens
                        )
                        self._last_generation = self._annotate_telemetry(telemetry)
                        return
            finally:
                runtime.close()

        generation_started = time.perf_counter()
        telemetry = NativeGenerationTelemetry(
            prompt_tokens=len(prompt_tokens),
            kv_used_cells=len(prompt_tokens),
            stop_reason="max_new_tokens" if max_new_tokens > 0 else "prompt_only",
        )
        if self._model_graph is not None:
            self._model_graph.reset()
        self._reset_generation_counters()
        self._clear_logits_cache()
        self._parallel_decode_disable_reason = ""
        if logits_processor and not (
            self._native_fast_path and self._disable_logits_processors
        ):
            raise RuntimeError(
                "Strict native generation does not allow Python logits_processor "
                "callables on the active hot path."
            )

        # Prefill
        all_tokens = list(prompt_tokens)
        token_history_buf = array("i", (int(token) for token in all_tokens))
        pos = 0
        ubatch = max(1, int(getattr(self, "num_ubatch", 1)))
        prefill_started = time.perf_counter()
        if len(prompt_tokens) == 0:
            logits = _zero_logits(self.profile.vocab_size)
        elif ubatch > 1 and len(prompt_tokens) > ubatch:
            logits = _zero_logits(self.profile.vocab_size)
            for i in range(0, len(prompt_tokens), ubatch):
                chunk = prompt_tokens[i : i + ubatch]
                logits = self._get_logits(chunk, start_pos=pos)
                pos += len(chunk)
                self._prefill_chunk_count += 1
        else:
            logits = self._get_logits(prompt_tokens, start_pos=0)
            pos = len(prompt_tokens)
            if prompt_tokens:
                self._prefill_chunk_count = 1
        telemetry.prefill_seconds = time.perf_counter() - prefill_started
        self._reset_graph_perf_stats_for_decode_window()

        for token_index in range(max_new_tokens):
            self._sample_os_thread_cpu()
            step_started = time.perf_counter()
            telemetry.python_hot_path_calls += 1
            generated_tokens = len(all_tokens) - len(prompt_tokens)
            token = self._sample(
                logits,
                token_history_buf,
                presence_penalty,
                repetition_penalty,
                temperature,
                top_p,
                top_k,
                min_p,
                disallow_eos=self._should_block_eos(generated_tokens),
                leading_response=(token_index == 0),
            )

            if token == self.token_eos():
                telemetry.stop_reason = "eos"
                telemetry.total_seconds = time.perf_counter() - generation_started
                telemetry.decode_seconds = max(
                    0.0, telemetry.total_seconds - telemetry.prefill_seconds
                )
                self._last_generation = self._annotate_telemetry(telemetry)
                return

            all_tokens.append(token)
            token_history_buf.append(int(token))
            telemetry.generated_tokens += 1
            telemetry.kv_used_cells = len(all_tokens)
            if token_index == 0:
                telemetry.first_token_latency_seconds = (
                    time.perf_counter() - generation_started
                )
            telemetry.per_token_latencies_seconds.append(
                time.perf_counter() - step_started
            )
            yield token

            logits = self._get_logits([token], start_pos=pos)
            pos += 1
        telemetry.total_seconds = time.perf_counter() - generation_started
        telemetry.decode_seconds = max(
            0.0, telemetry.total_seconds - telemetry.prefill_seconds
        )
        self._last_generation = self._annotate_telemetry(telemetry)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        _ = seed
        effective_repetition_penalty = (
            float(repetition_penalty)
            if repetition_penalty is not None
            else float(repeat_penalty)
        )
        tokens = self.prepare_prompt_tokens(prompt)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": effective_repetition_penalty,
        }
        if seed is not None and _callable_supports_keyword_arg(self.generate, "seed"):
            generate_kwargs["seed"] = seed
        generated = self.generate(tokens, **generate_kwargs)
        return self.decode_generated_tokens(generated[len(tokens) :])

    def generate_stream_text(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        _ = seed
        effective_repetition_penalty = (
            float(repetition_penalty)
            if repetition_penalty is not None
            else float(repeat_penalty)
        )
        tokens = self.prepare_prompt_tokens(prompt)
        emitted_text = ""
        generated_tokens: list[int] = []
        generate_stream_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": effective_repetition_penalty,
        }
        if seed is not None and _callable_supports_keyword_arg(
            self.generate_stream, "seed"
        ):
            generate_stream_kwargs["seed"] = seed
        for token in self.generate_stream(tokens, **generate_stream_kwargs):
            generated_tokens.append(int(token))
            text = self.decode_generated_tokens(generated_tokens)
            if text.startswith(emitted_text):
                delta = text[len(emitted_text) :]
            else:
                delta = text
            emitted_text = text
            if delta:
                yield delta

    def get_hidden_states(self, token_ids: list[int], layer: int = -1) -> Any:
        _ = token_ids
        _ = layer
        raise RuntimeError(
            "Strict native production mode disables hidden-state helpers."
        )

    def close(self) -> None:
        if getattr(self, "_model_graph", None) is not None:
            self._model_graph.close()
            self._model_graph = None
        if getattr(self, "_native_kv_cache", None) is not None:
            self._native_kv_cache.close()
            self._native_kv_cache = None
        if getattr(self, "forward_pass", None) is not None:
            close_fn = getattr(self.forward_pass, "close", None)
            if callable(close_fn):
                close_fn()
        if getattr(self, "loader", None) is not None:
            self.loader.close()
