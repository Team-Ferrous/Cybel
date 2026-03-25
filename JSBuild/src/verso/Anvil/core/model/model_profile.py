"""Model profiling for architecture-aware QSG behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.model.chat_templates import (
    default_chat_template_name,
    get_strict_prompt_contract,
    normalize_chat_template_name,
)
from core.model.model_contract import canonicalize_model_name
from config.settings import MODEL_PROFILES


@dataclass(frozen=True)
class ModelProfile:
    """Runtime profile inferred from model metadata and Ollama modelfile."""

    model_name: str
    family: str
    architecture: str
    vocab_size: int
    embedding_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    has_moe: bool
    chat_template: str
    propagator_strategy: str
    logits_proxy_top_k: int
    coconut_alpha: float
    grover_top_k: int
    grover_damping: float
    speculative_acceptance_threshold: float
    grover_iterations: int
    coconut_paths: int
    coconut_mode: str
    speculative_enabled: bool
    spec_num_candidates: int
    spec_max_draft_length: int
    spec_acceptance_threshold: float
    gqa: bool
    tied_embeddings: bool

    @classmethod
    def from_loader(cls, model_name: str, loader: Any) -> "ModelProfile":
        metadata_obj = loader.get_metadata() if hasattr(loader, "get_metadata") else {}
        metadata = metadata_obj if isinstance(metadata_obj, dict) else {}
        architecture = _normalize_architecture(metadata.get("general.architecture", ""))
        lower_name = model_name.lower()

        is_qwen = "qwen" in lower_name or "qwen" in architecture
        is_granite = "granite" in lower_name or "granite" in architecture

        family = "qwen" if is_qwen else "granite" if is_granite else "generic"
        template = _detect_chat_template(
            loader=loader,
            metadata=metadata,
            model_name=model_name,
        )
        if not template:
            template = default_chat_template_name(model_name, family=family)
        strategy = "mlp" if (is_qwen or is_granite) else "attn"
        vocab_size = int(_safe_loader_int(loader, "get_vocab_size"))
        embedding_dim = int(_safe_loader_int(loader, "get_embedding_dim"))
        n_layers = int(_safe_loader_int(loader, "get_layer_count"))

        attention_heads = _find_int_metadata(
            metadata,
            (
                "attention.head_count",
                "attn.head_count",
                "head_count",
            ),
        )
        kv_heads = _find_int_metadata(
            metadata,
            (
                "attention.head_count_kv",
                "attn.head_count_kv",
                "head_count_kv",
            ),
        )
        n_heads = int(attention_heads or 0)
        n_kv_heads = int(kv_heads or 0)
        if n_heads > 0 and (n_kv_heads <= 0 or n_kv_heads > n_heads):
            inferred_kv = _infer_kv_heads_from_byte_payload(
                metadata,
                (
                    "attention.head_count_kv",
                    "attn.head_count_kv",
                    "head_count_kv",
                ),
            )
            if inferred_kv is not None and 0 < inferred_kv <= n_heads:
                n_kv_heads = inferred_kv
        if n_heads > 0 and embedding_dim > 0:
            inferred_qkv_kv = _infer_kv_heads_from_qkv_tensors(
                loader=loader,
                n_heads=n_heads,
                embedding_dim=embedding_dim,
            )
            if inferred_qkv_kv is not None and 0 < inferred_qkv_kv <= n_heads:
                n_kv_heads = inferred_qkv_kv
        if n_heads > 0 and (n_kv_heads <= 0 or n_kv_heads > n_heads):
            n_kv_heads = n_heads
        gqa = bool(n_heads > 0 and n_kv_heads > 0 and n_kv_heads < n_heads)

        tied_embeddings = True
        try:
            tied_embeddings = not any(
                t.name == "output.weight" for t in loader.reader.tensors
            )
        except Exception:
            pass
        has_moe = _detect_moe(loader)

        if family == "qwen":
            profile = cls(
                model_name=model_name,
                family=family,
                architecture=architecture or "qwen",
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                has_moe=has_moe,
                chat_template=template,
                propagator_strategy=strategy,
                logits_proxy_top_k=48,
                coconut_alpha=0.22,
                grover_top_k=8,
                grover_damping=0.55,
                speculative_acceptance_threshold=0.72,
                grover_iterations=1,
                coconut_paths=8,
                coconut_mode="logits_proxy",
                speculative_enabled=True,
                spec_num_candidates=3,
                spec_max_draft_length=3,
                spec_acceptance_threshold=0.72,
                gqa=gqa,
                tied_embeddings=tied_embeddings,
            )
            return _apply_settings_override(profile)

        if family == "granite":
            profile = cls(
                model_name=model_name,
                family=family,
                architecture=architecture or "granite",
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                has_moe=has_moe,
                chat_template=template,
                propagator_strategy=strategy,
                logits_proxy_top_k=32,
                coconut_alpha=0.30,
                grover_top_k=4,
                grover_damping=0.70,
                speculative_acceptance_threshold=0.68,
                grover_iterations=2,
                coconut_paths=8,
                coconut_mode="logits_proxy",
                speculative_enabled=True,
                spec_num_candidates=4,
                spec_max_draft_length=4,
                spec_acceptance_threshold=0.68,
                gqa=gqa,
                tied_embeddings=tied_embeddings,
            )
            return _apply_settings_override(profile)

        profile = cls(
            model_name=model_name,
            family=family,
            architecture=architecture or "generic",
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            has_moe=has_moe,
            chat_template=template,
            propagator_strategy="attn",
            logits_proxy_top_k=32,
            coconut_alpha=0.25,
            grover_top_k=4,
            grover_damping=0.65,
            speculative_acceptance_threshold=0.70,
            grover_iterations=1,
            coconut_paths=8,
            coconut_mode="logits_proxy",
            speculative_enabled=True,
            spec_num_candidates=4,
            spec_max_draft_length=4,
            spec_acceptance_threshold=0.70,
            gqa=gqa,
            tied_embeddings=tied_embeddings,
        )
        return _apply_settings_override(profile)

    @classmethod
    def from_gguf(cls, loader: Any, model_name: str) -> "ModelProfile":
        """Backward-compatible constructor used by legacy call sites."""
        return cls.from_loader(model_name=model_name, loader=loader)

    @property
    def is_chatml(self) -> bool:
        return self.chat_template == "chatml"


def _find_int_metadata(
    metadata: Dict[str, Any], suffixes: tuple[str, ...]
) -> Optional[int]:
    for key, value in metadata.items():
        for suffix in suffixes:
            if key.endswith(suffix):
                parsed = _extract_int_like(value)
                if parsed is not None:
                    return parsed
    return None


def _extract_int_like(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return None
    if hasattr(value, "tolist"):
        try:
            return _extract_int_like(value.tolist())
        except Exception:
            return None
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        seq = list(value)
        if len(seq) >= 4 and isinstance(seq[1], str):
            # Skip GGUF descriptor envelope and optional length marker.
            seq = seq[3:]
            if len(seq) >= 2 and isinstance(seq[0], list) and len(seq[0]) == 1:
                if isinstance(seq[0][0], int):
                    seq = seq[1:]
        for item in seq:
            parsed = _extract_int_like(item)
            if parsed is not None:
                return parsed
    return None


def _infer_kv_heads_from_byte_payload(
    metadata: Dict[str, Any], suffixes: tuple[str, ...]
) -> Optional[int]:
    for key, value in metadata.items():
        if not any(key.endswith(suffix) for suffix in suffixes):
            continue
        byte_values = _flatten_single_int_lists(value)
        if not byte_values:
            continue
        if len(byte_values) >= 2 and byte_values[0] == len(byte_values) - 1:
            byte_values = byte_values[1:]
        non_zero = [v for v in byte_values if v > 0]
        if not non_zero:
            continue
        # Byte payloads in GGUF metadata for kv heads often repeat the same value.
        counts: Dict[int, int] = {}
        for value_int in non_zero:
            counts[value_int] = counts.get(value_int, 0) + 1
        return max(counts.items(), key=lambda item: item[1])[0]
    return None


def _infer_kv_heads_from_qkv_tensors(
    loader: Any, n_heads: int, embedding_dim: int
) -> Optional[int]:
    if n_heads <= 0 or embedding_dim <= 0 or embedding_dim % n_heads != 0:
        return None
    try:
        tensors = loader.reader.tensors
    except Exception:
        return None

    head_dim = embedding_dim // n_heads
    for tensor in tensors:
        name = str(getattr(tensor, "name", ""))
        if not name.endswith("attn_qkv.weight"):
            continue
        shape = tuple(int(v) for v in getattr(tensor, "shape", ()))
        if len(shape) != 2:
            continue

        fused_out = None
        if shape[0] == embedding_dim:
            fused_out = shape[1]
        elif shape[1] == embedding_dim:
            fused_out = shape[0]
        if fused_out is None or fused_out <= embedding_dim:
            continue

        remaining = fused_out - embedding_dim
        if remaining <= 0 or remaining % 2 != 0:
            continue
        kv_dim = remaining // 2
        if kv_dim <= 0 or kv_dim % head_dim != 0:
            continue
        kv_heads = kv_dim // head_dim
        if 0 < kv_heads <= n_heads:
            return int(kv_heads)
    return None


def _flatten_single_int_lists(value: Any) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    if hasattr(value, "tolist"):
        try:
            return _flatten_single_int_lists(value.tolist())
        except Exception:
            return []
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        out: list[int] = []
        for item in value:
            if isinstance(item, list) and len(item) == 1 and isinstance(item[0], int):
                out.append(int(item[0]))
            else:
                out.extend(_flatten_single_int_lists(item))
        return out
    return []


def _safe_loader_int(loader: Any, attr: str) -> int:
    getter = getattr(loader, attr, None)
    if getter is None:
        return 0
    try:
        return int(getter())
    except Exception:
        return 0


def _normalize_architecture(value: Any) -> str:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace").strip().lower()
    if isinstance(value, list):
        maybe_text = _decode_byte_values(value)
        if maybe_text is not None:
            return maybe_text.lower()
        return str(value).lower()
    return str(value).strip().lower()


def _decode_byte_values(values: list[Any]) -> Optional[str]:
    if not values or any(not isinstance(v, int) for v in values):
        return None
    if any(v < 0 or v > 255 for v in values):
        return None
    raw = bytes(values).split(b"\x00", maxsplit=1)[0]
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    printable = sum(ch.isprintable() for ch in text)
    if printable / max(len(text), 1) < 0.85:
        return None
    return text


def _detect_moe(loader: Any) -> bool:
    try:
        tensor_names = [t.name for t in loader.reader.tensors]
    except Exception:
        return False
    markers = ("experts", "ffn_gate_exps", "ffn_down_exps", "ffn_up_exps")
    return any(any(marker in name for marker in markers) for name in tensor_names)


def _detect_chat_template(loader: Any, metadata: Dict[str, Any], model_name: str) -> Optional[str]:
    strict_contract = get_strict_prompt_contract(model_name, strict=False)
    if strict_contract is not None:
        return strict_contract.template_name

    template = normalize_chat_template_name(_detect_template_from_metadata(metadata))
    if template:
        return template

    template = normalize_chat_template_name(_detect_template_from_tokens(loader))
    if template:
        return template

    lower_name = str(model_name or "").lower()
    architecture = _normalize_architecture(metadata.get("general.architecture", ""))
    if "qwen3.5" in lower_name or "qwen35" in lower_name or "qwen" in architecture:
        return "chatml"
    if "granite4" in lower_name or "granite" in architecture:
        return "granite"

    return None


def _detect_template_from_metadata(metadata: Dict[str, Any]) -> Optional[str]:
    for key, value in metadata.items():
        key_lower = str(key).lower()
        if "chat_template" not in key_lower and not key_lower.endswith(".template"):
            continue
        text = _metadata_text(value)
        if not text:
            continue
        lowered = text.lower()
        if "<|im_start|>" in lowered or "renderer qwen3.5" in lowered:
            return "chatml"
        if "<|start_of_role|>" in lowered:
            return "granite"
        if "<|start_header_id|>" in lowered:
            return "llama3"
    return None


def _detect_template_from_tokens(loader: Any) -> Optional[str]:
    get_vocab_tokens = getattr(loader, "get_vocab_tokens", None)
    if get_vocab_tokens is None:
        return None
    try:
        tokens = get_vocab_tokens()
    except Exception:
        return None
    if not tokens:
        return None

    # Special template markers are usually in the early special-token region.
    sample = tokens[:4096]
    joined = "\n".join(str(token) for token in sample).lower()
    if "<|im_start|>" in joined:
        return "chatml"
    if "<|start_of_role|>" in joined:
        return "granite"
    if "<|start_header_id|>" in joined:
        return "llama3"
    return None


def _metadata_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, list):
        maybe_text = _decode_byte_values(value)
        if maybe_text is not None:
            return maybe_text
        if value and all(isinstance(item, str) for item in value):
            return "\n".join(value)
    return None


def _apply_settings_override(profile: ModelProfile) -> ModelProfile:
    """Allow explicit profile overrides from config/settings.py."""
    candidates = [profile.model_name, profile.model_name.lower()]
    try:
        canonical = canonicalize_model_name(profile.model_name)
    except Exception:
        canonical = ""
    if canonical:
        candidates.extend((canonical, canonical.lower()))

    override = None
    for candidate in candidates:
        if not candidate:
            continue
        override = MODEL_PROFILES.get(candidate)
        if override:
            break
    if not override:
        return profile
    merged = dict(profile.__dict__)
    merged.update(override)
    # Keep aliases consistent.
    if "spec_acceptance_threshold" in merged:
        merged["speculative_acceptance_threshold"] = float(
            merged["spec_acceptance_threshold"]
        )
    return ModelProfile(**merged)
