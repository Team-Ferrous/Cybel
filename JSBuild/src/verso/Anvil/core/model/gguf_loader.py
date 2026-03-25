"""GGUF Model Loader - Direct model file access for QSG pipeline.

Loads model weights and metadata from Ollama's GGUF blob storage,
bypassing the Ollama API for direct inference.
"""

import importlib
import os
from pathlib import Path
from typing import Dict, Optional, Any

from core.model.model_contract import (
    ModelContract,
    model_contract_snapshot,
    resolve_model_contract,
)

np = importlib.import_module("numpy")

try:
    _svd = importlib.import_module("scipy.linalg").svd
except Exception:
    # Keep loader importable without scipy; NumPy provides compatible SVD for this path.
    _svd = np.linalg.svd

_gguf = importlib.import_module("gguf")
GGUFReader = _gguf.GGUFReader
dequantize = _gguf.dequantize


def _decode_byte_values(values: list[Any]) -> Optional[str]:
    """Decode likely UTF-8 byte-value sequences (e.g., [113, 119, 101, 110])."""
    if not values or any(not isinstance(v, int) for v in values):
        return None
    # Single-byte numeric lists are often scalar metadata, not strings.
    if len(values) < 2:
        return None
    if any(v < 0 or v > 255 for v in values):
        return None

    raw = bytes(values)
    # C-style metadata strings can include trailing NUL padding.
    raw = raw.split(b"\x00", maxsplit=1)[0]
    if not raw:
        return None

    text: Optional[str] = None
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None

    if not text.strip():
        return None
    if not any(ch.isalpha() for ch in text):
        return None

    printable = sum(ch.isprintable() for ch in text)
    if printable / max(len(text), 1) < 0.85:
        return None
    return text


def _coerce_metadata_value(value: Any) -> Any:
    """Normalize GGUF metadata values into Python scalars/strings/lists."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _coerce_metadata_value(value.item())
        return _coerce_metadata_value(value.tolist())

    if isinstance(value, (bytes, bytearray, memoryview)):
        decoded = bytes(value).split(b"\x00", maxsplit=1)[0].decode(
            "utf-8", errors="replace"
        )
        return decoded

    if isinstance(value, list):
        coerced = [_coerce_metadata_value(item) for item in value]
        maybe_text = _decode_byte_values(coerced)
        return maybe_text if maybe_text is not None else coerced

    if isinstance(value, tuple):
        coerced = [_coerce_metadata_value(item) for item in value]
        maybe_text = _decode_byte_values(coerced)
        return maybe_text if maybe_text is not None else coerced

    if isinstance(value, str):
        return value.replace("\x00", "")

    return value


def _is_single_int_container(value: Any) -> bool:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], int):
        return True
    if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], int):
        return True
    return False


def _normalize_field_parts(key: str, parts: list[Any]) -> Any:
    """Normalize GGUF field parts into the most likely scalar/array value."""
    coerced = [_coerce_metadata_value(part) for part in parts]
    if len(coerced) == 1:
        return coerced[0]

    # Typical GGUF envelope:
    # [type_tag, field_name, value_type, value_len, ...actual_value]
    if len(coerced) >= 4 and isinstance(coerced[1], str) and coerced[1] == key:
        payload = list(coerced[3:])
        if len(payload) >= 2 and _is_single_int_container(payload[0]):
            # Drop declared length prefix; keep actual payload.
            payload = payload[1:]
        if len(payload) == 1:
            return payload[0]
        return payload

    return coerced


class GGUFModelLoader:
    """
    Loads GGUF model files from Ollama's blob storage.

    Provides direct access to:
    - Token embeddings (for QSG context encoding)
    - Vocabulary tokens (for tokenization)
    - Model metadata (context length, architecture, etc.)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.contract: ModelContract = resolve_model_contract(model_name)
        self.model_path = self.contract.blob_path
        self._reader: Optional[GGUFReader] = None
        self._embedding_tensor: Optional[np.ndarray] = None
        self._vocab_tokens: Optional[list] = None
        self._tokenizer_merges: Optional[list[str]] = None
        self._metadata: Dict = {}

    def _resolve_from_search_paths(self, model_name: str) -> Optional[Path]:
        """Resolve model from explicit local GGUF search paths."""
        env_paths = os.getenv("ANVIL_GGUF_PATHS", "")
        if not env_paths:
            return None

        candidates: list[Path] = []
        for raw in env_paths.split(os.pathsep):
            raw = raw.strip()
            if not raw:
                continue
            root = Path(raw).expanduser()
            candidates.append(root / model_name)
            candidates.append(root / f"{model_name}.gguf")
            if ":" in model_name:
                normalized = model_name.replace(":", "-")
                candidates.append(root / normalized)
                candidates.append(root / f"{normalized}.gguf")

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_ollama_model(self, model_name: str) -> Path:
        """Locate the GGUF blob path for an Ollama model."""
        _ = model_name
        return self.contract.blob_path

    @property
    def reader(self) -> GGUFReader:
        """Lazy-load the GGUF reader."""
        if self._reader is None:
            self._reader = GGUFReader(str(self.model_path))
        return self._reader

    def get_metadata(self) -> Dict:
        """Extract model metadata from GGUF."""
        if not self._metadata:
            for key, field in self.reader.fields.items():
                parts = list(getattr(field, "parts", []) or [])
                if not parts:
                    continue

                val = _normalize_field_parts(key, parts)

                if isinstance(val, list) and len(val) == 1:
                    val = val[0]

                self._metadata[key] = val
            contract = getattr(self, "contract", None)
            model_path = getattr(self, "model_path", None)
            if contract is not None:
                self._metadata.setdefault(
                    "anvil.model.digest",
                    getattr(contract, "expected_digest", ""),
                )
            if model_path is not None:
                self._metadata.setdefault("anvil.model.path", str(model_path))
        return self._metadata

    def get_model_contract(self) -> dict[str, Any]:
        return model_contract_snapshot(self.contract)

    def get_model_digest(self) -> str:
        return self.contract.expected_digest

    def get_quantization_label(self) -> str:
        metadata = self.get_metadata()
        file_type = metadata.get("general.file_type")
        try:
            file_type_int = int(file_type)
        except Exception:
            return "unknown"
        try:
            constants = importlib.import_module("gguf.constants")
            enum_type = getattr(constants, "LlamaFileType", None)
            if enum_type is None:
                return str(file_type_int)
            return str(enum_type(file_type_int).name)
        except Exception:
            return str(file_type_int)

    def get_embedding_dim(self) -> int:
        """Get the model's embedding dimension."""
        metadata = self.get_metadata()
        # Try architecture-specific key first
        for key in [
            "granitehybrid.embedding_length",
            "qwen35.embedding_length",
            "llama.embedding_length",
            "granite.embedding_length",
            "general.embedding_length",
        ]:
            if key in metadata:
                return int(metadata[key])

        # Fall back to extracting from tensor shape
        for tensor in self.reader.tensors:
            if "token_embd" in tensor.name:
                # The embedding tensor is typically [dim, vocab] or [vocab, dim]
                # The smaller dimension is usually the embedding dimension
                return min(tensor.shape)

        raise RuntimeError("Could not determine embedding dimension")

    def get_vocab_size(self) -> int:
        """Get vocabulary size based on embedding tensor (ground truth).

        The embedding tensor is the authoritative source for vocab size.
        Tokenizers may report more tokens than actually have embeddings.
        """
        # First check embedding tensor - this is ground truth
        for tensor in self.reader.tensors:
            if "token_embd" in tensor.name:
                # Shape is [dim, vocab] or [vocab, dim] - take the larger dimension
                return max(tensor.shape)

        # Fallback to architecture-specific metadata
        metadata = self.get_metadata()
        for key in [
            "granitehybrid.vocab_size",
            "qwen35.vocab_size",
            "llama.vocab_size",
            "granite.vocab_size",
            "mistral.vocab_size",
            "qwen2.vocab_size",
            "phi.vocab_size",
        ]:
            if key in metadata:
                return int(metadata[key])

        raise RuntimeError("Could not determine vocab size")

    def get_context_length(self) -> int:
        """Get model's maximum context length."""
        metadata = self.get_metadata()
        for key in [
            "granitehybrid.context_length",
            "qwen35.context_length",
            "llama.context_length",
            "granite.context_length",
        ]:
            if key in metadata:
                return int(metadata[key])
        return 4096  # Default fallback

    def get_token_embeddings(self) -> np.ndarray:
        """
        Load and return the token embedding matrix.

        Returns:
            np.ndarray of shape [vocab_size, embedding_dim]
        """
        if self._embedding_tensor is not None:
            return self._embedding_tensor

        for tensor in self.reader.tensors:
            if tensor.name == "token_embd.weight":
                qtype = int(tensor.tensor_type.value)

                # Dequantize if needed (for Q4_K, Q8_0, etc.)
                if qtype != 0:  # 0 = F32, others are quantized
                    data = np.asarray(dequantize(tensor.data, tensor.tensor_type))
                else:
                    data = np.asarray(tensor.data, dtype=np.float32)

                # GGUF readers usually expose the matrix in runtime orientation already.
                # Only reshape if the payload is flat.
                if data.ndim == 1 and len(tensor.shape) > 1:
                    expected = int(np.prod(tensor.shape))
                    if data.size == expected:
                        data = data.reshape(tensor.shape)

                # The GGUF tensor is typically [dim, vocab] but we want [vocab, dim]
                # Transpose to [vocab, dim] if needed
                if data.shape[0] < data.shape[1]:
                    data = data.T

                self._embedding_tensor = data.astype(np.float32)

                return self._embedding_tensor

        raise RuntimeError("Could not find token embedding tensor")

    def get_vocab_tokens(self) -> list:
        """
        Get the vocabulary token list, truncated to match embedding size.

        Returns:
            List of token strings aligned with embedding tensor
        """
        if self._vocab_tokens is not None:
            return self._vocab_tokens

        # Get the authoritative vocab size from embedding tensor
        max_vocab_size = self.get_vocab_size()

        # Look for tokenizer tokens in metadata
        tokens_key = "tokenizer.ggml.tokens"
        if tokens_key in self.reader.fields:
            field = self.reader.fields[tokens_key]
            # field.data contains indices into field.parts for each token
            # field.parts[idx] is the actual bytes/ndarray for each token
            tokens = []
            for idx in field.data:
                part = field.parts[idx]
                if isinstance(part, (bytes, bytearray)):
                    tokens.append(part.decode("utf-8", errors="replace"))
                elif hasattr(part, "tobytes"):
                    tokens.append(part.tobytes().decode("utf-8", errors="replace"))
                elif isinstance(part, str):
                    tokens.append(part)
                else:
                    tokens.append(f"<unk_{len(tokens)}>")

            if tokens:
                # Truncate to embedding size
                self._vocab_tokens = tokens[:max_vocab_size]
                return self._vocab_tokens

        # Fallback: generate placeholder tokens
        self._vocab_tokens = [f"<tok_{i}>" for i in range(max_vocab_size)]
        return self._vocab_tokens

    def get_tokenizer_merges(self) -> list[str]:
        """Get tokenizer BPE merges from GGUF metadata, if available."""
        if self._tokenizer_merges is not None:
            return self._tokenizer_merges

        merges_key = "tokenizer.ggml.merges"
        field = self.reader.fields.get(merges_key)
        if field is None:
            self._tokenizer_merges = []
            return self._tokenizer_merges

        merges: list[str] = []
        for idx in field.data:
            part = field.parts[idx]
            if isinstance(part, (bytes, bytearray)):
                merges.append(part.decode("utf-8", errors="replace"))
            elif hasattr(part, "tobytes"):
                merges.append(part.tobytes().decode("utf-8", errors="replace"))
            elif isinstance(part, str):
                merges.append(part)
            else:
                merges.append(str(part))

        self._tokenizer_merges = merges
        return self._tokenizer_merges

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs (BOS, EOS, PAD, etc.), clamped to valid range."""
        metadata = self.get_metadata()
        special = {}
        max_valid_id = self.get_vocab_size() - 1

        token_mappings = {
            "bos": "tokenizer.ggml.bos_token_id",
            "eos": "tokenizer.ggml.eos_token_id",
            "pad": "tokenizer.ggml.padding_token_id",
            "unk": "tokenizer.ggml.unknown_token_id",
        }

        for name, key in token_mappings.items():
            if key in metadata:
                token_id = int(metadata[key])
                # Clamp to valid range - enterprise solution for vocab mismatch
                if token_id > max_valid_id:
                    token_id = 0 if name == "pad" else max_valid_id
                special[name] = token_id

        return special

    def get_lm_head_weight(self) -> np.ndarray:
        """
        Get the LM head weight matrix for logit computation.

        Model-agnostic: Returns output.weight if available (Llama, Mistral),
        otherwise falls back to token_embd.weight (tied embeddings like Granite 4).

        Returns:
            np.ndarray of shape [vocab_size, hidden_dim] for logit computation
        """
        # First, try to find a dedicated output weight
        output_weight = self.get_tensor("output.weight")
        if output_weight is not None:
            # Transpose to [vocab, dim] if needed
            if output_weight.shape[0] < output_weight.shape[1]:
                output_weight = output_weight.T
            return output_weight

        # Fall back to tied embeddings
        return self.get_token_embeddings()

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        """
        Load a specific tensor by name.
        Dequantizes to float32 if needed.
        """
        for tensor in self.reader.tensors:
            if tensor.name == name:
                qtype = int(tensor.tensor_type.value)
                if qtype == 0:
                    data = np.asarray(tensor.data, dtype=np.float32)
                elif qtype == 1:
                    data = np.asarray(tensor.data, dtype=np.float16)
                else:
                    data = np.asarray(dequantize(tensor.data, tensor.tensor_type))

                # Preserve GGUF runtime orientation; reshape only for flat payloads.
                if data.ndim == 1 and len(tensor.shape) > 1:
                    expected = int(np.prod(tensor.shape))
                    if data.size == expected:
                        data = data.reshape(tensor.shape)

                # WARNING: GGUF tensor shapes vary by architecture.
                # Some are [Out, In], some [In, Out].
                # We return raw shape here (numpy-ordered).
                # Caller (Engine) must handle transpose if needed based on known architecture.

                return data
        return None

    def get_layer_count(self) -> int:
        """Get number of transformer layers."""
        metadata = self.get_metadata()
        for key in [
            "granitehybrid.block_count",
            "qwen35.block_count",
            "llama.block_count",
            "granite.block_count",
        ]:
            if key in metadata:
                return int(metadata[key])

        # Count by tensor names
        max_layer = -1
        for tensor in self.reader.tensors:
            if "blk." in tensor.name:
                try:
                    # blk.0.attn...
                    parts = tensor.name.split(".")
                    idx = int(parts[1])
                    max_layer = max(max_layer, idx)
                except Exception:
                    pass
        return max_layer + 1

    def extract_propagator(
        self,
        rank: int = 128,
        layers: int = 4,
        strategy: str = "auto",
        profile: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Extract a spectral propagator matrix U.

        Args:
            rank: Rank for SVD approximation.
            layers: Number of initial layers to aggregate.
            strategy: "auto", "mlp", "attn", "identity".
            profile: Optional model profile with propagator_strategy.

        Returns:
            U matrix [dim, dim]
        """
        if strategy == "auto":
            strategy = getattr(profile, "propagator_strategy", "attn")

        if strategy == "identity":
            return np.eye(self.get_embedding_dim(), dtype=np.float32)

        if strategy == "mlp":
            mlp_prop = self._extract_propagator_mlp(rank=rank, layers=layers)
            if mlp_prop is not None:
                return mlp_prop
            return self._extract_propagator_attn(rank=rank, layers=layers)

        return self._extract_propagator_attn(rank=rank, layers=layers)

    def _extract_propagator_attn(self, rank: int = 128, layers: int = 4) -> np.ndarray:
        dim = self.get_embedding_dim()
        agg_matrix = np.zeros((dim, dim), dtype=np.float32)
        count = 0

        for i in range(min(layers, self.get_layer_count())):
            keys = [
                f"blk.{i}.attn_output.weight",
                f"blk.{i}.ffn_down.weight",
                f"blk.{i}.ffn_down_shexp.weight",
                f"blk.{i}.ssm_out.weight",
            ]

            for key in keys:
                tensor = self.get_tensor(key)
                if tensor is not None:
                    projected = self._project_to_dim_square(tensor, dim)
                    if projected is not None:
                        agg_matrix += projected
                        count += 1

        if count > 0:
            agg_matrix /= count
        else:
            for tensor in self.reader.tensors:
                if any(name in tensor.name for name in ["proj", "dense", "out"]):
                    t_data = self.get_tensor(tensor.name)
                    projected = self._project_to_dim_square(t_data, dim)
                    if projected is not None:
                        agg_matrix += projected
                        count += 1
                        if count >= layers:
                            break

            if count > 0:
                agg_matrix /= count
            else:
                return np.eye(dim, dtype=np.float32)

        return self._svd_approximation(agg_matrix, rank=rank)

    def _extract_propagator_mlp(
        self, rank: int = 128, layers: int = 4
    ) -> Optional[np.ndarray]:
        dim = self.get_embedding_dim()
        agg_matrix = np.zeros((dim, dim), dtype=np.float32)
        count = 0

        gate_keys = [
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_gate_inp.weight",
        ]
        down_keys = [
            "ffn_down.weight",
            "ffn_down_shexp.weight",
            "ffn_down_exps.weight",
        ]

        for i in range(min(layers, self.get_layer_count())):
            gate = self._first_tensor_for_keys(i, gate_keys)
            down = self._first_tensor_for_keys(i, down_keys)

            layer_matrix = self._compose_dim_transition(down, gate, dim)
            if layer_matrix is None:
                continue

            agg_matrix += layer_matrix
            count += 1

        if count == 0:
            return None

        agg_matrix /= count
        return self._svd_approximation(agg_matrix, rank=rank)

    def _first_tensor_for_keys(
        self, layer_idx: int, suffixes: list[str]
    ) -> Optional[np.ndarray]:
        for suffix in suffixes:
            tensor = self.get_tensor(f"blk.{layer_idx}.{suffix}")
            if tensor is not None and tensor.ndim == 2:
                return tensor
        return None

    def _project_to_dim_square(
        self, tensor: Optional[np.ndarray], dim: int
    ) -> Optional[np.ndarray]:
        if tensor is None or tensor.ndim != 2:
            return None

        projected = np.zeros((dim, dim), dtype=np.float32)
        rows = min(dim, tensor.shape[0])
        cols = min(dim, tensor.shape[1])
        projected[:rows, :cols] = tensor[:rows, :cols].astype(np.float32, copy=False)
        return projected

    def _compose_dim_transition(
        self, down: Optional[np.ndarray], gate: Optional[np.ndarray], dim: int
    ) -> Optional[np.ndarray]:
        if down is None and gate is None:
            return None

        if down is not None and gate is not None:
            for left in (down, down.T):
                for right in (gate, gate.T):
                    if left.ndim != 2 or right.ndim != 2:
                        continue
                    if left.shape[1] != right.shape[0]:
                        continue
                    candidate = left @ right
                    if candidate.shape[0] >= 1 and candidate.shape[1] >= 1:
                        return self._project_to_dim_square(candidate, dim)

        if down is not None:
            return self._project_to_dim_square(down, dim)
        return self._project_to_dim_square(gate, dim)

    def _svd_approximation(self, matrix: np.ndarray, rank: int) -> np.ndarray:
        U_svd, s, Vh = _svd(matrix, full_matrices=False)

        use_rank = max(1, min(rank, s.shape[0]))
        s_reduced = np.zeros_like(s)
        s_reduced[:use_rank] = s[:use_rank]

        U_approximated = (U_svd * s_reduced) @ Vh
        return U_approximated.astype(np.float32)

    def close(self):
        """Release resources."""
        self._reader = None
        self._embedding_tensor = None
        self._vocab_tokens = None
        self._tokenizer_merges = None


# Cached loader instances per model
_loader_cache: Dict[str, GGUFModelLoader] = {}


def get_loader(model_name: str) -> GGUFModelLoader:
    """Get a cached loader for a model."""
    contract = resolve_model_contract(model_name)
    cache_key = contract.canonical_name
    if cache_key not in _loader_cache:
        _loader_cache[cache_key] = GGUFModelLoader(contract.canonical_name)
    return _loader_cache[cache_key]
