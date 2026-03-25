"""Architecture-aware GGUF tensor cache for the native QSG engine."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from core.model.gguf_loader import GGUFModelLoader
from core.model.model_profile import ModelProfile
from core.native.quantized_matmul_wrapper import (
    QuantizedMatrix,
    dequantize_rows,
    interleave_quantized_rows,
    matvec,
)


@dataclass(frozen=True)
class LayerWeights:
    layer_idx: int
    weights: dict[str, Union[np.ndarray, QuantizedMatrix]]
    layer_type: str


class WeightStore:
    """Lazy tensor cache with light architecture-specific normalization."""

    def __init__(self, loader: GGUFModelLoader, profile: ModelProfile):
        self.loader = loader
        self.profile = profile
        self._cache: dict[str, np.ndarray] = {}
        self._f32_cache: dict[str, np.ndarray] = {}
        self._quant_cache: dict[str, QuantizedMatrix] = {}
        self._expert_cache: dict[tuple[str, int], Union[np.ndarray, QuantizedMatrix]] = {}
        self._expert_cache_limit = 8192
        self._raw_expert_data: dict[str, np.ndarray] = {}  # cache 3D expert tensors
        self._layer_cache: dict[int, LayerWeights] = {}
        self._layer_names_cache: dict[int, set[str]] = {}
        self._tensor_index = {tensor.name: tensor for tensor in loader.reader.tensors}
        self._layer_names: set[str] = set(self._tensor_index.keys())
        disable_moe_routed_raw = os.getenv("ANVIL_DISABLE_MOE_ROUTED")
        disable_moe_shared_raw = os.getenv("ANVIL_DISABLE_MOE_SHARED")
        self._disable_moe_routed = (
            str(disable_moe_routed_raw).strip().lower() in {"1", "true", "yes", "on"}
            if disable_moe_routed_raw is not None
            else False
        )
        self._disable_moe_shared = (
            str(disable_moe_shared_raw).strip().lower() in {"1", "true", "yes", "on"}
            if disable_moe_shared_raw is not None
            else False
        )
        # Full MoE expert tensors are 3D and can expand to tens of GB if dequantized.
        # Keep disabled in the default runtime path until native expert routing is enabled.
        self._load_moe_experts = os.getenv("ANVIL_NATIVE_LOAD_MOE_EXPERTS", "0") == "1"
        self._promote_f16_cache = os.getenv("ANVIL_NATIVE_PROMOTE_F16_CACHE", "1") == "1"
        graph_mode_on = str(os.getenv("ANVIL_NATIVE_GRAPH_MODE", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        architecture = str(getattr(profile, "architecture", "") or "").strip().lower()
        default_interleave = "4" if architecture == "granitehybrid" and graph_mode_on else "1"
        self._quant_interleave = max(
            1,
            int(os.getenv("ANVIL_QUANT_ROW_INTERLEAVE", default_interleave) or default_interleave),
        )
        self._attention_dims = self._infer_attention_dims()

    def get_tensor(self, name: str) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]
        tensor = self.loader.get_tensor(name)
        if tensor is None:
            return None
        tensor_arr = np.asarray(tensor)
        if not tensor_arr.flags["C_CONTIGUOUS"]:
            tensor_arr = np.ascontiguousarray(tensor_arr)
        self._cache[name] = tensor_arr
        return tensor_arr

    def get_tensor_quantized(self, name: str) -> Optional[QuantizedMatrix]:
        if name in self._quant_cache:
            return self._quant_cache[name]
        tensor = self._tensor_index.get(name)
        if tensor is None:
            return None
        if len(tensor.shape) != 2:
            return None
        tensor_type = getattr(tensor, "tensor_type", None)
        if tensor_type is None or not hasattr(tensor_type, "value"):
            return None
        if not hasattr(tensor, "data"):
            return None
        qtype = int(tensor_type.value)
        if qtype not in {8, 12, 14}:
            return None
        raw = np.asarray(tensor.data, dtype=np.uint8)
        if raw.ndim != 2:
            return None
        matrix = QuantizedMatrix(
            name=name,
            qtype=qtype,
            shape=(int(tensor.shape[0]), int(tensor.shape[1])),
            data=raw if raw.flags["C_CONTIGUOUS"] else np.ascontiguousarray(raw),
        )
        if self._quant_interleave > 1:
            matrix = interleave_quantized_rows(matrix, factor=self._quant_interleave)
        self._quant_cache[name] = matrix
        return matrix

    def get_weight(self, name: str) -> Optional[Union[np.ndarray, QuantizedMatrix]]:
        quant = self.get_tensor_quantized(name)
        if quant is not None:
            return quant
        tensor = self._tensor_index.get(name)
        if (
            tensor is not None
            and not self._load_moe_experts
            and len(getattr(tensor, "shape", ())) > 2
        ):
            tensor_type = getattr(tensor, "tensor_type", None)
            qvalue = getattr(tensor_type, "value", None)
            if qvalue is not None and int(qvalue) not in {0, 1}:
                return None
        dense = self.get_tensor(name)
        if dense is None:
            return None
        # Cast F16 weights once to avoid repeated per-token conversion during forward passes.
        if dense.dtype == np.float16 and self._promote_f16_cache:
            cached = self._f32_cache.get(name)
            if cached is None:
                cached = np.ascontiguousarray(dense, dtype=np.float32)
                self._f32_cache[name] = cached
            return cached
        return dense

    def get_embedding_weights(self) -> dict[str, Union[np.ndarray, QuantizedMatrix]]:
        token_embd = self.get_weight("token_embd.weight")
        output = self.get_weight("output.weight")
        if token_embd is None:
            raise RuntimeError("Missing token_embd.weight in GGUF model.")
        if isinstance(token_embd, np.ndarray) and token_embd.ndim == 2 and token_embd.shape[0] < token_embd.shape[1]:
            token_embd = token_embd.T
        if isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[0] < output.shape[1]:
            output = output.T
        return {"token_embd": token_embd, "output": output if output is not None else token_embd}

    def lookup_embeddings(self, token_ids: list[int]) -> np.ndarray:
        emb = self.get_embedding_weights()["token_embd"]
        if isinstance(emb, QuantizedMatrix):
            return dequantize_rows(emb, token_ids)
        return np.asarray(emb[token_ids], dtype=np.float32)

    def project_lm_head(self, hidden: np.ndarray) -> np.ndarray:
        emb = self.get_embedding_weights()
        lm = emb["output"]
        x = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if isinstance(lm, QuantizedMatrix):
            return matvec(x, lm)
        w = np.asarray(lm, dtype=np.float32)
        if w.ndim != 2:
            return x
        if w.shape[0] == x.shape[0]:
            return x @ w
        if w.shape[1] == x.shape[0]:
            return w @ x
        if w.shape[1] > x.shape[0]:
            x = np.pad(x, (0, w.shape[1] - x.shape[0]))
            return w @ x
        return w @ x[: w.shape[1]]

    def get_layer_type(self, layer_idx: int) -> str:
        cached = self._layer_cache.get(layer_idx)
        if cached is not None:
            return cached.layer_type
        names = self._layer_names_for(layer_idx)
        has_attn = any(
            name in names
            for name in (
                f"blk.{layer_idx}.attn_qkv.weight",
                f"blk.{layer_idx}.attn_q.weight",
                f"blk.{layer_idx}.attn_output.weight",
            )
        )
        has_ssm = f"blk.{layer_idx}.ssm_a" in names
        if has_attn and has_ssm:
            return "hybrid"
        if has_attn:
            return "attention"
        if has_ssm:
            return "ssm"
        return "ffn"

    def get_layer_weights(self, layer_idx: int) -> dict[str, Union[np.ndarray, QuantizedMatrix]]:
        if layer_idx in self._layer_cache:
            return self._layer_cache[layer_idx].weights

        names = self._layer_names_for(layer_idx)
        candidates = {
            "attn_norm": [
                f"blk.{layer_idx}.attn_norm.weight",
                f"blk.{layer_idx}.ssm_norm.weight",
            ],
            "attn_qkv": [f"blk.{layer_idx}.attn_qkv.weight"],
            "attn_q": [f"blk.{layer_idx}.attn_q.weight"],
            "attn_k": [f"blk.{layer_idx}.attn_k.weight"],
            "attn_v": [f"blk.{layer_idx}.attn_v.weight"],
            "attn_output": [f"blk.{layer_idx}.attn_output.weight"],
            "attn_gate": [f"blk.{layer_idx}.attn_gate.weight"],
            "attn_q_norm": [f"blk.{layer_idx}.attn_q_norm.weight"],
            "attn_k_norm": [f"blk.{layer_idx}.attn_k_norm.weight"],
            "post_attn_norm": [f"blk.{layer_idx}.post_attention_norm.weight"],
            "ffn_norm": [f"blk.{layer_idx}.ffn_norm.weight"],
            "ffn_gate": [f"blk.{layer_idx}.ffn_gate.weight"],
            "ffn_up": [f"blk.{layer_idx}.ffn_up.weight"],
            "ffn_down": [f"blk.{layer_idx}.ffn_down.weight"],
            "ffn_gate_inp": [f"blk.{layer_idx}.ffn_gate_inp.weight"],
            "ffn_gate_shexp": [f"blk.{layer_idx}.ffn_gate_shexp.weight"],
            "ffn_up_shexp": [f"blk.{layer_idx}.ffn_up_shexp.weight"],
            "ffn_down_shexp": [f"blk.{layer_idx}.ffn_down_shexp.weight"],
            "ssm_a": [f"blk.{layer_idx}.ssm_a"],
            "ssm_d": [f"blk.{layer_idx}.ssm_d"],
            "ssm_dt": [f"blk.{layer_idx}.ssm_dt", f"blk.{layer_idx}.ssm_dt.bias"],
            "ssm_in": [f"blk.{layer_idx}.ssm_in.weight"],
            "ssm_out": [f"blk.{layer_idx}.ssm_out.weight"],
            "ssm_alpha": [f"blk.{layer_idx}.ssm_alpha.weight"],
            "ssm_beta": [f"blk.{layer_idx}.ssm_beta.weight"],
            "ssm_conv1d": [f"blk.{layer_idx}.ssm_conv1d.weight"],
            "ssm_conv1d_bias": [f"blk.{layer_idx}.ssm_conv1d.bias"],
            "ssm_norm": [f"blk.{layer_idx}.ssm_norm.weight"],
        }
        if self._load_moe_experts:
            candidates.update(
                {
                    "ffn_gate_exps": [f"blk.{layer_idx}.ffn_gate_exps.weight"],
                    "ffn_up_exps": [f"blk.{layer_idx}.ffn_up_exps.weight"],
                    "ffn_down_exps": [f"blk.{layer_idx}.ffn_down_exps.weight"],
                }
            )

        resolved: dict[str, Union[np.ndarray, QuantizedMatrix]] = {}
        for key, options in candidates.items():
            for option in options:
                if option not in names:
                    continue
                tensor = self.get_weight(option)
                if tensor is not None:
                    resolved[key] = tensor
                    break

        if self._disable_moe_shared:
            resolved.pop("ffn_gate_shexp", None)
            resolved.pop("ffn_up_shexp", None)
            resolved.pop("ffn_down_shexp", None)
        if self._disable_moe_routed:
            resolved.pop("ffn_gate_inp", None)

        layer_type = self.get_layer_type(layer_idx)
        self._layer_cache[layer_idx] = LayerWeights(
            layer_idx=layer_idx,
            weights=resolved,
            layer_type=layer_type,
        )
        return resolved

    def release_layer(self, layer_idx: int) -> None:
        prefix = f"blk.{layer_idx}."
        to_delete = [name for name in self._cache if name.startswith(prefix)]
        for name in to_delete:
            del self._cache[name]
        f32_to_delete = [name for name in self._f32_cache if name.startswith(prefix)]
        for name in f32_to_delete:
            del self._f32_cache[name]
        q_to_delete = [name for name in self._quant_cache if name.startswith(prefix)]
        for name in q_to_delete:
            del self._quant_cache[name]
        expert_to_delete = [key for key in self._expert_cache if key[0].startswith(prefix)]
        for key in expert_to_delete:
            del self._expert_cache[key]
        self._layer_cache.pop(layer_idx, None)
        self._layer_names_cache.pop(layer_idx, None)

    def get_expert_tensor_names(self, layer_idx: int) -> tuple[Optional[str], Optional[str], Optional[str]]:
        if self._disable_moe_routed:
            return (None, None, None)
        names = self._layer_names_for(layer_idx)
        gate = f"blk.{layer_idx}.ffn_gate_exps.weight"
        up = f"blk.{layer_idx}.ffn_up_exps.weight"
        down = f"blk.{layer_idx}.ffn_down_exps.weight"
        return (
            gate if gate in names else None,
            up if up in names else None,
            down if down in names else None,
        )

    def get_expert_matrix(self, tensor_name: str, expert_idx: int) -> Optional[Union[np.ndarray, QuantizedMatrix]]:
        key = (tensor_name, int(expert_idx))
        cached = self._expert_cache.get(key)
        if cached is not None:
            return cached

        tensor = self._tensor_index.get(tensor_name)
        if tensor is None or len(getattr(tensor, "shape", ())) != 3:
            return None
        tensor_type = getattr(tensor, "tensor_type", None)
        qvalue = getattr(tensor_type, "value", None)
        if qvalue is None:
            return None
        qtype = int(qvalue)

        matrix: Optional[Union[np.ndarray, QuantizedMatrix]] = None
        if qtype in {8, 12, 14}:
            # Cache the full 3D raw tensor to avoid repeated np.asarray calls
            if tensor_name not in self._raw_expert_data:
                raw = np.asarray(tensor.data, dtype=np.uint8)
                if raw.ndim != 3 or raw.shape[0] <= 0:
                    return None
                self._raw_expert_data[tensor_name] = raw
            raw = self._raw_expert_data[tensor_name]
            idx = int(np.clip(expert_idx, 0, raw.shape[0] - 1))
            matrix = QuantizedMatrix(
                name=f"{tensor_name}::expert{idx}",
                qtype=qtype,
                shape=(int(tensor.shape[0]), int(tensor.shape[1])),
                data=np.ascontiguousarray(raw[idx]),
            )
            # Expert fused kernels assume canonical hidden-channel row order.
            # Keep expert projections non-interleaved; interleaving is still used
            # for regular 2D quantized layers on the hot attention path.
        else:
            dense = self.get_tensor(tensor_name)
            if dense is None or dense.ndim != 3 or dense.shape[-1] <= 0:
                return None
            idx = int(np.clip(expert_idx, 0, dense.shape[-1] - 1))
            matrix = np.asarray(dense[:, :, idx], dtype=np.float32)

        if matrix is None:
            return None
        if isinstance(matrix, np.ndarray) and not matrix.flags["C_CONTIGUOUS"]:
            matrix = np.ascontiguousarray(matrix)

        self._expert_cache[key] = matrix
        if len(self._expert_cache) > self._expert_cache_limit:
            oldest = next(iter(self._expert_cache))
            if oldest != key:
                del self._expert_cache[oldest]
        return matrix

    def attention_dims(self) -> tuple[int, int, int] | None:
        return self._attention_dims

    def _layer_names_for(self, layer_idx: int) -> set[str]:
        cached = self._layer_names_cache.get(layer_idx)
        if cached is not None:
            return cached
        prefix = f"blk.{layer_idx}."
        names = {name for name in self._layer_names if name.startswith(prefix)}
        self._layer_names_cache[layer_idx] = names
        return names

    def _infer_attention_dims(self) -> tuple[int, int, int] | None:
        for layer_idx in range(self.profile.n_layers):
            weights = self.get_layer_weights(layer_idx)
            q = weights.get("attn_q")
            k = weights.get("attn_k")
            v = weights.get("attn_v")
            if q is None or k is None or v is None:
                continue
            q_dim = self._weight_output_dim_any(q, self.profile.embedding_dim)
            k_dim = self._weight_output_dim_any(k, self.profile.embedding_dim)
            v_dim = self._weight_output_dim_any(v, self.profile.embedding_dim)
            if q_dim and k_dim and v_dim:
                return (q_dim, k_dim, v_dim)
        return None

    @staticmethod
    def _weight_output_dim_any(
        weight: Union[np.ndarray, QuantizedMatrix],
        input_dim: int,
    ) -> Optional[int]:
        if isinstance(weight, QuantizedMatrix):
            if weight.input_dim == input_dim:
                return int(weight.output_dim)
            if weight.output_dim == input_dim:
                return int(weight.input_dim)
            return int(max(weight.shape))
        return WeightStore._weight_output_dim(weight, input_dim)

    @staticmethod
    def _weight_output_dim(weight: np.ndarray, input_dim: int) -> Optional[int]:
        if weight.ndim != 2:
            return None
        if weight.shape[0] == input_dim:
            return int(weight.shape[1])
        if weight.shape[1] == input_dim:
            return int(weight.shape[0])
        return int(max(weight.shape))
