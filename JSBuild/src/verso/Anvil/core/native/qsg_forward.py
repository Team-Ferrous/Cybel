"""Native forward-pass for hybrid GGUF models (granite4 + qwen3.5)."""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np

from core.model.model_profile import ModelProfile
from core.native import quantized_matmul_wrapper as quant_ops
from core.native import simd_ops_wrapper as simd_ops
from core.native.weight_store import WeightStore

try:
    from core.native.fast_attention_wrapper import (
        fused_attention_mqa_f32 as _fused_attention_mqa,
    )
except Exception:
    _fused_attention_mqa = None

try:
    from core.native.native_kv_cache_wrapper import (
        NativeKVCacheWrapper as _NativeKVCacheWrapper,
    )
except Exception:
    _NativeKVCacheWrapper = None

from core.native.qsg_kv_cache import NativeKVCache


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


class QSGForwardPass:
    """Native forward pass supporting attention and SSM (Gated Delta Networks) layers."""

    def __init__(
        self,
        weight_store: WeightStore,
        profile: ModelProfile,
        max_seq_len: int = 8192,
    ):
        self.weights = weight_store
        self.profile = profile
        kv_max_len = int(
            os.getenv("ANVIL_NATIVE_KV_MAX_SEQ_LEN", str(max_seq_len)) or max_seq_len
        )
        use_native_kv = os.getenv("ANVIL_NATIVE_KV_CACHE", "0") == "1"
        if use_native_kv and _NativeKVCacheWrapper is not None:
            try:
                self.kv_cache = _NativeKVCacheWrapper(
                    profile=profile, max_seq_len=kv_max_len
                )
            except Exception:
                self.kv_cache = NativeKVCache(profile, max_seq_len=kv_max_len)
        else:
            self.kv_cache = NativeKVCache(profile, max_seq_len=kv_max_len)
        self.ssm_states: dict[int, np.ndarray] = {}
        self.rope_theta = self._get_rope_theta()
        self.metadata = self.weights.loader.get_metadata()
        arch = self.profile.architecture
        self.embedding_scale = float(
            self.metadata.get(f"{arch}.embedding_scale", 0.0) or 0.0
        )
        self.residual_scale = float(
            self.metadata.get(f"{arch}.residual_scale", 0.0) or 0.0
        )
        self.logit_scale = float(self.metadata.get(f"{arch}.logit_scale", 0.0) or 0.0)
        self.attention_scale = float(
            self.metadata.get(f"{arch}.attention.scale", 0.0) or 0.0
        )
        self.rms_eps = float(
            self.metadata.get(f"{arch}.attention.layer_norm_rms_epsilon", 1.0e-6)
            or 1.0e-6
        )
        self.rope_dim = int(self.metadata.get(f"{arch}.rope.dimension_count", 0) or 0)
        rope_finetuned_key = f"{arch}.rope.scaling.finetuned"
        rope_finetuned_meta = self.metadata.get(rope_finetuned_key)
        # Granite Hybrid enables RoPE only when this metadata flag is set.
        self.rope_finetuned = _metadata_bool(rope_finetuned_meta, default=True)
        meta_top_k = int(self.metadata.get(f"{arch}.expert_used_count", 1) or 1)
        top_k_override = int(os.getenv("ANVIL_MOE_TOP_K", "0") or "0")
        self.disable_moe_routed = os.getenv("ANVIL_DISABLE_MOE_ROUTED", "0") == "1"
        self.disable_moe_shared = os.getenv("ANVIL_DISABLE_MOE_SHARED", "0") == "1"
        if top_k_override > 0:
            self.moe_top_k = top_k_override
        elif self.profile.architecture == "granitehybrid":
            # Granite routed MoE dominates CPU decode latency; default to a low-latency cap.
            granite_cap_raw = os.getenv("ANVIL_GRANITE_MAX_MOE_TOP_K")
            granite_cap = 2 if granite_cap_raw is None else int(granite_cap_raw or "0")
            self.moe_top_k = (
                min(meta_top_k, granite_cap) if granite_cap > 0 else meta_top_k
            )
        else:
            self.moe_top_k = meta_top_k
        self.qwen_full_attention_interval = int(
            self.metadata.get("qwen35.full_attention_interval", 0) or 0
        )
        self.qwen_is_mrope = _metadata_bool(
            self.metadata.get("qwen35.rope.mrope_interleaved", False)
        )
        self.qwen_rope_sections = self._parse_rope_sections(
            self.metadata.get("qwen35.rope.dimension_sections")
            or self.metadata.get("qwen35.mrope_sections")
            or self.metadata.get("qwen35.rope.mrope_section")
        )
        self.qwen_d_inner = int(
            self.metadata.get("qwen35.ssm.inner_size", self.profile.embedding_dim)
            or self.profile.embedding_dim
        )
        self.qwen_d_state = int(self.metadata.get("qwen35.ssm.state_size", 0) or 0)
        self.qwen_n_groups = int(self.metadata.get("qwen35.ssm.group_count", 0) or 0)
        self.qwen_n_v_heads = int(
            self.metadata.get("qwen35.ssm.time_step_rank", 0) or 0
        )
        self.residual_max_ratio = float(
            os.getenv("ANVIL_RESIDUAL_MAX_RATIO", "1.5") or "1.5"
        )
        self.enable_residual_stabilizer = (
            os.getenv("ANVIL_ENABLE_RESIDUAL_STABILIZER", "0") == "1"
        )
        self.enable_fused_mqa = (
            os.getenv("ANVIL_ENABLE_FUSED_MQA", "1") == "1"
            and _fused_attention_mqa is not None
        )

    def reset(self) -> None:
        self.kv_cache.reset()
        self.ssm_states.clear()
        if hasattr(self, "_conv_bufs"):
            self._conv_bufs.clear()

    def close(self) -> None:
        kv_close = getattr(self.kv_cache, "close", None)
        if callable(kv_close):
            try:
                kv_close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Full forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: list[int],
        start_pos: int = 0,
    ) -> np.ndarray:
        """Full forward pass: tokens → logits.

        Returns logits over full vocabulary for the LAST token position.
        """
        emb = self.weights.get_embedding_weights()
        x = self.weights.lookup_embeddings(token_ids)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if self.embedding_scale:
            x = x * self.embedding_scale

        for layer_idx in range(self.profile.n_layers):
            x = self._forward_layer(x, layer_idx, start_pos)

        # Final RMSNorm
        final_norm = self.weights.get_tensor("output_norm.weight")
        if final_norm is not None:
            x = simd_ops.rmsnorm(x, final_norm, eps=self.rms_eps)

        # LM head: project last hidden state to vocabulary logits
        hidden = x[-1]  # [dim]
        logits = self.weights.project_lm_head(hidden)
        if self.logit_scale:
            logits = logits * (1.0 / self.logit_scale)
        return logits

    def get_hidden_states(
        self, token_ids: list[int], target_layer: int = -1
    ) -> np.ndarray:
        x = self.weights.lookup_embeddings(token_ids)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if target_layer < 0:
            target_layer = max(0, self.profile.n_layers + target_layer)
        for layer_idx in range(min(target_layer + 1, self.profile.n_layers)):
            x = self._forward_layer(x, layer_idx, 0)
        return x

    # ------------------------------------------------------------------
    # Layer dispatch
    # ------------------------------------------------------------------

    def _forward_layer(
        self, x: np.ndarray, layer_idx: int, start_pos: int
    ) -> np.ndarray:
        weights = self.weights.get_layer_weights(layer_idx)
        layer_type = self.weights.get_layer_type(layer_idx)
        if self.profile.architecture == "granitehybrid":
            return self._forward_layer_granite(
                x, layer_idx, weights, layer_type, start_pos
            )
        if self.profile.architecture == "qwen35":
            return self._forward_layer_qwen35(
                x, layer_idx, weights, layer_type, start_pos
            )
        if layer_type == "attention":
            x = self._forward_attention(x, layer_idx, weights, start_pos)
            x = self._forward_ffn(x, layer_idx, weights)
        elif layer_type == "ssm":
            x = self._forward_ssm(x, layer_idx, weights)
            x = self._forward_ffn(x, layer_idx, weights)
        elif layer_type == "hybrid":
            x = self._forward_attention(x, layer_idx, weights, start_pos)
            x = self._forward_ssm(x, layer_idx, weights)
            x = self._forward_ffn(x, layer_idx, weights)
        else:
            # FFN-only layer
            x = self._forward_ffn(x, layer_idx, weights)
        return x

    def _forward_layer_granite(
        self,
        x: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        layer_type: str,
        start_pos: int,
    ) -> np.ndarray:
        norm_w = weights.get("attn_norm")
        normed = (
            simd_ops.rmsnorm(x.copy(), norm_w, eps=self.rms_eps)
            if norm_w is not None
            else x.copy()
        )

        if layer_type == "attention":
            branch = self._forward_attention_branch(
                normed, layer_idx, weights, start_pos
            )
        elif layer_type == "ssm":
            branch = self._forward_granite_ssm_branch(normed, layer_idx, weights)
        elif layer_type == "hybrid":
            attn_branch = self._forward_attention_branch(
                normed, layer_idx, weights, start_pos
            )
            ssm_branch = self._forward_granite_ssm_branch(normed, layer_idx, weights)
            branch = attn_branch + ssm_branch
        else:
            branch = np.zeros_like(x, dtype=np.float32)

        if branch.shape[1] != x.shape[1]:
            branch = self._match_dim(branch, x.shape[1])
        if self.enable_residual_stabilizer:
            branch = self._stabilize_residual_branch(
                branch,
                x,
                max_ratio=self.residual_max_ratio,
            )
        if self.residual_scale:
            branch = branch * self.residual_scale
        ffn_input = x + branch

        ffn_out = self._forward_ffn(ffn_input, layer_idx, weights, add_residual=False)
        if ffn_out.shape[1] != x.shape[1]:
            ffn_out = self._match_dim(ffn_out, x.shape[1])
        if self.enable_residual_stabilizer:
            ffn_out = self._stabilize_residual_branch(
                ffn_out,
                ffn_input,
                max_ratio=self.residual_max_ratio,
            )
        if self.residual_scale:
            ffn_out = ffn_out * self.residual_scale
        return ffn_input + ffn_out

    def _forward_layer_qwen35(
        self,
        x: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        layer_type: str,
        start_pos: int,
    ) -> np.ndarray:
        norm_w = weights.get("attn_norm")
        normed = (
            simd_ops.rmsnorm(x.copy(), norm_w, eps=self.rms_eps)
            if norm_w is not None
            else x.copy()
        )

        is_recurrent = self._is_qwen35_recurrent_layer(layer_idx, weights)
        if is_recurrent:
            branch = self._forward_qwen35_recurrent_branch(
                normed, layer_idx, weights, start_pos
            )
        else:
            branch = self._forward_qwen35_full_attention_branch(
                normed, layer_idx, weights, start_pos
            )
        if branch.shape[1] != x.shape[1]:
            branch = self._match_dim(branch, x.shape[1])
        if self.enable_residual_stabilizer:
            branch = self._stabilize_residual_branch(
                branch,
                x,
                max_ratio=self.residual_max_ratio,
            )
        x = x + branch

        ffn_out = self._forward_ffn(x, layer_idx, weights, add_residual=False)
        if ffn_out.shape[1] != x.shape[1]:
            ffn_out = self._match_dim(ffn_out, x.shape[1])
        if self.enable_residual_stabilizer:
            ffn_out = self._stabilize_residual_branch(
                ffn_out,
                x,
                max_ratio=self.residual_max_ratio,
            )
        return x + ffn_out

    def _is_qwen35_recurrent_layer(
        self,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
    ) -> bool:
        if self.qwen_full_attention_interval > 0:
            return ((layer_idx + 1) % self.qwen_full_attention_interval) != 0
        if weights.get("ssm_a") is not None and weights.get("attn_qkv") is not None:
            return True
        return self.weights.get_layer_type(layer_idx) in {"ssm", "hybrid"}

    def _forward_qwen35_full_attention_branch(
        self,
        normed: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        start_pos: int,
    ) -> np.ndarray:
        return self._forward_attention_branch(
            normed=normed,
            layer_idx=layer_idx,
            weights=weights,
            start_pos=start_pos,
            qwen_full_attn=True,
        )

    def _forward_granite_ssm_branch(
        self,
        normed: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
    ) -> np.ndarray:
        a = weights.get("ssm_a")
        if a is None:
            return np.zeros_like(normed, dtype=np.float32)
        a_vec = np.asarray(a, dtype=np.float32).reshape(-1)
        state_dim = a_vec.shape[0]
        return self._forward_mamba_ssm(normed, layer_idx, weights, a_vec, state_dim)

    def _forward_qwen35_recurrent_branch(
        self,
        normed: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        start_pos: int,
    ) -> np.ndarray:
        seq_len = normed.shape[0]
        rows = []
        for t in range(seq_len):
            row = self._forward_qwen35_recurrent_token(
                normed[t],
                layer_idx=layer_idx,
                weights=weights,
                pos=start_pos + t,
            )
            rows.append(row)
        return np.asarray(rows, dtype=np.float32)

    def _forward_qwen35_recurrent_token(
        self,
        normed_row: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        pos: int,
    ) -> np.ndarray:
        _ = pos
        qkv_w = weights.get("attn_qkv")
        gate_w = weights.get("attn_gate")
        ssm_alpha_w = weights.get("ssm_alpha")
        ssm_beta_w = weights.get("ssm_beta")
        ssm_out_w = weights.get("ssm_out")
        ssm_norm_w = weights.get("ssm_norm")
        ssm_a = weights.get("ssm_a")
        ssm_dt = weights.get("ssm_dt")
        conv1d_w = weights.get("ssm_conv1d")

        if (
            qkv_w is None
            or gate_w is None
            or ssm_alpha_w is None
            or ssm_beta_w is None
            or ssm_out_w is None
            or ssm_a is None
            or ssm_dt is None
            or conv1d_w is None
        ):
            return np.zeros((self.profile.embedding_dim,), dtype=np.float32)

        d_inner = self.qwen_d_inner or self.profile.embedding_dim
        d_state = self.qwen_d_state or 128
        n_group = self.qwen_n_groups or 1
        n_v_heads = self.qwen_n_v_heads or max(1, d_inner // d_state)
        head_v_dim = d_inner // max(n_v_heads, 1)
        qk_dim = n_group * d_state
        conv_channels = d_inner + 2 * qk_dim

        x = np.asarray(normed_row, dtype=np.float32).reshape(-1)
        qkv_mixed = self._linear_project(x, qkv_w)
        z = self._linear_project(x, gate_w)
        beta = self._linear_project(x, ssm_beta_w)
        alpha = self._linear_project(x, ssm_alpha_w)
        dt_bias = np.asarray(ssm_dt, dtype=np.float32).reshape(-1)
        a_vec = np.asarray(ssm_a, dtype=np.float32).reshape(-1)
        if beta.shape[0] != n_v_heads:
            beta = self._match_dim(beta.reshape(1, -1), n_v_heads).reshape(-1)
        if alpha.shape[0] != n_v_heads:
            alpha = self._match_dim(alpha.reshape(1, -1), n_v_heads).reshape(-1)
        if dt_bias.shape[0] != n_v_heads:
            dt_bias = self._match_dim(dt_bias.reshape(1, -1), n_v_heads).reshape(-1)
        if a_vec.shape[0] != n_v_heads:
            a_vec = self._match_dim(a_vec.reshape(1, -1), n_v_heads).reshape(-1)

        beta = self._sigmoid(beta)
        gate = self._softplus(alpha + dt_bias) * a_vec

        if qkv_mixed.shape[0] != conv_channels:
            qkv_mixed = self._match_dim(
                qkv_mixed.reshape(1, -1), conv_channels
            ).reshape(-1)

        conv_kernel = self._normalize_conv_kernel(conv1d_w)
        if conv_kernel is None:
            conv_out = qkv_mixed
        else:
            conv_width = conv_kernel.shape[0]
            conv_key = f"qwen35_conv_buf_{layer_idx}"
            if not hasattr(self, "_conv_bufs"):
                self._conv_bufs = {}
            if conv_key not in self._conv_bufs:
                self._conv_bufs[conv_key] = np.zeros(
                    (conv_width - 1, conv_channels), dtype=np.float32
                )
            padded = np.vstack([self._conv_bufs[conv_key], qkv_mixed.reshape(1, -1)])
            conv_out = np.zeros((conv_channels,), dtype=np.float32)
            for k in range(conv_width):
                conv_out += conv_kernel[k] * padded[k]
            self._conv_bufs[conv_key] = padded[-(conv_width - 1) :].copy()

        conv_out = conv_out * self._sigmoid(conv_out)
        q_conv = conv_out[:qk_dim].reshape(n_group, d_state)
        k_conv = conv_out[qk_dim : 2 * qk_dim].reshape(n_group, d_state)
        v_conv = conv_out[2 * qk_dim :].reshape(n_v_heads, head_v_dim)

        eps = 1.0e-6
        q_norm = np.sqrt(np.sum(q_conv * q_conv, axis=1, keepdims=True) + eps)
        k_norm = np.sqrt(np.sum(k_conv * k_conv, axis=1, keepdims=True) + eps)
        q_conv = q_conv / q_norm
        k_conv = k_conv / k_norm

        if n_group != n_v_heads:
            repeats = max(1, n_v_heads // max(n_group, 1))
            q_conv = np.repeat(q_conv, repeats, axis=0)[:n_v_heads]
            k_conv = np.repeat(k_conv, repeats, axis=0)[:n_v_heads]

        state_key = f"qwen35_delta_{layer_idx}"
        if state_key not in self.ssm_states:
            self.ssm_states[state_key] = np.zeros(
                (n_v_heads, head_v_dim, head_v_dim), dtype=np.float32
            )
        state = self.ssm_states[state_key]

        scale = 1.0 / math.sqrt(float(head_v_dim))
        out_heads = np.zeros((n_v_heads, head_v_dim), dtype=np.float32)
        for h in range(n_v_heads):
            q_h = q_conv[h] * scale
            k_h = k_conv[h]
            v_h = v_conv[h]
            out_h, next_state = self._qwen_delta_step(
                state=state[h],
                q=q_h,
                k=k_h,
                v=v_h,
                gate=float(gate[h]),
                beta=float(beta[h]),
            )
            out_heads[h] = out_h
            state[h] = next_state
        self.ssm_states[state_key] = state

        if ssm_norm_w is not None:
            norm_w = np.asarray(ssm_norm_w, dtype=np.float32).reshape(-1)
            if norm_w.shape[0] != head_v_dim:
                norm_w = self._match_dim(norm_w.reshape(1, -1), head_v_dim).reshape(-1)
            out_heads = self._rmsnorm_lastdim(out_heads, norm_w, eps=self.rms_eps)

        if z.shape[0] != n_v_heads * head_v_dim:
            z = self._match_dim(z.reshape(1, -1), n_v_heads * head_v_dim).reshape(-1)
        z_heads = z.reshape(n_v_heads, head_v_dim)
        z_silu = z_heads * self._sigmoid(z_heads)
        gated = out_heads * z_silu
        out = self._linear_project(gated.reshape(-1), ssm_out_w)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _forward_attention_branch(
        self,
        normed: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        start_pos: int,
        qwen_full_attn: bool = False,
    ) -> np.ndarray:
        seq_len = normed.shape[0]
        q = k = v = None
        gate_direct: Optional[np.ndarray] = None

        q_w = weights.get("attn_q")
        k_w = weights.get("attn_k")
        v_w = weights.get("attn_v")
        q_norm_w = weights.get("attn_q_norm")
        k_norm_w = weights.get("attn_k_norm")

        if q_w is not None and k_w is not None and v_w is not None:
            q_full = self._linear_project_batch(normed, q_w)
            if qwen_full_attn and q_full.shape[1] > self.profile.embedding_dim:
                head_dim_hint = 0
                if q_norm_w is not None:
                    head_dim_hint = int(np.asarray(q_norm_w).reshape(-1).shape[0])
                elif k_norm_w is not None:
                    head_dim_hint = int(np.asarray(k_norm_w).reshape(-1).shape[0])
                q_split, gate_split = self._split_qwen_q_and_gate(
                    q_full, head_dim_hint=head_dim_hint
                )
                q = q_split
                gate_direct = gate_split
            else:
                q = q_full
            k = self._linear_project_batch(normed, k_w)
            v = self._linear_project_batch(normed, v_w)
        else:
            qkv_w = weights.get("attn_qkv")
            if qkv_w is None:
                return np.zeros_like(normed, dtype=np.float32)
            fused = self._linear_project_batch(normed, qkv_w)
            dims = self.weights.attention_dims()
            if dims is None or sum(dims) > fused.shape[1]:
                model_dim = int(self.profile.embedding_dim)
                q_dim = min(model_dim, fused.shape[1])
                remaining = fused.shape[1] - q_dim
                kv_dim = max(0, remaining // 2)
                q_dim = fused.shape[1] - 2 * kv_dim
                if kv_dim <= 0 or q_dim <= 0:
                    return np.zeros_like(normed, dtype=np.float32)
                dims = (q_dim, kv_dim, kv_dim)
            q_dim, k_dim, v_dim = dims
            q = fused[:, :q_dim]
            k = fused[:, q_dim : q_dim + k_dim]
            v = fused[:, q_dim + k_dim : q_dim + k_dim + v_dim]

        if q is None or k is None or v is None:
            return np.zeros_like(normed, dtype=np.float32)

        q_dim = int(q.shape[1])
        k_dim = int(k.shape[1])
        v_dim = int(v.shape[1])
        if q_dim <= 0 or k_dim <= 0 or v_dim <= 0:
            return np.zeros_like(normed, dtype=np.float32)

        if q_norm_w is not None:
            head_dim = int(np.asarray(q_norm_w).reshape(-1).shape[0])
        elif k_norm_w is not None:
            head_dim = int(np.asarray(k_norm_w).reshape(-1).shape[0])
        else:
            guess = self.profile.embedding_dim // max(1, self.profile.n_heads)
            if guess <= 0:
                guess = math.gcd(q_dim, k_dim)
            head_dim = max(1, guess)
            while head_dim > 1 and (
                q_dim % head_dim != 0 or k_dim % head_dim != 0 or v_dim % head_dim != 0
            ):
                head_dim -= 1

        n_heads = max(1, q_dim // max(1, head_dim))
        n_kv_heads = max(1, k_dim // max(1, head_dim))
        q = q[:, : n_heads * head_dim]
        k = k[:, : n_kv_heads * head_dim]
        v = v[:, : n_kv_heads * head_dim]

        q_heads = q.reshape(seq_len, n_heads, head_dim)
        k_heads_new = k.reshape(seq_len, n_kv_heads, head_dim)
        v_heads_new = v.reshape(seq_len, n_kv_heads, head_dim)

        if q_norm_w is not None:
            q_norm_vec = np.asarray(q_norm_w, dtype=np.float32).reshape(-1)
            if q_norm_vec.shape[0] == head_dim:
                q_heads = self._rmsnorm_lastdim(q_heads, q_norm_vec, eps=self.rms_eps)
        if k_norm_w is not None:
            k_norm_vec = np.asarray(k_norm_w, dtype=np.float32).reshape(-1)
            if k_norm_vec.shape[0] == head_dim:
                k_heads_new = self._rmsnorm_lastdim(
                    k_heads_new, k_norm_vec, eps=self.rms_eps
                )

        for t in range(seq_len):
            q_t = q_heads[t]
            k_t = k_heads_new[t]
            rope_dim = head_dim if self.rope_dim <= 0 else min(head_dim, self.rope_dim)
            if rope_dim > 0:
                if (
                    qwen_full_attn
                    and self.qwen_is_mrope
                    and any(self.qwen_rope_sections)
                ):
                    self._apply_qwen_mrope_inplace(
                        q_t,
                        k_t,
                        rope_dim=rope_dim,
                        pos=start_pos + t,
                    )
                else:
                    simd_ops.rope(
                        q_t[:, :rope_dim],
                        k_t[:, :rope_dim],
                        n_heads,
                        n_kv_heads,
                        rope_dim,
                        start_pos + t,
                        self.rope_theta,
                    )
            q_heads[t] = q_t
            k_heads_new[t] = k_t

        k_flat = k_heads_new.reshape(seq_len, -1)
        v_flat = v_heads_new.reshape(seq_len, -1)
        self.kv_cache.append(layer_idx, k_flat, v_flat, start_pos)
        scale = (
            self.attention_scale
            if self.attention_scale > 0.0
            else (1.0 / math.sqrt(float(head_dim)))
        )
        kv_len = int(start_pos) + int(seq_len)
        attn_out = self._try_native_kv_flash_attention(
            layer_idx=layer_idx,
            q_heads=q_heads,
            seq_len=seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            kv_len=kv_len,
            scale=float(scale),
            start_pos=int(start_pos),
        )
        k_heads = None
        v_heads = None
        if attn_out is None:
            k_full, v_full = self.kv_cache.get(layer_idx)
            kv_len = k_full.shape[0]
            if kv_len == 0:
                return np.zeros_like(normed, dtype=np.float32)
            k_heads = k_full.reshape(kv_len, n_kv_heads, head_dim)
            v_heads = v_full.reshape(kv_len, n_kv_heads, head_dim)
            heads_per_kv = max(1, n_heads // max(1, n_kv_heads))
            attn_out = self._try_fused_mqa_attention(
                layer_idx=layer_idx,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
                seq_len=seq_len,
                kv_len=kv_len,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                scale=float(scale),
                start_pos=int(start_pos),
            )
        if attn_out is None:
            if k_heads is None or v_heads is None:
                k_full, v_full = self.kv_cache.get(layer_idx)
                kv_len = k_full.shape[0]
                if kv_len == 0:
                    return np.zeros_like(normed, dtype=np.float32)
                k_heads = k_full.reshape(kv_len, n_kv_heads, head_dim)
                v_heads = v_full.reshape(kv_len, n_kv_heads, head_dim)
            heads_per_kv = max(1, n_heads // max(1, n_kv_heads))
            attn_out = np.zeros((seq_len, n_heads, head_dim), dtype=np.float32)
            for h in range(n_heads):
                kv_h = min(n_kv_heads - 1, h // heads_per_kv)
                for t in range(seq_len):
                    scores = np.dot(k_heads[:, kv_h, :], q_heads[t, h, :]) * scale
                    causal_len = min(kv_len, start_pos + t + 1)
                    if causal_len < kv_len:
                        scores[causal_len:] = -1e9
                    probs = simd_ops.softmax(scores.copy())
                    attn_out[t, h, :] = probs @ v_heads[:, kv_h, :]

        context = attn_out.reshape(seq_len, -1)
        # Qwen3.5 full attention gates the attention output before the final O projection.
        if gate_direct is not None and gate_direct.shape == context.shape:
            context = context * self._sigmoid(gate_direct)
        else:
            gate_w = weights.get("attn_gate")
            if gate_w is not None:
                gate_proj = self._linear_project_batch(normed, gate_w)
                if gate_proj.shape == context.shape:
                    context = context * self._sigmoid(gate_proj)

        output_w = weights.get("attn_output")
        projected = (
            self._linear_project_batch(context, output_w)
            if output_w is not None
            else context
        )

        return np.asarray(projected, dtype=np.float32)

    # ------------------------------------------------------------------
    # Attention block
    # ------------------------------------------------------------------

    def _forward_attention(
        self,
        x: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        start_pos: int,
    ) -> np.ndarray:
        dim = self.profile.embedding_dim
        seq_len = x.shape[0]

        # 1. Pre-attention RMSNorm
        norm_w = weights.get("attn_norm")
        normed = (
            simd_ops.rmsnorm(x.copy(), norm_w, eps=self.rms_eps)
            if norm_w is not None
            else x.copy()
        )

        # 2. QKV projection
        q, k, v = self._project_qkv(normed, weights)
        if q is None or k is None or v is None:
            return x

        # Determine head dimensions
        n_heads = self.profile.n_heads
        n_kv_heads = self.profile.n_kv_heads or n_heads
        if n_heads <= 0 or n_kv_heads <= 0:
            return x
        q_head_dim = q.shape[1] // n_heads if q.shape[1] >= n_heads else 0
        k_head_dim = k.shape[1] // n_kv_heads if k.shape[1] >= n_kv_heads else 0
        v_head_dim = v.shape[1] // n_kv_heads if v.shape[1] >= n_kv_heads else 0
        head_dim = min(q_head_dim, k_head_dim, v_head_dim)
        if head_dim <= 0:
            return x
        if q.shape[1] != n_heads * head_dim:
            q = q[:, : n_heads * head_dim]
        if k.shape[1] != n_kv_heads * head_dim:
            k = k[:, : n_kv_heads * head_dim]
        if v.shape[1] != n_kv_heads * head_dim:
            v = v[:, : n_kv_heads * head_dim]

        # 3. Apply RoPE
        for t in range(seq_len):
            q_t = q[t].reshape(n_heads, head_dim)
            k_t = k[t].reshape(n_kv_heads, head_dim)
            rope_dim = head_dim if self.rope_dim <= 0 else min(head_dim, self.rope_dim)
            if rope_dim > 0:
                simd_ops.rope(
                    q_t[:, :rope_dim],
                    k_t[:, :rope_dim],
                    n_heads,
                    n_kv_heads,
                    rope_dim,
                    start_pos + t,
                    self.rope_theta,
                )
            q[t] = q_t.reshape(-1)
            k[t] = k_t.reshape(-1)

        # 4. KV cache update
        self.kv_cache.append(layer_idx, k, v, start_pos)
        q_heads = q.reshape(seq_len, n_heads, head_dim)
        scale = 1.0 / math.sqrt(head_dim)

        kv_len = int(start_pos) + int(seq_len)
        attn_out = self._try_native_kv_flash_attention(
            layer_idx=layer_idx,
            q_heads=q_heads,
            seq_len=seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            kv_len=kv_len,
            scale=float(scale),
            start_pos=int(start_pos),
        )
        k_heads = None
        v_heads = None
        if attn_out is None:
            k_full, v_full = self.kv_cache.get(layer_idx)
            kv_len = k_full.shape[0]
            if kv_len == 0:
                return x
            k_heads = k_full.reshape(kv_len, n_kv_heads, head_dim)
            v_heads = v_full.reshape(kv_len, n_kv_heads, head_dim)
            attn_out = self._try_fused_mqa_attention(
                layer_idx=layer_idx,
                q_heads=q_heads,
                k_heads=k_heads,
                v_heads=v_heads,
                seq_len=seq_len,
                kv_len=kv_len,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                scale=float(scale),
                start_pos=int(start_pos),
            )
        if attn_out is None:
            if k_heads is None or v_heads is None:
                k_full, v_full = self.kv_cache.get(layer_idx)
                kv_len = k_full.shape[0]
                if kv_len == 0:
                    return x
                k_heads = k_full.reshape(kv_len, n_kv_heads, head_dim)
                v_heads = v_full.reshape(kv_len, n_kv_heads, head_dim)
            heads_per_kv = n_heads // n_kv_heads if n_kv_heads > 0 else 1
            attn_out = np.zeros((seq_len, n_heads, head_dim), dtype=np.float32)
            for h in range(n_heads):
                kv_h = h // heads_per_kv  # GQA: map query head to KV head
                for t in range(seq_len):
                    # scores[j] = q[t, h] · k[j, kv_h] for all j
                    scores = np.dot(k_heads[:, kv_h, :], q_heads[t, h, :]) * scale
                    # Causal mask
                    causal_len = min(kv_len, start_pos + t + 1)
                    if causal_len < kv_len:
                        scores[causal_len:] = -1e9
                    # Softmax
                    probs = simd_ops.softmax(scores.copy())
                    # Weighted sum of values
                    attn_out[t, h, :] = probs @ v_heads[:, kv_h, :]

        # Reshape back to [seq, dim]
        context = attn_out.reshape(seq_len, -1)

        # 6. Output projection + optional gate
        output_w = weights.get("attn_output")
        if output_w is not None:
            projected = self._linear_project_batch(context, output_w)
        else:
            projected = context

        gate_w = weights.get("attn_gate")
        if gate_w is not None:
            gate_proj = self._linear_project_batch(normed, gate_w)
            gate_vals = self._sigmoid(gate_proj)
            # Match dimensions
            if gate_vals.shape == projected.shape:
                projected = projected * gate_vals

        # 7. Residual
        if self.enable_residual_stabilizer:
            projected = self._stabilize_residual_branch(
                projected,
                x,
                max_ratio=self.residual_max_ratio,
            )
        return x + projected

    def _try_native_kv_flash_attention(
        self,
        layer_idx: int,
        q_heads: np.ndarray,
        seq_len: int,
        n_heads: int,
        head_dim: int,
        kv_len: int,
        scale: float,
        start_pos: int,
    ) -> np.ndarray | None:
        flash_fn = getattr(self.kv_cache, "flash_attention", None)
        if not callable(flash_fn):
            return None
        if seq_len != 1:
            return None
        if kv_len <= 0 or kv_len != (start_pos + 1):
            return None
        if q_heads.shape != (seq_len, n_heads, head_dim):
            return None
        try:
            q = np.asarray(q_heads[0], dtype=np.float32)
            out = flash_fn(
                layer_idx=int(layer_idx),
                q=q,
                n_heads=int(n_heads),
                kv_len=int(kv_len),
                scale=float(scale),
            )
            if out is None:
                return None
            out_f = np.asarray(out, dtype=np.float32)
            if out_f.shape != (n_heads, head_dim):
                return None
            return out_f.reshape(1, n_heads, head_dim)
        except Exception:
            return None

    def _try_fused_mqa_attention(
        self,
        layer_idx: int,
        q_heads: np.ndarray,
        k_heads: np.ndarray,
        v_heads: np.ndarray,
        seq_len: int,
        kv_len: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        scale: float,
        start_pos: int,
    ) -> np.ndarray | None:
        _ = layer_idx
        if not self.enable_fused_mqa or _fused_attention_mqa is None:
            return None
        # The native fused kernel currently has no causal-mask input. Restrict to
        # decode hot-path where seq_q == 1 and full KV is already causal-safe.
        if seq_len != 1:
            return None
        if kv_len <= 0 or kv_len != (start_pos + 1):
            return None
        if (
            q_heads.shape != (seq_len, n_heads, head_dim)
            or k_heads.shape != (kv_len, n_kv_heads, head_dim)
            or v_heads.shape != (kv_len, n_kv_heads, head_dim)
        ):
            return None
        try:
            q = np.transpose(np.asarray(q_heads, dtype=np.float32), (1, 0, 2))[
                np.newaxis, ...
            ]
            k = np.transpose(np.asarray(k_heads, dtype=np.float32), (1, 0, 2))[
                np.newaxis, ...
            ]
            v = np.transpose(np.asarray(v_heads, dtype=np.float32), (1, 0, 2))[
                np.newaxis, ...
            ]
            out = _fused_attention_mqa(q, k, v, scale=float(scale))
            out_f = np.asarray(out, dtype=np.float32)
            if out_f.shape != (1, n_heads, seq_len, head_dim):
                return None
            return np.transpose(out_f[0], (1, 0, 2))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # SSM block (Gated Delta Networks — linear recurrence)
    # ------------------------------------------------------------------

    def _forward_ssm(
        self,
        x: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
    ) -> np.ndarray:
        seq_len = x.shape[0]

        # Pre-SSM norm — use attn_norm (not ssm_norm, which is for inner state)
        norm_w = weights.get("attn_norm")
        normed = (
            simd_ops.rmsnorm(x.copy(), norm_w, eps=self.rms_eps)
            if norm_w is not None
            else x.copy()
        )

        # Detect which SSM variant based on available tensors
        ssm_in_w = weights.get("ssm_in")
        ssm_alpha_w = weights.get("ssm_alpha")
        a = weights.get("ssm_a")

        if a is None:
            return x

        a_vec = np.asarray(a, dtype=np.float32).reshape(-1)
        state_dim = a_vec.shape[0]

        if layer_idx not in self.ssm_states:
            self.ssm_states[layer_idx] = np.zeros(state_dim, dtype=np.float32)

        if ssm_alpha_w is not None:
            # Gated Delta Networks (qwen3.5): alpha/beta projections
            beta_w = weights.get("ssm_beta")
            alpha_proj = self._linear_project_batch(normed, ssm_alpha_w)

            if seq_len == 1:
                x_proj = alpha_proj[0]
                self.ssm_states[layer_idx] = simd_ops.ssm_step(
                    self.ssm_states[layer_idx], a_vec, x_proj
                )
                h_out = self.ssm_states[layer_idx].reshape(1, -1)
            else:
                H = simd_ops.ssm_parallel_scan(
                    a_vec, alpha_proj, self.ssm_states[layer_idx]
                )
                self.ssm_states[layer_idx] = H[-1].copy()
                h_out = H

            if beta_w is not None:
                ssm_out = self._linear_project_batch(h_out, beta_w)
            else:
                ssm_out = h_out

        elif ssm_in_w is not None:
            # Mamba-1 SSM (granite4 hybrid)
            # Pipeline: in_proj → split(conv_data, gate) → conv1d → silu → state_update → gate → norm → out_proj
            ssm_out = self._forward_mamba_ssm(
                normed, layer_idx, weights, a_vec, state_dim
            )
        else:
            return x

        # Ensure output matches input dim for residual
        if ssm_out.shape[1] != x.shape[1]:
            ssm_out = self._match_dim(ssm_out, x.shape[1])

        if self.enable_residual_stabilizer:
            ssm_out = self._stabilize_residual_branch(
                ssm_out,
                x,
                max_ratio=self.residual_max_ratio,
            )
        return x + ssm_out

    def _forward_mamba_ssm(
        self,
        normed: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        a_vec: np.ndarray,
        state_dim: int,
    ) -> np.ndarray:
        """Mamba-2 SSM block for granite4 hybrid architecture.

        Mamba-2 in-projection splits: [z, xBC, dt]
          z:   d_inner (3072) channels — gate path
          xBC: d_conv  (3328) channels — conv path containing x+B+C
          dt:  n_heads (48)   channels — per-head timestep

        After conv1d + SiLU on xBC(3328), split into:
          x: d_inner (3072) channels — SSM input
          B: d_state (128) — input-dependent state projection
          C: d_state (128) — output-dependent state readout

        SSM per-head: h = exp(A*dt)*h + B*dt*x, y = C*h + D*x
        Output: RMSNorm(y) * SiLU(z) → out_proj
        """
        seq_len = normed.shape[0]
        ssm_in_w = weights["ssm_in"]  # [1536, 6448]
        conv1d_w = weights.get("ssm_conv1d")  # [4, 3328]
        conv1d_b = weights.get("ssm_conv1d_bias")  # [3328]
        ssm_d = weights.get("ssm_d")
        ssm_dt_bias = weights.get("ssm_dt")
        ssm_norm_w = weights.get("ssm_norm")  # [3072]
        ssm_out_w = weights.get("ssm_out")  # [3072, 1536]

        conv_kernel = None
        if isinstance(conv1d_w, np.ndarray) and conv1d_w.ndim == 2:
            # GGUF runtime tensors may expose conv weights as [channels, kernel] even when
            # metadata shape is [kernel, channels]. Normalize to [kernel, channels].
            if conv1d_w.shape[0] <= 16 and conv1d_w.shape[1] > conv1d_w.shape[0]:
                conv_kernel = np.asarray(conv1d_w, dtype=np.float32)
            elif conv1d_w.shape[1] <= 16 and conv1d_w.shape[0] > conv1d_w.shape[1]:
                conv_kernel = np.asarray(conv1d_w.T, dtype=np.float32)
            else:
                conv_kernel = np.asarray(conv1d_w, dtype=np.float32)

        # Dimensions
        n_heads = a_vec.shape[0]  # 48
        d_inner = ssm_norm_w.shape[0] if ssm_norm_w is not None else 3072
        d_conv = conv_kernel.shape[1] if conv_kernel is not None else 3328
        d_state = (d_conv - d_inner) // 2  # (3328-3072)//2 = 128

        # 1. In-projection: [seq, 1536] → [seq, 6448]
        inner = self._linear_project_batch(normed, ssm_in_w)

        # 2. Split into [z, xBC, dt]
        z = inner[:, :d_inner]  # [seq, 3072] — gate
        xBC = inner[:, d_inner : d_inner + d_conv]  # [seq, 3328] — conv path
        dt_proj = inner[:, d_inner + d_conv :]  # [seq, 48]  — timestep

        # 3. Conv1d (causal) on xBC
        if conv_kernel is not None:
            kernel_width = conv_kernel.shape[0]
            conv_key = f"conv_buf_{layer_idx}"
            if not hasattr(self, "_conv_bufs"):
                self._conv_bufs = {}
            if conv_key not in self._conv_bufs:
                self._conv_bufs[conv_key] = np.zeros(
                    (kernel_width - 1, d_conv), dtype=np.float32
                )

            padded = np.vstack([self._conv_bufs[conv_key], xBC])
            conv_out = np.zeros_like(xBC)
            for t in range(seq_len):
                for k in range(kernel_width):
                    conv_out[t] += conv_kernel[k] * padded[t + k]
            if conv1d_b is not None:
                conv_bias = np.asarray(conv1d_b, dtype=np.float32).reshape(-1)
                if conv_bias.shape[0] == d_conv:
                    conv_out += conv_bias.reshape(1, -1)
            self._conv_bufs[conv_key] = padded[-(kernel_width - 1) :].copy()
        else:
            conv_out = xBC

        # 4. SiLU activation on conv output
        silu_conv = conv_out * self._sigmoid(conv_out)

        # 5. Split conv output into [x, B, C]
        x_ssm = silu_conv[:, :d_inner]  # [seq, 3072]
        B = silu_conv[:, d_inner : d_inner + d_state]  # [seq, 128]
        C = silu_conv[:, d_inner + d_state : d_inner + 2 * d_state]  # [seq, 128]

        # 6. Compute per-head parameters
        dt_bias_v = (
            np.asarray(ssm_dt_bias, dtype=np.float32).reshape(-1)
            if ssm_dt_bias is not None
            else np.zeros(n_heads, dtype=np.float32)
        )
        d_vec = (
            np.asarray(ssm_d, dtype=np.float32).reshape(-1)
            if ssm_d is not None
            else np.zeros(n_heads, dtype=np.float32)
        )

        # GGUF conversion stores Mamba A in continuous form for runtime scan.
        A = np.asarray(a_vec, dtype=np.float32).reshape(-1)

        head_dim = d_inner // n_heads  # 64

        # 7. SSM state: [n_heads, head_dim, d_state]
        state_key = f"mamba2_{layer_idx}"
        if state_key not in self.ssm_states:
            self.ssm_states[state_key] = np.zeros(
                (n_heads, head_dim, d_state), dtype=np.float32
            )

        # 8. Mamba-2 selective scan (reference: HuggingFace torch_forward)
        # Reshape x to [seq, n_heads, head_dim]
        x_heads = x_ssm.reshape(seq_len, n_heads, head_dim)

        # B/C: repeat_interleave for n_groups → n_heads (if n_groups < n_heads)
        n_groups = B.shape[1] // d_state if d_state > 0 else 1
        heads_per_group = n_heads // max(n_groups, 1)

        rows = []
        for t in range(seq_len):
            # dt: softplus(dt_proj + dt_bias)
            dt_t = self._softplus(
                dt_proj[t, :n_heads] + dt_bias_v[:n_heads]
            )  # [n_heads]

            # B_t: [n_groups, d_state] → repeat to [n_heads, d_state]
            B_t = B[t].reshape(n_groups, -1)  # [n_groups, d_state]
            B_t = np.repeat(B_t, heads_per_group, axis=0)  # [n_heads, d_state]

            # C_t: same expansion
            C_t = C[t].reshape(n_groups, -1)
            C_t = np.repeat(C_t, heads_per_group, axis=0)  # [n_heads, d_state]

            state = self.ssm_states[state_key]  # [n_heads, head_dim, d_state]
            x_t = x_heads[t, :n_heads, :]  # [n_heads, head_dim]
            dA = (
                self._safe_exp(A[:n_heads] * dt_t[:n_heads])
                .astype(np.float32)
                .reshape(n_heads, 1, 1)
            )
            dB = (
                (dt_t[:n_heads, np.newaxis] * B_t[:n_heads])
                .astype(np.float32)
                .reshape(n_heads, 1, d_state)
            )
            dBx = x_t[:, :, np.newaxis] * dB  # [n_heads, head_dim, d_state]
            state[:n_heads] = dA * state[:n_heads] + dBx

            # y = state @ C + D * x
            y_heads = np.einsum("hds,hs->hd", state[:n_heads], C_t[:n_heads]).astype(
                np.float32
            )
            y_heads += d_vec[:n_heads, np.newaxis] * x_t
            rows.append(y_heads.reshape(-1))
        recurrent = np.asarray(rows, dtype=np.float32)  # [seq, d_inner]

        # 9. GraniteMoeHybridRMSNormGated ordering: RMSNorm(y * SiLU(z)).
        gate_silu = z * self._sigmoid(z)
        gated = recurrent * gate_silu
        if ssm_norm_w is not None and gated.shape[1] == ssm_norm_w.shape[0]:
            gated = simd_ops.rmsnorm(gated, ssm_norm_w, eps=self.rms_eps)

        # 11. Out-projection
        if ssm_out_w is not None:
            ssm_out = self._linear_project_batch(gated, ssm_out_w)
        else:
            ssm_out = gated

        return ssm_out

    @staticmethod
    def _expand_state(x: np.ndarray, target_dim: int) -> np.ndarray:
        """Expand state_dim to target output dim via repetition."""
        if x.shape[1] == target_dim:
            return x
        if x.shape[1] > target_dim:
            return x[:, :target_dim]
        repeats = math.ceil(target_dim / x.shape[1])
        return np.tile(x, (1, repeats))[:, :target_dim]

    # ------------------------------------------------------------------
    # FFN block (SwiGLU)
    # ------------------------------------------------------------------

    def _forward_ffn(
        self,
        x: np.ndarray,
        layer_idx: int,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
        add_residual: bool = True,
    ) -> np.ndarray:
        norm_w = weights.get("post_attn_norm")
        if norm_w is None:
            norm_w = weights.get("ffn_norm")
        normed = (
            simd_ops.rmsnorm(x.copy(), norm_w, eps=self.rms_eps)
            if norm_w is not None
            else x.copy()
        )

        gate_w = weights.get("ffn_gate")
        up_w = weights.get("ffn_up")
        down_w = weights.get("ffn_down")

        # Standard SwiGLU FFN (non-MoE, 2D weights only)
        if (
            gate_w is not None
            and up_w is not None
            and down_w is not None
            and self._is_matrix_weight(gate_w)
            and self._is_matrix_weight(up_w)
            and self._is_matrix_weight(down_w)
        ):
            gate = self._linear_project_batch(normed, gate_w)
            up = self._linear_project_batch(normed, up_w)
            hidden = simd_ops.swiglu(gate, up)
            out = self._linear_project_batch(hidden, down_w)
            if out.shape[1] != x.shape[1]:
                out = self._match_dim(out, x.shape[1])
            return (x + out) if add_residual else out

        # MoE FFN: shared expert + routed top-1 expert
        gate_shexp = weights.get("ffn_gate_shexp")
        up_shexp = weights.get("ffn_up_shexp")
        down_shexp = weights.get("ffn_down_shexp")
        router_w = weights.get("ffn_gate_inp")

        shared_out: Optional[np.ndarray] = None

        if (
            not self.disable_moe_shared
            and gate_shexp is not None
            and up_shexp is not None
            and down_shexp is not None
            and self._is_matrix_weight(gate_shexp)
            and self._is_matrix_weight(up_shexp)
            and self._is_matrix_weight(down_shexp)
        ):
            gate = self._linear_project_batch(normed, gate_shexp)
            up_proj = self._linear_project_batch(normed, up_shexp)
            hidden = simd_ops.swiglu(gate, up_proj)
            shared_out = self._linear_project_batch(hidden, down_shexp)

            if shared_out.shape[1] != x.shape[1]:
                shared_out = self._match_dim(shared_out, x.shape[1])

        routed_out: Optional[np.ndarray] = None
        gate_exps_name, up_exps_name, down_exps_name = (
            self.weights.get_expert_tensor_names(layer_idx)
        )
        if (
            not self.disable_moe_routed
            and router_w is not None
            and self._is_matrix_weight(router_w)
            and gate_exps_name is not None
            and up_exps_name is not None
            and down_exps_name is not None
        ):
            routed_out = self._forward_moe_experts_lazy(
                normed=normed,
                router_w=router_w,
                gate_name=gate_exps_name,
                up_name=up_exps_name,
                down_name=down_exps_name,
                top_k=max(1, self.moe_top_k),
            )

        if shared_out is not None and routed_out is not None:
            out = shared_out + routed_out
            return (x + out) if add_residual else out
        if shared_out is not None:
            return (x + shared_out) if add_residual else shared_out
        if routed_out is not None:
            return (x + routed_out) if add_residual else routed_out

        if add_residual:
            return x  # No FFN weights found
        return np.zeros_like(x, dtype=np.float32)

    def _forward_moe_experts_lazy(
        self,
        normed: np.ndarray,
        router_w: np.ndarray | quant_ops.QuantizedMatrix,
        gate_name: str,
        up_name: str,
        down_name: str,
        top_k: int = 1,
    ) -> Optional[np.ndarray]:
        """Route through top-k experts using fused C kernel when possible."""
        try:
            seq_len = normed.shape[0]
            router_logits = self._linear_project_batch(normed, router_w)
            if router_logits.ndim != 2 or router_logits.shape[1] <= 0:
                return None

            results = np.zeros((seq_len, normed.shape[1]), dtype=np.float32)
            for t in range(seq_len):
                logits_t = np.asarray(router_logits[t], dtype=np.float32).reshape(-1)
                k = min(max(1, int(top_k)), logits_t.shape[0])
                if k == logits_t.shape[0]:
                    top_idx = np.arange(logits_t.shape[0], dtype=np.int64)
                else:
                    top_idx = np.argpartition(logits_t, -k)[-k:]
                top_scores = logits_t[top_idx]
                order = np.argsort(-top_scores)
                top_idx = top_idx[order]
                top_scores = top_scores[order]
                probs = np.exp(top_scores - np.max(top_scores))
                probs = probs / (np.sum(probs) + 1e-8)

                out_t = np.zeros((normed.shape[1],), dtype=np.float32)
                selected: list[
                    tuple[
                        np.ndarray | quant_ops.QuantizedMatrix,
                        np.ndarray | quant_ops.QuantizedMatrix,
                        np.ndarray | quant_ops.QuantizedMatrix,
                        float,
                    ]
                ] = []
                all_quantized = True
                for expert_idx, softmax_weight in zip(top_idx.tolist(), probs.tolist()):
                    g_w = self.weights.get_expert_matrix(gate_name, int(expert_idx))
                    u_w = self.weights.get_expert_matrix(up_name, int(expert_idx))
                    d_w = self.weights.get_expert_matrix(down_name, int(expert_idx))
                    if g_w is None or u_w is None or d_w is None:
                        continue
                    if not (
                        isinstance(g_w, quant_ops.QuantizedMatrix)
                        and isinstance(u_w, quant_ops.QuantizedMatrix)
                        and isinstance(d_w, quant_ops.QuantizedMatrix)
                    ):
                        all_quantized = False
                    selected.append((g_w, u_w, d_w, float(softmax_weight)))

                if selected and all_quantized:
                    gate_mats = [entry[0] for entry in selected]
                    up_mats = [entry[1] for entry in selected]
                    down_mats = [entry[2] for entry in selected]
                    moe_weights = [entry[3] for entry in selected]
                    out_t = quant_ops.fused_moe_ffn(
                        normed[t],
                        expert_weights=moe_weights,
                        gate_matrices=gate_mats,  # type: ignore[arg-type]
                        up_matrices=up_mats,  # type: ignore[arg-type]
                        down_matrices=down_mats,  # type: ignore[arg-type]
                    )
                else:
                    for g_w, u_w, d_w, softmax_weight in selected:
                        if (
                            isinstance(g_w, quant_ops.QuantizedMatrix)
                            and isinstance(u_w, quant_ops.QuantizedMatrix)
                            and isinstance(d_w, quant_ops.QuantizedMatrix)
                        ):
                            out = quant_ops.fused_expert_swiglu(
                                normed[t], g_w, u_w, d_w
                            )
                        else:
                            gate = self._linear_project(normed[t], g_w)
                            up = self._linear_project(normed[t], u_w)
                            hidden = simd_ops.swiglu(gate, up)
                            out = self._linear_project(hidden, d_w)

                        if out.shape[0] != normed.shape[1]:
                            out = self._match_dim(
                                out.reshape(1, -1), normed.shape[1]
                            ).reshape(-1)
                        out_t += float(softmax_weight) * out
                results[t] = out_t

            return results
        except Exception:
            return None  # Skip if expert routing fails

    # ------------------------------------------------------------------
    # QKV projection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_qwen_q_and_gate(
        q_full: np.ndarray,
        head_dim_hint: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        q_full_f = np.asarray(q_full, dtype=np.float32)
        if q_full_f.ndim != 2:
            return q_full_f, None

        seq_len, total_dim = q_full_f.shape
        if total_dim <= 1:
            return q_full_f, None

        # Qwen3.5 full-attn packs each head as [q_head, gate_head].
        if head_dim_hint > 0:
            pair_dim = 2 * int(head_dim_hint)
            if pair_dim > 0 and (total_dim % pair_dim) == 0:
                n_heads = total_dim // pair_dim
                packed = q_full_f.reshape(seq_len, n_heads, 2, int(head_dim_hint))
                q = packed[:, :, 0, :].reshape(seq_len, n_heads * int(head_dim_hint))
                gate = packed[:, :, 1, :].reshape(seq_len, n_heads * int(head_dim_hint))
                return q, gate

        # Fallback for unknown layouts: split contiguous halves.
        half = total_dim // 2
        if half <= 0:
            return q_full_f, None
        q = q_full_f[:, :half]
        gate = q_full_f[:, half : half + half]
        return q, gate

    @staticmethod
    def _qwen_delta_step(
        state: np.ndarray,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gate: float,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        s_h = np.asarray(state, dtype=np.float32)
        q_h = np.asarray(q, dtype=np.float32).reshape(-1)
        k_h = np.asarray(k, dtype=np.float32).reshape(-1)
        v_h = np.asarray(v, dtype=np.float32).reshape(-1)

        gated_state = s_h * float(np.exp(np.clip(gate, -60.0, 60.0)))
        sk = gated_state @ k_h
        d = (v_h - sk) * float(beta)
        next_state = gated_state + np.outer(d, k_h)
        out = next_state @ q_h
        return np.asarray(out, dtype=np.float32), np.asarray(
            next_state, dtype=np.float32
        )

    def _project_qkv(
        self,
        normed: np.ndarray,
        weights: dict[str, np.ndarray | quant_ops.QuantizedMatrix],
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        qkv_w = weights.get("attn_qkv")
        if qkv_w is not None:
            fused = self._linear_project_batch(normed, qkv_w)
            dims = self.weights.attention_dims()
            if dims is None or sum(dims) > fused.shape[1]:
                model_dim = int(self.profile.embedding_dim)
                n_heads = max(1, int(self.profile.n_heads))
                n_kv_heads = max(
                    1, int(self.profile.n_kv_heads or self.profile.n_heads)
                )
                q_dim = min(model_dim, fused.shape[1])
                remaining = fused.shape[1] - q_dim
                kv_dim = max(0, remaining // 2)
                kv_dim -= kv_dim % n_kv_heads
                q_dim = fused.shape[1] - 2 * kv_dim
                if kv_dim <= 0 or q_dim <= 0 or (q_dim % n_heads) != 0:
                    return None, None, None
                dims = (q_dim, kv_dim, kv_dim)
            q_dim, k_dim, v_dim = dims
            q = fused[:, :q_dim]
            k = fused[:, q_dim : q_dim + k_dim]
            v = fused[:, q_dim + k_dim : q_dim + k_dim + v_dim]
            return q, k, v

        q_w = weights.get("attn_q")
        k_w = weights.get("attn_k")
        v_w = weights.get("attn_v")
        if q_w is None or k_w is None or v_w is None:
            return None, None, None
        q = self._linear_project_batch(normed, q_w)
        k = self._linear_project_batch(normed, k_w)
        v = self._linear_project_batch(normed, v_w)
        return q, k, v

    # ------------------------------------------------------------------
    # Linear algebra helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        x_f = np.asarray(x, dtype=np.float32)
        return np.log1p(np.exp(-np.abs(x_f))) + np.maximum(x_f, 0.0)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x_f = np.asarray(x, dtype=np.float32)
        x_f = np.clip(x_f, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x_f))

    @staticmethod
    def _safe_exp(x: np.ndarray, lo: float = -60.0, hi: float = 60.0) -> np.ndarray:
        x_f = np.asarray(x, dtype=np.float32)
        return np.exp(np.clip(x_f, lo, hi))

    @staticmethod
    def _normalize_conv_kernel(
        conv1d_w: np.ndarray | quant_ops.QuantizedMatrix | None,
    ) -> np.ndarray | None:
        if not isinstance(conv1d_w, np.ndarray) or conv1d_w.ndim != 2:
            return None
        kernel = np.asarray(conv1d_w, dtype=np.float32)
        if kernel.shape[0] <= 16 and kernel.shape[1] > kernel.shape[0]:
            return kernel
        if kernel.shape[1] <= 16 and kernel.shape[0] > kernel.shape[1]:
            return kernel.T
        return kernel

    @staticmethod
    def _rmsnorm_lastdim(
        x: np.ndarray, weight: np.ndarray, eps: float = 1.0e-6
    ) -> np.ndarray:
        x_f = np.asarray(x, dtype=np.float32)
        w_f = np.asarray(weight, dtype=np.float32).reshape(-1)
        if x_f.shape[-1] != w_f.shape[0]:
            return x_f
        rms = np.sqrt(np.mean(x_f * x_f, axis=-1, keepdims=True) + eps)
        return (x_f / rms) * w_f.reshape((1,) * (x_f.ndim - 1) + (-1,))

    @staticmethod
    def _linear_project(
        x: np.ndarray,
        weight: np.ndarray | quant_ops.QuantizedMatrix,
    ) -> np.ndarray:
        """Project vector x through weight matrix.

        Weight is expected [out_dim, in_dim] (standard convention for GGUF).
        Result: y = W @ x  (shape [out_dim]).

        If weight shape is [in_dim, out_dim], we detect and transpose.
        """
        x_f = np.asarray(x, dtype=np.float32).reshape(-1)
        if isinstance(weight, quant_ops.QuantizedMatrix):
            return quant_ops.matvec(x_f, weight)

        w_f = np.asarray(weight, dtype=np.float32)

        if w_f.ndim == 1:
            # Element-wise scale
            if w_f.shape[0] == x_f.shape[0]:
                return x_f * w_f
            return x_f  # Can't use 1D weight with mismatched dim

        # Determine orientation:
        # Convention: W is [out_dim, in_dim], result = W @ x = [out_dim]
        # If W.shape[1] == x.shape[0]: y = W @ x
        # If W.shape[0] == x.shape[0]: y = W^T @ x
        if w_f.shape[1] == x_f.shape[0]:
            return simd_ops.matvec(x_f, w_f.T)
        elif w_f.shape[0] == x_f.shape[0]:
            return simd_ops.matvec(x_f, w_f)
        else:
            # Dimension mismatch — truncate/pad input to match
            in_dim = min(w_f.shape[0], w_f.shape[1])
            if x_f.shape[0] > in_dim:
                x_f = x_f[:in_dim]
            elif x_f.shape[0] < in_dim:
                padded = np.zeros(in_dim, dtype=np.float32)
                padded[: x_f.shape[0]] = x_f
                x_f = padded
            if w_f.shape[0] == in_dim:
                return simd_ops.matvec(x_f, w_f)
            else:
                return simd_ops.matvec(x_f, w_f.T)

    @staticmethod
    def _linear_project_batch(
        x: np.ndarray,
        weight: np.ndarray | quant_ops.QuantizedMatrix,
    ) -> np.ndarray:
        """Project a batch of vectors through weight matrix using native SIMD kernels."""
        x_f = np.asarray(x, dtype=np.float32)
        if isinstance(weight, quant_ops.QuantizedMatrix):
            if x_f.ndim == 1:
                return quant_ops.matvec(x_f, weight)
            if x_f.ndim == 2 and x_f.shape[0] == 1:
                return quant_ops.matvec(x_f[0], weight).reshape(1, weight.output_dim)
            return quant_ops.matmul(x_f, weight)

        w_f = np.asarray(weight, dtype=np.float32)

        if x_f.ndim == 1:
            return QSGForwardPass._linear_project(x_f, w_f)

        if w_f.ndim == 1:
            # Element-wise scale
            if w_f.shape[0] == x_f.shape[1]:
                return x_f * w_f[np.newaxis, :]
            return x_f

        # Determine orientation and do batch matmul
        # W is [out_dim, in_dim]: result = x @ W.T = [batch, out_dim]
        # W is [in_dim, out_dim]: result = x @ W   = [batch, out_dim]
        if w_f.shape[1] == x_f.shape[1]:
            # W[out, in], x[batch, in] → x @ W.T
            return simd_ops.matmul(x_f, w_f.T)
        elif w_f.shape[0] == x_f.shape[1]:
            # W[in, out], x[batch, in] → x @ W
            return simd_ops.matmul(x_f, w_f)
        else:
            # Dimension mismatch — pad/truncate input
            in_dim = w_f.shape[1] if w_f.shape[1] <= w_f.shape[0] else w_f.shape[0]
            if x_f.shape[1] > in_dim:
                x_f = x_f[:, :in_dim]
            elif x_f.shape[1] < in_dim:
                padded = np.zeros((x_f.shape[0], in_dim), dtype=np.float32)
                padded[:, : x_f.shape[1]] = x_f
                x_f = padded
            if w_f.shape[0] == in_dim:
                return simd_ops.matmul(x_f, w_f)
            else:
                return simd_ops.matmul(x_f, w_f.T)

    @staticmethod
    def _reduce_to_state(x: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce feature dim to state_dim via mean-pooling chunks."""
        if x.shape[1] == target_dim:
            return x
        if x.shape[1] < target_dim:
            # Pad with zeros
            padded = np.zeros((x.shape[0], target_dim), dtype=np.float32)
            padded[:, : x.shape[1]] = x
            return padded
        # Mean-pool chunks
        chunks = np.array_split(x, target_dim, axis=1)
        return np.stack([chunk.mean(axis=1) for chunk in chunks], axis=1)

    @staticmethod
    def _match_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
        """Match output dimension to target via truncation or zero-padding."""
        if x.shape[1] == target_dim:
            return x
        if x.shape[1] > target_dim:
            return x[:, :target_dim]
        padded = np.zeros((x.shape[0], target_dim), dtype=np.float32)
        padded[:, : x.shape[1]] = x
        return padded

    @staticmethod
    def _stabilize_residual_branch(
        branch: np.ndarray,
        residual: np.ndarray,
        max_ratio: float = 1.5,
    ) -> np.ndarray:
        if branch.shape != residual.shape:
            return branch
        b = np.asarray(branch, dtype=np.float32)
        r = np.asarray(residual, dtype=np.float32)
        b_rms = np.sqrt(np.mean(b * b, axis=1, keepdims=True) + 1e-8)
        r_rms = np.sqrt(np.mean(r * r, axis=1, keepdims=True) + 1e-8)
        scale = np.clip((max_ratio * r_rms) / b_rms, 0.0, 1.0)
        return b * scale

    def _get_rope_theta(self) -> float:
        meta = self.weights.loader.get_metadata()
        arch = self.profile.architecture
        for key in (f"{arch}.rope.freq_base", "rope.freq_base"):
            if key in meta:
                return float(meta[key])
        return 10000.0

    @staticmethod
    def _is_matrix_weight(weight: np.ndarray | quant_ops.QuantizedMatrix) -> bool:
        if isinstance(weight, quant_ops.QuantizedMatrix):
            return True
        return isinstance(weight, np.ndarray) and weight.ndim == 2

    @staticmethod
    def _parse_rope_sections(raw: object) -> tuple[int, int, int, int]:
        if raw is None:
            return (0, 0, 0, 0)
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if not isinstance(raw, (list, tuple)):
            return (0, 0, 0, 0)
        vals: list[int] = []
        for item in raw:
            if isinstance(item, np.ndarray):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                if len(item) == 0:
                    vals.append(0)
                else:
                    vals.append(int(item[0]))
            else:
                vals.append(int(item))
        while len(vals) < 4:
            vals.append(0)
        return tuple(max(0, int(v)) for v in vals[:4])

    def _apply_qwen_mrope_inplace(
        self,
        q_heads: np.ndarray,
        k_heads: np.ndarray,
        rope_dim: int,
        pos: int,
    ) -> None:
        half = rope_dim // 2
        if half <= 0:
            return

        sec_t, sec_h, sec_w, sec_e = self.qwen_rope_sections
        sect_dims = sec_t + sec_h + sec_w + sec_e
        if sect_dims <= 0:
            simd_ops.rope(
                q_heads[:, :rope_dim],
                k_heads[:, :rope_dim],
                q_heads.shape[0],
                k_heads.shape[0],
                rope_dim,
                pos,
                self.rope_theta,
            )
            return

        # Llama.cpp text MRoPE expands scalar position into [t, h, w, e] = [p, p, p, 0].
        p_t = float(pos)
        p_h = float(pos)
        p_w = float(pos)
        p_e = 0.0
        sec_w_off = sec_t + sec_h
        sec_e_off = sec_w_off + sec_w

        positions = np.zeros((half,), dtype=np.float32)
        for i in range(half):
            sector = i % sect_dims
            if self.qwen_is_mrope:
                if (sector % 3 == 1) and (sector < 3 * sec_h):
                    positions[i] = p_h
                elif (sector % 3 == 2) and (sector < 3 * sec_w):
                    positions[i] = p_w
                elif (sector % 3 == 0) and (sector < 3 * sec_t):
                    positions[i] = p_t
                else:
                    positions[i] = p_e
            else:
                if sec_t <= sector < sec_w_off:
                    positions[i] = p_h
                elif sec_w_off <= sector < sec_e_off:
                    positions[i] = p_w
                elif sector >= sec_e_off:
                    positions[i] = p_e
                else:
                    positions[i] = p_t

        inv_freq = np.power(
            float(self.rope_theta),
            -2.0 * np.arange(half, dtype=np.float32) / float(rope_dim),
        )
        angles = positions * inv_freq
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)

        def _rotate_inplace(x: np.ndarray) -> None:
            x1 = x[:, :half].copy()
            x2 = x[:, half:rope_dim].copy()
            x[:, :half] = x1 * cos - x2 * sin
            x[:, half:rope_dim] = x1 * sin + x2 * cos

        _rotate_inplace(q_heads[:, :rope_dim])
        _rotate_inplace(k_heads[:, :rope_dim])
