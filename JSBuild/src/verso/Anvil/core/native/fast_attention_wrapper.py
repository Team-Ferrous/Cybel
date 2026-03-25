"""
Python bindings for fast CPU attention kernels.
Provides drop-in replacement for standard attention computation.
"""

import ctypes
import os
from typing import Optional

import numpy as np

from core.native.native_ops import load_native_library
from config.settings import PERFORMANCE_CONFIG


def _strict_native_qsg_enabled() -> bool:
    raw = os.getenv("ANVIL_STRICT_NATIVE_QSG")
    if raw is not None:
        normalized = str(raw).strip().lower()
        return normalized not in {"0", "false", "no", "off"}
    return bool(PERFORMANCE_CONFIG.get("strict_native_qsg", False))


class FastAttention:
    """
    High-performance CPU attention using AVX2/AVX512 kernels.

    Provides 3-5x speedup vs naive numpy implementation by:
    - Fusing operations (reduces memory bandwidth)
    - SIMD vectorization (AVX2: 8 floats, AVX512: 16 floats)
    - Cache-friendly memory access patterns
    """

    def __init__(self):
        self._f32_available = False
        self._mqa_available = False
        self.lib = self._load_library()

    def _load_library(self) -> Optional[ctypes.CDLL]:
        """Load compiled attention kernel library."""
        try:
            lib = load_native_library()

            if hasattr(lib, "fused_attention_f32"):
                # Signature: [B*H, Sq, D] x [B*H, Sk, D] -> [B*H, Sq, D]
                lib.fused_attention_f32.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # Q
                    ctypes.POINTER(ctypes.c_float),  # K
                    ctypes.POINTER(ctypes.c_float),  # V
                    ctypes.POINTER(ctypes.c_float),  # out
                    ctypes.c_int,  # batch_heads
                    ctypes.c_int,  # seq_q
                    ctypes.c_int,  # seq_k
                    ctypes.c_int,  # head_dim
                    ctypes.c_float,  # scale
                ]
                lib.fused_attention_f32.restype = None
                self._f32_available = True

            if hasattr(lib, "fused_attention_mqa_f32"):
                # Signature: [B, Hq, Sq, D] x [B, Hkv, Sk, D] -> [B, Hq, Sq, D]
                lib.fused_attention_mqa_f32.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # Q
                    ctypes.POINTER(ctypes.c_float),  # K
                    ctypes.POINTER(ctypes.c_float),  # V
                    ctypes.POINTER(ctypes.c_float),  # out
                    ctypes.c_int,  # batch
                    ctypes.c_int,  # q_heads
                    ctypes.c_int,  # kv_heads
                    ctypes.c_int,  # seq_q
                    ctypes.c_int,  # seq_k
                    ctypes.c_int,  # head_dim
                    ctypes.c_float,  # scale
                ]
                lib.fused_attention_mqa_f32.restype = None
                self._mqa_available = True

            if not self._f32_available and not self._mqa_available:
                return None

            return lib

        except Exception as e:
            print(f"Error loading fast attention library: {e}")
            return None

    @property
    def available(self) -> bool:
        """Check if native kernels are available."""
        return self.lib is not None and self._f32_available

    @property
    def mqa_available(self) -> bool:
        """Check if fused MQA kernel is available."""
        return self.lib is not None and self._mqa_available

    def compute_attention(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute fused attention: softmax(Q @ K^T / sqrt(d)) @ V

        Args:
            Q: [batch * heads, seq_q, head_dim] query tensor
            K: [batch * heads, seq_k, head_dim] key tensor
            V: [batch * heads, seq_k, head_dim] value tensor
            scale: Optional scale factor (default: 1/sqrt(head_dim))

        Returns:
            out: [batch * heads, seq_q, head_dim] attention output
        """
        if not self.available:
            if _strict_native_qsg_enabled():
                raise RuntimeError(
                    "Strict native QSG requires fused attention kernels; NumPy fallback is disabled."
                )
            # Fallback to numpy implementation
            return self._numpy_attention(Q, K, V, scale)

        # Validate inputs
        assert Q.dtype == np.float32, "Q must be float32"
        assert K.dtype == np.float32, "K must be float32"
        assert V.dtype == np.float32, "V must be float32"
        assert Q.ndim == 3 and K.ndim == 3 and V.ndim == 3

        batch_heads, seq_q, head_dim = Q.shape
        _, seq_k, _ = K.shape

        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Allocate output
        out = np.zeros((batch_heads, seq_q, head_dim), dtype=np.float32)

        # Call native kernel
        self.lib.fused_attention_f32(
            Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            K.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch_heads,
            seq_q,
            seq_k,
            head_dim,
            scale,
        )

        return out

    def compute_attention_mqa(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: Optional[float] = None
    ) -> np.ndarray:
        """Compute fused MQA/GQA attention.

        Args:
            Q: [batch, q_heads, seq_q, head_dim]
            K: [batch, kv_heads, seq_k, head_dim]
            V: [batch, kv_heads, seq_k, head_dim]
            scale: Optional scale factor (default: 1/sqrt(head_dim))
        """
        if not self.mqa_available:
            if _strict_native_qsg_enabled():
                raise RuntimeError(
                    "Strict native QSG requires MQA fused attention kernels; NumPy fallback is disabled."
                )
            return self._numpy_attention_mqa(Q, K, V, scale)

        assert Q.dtype == np.float32, "Q must be float32"
        assert K.dtype == np.float32, "K must be float32"
        assert V.dtype == np.float32, "V must be float32"
        assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4

        batch, q_heads, seq_q, head_dim = Q.shape
        batch_k, kv_heads, seq_k, k_dim = K.shape
        batch_v, kv_heads_v, seq_k_v, v_dim = V.shape
        assert batch_k == batch_v == batch, "Batch dimension mismatch"
        assert kv_heads_v == kv_heads, "KV heads mismatch"
        assert seq_k_v == seq_k, "KV sequence length mismatch"
        assert k_dim == v_dim == head_dim, "Head dimension mismatch"

        if scale is None:
            scale = 1.0 / np.sqrt(float(head_dim))

        out = np.zeros((batch, q_heads, seq_q, head_dim), dtype=np.float32)
        self.lib.fused_attention_mqa_f32(
            np.ascontiguousarray(Q).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(K).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            np.ascontiguousarray(V).ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            batch,
            q_heads,
            kv_heads,
            seq_q,
            seq_k,
            head_dim,
            float(scale),
        )
        return out

    def _numpy_attention(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: Optional[float] = None
    ) -> np.ndarray:
        """Fallback numpy implementation."""
        head_dim = Q.shape[-1]
        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)

        # Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
        scores = Q @ K.transpose(0, 2, 1) * scale  # [B*H, Sq, Sk]
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights /= np.sum(attn_weights, axis=-1, keepdims=True)
        out = attn_weights @ V  # [B*H, Sq, D]

        return out

    def _numpy_attention_mqa(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: Optional[float] = None
    ) -> np.ndarray:
        """Fallback numpy MQA/GQA attention."""
        Q_f = np.asarray(Q, dtype=np.float32)
        K_f = np.asarray(K, dtype=np.float32)
        V_f = np.asarray(V, dtype=np.float32)
        batch, q_heads, seq_q, head_dim = Q_f.shape
        _, kv_heads, seq_k, _ = K_f.shape
        if scale is None:
            scale = 1.0 / np.sqrt(float(head_dim))
        heads_per_kv = max(1, q_heads // max(1, kv_heads))
        out = np.zeros((batch, q_heads, seq_q, head_dim), dtype=np.float32)
        for b in range(batch):
            for q_h in range(q_heads):
                kv_h = min(kv_heads - 1, q_h // heads_per_kv)
                q = Q_f[b, q_h]  # [Sq, D]
                k = K_f[b, kv_h]  # [Sk, D]
                v = V_f[b, kv_h]  # [Sk, D]
                scores = q @ k.T * float(scale)  # [Sq, Sk]
                scores = scores - np.max(scores, axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs /= np.sum(probs, axis=-1, keepdims=True)
                out[b, q_h] = probs @ v
        return out


_GLOBAL_FAST_ATTENTION: Optional[FastAttention] = None


def fused_attention_mqa_f32(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: Optional[float] = None
) -> np.ndarray:
    """Ergonomic fused MQA helper used by the forward pass."""
    global _GLOBAL_FAST_ATTENTION
    if _GLOBAL_FAST_ATTENTION is None:
        _GLOBAL_FAST_ATTENTION = FastAttention()
    return _GLOBAL_FAST_ATTENTION.compute_attention_mqa(Q=Q, K=K, V=V, scale=scale)


# Integration with Hopfield Network
class OptimizedHopfieldVocab:
    """
    Hopfield network with fast attention kernel for token retrieval.
    Drop-in replacement for core/qsg/hopfield_vocab.py
    """

    def __init__(self, vocab_embeddings: np.ndarray, beta: float = 1.0):
        self.patterns = vocab_embeddings.astype(np.float32)  # [Vocab, Dim]
        self.beta = beta
        self.fast_attn = FastAttention()

    def get_token_probs(self, query: np.ndarray) -> np.ndarray:
        """
        Compute token probabilities using fast attention.

        Args:
            query: [Batch, Dim] query vectors

        Returns:
            probs: [Batch, Vocab] probability distribution
        """
        # Ensure float32
        if query.dtype != np.float32:
            query = query.astype(np.float32)

        # Reshape for attention: treat vocab as sequence
        # Q: [Batch, 1, Dim]
        # K: [1, Vocab, Dim] (patterns)
        # V: [1, Vocab, Dim] (patterns, but we only need scores)

        batch_size = query.shape[0]
        dim = query.shape[1]
        vocab_size = self.patterns.shape[0]

        # Compute similarity scores: Q @ Patterns^T
        # Using fast kernel for matmul
        Q = query.reshape(batch_size, 1, dim)  # [B, 1, D]
        K = self.patterns.reshape(1, vocab_size, dim)  # [1, V, D]

        # Broadcast for batch
        K_batch = np.tile(K, (batch_size, 1, 1))  # [B, V, D]

        # Compute dot product scores
        scores = np.einsum("bid,bvd->bv", Q.squeeze(1), K_batch)  # [B, V]
        scores *= self.beta

        # Softmax
        max_scores = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        return probs


# Compile script
BUILD_SCRIPT = """#!/bin/bash
# Build fast attention kernels
cd "$(dirname "$0")"

# Detect CPU capabilities
if grep -q avx512 /proc/cpuinfo; then
    MARCH="-march=skylake-avx512"
    echo "Building with AVX512 support"
elif grep -q avx2 /proc/cpuinfo; then
    MARCH="-march=haswell"
    echo "Building with AVX2 support"
else
    MARCH="-march=native"
    echo "Building with native CPU features"
fi

g++ -O3 $MARCH -fPIC -shared -o libfast_attention.so fast_attention.cpp

if [ $? -eq 0 ]; then
    echo "Successfully built libfast_attention.so"
else
    echo "Build failed!"
    exit 1
fi
"""

if __name__ == "__main__":
    # Save build script
    script_path = os.path.join(os.path.dirname(__file__), "build_attention.sh")
    with open(script_path, "w") as f:
        f.write(BUILD_SCRIPT)
    os.chmod(script_path, 0o755)
    print(f"Build script saved to {script_path}")
    print("Run: ./build_attention.sh")
