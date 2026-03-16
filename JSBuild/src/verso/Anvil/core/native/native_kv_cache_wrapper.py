"""Optional ctypes-backed KV cache with CPU Flash Attention and automatic Python fallback."""

from __future__ import annotations

import ctypes
import importlib
import os
from typing import Optional

from core.model.model_profile import ModelProfile
from core.native.native_ops import load_native_library
from core.native.qsg_kv_cache import NativeKVCache as PythonKVCache
from config.settings import PERFORMANCE_CONFIG

np = importlib.import_module("numpy")


class _KVCacheMetricsSnapshot(ctypes.Structure):
    _fields_ = [
        ("active_page_slots", ctypes.c_int),
        ("resident_page_count", ctypes.c_int),
        ("shared_page_slots", ctypes.c_int),
        ("snapshot_count", ctypes.c_int),
        ("active_tokens", ctypes.c_int),
        ("committed_token_capacity", ctypes.c_int),
        ("copy_on_write_events", ctypes.c_int),
        ("prefix_share_events", ctypes.c_int),
        ("page_tokens", ctypes.c_int),
        ("fragmentation_ratio", ctypes.c_float),
    ]


def _strict_native_qsg_enabled() -> bool:
    raw = os.getenv("ANVIL_STRICT_NATIVE_QSG")
    if raw is not None:
        normalized = str(raw).strip().lower()
        return normalized not in {"0", "false", "no", "off"}
    return bool(PERFORMANCE_CONFIG.get("strict_native_qsg", False))


class NativeKVCacheWrapper:
    """Drop-in replacement for NativeKVCache using native contiguous buffers.

    Features beyond the Python fallback:
      - Pre-allocated contiguous C++ memory (zero allocation on append)
      - Batch append for prefill
      - Integrated CPU Flash Attention with tiled online softmax
    """

    def __init__(self, profile: ModelProfile, max_seq_len: int = 8192):
        self.profile = profile
        self.max_seq_len = int(max_seq_len)
        self._strict_native = _strict_native_qsg_enabled()
        self._fallback = PythonKVCache(profile=profile, max_seq_len=max_seq_len)
        self._fallback_layers: set[int] = set()
        self._layer_len: dict[int, int] = {}
        self._layer_width: dict[int, int] = {}
        self._snapshot_lengths: dict[int, dict[int, int]] = {}

        self._lib: Optional[ctypes.CDLL] = None
        self._handle: Optional[int] = None
        self._token_width = 0
        self._n_kv_heads = 0
        self._head_dim = 0
        self._has_flash_attention = False
        self._has_batch_append = False
        self._has_prefix_snapshots = False
        self._has_metrics = False
        self._load_native()

    def _load_native(self) -> None:
        n_heads = int(self.profile.n_heads) if int(self.profile.n_heads) > 0 else 1
        n_kv_heads = int(self.profile.n_kv_heads) if int(self.profile.n_kv_heads) > 0 else n_heads
        head_dim = int(self.profile.embedding_dim // max(1, n_heads))
        if head_dim <= 0:
            return
        self._n_kv_heads = n_kv_heads
        self._head_dim = head_dim
        self._token_width = n_kv_heads * head_dim
        if self._token_width <= 0:
            return

        try:
            lib = load_native_library()
            create = getattr(lib, "kv_cache_create")
            append = getattr(lib, "kv_cache_append")
            get_k = getattr(lib, "kv_cache_get_k")
            get_v = getattr(lib, "kv_cache_get_v")
            reset = getattr(lib, "kv_cache_reset")
            destroy = getattr(lib, "kv_cache_destroy")
        except Exception:
            return

        float_p = ctypes.POINTER(ctypes.c_float)
        create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        create.restype = ctypes.c_void_p
        append.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, float_p, float_p]
        append.restype = ctypes.c_int
        get_k.argtypes = [ctypes.c_void_p, ctypes.c_int]
        get_k.restype = float_p
        get_v.argtypes = [ctypes.c_void_p, ctypes.c_int]
        get_v.restype = float_p
        reset.argtypes = [ctypes.c_void_p]
        reset.restype = ctypes.c_int
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = None

        # Check for batch append
        if hasattr(lib, "kv_cache_append_batch"):
            lib.kv_cache_append_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                float_p, float_p, ctypes.c_int,
            ]
            lib.kv_cache_append_batch.restype = ctypes.c_int
            self._has_batch_append = True

        # Check for Flash Attention
        if hasattr(lib, "kv_cache_flash_attention"):
            lib.kv_cache_flash_attention.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                float_p, float_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ]
            lib.kv_cache_flash_attention.restype = ctypes.c_int
            self._has_flash_attention = True

        if hasattr(lib, "kv_cache_snapshot_prefix"):
            lib.kv_cache_snapshot_prefix.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.kv_cache_snapshot_prefix.restype = ctypes.c_int
            if hasattr(lib, "kv_cache_restore_prefix") and hasattr(lib, "kv_cache_release_snapshot"):
                lib.kv_cache_restore_prefix.argtypes = [ctypes.c_void_p, ctypes.c_int]
                lib.kv_cache_restore_prefix.restype = ctypes.c_int
                lib.kv_cache_release_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_int]
                lib.kv_cache_release_snapshot.restype = ctypes.c_int
                self._has_prefix_snapshots = True

        if hasattr(lib, "kv_cache_get_metrics"):
            lib.kv_cache_get_metrics.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_KVCacheMetricsSnapshot),
            ]
            lib.kv_cache_get_metrics.restype = ctypes.c_int
            self._has_metrics = True

        handle = create(self.max_seq_len, int(self.profile.n_layers), n_kv_heads, head_dim)
        if not handle:
            return
        self._lib = lib
        self._handle = int(handle)

    @property
    def available(self) -> bool:
        return self._lib is not None and self._handle is not None

    @property
    def has_flash_attention(self) -> bool:
        return self._has_flash_attention and self.available

    @property
    def has_prefix_snapshots(self) -> bool:
        return self._has_prefix_snapshots and self.available

    def _to_seq2d(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"KV cache append expects 1D/2D tensors, got shape={arr.shape}")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray, pos: int) -> None:
        layer = int(layer_idx)
        if layer in self._fallback_layers or not self.available:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV cache execution and "
                    "does not allow Python fallback append path."
                )
            self._fallback.append(layer_idx=layer, k=k, v=v, pos=pos)
            return

        k_f = self._to_seq2d(k)
        v_f = self._to_seq2d(v)
        if k_f.shape != v_f.shape:
            self._fallback_layers.add(layer)
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV cache append with matching tensor shapes."
                )
            self._fallback.append(layer_idx=layer, k=k_f, v=v_f, pos=pos)
            return

        width = int(k_f.shape[1])
        self._layer_width[layer] = width
        if width != self._token_width:
            self._fallback_layers.add(layer)
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV cache append width to match KV head layout."
                )
            self._fallback.append(layer_idx=layer, k=k_f, v=v_f, pos=pos)
            return

        seq_len = int(k_f.shape[0])
        next_len = int(pos) + seq_len
        if pos < 0 or next_len > self.max_seq_len:
            self._fallback_layers.add(layer)
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV cache append to stay within max_seq_len."
                )
            self._fallback.append(layer_idx=layer, k=k_f, v=v_f, pos=pos)
            return

        float_p = ctypes.POINTER(ctypes.c_float)

        # Use batch append if available and seq_len > 1
        if self._has_batch_append and seq_len > 1:
            ok = self._lib.kv_cache_append_batch(
                ctypes.c_void_p(self._handle),
                layer, int(pos),
                k_f.ctypes.data_as(float_p),
                v_f.ctypes.data_as(float_p),
                seq_len,
            )
            if int(ok) != 1:
                self._fallback_layers.add(layer)
                if self._strict_native:
                    raise RuntimeError(
                        "Strict native QSG requires native batched KV cache append to succeed."
                    )
                self._fallback.append(layer_idx=layer, k=k_f, v=v_f, pos=pos)
                return
        else:
            for t in range(seq_len):
                ok = self._lib.kv_cache_append(
                    ctypes.c_void_p(self._handle),
                    layer, int(pos + t),
                    k_f[t].ctypes.data_as(float_p),
                    v_f[t].ctypes.data_as(float_p),
                )
                if int(ok) != 1:
                    self._fallback_layers.add(layer)
                    if self._strict_native:
                        raise RuntimeError(
                            "Strict native QSG requires native KV cache append to succeed."
                        )
                    self._fallback.append(layer_idx=layer, k=k_f, v=v_f, pos=pos)
                    return

        self._layer_len[layer] = max(self._layer_len.get(layer, 0), next_len)

    def get(self, layer_idx: int) -> tuple[np.ndarray, np.ndarray]:
        layer = int(layer_idx)
        if layer in self._fallback_layers or not self.available:
            if self._strict_native:
                raise RuntimeError("Strict native QSG requires native KV cache reads only.")
            return self._fallback.get(layer)

        length = int(self._layer_len.get(layer, 0))
        width = int(self._layer_width.get(layer, self._token_width))
        if length <= 0 or width <= 0:
            empty = np.zeros((0, max(width, 1)), dtype=np.float32)
            return empty, empty

        k_ptr = self._lib.kv_cache_get_k(ctypes.c_void_p(self._handle), layer)
        v_ptr = self._lib.kv_cache_get_v(ctypes.c_void_p(self._handle), layer)
        if not k_ptr or not v_ptr:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV cache pointers for reads."
                )
            return self._fallback.get(layer)

        native_shape = (length, self._token_width)
        k_all = np.ctypeslib.as_array(k_ptr, shape=native_shape)
        v_all = np.ctypeslib.as_array(v_ptr, shape=native_shape)
        return (
            np.asarray(k_all[:length, :width], dtype=np.float32).copy(),
            np.asarray(v_all[:length, :width], dtype=np.float32).copy(),
        )

    def flash_attention(
        self,
        layer_idx: int,
        q: np.ndarray,
        n_heads: int,
        kv_len: int,
        scale: float,
    ) -> Optional[np.ndarray]:
        """CPU Flash Attention: tiled online-softmax for a single query token.

        Args:
            layer_idx: Index of the KV cache layer.
            q: Query tensor [n_heads, head_dim].
            n_heads: Number of query heads.
            kv_len: Current KV sequence length.
            scale: Attention scale factor (typically 1/sqrt(head_dim)).

        Returns:
            Attention output [n_heads, head_dim], or None if not available.
        """
        if not self.has_flash_attention:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native Flash Attention in KV cache path."
                )
            return None
        if int(layer_idx) in self._fallback_layers:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires Flash Attention using native layer state."
                )
            return None

        q_f = np.ascontiguousarray(q, dtype=np.float32).reshape(n_heads, self._head_dim)
        out = np.zeros((n_heads, self._head_dim), dtype=np.float32)

        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.kv_cache_flash_attention(
            ctypes.c_void_p(self._handle),
            int(layer_idx),
            q_f.ctypes.data_as(float_p),
            out.ctypes.data_as(float_p),
            n_heads, kv_len,
            ctypes.c_float(float(scale)),
        )
        return out if int(ok) == 1 else None

    def reset(self) -> None:
        self._layer_len.clear()
        self._layer_width.clear()
        self._snapshot_lengths.clear()
        self._fallback_layers.clear()
        self._fallback.reset()
        if self.available:
            self._lib.kv_cache_reset(ctypes.c_void_p(self._handle))

    def get_current_length(self) -> int:
        if self._layer_len:
            return max(self._layer_len.values())
        if self._strict_native:
            return 0
        return self._fallback.get_current_length()

    def snapshot_prefix(self, length_tokens: int | None = None) -> int:
        if not self.has_prefix_snapshots:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV prefix snapshot support."
                )
            raise RuntimeError("Native KV prefix snapshots are unavailable.")
        length = self.get_current_length() if length_tokens is None else int(length_tokens)
        snapshot_id = int(
            self._lib.kv_cache_snapshot_prefix(
                ctypes.c_void_p(self._handle),
                int(max(0, length)),
            )
        )
        if snapshot_id <= 0:
            raise RuntimeError("Native KV prefix snapshot creation failed.")
        self._snapshot_lengths[snapshot_id] = dict(self._layer_len)
        return snapshot_id

    def restore_prefix(self, snapshot_id: int) -> None:
        if not self.has_prefix_snapshots:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV prefix restore support."
                )
            raise RuntimeError("Native KV prefix restore is unavailable.")
        ok = int(
            self._lib.kv_cache_restore_prefix(
                ctypes.c_void_p(self._handle),
                int(snapshot_id),
            )
        )
        if ok != 1:
            raise RuntimeError(f"Native KV prefix restore failed for snapshot {snapshot_id}.")
        self._layer_len = dict(self._snapshot_lengths.get(int(snapshot_id), {}))
        self._layer_width = {layer: self._token_width for layer in self._layer_len}

    def release_snapshot(self, snapshot_id: int) -> None:
        if not self.has_prefix_snapshots:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV snapshot release support."
                )
            raise RuntimeError("Native KV snapshot release is unavailable.")
        ok = int(
            self._lib.kv_cache_release_snapshot(
                ctypes.c_void_p(self._handle),
                int(snapshot_id),
            )
        )
        if ok != 1:
            raise RuntimeError(f"Native KV snapshot release failed for snapshot {snapshot_id}.")
        self._snapshot_lengths.pop(int(snapshot_id), None)

    def metrics_snapshot(self) -> dict[str, int | float]:
        if not self.available or not self._has_metrics:
            if self._strict_native:
                raise RuntimeError(
                    "Strict native QSG requires native KV metrics snapshots."
                )
            return {
                "active_page_slots": 0,
                "resident_page_count": 0,
                "shared_page_slots": 0,
                "snapshot_count": 0,
                "active_tokens": self.get_current_length(),
                "committed_token_capacity": 0,
                "copy_on_write_events": 0,
                "prefix_share_events": 0,
                "page_tokens": 0,
                "fragmentation_ratio": 0.0,
            }
        snapshot = _KVCacheMetricsSnapshot()
        ok = int(
            self._lib.kv_cache_get_metrics(
                ctypes.c_void_p(self._handle),
                ctypes.byref(snapshot),
            )
        )
        if ok != 1:
            raise RuntimeError("Native KV metrics snapshot failed.")
        return {
            "active_page_slots": int(snapshot.active_page_slots),
            "resident_page_count": int(snapshot.resident_page_count),
            "shared_page_slots": int(snapshot.shared_page_slots),
            "snapshot_count": int(snapshot.snapshot_count),
            "active_tokens": int(snapshot.active_tokens),
            "committed_token_capacity": int(snapshot.committed_token_capacity),
            "copy_on_write_events": int(snapshot.copy_on_write_events),
            "prefix_share_events": int(snapshot.prefix_share_events),
            "page_tokens": int(snapshot.page_tokens),
            "fragmentation_ratio": float(snapshot.fragmentation_ratio),
        }

    def close(self) -> None:
        if not self.available:
            return
        self._lib.kv_cache_destroy(ctypes.c_void_p(self._handle))
        self._handle = None
        self._snapshot_lengths.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
