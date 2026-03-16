"""Holographic memory operations with graceful fallback.

Uses native quantum ops when available. Falls back to TensorFlow/NumPy operations
when native custom ops are unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

from . import quantum_ops


def _to_numpy(value: Any) -> np.ndarray:
    if tf is not None and isinstance(value, tf.Tensor):
        return value.numpy()
    return np.asarray(value, dtype=np.float32)


def _to_tensor(value: np.ndarray):
    if tf is None:
        return value
    return tf.convert_to_tensor(value, dtype=tf.float32)


def holographic_bundle(vectors):
    try:
        return quantum_ops.holographic_bundle(vectors)
    except Exception:
        arr = _to_numpy(vectors)
        if arr.ndim == 1:
            out = arr.astype(np.float32)
        else:
            out = np.mean(arr, axis=0).astype(np.float32)
        return _to_tensor(out)


def modern_hopfield_retrieve(query, memory, beta: float = 1.0):
    try:
        return quantum_ops.modern_hopfield_retrieve(query, memory, beta=beta)
    except Exception:
        q = _to_numpy(query)
        m = _to_numpy(memory)
        logits = float(beta) * (q @ m.T)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits)
        weights = exp / (np.sum(exp, axis=-1, keepdims=True) + 1e-9)
        out = (weights @ m).astype(np.float32)
        return _to_tensor(out)


def crystallize_memory(knowledge, importance, threshold: float = 0.5):
    try:
        return quantum_ops.crystallize_memory(knowledge, importance, threshold)
    except Exception:
        k = _to_numpy(knowledge)
        i = _to_numpy(importance)
        out = np.where(i >= float(threshold), k, 0.0).astype(np.float32)
        return _to_tensor(out)


def serialize_bundle(tensor) -> bytes:
    if tf is not None and isinstance(tensor, tf.Tensor):
        return tf.io.serialize_tensor(tensor).numpy()
    return _to_numpy(tensor).astype(np.float32).tobytes()


def serialize_hd_state(tensor) -> bytes:
    return serialize_bundle(tensor)


def deserialize_bundle(blob: bytes):
    if tf is not None:
        try:
            return tf.io.parse_tensor(blob, out_type=tf.float32)
        except Exception:
            pass
    return np.frombuffer(blob, dtype=np.float32)


def deserialize_hd_state(data: bytes, dtype=None, *, allow_tensorflow_parse: bool = False):
    if allow_tensorflow_parse and tf is not None:
        out_type = dtype if dtype is not None else tf.float32
        try:
            return tf.io.parse_tensor(data, out_type=out_type)
        except Exception:
            pass

    np_dtype = np.float32 if dtype is None else dtype
    return np.frombuffer(data, dtype=np_dtype)
