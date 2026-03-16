"""Embedding backends for Saguaro indexing/query."""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    from saguaro.indexing.native_indexer_bindings import (
        NativeIndexer,
        NativeIndexerError,
    )

    _HAS_NATIVE = True
except Exception:
    NativeIndexer = None  # type: ignore[assignment]
    NativeIndexerError = RuntimeError  # type: ignore[assignment]
    _HAS_NATIVE = False

_TF_MODULE = None
_HAS_TF: bool | None = None


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|\S")
_TOKEN_BUCKETS = 2**31 - 1


def _load_tensorflow() -> Any | None:
    global _TF_MODULE, _HAS_TF
    if _TF_MODULE is not None:
        return _TF_MODULE
    if _HAS_TF is False:
        return None
    try:
        import tensorflow as tf_module  # type: ignore

        _TF_MODULE = tf_module
        _HAS_TF = True
        return _TF_MODULE
    except Exception:
        _HAS_TF = False
        return None


def _tensorflow_available() -> bool:
    return _load_tensorflow() is not None


def _stable_token_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % _TOKEN_BUCKETS


class EmbeddingBackend(ABC):
    """Abstract backend for embedding/token operations."""

    @abstractmethod
    def tokenize(self, text: str, max_length: int = 512) -> np.ndarray:
        """Handle tokenize."""
        raise NotImplementedError

    @abstractmethod
    def projection(self, vocab_size: int, dim: int, seed: int = 42) -> np.ndarray:
        """Handle projection."""
        raise NotImplementedError

    def prepare_projection(self, projection: Any) -> Any:
        """Prepare a projection object for repeated hot-path use."""
        return projection

    @abstractmethod
    def embed(self, tokens: np.ndarray, projection: np.ndarray) -> np.ndarray:
        """Handle embed."""
        raise NotImplementedError

    @abstractmethod
    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        """Handle bundle."""
        raise NotImplementedError

    @abstractmethod
    def serialize(self, vector: np.ndarray) -> bytes:
        """Handle serialize."""
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, blob: bytes, dtype: Any = np.float32) -> np.ndarray:
        """Handle deserialize."""
        raise NotImplementedError


class NumPyBackend(EmbeddingBackend):
    """Legacy backend retained for compatibility tests only."""

    def tokenize(self, text: str, max_length: int = 512) -> np.ndarray:
        """Handle tokenize."""
        words = _TOKEN_RE.findall(text if text else "")
        token_ids = [
            _stable_token_id(word) for word in words[: max(1, int(max_length))]
        ]
        if not token_ids:
            token_ids = [0]
        return np.asarray(token_ids, dtype=np.int32)

    def projection(self, vocab_size: int, dim: int, seed: int = 42) -> np.ndarray:
        """Handle projection."""
        rng = np.random.default_rng(seed)
        init_range = 1.0 / np.sqrt(max(int(dim), 1))
        return rng.uniform(
            -init_range, init_range, (int(vocab_size), int(dim))
        ).astype(np.float32)

    def embed(self, tokens: np.ndarray, projection: np.ndarray) -> np.ndarray:
        """Handle embed."""
        proj = np.asarray(projection, dtype=np.float32)
        if proj.ndim != 2 or proj.shape[0] == 0:
            return np.zeros((1, 0), dtype=np.float32)
        toks = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if toks.size == 0:
            return np.zeros((1, proj.shape[1]), dtype=np.float32)
        idx = np.mod(toks, proj.shape[0]).astype(np.int64)
        return proj[idx]

    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        """Handle bundle."""
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim == 1:
            return vec
        if vec.size == 0:
            dim = int(vec.shape[1]) if vec.ndim == 2 else 0
            return np.zeros((dim,), dtype=np.float32)
        return vec.mean(axis=0, dtype=np.float32)

    def serialize(self, vector: np.ndarray) -> bytes:
        """Handle serialize."""
        return np.asarray(vector, dtype=np.float32).tobytes()

    def deserialize(self, blob: bytes, dtype: Any = np.float32) -> np.ndarray:
        """Handle deserialize."""
        if not blob:
            return np.zeros((0,), dtype=dtype)
        return np.frombuffer(blob, dtype=np.float32).astype(dtype, copy=False)


class NativeIndexerBackend(NumPyBackend):
    """Native C++ backed tokenization and bundling with CPU-first fallbacks."""

    def __init__(self) -> None:
        if not _HAS_NATIVE or NativeIndexer is None:
            raise RuntimeError("Native indexer backend is unavailable.")
        self._indexer = NativeIndexer()
        if not self._indexer.is_available():
            raise RuntimeError("Native indexer backend reported unavailable state.")

    def tokenize(self, text: str, max_length: int = 512) -> np.ndarray:
        tokens, lengths = self._indexer.tokenize_batch(
            [text if text else ""],
            max_length=max_length,
        )
        if tokens.size == 0:
            return np.asarray([0], dtype=np.int32)
        length = int(lengths[0]) if lengths.size else tokens.shape[1]
        length = max(1, min(length, tokens.shape[1]))
        return np.asarray(tokens[0, :length], dtype=np.int32)

    def embed(self, tokens: np.ndarray, projection: np.ndarray) -> np.ndarray:
        proj = np.asarray(projection, dtype=np.float32)
        if proj.ndim != 2 or proj.shape[0] == 0:
            return np.zeros((1, 0), dtype=np.float32)
        toks = np.asarray(tokens, dtype=np.int32).reshape(1, -1)
        if toks.size == 0:
            return np.zeros((1, proj.shape[1]), dtype=np.float32)
        return self._indexer.embed_lookup(toks, proj, int(proj.shape[0]))[0]

    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim == 1:
            return vec
        if vec.size == 0:
            dim = int(vec.shape[1]) if vec.ndim == 2 else 0
            return np.zeros((dim,), dtype=np.float32)
        return self._indexer.holographic_bundle(vec)


class TensorFlowBackend(EmbeddingBackend):
    """TensorFlow-backed embedding operations with NumPy outputs."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self._prepared_projection_cache: dict[
            tuple[int, tuple[int, ...], str], Any
        ] = {}

    def tokenize(self, text: str, max_length: int = 512) -> np.ndarray:
        """Handle tokenize."""
        tf = _load_tensorflow()
        assert tf is not None
        tensor_text = tf.convert_to_tensor(text if text else "", dtype=tf.string)
        text_spaced = tf.strings.regex_replace(tensor_text, r"([^A-Za-z0-9_])", r" \1 ")
        words = tf.strings.split(text_spaced)
        tokens = tf.strings.to_hash_bucket_fast(words, _TOKEN_BUCKETS)
        tokens = tf.cast(tokens, tf.int32)[: max(1, int(max_length))]
        arr = tokens.numpy()
        if arr.size == 0:
            return np.asarray([0], dtype=np.int32)
        return arr.astype(np.int32, copy=False)

    def projection(self, vocab_size: int, dim: int, seed: int = 42) -> np.ndarray:
        """Handle projection."""
        tf = _load_tensorflow()
        assert tf is not None
        tf.random.set_seed(seed)
        init_range = 1.0 / tf.math.sqrt(tf.cast(tf.math.maximum(dim, 1), tf.float32))
        tensor = tf.random.uniform(
            [int(vocab_size), int(dim)],
            -init_range,
            init_range,
            seed=seed,
            dtype=tf.float32,
        )
        return tensor.numpy()

    def prepare_projection(self, projection: Any) -> Any:
        """Handle prepare projection."""
        tf = _load_tensorflow()
        assert tf is not None
        if tf.is_tensor(projection):
            return projection

        array = np.asarray(projection, dtype=np.float32)
        key = (
            int(array.__array_interface__["data"][0]),
            tuple(int(v) for v in array.shape),
            str(array.dtype),
        )
        cached = self._prepared_projection_cache.get(key)
        if cached is None:
            cached = tf.convert_to_tensor(array, dtype=tf.float32)
            self._prepared_projection_cache[key] = cached
        return cached

    def embed(self, tokens: np.ndarray, projection: np.ndarray) -> np.ndarray:
        """Handle embed."""
        tf = _load_tensorflow()
        assert tf is not None
        proj = self.prepare_projection(projection)
        toks = tf.convert_to_tensor(np.asarray(tokens, dtype=np.int32), dtype=tf.int32)
        vocab_size = tf.shape(proj)[0]
        clipped = tf.math.floormod(toks, vocab_size)
        return tf.nn.embedding_lookup(proj, clipped)

    def bundle(self, vectors: np.ndarray) -> np.ndarray:
        """Handle bundle."""
        tf = _load_tensorflow()
        assert tf is not None
        vec = (
            vectors
            if tf.is_tensor(vectors)
            else tf.convert_to_tensor(
                np.asarray(vectors, dtype=np.float32), dtype=tf.float32
            )
        )
        if len(vec.shape) == 1:
            return vec.numpy()
        return tf.reduce_mean(vec, axis=0).numpy()

    def serialize(self, vector: np.ndarray) -> bytes:
        """Handle serialize."""
        tf = _load_tensorflow()
        assert tf is not None
        vec = tf.convert_to_tensor(
            np.asarray(vector, dtype=np.float32), dtype=tf.float32
        )
        return tf.io.serialize_tensor(vec).numpy()

    def deserialize(self, blob: bytes, dtype: Any = np.float32) -> np.ndarray:
        """Handle deserialize."""
        tf = _load_tensorflow()
        assert tf is not None
        if not blob:
            return np.zeros((0,), dtype=dtype)
        tensor = tf.io.parse_tensor(blob, out_type=tf.float32)
        return tf.cast(tensor, tf.as_dtype(dtype)).numpy()


def get_backend(prefer_tensorflow: bool = True) -> EmbeddingBackend:
    """Return an embedding backend instance.

    Args:
        prefer_tensorflow: Use TensorFlow when available if True.
    """
    forced = str(os.getenv("SAGUARO_BACKEND", "auto") or "auto").strip().lower()
    prefer_native = str(
        os.getenv("SAGUARO_PREFER_NATIVE_BACKEND", "1") or "1"
    ).strip().lower() in {"1", "true", "yes", "on"}

    if forced == "native":
        if not _HAS_NATIVE:
            raise RuntimeError("Native backend requested but native indexer is unavailable.")
        return NativeIndexerBackend()
    if forced == "tensorflow":
        if _HAS_TF:
            return TensorFlowBackend()
        raise RuntimeError(
            "TensorFlow backend requested but TensorFlow is unavailable."
        )
    if forced == "numpy":
        return NumPyBackend()

    if prefer_native and _HAS_NATIVE:
        try:
            return NativeIndexerBackend()
        except NativeIndexerError:
            pass
        except RuntimeError:
            pass

    if _tensorflow_available() and prefer_tensorflow:
        return TensorFlowBackend()
    if _HAS_NATIVE:
        try:
            return NativeIndexerBackend()
        except NativeIndexerError:
            pass
        except RuntimeError:
            pass
    return NumPyBackend()


def backend_name(backend: EmbeddingBackend | None) -> str:
    """Handle backend name."""
    if backend is None:
        return "unknown"
    return backend.__class__.__name__
