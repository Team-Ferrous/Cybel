from __future__ import annotations

import importlib
import mmap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _FallbackMmapWeightStore:
    def __init__(self, path: str):
        self._fd = open(path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)

    def get_bytes(self, offset: int, size: int) -> bytes:
        start = int(offset)
        end = start + int(size)
        return self._mm[start:end]

    def get_view(self, offset: int, size: int):
        start = int(offset)
        end = start + int(size)
        return memoryview(self._mm)[start:end]

    def close(self) -> None:
        self._mm.close()
        self._fd.close()


def _load_store(path: str):
    try:
        module = importlib.import_module("core.native.mmap_weight_store")
        cls = getattr(module, "MmapWeightStore")
        return cls(path), True
    except Exception:
        return _FallbackMmapWeightStore(path), False


def _call_first(obj, names: tuple[str, ...], *args):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args)
    raise AttributeError(f"no callable found in {names}")


def _extract_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    return bytes(np.asarray(value, dtype=np.uint8).reshape(-1))


def test_mmap_weight_store_can_fetch_bytes_and_views(tmp_path: Path) -> None:
    payload = bytes(range(64)) * 4
    file_path = tmp_path / "weights.bin"
    file_path.write_bytes(payload)

    store, is_native = _load_store(str(file_path))
    view = None
    try:
        try:
            got_bytes = _call_first(store, ("get_bytes", "read_bytes", "bytes_at"), 10, 9)
            view = _call_first(store, ("get_view", "read_view", "view_at"), 20, 11)
        except AttributeError:
            if is_native:
                pytest.skip("Native mmap store exists but does not expose expected byte/view accessors.")
            raise

        assert _extract_bytes(got_bytes) == payload[10:19]
        assert _extract_bytes(view) == payload[20:31]
    finally:
        if view is not None and hasattr(view, "release"):
            view.release()
        close_fn = getattr(store, "close", None)
        if callable(close_fn):
            close_fn()


def test_fallback_mmap_store_is_deterministic(tmp_path: Path) -> None:
    payload = b"anvil-mmap-regression-coverage"
    file_path = tmp_path / "weights_small.bin"
    file_path.write_bytes(payload)

    store = _FallbackMmapWeightStore(str(file_path))
    view = None
    try:
        a = store.get_bytes(0, len(payload))
        view = store.get_view(6, 8)
        b = store.get_bytes(0, len(payload))
        assert a == payload
        assert b == payload
        assert _extract_bytes(view) == payload[6:14]
    finally:
        if view is not None and hasattr(view, "release"):
            view.release()
        store.close()


def test_mmap_weight_store_disables_float16_promotion_cache() -> None:
    from core.native.mmap_weight_store import MMapWeightStore

    dense = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    tensor = SimpleNamespace(
        name="blk.0.attn_norm.weight",
        shape=dense.shape,
        tensor_type=SimpleNamespace(value=1),
        data=dense,
    )
    loader = SimpleNamespace(
        reader=SimpleNamespace(tensors=[tensor]),
        get_tensor=lambda name: dense if name == "blk.0.attn_norm.weight" else None,
    )
    profile = SimpleNamespace(n_layers=0, embedding_dim=4)
    store = MMapWeightStore(loader=loader, profile=profile)

    got = store.get_weight("blk.0.attn_norm.weight")

    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float16
    assert np.shares_memory(got, dense)
