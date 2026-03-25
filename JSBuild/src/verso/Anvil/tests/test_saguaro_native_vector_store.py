from __future__ import annotations

from array import array
from pathlib import Path

import pytest

import saguaro.storage.native_vector_store as native_vector_store_module
from saguaro.storage.native_vector_store import (
    NativeMemoryMappedVectorStore,
    native_vector_store_available,
)
from saguaro.storage.vector_store import VectorStore


def test_native_vector_store_query_and_remove(tmp_path: Path) -> None:
    assert native_vector_store_available() is True

    store = NativeMemoryMappedVectorStore(str(tmp_path / "vectors"), dim=4)
    store.add([1.0, 0.0, 0.0, 0.0], {"file": "pkg/a.py", "name": "alpha"})
    store.add([0.0, 1.0, 0.0, 0.0], {"file": "pkg/b.py", "name": "beta"})
    store.save()

    results = store.query([1.0, 0.0, 0.0, 0.0], k=1, query_text="alpha")
    assert results[0]["name"] == "alpha"

    removed = store.remove_file("pkg/a.py")
    store.save()
    assert removed == 1

    reloaded = NativeMemoryMappedVectorStore(str(tmp_path / "vectors"), dim=4)
    post = reloaded.query([1.0, 0.0, 0.0, 0.0], k=1)
    assert post[0]["name"] == "beta"


def test_native_vector_store_bulk_remove_files(tmp_path: Path) -> None:
    store = NativeMemoryMappedVectorStore(str(tmp_path / "vectors"), dim=4)
    store.add([1.0, 0.0, 0.0, 0.0], {"file": "pkg/a.py", "name": "alpha"})
    store.add([0.0, 1.0, 0.0, 0.0], {"file": "pkg/b.py", "name": "beta"})
    store.add([0.0, 0.0, 1.0, 0.0], {"file": "pkg/c.py", "name": "gamma"})

    removed = store.remove_files(["pkg/a.py", "pkg/c.py"])
    assert removed == 2
    assert len(store) == 1

    store.save()
    reloaded = NativeMemoryMappedVectorStore(str(tmp_path / "vectors"), dim=4)
    results = reloaded.query([0.0, 1.0, 0.0, 0.0], k=1, query_text="beta")
    assert results[0]["name"] == "beta"


def test_vector_store_factory_prefers_native_store(tmp_path: Path) -> None:
    store = VectorStore(str(tmp_path / "vectors"), dim=4)
    assert isinstance(store, NativeMemoryMappedVectorStore)


def test_vector_store_factory_rejects_legacy_store(tmp_path: Path) -> None:
    storage = tmp_path / "vectors"
    storage.mkdir()
    (storage / "index.pkl").write_bytes(b"legacy")

    with pytest.raises(RuntimeError, match="Legacy pickle vector stores"):
        VectorStore(str(storage), dim=4)


def test_native_vector_store_requires_prebuilt_library(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        native_vector_store_module,
        "_native_lib_path",
        lambda: tmp_path / "missing" / "libanvil_saguaro_vector_store.so",
    )
    native_vector_store_module._LIB = None

    with pytest.raises(RuntimeError, match="Prebuilt native vector-store library"):
        native_vector_store_module._load_native_library()


def test_native_vector_store_add_batch_uses_bulk_write_path(tmp_path: Path) -> None:
    store = NativeMemoryMappedVectorStore(str(tmp_path / "vectors"), dim=4)
    store.add = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("row add path"))

    vectors = [
        array("f", [1.0, 0.0, 0.0, 0.0]),
        array("f", [0.0, 1.0, 0.0, 0.0]),
    ]
    metas = [
        {"file": "pkg/a.py", "name": "alpha", "entity_id": "a"},
        {"file": "pkg/b.py", "name": "beta", "entity_id": "b"},
    ]

    added = store.add_batch(vectors, metas)

    assert added == 2
    assert len(store) == 2
    store.save()
    results = store.query([1.0, 0.0, 0.0, 0.0], k=1, query_text="alpha")
    assert results[0]["name"] == "alpha"
