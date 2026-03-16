from __future__ import annotations

from pathlib import Path

from saguaro.storage.native_vector_store import (
    NativeMemoryMappedVectorStore,
    native_vector_store_perf_counters,
)


def test_native_vector_store_perf_counters_move_after_query(tmp_path: Path) -> None:
    before = native_vector_store_perf_counters()

    store = NativeMemoryMappedVectorStore(str(tmp_path / "vectors"), dim=4)
    store.add([1.0, 0.0, 0.0, 0.0], {"file": "pkg/a.py", "name": "alpha"})
    store.save()
    store.query([1.0, 0.0, 0.0, 0.0], k=1, query_text="alpha")

    after = native_vector_store_perf_counters()
    instance = store.perf_counters()

    assert after["available"] == 1
    assert after["query_calls"] >= before.get("query_calls", 0)
    assert after["remap_count"] >= before.get("remap_count", 0)
    assert instance["native_perf_available"] == 1
    assert "open_handles" in instance
    assert "query_calls" in instance
