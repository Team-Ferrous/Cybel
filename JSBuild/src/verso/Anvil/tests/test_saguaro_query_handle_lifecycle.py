from __future__ import annotations

from pathlib import Path

from saguaro.storage.native_vector_store import NativeMemoryMappedVectorStore


def test_native_query_handle_is_reused_until_store_close(tmp_path: Path) -> None:
    store = NativeMemoryMappedVectorStore(
        str(tmp_path / "vectors"),
        dim=4,
        active_dim=4,
        total_dim=8,
    )
    store.add([1.0, 0.0, 0.0, 0.0], {"file": "pkg/a.py", "name": "alpha"})
    store.add([0.0, 1.0, 0.0, 0.0], {"file": "pkg/b.py", "name": "beta"})
    store.save()

    first = store.query([1.0, 0.0, 0.0, 0.0], k=1, query_text="alpha")
    second = store.query([0.0, 1.0, 0.0, 0.0], k=1, query_text="beta")

    counters = store.perf_counters()
    assert first[0]["name"] == "alpha"
    assert second[0]["name"] == "beta"
    assert counters["query_count"] == 2
    assert counters["query_handle_opens"] == 1
    assert counters["query_handle_closes"] == 0

    store.close()
    closed_counters = store.perf_counters()
    assert closed_counters["query_handle_closes"] == 1
