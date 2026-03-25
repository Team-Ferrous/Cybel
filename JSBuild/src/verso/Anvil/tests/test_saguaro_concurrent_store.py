from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from saguaro.storage.native_vector_store import NativeMemoryMappedVectorStore


def test_native_store_handles_concurrent_writers(tmp_path) -> None:
    store = NativeMemoryMappedVectorStore(str(tmp_path / "vectors"), dim=32)

    def _write(idx: int) -> None:
        store.add(
            [float((idx % 7) - 3)] * 32,
            meta={
                "entity_id": f"entity:{idx}",
                "name": f"fn_{idx}",
                "type": "function",
                "file": f"/tmp/f_{idx % 5}.py",
                "line": idx + 1,
                "end_line": idx + 1,
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        for idx in range(200):
            pool.submit(_write, idx)

    store.save()
    assert len(store) == 200
