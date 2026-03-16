from __future__ import annotations

import json
from array import array
from pathlib import Path

from saguaro.storage.native_vector_store import NativeMemoryMappedVectorStore


def test_native_vector_store_migrates_legacy_full_width_layout(tmp_path: Path) -> None:
    storage = tmp_path / "vectors"
    storage.mkdir()

    rows = array(
        "f",
        [
            1.0,
            0.0,
            0.0,
            0.0,
            9.0,
            9.0,
            9.0,
            9.0,
            0.0,
            1.0,
            0.0,
            0.0,
            8.0,
            8.0,
            8.0,
            8.0,
        ],
    )
    (storage / "vectors.bin").write_bytes(memoryview(rows).cast("B"))
    (storage / "metadata.json").write_text(
        json.dumps(
            [
                {"file": "pkg/a.py", "name": "alpha", "entity_id": "alpha"},
                {"file": "pkg/b.py", "name": "beta", "entity_id": "beta"},
            ]
        ),
        encoding="utf-8",
    )
    (storage / "index_meta.json").write_text(
        json.dumps(
            {
                "dim": 8,
                "active_dim": 4,
                "total_dim": 8,
                "count": 2,
                "capacity": 2,
                "version": 3,
                "format": "native_mmap",
            }
        ),
        encoding="utf-8",
    )

    store = NativeMemoryMappedVectorStore(
        str(storage),
        dim=8,
        active_dim=4,
        total_dim=8,
    )
    assert store._format_version == 3  # noqa: SLF001

    store.save()
    reloaded = NativeMemoryMappedVectorStore(
        str(storage),
        dim=8,
        active_dim=4,
        total_dim=8,
    )

    assert reloaded._format_version == 4  # noqa: SLF001
    assert reloaded._storage_dim == 4  # noqa: SLF001
    assert (storage / "norms.bin").exists()
    assert (storage / "vectors.bin").stat().st_size == 2 * 4 * 4

    results = reloaded.query([1.0, 0.0, 0.0, 0.0], k=1, query_text="alpha")
    assert results[0]["name"] == "alpha"
