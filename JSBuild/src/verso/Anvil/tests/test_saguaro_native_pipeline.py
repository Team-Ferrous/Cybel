from __future__ import annotations

from array import array
from types import SimpleNamespace

from saguaro.indexing import native_worker


def test_native_worker_enforces_batch_quota_and_emits_segment_identity(monkeypatch, tmp_path) -> None:
    source = tmp_path / "quota_demo.py"
    source.write_text("def alpha():\n    return 1\n", encoding="utf-8")

    class _FakeParser:
        def parse_file(self, _path: str):
            return [
                SimpleNamespace(
                    name="alpha",
                    type="function",
                    content="def alpha(): return 1",
                    start_line=1,
                    end_line=1,
                    file_path=str(source),
                ),
                SimpleNamespace(
                    name="beta",
                    type="function",
                    content="def beta(): return 2",
                    start_line=2,
                    end_line=2,
                    file_path=str(source),
                ),
            ]

    class _FakeRuntime:
        def full_pipeline(self, *, texts, projection_buffer, vocab_size, dim, max_length, trie, num_threads):
            _ = (projection_buffer, vocab_size, max_length, trie, num_threads)
            return [array("f", [float(i + 1) for i in range(dim)]) for _text in texts]

    monkeypatch.setattr(native_worker, "_parser", _FakeParser())
    monkeypatch.setattr(native_worker, "_native_indexer", _FakeRuntime())
    monkeypatch.setattr(native_worker, "_worker_shm", SimpleNamespace(buf=bytearray(4096)))
    monkeypatch.setenv("SAGUARO_INDEX_SPILL_RESULTS", "0")

    meta, vectors, touched, metrics = native_worker.process_batch_worker_native(
        [str(source)],
        active_dim=8,
        total_dim=16,
        vocab_size=64,
        repo_path=str(tmp_path),
        batch_capacity=1,
        max_total_texts=10_000,
    )

    assert len(meta) == 2
    assert len(vectors or []) == 2
    assert len(vectors[0]) == 8
    assert touched == [str(source)]
    assert meta[0]["entity_id"]
    assert meta[0]["segment_id"]
    assert metrics["quota_hits"] >= 1
    assert metrics["entities_dropped_by_quota"] == 0
