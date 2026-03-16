from __future__ import annotations

from saguaro import health as health_module
from saguaro.indexing import native_indexer_bindings
from saguaro.storage import native_vector_store


class _DummyIndexer:
    def capability_report(self) -> dict:
        return {
            "native_indexer": {
                "ok": True,
                "max_threads": 12,
                "isa_baseline": "avx2",
                "avx2_enabled": True,
                "fma_enabled": True,
            },
            "parallel_runtime": {
                "compiled": True,
                "default_threads": 6,
                "max_threads": 12,
                "affinity_mode": "sched_affinity",
                "affinity_cpus": [0, 1, 2, 3, 4, 5],
            },
            "trie_ops": {"available": True},
            "ops": {},
            "manifest": {"generated": True},
        }


def test_collect_native_compute_report_uses_native_thread_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        native_indexer_bindings,
        "get_native_indexer",
        lambda: _DummyIndexer(),
    )
    monkeypatch.setattr(
        native_vector_store,
        "native_vector_store_perf_counters",
        lambda: {"available": 1},
    )

    report = health_module.collect_native_compute_report()

    assert report["parallel_runtime"]["default_threads"] == 6
    assert report["parallel_runtime"]["max_threads"] == 12
    assert report["parallel_runtime"]["affinity_mode"] == "sched_affinity"
