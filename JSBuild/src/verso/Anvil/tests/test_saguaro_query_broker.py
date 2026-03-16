from __future__ import annotations

from core.parallel_executor import SaguaroQueryBroker


class _StubSubstrate:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def batch_query(self, queries: list[str], *, k: int = 5, **_kwargs):
        self.calls.append((list(queries), int(k)))
        return {
            query: [{"file": f"{query.replace(' ', '_')}.py"}]
            for query in queries
        }


def test_query_broker_microbatches_concurrent_queries() -> None:
    substrate = _StubSubstrate()
    broker = SaguaroQueryBroker(
        substrate,
        batch_window_ms=20,
        max_batch_size=8,
    )

    future_a = broker.submit("alpha query", 2)
    future_b = broker.submit("beta query", 3)
    future_c = broker.submit("gamma query", 1)

    assert future_a.result(timeout=1.0) == ["alpha_query.py"]
    assert future_b.result(timeout=1.0) == ["beta_query.py"]
    assert future_c.result(timeout=1.0) == ["gamma_query.py"]

    assert len(substrate.calls) == 1
    assert substrate.calls[0][0] == ["alpha query", "beta query", "gamma query"]
    assert substrate.calls[0][1] == 3
    stats = broker.snapshot_stats()
    assert stats["batches"] == 1
    assert stats["queries_routed"] == 3
    assert stats["batch_size_max"] == 3
