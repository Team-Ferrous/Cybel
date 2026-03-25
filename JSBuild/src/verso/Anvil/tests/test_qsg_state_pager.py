import numpy as np

from core.qsg.state_pager import QSGStatePager


def _make_pager() -> QSGStatePager:
    return QSGStatePager(
        dim=4,
        state_page_rows=2,
        soft_compaction_threshold=0.18,
        hard_compaction_threshold=0.30,
    )


def test_alloc_gather_scatter_release_roundtrip():
    pager = _make_pager()
    refs = pager.alloc_rows("req-a", 2)

    values = np.asarray([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=np.float32)
    pager.scatter_updates(refs, values)
    gathered = pager.gather_active(refs)
    np.testing.assert_allclose(gathered, values)

    pager.release_request("req-a")
    assert pager.get_request_rows("req-a") == []
    assert pager.metrics_snapshot()["used_rows"] == 0


def test_clone_and_compact():
    pager = _make_pager()
    base_refs = pager.alloc_rows("req-a", 2)
    pager.scatter_updates(
        base_refs,
        np.asarray([[5.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    clone_refs = pager.clone_cow("req-a", target_request_id="req-b")
    gathered_clone = pager.gather_active(clone_refs)
    np.testing.assert_allclose(
        gathered_clone,
        np.asarray([[5.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )

    pager.release_request("req-a")
    fragmented_before = pager.fragmentation_ratio()
    assert fragmented_before > 0.0

    assert pager.soft_compact_if_needed() is True
    metrics = pager.metrics_snapshot()
    assert metrics["compaction_count"] >= 1
