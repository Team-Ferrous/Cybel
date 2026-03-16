from __future__ import annotations

import numpy as np

from saguaro.chronicle import diff as diff_module
from saguaro.chronicle.diff import SemanticDiff
from saguaro.sentinel.engines import semantic as semantic_module
from saguaro.sentinel.engines.semantic import SemanticEngine


class _StorageStub:
    def __init__(self, snapshot: dict[str, object] | None) -> None:
        self._snapshot = snapshot

    def get_latest_snapshot(self) -> dict[str, object] | None:
        return self._snapshot


def test_decode_blob_prefers_raw_and_skips_deserializer_for_aligned_blob(
    monkeypatch,
) -> None:
    calls: list[object] = []

    def _should_not_run(*_args: object, **_kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("deserialize_hd_state should not be called")

    monkeypatch.setattr(diff_module, "deserialize_hd_state", _should_not_run)
    payload = np.asarray([0.25, -0.5, 1.0], dtype=np.float32).tobytes()

    decoded, details = SemanticDiff._decode_blob(payload, np.float32)

    assert calls == []
    assert details["status"] == "ok"
    assert details["decode_mode"] == "raw"
    assert np.allclose(decoded, np.asarray([0.25, -0.5, 1.0], dtype=np.float32))


def test_decode_blob_marks_unaligned_payload_indeterminate(monkeypatch) -> None:
    monkeypatch.setattr(diff_module, "deserialize_hd_state", None)

    decoded, details = SemanticDiff._decode_blob(b"\x01\x02\x03", np.float32)

    assert decoded.size == 0
    assert details["status"] == "indeterminate"
    assert details["reason"] == "invalid_blob_size"


def test_calculate_drift_empty_blob_is_indeterminate() -> None:
    drift, details = SemanticDiff.calculate_drift(b"", b"\x00\x00\x80?")

    assert drift == 0.0
    assert details["status"] == "indeterminate"
    assert details["reason"] == "empty_state_blob"


def test_calculate_drift_supports_projection_state_blobs() -> None:
    baseline = (
        "src/a.py:hash-a:0:1\n"
        "src/b.py:hash-b:0:1\n"
    ).encode("utf-8")
    current = (
        "src/a.py:hash-a:0:1\n"
        "src/b.py:hash-c:0:2\n"
        "src/c.py:hash-d:0:1\n"
    ).encode("utf-8")

    drift, details = SemanticDiff.calculate_drift(baseline, current)

    assert details["status"] == "ok"
    assert details["comparison_mode"] == "projection"
    assert details["decode_mode"] == "projection"
    assert details["path_count"] == 3
    assert details["added_count"] == 1
    assert details["changed_count"] == 1
    assert details["removed_count"] == 0
    assert drift == 2 / 3


def test_semantic_engine_skips_violation_when_state_decode_is_indeterminate(
    monkeypatch, tmp_path
) -> None:
    snapshot = {
        "id": 7,
        "description": "baseline",
        "hd_state_blob": b"\x01\x02\x03",
    }
    monkeypatch.setattr(
        semantic_module, "ChronicleStorage", lambda: _StorageStub(snapshot)
    )
    engine = SemanticEngine(str(tmp_path))
    monkeypatch.setattr(engine, "_calculate_current_state", lambda: b"\x04\x05\x06")

    violations = engine.run()

    assert violations == []


def test_semantic_engine_still_emits_true_drift_for_valid_comparable_states(
    monkeypatch, tmp_path
) -> None:
    baseline = np.asarray([1.0, 0.0], dtype=np.float32).tobytes()
    current = np.asarray([-1.0, 0.0], dtype=np.float32).tobytes()
    snapshot = {"id": 9, "description": "baseline", "hd_state_blob": baseline}
    monkeypatch.setattr(
        semantic_module, "ChronicleStorage", lambda: _StorageStub(snapshot)
    )
    engine = SemanticEngine(str(tmp_path))
    monkeypatch.setattr(engine, "_calculate_current_state", lambda: current)

    violations = engine.run()

    assert len(violations) == 1
    assert violations[0]["rule_id"] == "SEMANTIC-DRIFT"


def test_semantic_engine_prefers_ledger_projection_blob(tmp_path) -> None:
    repo = tmp_path
    (repo / ".saguaro").mkdir()
    tracked = repo / "tracked.py"
    tracked.write_text("print('tracked')\n", encoding="utf-8")

    engine = SemanticEngine(str(repo))
    engine.storage = _StorageStub(None)
    engine.storage.get_latest_snapshot = lambda: None
    engine._state_ledger = None

    ledger = semantic_module.StateLedger(str(repo))
    ledger.record_changes(changed_files=[str(tracked)], reason="test")

    blob = engine._calculate_current_state()

    assert blob == ledger.state_projection_blob()


def test_semantic_engine_policy_controls_decode_mode_and_fail_open(
    monkeypatch, tmp_path
) -> None:
    snapshot = {"id": 13, "description": "baseline", "hd_state_blob": b"baseline"}
    monkeypatch.setattr(
        semantic_module, "ChronicleStorage", lambda: _StorageStub(snapshot)
    )
    engine = SemanticEngine(str(tmp_path))
    engine.set_policy(
        {
            "semantic_decode_mode": "raw",
            "semantic_fail_open_on_decode_error": False,
            "semantic_drift_enabled": True,
        }
    )
    monkeypatch.setattr(engine, "_calculate_current_state", lambda: b"current")

    captured: dict[str, object] = {}

    def _capture(*_args: object, **kwargs: object):
        captured.update(kwargs)
        return 0.0, {"status": "ok", "similarity": 1.0}

    monkeypatch.setattr(semantic_module.SemanticDiff, "calculate_drift", _capture)

    violations = engine.run()

    assert violations == []
    assert captured["decode_mode"] == "raw"
    assert captured["fail_open_on_decode_error"] is False


def test_semantic_engine_can_disable_drift_check(monkeypatch, tmp_path) -> None:
    snapshot = {"id": 1, "description": "baseline", "hd_state_blob": b"baseline"}
    monkeypatch.setattr(
        semantic_module, "ChronicleStorage", lambda: _StorageStub(snapshot)
    )
    engine = SemanticEngine(str(tmp_path))
    engine.set_policy({"semantic_drift_enabled": False})

    called: list[object] = []

    def _capture(*_args: object, **_kwargs: object):
        called.append(True)
        return 0.0, {"status": "ok"}

    monkeypatch.setattr(semantic_module.SemanticDiff, "calculate_drift", _capture)

    violations = engine.run()

    assert violations == []
    assert called == []
