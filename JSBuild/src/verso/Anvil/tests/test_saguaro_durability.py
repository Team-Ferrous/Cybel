from __future__ import annotations

import json
import multiprocessing
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from saguaro.api import SaguaroAPI
from saguaro.errors import SaguaroBusyError, SaguaroStateCorruptionError
from saguaro.indexing.tracker import IndexTracker
from saguaro.reality.store import RealityGraphStore
from saguaro.storage.atomic_fs import atomic_write_json, atomic_write_yaml
from saguaro.storage.index_state import (
    INDEX_ARTIFACTS,
    load_journal,
    load_snapshot_descriptors,
)
from saguaro.storage.locks import RepoLockManager
from saguaro.storage.memmap_vector_store import MemoryMappedVectorStore


def _write_repo(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "auth.py").write_text(
        "class AuthManager:\n"
        "    def login(self, username: str, password: str) -> bool:\n"
        "        return bool(username and password)\n",
        encoding="utf-8",
    )


def _copy_live_index_to_stage(repo_root: Path, generation_id: str) -> Path:
    saguaro_dir = repo_root / ".saguaro"
    stage_dir = saguaro_dir / "staging" / f"index-{generation_id}"
    for rel_path in list(INDEX_ARTIFACTS.values()) + ["index_manifest.json"]:
        src = saguaro_dir / rel_path
        dest = stage_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return stage_dir


def _hold_lock(saguaro_dir: str, mode: str, duration: float, queue: multiprocessing.Queue) -> None:
    manager = RepoLockManager(saguaro_dir)
    with manager.acquire("index", mode=mode, operation=f"hold-{mode}", timeout_seconds=1.0):
        queue.put("locked")
        time.sleep(duration)


def test_atomic_write_json_preserves_previous_contents_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"status":"stable"}\n', encoding="utf-8")

    with pytest.raises(TypeError):
        atomic_write_json(str(path), {"broken": {1, 2, 3}})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "stable"}


def test_atomic_write_yaml_preserves_previous_contents_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("status: stable\n", encoding="utf-8")

    with pytest.raises(yaml.representer.RepresenterError):
        atomic_write_yaml(str(path), {"broken": object()})

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"status": "stable"}


def test_tracker_raises_corruption_for_invalid_json(tmp_path: Path) -> None:
    tracking = tmp_path / "tracking.json"
    tracking.write_text('{"broken": ', encoding="utf-8")

    with pytest.raises(SaguaroStateCorruptionError):
        IndexTracker(str(tracking))


def test_memmap_store_raises_for_mismatched_file_size(tmp_path: Path) -> None:
    storage = tmp_path / "vectors"
    storage.mkdir()
    (storage / "metadata.json").write_text("[]\n", encoding="utf-8")
    (storage / "index_meta.json").write_text(
        json.dumps({"dim": 4, "count": 1, "capacity": 8, "version": 2, "format": "memmap"}),
        encoding="utf-8",
    )
    (storage / "vectors.bin").write_bytes(b"short")

    with pytest.raises(SaguaroStateCorruptionError):
        MemoryMappedVectorStore(str(storage), dim=4)


def test_memmap_store_clear_persists_without_deadlock(tmp_path: Path) -> None:
    storage = tmp_path / "vectors"
    store = MemoryMappedVectorStore(str(storage), dim=4)
    store.add(np.ones(4, dtype=np.float32), {"file": "pkg/auth.py", "name": "AuthManager"})
    assert len(store) == 1

    store.clear()

    reloaded = MemoryMappedVectorStore(str(storage), dim=4)
    assert len(reloaded) == 0


def test_repo_lock_manager_serializes_writer_against_reader(tmp_path: Path) -> None:
    saguaro_dir = tmp_path / ".saguaro"
    saguaro_dir.mkdir()
    queue: multiprocessing.Queue[str] = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_hold_lock,
        args=(str(saguaro_dir), "exclusive", 0.75, queue),
    )
    proc.start()
    assert queue.get(timeout=2.0) == "locked"

    manager = RepoLockManager(str(saguaro_dir))
    with pytest.raises(SaguaroBusyError):
        manager.acquire(
            "index",
            mode="shared",
            operation="reader-during-writer",
            timeout_seconds=0.2,
        )

    proc.join(timeout=3.0)
    assert proc.exitcode == 0


def test_repo_lock_manager_allows_concurrent_readers(tmp_path: Path) -> None:
    saguaro_dir = tmp_path / ".saguaro"
    saguaro_dir.mkdir()
    manager = RepoLockManager(str(saguaro_dir))
    queue: multiprocessing.Queue[str] = multiprocessing.Queue()

    with manager.acquire("index", mode="shared", operation="reader-one", timeout_seconds=1.0):
        proc = multiprocessing.Process(
            target=_hold_lock,
            args=(str(saguaro_dir), "shared", 0.1, queue),
        )
        proc.start()
        assert queue.get(timeout=2.0) == "locked"
        proc.join(timeout=3.0)
        assert proc.exitcode == 0


def test_repo_lock_manager_flags_orphaned_metadata(tmp_path: Path) -> None:
    saguaro_dir = tmp_path / ".saguaro"
    saguaro_dir.mkdir()
    manager = RepoLockManager(str(saguaro_dir))
    atomic_write_json(
        str(saguaro_dir / "locks" / "index.meta.json"),
        {
            "name": "index",
            "mode": "exclusive",
            "operation": "crashed-index",
            "pid": 999999,
            "hostname": "test-host",
            "started_at": time.time() - 60.0,
        },
    )

    status = manager.describe_lock("index")

    assert status["locked"] is False
    assert status["orphaned_metadata"] is True
    assert status["lock_state"] == "stale_metadata"


def test_api_health_reports_integrity_and_locks(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    report = api.health()

    assert report["integrity"]["status"] == "ready"
    assert report["integrity"]["manifest_generation_id"]
    assert "index" in report["locks"]
    assert "state" in report["locks"]
    assert "state_journal" in report
    assert "snapshots" in report
    assert "task_arbiter" in report


def test_api_health_reports_manifest_sha_mismatch(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    tracking_path = tmp_path / ".saguaro" / "tracking.json"
    tracking_path.write_text('{"tampered": true}\n', encoding="utf-8")

    report = api.health()

    assert report["integrity"]["status"] == "mismatch"
    assert "sha256:tracking.json" in report["integrity"]["mismatches"]


def test_recover_clears_orphaned_lock_metadata_without_manifest(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    atomic_write_json(
        str(tmp_path / ".saguaro" / "locks" / "index.meta.json"),
        {
            "name": "index",
            "mode": "exclusive",
            "operation": "crashed-index",
            "pid": 999999,
            "hostname": "test-host",
            "started_at": time.time() - 30.0,
        },
    )

    result = api.recover()

    assert result["status"] == "warning"
    assert result["removed_lock_metadata"]
    assert not (tmp_path / ".saguaro" / "locks" / "index.meta.json").exists()


def test_state_export_restore_round_trip(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    exported = api.state_export()
    assert exported["status"] == "ok"
    assert Path(exported["path"]).exists()
    assert exported["snapshot_id"]
    assert exported["task_receipt"]["task_class"] == "state_export"

    tracking_path = tmp_path / ".saguaro" / "tracking.json"
    original = tracking_path.read_text(encoding="utf-8")
    tracking_path.write_text('{"mutated": true}\n', encoding="utf-8")

    restored = api.state_restore(bundle_path=exported["path"], force=True)

    assert restored["status"] == "ok"
    assert tracking_path.read_text(encoding="utf-8") == original
    assert restored["integrity"]["status"] == "ready"
    assert restored["task_receipt"]["task_class"] == "restore"


def test_debuginfo_bundle_contains_bundle_payload(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    result = api.debuginfo(event_limit=10)

    assert result["status"] == "ok"
    assert result["task_receipt"]["task_class"] == "debuginfo"
    archive_path = Path(result["path"])
    assert archive_path.exists()
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("/bundle.json") for name in names)


def test_verify_preflight_blocks_on_orphaned_lock_metadata(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    atomic_write_json(
        str(tmp_path / ".saguaro" / "locks" / "index.meta.json"),
        {
            "name": "index",
            "mode": "exclusive",
            "operation": "crashed-index",
            "pid": 999999,
            "hostname": "test-host",
            "started_at": time.time() - 30.0,
        },
    )

    result = api.verify(path=".", engines="native", preflight_only=True)

    assert result["status"] == "error"
    assert result["preflight"]["status"] == "blocked"
    assert result["preflight"]["degraded_mode"]["active"] is True
    assert any(
        issue["code"] == "orphaned_lock_metadata:index"
        for issue in result["preflight"]["issues"]
    )
    assert result["engine_receipts"][0]["engine"] == "native"


def test_api_recover_promotes_latest_intact_staging_generation(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    generation_id = "recovery-fixture"
    _copy_live_index_to_stage(tmp_path, generation_id)

    tracking_path = tmp_path / ".saguaro" / "tracking.json"
    tracking_path.write_text('{"broken": ', encoding="utf-8")

    health_before = api.health()
    assert health_before["integrity"]["status"] in {"mismatch", "corrupt"}

    recovered = api.recover()

    assert recovered["status"] == "ok"
    assert recovered["action"] == "promoted_staging_generation"
    assert recovered["manifest_generation_id"] is not None
    assert recovered["task_receipt"]["task_class"] == "repair"
    assert recovered["snapshot_id"]

    health_after = api.health()
    assert health_after["integrity"]["status"] == "ready"
    assert (tmp_path / ".saguaro" / "tracking.json").exists()
    assert list((tmp_path / ".saguaro").glob("tracking.json.corrupt-*"))


def test_index_records_state_journal_and_snapshot_descriptor(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))

    result = api.index(path=".", force=True)

    assert result["status"] == "ok"
    assert result["task_receipt"]["task_class"] == "index"
    assert result["snapshot_id"]

    journal = load_journal(str(tmp_path / ".saguaro"))
    assert any(item["event"] == "index_committed" for item in journal)
    snapshots = load_snapshot_descriptors(str(tmp_path / ".saguaro"))
    assert any(item["category"] == "index_generation" for item in snapshots)


def test_impact_uses_runtime_corroboration(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    store = RealityGraphStore(str(tmp_path))
    store.record_event(
        "test_execution",
        run_id="impact-run",
        files=["pkg/auth.py"],
        tests=["tests/test_auth.py"],
        operation_class="verification",
        trace_id="trace::impact-run",
    )

    result = api.impact("pkg/auth.py")

    assert result["runtime_corroboration"]["event_count"] >= 1
    assert result["confidence_score"] > 0.0
    assert result["analysis_mode"] in {"heuristic_scan_runtime", "code_graph_runtime", "heuristic_scan", "code_graph"}


def test_api_query_rejects_manifest_mismatch_until_recovered(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    generation_id = "query-recovery-fixture"
    _copy_live_index_to_stage(tmp_path, generation_id)

    manifest = json.loads((tmp_path / ".saguaro" / "index_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["tracking.json"]["size"] = -1
    atomic_write_json(str(tmp_path / ".saguaro" / "index_manifest.json"), manifest)

    broken = api.query("authentication login", k=3, strategy="hybrid", auto_refresh=False)
    assert broken["results"] == []
    assert "recover" in broken["error"]

    recovered = api.recover()
    assert recovered["status"] == "ok"

    result = api.query("authentication login", k=3, strategy="hybrid", auto_refresh=False)
    assert result["results"]
    assert result["integrity"]["status"] == "ready"
