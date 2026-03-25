from shared_kernel.event_store import EventStore


def test_event_store_export_produces_stable_capsule_and_replay_hash(tmp_path):
    store = EventStore(str(tmp_path / "events.db"))
    store.emit(
        event_type="campaign.verification_lane",
        payload={"files": ["core/a.py"], "artifacts": ["artifacts/proof.json"]},
        source="verification",
        run_id="run-1",
    )
    store.record_checkpoint("run-1", "verification", "passed", artifacts=["artifacts/proof.json"])

    export = store.export_run("run-1")

    assert export["replay"]["deterministic_hash"]
    assert export["mission_capsule"]["artifact_refs"] == ["artifacts/proof.json"]
    assert export["closure_summary"]["capsule_id"].startswith("capsule_")
