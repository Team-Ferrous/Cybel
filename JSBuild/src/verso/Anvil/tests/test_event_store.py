from shared_kernel.event_store import EventStore


def test_event_store_exports_run_with_links_and_replay_metadata(tmp_path):
    store = EventStore(str(tmp_path / "events.db"))
    store.emit(
        "phase_transition",
        {"files": ["src/feature.py"], "tests": ["tests/test_feature.py"]},
        source="test",
        metadata={"symbols": ["feature"]},
        run_id="run-1",
        links=[
            {
                "link_type": "touches",
                "target_type": "code",
                "target_ref": "src/feature.py",
            }
        ],
    )

    payload = store.export_run("run-1", output_path=str(tmp_path / "run-1.json"))

    assert payload["status"] == "ok"
    assert payload["run_id"] == "run-1"
    assert payload["replay"]["inspectable_without_model"] is True
    assert payload["replay"]["event_count"] == 1
    assert len(payload["links"]) >= 3
    assert (tmp_path / "run-1.json").exists()


def test_event_store_exports_mission_checkpoints(tmp_path):
    store = EventStore(str(tmp_path / "events.db"))
    store.emit("run_started", {}, run_id="run-2")
    checkpoint = store.record_checkpoint(
        "run-2",
        "execute",
        "started",
        metadata={"step": "phase_enter"},
        artifacts=[".anvil/flight_recorder/run-2/events.json"],
    )

    payload = store.export_run("run-2", output_path=str(tmp_path / "run-2.json"))

    assert checkpoint["phase"] == "execute"
    assert payload["replay"]["checkpoint_count"] == 1
    assert payload["checkpoints"][0]["phase"] == "execute"
    assert payload["checkpoints"][0]["metadata"]["step"] == "phase_enter"


def test_event_store_builds_resume_payload_from_latest_checkpoint(tmp_path):
    store = EventStore(str(tmp_path / "events.db"))
    store.record_checkpoint(
        "run-3",
        "plan",
        "compiled",
        checkpoint_type="compiled_plan",
        metadata={"compiled_plan_path": ".anvil/missions/run-3/compiled_plan.json"},
        artifacts=[".anvil/missions/run-3/compiled_plan.json"],
    )
    store.record_checkpoint(
        "run-3",
        "execute",
        "completed",
        metadata={"step": "tool_execution"},
        artifacts=[".anvil/missions/run-3/compiled_plan.json"],
    )

    payload = store.build_resume_payload("run-3")

    assert payload["status"] == "ok"
    assert payload["checkpoint_count"] == 2
    assert payload["latest_checkpoint"]["phase"] == "execute"
    assert payload["artifacts"] == [".anvil/missions/run-3/compiled_plan.json"]
