from io import StringIO
import json

from rich.console import Console

from core.subagent_communication import MessageBus, MessageType
from core.telemetry.black_box import BlackBoxRecorder


def test_black_box_recorder_exports_run_trace_and_message_segments(tmp_path):
    console = Console(file=StringIO(), force_terminal=False, width=120)
    bus = MessageBus(console=console)
    bus.register_agent("master")
    bus.register_agent("worker")

    recorder = BlackBoxRecorder(str(tmp_path))
    recorder.start_run(
        run_id="run-42",
        task_id="task-42",
        task="Implement black box recording",
        metadata={"agent_name": "tester"},
    )
    recorder.bind_message_bus(bus)
    bus.set_trace_context(run_id="run-42", task_id="task-42", phase="execute")
    bus.send(
        sender="master",
        recipient="worker",
        message_type=MessageType.REQUEST,
        payload={"task": "inspect"},
    )
    recorder.record_event(
        "phase_transition",
        phase="execute",
        status="completed",
        files=["core/unified_chat_loop.py"],
        metadata={"step": "execute"},
    )
    recorder.record_verification(
        modified_files=["core/unified_chat_loop.py"],
        passed=True,
        issues=[],
    )

    manifest = recorder.finalize(
        stop_reason="completed",
        success=True,
        message_bus=bus,
    )

    assert manifest["status"] == "ok"
    assert manifest["message_segment_count"] >= 1
    assert "events" in manifest["artifacts"]
    assert "reality" in manifest["artifacts"]
    assert "message_trace" in manifest["artifacts"]

    message_trace = json.loads(
        (tmp_path / ".anvil" / "flight_recorder" / "run-42" / "message_trace.json").read_text(
            encoding="utf-8"
        )
    )
    assert message_trace["total_segments"] >= 1

    events_payload = json.loads(
        (tmp_path / ".anvil" / "flight_recorder" / "run-42" / "events.json").read_text(
            encoding="utf-8"
        )
    )
    assert events_payload["replay"]["inspectable_without_model"] is True


def test_black_box_recorder_exports_qsg_replay_tapes_and_runtime_status(tmp_path):
    recorder = BlackBoxRecorder(str(tmp_path))
    recorder.start_run(
        run_id="run-qsg",
        task_id="task-qsg",
        task="Replay QSG request",
        metadata={"agent_name": "tester"},
    )
    recorder.event_store.record_qsg_replay_event(
        request_id="req-7",
        stage="resume",
        payload={
            "request_id": "req-7",
            "mission_replay_descriptor": {
                "request_id": "req-7",
                "replay_tape_path": "/tmp/replay.json",
                "capability_digest": "cap-7",
            },
        },
    )
    recorder.record_event(
        "qsg_runtime_snapshot",
        phase="observe",
        status="ok",
        metadata={
            "request_id": "req-7",
            "mission_replay_descriptor": {
                "request_id": "req-7",
                "replay_tape_path": "/tmp/replay.json",
                "capability_digest": "cap-7",
            },
        },
    )

    manifest = recorder.finalize(
        stop_reason="completed",
        success=True,
        extra_metadata={
            "qsg_runtime_status": {
                "capability_vector": {"native_isa_baseline": "avx2"},
                "controller_state": {"frontier": {"selected_mode": "prompt_lookup"}},
            }
        },
    )

    assert manifest["qsg_replay"]["request_ids"] == ["req-7"]
    assert manifest["qsg_replay"]["descriptor_count"] == 2
    assert "qsg_runtime_status" in manifest["artifacts"]
    assert "qsg_replay_tape_req-7" in manifest["artifacts"]
    assert "qsg_mission_replays" in manifest["artifacts"]
