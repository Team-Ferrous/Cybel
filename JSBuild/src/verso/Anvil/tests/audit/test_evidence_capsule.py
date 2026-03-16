from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import jsonschema

sys.path.insert(0, os.getcwd())

from audit.evidence_capsule import build_evidence_capsule
from audit.evidence_capsule import stable_result_hash
from audit.runtime_logging import SuiteEventLogger
from audit.runtime_logging import run_logged_subprocess
from audit.runtime_logging import set_active_logger


def test_build_evidence_capsule_stable_hash_changes_with_metrics() -> None:
    base = build_evidence_capsule(
        sequence_id=1,
        tool_run_id="unit:1",
        source="unit_test",
        command=["python", "-c", "print('ok')"],
        cwd="/tmp",
        exit_code=0,
        wall_time_ms=10.0,
        user_time_ms=2.0,
        sys_time_ms=1.0,
        max_rss_mb=32.0,
        stdout_path="/tmp/stdout.log",
        stderr_path="/tmp/stderr.log",
        artifact_paths={"stdout_log": "/tmp/stdout.log"},
        failing_tests=[],
        compiler_diagnostics=[],
        benchmark_metrics={"decode_tps": 11.0},
        summary="unit test succeeded",
        replay={
            "checkpoint_metadata_path": "/tmp/checkpoint.json",
            "flight_recorder_timeline_path": "/tmp/events.ndjson",
            "terminal_transcript_path": "/tmp/terminal.log",
            "inspectable_without_model": True,
        },
        stdout_text="ok\n",
        stderr_text="",
    )
    changed = build_evidence_capsule(
        sequence_id=1,
        tool_run_id="unit:1",
        source="unit_test",
        command=["python", "-c", "print('ok')"],
        cwd="/tmp",
        exit_code=0,
        wall_time_ms=10.0,
        user_time_ms=2.0,
        sys_time_ms=1.0,
        max_rss_mb=32.0,
        stdout_path="/tmp/stdout.log",
        stderr_path="/tmp/stderr.log",
        artifact_paths={"stdout_log": "/tmp/stdout.log"},
        failing_tests=[],
        compiler_diagnostics=[],
        benchmark_metrics={"decode_tps": 12.0},
        summary="unit test succeeded",
        replay={
            "checkpoint_metadata_path": "/tmp/checkpoint.json",
            "flight_recorder_timeline_path": "/tmp/events.ndjson",
            "terminal_transcript_path": "/tmp/terminal.log",
            "inspectable_without_model": True,
        },
        stdout_text="ok\n",
        stderr_text="",
    )

    assert base["result_hash"] == stable_result_hash(base)
    assert changed["result_hash"] == stable_result_hash(changed)
    assert base["result_hash"] != changed["result_hash"]


def test_run_logged_subprocess_persists_base_evidence_capsule(tmp_path: Path) -> None:
    logger = SuiteEventLogger(
        run_id="evidence-test",
        run_root=tmp_path,
        events_path=tmp_path / "events.ndjson",
        transcript_path=tmp_path / "terminal_transcript.log",
        console_log_path=tmp_path / "console.log",
        ui_mode="plain",
        log_level="trace",
    )
    logger.start()
    set_active_logger(logger)
    try:
        stdout_path = tmp_path / "artifact" / "stdout.log"
        stderr_path = tmp_path / "artifact" / "stderr.log"
        completed = run_logged_subprocess(
            cmd=[sys.executable, "-c", "print('phase9 evidence')"],
            cwd=tmp_path,
            env=os.environ.copy(),
            source="unit_test",
            phase="phase9",
            attempt_id="attempt-1",
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    finally:
        set_active_logger(None)
        logger.close()

    assert completed.returncode == 0
    evidence_path = tmp_path / "artifact" / "evidence_capsule.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("audit/schemas/evidence_capsule.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(evidence, schema)
    assert evidence["artifact_paths"]["flight_recorder_timeline"] == str(
        tmp_path / "events.ndjson"
    )
    assert evidence["artifact_paths"]["terminal_transcript"] == str(
        tmp_path / "terminal_transcript.log"
    )
    assert evidence["replay"]["inspectable_without_model"] is True
    assert evidence["summary"] == "unit_test exited with return_code=0"
