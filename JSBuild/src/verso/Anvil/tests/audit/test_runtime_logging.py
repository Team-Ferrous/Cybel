from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from audit.runtime_logging import SuiteEventLogger


def test_suite_event_logger_writes_event_and_transcript(tmp_path: Path) -> None:
    logger = SuiteEventLogger(
        run_id="run123",
        run_root=tmp_path,
        events_path=tmp_path / "events.ndjson",
        transcript_path=tmp_path / "terminal_transcript.log",
        console_log_path=tmp_path / "console.log",
        ui_mode="plain",
        log_level="trace",
    )
    logger.start()
    try:
        logger.set_state(
            phase="preflight",
            lane="preflight",
            planned_attempts=5,
            completed_attempts=1,
            thread_tuple="4x8x16",
        )
        logger.emit(
            level="info",
            source="unit_test",
            event_type="probe",
            message="probe completed",
            phase="preflight",
            lane="preflight",
            payload={"decode_tps": 12.5, "ttft_ms": 111.0},
        )
    finally:
        logger.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(events) == 1
    assert events[0]["event_type"] == "probe"
    assert events[0]["phase"] == "preflight"
    transcript = (tmp_path / "terminal_transcript.log").read_text(encoding="utf-8")
    assert "probe completed" in transcript
    assert "unit_test:probe" in transcript
    console_log = (tmp_path / "console.log").read_text(encoding="utf-8")
    assert "probe completed" in console_log
