from types import SimpleNamespace
from unittest.mock import MagicMock

import cli.repl as repl_module
from cli.repl import AgentREPL


class _DummyREPL:
    def __init__(self):
        self.enhanced_loop_enabled = True
        self.current_mission_id = None
        self.console = MagicMock()
        self.brain = SimpleNamespace(
            model_name="qwen35",
            runtime_status=lambda: {
                "backend": "native_qsg",
                "digest": "abc1234567890def",
                "decode_threads": 4,
                "batch_threads": 8,
                "openmp_enabled": True,
                "avx2_enabled": True,
                "capability_vector": {"native_isa_baseline": "avx2"},
                "controller_state": {
                    "frontier": {"selected_mode": "prompt_lookup"},
                    "drift": {"selected_mode": "adaptive"},
                },
                "repo_coupled_runtime": {
                    "delta_watermark": {"delta_id": "delta-7"},
                },
            },
        )
        self.renderer = SimpleNamespace(
            start_live_dashboard=MagicMock(),
            stop_live_dashboard=MagicMock(),
            print_response=MagicMock(),
        )
        self.loop_orchestrator = SimpleNamespace(run=lambda _objective: "done")


def test_repl_mission_emits_trace_id(monkeypatch):
    repl = _DummyREPL()
    events = []

    def _capture(*args, **kwargs):
        component = kwargs.get("component")
        event = kwargs.get("event")
        mission_id = kwargs.get("mission_id")
        events.append((component, event, mission_id))

    monkeypatch.setattr(repl_module, "emit_structured_event", _capture)
    monkeypatch.setattr(repl_module, "get_active_log_file", lambda: "/tmp/granite.log")
    monkeypatch.setattr(repl_module, "_short_git_sha", lambda: "abc1234")

    AgentREPL.run_mission(repl, "test objective")

    mission_ids = {mission_id for _component, _event, mission_id in events if mission_id}
    assert len(mission_ids) == 1
    start_events = [event for _c, event, _m in events if event == "repl.mission.start"]
    end_events = [event for _c, event, _m in events if event == "repl.mission.complete"]
    assert start_events
    assert end_events
    printed = "\n".join(str(call.args[0]) for call in repl.console.print.call_args_list)
    assert "frontier prompt_lookup" in printed
    assert "drift adaptive" in printed
    assert "delta delta-7" in printed
