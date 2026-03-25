from types import SimpleNamespace

from cli.commands.features import TimelineCommand


class _HistoryStub:
    def get_timeline(self, limit=20):
        return [
            {
                "event_type": "tool:start",
                "wall_clock": "2026-03-02T12:00:00-07:00",
                "monotonic_elapsed_ms": 100,
                "payload": {"tool_name": "read_file"},
            },
            {
                "event_type": "tool:end",
                "wall_clock": "2026-03-02T12:00:01-07:00",
                "monotonic_elapsed_ms": 800,
                "payload": {"tool_name": "read_file", "status": "ok"},
            },
        ][:limit]

    def export_audit(self, output_path: str):
        return output_path


def test_timeline_command_lists_events():
    command = TimelineCommand()
    context = SimpleNamespace(history=_HistoryStub())
    result = command.execute([], context)
    assert "Timeline (2 events)" in result
    assert "tool:start" in result
    assert "tool:end" in result


def test_timeline_command_exports_audit():
    command = TimelineCommand()
    context = SimpleNamespace(history=_HistoryStub())
    result = command.execute(["export", ".anvil/export.json"], context)
    assert "Audit exported to .anvil/export.json" == result
