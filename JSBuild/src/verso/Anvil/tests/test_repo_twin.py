import time

from core.connectivity.repo_twin import ConnectivityRepoTwin
from saguaro.state.ledger import StateLedger
from shared_kernel.event_store import EventStore


class _Presence:
    def snapshot(self):
        return {
            "peer_count": 1,
            "peers": [{"instance_id": "inst-b", "connected": False}],
        }


class _Ownership:
    def get_status_snapshot(self):
        return {"total_claimed_files": 2, "file_owners": {"core/a.py": [{"owner": "agent-a"}]}}


class _Telemetry:
    def summarize(self):
        return {"event_count": 3}


class _Architect:
    def snapshot(self, *, presence=None):
        return {"leader_id": "inst-a", "peer_count": len((presence or {}).get("peers", []))}


def test_repo_twin_summarizes_presence_ownership_and_blocked_promotions(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "alpha.py"
    source.write_text("print('a')\n", encoding="utf-8")

    ledger = StateLedger(str(repo))
    ledger.record_changes(changed_files=[str(source)], reason="unit_test")
    events = EventStore(str(tmp_path / "events.db"))
    events.emit(
        event_type="campaign.verification_lane",
        payload={"promotion_blocked": True},
        run_id="cmp-1",
        source="test",
    )

    twin = ConnectivityRepoTwin(
        state_ledger=ledger,
        presence_service=_Presence(),
        ownership_registry=_Ownership(),
        telemetry=_Telemetry(),
        event_store=events,
        architect_plane=_Architect(),
    )
    payload = twin.capture(label="snapshot", campaign_id="cmp-1")

    assert payload["summary"]["peer_count"] == 1
    assert payload["summary"]["claimed_file_count"] == 2
    assert payload["summary"]["blocked_promotion_count"] == 1
    assert payload["architect"]["leader_id"] == "inst-a"
