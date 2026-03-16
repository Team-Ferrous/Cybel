import json
import time

from core.connectivity.repo_presence import RepoPresenceService
from core.networking.instance_identity import InstanceRegistry
from core.networking.peer_discovery import PeerDiscovery
from core.networking.peer_transport import PeerTransport


class _OwnershipSnapshot:
    def __init__(self, claimed_files: int) -> None:
        self.claimed_files = claimed_files

    def get_status_snapshot(self):
        return {
            "total_claimed_files": self.claimed_files,
            "file_owners": {},
        }


def test_repo_presence_projects_campaign_and_claim_state(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    peers_dir = tmp_path / "peers"
    transport_root = tmp_path / "transport"

    reg1 = InstanceRegistry(anvil_dir=str(tmp_path / "one" / ".anvil"), project_root=str(repo_root))
    reg2 = InstanceRegistry(anvil_dir=str(tmp_path / "two" / ".anvil"), project_root=str(repo_root))

    disc1 = PeerDiscovery(reg1.identity, method="filesystem", shared_peers_dir=str(peers_dir))
    disc2 = PeerDiscovery(reg2.identity, method="filesystem", shared_peers_dir=str(peers_dir))
    transport1 = PeerTransport(reg1.identity, provider="filesystem", transport_root=str(transport_root))
    transport2 = PeerTransport(reg2.identity, provider="filesystem", transport_root=str(transport_root))

    service1 = RepoPresenceService(
        instance_registry=reg1,
        peer_discovery=disc1,
        peer_transport=transport1,
        ownership_registry=_OwnershipSnapshot(2),
        campaign_getter=lambda: {"campaign_id": "cmp-1", "phase_id": "development"},
        capability_getter=lambda: {"analysis_capacity": 0.9, "verification_capacity": 0.8},
    )
    service2 = RepoPresenceService(
        instance_registry=reg2,
        peer_discovery=disc2,
        peer_transport=transport2,
        ownership_registry=_OwnershipSnapshot(1),
        campaign_getter=lambda: {"campaign_id": "cmp-1", "phase_id": "verification"},
        capability_getter=lambda: {"analysis_capacity": 0.4, "verification_capacity": 1.0},
    )

    service1.refresh()
    service2.refresh()
    snapshot = service1.refresh()

    assert snapshot["peer_count"] == 1
    peer = snapshot["peers"][0]
    assert peer["campaign_id"] == "cmp-1"
    assert peer["phase_id"] == "verification"
    assert peer["transport_provider"] == "filesystem"

    local = snapshot["local"]
    assert local["active_claim_count"] == 2
    assert local["verification_state"] == "ready"

    transport1.close()
    transport2.close()


def test_peer_discovery_drops_stale_peer_records(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    peers_dir = tmp_path / "peers"

    reg1 = InstanceRegistry(anvil_dir=str(tmp_path / "one" / ".anvil"), project_root=str(repo_root))
    reg2 = InstanceRegistry(anvil_dir=str(tmp_path / "two" / ".anvil"), project_root=str(repo_root))
    disc1 = PeerDiscovery(reg1.identity, method="filesystem", shared_peers_dir=str(peers_dir))
    disc2 = PeerDiscovery(reg2.identity, method="filesystem", shared_peers_dir=str(peers_dir))

    disc1.refresh()
    disc2.refresh()
    disc1.refresh()
    assert len(disc1.get_peers()) == 1

    stale_path = peers_dir / f"{reg2.identity.instance_id}.json"
    payload = json.loads(stale_path.read_text(encoding="utf-8"))
    payload["last_seen"] = time.time() - 600
    stale_path.write_text(json.dumps(payload), encoding="utf-8")

    disc1.refresh()
    assert disc1.get_peers() == []
