import time

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import (
    LatentPackageRecord,
    MemoryFabricAuditor,
    MemoryFabricStore,
    RetentionPolicy,
)


def test_memory_governance_audit_flags_expired_latent_packages(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    memory = fabric.create_memory(
        memory_kind="hypothesis",
        payload_json={"statement": "temporal hierarchy improves replay"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="temporal hierarchy improves replay",
    )
    package = LatentPackageRecord(
        latent_package_id="latent-expired",
        memory_id=memory.memory_id,
        model_family="qsg-python",
        model_revision="v1",
        hidden_dim=1,
        qsg_runtime_version="qsg.v1",
        capture_stage="hypothesis_ranking",
        summary_text="expired latent package",
        created_at=time.time() - 3600,
        expires_at=time.time() - 60,
    )
    fabric.put_latent_package(package, tensor=[[1.0]])

    audit = MemoryFabricAuditor(fabric).audit_campaign("campaign-1")

    assert audit["memory_count"] == 1
    assert "latent-expired" in audit["expired_latent_packages"]
    assert RetentionPolicy.is_expired(package.expires_at) is True
