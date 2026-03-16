import time

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import (
    LatentPackageRecord,
    MemoryConsolidationJobs,
    MemoryFabricStore,
    MemoryProjector,
)


def test_memory_consolidation_jobs_create_summaries_and_archive_expired_latent(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    projector = MemoryProjector()
    jobs = MemoryConsolidationJobs(fabric, projector)

    convo = fabric.create_memory(
        memory_kind="conversation_turn",
        payload_json={"role": "user", "content": "Need telemetry replay fixed"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        session_id="session-1",
        source_system="test",
        summary_text="Need telemetry replay fixed",
        retention_class="durable",
        provenance_json={"source": "chat"},
    )
    fabric.create_memory(
        memory_kind="conversation_turn",
        payload_json={"role": "assistant", "content": "Telemetry replay needs durable checkpoints"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        session_id="session-1",
        source_system="test",
        summary_text="Telemetry replay needs durable checkpoints",
        retention_class="durable",
        provenance_json={"source": "chat"},
    )
    projector.project_memory(fabric, convo)

    claim_a = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"topic": "telemetry"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        source_system="test",
        summary_text="Telemetry schema contract reduces drift",
        provenance_json={"source": "doc-a"},
    )
    claim_b = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"topic": "telemetry"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        source_system="test",
        summary_text="Telemetry schema contract reduces drift",
        provenance_json={"source": "doc-b"},
    )
    fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"topic": "telemetry"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        source_system="test",
        summary_text="Telemetry schema contract fails without validation",
        provenance_json={"source": "doc-c"},
    )
    projector.project_memory(fabric, claim_a, include_multivector=True)
    projector.project_memory(fabric, claim_b, include_multivector=True)

    package = LatentPackageRecord(
        latent_package_id="expired-latent",
        memory_id=claim_a.memory_id,
        model_family="qsg-python",
        model_revision="v1",
        hidden_dim=1,
        qsg_runtime_version="qsg.v1",
        capture_stage="unit_test",
        summary_text="expired package",
        created_at=time.time() - 3600,
        expires_at=time.time() - 60,
    )
    original_uri = fabric.put_latent_package(package, tensor=[[1.0]])

    report = jobs.run_milestone_consolidation("campaign-1")
    session_summaries = fabric.list_memories("campaign-1", memory_kinds=["session_summary"])
    claim_clusters = fabric.list_memories("campaign-1", memory_kinds=["claim_cluster"])
    temporal = fabric.list_memories("campaign-1", memory_kinds=["temporal_summary"])
    community = fabric.list_memories("campaign-1", memory_kinds=["community_summary"])
    telemetry = fabric.list_memories("campaign-1", memory_kinds=["telemetry_snapshot"])
    archived_package = fabric.latest_latent_package(claim_a.memory_id)

    assert report["job_count"] == 7
    assert session_summaries
    assert claim_clusters
    assert temporal
    assert community
    assert telemetry
    assert archived_package is not None
    assert archived_package["tensor_uri"] != original_uri
    assert "latent_archive" in archived_package["tensor_uri"]
