from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import LatentPackageRecord, MemoryFabricStore, MemoryProjector


def test_memory_state_store_roundtrip(tmp_path):
    db_path = tmp_path / "state.db"
    storage_root = tmp_path / "fabric"
    store = CampaignStateStore(str(db_path))
    fabric = MemoryFabricStore(store, storage_root=str(storage_root))
    projector = MemoryProjector()

    memory = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"claim": "latent replay needs canonical evidence"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="latent replay canonical evidence",
        claim_id="claim-1",
    )
    fabric.register_alias(
        memory.memory_id,
        "research_claims",
        "claim-1",
        campaign_id="campaign-1",
    )
    projector.project_memory(fabric, memory, include_multivector=True)
    package = LatentPackageRecord(
        latent_package_id="latent-1",
        memory_id=memory.memory_id,
        branch_id="branch-1",
        model_family="qsg-python",
        model_revision="v1",
        tokenizer_hash="tokenizer:unknown",
        prompt_protocol_hash="almf.v1",
        hidden_dim=1,
        qsg_runtime_version="qsg.v1",
        capture_stage="hypothesis_ranking",
        summary_text="latent replay canonical evidence",
    )
    fabric.put_latent_package(package, tensor=[[1.0]])
    store.close()

    reopened = CampaignStateStore(str(db_path))
    reopened_fabric = MemoryFabricStore(reopened, storage_root=str(storage_root))
    restored = reopened_fabric.get_memory(memory.memory_id)

    assert restored is not None
    assert restored["claim_id"] == "claim-1"
    assert (
        reopened_fabric.resolve_alias(
            campaign_id="campaign-1",
            source_table="research_claims",
            source_id="claim-1",
        )
        == memory.memory_id
    )
    assert reopened_fabric.get_embedding(memory.memory_id) is not None
    assert reopened_fabric.get_multivector(memory.memory_id) is not None
    assert reopened_fabric.latest_latent_package(memory.memory_id)["latent_package_id"] == "latent-1"
