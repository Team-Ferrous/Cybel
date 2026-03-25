from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import (
    LatentPackageRecord,
    MemoryFabricSnapshotter,
    MemoryFabricStore,
    MemoryProjector,
)


def test_memory_snapshot_restore_round_trip(tmp_path):
    store = CampaignStateStore(str(tmp_path / "source.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "source_fabric"))
    projector = MemoryProjector()
    memory = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"topic": "telemetry", "statement": "Telemetry contracts reduce drift."},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        source_system="test",
        summary_text="Telemetry contracts reduce drift.",
        provenance_json={"source": "unit-test"},
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
        model_family="qsg-python",
        model_revision="v1",
        tokenizer_hash="tok",
        prompt_protocol_hash="almf.v1",
        hidden_dim=2,
        qsg_runtime_version="qsg.v1",
        capture_stage="unit_test",
        summary_text="latent snapshot",
        compatibility_json={
            "model_family": "qsg-python",
            "hidden_dim": 2,
            "tokenizer_hash": "tok",
            "prompt_protocol_hash": "almf.v1",
            "qsg_runtime_version": "qsg.v1",
        },
    )
    fabric.put_latent_package(package, tensor=[[0.1, 0.2]])

    snapshot = MemoryFabricSnapshotter(fabric).snapshot_campaign("campaign-1")

    restored = MemoryFabricStore.from_db_path(
        str(tmp_path / "restored.db"),
        storage_root=str(tmp_path / "restored_fabric"),
    )
    report = MemoryFabricSnapshotter(restored).restore_campaign(
        snapshot["snapshot_dir"],
        target_campaign_id="campaign-2",
    )

    restored_memory = restored.resolve_alias(
        campaign_id="campaign-2",
        source_table="research_claims",
        source_id="claim-1",
    )
    restored_package = restored.latest_latent_package(restored_memory)

    assert report["restored_memories"] == 1
    assert restored_memory
    assert restored.get_embedding(restored_memory) is not None
    assert restored.get_multivector(restored_memory) is not None
    assert restored_package is not None
    assert restored.load_latent_tensor(restored_package).shape == (1, 2)
