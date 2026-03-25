import time

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import (
    MemoryCommunityBuilder,
    MemoryFabricStore,
    MemoryTemporalTreeBuilder,
)


def test_memory_temporal_hierarchy_and_communities(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    now = time.time()
    fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"summary": "today"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="today claim",
        observed_at=now,
        repo_context="core/research",
    )
    fabric.create_memory(
        memory_kind="hypothesis",
        payload_json={"summary": "yesterday"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="yesterday hypothesis",
        observed_at=now - 86400,
        repo_context="core/research",
    )

    memories = fabric.list_memories("campaign-1")
    temporal = MemoryTemporalTreeBuilder().build(memories)
    communities = MemoryCommunityBuilder().build(memories)

    assert temporal["total_days"] == 2
    assert temporal["total_memories"] == 2
    assert communities["community_count"] >= 1
