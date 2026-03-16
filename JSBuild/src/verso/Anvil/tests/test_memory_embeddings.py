from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import MemoryFabricStore, MemoryProjector, MemoryRetrievalPlanner


def test_memory_embeddings_and_planner_rank_relevant_memory(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    projector = MemoryProjector()
    planner = MemoryRetrievalPlanner(fabric, projector)

    telemetry = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"summary": "contract-driven telemetry reduces missing metrics"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="contract-driven telemetry reduces missing metrics",
    )
    repo = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"summary": "repo cache lowers repeated file system scans"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="repo cache lowers repeated file system scans",
    )
    projector.project_memory(fabric, telemetry, include_multivector=True)
    projector.project_memory(fabric, repo, include_multivector=True)

    result = planner.retrieve(
        campaign_id="campaign-1",
        query_text="telemetry contract missing metrics",
        memory_kinds=["research_claim"],
        limit=2,
    )

    assert result["results"][0]["memory_id"] == telemetry.memory_id
    assert result["results"][1]["memory_id"] == repo.memory_id
