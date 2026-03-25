from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import MemoryFabricStore, MemoryProjector, MemoryReranker


def test_memory_multivector_reranker_prefers_token_aligned_candidate(tmp_path):
    store = CampaignStateStore(str(tmp_path / "state.db"))
    fabric = MemoryFabricStore(store, storage_root=str(tmp_path / "fabric"))
    projector = MemoryProjector()
    reranker = MemoryReranker(fabric, projector)

    good = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"summary": "telemetry contract enforcement"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="telemetry contract enforcement",
    )
    weak = fabric.create_memory(
        memory_kind="research_claim",
        payload_json={"summary": "general roadmap planning"},
        campaign_id="campaign-1",
        workspace_id="campaign-1",
        summary_text="general roadmap planning",
    )
    projector.project_memory(fabric, good, include_multivector=True)
    projector.project_memory(fabric, weak, include_multivector=True)

    ranked = reranker.rerank(
        "telemetry contract",
        [
            {"memory_id": weak.memory_id, "summary_text": weak.summary_text, "_dense_score": 0.25},
            {"memory_id": good.memory_id, "summary_text": good.summary_text, "_dense_score": 0.25},
        ],
    )

    assert ranked[0]["memory_id"] == good.memory_id
