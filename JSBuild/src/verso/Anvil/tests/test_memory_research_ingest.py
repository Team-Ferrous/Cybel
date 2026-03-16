from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import MemoryFabricStore
from core.research.hypothesis_lab import HypothesisLab
from core.research.store import ResearchStore


def test_research_and_hypothesis_ingest_into_almf(tmp_path):
    state_store = CampaignStateStore(str(tmp_path / "state.db"))
    research_store = ResearchStore("campaign-1", state_store)
    source_id = research_store.record_source(
        "web",
        "memory://telemetry",
        {"topic": "telemetry", "digest": "digest-1"},
    )
    document_id = research_store.record_document(
        source_id,
        "Telemetry",
        {"digest": "doc-1", "content": "Telemetry contract evidence"},
    )
    research_store.record_chunk(
        document_id,
        content="Telemetry contract evidence",
        topic="telemetry",
        position=0,
    )
    claim_id = research_store.record_claim(
        document_id,
        "telemetry",
        "Contract telemetry reduces missing measurements",
        0.9,
        {"url": "memory://telemetry"},
    )
    lab = HypothesisLab("campaign-1", state_store)
    hypotheses = lab.generate("improve telemetry", research_store.list_claims())
    fabric = MemoryFabricStore(state_store, storage_root=str(tmp_path / "fabric"))

    claim_memory_id = fabric.resolve_alias(
        campaign_id="campaign-1",
        source_table="research_claims",
        source_id=claim_id,
    )
    hypothesis_memory_id = fabric.resolve_alias(
        campaign_id="campaign-1",
        source_table="hypotheses",
        source_id=hypotheses[0]["hypothesis_id"],
    )
    edges = fabric.list_edges(src_memory_id=hypothesis_memory_id, edge_type="supports")

    assert claim_memory_id is not None
    assert hypothesis_memory_id is not None
    assert any(edge["dst_memory_id"] == claim_memory_id for edge in edges)
