from __future__ import annotations

from uuid import uuid4

from core.campaign.control_plane import CampaignControlPlane
from core.campaign.state_store import CampaignStateStore
from core.research.crawler import ResearchCrawler


def test_research_crawler_prioritizes_frontier_and_stops_after_low_yield(tmp_path) -> None:
    store = CampaignStateStore(str(tmp_path / "state.db"))
    crawler = ResearchCrawler(f"campaign-{uuid4().hex[:8]}", store)

    crawler.enqueue("memory://docs", topic="docs", priority=0.2)
    crawler.enqueue("memory://telemetry", topic="telemetry", priority=0.9)

    batch = crawler.dequeue_batch(limit=2)

    assert [row["url"] for row in batch] == ["memory://telemetry", "memory://docs"]

    stop = crawler.stop_conditions(
        [0.18, 0.12, 0.08],
        coverage_complete=True,
        unknowns_remaining=0,
        impact_deltas=[0.04, 0.03, 0.02],
    )

    assert stop["remaining_frontier"] == 0
    assert stop["yield_below_threshold"] is True
    assert stop["impact_below_threshold"] is True
    assert stop["stop_allowed"] is True


def test_control_plane_run_research_emits_digest_with_claims_and_frontier_summary(
    tmp_path,
) -> None:
    control = CampaignControlPlane.create(
        f"research-{uuid4().hex[:8]}",
        "Research Frontier",
        str(tmp_path / "campaigns"),
        objective="Prioritize high-value research evidence.",
        root_dir=str(tmp_path / "repo"),
    )

    digest = control.run_research(
        sources=[
            {
                "topic": "telemetry",
                "url": "memory://telemetry",
                "title": "Telemetry",
                "content": "Telemetry contracts reduce missing measurements during lane promotion.",
                "summary": "Telemetry contracts reduce missing measurements.",
                "confidence": 0.9,
                "novelty_score": 0.8,
                "complexity_score": 0.3,
                "applicability_score": 0.95,
            },
            {
                "topic": "repo analysis",
                "url": "memory://repo-analysis",
                "title": "Repo Analysis",
                "content": "Pre-computed repo analysis packs lower planning latency.",
                "summary": "Pre-computed analysis packs lower planning latency.",
                "confidence": 0.75,
                "novelty_score": 0.65,
                "complexity_score": 0.25,
                "applicability_score": 0.8,
            },
        ]
    )

    claim_topics = {claim["topic"] for claim in digest["claims"]}

    assert claim_topics == {"telemetry", "repo analysis"}
    assert digest["frontier"]["remaining_frontier"] == 0
    assert digest["coverage"]["frontier_size"] == 0
    assert len(digest["clusters"]) >= 1
    assert (
        tmp_path / "campaigns" / control.campaign_id / "artifacts" / "research" / "research_digest.json"
    ).exists()
