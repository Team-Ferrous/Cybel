from core.collaboration.negotiation import CollaborationNegotiator


def test_semantic_conflict_radar_scores_symbol_overlap_as_high_risk():
    payload = CollaborationNegotiator.score_semantic_conflict(
        {
            "files": ["core/networking/peer_discovery.py"],
            "context_symbols": ["PeerDiscovery.refresh"],
        },
        {
            "files": ["core/networking/peer_discovery.py"],
            "context_symbols": ["PeerDiscovery.refresh", "PeerDiscovery.status"],
        },
    )

    assert payload["risk_level"] == "high"
    assert payload["symbol_overlap"] == ["PeerDiscovery.refresh"]
    assert payload["file_overlap"] == ["core/networking/peer_discovery.py"]
