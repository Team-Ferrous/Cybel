from core.campaign.ledger import TheLedger


def make_ledger(tmp_path, campaign_id="campaign-1"):
    return TheLedger(
        campaign_name="test campaign",
        campaign_id=campaign_id,
        db_path=str(tmp_path / "ledger.db"),
    )


def test_record_and_retrieve_resources(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record_resource(
        name="competitor-repo",
        path="/tmp/competitor",
        role="competitor",
        read_only=True,
        metadata={"source": "github"},
    )

    resources = ledger.get_resources()
    assert resources == [
        {
            "name": "competitor-repo",
            "path": "/tmp/competitor",
            "role": "competitor",
            "read_only": True,
            "metadata": {"source": "github"},
            "recorded_at": resources[0]["recorded_at"],
        }
    ]


def test_record_and_retrieve_evidence(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record_evidence(
        name="gap-analysis",
        summary="Benchmarks are missing from competitor claims.",
        evidence_type="research",
        confidence="medium",
        payload={"sources": 3},
        source_phase="phase_research",
    )

    evidence = ledger.get_evidence()
    assert evidence[0]["name"] == "gap-analysis"
    assert evidence[0]["payload"] == {"sources": 3}
    summary = ledger.get_context_summary()
    assert "gap-analysis" in summary
    assert "phase_research" in summary
