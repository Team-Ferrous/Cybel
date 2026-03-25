from core.campaign.ledger import TheLedger


def make_ledger(tmp_path, campaign_id="campaign-1"):
    return TheLedger(
        campaign_name="test campaign",
        campaign_id=campaign_id,
        db_path=str(tmp_path / "ledger.db"),
    )


def test_record_and_retrieve_metric(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record_metric("total_files", 10)
    assert ledger.get_metric("total_files") == 10


def test_record_and_retrieve_artifact(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record_artifact("summary", "hello")
    artifacts = ledger.get_all_artifacts()
    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "summary"


def test_phase_result_persistence(tmp_path):
    ledger = make_ledger(tmp_path)
    ledger.record_phase_result("phase_a", "passed", {"ok": True})
    phase_results = ledger.get_phase_results()
    assert len(phase_results) == 1
    assert phase_results[0]["phase_id"] == "phase_a"


def test_context_summary_budget(tmp_path):
    ledger = make_ledger(tmp_path)
    for index in range(100):
        ledger.record_artifact(f"artifact_{index}", "x" * 200)
    summary = ledger.get_context_summary(budget_tokens=25)
    assert len(summary.split()) <= 30


def test_sqlite_durability(tmp_path):
    ledger = make_ledger(tmp_path, campaign_id="durable")
    ledger.record_metric("passes", 3)
    ledger.close()

    restored = make_ledger(tmp_path, campaign_id="durable")
    assert restored.get_metric("passes") == 3
