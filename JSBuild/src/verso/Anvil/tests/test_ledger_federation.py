from saguaro.state.ledger import StateLedger


def test_state_ledger_can_push_and_pull_peer_bundles(tmp_path):
    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()
    changed = repo_a / "alpha.py"
    changed.write_text("print('a')\n", encoding="utf-8")

    ledger_a = StateLedger(str(repo_a))
    ledger_b = StateLedger(str(repo_b))
    ledger_a.record_changes(changed_files=[str(changed)], reason="unit_test")

    peer_a = ledger_a.peer_add("repo-b", "local://repo-b")["peer"]["peer_id"]
    peer_b = ledger_b.peer_add("repo-a", "local://repo-a")["peer"]["peer_id"]

    bundle = ledger_a.sync_push(peer_a)
    pulled = ledger_b.sync_pull(peer_b, bundle["bundle_path"])

    assert pulled["status"] == "ok"
    assert pulled["applied_events"] == 1
    events = ledger_b.list_events(limit=10)
    assert events
    assert events[0]["path"] == "alpha.py"
