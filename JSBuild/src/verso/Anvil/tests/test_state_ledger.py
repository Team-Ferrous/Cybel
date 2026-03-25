from saguaro.state.ledger import StateLedger


def test_state_ledger_exposes_delta_watermark_and_changeset(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "alpha.py"
    source.write_text("print('a')\n", encoding="utf-8")

    ledger = StateLedger(str(repo))
    before = ledger.delta_watermark()

    result = ledger.record_changes(changed_files=[str(source)], reason="unit_test")
    after = ledger.delta_watermark()
    changeset = ledger.changeset_since(before)

    assert result["events_written"] == 1
    assert after["delta_id"] == after["event_id"]
    assert after["logical_clock"] > before["logical_clock"]
    assert changeset["changed_files"] == ["alpha.py"]
    assert changeset["watermark"]["delta_id"] == after["delta_id"]


def test_state_ledger_compares_filesystem_and_projects_state(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    alpha = repo / "alpha.py"
    alpha.write_text("print('a')\n", encoding="utf-8")

    ledger = StateLedger(str(repo))
    ledger.record_changes(changed_files=[str(alpha)], reason="unit_test")

    alpha.write_text("print('b')\n", encoding="utf-8")
    beta = repo / "beta.py"
    beta.write_text("print('beta')\n", encoding="utf-8")

    comparison = ledger.compare_with_filesystem(["alpha.py", "beta.py"])
    projection = ledger.state_projection_lines()

    assert comparison["changed_files"] == ["alpha.py", "beta.py"]
    assert comparison["deleted_files"] == []
    assert any(line.startswith("alpha.py:") for line in projection)
