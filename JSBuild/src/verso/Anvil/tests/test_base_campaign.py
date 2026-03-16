import os
import subprocess

import pytest

from core.campaign.base_campaign import BaseCampaignLoop, gate, phase


class OrderedCampaign(BaseCampaignLoop):
    campaign_name = "ordered"

    @phase(order=1, name="second", depends_on=["phase_first"])
    def phase_second(self):
        return {"value": 2}

    @gate(phase="phase_second")
    def gate_second(self, result):
        assert result["value"] == 2

    @phase(order=0, name="first")
    def phase_first(self):
        return {"value": 1}

    @gate(phase="phase_first")
    def gate_first(self, result):
        assert result["value"] == 1


class FlakyCampaign(BaseCampaignLoop):
    campaign_name = "flaky"

    def __init__(self, *args, **kwargs):
        self.counter = 0
        super().__init__(*args, **kwargs)

    @phase(order=0, name="flaky")
    def phase_flaky(self):
        self.counter += 1
        return {"attempt": self.counter}

    @gate(phase="phase_flaky")
    def gate_flaky(self, result):
        assert result["attempt"] >= 2


class AlwaysFailCampaign(BaseCampaignLoop):
    campaign_name = "always_fail"

    @phase(order=0, name="fail")
    def phase_fail(self):
        return {"ok": False}

    @gate(phase="phase_fail")
    def gate_fail(self, result):
        assert result["ok"] is True


def make_config(tmp_path):
    return {
        "state_dir": str(tmp_path / "state"),
        "ledger_db_path": str(tmp_path / "ledger.db"),
        "reports_dir": str(tmp_path / "reports"),
        "max_retries_per_phase": 2,
    }


def test_phase_discovery_ordering(tmp_path):
    campaign = OrderedCampaign(root_dir=str(tmp_path), config=make_config(tmp_path))
    names = [phase_info["method_name"] for phase_info in campaign._phases]
    assert names == ["phase_first", "phase_second"]


def test_gate_discovery_mapping(tmp_path):
    campaign = OrderedCampaign(root_dir=str(tmp_path), config=make_config(tmp_path))
    assert set(campaign._gates.keys()) == {"phase_first", "phase_second"}


def test_dependency_resolution(tmp_path):
    campaign = OrderedCampaign(root_dir=str(tmp_path), config=make_config(tmp_path))
    second = [p for p in campaign._phases if p["method_name"] == "phase_second"][0]

    assert campaign._dependencies_met(second) is False
    campaign.state.phase_statuses["phase_first"] = "passed"
    assert campaign._dependencies_met(second) is True


def test_retry_on_gate_failure(tmp_path):
    campaign = FlakyCampaign(root_dir=str(tmp_path), config=make_config(tmp_path))
    report = campaign.run()
    assert report.phase_statuses["phase_flaky"] == "passed"


def test_max_retries_exceeded(tmp_path):
    campaign = AlwaysFailCampaign(
        root_dir=str(tmp_path),
        config={**make_config(tmp_path), "max_retries_per_phase": 1},
    )
    report = campaign.run()
    assert report.phase_statuses["phase_fail"] == "failed"


def test_state_persistence_and_recovery(tmp_path):
    cfg = make_config(tmp_path)
    campaign = OrderedCampaign(root_dir=str(tmp_path), config=cfg, campaign_id="persist")
    campaign.state.phase_statuses["phase_first"] = "passed"
    campaign.state.save()

    recovered = OrderedCampaign(root_dir=str(tmp_path), config=cfg, campaign_id="persist")
    assert recovered.state.phase_statuses["phase_first"] == "passed"


def test_discover_files_excludes_correctly(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "b.py").write_text("print('b')\n", encoding="utf-8")

    campaign = OrderedCampaign(root_dir=str(tmp_path), config=make_config(tmp_path))
    files = campaign.discover_files(extensions=[".py"])
    assert files == [os.path.join("src", "a.py")]


def test_run_shell_timeout(tmp_path):
    campaign = OrderedCampaign(root_dir=str(tmp_path), config=make_config(tmp_path))
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.run_shell("python -c 'import time; time.sleep(2)'", timeout=1)
