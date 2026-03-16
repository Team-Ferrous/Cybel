import os

from core.campaign.runner import CampaignRunner


def make_runner(tmp_path):
    return CampaignRunner(
        config={
            "generated_dir": str(tmp_path / "generated"),
            "custom_dir": str(tmp_path / "custom"),
            "state_dir": str(tmp_path / "state"),
            "ledger_db_path": str(tmp_path / "ledger.db"),
            "max_retries_per_phase": 1,
            "halt_on_failure": True,
        }
    )


def write_campaign(path, with_resume_gate=False):
    marker_logic = ""
    if with_resume_gate:
        marker_logic = """
    @phase(order=1, name='phase_two', depends_on=['phase_one'])
    def phase_two(self):
        marker = os.path.join(self.root_dir, 'marker.txt')
        return {'ok': os.path.exists(marker)}

    @gate(phase='phase_two')
    def gate_two(self, result):
        assert result.get('ok') is True
"""
    content = f"""
import os
from core.campaign.base_campaign import BaseCampaignLoop, gate, phase

class SampleCampaign(BaseCampaignLoop):
    campaign_name = 'sample'

    @phase(order=0, name='phase_one')
    def phase_one(self):
        self.ledger.record_metric('ran', 1)
        return {{'ok': True}}

    @gate(phase='phase_one')
    def gate_one(self, result):
        assert result.get('ok') is True
{marker_logic}
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def test_load_and_run_campaign_file(tmp_path):
    runner = make_runner(tmp_path)
    campaign_path = tmp_path / "generated" / "sample.py"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    write_campaign(campaign_path)

    report = runner.run_campaign(str(campaign_path), root_dir=str(tmp_path))
    assert report.campaign_name == "sample"
    assert report.phase_statuses["phase_one"] == "passed"


def test_dynamic_class_discovery(tmp_path):
    runner = make_runner(tmp_path)
    campaign_path = tmp_path / "generated" / "discover.py"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    write_campaign(campaign_path)

    module = runner._load_module(str(campaign_path))
    cls = runner._find_campaign_class(module)
    assert cls.__name__ == "SampleCampaign"


def test_resume_from_state(tmp_path):
    runner = make_runner(tmp_path)
    campaign_path = tmp_path / "generated" / "resume.py"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    write_campaign(campaign_path, with_resume_gate=True)

    first = runner.run_campaign(str(campaign_path), root_dir=str(tmp_path))
    assert first.phase_statuses["phase_two"] == "failed"

    marker = tmp_path / "marker.txt"
    marker.write_text("ok\n", encoding="utf-8")

    resumed = runner.resume_campaign(first.campaign_id, root_dir=str(tmp_path))
    assert resumed.phase_statuses["phase_two"] == "passed"


def test_report_generation(tmp_path):
    runner = make_runner(tmp_path)
    campaign_path = tmp_path / "generated" / "report.py"
    campaign_path.parent.mkdir(parents=True, exist_ok=True)
    write_campaign(campaign_path)

    report = runner.run_campaign(str(campaign_path), root_dir=str(tmp_path))
    payload = report.to_dict()

    assert payload["campaign_id"]
    assert payload["campaign_name"] == "sample"
    assert isinstance(payload["metrics"], dict)
