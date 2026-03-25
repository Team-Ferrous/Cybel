from __future__ import annotations

import time
from types import SimpleNamespace

from cli.commands.manager import ManagerCommand
from core.campaign.runner import CampaignRunner


def _runner(tmp_path):
    return CampaignRunner(
        config={
            "generated_dir": str(tmp_path / "generated"),
            "custom_dir": str(tmp_path / "custom"),
            "state_dir": str(tmp_path / "state"),
            "workspace_dir": str(tmp_path / "managed"),
            "ledger_db_path": str(tmp_path / "ledger.db"),
        }
    )


def test_campaign_daemon_and_manager_surface(tmp_path) -> None:
    runner = _runner(tmp_path)
    target_repo = tmp_path / "repo"
    target_repo.mkdir()
    created = runner.create_autonomy_campaign(
        name="Detached Test",
        objective="Detach a campaign worker",
        directives=["persist logs"],
        root_dir=str(target_repo),
    )
    campaign_id = created["campaign_id"]

    launched = runner.detach_campaign(campaign_id)
    assert launched["state"] == "running"
    assert launched["detached_lane"]["workspace_dir"].endswith("/workspace")

    final = runner.detached_status(campaign_id)
    for _ in range(40):
        if final.get("state") != "running":
            break
        time.sleep(0.1)
        final = runner.detached_status(campaign_id)
    assert final["state"] in {"completed", "running"}

    command = ManagerCommand()
    context = SimpleNamespace(
        brain=None,
        console=None,
        ownership_registry=None,
        campaign_runner_config=runner.config,
    )
    listing = command.execute(["campaigns"], context)
    status = command.execute(["status", campaign_id], context)

    assert campaign_id in str(listing)
    assert campaign_id in str(status)
