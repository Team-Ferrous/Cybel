import json

from core.agents.domain.repo.repo_campaign_analysis_subagent import (
    RepoCampaignAnalysisSubagent,
)
from core.campaign.runner import CampaignRunner


def make_runner(tmp_path):
    return CampaignRunner(
        config={
            "generated_dir": str(tmp_path / "generated"),
            "custom_dir": str(tmp_path / "custom"),
            "state_dir": str(tmp_path / "state"),
            "workspace_dir": str(tmp_path / "managed"),
            "ledger_db_path": str(tmp_path / "ledger.db"),
        }
    )


def test_repo_campaign_subagent_prompt_and_tools():
    assert "repo dossier" in RepoCampaignAnalysisSubagent.system_prompt.lower()
    assert "saguaro_query" in RepoCampaignAnalysisSubagent.tools


def test_campaign_repo_flow_persists_repo_dossiers(tmp_path):
    runner = make_runner(tmp_path)
    target_repo = tmp_path / "target_repo"
    analysis_repo = tmp_path / "analysis_repo"
    target_repo.mkdir()
    analysis_repo.mkdir()
    (analysis_repo / "module.py").write_text(
        "# TODO: refine\n\n" "def reusable_component(x):\n" "    return x + 1\n",
        encoding="utf-8",
    )

    created = runner.create_autonomy_campaign(
        name="Repo Dossier Campaign",
        objective="Ingest attached repos with dossier reuse evidence",
        directives=["Persist campaign-grade repo dossiers"],
        root_dir=str(target_repo),
    )
    campaign_id = created["campaign_id"]

    acquired = runner.acquire_repos(campaign_id, repo_specs=[str(analysis_repo)])
    assert acquired["count"] == 1
    assert acquired["repo_dossiers"]
    assert acquired["repo_dossiers"][0]["schema_version"] == "repo_dossier.v1"

    research = runner.run_research(campaign_id)
    assert research["repo_dossiers"]
    assert research["repo_analysis"]["imported_repo_dossiers"] >= 1

    dossier_path = (
        tmp_path
        / "managed"
        / campaign_id
        / "artifacts"
        / "research"
        / "repo_dossiers.json"
    )
    assert dossier_path.exists()
    payload = json.loads(dossier_path.read_text(encoding="utf-8"))
    assert payload["repo_dossiers"]
