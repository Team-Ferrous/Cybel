from __future__ import annotations

from pathlib import Path

from core.campaign.repo_cache import RepoCache
from core.campaign.repo_registry import CampaignRepoRegistry
from core.campaign.state_store import CampaignStateStore
from core.campaign.workspace import CampaignWorkspace
from core.research.repo_acquisition import RepoAcquisitionService


def test_repo_acquisition_uses_native_comparative_pack(tmp_path: Path) -> None:
    workspace = CampaignWorkspace.create("native-intake", base_dir=str(tmp_path / "campaigns"))
    state_store = CampaignStateStore(workspace.db_path)
    repo_cache = RepoCache(workspace)
    repo_registry = CampaignRepoRegistry(workspace, state_store)
    acquisition = RepoAcquisitionService(repo_cache, repo_registry)

    source = tmp_path / "source_repo"
    source.mkdir(parents=True, exist_ok=True)
    (source / "demo.py").write_text(
        "def acquire_me(token: str) -> bool:\n    return bool(token)\n",
        encoding="utf-8",
    )
    (source / "CMakeLists.txt").write_text("project(native_intake)\n", encoding="utf-8")

    result = acquisition.acquire_local("demo", str(source))

    assert result["analysis_pack"]["producer"] == "ComparativeAnalysisService.native_index"
    assert result["repo_dossier"]["repo_id"] == result["repo"].repo_id
    assert result["corpus_session"]["corpus_id"] == result["repo"].repo_id
    assert Path(result["corpus_session"]["artifact_paths"]["index_manifest"]).exists()
