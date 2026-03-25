from __future__ import annotations

from config.settings import CAMPAIGN_CONFIG
from core.campaign.base_campaign import BaseCampaignLoop
from core.campaign.daemon import CampaignDaemon
from core.campaign.control_plane import CampaignControlPlane
from core.campaign.ledger import TheLedger
from core.campaign.worktree_manager import CampaignWorktreeManager
from core.env_manager import EnvironmentManager
import importlib.util
import inspect
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

DEFAULT_CAMPAIGN_CONFIG = CAMPAIGN_CONFIG


class CampaignRunner:
    """Load classic campaign modules and façade managed-autonomy workspaces."""

    def __init__(
        self,
        brain_factory=None,
        console=None,
        config: Optional[Dict[str, Any]] = None,
        ownership_registry=None,
    ):
        self.brain_factory = brain_factory
        self.console = console
        self.ownership_registry = ownership_registry

        merged = dict(DEFAULT_CAMPAIGN_CONFIG)
        merged.update(config or {})
        self.config = merged

        self.generated_dir = self.config.get("generated_dir", ".anvil/campaigns/generated")
        self.custom_dir = self.config.get("custom_dir", ".anvil/campaigns/custom")
        self.state_dir = self.config.get("state_dir", ".anvil/campaigns/state")
        self.workspace_dir = self.config.get("workspace_dir") or os.path.dirname(
            os.path.abspath(self.state_dir)
        )
        self.index_path = os.path.join(self.state_dir, "index.json")

        for path in [
            self.generated_dir,
            self.custom_dir,
            self.state_dir,
            self.workspace_dir,
        ]:
            os.makedirs(path, exist_ok=True)

    def run_campaign(self, campaign_path: str, root_dir: str = "."):
        resolved_path = self._resolve_campaign_path(campaign_path)
        module = self._load_module(resolved_path)
        campaign_class = self._find_campaign_class(module)

        campaign = campaign_class(
            root_dir=root_dir,
            brain_factory=self.brain_factory,
            console=self.console,
            config=self.config,
            ownership_registry=self.ownership_registry,
        )
        campaign.state.campaign_path = resolved_path
        campaign.state.save()

        report = campaign.run()
        self._set_index(report.campaign_id, resolved_path)
        return report

    def resume_campaign(self, campaign_id: str, root_dir: str = "."):
        state = self.get_campaign_status(campaign_id)
        if not state:
            raise ValueError(f"No campaign state found for '{campaign_id}'")

        campaign_path = state.get("campaign_path") or self._get_index().get(campaign_id)
        if not campaign_path:
            raise ValueError(f"Campaign path unavailable for '{campaign_id}'. Cannot resume.")

        module = self._load_module(campaign_path)
        campaign_class = self._find_campaign_class(module)

        campaign = campaign_class(
            root_dir=root_dir,
            brain_factory=self.brain_factory,
            console=self.console,
            config=self.config,
            campaign_id=campaign_id,
            ownership_registry=self.ownership_registry,
        )
        campaign.state.campaign_path = campaign_path
        campaign.state.save()
        report = campaign.run()
        self._set_index(campaign_id, campaign_path)
        return report

    def create_autonomy_campaign(
        self,
        *,
        name: str,
        objective: str,
        directives: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        root_dir: str = ".",
    ) -> Dict[str, Any]:
        campaign_id = self._make_campaign_id(name)
        control_plane = CampaignControlPlane.create(
            campaign_id,
            name,
            self.workspace_dir,
            objective=objective,
            directives=directives,
            constraints=constraints,
            root_dir=root_dir,
        )
        environment_profile = EnvironmentManager(root_dir=root_dir).capture_profile()
        control_plane.workspace.write_json(
            "artifacts/intake/environment_profile.json",
            environment_profile,
        )
        control_plane.workspace.save_metadata({"environment_profile": environment_profile})
        status = control_plane.status()
        return {
            "campaign_id": campaign_id,
            "workspace": status["workspace"],
            "campaign": status["campaign"],
        }

    def attach_repo(
        self,
        campaign_id: str,
        *,
        repo_path: str,
        role: str,
        name: Optional[str] = None,
        write_policy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).attach_repo(
            repo_path=repo_path,
            role=role,
            name=name,
            write_policy=write_policy,
            metadata=metadata,
        )

    def acquire_repos(
        self,
        campaign_id: str,
        *,
        repo_specs: List[str | Dict[str, Any]],
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).acquire_repos(repo_specs=repo_specs)

    def run_research(
        self,
        campaign_id: str,
        *,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).run_research(sources=sources)

    def run_eid(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).run_eid()

    def build_questionnaire(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).build_questionnaire()

    def build_feature_map(
        self,
        campaign_id: str,
        *,
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).build_feature_map(candidates=candidates)

    def build_roadmap(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).build_roadmap()

    def promote_final_roadmap(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).promote_final_roadmap()

    def list_artifacts(self, campaign_id: str) -> List[Dict[str, Any]]:
        return self._open_control_plane(campaign_id).list_artifacts()

    def approve_artifact(
        self,
        campaign_id: str,
        artifact_id: str,
        *,
        approved_by: str = "user",
        state: str = "approved",
        notes: str = "",
    ) -> None:
        self._open_control_plane(campaign_id).approve_artifact(
            artifact_id,
            approved_by=approved_by,
            state=state,
            notes=notes,
        )

    def continue_autonomy_campaign(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).continue_campaign()

    def timeline(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).build_mission_timeline()

    def specialist_dashboard(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).build_specialist_dashboard()

    def speculate_roadmap_item(
        self,
        campaign_id: str,
        item_id: str,
        *,
        verifier=None,
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).run_speculative_roadmap_item(
            item_id,
            verifier=verifier,
        )

    def promote_speculative_branch(
        self,
        campaign_id: str,
        comparison_id: str,
        branch_lane_id: str,
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).promote_speculative_branch(
            comparison_id,
            branch_lane_id,
        )

    def detach_campaign(self, campaign_id: str) -> Dict[str, Any]:
        status = self.get_campaign_status(campaign_id) or {}
        control_plane = self._open_control_plane(campaign_id)
        detached_lane = CampaignWorktreeManager(control_plane.workspace.root_dir).prepare(
            f"detached_{campaign_id}",
            [],
        )
        profile = (
            (status.get("workspace") or {}).get("metadata", {}).get("environment_profile")
            or EnvironmentManager(root_dir=".").capture_profile()
        )
        return CampaignDaemon(self.workspace_dir).launch_continue(
            campaign_id,
            environment_profile=profile,
            detached_lane=detached_lane,
        )

    def detached_status(self, campaign_id: str) -> Dict[str, Any]:
        return CampaignDaemon(self.workspace_dir).status(campaign_id)

    def detached_log(self, campaign_id: str, *, lines: int = 20) -> str:
        return CampaignDaemon(self.workspace_dir).tail_log(campaign_id, lines=lines)

    def cancel_detached(self, campaign_id: str) -> Dict[str, Any]:
        return CampaignDaemon(self.workspace_dir).cancel(campaign_id)

    def list_detached_campaigns(self) -> List[Dict[str, Any]]:
        return CampaignDaemon(self.workspace_dir).list_statuses()

    def run_audit(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).run_audit()

    def closure_proof(self, campaign_id: str) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).build_completion_proof()

    def adopt_rule_proposal(
        self,
        campaign_id: str,
        rule_id: str,
        *,
        approved_by: str = "user",
        notes: str = "",
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).adopt_rule_proposal(
            rule_id,
            approved_by=approved_by,
            notes=notes,
        )

    def record_rule_outcome(
        self,
        campaign_id: str,
        rule_id: str,
        *,
        outcome_status: str,
        regression_delta: float = 0.0,
        notes: str = "",
    ) -> Dict[str, Any]:
        return self._open_control_plane(campaign_id).record_rule_outcome(
            rule_id,
            outcome_status=outcome_status,
            regression_delta=regression_delta,
            notes=notes,
        )

    def list_campaigns(self) -> List[Dict[str, Any]]:
        campaigns: List[Dict[str, Any]] = []

        for folder, category in [
            (self.generated_dir, "generated"),
            (self.custom_dir, "custom"),
        ]:
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                if not filename.endswith(".py") or filename.startswith("__"):
                    continue
                full_path = os.path.join(folder, filename)
                campaigns.append(
                    {
                        "name": os.path.splitext(filename)[0],
                        "path": full_path,
                        "category": category,
                    }
                )

        for entry in sorted(os.listdir(self.workspace_dir)):
            candidate = os.path.join(self.workspace_dir, entry)
            metadata_path = os.path.join(candidate, "campaign.json")
            if not os.path.isdir(candidate) or not os.path.exists(metadata_path):
                continue
            campaigns.append(
                {
                    "name": entry,
                    "path": candidate,
                    "category": "managed",
                }
            )
        return campaigns

    def get_campaign_status(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        state_path = os.path.join(self.state_dir, f"{campaign_id}.json")
        status: Optional[Dict[str, Any]] = None
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as handle:
                status = json.load(handle)
        try:
            control_plane_status = self._open_control_plane(campaign_id).status()
        except Exception:
            return status
        merged = dict(status or {})
        merged["control_plane"] = control_plane_status["campaign"]
        merged["workspace"] = control_plane_status["workspace"]
        merged["snapshot"] = control_plane_status["snapshot"]
        merged["detached"] = self.detached_status(campaign_id)
        merged.setdefault(
            "campaign_name",
            control_plane_status["campaign"].get("campaign_name")
            or control_plane_status["campaign"].get("name"),
        )
        merged.setdefault("objective", control_plane_status["campaign"].get("objective"))
        return merged

    def get_ledger_summary(self, campaign_id: str, budget_tokens: int = 2000) -> str:
        state = self.get_campaign_status(campaign_id)
        if state is None:
            raise ValueError(f"No campaign state found for '{campaign_id}'")

        ledger = TheLedger(
            campaign_name=state.get("campaign_name")
            or (state.get("control_plane") or {}).get("name")
            or "Campaign",
            campaign_id=campaign_id,
            db_path=self.config.get(
                "ledger_db_path", ".anvil/campaigns/campaign_ledger.db"
            ),
        )
        return ledger.get_context_summary(budget_tokens=budget_tokens)

    def _resolve_campaign_path(self, campaign_path: str) -> str:
        candidate = campaign_path

        if os.path.exists(candidate):
            return os.path.abspath(candidate)

        candidate_with_ext = f"{candidate}.py" if not candidate.endswith(".py") else candidate
        for folder in [self.generated_dir, self.custom_dir]:
            full_path = os.path.join(folder, os.path.basename(candidate_with_ext))
            if os.path.exists(full_path):
                return os.path.abspath(full_path)

        raise FileNotFoundError(f"Campaign file not found: {campaign_path}")

    @staticmethod
    def _load_module(campaign_path: str):
        module_name = f"campaign_module_{abs(hash(campaign_path))}"
        spec = importlib.util.spec_from_file_location(module_name, campaign_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load campaign module: {campaign_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _find_campaign_class(module):
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj is BaseCampaignLoop:
                continue
            if issubclass(obj, BaseCampaignLoop):
                return obj
        raise ValueError("No BaseCampaignLoop subclass found in module")

    def _get_index(self) -> Dict[str, str]:
        if not os.path.exists(self.index_path):
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _set_index(self, campaign_id: str, campaign_path: str) -> None:
        index = self._get_index()
        index[campaign_id] = campaign_path
        with open(self.index_path, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2)

    def _open_control_plane(self, campaign_id: str) -> CampaignControlPlane:
        return CampaignControlPlane.open(campaign_id, self.workspace_dir)

    @staticmethod
    def _make_campaign_id(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "campaign"
        return f"{slug}_{int(time.time())}"
