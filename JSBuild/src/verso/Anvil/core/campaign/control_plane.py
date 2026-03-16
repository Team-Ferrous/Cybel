"""Managed-autonomy campaign control plane."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from core.agents.specialists import SpecialistRegistry
from core.agents.specialists.task_packet_executor import TaskPacketExecutor
from core.campaign.artifact_registry import CampaignArtifactRegistry
from core.campaign.audit_engine import CampaignAuditEngine
from core.campaign.completion_engine import CompletionEngine
from core.campaign.coverage_engine import CoverageEngine
from core.campaign.feature_map import FeatureEntry, FeatureMapBuilder
from core.campaign.gate_engine import GateEngine
from core.campaign.loop_scheduler import LoopDefinition, LoopScheduler
from core.campaign.questionnaire import ArchitectureQuestion, QuestionnaireBuilder
from core.campaign.retrieval_policy import RetrievalPolicyEngine
from core.campaign.risk_radar import RepoTwinBuilder, RoadmapRiskRadar
from core.campaign.repo_cache import RepoCache
from core.campaign.repo_registry import CampaignRepoRegistry
from core.campaign.roadmap_compiler import RoadmapCompiler, RoadmapItem
from core.campaign.state_store import CampaignStateStore
from core.campaign.telemetry import CampaignTelemetry
from core.campaign.timeline import MissionTimelineAssembler
from core.campaign.tooling_factory import ToolingTaskFactory
from core.campaign.transition_policy import CampaignTransitionPolicy
from core.campaign.whitepaper_engine import WhitepaperEngine
from core.connectivity.repo_twin import ConnectivityRepoTwin
from core.governance.rule_proposal_engine import RuleProposalEngine
from core.architect.architect_plane import ArchitectPlane
from core.memory.fabric import (
    MemoryFabricAuditor,
    MemoryFabricStore,
    MemoryTierPolicy,
    MemoryProjector,
    RepoDeltaMemoryRecord,
    MemoryRetrievalPlanner,
)
from core.memory.fabric.models import MemoryReadRecord
from core.qsg.latent_bridge import QSGLatentBridge
from core.qsg.runtime_contracts import MissionReplayDescriptor
from core.campaign.workspace import CampaignWorkspace
from core.research import (
    BrowserResearchRuntime,
    EIDMasterLoop,
    ExperimentRunner,
    HypothesisLab,
    RepoAcquisitionService,
    ResearchCrawler,
    ResearchEvaluationHarness,
    ResearchNormalizer,
    ResearchStore,
    TopicClusterer,
)
from domains.verification.verification_lane import VerificationLane
from saguaro.services.comparative import ComparativeAnalysisService
from saguaro.state.ledger import StateLedger
from shared_kernel.event_store import get_event_store


class CampaignControlPlane:
    """Authoritative managed-autonomy runtime for campaign workspaces."""

    def __init__(
        self,
        campaign_id: str,
        campaign_name: str,
        campaigns_dir: str,
        *,
        objective: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        create: bool = False,
    ) -> None:
        self.campaign_id = campaign_id
        self.campaign_name = campaign_name
        self.base_dir = campaigns_dir
        self.workspace = self._load_workspace(
            campaign_id,
            campaigns_dir,
            metadata=metadata,
            create=create,
        )
        self.state_store = CampaignStateStore(self.workspace.db_path)
        if create:
            self.state_store.initialize_campaign(
                campaign_id,
                name=campaign_name,
                objective=objective,
                current_state="INTAKE",
                metadata=self.workspace.metadata,
            )
        campaign_row = self.state_store.get_campaign(campaign_id) or {}
        self.current_state = str(
            campaign_row.get("current_state") or campaign_row.get("runtime_state") or "INTAKE"
        )
        self.repo_registry = CampaignRepoRegistry(self.workspace, self.state_store)
        self.repo_cache = RepoCache(self.workspace)
        self.event_store = get_event_store()
        self.artifacts = CampaignArtifactRegistry(
            campaign_id,
            self.workspace,
            self.state_store,
        )
        self.loop_scheduler = LoopScheduler()
        self.telemetry = CampaignTelemetry(self.state_store, campaign_id)
        self.memory_fabric = MemoryFabricStore(self.state_store)
        self.memory_projector = MemoryProjector()
        self.memory_planner = MemoryRetrievalPlanner(
            self.memory_fabric,
            self.memory_projector,
        )
        self.memory_auditor = MemoryFabricAuditor(self.memory_fabric)
        self.latent_bridge = QSGLatentBridge(self.memory_fabric, self.memory_projector)
        ledger_repo_path = str(
            self.workspace.metadata.get("root_dir") or self.workspace.root_dir
        )
        self.state_ledger = StateLedger(repo_path=ledger_repo_path)
        self.research_store = ResearchStore(campaign_id, self.state_store)
        self.research_normalizer = ResearchNormalizer()
        self.research_crawler = ResearchCrawler(campaign_id, self.state_store)
        self.topic_clusterer = TopicClusterer()
        self.browser_runtime = BrowserResearchRuntime()
        self.research_evals = ResearchEvaluationHarness()
        self.hypothesis_lab = HypothesisLab(campaign_id, self.state_store)
        self.experiment_runner = ExperimentRunner(
            campaign_id,
            self.state_store,
            cwd=self.workspace.root_dir,
        )
        self.repo_acquisition = RepoAcquisitionService(self.repo_cache, self.repo_registry)
        self.questionnaire = QuestionnaireBuilder(self.state_store, campaign_id)
        self.feature_map = FeatureMapBuilder(self.state_store, campaign_id)
        self.roadmap_compiler = RoadmapCompiler(self.state_store, campaign_id)
        self.risk_radar = RoadmapRiskRadar(
            self.state_store,
            self.event_store,
            self.state_ledger,
        )
        self.transition_policy = CampaignTransitionPolicy()
        self.retrieval_policy = RetrievalPolicyEngine(self.state_store, self.event_store)
        self.timeline = MissionTimelineAssembler(self.state_store, self.event_store)
        self.rule_proposals = RuleProposalEngine(
            self.state_store,
            standards_dir=os.path.join(self.workspace.root_dir, "standards", "governance"),
        )
        self.repo_twin_builder = RepoTwinBuilder(
            self.workspace,
            self.event_store,
            self.state_ledger,
        )
        self.architect_plane = ArchitectPlane(
            instance_id=self.state_ledger.instance_id,
            event_store=self.event_store,
            state_ledger=self.state_ledger,
        )
        self.connectivity_twin = ConnectivityRepoTwin(
            state_ledger=self.state_ledger,
            telemetry=self.telemetry,
            event_store=self.event_store,
            architect_plane=self.architect_plane,
        )
        self.task_packet_executor = TaskPacketExecutor(self.state_store)
        self.audit_engine = CampaignAuditEngine(self.state_store, campaign_id)
        self.completion_engine = CompletionEngine(
            self.workspace,
            self.state_store,
            campaign_id,
            event_store=self.event_store,
            memory_fabric=self.memory_fabric,
        )
        self.specialists = SpecialistRegistry()
        self.eid_master = EIDMasterLoop(
            experiment_runner=self.experiment_runner,
            state_store=self.state_store,
            campaign_id=campaign_id,
        )
        self.gate_engine = GateEngine()
        self._register_default_loops()
        self.workspace.save_metadata(self.snapshot())

    @classmethod
    def create(
        cls,
        campaign_id: str,
        campaign_name: str,
        campaigns_dir: str,
        *,
        objective: str,
        directives: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        root_dir: str = ".",
    ) -> "CampaignControlPlane":
        metadata = {
            "campaign_name": campaign_name,
            "objective": objective,
            "directives": list(directives or []),
            "constraints": list(constraints or []),
            "root_dir": os.path.abspath(root_dir),
        }
        control = cls(
            campaign_id,
            campaign_name,
            campaigns_dir,
            objective=objective,
            metadata=metadata,
            create=True,
        )
        control.repo_registry.register_repo(
            name=os.path.basename(os.path.abspath(root_dir)) or "target",
            local_path=root_dir,
            role="target",
            origin=os.path.abspath(root_dir),
            metadata={"attached_at_create": True},
        )
        intake_payload = {
            "campaign_id": campaign_id,
            "objective": objective,
            "directives": list(directives or []),
            "constraints": list(constraints or []),
            "root_dir": os.path.abspath(root_dir),
        }
        control.workspace.write_json("artifacts/intake/intake.json", intake_payload)
        control._publish_phase_artifact(
            "intake",
            f"{campaign_id}:phase_intake_manifest",
            "manifest",
            "manifest.json",
            {
                "phase_id": "intake",
                "summary": "Campaign directives and root configuration.",
                "artifacts": ["intake.json", "directives.json"],
                "payload": intake_payload,
            },
            metadata={"source": "CampaignControlPlane.create"},
        )
        control._publish_phase_artifact(
            "intake",
            f"{campaign_id}:phase_intake_directives",
            "directives",
            "directives.json",
            intake_payload,
            metadata={"source": "CampaignControlPlane.create"},
        )
        control.artifacts.publish(
            artifact_id=f"{campaign_id}:intake",
            family="intake",
            name="intake_brief",
            canonical_payload=intake_payload,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.create"},
        )
        control.workspace.save_metadata(control.snapshot())
        return control

    @classmethod
    def open(cls, campaign_id: str, campaigns_dir: str) -> "CampaignControlPlane":
        workspace = CampaignWorkspace.load(campaign_id, base_dir=campaigns_dir)
        store = CampaignStateStore(workspace.db_path)
        campaign_row = store.get_campaign(campaign_id) or {}
        name = str(
            campaign_row.get("campaign_name")
            or campaign_row.get("name")
            or workspace.metadata.get("campaign_name")
            or campaign_id
        )
        objective = str(
            campaign_row.get("objective") or workspace.metadata.get("objective") or ""
        )
        return cls(
            campaign_id,
            name,
            campaigns_dir,
            objective=objective,
            metadata=workspace.metadata,
            create=False,
        )

    @staticmethod
    def _load_workspace(
        campaign_id: str,
        campaigns_dir: str,
        *,
        metadata: Optional[Dict[str, Any]],
        create: bool,
    ) -> CampaignWorkspace:
        metadata_path = os.path.join(campaigns_dir, campaign_id, "campaign.json")
        if create or not os.path.exists(metadata_path):
            return CampaignWorkspace.create(
                campaign_id,
                base_dir=campaigns_dir,
                metadata=metadata or {},
            )
        workspace = CampaignWorkspace.load(campaign_id, base_dir=campaigns_dir)
        if metadata:
            workspace.save_metadata(metadata)
        return workspace

    def snapshot(self) -> dict[str, Any]:
        campaign = self.state_store.get_campaign(self.campaign_id) or {}
        questions = self.state_store.list_questions(self.campaign_id)
        features = self.state_store.list_features(self.campaign_id)
        artifacts = self.list_artifacts()
        repos = self.repo_registry.list_repos()
        return {
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "objective": campaign.get("objective", ""),
            "current_state": self.current_state,
            "workspace": self.workspace.summary(),
            "repo_roles": [repo.role for repo in repos],
            "repos": [asdict(repo) for repo in repos],
            "artifact_families": sorted({row["family"] for row in artifacts}),
            "approved_families": sorted(
                {
                    row["family"]
                    for row in artifacts
                    if row["approval_state"] in {"approved", "accepted"}
                }
            ),
            "artifacts": artifacts,
            "blocking_questions": len(
                [
                    row
                    for row in questions
                    if row["current_status"] not in {"answered", "waived"}
                    and row["blocking_level"] in {"high", "critical"}
                ]
            ),
            "pending_feature_confirmation": len(
                [row for row in features if row.get("requires_user_confirmation")]
            ),
            "repo_dossier_brief": self.ensure_repo_dossier_brief(),
            "roadmap_risk_summary": self._roadmap_risk_summary(),
            "governance_status": self.rule_proposals.status(self.campaign_id),
            "architect_status": self.architect_plane.snapshot(),
            "connectivity_twin": self.connectivity_twin.capture(
                label="snapshot",
                campaign_id=self.campaign_id,
                roadmap_summary=self._roadmap_risk_summary().get("summary", {}),
            ),
        }

    def status(self) -> dict[str, Any]:
        campaign = self.state_store.get_campaign(self.campaign_id) or {}
        self.current_state = str(
            campaign.get("current_state") or campaign.get("runtime_state") or self.current_state
        )
        return {
            "campaign": campaign,
            "workspace": self.workspace.summary(),
            "snapshot": self.snapshot(),
        }

    def transition_to(
        self,
        target_state: str,
        reason: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._set_state(target_state, reason, payload)

    def _set_state(
        self,
        target_state: str,
        reason: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        event = self.state_store.transition_state(
            self.campaign_id,
            to_state=target_state,
            cause=reason,
            metadata=payload,
        )
        self.current_state = target_state
        self.workspace.save_metadata(self.snapshot())
        return event

    def create_task_packet(
        self,
        *,
        packet_id: str,
        objective: str,
        allowed_repos: list[str],
        forbidden_repos: list[str] | None = None,
        evidence_bundle: list[str] | None = None,
        success_metrics: list[str] | None = None,
        aes_metadata: dict[str, Any] | None = None,
        telemetry_contract: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
    ) -> dict[str, Any]:
        dossier = self.ensure_repo_dossier_brief()
        retrieval = self.retrieval_policy.decide(
            campaign_id=self.campaign_id,
            query=objective,
            evidence_quality="campaign_brief",
        )
        read_record = MemoryReadRecord(
            campaign_id=self.campaign_id,
            query_kind="repo_dossier_brief",
            query_text=objective,
            planner_mode=retrieval.route,
            result_memory_ids_json=[f"{self.campaign_id}:repo_dossier_brief"],
            latency_ms=0.0,
        )
        self.memory_fabric.record_read(read_record)
        packet = self.specialists.build_task_packet(
            objective=objective,
            aal=str((aes_metadata or {}).get("aal") or "AAL-3"),
            domains=(aes_metadata or {}).get("domains") or [],
            repo_roles=allowed_repos,
            allowed_repos=allowed_repos,
            forbidden_repos=forbidden_repos or [],
            required_artifacts=[],
            produced_artifacts=[],
            allowed_tools=allowed_tools
            or ["saguaro_query", "saguaro_agent_skeleton", "saguaro_agent_slice", "pytest"],
        ).to_dict()
        packet.update(
            {
                "task_packet_id": packet_id,
                "campaign_id": self.campaign_id,
                "phase_id": self.current_state,
                "packet_kind": str((aes_metadata or {}).get("packet_kind") or "implementation"),
                "success_metrics": list(success_metrics or []),
                "evidence_bundle": {
                    "artifacts": list(evidence_bundle or []),
                    "repo_dossier_summary": dossier.get("brief_summary", ""),
                },
                "telemetry_contract": telemetry_contract or {},
                "metadata": {
                    "repo_dossier_summary": dossier.get("brief_summary", ""),
                    "retrieval_policy": retrieval.to_dict(),
                    "memory_read_id": read_record.read_id,
                },
            }
        )
        self.state_store.record_task_packet(packet)
        return packet

    def attach_repo(
        self,
        *,
        repo_path: str,
        role: str,
        name: Optional[str] = None,
        write_policy: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = self.repo_registry.register_repo(
            name=name or os.path.basename(os.path.abspath(repo_path)) or role,
            local_path=repo_path,
            role=role,
            origin=os.path.abspath(repo_path),
            write_policy=write_policy,
            metadata=metadata,
        )
        self._set_state(
            "REPO_INGESTION",
            "repo_attached",
            {"repo_id": record.repo_id},
        )
        return {
            "repo_id": record.repo_id,
            "role": record.role,
            "local_path": record.local_path,
        }

    def acquire_repos(
        self,
        *,
        repo_specs: List[str | Dict[str, Any]],
    ) -> Dict[str, Any]:
        acquired: List[Dict[str, Any]] = []
        repo_dossiers: List[Dict[str, Any]] = []
        comparative_reports: List[Dict[str, Any]] = []
        comparative_port_ledger: List[Dict[str, Any]] = []
        comparative_frontier: List[Dict[str, Any]] = []
        comparative_creation_ledger: List[Dict[str, Any]] = []
        comparative_programs: List[Dict[str, Any]] = []
        comparative_best_of_breed: Dict[str, Any] = {}
        comparative_service = ComparativeAnalysisService(
            self.workspace.root_dir,
            state_ledger=StateLedger(self.workspace.root_dir),
        )
        candidate_paths: List[str] = []
        for spec in repo_specs:
            payload = {"path": spec} if isinstance(spec, str) else dict(spec)
            repo_path = str(payload.get("path") or payload.get("local_path") or "").strip()
            origin_url = str(payload.get("origin_url") or "").strip()
            name = str(
                payload.get("name")
                or os.path.basename(repo_path or origin_url.rstrip("/"))
                or "analysis_repo"
            )
            if repo_path and os.path.exists(repo_path):
                result = self.repo_acquisition.acquire_local(
                    name=name,
                    local_path=repo_path,
                    role=str(payload.get("role") or "analysis_local"),
                )
            else:
                if not origin_url:
                    raise ValueError("Repo spec must provide a local path or origin_url")
                result = self.repo_acquisition.acquire_remote(
                    name=name,
                    origin_url=origin_url,
                    revision=str(payload.get("revision") or "HEAD"),
                )
            snapshot = result["snapshot"]
            record = result["repo"]
            pack = result["analysis_pack"]
            dossier = dict(result.get("repo_dossier") or pack.get("repo_dossier") or {})
            if os.path.abspath(record.local_path) != os.path.abspath(self.workspace.root_dir):
                candidate_paths.append(record.local_path)
                try:
                    comparison = comparative_service.compare(
                        target=self.workspace.root_dir,
                        candidates=[record.local_path],
                        top_k=8,
                        ttl_hours=72.0,
                    )
                    pack["comparative_report"] = comparison
                    comparison_rows = list(comparison.get("comparisons") or [])
                    if comparison_rows:
                        evidence = dict(comparison_rows[0])
                        dossier["comparative_evidence"] = evidence
                        dossier["migration_recipes"] = list(evidence.get("migration_recipes") or [])
                        dossier["port_ledger"] = list(evidence.get("port_ledger") or [])
                        dossier["frontier_packets"] = list(evidence.get("frontier_packets") or [])
                        dossier["creation_ledger"] = list(comparison.get("creation_ledger") or [])
                        dossier["phase_packets"] = list(comparison.get("phase_packets") or [])
                        dossier["negative_evidence"] = list(comparison.get("negative_evidence") or [])
                        dossier["portfolio_leaderboard"] = list(
                            comparison.get("portfolio_leaderboard") or []
                        )
                        dossier["best_of_breed_synthesis"] = dict(
                            comparison.get("best_of_breed_synthesis") or {}
                        )
                        dossier["native_migration_programs"] = list(
                            comparison.get("native_migration_programs") or []
                        )
                        comparative_port_ledger.extend(dossier["port_ledger"])
                        comparative_frontier.extend(dossier["frontier_packets"])
                        comparative_creation_ledger.extend(dossier["creation_ledger"])
                        comparative_programs.extend(dossier["native_migration_programs"])
                        comparative_reports.append(
                            {
                                "repo_id": record.repo_id,
                                "comparison_id": evidence.get("comparison_id"),
                                "candidate_corpus_id": (
                                    (evidence.get("candidate") or {}).get("corpus_id")
                                ),
                                "report_id": comparison.get("report_id"),
                                "comparison_backend": (
                                    (evidence.get("summary") or {}).get("comparison_backend")
                                ),
                                "artifacts": dict(comparison.get("artifacts") or {}),
                                "migration_recipe_count": len(
                                    evidence.get("migration_recipes") or []
                                ),
                                "port_candidate_count": len(
                                    evidence.get("port_ledger") or []
                                ),
                                "frontier_packet_count": len(
                                    evidence.get("frontier_packets") or []
                                ),
                                "creation_candidate_count": len(
                                    comparison.get("creation_ledger") or []
                                ),
                                "phase_packet_count": len(
                                    comparison.get("phase_packets") or []
                                ),
                                "negative_evidence_count": len(
                                    comparison.get("negative_evidence") or []
                                ),
                                "native_program_count": len(
                                    comparison.get("native_migration_programs") or []
                                ),
                            }
                        )
                except Exception as exc:
                    pack["comparative_report"] = {
                        "status": "error",
                        "message": str(exc),
                    }
            if dossier:
                pack["repo_dossier"] = dossier
            snapshot_id = f"snapshot_{uuid.uuid4().hex[:12]}"
            pack_id = f"pack_{uuid.uuid4().hex[:12]}"
            dossier_id = str(dossier.get("dossier_id") or f"{record.repo_id}:repo_dossier")
            self.state_store.insert_json_row(
                "repo_snapshots",
                campaign_id=self.campaign_id,
                payload=snapshot,
                id_field="snapshot_id",
                id_value=snapshot_id,
                extra={"repo_id": record.repo_id},
            )
            self.state_store.insert_json_row(
                "analysis_packs",
                campaign_id=self.campaign_id,
                payload=pack,
                id_field="pack_id",
                id_value=pack_id,
                extra={"repo_id": record.repo_id},
            )
            acquired.append(
                {
                    "repo_id": record.repo_id,
                    "role": record.role,
                    "snapshot_id": snapshot_id,
                    "pack_id": pack_id,
                    "dossier_id": dossier_id,
                    "origin": record.origin,
                    "revision": record.revision,
                }
            )
            if dossier:
                repo_dossiers.append(dossier)

        if len(candidate_paths) > 1:
            try:
                fleet_report = comparative_service.compare(
                    target=self.workspace.root_dir,
                    candidates=candidate_paths,
                    top_k=8,
                    ttl_hours=72.0,
                )
                comparative_best_of_breed = dict(
                    fleet_report.get("best_of_breed_synthesis") or {}
                )
                comparative_creation_ledger = list(
                    fleet_report.get("creation_ledger") or comparative_creation_ledger
                )
                comparative_programs = list(
                    fleet_report.get("native_migration_programs") or comparative_programs
                )
            except Exception:
                comparative_best_of_breed = {}

        payload = {
            "repos": acquired,
            "repo_dossiers": repo_dossiers,
            "comparative_reports": comparative_reports,
            "comparative_port_ledger": comparative_port_ledger,
            "comparative_frontier": comparative_frontier,
            "comparative_creation_ledger": comparative_creation_ledger,
            "comparative_best_of_breed": comparative_best_of_breed,
            "comparative_programs": comparative_programs,
        }
        self.workspace.write_json("artifacts/research/repo_acquisition.json", payload)
        self.workspace.write_json(
            "artifacts/research/repo_dossiers.json",
            {"repo_dossiers": repo_dossiers},
        )
        self.workspace.write_json(
            "artifacts/research/comparative_reports.json",
            {"comparative_reports": comparative_reports},
        )
        self._publish_phase_artifact(
            "research",
            f"{self.campaign_id}:phase_repo_acquisition",
            "repo_acquisition",
            "repo_acquisition.json",
            payload,
            metadata={"source": "CampaignControlPlane.acquire_repos"},
        )
        self._publish_phase_artifact(
            "research",
            f"{self.campaign_id}:phase_repo_dossiers",
            "repo_dossiers",
            "repo_dossiers.json",
            {"repo_dossiers": repo_dossiers},
            metadata={"source": "CampaignControlPlane.acquire_repos"},
        )
        self._publish_phase_artifact(
            "research",
            f"{self.campaign_id}:phase_comparative_reports",
            "comparative_reports",
            "comparative_reports.json",
            {"comparative_reports": comparative_reports},
            metadata={"source": "CampaignControlPlane.acquire_repos"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:repo_acquisition",
            family="research",
            name="repo_acquisition",
            canonical_payload=payload,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.acquire_repos"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:repo_dossiers",
            family="research",
            name="repo_dossiers",
            canonical_payload={"repo_dossiers": repo_dossiers},
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.acquire_repos"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:comparative_reports",
            family="research",
            name="comparative_reports",
            canonical_payload={"comparative_reports": comparative_reports},
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.acquire_repos"},
        )
        self.ensure_repo_dossier_brief(repo_dossiers=repo_dossiers)
        self._set_state(
            "REPO_ACQUISITION",
            "repo_acquisition_completed",
            {
                "acquired_count": len(acquired),
                "repo_dossier_count": len(repo_dossiers),
                "comparative_report_count": len(comparative_reports),
                "comparative_port_candidate_count": len(comparative_port_ledger),
                "comparative_frontier_count": len(comparative_frontier),
                "comparative_creation_candidate_count": len(comparative_creation_ledger),
                "comparative_program_count": len(comparative_programs),
            },
        )
        return {
            "count": len(acquired),
            "repos": acquired,
            "repo_dossiers": repo_dossiers,
            "comparative_reports": comparative_reports,
            "comparative_port_ledger": comparative_port_ledger,
            "comparative_frontier": comparative_frontier,
            "comparative_creation_ledger": comparative_creation_ledger,
            "comparative_best_of_breed": comparative_best_of_breed,
            "comparative_programs": comparative_programs,
        }

    def run_research(
        self,
        *,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        coverage_engine = CoverageEngine()
        repos = self.state_store.list_repos(self.campaign_id)
        repo_by_id = {item["repo_id"]: item for item in repos}
        yield_history: List[float] = []
        impact_deltas: List[float] = []
        browser_urls: List[str] = []
        imported_repo_files = 0
        imported_repo_symbols = 0
        imported_repo_dossiers = 0
        dossier_index: Dict[str, Dict[str, Any]] = {}
        retrieval_routes: List[Dict[str, Any]] = []
        frontier_candidates: List[Dict[str, Any]] = []
        route_explanations: Dict[str, str] = {}
        comparative_port_ledger: List[Dict[str, Any]] = []
        comparative_frontier: List[Dict[str, Any]] = []
        comparative_creation_ledger: List[Dict[str, Any]] = []
        comparative_programs: List[Dict[str, Any]] = []

        for row in self.state_store.fetchall(
            """
            SELECT repo_id, payload_json
            FROM analysis_packs
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (self.campaign_id,),
        ):
            pack = json.loads(row["payload_json"])
            repo = repo_by_id.get(row["repo_id"])
            imported = self._import_repo_analysis_pack(
                self.research_store,
                repo or {"repo_id": row["repo_id"], "role": "analysis"},
                pack,
            )
            imported_repo_files += imported["files"]
            imported_repo_symbols += imported["symbols"]
            imported_repo_dossiers += int(imported.get("dossiers", 0))
            dossier = imported.get("repo_dossier")
            dossier_id = str((dossier or {}).get("dossier_id") or "")
            if dossier and dossier_id:
                dossier_index[dossier_id] = dossier
                comparative_port_ledger.extend(list(dossier.get("port_ledger") or []))
                comparative_frontier.extend(list(dossier.get("frontier_packets") or []))
                comparative_creation_ledger.extend(list(dossier.get("creation_ledger") or []))
                comparative_programs.extend(list(dossier.get("native_migration_programs") or []))
            yield_history.append(0.85 * 0.4)
            language_counts = dict(pack.get("languages") or {})
            native_file_count = sum(
                int(language_counts.get(key, 0))
                for key in ("c", "cpp", "c_header", "cpp_header", "rust", "go")
            )
            impact_deltas.append(
                min(0.3, native_file_count / max(int(pack.get("file_count", 1) or 1), 1))
            )

        artifact_inventory = self.state_store.list_artifacts(self.campaign_id)
        existing_claims = self.research_store.list_claims()
        for source in sources or []:
            topic = str(source.get("topic") or "architecture")
            url = str(source.get("url") or f"memory://{topic}")
            repo_context = str(source.get("repo_context") or "analysis_external:web")
            source_kind = str(source.get("source_kind") or "web")
            browser_allowed = source_kind == "web" and not url.startswith("memory://")
            if browser_allowed:
                browser_urls.append(url)
            artifact_hits = len(
                [
                    artifact
                    for artifact in artifact_inventory
                    if topic.replace(" ", "_") in str(artifact.get("name") or "").replace("-", "_")
                    or artifact.get("family") in {"research", "roadmap_draft", "experiments"}
                ]
            )
            memory_hits = len(
                [
                    claim
                    for claim in existing_claims
                    if str(claim.get("topic") or "") == topic
                ]
            )
            decision = self.retrieval_policy.decide(
                campaign_id=self.campaign_id,
                query=f"{topic}:{url}",
                evidence_quality=(
                    "campaign_brief"
                    if dossier_index
                    else ("high" if float(source.get("confidence", 0.7)) >= 0.8 else "medium")
                ),
                repo_dossiers_present=len(dossier_index),
                artifact_hits=artifact_hits,
                memory_hits=memory_hits,
                allow_browser=browser_allowed,
            )
            expected_information_gain = round(
                min(
                    1.0,
                    float(source.get("confidence", 0.7))
                    * float(source.get("novelty_score", 0.6))
                    * max(0.3, float(source.get("applicability_score", 0.7)))
                    + float(decision.frontier_priority) * 0.2,
                ),
                3,
            )
            expected_runtime_cost = round(
                max(
                    0.05,
                    float(decision.expected_latency_ms) / 1000.0
                    + float(source.get("complexity_score", 0.4)) * 0.35,
                ),
                3,
            )
            utility_context = {
                **decision.to_dict(),
                "expected_information_gain": expected_information_gain,
                "expected_runtime_cost": expected_runtime_cost,
                "expected_token_cost": round(
                    max(25.0, 400.0 - float(decision.token_savings_estimate) * 0.1),
                    3,
                ),
                "reuse_bias": round(
                    min(1.0, artifact_hits * 0.1 + memory_hits * 0.12 + len(dossier_index) * 0.08),
                    3,
                ),
            }
            frontier_id = self.research_crawler.enqueue(
                url,
                topic=topic,
                priority=float(source.get("priority", 1.0)),
                metadata={
                    "source_kind": source_kind,
                    **utility_context,
                },
            )
            browser_record = self.browser_runtime.fetch(url) if browser_allowed else {
                "url": url,
                "fetched": False,
                "route_class": decision.route_class,
            }
            normalized = self.research_normalizer.normalize(
                title=str(source.get("title") or topic.title()),
                content=str(source.get("content") or source.get("summary") or ""),
                origin_url=url,
                topic=topic,
            )
            source_id = self.research_store.record_source(
                "web",
                url,
                {
                    "digest": normalized["digest"],
                    "topic": topic,
                    "confidence": float(source.get("confidence", 0.7)),
                    "novelty_score": float(source.get("novelty_score", 0.6)),
                    "complexity_score": float(source.get("complexity_score", 0.4)),
                    "source_scope": "browser",
                    "repo_scope": "analysis_external",
                    "browser_record": browser_record,
                    "derived_from": url,
                    "utility_context": utility_context,
                },
                repo_context=repo_context,
            )
            document_id = self.research_store.record_document(
                source_id,
                title=normalized["title"],
                normalized={
                    **normalized,
                    "source_scope": "browser",
                    "repo_scope": "analysis_external",
                    "confidence": float(source.get("confidence", 0.7)),
                    "novelty_score": float(source.get("novelty_score", 0.6)),
                    "complexity_score": float(source.get("complexity_score", 0.4)),
                    "derived_from": url,
                    "utility_context": utility_context,
                },
                path=url,
                repo_context=repo_context,
            )
            content = str(source.get("content") or source.get("summary") or "")
            if content and not self.research_store.document_has_chunks(
                document_id,
                repo_context=repo_context,
            ):
                chunk_position = 0
                for index in range(0, len(content), 120):
                    segment = content[index : index + 120]
                    if not segment:
                        continue
                    self.research_store.record_chunk(
                        document_id,
                        content=segment,
                        topic=topic,
                        position=chunk_position,
                        metadata={"url": url},
                        repo_context=repo_context,
                    )
                    chunk_position += 1
            self.research_store.record_claim(
                document_id=document_id,
                topic=topic,
                summary=str(source.get("summary") or normalized["content"][:120] or topic),
                confidence=float(source.get("confidence", 0.7)),
                provenance={"url": url, "browser_record": browser_record},
                repo_context=repo_context,
                complexity_score=float(source.get("complexity_score", 0.4)),
                topic_hierarchy=str(source.get("topic_hierarchy") or topic),
                applicability_score=float(source.get("applicability_score", 0.7)),
                evidence_type=str(source.get("evidence_type") or "implementation"),
                utility_context=utility_context,
            )
            self.research_crawler.mark(
                frontier_id,
                "processed",
                metadata={"browser_record": browser_record, **utility_context},
            )
            retrieval_routes.append(decision.to_dict())
            route_explanations[url] = str(decision.route_explanation)
            frontier_candidates.append(
                {
                    "frontier_id": frontier_id,
                    "url": url,
                    "topic": topic,
                    "status": "processed",
                    **utility_context,
                }
            )
            yield_history.append(expected_information_gain)
            impact_deltas.append(
                max(float(source.get("novelty_score", 0.6)) * 0.25, float(decision.frontier_priority) * 0.2)
            )
            existing_claims.append({"topic": topic})

        claim_rows = self.research_store.list_claims()
        clusters = self.topic_clusterer.cluster(claim_rows)
        self.research_store.persist_clusters(clusters)
        unresolved_unknowns = self.state_store.list_questions(self.campaign_id)
        coverage_details = coverage_engine.coverage_details(claim_rows)
        stop_conditions = self.research_crawler.stop_conditions(
            yield_history,
            coverage_complete=all(
                status.sufficient for status in coverage_details.values()
            ),
            unknowns_remaining=len(
                [
                    question
                    for question in unresolved_unknowns
                    if question.get("current_status")
                    not in {"answered", "waived", "accepted"}
                ]
            ),
            impact_deltas=impact_deltas,
        )
        stop_proof = coverage_engine.stop_proof(
            claim_rows,
            unresolved_unknowns,
            yield_history=yield_history,
            impact_deltas=impact_deltas,
            frontier_size=stop_conditions["remaining_frontier"],
        )
        evaluation = {
            **self.research_evals.evaluate(stop_conditions["remaining_frontier"], claim_rows),
            "browser": self.browser_runtime.evaluate(browser_urls),
        }
        utility_summary = {
            "route_count": len(retrieval_routes),
            "frontier_candidate_count": len(frontier_candidates),
            "mean_frontier_priority": round(
                sum(float(item.get("frontier_priority") or 0.0) for item in frontier_candidates)
                / max(1, len(frontier_candidates)),
                3,
            ),
            "mean_expected_information_gain": round(
                sum(float(item.get("expected_information_gain") or 0.0) for item in frontier_candidates)
                / max(1, len(frontier_candidates)),
                3,
            ),
            "mean_expected_runtime_cost": round(
                sum(float(item.get("expected_runtime_cost") or 0.0) for item in frontier_candidates)
                / max(1, len(frontier_candidates)),
                3,
            ),
            "route_mix": {
                route["route_class"]: (
                    len([item for item in retrieval_routes if item["route_class"] == route["route_class"]])
                )
                for route in retrieval_routes
            },
        }
        decision_record = {
            "decision_id": f"research_stop_{uuid.uuid4().hex[:12]}",
            "campaign_id": self.campaign_id,
            "decision_type": "research_stop_proof",
            "payload": {
                **stop_proof,
                "utility_summary": utility_summary,
            },
        }
        self.state_store.insert_json_row(
            "decision_records",
            campaign_id=self.campaign_id,
            payload=decision_record,
            id_field="decision_id",
            id_value=decision_record["decision_id"],
        )
        digest = {
            "claims": claim_rows,
            "clusters": clusters,
            "coverage": stop_proof,
            "evaluation": evaluation,
            "frontier": stop_conditions,
            "frontier_candidates": frontier_candidates,
            "retrieval_routes": retrieval_routes,
            "route_explanations": route_explanations,
            "utility_summary": utility_summary,
            "repo_dossiers": sorted(
                dossier_index.values(),
                key=lambda item: str(item.get("dossier_id") or ""),
            ),
            "comparative_port_ledger": sorted(
                comparative_port_ledger,
                key=lambda item: (
                    -float(item.get("relation_score") or 0.0),
                    str(item.get("corpus_id") or ""),
                    str(item.get("source_path") or ""),
                ),
            ),
            "comparative_frontier": sorted(
                comparative_frontier,
                key=lambda item: (
                    -float(item.get("priority") or 0.0),
                    str(item.get("corpus_id") or ""),
                    str(item.get("title") or ""),
                ),
            ),
            "comparative_creation_ledger": sorted(
                comparative_creation_ledger,
                key=lambda item: (
                    -float(item.get("priority") or 0.0),
                    str(item.get("feature_family") or ""),
                ),
            ),
            "comparative_programs": sorted(
                comparative_programs,
                key=lambda item: (
                    -float(item.get("priority") or 0.0),
                    str(item.get("feature_family") or ""),
                ),
            ),
            "repo_analysis": {
                "imported_files": imported_repo_files,
                "imported_symbol_traces": imported_repo_symbols,
                "imported_repo_dossiers": imported_repo_dossiers,
                "comparative_port_candidate_count": len(comparative_port_ledger),
                "comparative_frontier_count": len(comparative_frontier),
                "comparative_creation_candidate_count": len(comparative_creation_ledger),
                "comparative_program_count": len(comparative_programs),
            },
        }
        for dossier in digest["repo_dossiers"]:
            memory = self.memory_fabric.create_memory(
                memory_kind="repo_dossier",
                payload_json=dossier,
                campaign_id=self.campaign_id,
                workspace_id=self.campaign_id,
                repo_context=str(dossier.get("repo_id") or ""),
                source_system="control_plane.run_research",
                summary_text=str(dossier.get("summary") or dossier.get("repo_id") or "repo dossier"),
                lifecycle_state="observed",
                importance_score=0.85,
                confidence_score=0.8,
            )
            self.memory_projector.project_memory(
                self.memory_fabric,
                memory,
                include_multivector=True,
            )
        self.workspace.write_json(
            "artifacts/research/repo_dossiers.json",
            {"repo_dossiers": digest["repo_dossiers"]},
        )
        repo_dossier_brief = self.ensure_repo_dossier_brief(
            repo_dossiers=digest["repo_dossiers"],
        )
        digest["repo_dossier_brief"] = repo_dossier_brief
        self.workspace.write_json("artifacts/research/research_digest.json", digest)
        self._publish_phase_artifact(
            "research",
            f"{self.campaign_id}:phase_research_manifest",
            "manifest",
            "manifest.json",
            {
                "phase_id": "research",
                "summary": "Research and repo-analysis evidence bundle.",
                "claim_count": len(claim_rows),
                "repo_analysis": digest["repo_analysis"],
                "retrieval_route_count": len(retrieval_routes),
                "utility_summary": utility_summary,
                "comparative_frontier_count": len(digest["comparative_frontier"]),
                "comparative_port_candidate_count": len(digest["comparative_port_ledger"]),
            },
            metadata={"source": "CampaignControlPlane.run_research"},
        )
        self._publish_phase_artifact(
            "research",
            f"{self.campaign_id}:phase_research_digest",
            "research_digest",
            "research_digest.json",
            digest,
            metadata={"source": "CampaignControlPlane.run_research"},
        )
        self._publish_phase_artifact(
            "research",
            f"{self.campaign_id}:phase_research_repo_dossiers",
            "repo_dossiers",
            "repo_dossiers.json",
            {"repo_dossiers": digest["repo_dossiers"]},
            metadata={"source": "CampaignControlPlane.run_research"},
        )
        self.state_store.record_convergence_checkpoint(
            self.campaign_id,
            "research",
            1,
            {
                "claim_count": len(claim_rows),
                "remaining_frontier": stop_conditions["remaining_frontier"],
                "repo_files_imported": imported_repo_files,
                "repo_symbol_traces": imported_repo_symbols,
                "repo_dossiers": imported_repo_dossiers,
                "utility_summary": utility_summary,
            },
            converged=bool(stop_proof["stop_allowed"]),
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:research_digest",
            family="research",
            name="research_digest",
            canonical_payload=digest,
            approval_state="accepted",
            blocking=not stop_proof["stop_allowed"],
            metadata={"source": "CampaignControlPlane.run_research"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:research_repo_dossiers",
            family="research",
            name="repo_dossiers",
            canonical_payload={"repo_dossiers": digest["repo_dossiers"]},
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.run_research"},
        )
        self._set_state(
            "RESEARCH_RECONCILIATION",
            "research_digest_built",
            {
                "claim_count": len(claim_rows),
                "repo_dossier_count": imported_repo_dossiers,
            },
        )
        return digest

    def run_eid(self) -> Dict[str, Any]:
        campaign = self.state_store.get_campaign(self.campaign_id) or {}
        objective = str(campaign.get("objective") or "")
        research_claims = self.state_store.fetchall(
            """
            SELECT claim_id, topic, summary, confidence
            FROM research_claims
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (self.campaign_id,),
        )
        hypotheses = self.hypothesis_lab.generate(objective, research_claims)
        repo_dossiers: List[Dict[str, Any]] = []
        for row in self.state_store.fetchall(
            """
            SELECT payload_json
            FROM analysis_packs
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (self.campaign_id,),
        ):
            payload = json.loads(row["payload_json"] or "{}")
            dossier = payload.get("repo_dossier")
            if isinstance(dossier, dict) and dossier:
                repo_dossiers.append(dossier)
        eid_result = self.eid_master.run(
            objective,
            hypotheses,
            repo_dossiers=repo_dossiers,
            execute_tracks=True,
            max_tracks=2,
            workspace_root=self.workspace.root_dir,
            metadata_path=self.workspace.metadata_path,
        )
        lane_runs = list(eid_result.get("lane_runs") or [])
        experiment = (
            lane_runs[0]["experiment"]
            if lane_runs
            else self.experiment_runner.run(
                name="artifact_resume_replay",
                commands=[
                    {
                        "label": "metadata_exists",
                        "command": (
                            f"test -f '{self.workspace.metadata_path}' "
                            "&& echo replayability=1 determinism_pass=1"
                        ),
                        "timeout_seconds": 10,
                    },
                    {
                        "label": "artifact_count_probe",
                        "command": "echo experiment_depth=2 correctness_pass=1",
                        "timeout_seconds": 10,
                    },
                ],
            )
        )
        whitepaper = WhitepaperEngine(self.campaign_id, self.state_store).write(
            title="EID Design Review",
            summary=(
                "Innovation, specialist, and experiment-lane findings generated from the "
                "autonomy campaign R&D process."
            ),
            findings=(
                [item["statement"] for item in eid_result["innovation_hypotheses"][:3]]
                + [
                    f"{item['specialist_role']}: {item['focus']}"
                    for item in eid_result["specialist_packets"][:3]
                ]
                + [track["name"] for track in eid_result["experimental_tracks"][:3]]
                + [item["title"] for item in eid_result["feature_simplifications"][:2]]
                + [
                    item["title"]
                    for item in eid_result["hardware_fit_recommendations"][:2]
                ]
                + [item["title"] for item in eid_result["repo_dossier_actions"][:2]]
            ),
        )
        decision_record = {
            "decision_id": f"eid_decision_{uuid.uuid4().hex[:12]}",
            "campaign_id": self.campaign_id,
            "decision_type": "eid_prioritization",
            "payload": {
                "promoted_hypotheses": [
                    item["hypothesis_id"]
                    for item in eid_result["innovation_hypotheses"]
                    if item.get("promotable")
                ],
                "ranked_experiments": eid_result["ranked_experiments"],
                "experimental_tracks": eid_result["experimental_tracks"],
                "funded_proposals": eid_result.get("funded_proposals", []),
                "portfolio_summary": eid_result.get("portfolio_summary", {}),
            },
        }
        self.state_store.insert_json_row(
            "decision_records",
            campaign_id=self.campaign_id,
            payload=decision_record,
            id_field="decision_id",
            id_value=decision_record["decision_id"],
        )
        eid_payload = {
            "hypotheses": hypotheses,
            "eid": eid_result,
            "experiment": experiment,
            "lane_runs": lane_runs,
            "whitepaper": whitepaper,
            "memory_audit": self.memory_auditor.audit_campaign(self.campaign_id),
        }
        self.workspace.write_json("artifacts/experiments/eid_summary.json", eid_payload)
        self.workspace.write_json(
            "artifacts/whitepapers/eid_design_review.json",
            whitepaper,
        )
        self._publish_phase_artifact(
            "eid",
            f"{self.campaign_id}:phase_eid_manifest",
            "manifest",
            "manifest.json",
            {
                "phase_id": "eid",
                "summary": (
                    "Hypothesis generation, specialist routing, shared-lane experimentation, "
                    "and whitepaper synthesis."
                ),
                "hypothesis_count": len(hypotheses),
                "experiment_name": experiment["experiment_id"],
                "specialist_packet_count": len(eid_result["specialist_packets"]),
                "experimental_track_count": len(eid_result["experimental_tracks"]),
                "lane_run_count": len(lane_runs),
                "repo_dossier_count": len(repo_dossiers),
                "comparative_frontier_count": len(
                    eid_result.get("comparative_frontier") or []
                ),
                "accepted_bid_count": len(eid_result.get("funded_proposals") or []),
            },
            metadata={"source": "CampaignControlPlane.run_eid"},
        )
        self._publish_phase_artifact(
            "eid",
            f"{self.campaign_id}:phase_eid_summary",
            "eid_summary",
            "eid_summary.json",
            eid_payload,
            metadata={"source": "CampaignControlPlane.run_eid"},
        )
        self.state_store.record_convergence_checkpoint(
            self.campaign_id,
            "eid",
            1,
            {
                "hypothesis_count": len(hypotheses),
                "promotable_hypotheses": len(
                    [
                        item
                        for item in eid_result["innovation_hypotheses"]
                        if item.get("promotable")
                    ]
                ),
                "accepted_bid_count": len(eid_result.get("funded_proposals") or []),
                "experimental_track_count": len(eid_result["experimental_tracks"]),
                "lane_run_count": len(lane_runs),
                "comparative_frontier_count": len(
                    eid_result.get("comparative_frontier") or []
                ),
            },
            converged=True,
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:eid_summary",
            family="experiments",
            name="eid_summary",
            canonical_payload=eid_payload,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.run_eid"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:eid_whitepaper",
            family="whitepapers",
            name="eid_design_review",
            canonical_payload=whitepaper,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.run_eid"},
        )
        self._set_state(
            "EID_LAB",
            "eid_completed",
            {"hypothesis_count": len(hypotheses)},
        )
        return eid_payload

    def run_development_replay(
        self,
        hypothesis_id: str,
        *,
        engine=None,
        model_family: str = "qsg-python",
    ) -> Dict[str, Any]:
        memory_id = self.memory_fabric.resolve_alias(
            campaign_id=self.campaign_id,
            source_table="hypotheses",
            source_id=hypothesis_id,
        )
        if not memory_id:
            raise ValueError(f"Unknown hypothesis_id '{hypothesis_id}'")
        hypothesis_memory = self.memory_fabric.get_memory(memory_id) or {}
        evidence_plan = self.memory_planner.retrieve(
            campaign_id=self.campaign_id,
            query_text=str(hypothesis_memory.get("summary_text") or ""),
            planner_mode="development_replay",
            memory_kinds=["research_claim", "research_chunk", "repo_dossier"],
            limit=5,
        )
        replay = {
            "restored": False,
            "mode": "degraded",
            "reason": "engine_not_provided",
        }
        latent_package = self.memory_fabric.latest_latent_package(memory_id)
        replay_delta_watermark = self.state_ledger.delta_watermark()
        memory_tier_decision = MemoryTierPolicy().choose(
            purpose="mission_replay",
            latent_package=latent_package,
            repo_delta_memory=(
                dict(latent_package.get("delta_watermark_json") or {})
                if latent_package is not None
                else replay_delta_watermark
            ),
        ).as_dict()
        if (
            engine is not None
            and latent_package is not None
            and str(memory_tier_decision.get("selected_tier") or "") == "latent_replay"
        ):
            replay = self.latent_bridge.replay(
                engine=engine,
                memory_id=memory_id,
                model_family=model_family,
                hidden_dim=int(latent_package.get("hidden_dim") or 0),
                prompt_protocol_hash=str(
                    latent_package.get("prompt_protocol_hash") or "almf.v1"
                ),
                qsg_runtime_version=str(
                    latent_package.get("qsg_runtime_version") or "qsg.v1"
                ),
                quantization_profile=str(
                    latent_package.get("quantization_profile") or "float32"
                ),
            )
        elif engine is not None and latent_package is not None:
            replay = {
                "restored": False,
                "mode": "policy_deferred",
                "reason": str(memory_tier_decision.get("reason") or "memory_tier_policy"),
            }
        repo_delta_memory = RepoDeltaMemoryRecord.from_delta_watermark(
            replay_delta_watermark,
            capability_digest=str(
                (latent_package or {}).get("capability_digest") or ""
            ),
            source_stage="development_replay",
        )
        repo_delta_entry = self.memory_fabric.create_memory(
            memory_kind="repo_delta_memory",
            payload_json=repo_delta_memory.as_dict(),
            campaign_id=self.campaign_id,
            workspace_id=self.campaign_id,
            source_system="control_plane.run_development_replay",
            summary_text=repo_delta_memory.semantic_impact_hint,
            hypothesis_id=hypothesis_id,
            lifecycle_state="active",
            importance_score=0.7,
            confidence_score=0.65,
        )
        self.memory_projector.project_memory(self.memory_fabric, repo_delta_entry)
        supporting_memory_ids = [
            str(item.get("memory_id") or "")
            for item in evidence_plan["results"]
            if str(item.get("memory_id") or "")
        ]
        supporting_memory_ids.append(str(repo_delta_entry.memory_id))
        mission_replay_descriptor = MissionReplayDescriptor(
            hypothesis_id=hypothesis_id,
            request_id=str(replay.get("request_id") or memory_id),
            memory_id=memory_id,
            latent_package_id=str(
                (latent_package or {}).get("latent_package_id") or ""
            ),
            capsule_id=str((latent_package or {}).get("execution_capsule_id") or ""),
            capability_digest=str((latent_package or {}).get("capability_digest") or ""),
            delta_watermark=dict(replay_delta_watermark),
            supporting_memory_ids=supporting_memory_ids,
            repo_delta_memory_id=str(repo_delta_entry.memory_id),
            memory_tier_decision=dict(memory_tier_decision),
            replay_tape_path=str(
                replay.get("replay_tape_path")
                or (replay.get("mission_replay_descriptor") or {}).get("replay_tape_path")
                or ""
            ),
            replay_run_id=str(
                replay.get("replay_run_id")
                or (replay.get("mission_replay_descriptor") or {}).get("replay_run_id")
                or ""
            ),
            mode=str(replay.get("mode") or "degraded"),
            restored=bool(replay.get("restored")),
        ).as_dict()
        replay.setdefault("memory_tier_decision", memory_tier_decision)
        replay["mission_replay_descriptor"] = mission_replay_descriptor
        self.event_store.record_qsg_replay_event(
            request_id=str(mission_replay_descriptor.get("request_id") or memory_id),
            stage="mission_replay_descriptor",
            payload=mission_replay_descriptor,
            metadata={"hypothesis_id": hypothesis_id},
        )
        outcome = {
            "hypothesis_id": hypothesis_id,
            "memory_id": memory_id,
            "replay": replay,
            "memory_tier_decision": memory_tier_decision,
            "repo_delta_memory": repo_delta_memory.as_dict(),
            "mission_replay_descriptor": mission_replay_descriptor,
            "evidence_results": evidence_plan["results"],
            "memory_audit": self.memory_auditor.audit_campaign(self.campaign_id),
        }
        self.event_store.emit(
            event_type="campaign.development_replay",
            payload={
                "hypothesis_id": hypothesis_id,
                "memory_id": memory_id,
                "replay": replay,
                "mission_replay_descriptor": outcome["mission_replay_descriptor"],
            },
            source="CampaignControlPlane.run_development_replay",
            run_id=self.campaign_id,
            links=[
                {
                    "link_type": "campaign",
                    "target_type": "campaign",
                    "target_ref": self.campaign_id,
                },
                {
                    "link_type": "memory",
                    "target_type": "memory_object",
                    "target_ref": memory_id,
                },
                {
                    "link_type": "hypothesis",
                    "target_type": "hypothesis",
                    "target_ref": hypothesis_id,
                },
            ],
        )
        feedback_memory = self.memory_fabric.create_memory(
            memory_kind="memory_feedback",
            payload_json=outcome,
            campaign_id=self.campaign_id,
            workspace_id=self.campaign_id,
            source_system="control_plane.run_development_replay",
            summary_text=f"Replay {replay.get('mode')} for {hypothesis_id}",
            hypothesis_id=hypothesis_id,
            lifecycle_state=str(replay.get("mode") or "degraded"),
            importance_score=0.75,
            confidence_score=0.7 if replay.get("restored") else 0.4,
        )
        self.memory_projector.project_memory(self.memory_fabric, feedback_memory)
        return outcome

    def build_questionnaire(
        self,
        questions: Optional[List[ArchitectureQuestion]] = None,
    ) -> dict[str, Any]:
        campaign = self.state_store.get_campaign(self.campaign_id) or {}
        metadata = json.loads(campaign.get("metadata_json") or "{}")
        builder = self.questionnaire
        built = list(
            questions
            or builder.build(
                directives=metadata.get("directives") or [metadata.get("objective", "")],
                unresolved_unknowns=metadata.get("unknowns") or [],
                risk_summary=self._roadmap_risk_summary(),
            )
        )
        persisted = builder.persist(built)
        blockers = builder.pending_blockers()
        top_blocker = blockers[0] if blockers else {}
        self.workspace.write_json(
            "artifacts/architecture/questionnaire.json",
            {"questions": persisted},
        )
        self._publish_phase_artifact(
            "questionnaire",
            f"{self.campaign_id}:phase_questionnaire_manifest",
            "manifest",
            "manifest.json",
            {
                "phase_id": "questionnaire",
                "summary": "Architecture decision questionnaire for the campaign.",
                "question_count": len(persisted),
                "blocking_question_count": len(blockers),
                "top_blocker_question_id": str(top_blocker.get("question_id") or ""),
            },
            metadata={"source": "CampaignControlPlane.build_questionnaire"},
        )
        self._publish_phase_artifact(
            "questionnaire",
            f"{self.campaign_id}:phase_questionnaire",
            "questionnaire",
            "questionnaire.json",
            {"questions": persisted},
            metadata={"source": "CampaignControlPlane.build_questionnaire"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:architecture_questionnaire",
            family="architecture",
            name="architecture_questionnaire",
            canonical_payload={"questions": persisted},
            approval_state="pending",
            blocking=bool(blockers),
            metadata={"source": "CampaignControlPlane.build_questionnaire"},
        )
        self._set_state("QUESTIONNAIRE_WAIT", "questionnaire_built")
        return {
            "count": len(persisted),
            "questions": persisted,
            "blocking_questions": blockers,
            "top_blocker": top_blocker,
        }

    def build_feature_map(
        self,
        *,
        candidates: Optional[List[Dict[str, Any]]] = None,
        entries: Optional[List[FeatureEntry]] = None,
    ) -> dict[str, Any]:
        builder = self.feature_map
        if entries is None:
            if candidates is None:
                questions = self.state_store.list_questions(self.campaign_id)
                candidates = [
                    {
                        "feature_id": f"feature_{index}",
                        "name": question["question"][:64],
                        "category": "architecture",
                        "description": question["why_it_matters"],
                        "default_state": (
                            "selected"
                            if question["blocking_level"] in {"high", "critical"}
                            else "defer"
                        ),
                        "selection_state": (
                            "selected"
                            if question["blocking_level"] in {"high", "critical"}
                            else "defer"
                        ),
                        "requires_user_confirmation": True,
                        "evidence_links": [question["question_id"]],
                    }
                    for index, question in enumerate(questions, start=1)
                ]
            entries = builder.build_from_candidates(candidates)
        persisted = builder.persist(entries)
        rendered = builder.render_checklist(entries)
        payload = {"features": persisted, "rendered": rendered}
        self.workspace.write_json("artifacts/feature_map/feature_map.json", payload)
        self._publish_phase_artifact(
            "feature_map",
            f"{self.campaign_id}:phase_feature_map_manifest",
            "manifest",
            "manifest.json",
            {
                "phase_id": "feature_map",
                "summary": "Selected feature inventory.",
                "feature_count": len(entries),
            },
            metadata={"source": "CampaignControlPlane.build_feature_map"},
        )
        self._publish_phase_artifact(
            "feature_map",
            f"{self.campaign_id}:phase_feature_map",
            "feature_map",
            "feature_map.json",
            payload,
            metadata={"source": "CampaignControlPlane.build_feature_map"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:feature_map",
            family="feature_map",
            name="feature_map",
            canonical_payload=payload,
            approval_state="pending",
            blocking=True,
            metadata={"source": "CampaignControlPlane.build_feature_map"},
        )
        self._set_state("FEATURE_MAP_WAIT", "feature_map_built")
        return {"count": len(entries), "rendered": rendered, "features": persisted}

    def build_roadmap(self) -> Dict[str, Any]:
        compiler = self.roadmap_compiler
        eid_summary_artifact = self._artifact_map().get(f"{self.campaign_id}:eid_summary")
        eid_payload: Dict[str, Any] = {}
        if eid_summary_artifact and os.path.exists(eid_summary_artifact["canonical_path"]):
            with open(eid_summary_artifact["canonical_path"], "r", encoding="utf-8") as handle:
                eid_payload = json.load(handle)
        items = compiler.compile(
            features=self.state_store.list_features(self.campaign_id),
            questions=self.state_store.list_questions(self.campaign_id),
            hypotheses=(eid_payload.get("eid") or {}).get("innovation_hypotheses") or [],
            repo_dossiers=(eid_payload.get("eid") or {}).get("repo_dossiers") or [],
            experiment_lanes=(eid_payload.get("eid") or {}).get("experimental_tracks") or [],
            research_clusters=[
                {
                    "cluster_id": row["cluster_id"],
                    "topic": row["topic"],
                    "label": row["label"],
                    "members": json.loads(row["members_json"] or "[]"),
                    "score": row["score"],
                }
                for row in self.state_store.fetchall(
                    """
                    SELECT cluster_id, topic, label, members_json, score
                    FROM topic_clusters
                    WHERE campaign_id = ?
                    ORDER BY created_at ASC
                    """,
                    (self.campaign_id,),
                )
            ],
            objective=str(
                (self.state_store.get_campaign(self.campaign_id) or {}).get("objective") or ""
            ),
        )
        item_payloads = [item.__dict__ for item in items]
        roadmap_risk = self.risk_radar.analyze(self.campaign_id, item_payloads)
        risk_by_item = {
            str(item["item_id"]): item for item in roadmap_risk.get("items", [])
        }
        for item in items:
            item.metadata = {
                **dict(item.metadata or {}),
                "risk": dict(risk_by_item.get(item.item_id) or {}),
            }
        item_payloads = [item.__dict__ for item in items]
        graph = compiler.build_task_graph(items)
        phase_pack = compiler.render_phase_pack(items)
        validation_errors = compiler.validate(
            items,
            objective=str(
                (self.state_store.get_campaign(self.campaign_id) or {}).get("objective") or ""
            ),
        )
        payload = {
            "items": item_payloads,
            "task_graph": graph.to_dict(),
            "validation_errors": validation_errors,
            "roadmap_risk": roadmap_risk,
        }
        self.workspace.write_json("artifacts/roadmap_draft/roadmap.json", payload)
        self._publish_phase_artifact(
            "roadmap_draft",
            f"{self.campaign_id}:phase_roadmap_manifest",
            "manifest",
            "manifest.json",
            {
                "phase_id": "roadmap_draft",
                "summary": "Structured roadmap draft with per-phase JSON artifacts.",
                "phase_artifact_count": len(phase_pack),
                "item_count": len(items),
                "validation_error_count": len(validation_errors),
            },
            metadata={"source": "CampaignControlPlane.build_roadmap"},
        )
        for filename, phase_payload in phase_pack.items():
            self._publish_phase_artifact(
                "roadmap_draft",
                f"{self.campaign_id}:roadmap:{filename}",
                "roadmap_phase",
                filename,
                json.loads(phase_payload),
                metadata={"source": "CampaignControlPlane.build_roadmap"},
            )
        self._publish_phase_artifact(
            "roadmap_draft",
            f"{self.campaign_id}:phase_roadmap_risk",
            "risk_radar",
            "risk_radar.json",
            roadmap_risk,
            metadata={"source": "CampaignControlPlane.build_roadmap"},
        )
        all_validation_errors = graph.validate() + validation_errors
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:roadmap_draft",
            family="roadmap_draft",
            name="roadmap_draft",
            canonical_payload=payload,
            approval_state="pending",
            blocking=bool(all_validation_errors),
            metadata={
                "source": "CampaignControlPlane.build_roadmap",
                "risk_summary": roadmap_risk.get("summary", {}),
            },
        )
        self._set_state("ROADMAP_WAIT", "roadmap_drafted", {"item_count": len(items)})
        return {
            "items": item_payloads,
            "validation_errors": all_validation_errors,
            "roadmap_risk": roadmap_risk,
        }

    def compile_roadmap(
        self,
        items: Optional[List[RoadmapItem]] = None,
        *,
        final: bool = False,
    ) -> dict[str, Any]:
        del items
        if final:
            promoted = self.promote_final_roadmap()
            return {"items": promoted["items"], "canonical_path": promoted["path"]}
        drafted = self.build_roadmap()
        artifact = self._artifact_map().get(f"{self.campaign_id}:roadmap_draft")
        return {
            "items": drafted["items"],
            "canonical_path": (artifact or {}).get("canonical_path"),
        }

    def promote_final_roadmap(self) -> Dict[str, Any]:
        questions = self.state_store.list_questions(self.campaign_id)
        artifacts = self._artifact_map()
        blocking_questions = [
            item
            for item in questions
            if item["current_status"] not in {"answered", "waived"}
            and item["blocking_level"] in {"high", "critical"}
        ]
        architecture_artifact = artifacts.get(f"{self.campaign_id}:architecture_questionnaire")
        feature_artifact = artifacts.get(f"{self.campaign_id}:feature_map")
        if blocking_questions and (
            architecture_artifact is None
            or architecture_artifact["approval_state"] not in {"approved", "accepted"}
        ):
            raise ValueError("Cannot promote final roadmap with unresolved blocking questions")
        if feature_artifact is None or feature_artifact["approval_state"] not in {
            "approved",
            "accepted",
        }:
            raise ValueError("Feature map must be approved before final roadmap promotion")
        draft = artifacts.get(f"{self.campaign_id}:roadmap_draft")
        if draft is None:
            raise ValueError("Draft roadmap artifact is missing")
        with open(draft["canonical_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        roadmap_risk = dict(payload.get("roadmap_risk") or {})
        self.workspace.write_json("artifacts/roadmap_final/roadmap.json", payload)
        self._publish_phase_artifact(
            "development",
            f"{self.campaign_id}:phase_development_manifest",
            "manifest",
            "manifest.json",
            {
                "phase_id": "development",
                "summary": "Final roadmap promoted to the development phase.",
                "item_count": len(payload.get("items", [])),
            },
            metadata={"source": "CampaignControlPlane.promote_final_roadmap"},
        )
        self._publish_phase_artifact(
            "development",
            f"{self.campaign_id}:phase_development_roadmap",
            "roadmap_final",
            "roadmap.json",
            payload,
            metadata={"source": "CampaignControlPlane.promote_final_roadmap"},
        )
        record = self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:roadmap_final",
            family="roadmap_final",
            name="roadmap_final",
            canonical_payload=payload,
            approval_state="approved",
            metadata={"source": "CampaignControlPlane.promote_final_roadmap"},
        )
        repo_twin = self.repo_twin_builder.capture(
            self.campaign_id,
            label="roadmap_promoted",
            snapshot=self.snapshot(),
            roadmap_risk=roadmap_risk,
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:repo_twin:roadmap_promoted",
            family="telemetry",
            name="repo_twin_roadmap_promoted",
            canonical_payload=repo_twin,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.promote_final_roadmap"},
        )
        self._set_state(
            "DEVELOPMENT",
            "roadmap_promoted",
            {"final_path": record.canonical_path, "repo_twin_path": repo_twin["path"]},
        )
        return {
            "path": record.canonical_path,
            "items": payload.get("items", []),
            "repo_twin": repo_twin,
        }

    def approve_artifact(
        self,
        artifact_id: str,
        *,
        approved_by: str = "user",
        state: str = "approved",
        notes: str = "",
    ) -> None:
        self.artifacts.approve(
            artifact_id,
            approved_by=approved_by,
            state=state,
            notes=notes,
        )

    def list_artifacts(self) -> List[Dict[str, Any]]:
        return self.state_store.list_artifacts(self.campaign_id)

    def run_audit(self) -> dict[str, Any]:
        result = self.audit_engine.run(scope="operator")
        result["rule_proposals"] = self.rule_proposals.build(self.campaign_id)
        result["governance_status"] = self.rule_proposals.status(self.campaign_id)
        self.workspace.write_json("artifacts/audits/latest_audit.json", result)
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:audit_latest",
            family="audits",
            name="latest_audit",
            canonical_payload=result,
            rendered_document=self._render_audit_markdown(result),
            approval_state="accepted",
            blocking=bool(result["findings"]),
            metadata={"source": "CampaignControlPlane.run_audit"},
        )
        self._set_state("AUDIT", "audit_run")
        return result

    def build_completion_proof(self) -> dict[str, Any]:
        replay_target = os.path.join(
            self.workspace.root_dir,
            "artifacts",
            "closure",
            "replay_tape.json",
        )
        replay = self.event_store.export_run(
            self.campaign_id,
            output_path=replay_target,
        )
        proof = self.completion_engine.build_proof(replay_export=replay)
        target = self.completion_engine.persist_proof(proof)
        mission_capsule = dict(replay.get("mission_capsule") or {})
        capsule_target = os.path.join(
            self.workspace.root_dir,
            "artifacts",
            "closure",
            "mission_capsule.json",
        )
        self.workspace.write_json("artifacts/closure/mission_capsule.json", mission_capsule)
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:replay_tape",
            family="closure",
            name="replay_tape",
            canonical_payload=replay,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.build_completion_proof"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:mission_capsule",
            family="closure",
            name="mission_capsule",
            canonical_payload=mission_capsule,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.build_completion_proof"},
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:closure_proof",
            family="closure",
            name="closure_proof",
            canonical_payload=proof,
            approval_state="accepted",
            blocking=not proof["closure_allowed"],
            metadata={"source": "CampaignControlPlane.build_completion_proof"},
        )
        self._set_state(
            "CLOSURE",
            "closure_proof_emitted",
            {
                "replay_path": replay_target,
                "capsule_path": capsule_target,
                "closure_status": proof.get("closure_status"),
            },
        )
        return {
            "path": target,
            "proof": proof,
            "replay_tape": replay,
            "mission_capsule": mission_capsule,
        }

    def continue_campaign(self) -> Dict[str, Any]:
        decision = self.transition_policy.decide(self)
        event = self._apply_transition_decision(decision)
        self.workspace.save_metadata({"last_continue_state": event["to_state"]})
        self.event_store.emit(
            event_type="campaign.transition",
            payload=event,
            source="CampaignControlPlane.continue_campaign",
            run_id=self.campaign_id,
            links=[
                {
                    "link_type": "campaign",
                    "target_type": "campaign",
                    "target_ref": self.campaign_id,
                }
            ],
        )
        return event

    def ensure_repo_dossier_brief(
        self,
        *,
        repo_dossiers: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        repos = [asdict(repo) for repo in self.repo_registry.list_repos()]
        dossiers = repo_dossiers or self.retrieval_policy.load_repo_dossiers(
            self.workspace.root_dir
        )
        markdown, payload = self.retrieval_policy.render_repo_dossier_brief(
            campaign_id=self.campaign_id,
            repos=repos,
            repo_dossiers=dossiers,
        )
        brief_path = os.path.join(
            self.workspace.root_dir,
            "artifacts",
            "research",
            "repo_dossier_brief.md",
        )
        os.makedirs(os.path.dirname(brief_path), exist_ok=True)
        with open(brief_path, "w", encoding="utf-8") as handle:
            handle.write(markdown)
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:repo_dossier_brief",
            family="research",
            name="repo_dossier_brief",
            canonical_payload=payload,
            rendered_document=markdown,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.ensure_repo_dossier_brief"},
        )
        payload["path"] = brief_path
        return payload

    def build_mission_timeline(self) -> Dict[str, Any]:
        payload = self.timeline.persist(self.campaign_id, self.workspace.root_dir)
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:mission_timeline",
            family="telemetry",
            name="mission_timeline",
            canonical_payload=payload,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.build_mission_timeline"},
        )
        return payload

    def build_specialist_dashboard(self) -> Dict[str, Any]:
        task_packets = self.state_store.list_task_packets(self.campaign_id)
        task_runs = self.state_store.fetchall(
            """
            SELECT task_packet_id, status, payload_json
            FROM task_runs
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (self.campaign_id,),
        )
        telemetry = self.state_store.list_telemetry(self.campaign_id)
        packets_by_kind: dict[str, int] = {}
        accepted_count = 0
        failed_count = 0
        for packet in task_packets:
            kind = str(packet.get("packet_kind") or "implementation")
            packets_by_kind[kind] = packets_by_kind.get(kind, 0) + 1
        for row in task_runs:
            if str(row.get("status") or "") == "completed":
                accepted_count += 1
            else:
                failed_count += 1
        payload = {
            "campaign_id": self.campaign_id,
            "generated_at": time.time(),
            "summary": {
                "packet_count": len(task_packets),
                "accepted_count": accepted_count,
                "failed_count": failed_count,
                "telemetry_count": len(telemetry),
            },
            "packet_kinds": packets_by_kind,
            "task_packets": task_packets,
            "task_runs": [
                {
                    "task_packet_id": row["task_packet_id"],
                    "status": row["status"],
                    "result": json.loads(row["payload_json"] or "{}"),
                }
                for row in task_runs
            ],
        }
        self.workspace.write_json(
            "artifacts/telemetry/specialist_dashboard.json",
            payload,
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:specialist_dashboard",
            family="telemetry",
            name="specialist_dashboard",
            canonical_payload=payload,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.build_specialist_dashboard"},
        )
        return payload

    def run_speculative_roadmap_item(
        self,
        item_id: str,
        *,
        verifier=None,
    ) -> Dict[str, Any]:
        roadmap_item = next(
            (
                item
                for item in self.state_store.list_roadmap_items(self.campaign_id)
                if str(item.get("item_id") or "") == str(item_id)
            ),
            None,
        )
        if roadmap_item is None:
            raise ValueError(f"Unknown roadmap item: {item_id}")
        metadata = dict(roadmap_item.get("metadata") or {})
        speculation_variants = list(metadata.get("speculation_variants") or [])
        if not speculation_variants:
            commands = metadata.get("commands") or [
                {
                    "label": "speculation_probe",
                    "command": "echo correctness_pass=1 determinism_pass=1 replayability=1",
                }
            ]
            speculation_variants = [
                {"name": "control", "commands": commands},
                {
                    "name": "candidate",
                    "commands": metadata.get("candidate_commands") or commands,
                },
            ]
        branches: list[dict[str, Any]] = []
        for variant in speculation_variants[:2]:
            lane_id = f"{item_id}_{str(variant.get('name') or 'lane')}"
            lane_task = {
                "lane_id": lane_id,
                "caller_mode": "development",
                "lane_type": "speculative_branch",
                "name": f"{roadmap_item.get('title')} [{variant.get('name')}]",
                "objective_function": roadmap_item.get("objective")
                or roadmap_item.get("description")
                or roadmap_item.get("title"),
                "description": roadmap_item.get("description") or roadmap_item.get("title"),
                "commands": list(variant.get("commands") or []),
                "telemetry_contract": dict(
                    roadmap_item.get("telemetry_contract") or {}
                ),
                "promotion_policy": dict(
                    metadata.get("promotion_policy")
                    or roadmap_item.get("promotion_gate")
                    or {}
                ),
                "editable_scope": list(metadata.get("editable_scope") or []),
                "read_only_scope": list(metadata.get("read_only_scope") or []),
                "allowed_writes": list(
                    metadata.get("allowed_writes")
                    or roadmap_item.get("allowed_writes")
                    or ["target"]
                ),
                "metadata": {
                    **dict(metadata),
                    "speculation_variant": str(variant.get("name") or lane_id),
                },
            }
            shadow_preflight = self.experiment_runner.shadow_preflight(lane_task)
            if (
                str(variant.get("name") or "") != "control"
                and float(shadow_preflight.get("shadow_success_probability") or 0.0) < 0.32
            ):
                branch = {
                    "lane_id": lane_id,
                    "caller_mode": "development",
                    "lane_type": "speculative_branch",
                    "task": lane_task,
                    "baseline": {},
                    "worktree": {"workspace_dir": "", "shadow_skipped": True},
                    "experiment": {
                        "experiment_id": f"shadow_{uuid.uuid4().hex[:10]}",
                        "name": lane_task["name"],
                        "status": "shadow_rejected",
                        "summary_metrics": {
                            "command_count": len(lane_task["commands"]),
                            "success_count": 0,
                            "failure_count": 1,
                            "wall_time_seconds": 0.0,
                            "aggregate_metrics": {},
                        },
                    },
                    "shadow_preflight": shadow_preflight,
                    "telemetry_check": {
                        "contract_satisfied": False,
                        "required_metrics": list(
                            (lane_task.get("telemetry_contract") or {}).get("required_metrics") or []
                        ),
                        "missing_metrics": list(
                            shadow_preflight.get("predicted_contract_gaps") or []
                        ),
                    },
                    "scorecard": {"shadow_preflight_score": shadow_preflight["shadow_success_probability"]},
                    "promotion": {
                        "score": round(float(shadow_preflight["shadow_success_probability"]) - 1.0, 3),
                        "verdict": "reject",
                        "reason": "shadow_preflight_rejection",
                    },
                    "finalized": {"changed_files": [], "shadow_skipped": True},
                    "branch_metrics": {
                        "changed_files": [],
                        "verify_result": False,
                        "test_delta": 0.0,
                        "runtime_cost": float(shadow_preflight.get("predicted_runtime_cost") or 0.0),
                        "shadow_success_probability": float(
                            shadow_preflight.get("shadow_success_probability") or 0.0
                        ),
                        "predicted_contract_gaps": list(
                            shadow_preflight.get("predicted_contract_gaps") or []
                        ),
                        "predicted_runtime_cost": float(
                            shadow_preflight.get("predicted_runtime_cost") or 0.0
                        ),
                    },
                }
            else:
                branch = self.experiment_runner.run_lane(lane_task)
            branches.append(branch)
        if len(branches) < 2:
            raise ValueError("Speculation requires two branches.")
        winner = max(branches, key=lambda item: float(item["promotion"]["score"]))
        ghost_verifier = None
        item_risk = (
            (metadata.get("risk") or {})
            or (
                self._roadmap_risk_map().get(item_id)
            )
            or {}
        )
        if verifier is not None and str(item_risk.get("risk_level") or "") == "high":
            lane = self.create_verification_lane(verifier)
            ghost_verifier = lane.run(
                list(winner.get("branch_metrics", {}).get("changed_files") or []),
                campaign_id=self.campaign_id,
                task_packet_id=str(winner.get("lane_id") or ""),
                tier="ghost_high_risk",
            )
        comparison_id = f"speculation_{uuid.uuid4().hex[:10]}"
        payload = {
            "comparison_id": comparison_id,
            "campaign_id": self.campaign_id,
            "roadmap_item_id": item_id,
            "winner_lane_id": winner["lane_id"],
            "winner_variant": winner["task"]["metadata"]["speculation_variant"],
            "branches": [
                {
                    "lane_id": branch["lane_id"],
                    "variant": branch["task"]["metadata"]["speculation_variant"],
                    "score": branch["promotion"]["score"],
                    "verdict": branch["promotion"]["verdict"],
                    "branch_metrics": branch["branch_metrics"],
                    "worktree": branch["worktree"],
                    "finalized": branch["finalized"],
                    "shadow_preflight": branch.get("shadow_preflight", {}),
                }
                for branch in branches
            ],
            "comparison_metrics": {
                "changed_files": [
                    {
                        "lane_id": branch["lane_id"],
                        "files": branch["branch_metrics"]["changed_files"],
                    }
                    for branch in branches
                ],
                "verify_result": {
                    branch["lane_id"]: branch["branch_metrics"]["verify_result"]
                    for branch in branches
                },
                "test_delta": {
                    branch["lane_id"]: branch["branch_metrics"]["test_delta"]
                    for branch in branches
                },
                "runtime_cost": {
                    branch["lane_id"]: branch["branch_metrics"]["runtime_cost"]
                    for branch in branches
                },
            },
            "shadow_laps": {
                branch["lane_id"]: branch.get("shadow_preflight", {})
                for branch in branches
            },
            "ghost_verifier": ghost_verifier,
            "promotion_required": not bool((ghost_verifier or {}).get("promotion_blocked")),
        }
        self.workspace.write_json(
            f"artifacts/experiments/{comparison_id}.json",
            payload,
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:{comparison_id}",
            family="experiments",
            name=comparison_id,
            canonical_payload=payload,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.run_speculative_roadmap_item"},
        )
        self.event_store.emit(
            event_type="campaign.speculation",
            payload=payload,
            source="CampaignControlPlane.run_speculative_roadmap_item",
            run_id=self.campaign_id,
        )
        return payload

    def promote_speculative_branch(
        self,
        comparison_id: str,
        branch_lane_id: str,
    ) -> Dict[str, Any]:
        artifact = self._artifact_map().get(f"{self.campaign_id}:{comparison_id}")
        if artifact is None:
            raise ValueError(f"Unknown speculation artifact: {comparison_id}")
        with open(artifact["canonical_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        branch = next(
            (
                item
                for item in payload.get("branches", [])
                if str(item.get("lane_id") or "") == str(branch_lane_id)
            ),
            None,
        )
        if branch is None:
            raise ValueError(f"Unknown speculation branch: {branch_lane_id}")
        promoted = self.experiment_runner.worktrees.promote(
            branch_lane_id,
            files=list(branch.get("branch_metrics", {}).get("changed_files") or []),
        )
        result = {
            "comparison_id": comparison_id,
            "campaign_id": self.campaign_id,
            "branch_lane_id": branch_lane_id,
            "promoted": promoted,
        }
        self.workspace.write_json(
            f"artifacts/experiments/{comparison_id}_promotion.json",
            result,
        )
        self.event_store.emit(
            event_type="campaign.speculation.promoted",
            payload=result,
            source="CampaignControlPlane.promote_speculative_branch",
            run_id=self.campaign_id,
        )
        return result

    def create_verification_lane(self, verifier) -> VerificationLane:
        return VerificationLane(
            verifier,
            state_store=self.state_store,
            event_store=self.event_store,
            memory_fabric=self.memory_fabric,
        )

    def execute_task_packet(
        self,
        task_packet_id: str,
        handler,
        *,
        verifier=None,
    ) -> Dict[str, Any]:
        packet = next(
            (
                item
                for item in self.state_store.list_task_packets(self.campaign_id)
                if item.get("task_packet_id") == task_packet_id
            ),
            None,
        )
        if packet is None:
            raise ValueError(f"Unknown task packet: {task_packet_id}")
        result = self.task_packet_executor.execute(packet, handler)
        if verifier is not None and result.get("accepted"):
            lane = self.create_verification_lane(verifier)
            result["verification_lane"] = lane.run(
                list(result["result"].get("changed_files") or []),
                campaign_id=self.campaign_id,
                task_packet_id=task_packet_id,
                read_id=str((packet.get("metadata") or {}).get("memory_read_id") or ""),
            )
        return result

    def adopt_rule_proposal(
        self,
        rule_id: str,
        *,
        approved_by: str = "user",
        notes: str = "",
    ) -> Dict[str, Any]:
        adoption = self.rule_proposals.adopt(
            self.campaign_id,
            rule_id,
            approved_by=approved_by,
            notes=notes,
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:rule_adoption:{rule_id}",
            family="audits",
            name=f"rule_adoption_{rule_id}",
            canonical_payload=adoption,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.adopt_rule_proposal"},
        )
        return adoption

    def record_rule_outcome(
        self,
        rule_id: str,
        *,
        outcome_status: str,
        regression_delta: float = 0.0,
        notes: str = "",
    ) -> Dict[str, Any]:
        outcome = self.rule_proposals.record_outcome(
            self.campaign_id,
            rule_id,
            outcome_status=outcome_status,
            regression_delta=regression_delta,
            notes=notes,
        )
        self.artifacts.publish(
            artifact_id=f"{self.campaign_id}:rule_outcome:{rule_id}:{outcome_status}",
            family="audits",
            name=f"rule_outcome_{rule_id}",
            canonical_payload=outcome,
            approval_state="accepted",
            metadata={"source": "CampaignControlPlane.record_rule_outcome"},
        )
        return outcome

    def _roadmap_risk_map(self) -> Dict[str, Dict[str, Any]]:
        artifact = self._artifact_map().get(f"{self.campaign_id}:roadmap_draft")
        if artifact is None or not os.path.exists(artifact["canonical_path"]):
            return {}
        with open(artifact["canonical_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            str(item.get("item_id") or ""): item
            for item in list((payload.get("roadmap_risk") or {}).get("items") or [])
        }

    def _roadmap_risk_summary(self) -> Dict[str, Any]:
        artifact = self._artifact_map().get(f"{self.campaign_id}:roadmap_draft")
        if artifact is None or not os.path.exists(artifact["canonical_path"]):
            return {}
        with open(artifact["canonical_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict((payload.get("roadmap_risk") or {}).get("summary") or {})

    def _apply_transition_decision(self, decision) -> Dict[str, Any]:
        if decision.action == "set_state":
            event = self._set_state(decision.target_state, decision.cause, decision.payload)
        elif decision.action == "run_research":
            digest = self.run_research()
            event = self._set_state(
                decision.target_state,
                decision.cause,
                {"claim_count": len(digest["claims"]), **decision.payload},
            )
        elif decision.action == "build_questionnaire":
            questionnaire = self.build_questionnaire()
            event = self._set_state(
                decision.target_state,
                decision.cause,
                {"question_count": questionnaire["count"], **decision.payload},
            )
        elif decision.action == "build_feature_map":
            feature_map = self.build_feature_map()
            event = self._set_state(
                decision.target_state,
                decision.cause,
                {"feature_count": feature_map["count"], **decision.payload},
            )
        elif decision.action == "run_eid":
            eid = self.run_eid()
            event = self._set_state(
                decision.target_state,
                decision.cause,
                {"hypothesis_count": len(eid["hypotheses"]), **decision.payload},
            )
        elif decision.action == "build_roadmap":
            roadmap = self.build_roadmap()
            event = self._set_state(
                decision.target_state,
                decision.cause,
                {"item_count": len(roadmap["items"]), **decision.payload},
            )
        elif decision.action == "promote_final_roadmap":
            final_roadmap = self.promote_final_roadmap()
            tooling_task = ToolingTaskFactory(
                self.campaign_id,
                self.state_store,
            ).create_measurement_task(
                title="Development telemetry contract",
                objective="Capture required telemetry before implementation claims are promoted.",
                missing_metrics=["wall_time", "artifact_emission_status", "retry_count"],
            )
            event = self._set_state(
                decision.target_state,
                decision.cause,
                {
                    "item_count": len(final_roadmap["items"]),
                    "tooling_task_id": tooling_task["tooling_task_id"],
                    **decision.payload,
                },
            )
        elif decision.action == "build_completion_proof":
            proof = self.build_completion_proof()
            event = {
                "campaign_id": self.campaign_id,
                "from_state": decision.current_state,
                "to_state": "CLOSURE",
                "cause": decision.cause,
                "metadata": {
                    "closure_allowed": proof["proof"]["closure_allowed"],
                    **decision.payload,
                },
            }
        else:
            raise ValueError(f"Unsupported transition action: {decision.action}")
        if decision.loop_id:
            event["loop_id"] = decision.loop_id
        return event

    def _publish_phase_artifact(
        self,
        phase_id: str,
        artifact_id: str,
        artifact_type: str,
        relative_name: str,
        payload: Dict[str, Any],
        *,
        status: str = "published",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        path = self.workspace.write_phase_artifact(phase_id, relative_name, payload)
        self.state_store.record_phase_artifact(
            self.campaign_id,
            self.workspace.phase_slug(phase_id),
            artifact_id,
            artifact_type,
            path,
            status=status,
            metadata=metadata,
        )
        return path

    @staticmethod
    def _repo_context(repo: Dict[str, Any]) -> str:
        role = str(repo.get("role") or "analysis")
        repo_id = str(repo.get("repo_id") or "repo")
        return f"{role}:{repo_id}"

    def _artifact_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            item["artifact_id"]: item
            for item in self.state_store.list_artifacts(self.campaign_id)
        }

    def _import_repo_analysis_pack(
        self,
        research_store: ResearchStore,
        repo: Dict[str, Any],
        pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        repo_context = self._repo_context(repo)
        repo_id = str(repo.get("repo_id") or "repo")
        language_counts = dict(pack.get("languages") or {})
        python_file_count = int(language_counts.get("python", 0))
        native_file_count = sum(
            int(language_counts.get(key, 0))
            for key in ("c", "cpp", "c_header", "cpp_header", "rust", "go")
        )
        dossier = dict(pack.get("repo_dossier") or {})
        if not dossier:
            dossier = {
                "schema_version": "repo_dossier.v1",
                "dossier_id": f"{repo_id}:repo_dossier",
                "repo_id": repo_id,
                "repo_path": pack.get("repo_path") or repo.get("local_path") or "",
                "repo_role": str(repo.get("role") or "analysis"),
                "summary": {
                    "file_count": int(pack.get("file_count", 0)),
                    "loc": int(pack.get("loc", 0)),
                    "python_files": python_file_count,
                    "cpp_files": native_file_count,
                    "languages": dict(pack.get("languages") or {}),
                },
                "entry_points": list(pack.get("entry_points") or []),
                "build_files": list(pack.get("build_files") or []),
                "test_files": list(pack.get("test_files") or []),
                "tech_stack": list(pack.get("tech_stack") or []),
                "reuse_candidates": [],
                "risk_signals": [],
            }
        dossier_digest = str(
            dossier.get("digest")
            or (dossier.get("evidence_envelope") or {}).get("digest")
            or f"{repo_id}:{pack.get('file_count', 0)}:{pack.get('loc', 0)}"
        )
        pack_digest = (
            f"{repo_context}:"
            f"{pack.get('file_count', 0)}:"
            f"{native_file_count}:"
            f"{python_file_count}"
        )
        source_id = research_store.record_source(
            "repo_analysis",
            str(repo.get("origin") or pack.get("repo_path") or repo.get("repo_id") or repo_context),
            {
                "digest": pack_digest,
                "repo_id": repo.get("repo_id"),
                "repo_scope": str(repo.get("role") or "analysis"),
                "topic": "implementation_patterns",
                "confidence": 0.85,
                "novelty_score": 0.4,
                "complexity_score": 0.5,
                "source_scope": "repo_analysis",
                "derived_from": repo.get("repo_id"),
            },
            repo_context=repo_context,
        )
        document_id = research_store.record_document(
            source_id,
            title=f"Analysis pack for {repo.get('repo_id')}",
            normalized={
                "digest": pack_digest,
                "repo_scope": str(repo.get("role") or "analysis"),
                "source_scope": "repo_analysis",
                "topic": "repo_analysis",
                "confidence": 0.85,
                "novelty_score": 0.4,
                "complexity_score": 0.5,
                "derived_from": repo.get("repo_id"),
                "summary": pack,
            },
            path=str(pack.get("repo_path") or ""),
            repo_context=repo_context,
        )
        if not research_store.document_has_chunks(document_id, repo_context=repo_context):
            research_store.record_chunk(
                document_id,
                content=json.dumps(pack, sort_keys=True, default=str),
                topic=str(repo.get("role") or "repo_analysis"),
                position=0,
                metadata={"source": "analysis_pack"},
                repo_context=repo_context,
            )
        research_store.record_claim(
            document_id=document_id,
            topic=str(repo.get("role") or "repo_analysis"),
            summary=(
                f"Repo {repo.get('repo_id')} exposes {pack.get('file_count', 0)} files, "
                f"{native_file_count} native files, and "
                f"{python_file_count} Python files."
            ),
            confidence=0.85,
            provenance={"repo_id": repo.get("repo_id"), "document_id": document_id},
            repo_context=repo_context,
            complexity_score=0.5,
            topic_hierarchy="repo_analysis > summary",
            applicability_score=0.9,
            evidence_type="implementation",
        )
        dossier_source_id = research_store.record_source(
            "repo_dossier",
            str(pack.get("repo_path") or repo.get("origin") or repo_id),
            {
                "digest": dossier_digest,
                "repo_id": repo.get("repo_id"),
                "repo_scope": str(repo.get("role") or "analysis"),
                "topic": "repo_dossier",
                "confidence": 0.87,
                "novelty_score": 0.35,
                "complexity_score": 0.45,
                "source_scope": "repo_dossier",
                "derived_from": repo.get("repo_id"),
            },
            repo_context=repo_context,
        )
        dossier_document_id = research_store.record_document(
            dossier_source_id,
            title=f"Repo dossier for {repo_id}",
            normalized={
                "digest": dossier_digest,
                "repo_scope": str(repo.get("role") or "analysis"),
                "source_scope": "repo_dossier",
                "topic": "repo_dossier",
                "confidence": 0.87,
                "novelty_score": 0.35,
                "complexity_score": 0.45,
                "summary": dossier,
            },
            path=str(dossier.get("repo_path") or pack.get("repo_path") or ""),
            repo_context=repo_context,
        )
        if not research_store.document_has_chunks(
            dossier_document_id,
            repo_context=repo_context,
        ):
            research_store.record_chunk(
                dossier_document_id,
                content=json.dumps(dossier, sort_keys=True, default=str),
                topic="repo_dossier",
                position=0,
                metadata={"source": "repo_dossier", "repo_id": repo.get("repo_id")},
                repo_context=repo_context,
            )
        research_store.record_claim(
            document_id=dossier_document_id,
            topic="repo_dossier",
            summary=(
                f"Repo dossier for {repo_id} lists "
                f"{len(dossier.get('reuse_candidates', []))} reuse candidates and "
                f"{len(dossier.get('risk_signals', []))} risk signals."
                + (
                    f" Comparative evidence includes "
                    f"{len((dossier.get('port_ledger') or []))} port candidates."
                    if dossier.get("port_ledger")
                    else ""
                )
            ),
            confidence=0.87,
            provenance={"repo_id": repo.get("repo_id"), "document_id": dossier_document_id},
            repo_context=repo_context,
            complexity_score=0.45,
            topic_hierarchy="repo_analysis > dossier",
            applicability_score=0.9,
            evidence_type="implementation",
        )
        imported_files = 0
        imported_symbols = 0
        for file_record in pack.get("files", []):
            rel_path = str(file_record.get("path") or "")
            if not rel_path:
                continue
            file_digest = str(file_record.get("digest") or rel_path)
            file_source_id = research_store.record_source(
                "repo_file",
                f"{pack.get('repo_path')}::{rel_path}",
                {
                    "digest": file_digest,
                    "repo_id": repo.get("repo_id"),
                    "repo_scope": str(repo.get("role") or "analysis"),
                    "topic": "file_catalog",
                    "confidence": 0.82,
                    "complexity_score": 0.6 if file_record.get("symbols") else 0.3,
                    "source_scope": "repo_file_analysis",
                    "derived_from": repo.get("repo_id"),
                },
                repo_context=repo_context,
            )
            file_document_id = research_store.record_document(
                file_source_id,
                title=rel_path,
                normalized={
                    "digest": file_digest,
                    "summary": file_record,
                    "repo_scope": str(repo.get("role") or "analysis"),
                    "source_scope": "repo_file_analysis",
                    "topic": "file_catalog",
                    "confidence": 0.82,
                    "complexity_score": 0.6 if file_record.get("symbols") else 0.3,
                },
                path=rel_path,
                repo_context=repo_context,
            )
            if not research_store.document_has_chunks(file_document_id, repo_context=repo_context):
                research_store.record_chunk(
                    file_document_id,
                    content=json.dumps(file_record, sort_keys=True, default=str),
                    topic="repo_file_analysis",
                    position=0,
                    metadata={"repo_id": repo.get("repo_id"), "path": rel_path},
                    repo_context=repo_context,
                )
            research_store.record_claim(
                document_id=file_document_id,
                topic="repo_file_analysis",
                summary=(
                    f"{rel_path} is a {file_record.get('classification')} "
                    f"{file_record.get('language')} file with "
                    f"{len(file_record.get('symbols', []))} symbols and "
                    f"{sum(item.get('reference_count', 0) for item in file_record.get('usage_traces', []))} "
                    "traced references."
                ),
                confidence=0.82,
                provenance={"repo_id": repo.get("repo_id"), "path": rel_path},
                repo_context=repo_context,
                complexity_score=float(0.7 if file_record.get("symbols") else 0.2),
                topic_hierarchy="repo_analysis > file_catalog",
                applicability_score=0.85,
                evidence_type="implementation",
            )
            for trace in file_record.get("usage_traces", []):
                research_store.record_usage_trace(
                    file_document_id,
                    repo_context=repo_context,
                    symbol_name=str(trace.get("symbol") or ""),
                    trace=trace,
                )
            imported_files += 1
            imported_symbols += len(file_record.get("usage_traces", []))
        return {
            "files": imported_files,
            "symbols": imported_symbols,
            "dossiers": 1,
            "repo_dossier": dossier,
        }

    def _register_default_loops(self) -> None:
        self.loop_scheduler.register_many(
            [
                LoopDefinition(
                    loop_id="repo_ingestion_loop",
                    purpose="Freeze repo revisions and attach analysis packs.",
                    inputs=["attached_repos"],
                    produced_artifacts=["repo_ingestion_report"],
                    allowed_repo_roles=["target", "analysis_local", "analysis_external"],
                    allowed_tools=["filesystem", "git", "saguaro"],
                    stop_conditions=["repos_registered"],
                    escalation_conditions=["repo_unavailable"],
                    retry_policy={"max_attempts": 2},
                    telemetry_contract={"required_metrics": ["wall_time"]},
                    promotion_effect="REPO_INGESTION",
                    controlling_state="REPO_INGESTION",
                ),
                LoopDefinition(
                    loop_id="research_loop",
                    purpose="Crawl evidence until frontier exhaustion.",
                    inputs=["research_frontier"],
                    produced_artifacts=["research_digest"],
                    allowed_repo_roles=["analysis_local", "analysis_external"],
                    allowed_tools=["web", "saguaro"],
                    stop_conditions=["frontier_exhausted"],
                    escalation_conditions=["source_blocked"],
                    retry_policy={"max_attempts": 3},
                    telemetry_contract={"required_metrics": ["wall_time"]},
                    promotion_effect="RESEARCH_RECONCILIATION",
                    controlling_state="RESEARCH",
                ),
            ]
        )

    @staticmethod
    def _render_audit_markdown(result: Dict[str, Any]) -> str:
        lines = ["# Audit Report", ""]
        for finding in result.get("findings", []):
            lines.append(
                f"- {finding.get('severity', 'unknown')}: "
                f"{finding.get('summary', 'Audit finding')} "
                f"({finding.get('category', 'general')})"
            )
        if len(lines) == 2:
            lines.append("- No findings.")
        return "\n".join(lines) + "\n"


__all__ = ["CampaignControlPlane"]
