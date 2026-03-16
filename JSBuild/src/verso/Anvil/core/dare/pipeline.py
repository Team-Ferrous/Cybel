"""High-level DARE orchestration pipeline."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from core.campaign.runner import CampaignRunner
from core.dare.campaign_sculptor import CampaignSculptor
from core.dare.deep_analyzer import DeepAnalyzer
from core.dare.knowledge_base import DareKnowledgeBase
from core.dare.models import (
    AnalysisFinding,
    AnalysisReport,
    ComponentSpec,
    DareState,
    InnovationProposal,
    PhaseSpec,
    RepoProfile,
    ResearchReport,
    ResearchSource,
    TechnicalRoadmap,
    utc_now_iso,
)
from core.dare.repo_ingestion import RepoIngestionEngine
from core.dare.synthesizer import NovelMethodSynthesizer
from core.dare.web_research import WebResearchEngine


class DarePipeline:
    """Orchestrate DARE analysis, research, synthesis, and campaign execution."""

    def __init__(
        self,
        root_dir: str = ".",
        console=None,
        brain: Any = None,
        ownership_registry=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.console = console
        self.brain = brain
        self.ownership_registry = ownership_registry
        self.config = dict(config or {})
        self.state_path = os.path.join(self.root_dir, ".anvil", "dare", "state.json")
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.state = self._load_state()
        self.kb = DareKnowledgeBase(self.root_dir)
        self.ingestion = RepoIngestionEngine([], self.kb)
        self.analyzer = DeepAnalyzer(self.kb)
        self.research_engine = WebResearchEngine(self.kb, workspace_root=self.root_dir)
        self.synthesizer = NovelMethodSynthesizer()
        self.sculptor = CampaignSculptor(root_dir=self.root_dir)

    def analyze(self, paths: List[str]) -> Dict[str, RepoProfile]:
        repo_paths = [os.path.abspath(path) for path in (paths or [self.root_dir])]
        repo_roles = {repo_paths[0]: "primary"} if repo_paths else {}
        self.ingestion = RepoIngestionEngine(repo_paths, self.kb, repo_roles=repo_roles)
        profiles = self.ingestion.ingest_all()
        reports = [self.analyzer.analyze(profile) for profile in profiles.values()]
        self.state.repos = repo_paths
        self.state.repo_profiles = [profile.to_dict() for profile in profiles.values()]
        self.state.analysis_reports = [report.to_dict() for report in reports]
        self._save_state()
        return profiles

    def research(self, topic: str, depth: str = "deep") -> ResearchReport:
        report = self.research_engine.research_topic(topic, depth=depth)
        self.state.research_reports.append(report.to_dict())
        self._save_state()
        return report

    def compete(self, domain: str):
        primary = self._latest_repo_profile()
        report = self.research_engine.competitive_analysis(domain, our_repo=primary)
        self.state.competitive_reports.append(report.to_dict())
        self._save_state()
        return report

    def synthesize(self, constraints: Optional[Dict[str, Any]] = None) -> TechnicalRoadmap:
        analysis = self._latest_analysis_report()
        research = self._latest_research_report()
        constraints = dict(constraints or {})
        if "objective" not in constraints:
            constraints["objective"] = (
                constraints.get("description")
                or f"DARE roadmap for {os.path.basename(analysis.repo_path if analysis else self.root_dir)}"
            )
        roadmap = self.synthesizer.synthesize(
            analysis=analysis or self._empty_analysis(),
            research=research or self._empty_research(),
            target_constraints=constraints,
        )
        self.state.roadmap = roadmap.to_dict()
        self._save_state()
        self.kb.store(
            category="synthesis",
            topic="latest_roadmap",
            content=roadmap.to_markdown(),
            source="dare-pipeline",
            confidence=roadmap.confidence,
            tags=["roadmap", "synthesis"],
            metadata=roadmap.to_dict(),
        )
        return roadmap

    def sculpt(self, description: Optional[str] = None) -> str:
        roadmap = self._latest_roadmap()
        if roadmap is None or description:
            roadmap = self.synthesize({"description": description or "Generate DARE campaign"})
        output_dir = self.config.get(
            "generated_dir",
            os.path.join(self.root_dir, ".anvil", "campaigns", "generated"),
        )
        campaign_path = self.sculptor.sculpt_campaign(roadmap, self.kb, output_dir=output_dir)
        self.state.campaign_path = campaign_path
        self._save_state()
        return campaign_path

    def run(self):
        campaign_path = self.state.campaign_path or self.sculpt()
        runner = CampaignRunner(
            brain_factory=(lambda: self.brain),
            console=self.console,
            config=self.config,
            ownership_registry=self.ownership_registry,
        )
        report = runner.run_campaign(campaign_path, root_dir=self.root_dir)
        self.state.updated_at = report.completed_at or self.state.updated_at
        self._save_state()
        return report

    def run_refinement(
        self,
        refinement_context: str,
        initial_roadmap: Optional[TechnicalRoadmap] = None,
    ) -> TechnicalRoadmap:
        roadmap = initial_roadmap or self._latest_roadmap() or self.synthesize()
        experimental_results = self.synthesizer.run_validation_experiments(
            roadmap.proposals,
            scratch_dir=os.path.join(self.root_dir, ".anvil", "dare", "experiments"),
        )
        refined = self.synthesizer.refine_roadmap(roadmap, experimental_results)
        refined.metadata["refinement_context"] = refinement_context
        self.state.roadmap = refined.to_dict()
        self._save_state()
        return refined

    def status(self) -> Dict[str, Any]:
        return {
            "repos": self.state.repos,
            "analysis_reports": len(self.state.analysis_reports),
            "research_reports": len(self.state.research_reports),
            "competitive_reports": len(self.state.competitive_reports),
            "campaign_path": self.state.campaign_path,
            "knowledge_entries": len(self.kb.list_entries(limit=1000)),
            "updated_at": self.state.updated_at,
        }

    def kb_search(self, query: str, category: Optional[str] = None) -> List[dict]:
        return [entry.to_dict() for entry in self.kb.query(query, category=category, limit=10)]

    def kb_report(self, category: Optional[str] = None) -> str:
        return self.kb.get_full_report(category=category)

    def _load_state(self) -> DareState:
        if not os.path.exists(self.state_path):
            return DareState()
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return DareState()
        return DareState(**payload)

    def _save_state(self) -> None:
        self.state.updated_at = utc_now_iso()
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(self.state.to_dict(), handle, indent=2)

    def _latest_repo_profile(self) -> Optional[RepoProfile]:
        if not self.state.repo_profiles:
            return None
        return RepoProfile(**self.state.repo_profiles[-1])

    def _latest_analysis_report(self) -> Optional[AnalysisReport]:
        if not self.state.analysis_reports:
            return None
        payload = dict(self.state.analysis_reports[-1])
        payload["findings"] = [AnalysisFinding(**item) for item in payload.get("findings", [])]
        payload["layer_analyses"] = [dict(item) for item in payload.get("layer_analyses", [])]
        from core.dare.models import LayerAnalysis

        payload["layer_analyses"] = [LayerAnalysis(**item) for item in payload.get("layer_analyses", [])]
        return AnalysisReport(**payload)

    def _latest_research_report(self) -> Optional[ResearchReport]:
        if not self.state.research_reports:
            return None
        payload = dict(self.state.research_reports[-1])
        payload["sources"] = [ResearchSource(**item) for item in payload.get("sources", [])]
        return ResearchReport(**payload)

    def _latest_roadmap(self) -> Optional[TechnicalRoadmap]:
        if not self.state.roadmap:
            return None
        payload = dict(self.state.roadmap)
        payload["proposals"] = [InnovationProposal(**item) for item in payload.get("proposals", [])]
        payload["components"] = [ComponentSpec(**item) for item in payload.get("components", [])]
        payload["phases"] = [PhaseSpec(**item) for item in payload.get("phases", [])]
        return TechnicalRoadmap(**payload)

    @staticmethod
    def _empty_analysis() -> AnalysisReport:
        return AnalysisReport(
            repo_path=".",
            summary="No repository analysis available yet.",
            architecture="unknown",
        )

    @staticmethod
    def _empty_research() -> ResearchReport:
        return ResearchReport(
            topic="unknown",
            summary="No research report available yet.",
        )
