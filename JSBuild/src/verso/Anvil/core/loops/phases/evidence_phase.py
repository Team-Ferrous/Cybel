from typing import Any, Dict, Optional
from core.loops.phases.base_phase import BasePhase
from core.utils.logger import get_logger
from rich import box
from rich.panel import Panel

logger = get_logger(__name__)


class EvidencePhase(BasePhase):
    """
    Phase 2: Evidence Gathering - Search codebase and load context.
    """

    def execute(
        self, user_input: str, context: Dict[str, Any], dashboard: Optional[Any] = None
    ) -> Dict[str, Any]:
        logger.info("Executing Evidence Phase")

        request_type = context.get("request_type")
        evidence = {
            "codebase_files": [],
            "file_contents": {},
            "web_results": [],
            "search_results": [],
            "errors": [],
            "question_type": "simple",
        }
        complexity_profile = context.get("complexity_profile")
        if complexity_profile is not None:
            evidence["complexity_profile"] = complexity_profile
            evidence["complexity_score"] = complexity_profile.score

        if dashboard:
            dashboard.update_phase(
                "Evidence Gathering",
                status="in_progress",
                message="Analyzing request...",
            )
        else:
            self.console.print("\n")
            self.console.print(
                Panel(
                    "[bold yellow]Phase 1: Evidence Gathering[/bold yellow]\n"
                    "[dim]Analyzing request, gathering context...[/dim]",
                    border_style="yellow",
                    box=box.HEAVY,
                    padding=(0, 2),
                )
            )
            self.console.print("")

        if request_type in ["question", "explanation", "investigation"]:
            # Logic from _handle_question
            question_type = self.loop._classify_question_type(user_input)
            evidence["question_type"] = question_type
            logger.info(f"Question type classified as: {question_type}")
            self.console.print(
                f"  [cyan]→ Question type:[/cyan] [bold]{question_type}[/bold]"
            )

            if question_type == "research":
                logger.info("Delegating to ResearchSubagent")
                if dashboard:
                    dashboard.add_agent(
                        "Evidence Gathering", "ResearchSubagent", status="running"
                    )
                evidence = self.loop._delegate_to_research_subagent(user_input)
                evidence["question_type"] = question_type  # Preserve for COCONUT
                if dashboard:
                    dashboard.update_agent(
                        "ResearchSubagent", status="completed", progress=1.0
                    )
            elif question_type == "architecture":
                # Architecture questions need deep repo analysis, not just search
                logger.info("Delegating to RepoAnalysisSubagent for architecture")
                if dashboard:
                    dashboard.add_agent(
                        "Evidence Gathering", "RepoAnalysisSubagent", status="running"
                    )
                evidence = self.loop._delegate_to_repo_analysis_subagent(user_input)
                evidence["question_type"] = question_type  # Preserve for COCONUT
                if self.loop.enhanced_mode and not evidence.get("file_contents"):
                    logger.info(
                        "RepoAnalysisSubagent returned no file contents; enriching architecture evidence via enhanced gather"
                    )
                    self.loop._gather_evidence_enhanced(
                        user_input, evidence, dashboard=dashboard
                    )
                if dashboard:
                    dashboard.update_agent(
                        "RepoAnalysisSubagent", status="completed", progress=1.0
                    )
            elif question_type == "investigation":
                logger.info("Gathering enhanced evidence for investigation")
                if dashboard:
                    dashboard.add_agent(
                        "Evidence Gathering", "SemanticSearch", status="running"
                    )
                if self.loop.enhanced_mode:
                    self.loop._gather_evidence_enhanced(
                        user_input, evidence, dashboard=dashboard
                    )
                else:
                    self.loop._gather_evidence_basic(user_input, evidence)
                if dashboard:
                    dashboard.update_agent(
                        "SemanticSearch", status="completed", progress=1.0
                    )
            else:
                logger.info("Gathering basic evidence (inline)")
                self.loop._gather_evidence_basic(user_input, evidence)

        elif request_type in ["modification", "creation", "deletion"]:
            # Logic from _handle_action context gathering
            logger.info(f"Gathering action context for {request_type}")
            if self.loop.enhanced_mode:
                action_context = self.loop._gather_action_context(user_input)
                evidence.update(action_context)

        if dashboard:
            dashboard.update_phase("Evidence Gathering", status="completed")

        return evidence
