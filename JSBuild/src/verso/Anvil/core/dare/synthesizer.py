"""Novel-method synthesis and roadmap construction for DARE."""

from __future__ import annotations

from typing import Dict, List

from core.dare.models import (
    AnalysisReport,
    ComponentSpec,
    ExperimentResult,
    InnovationProposal,
    PhaseSpec,
    ResearchReport,
    TechnicalRoadmap,
)


class NovelMethodSynthesizer:
    """Convert analysis and research into an executable technical roadmap."""

    def synthesize(
        self,
        analysis: AnalysisReport,
        research: ResearchReport,
        target_constraints: Dict,
    ) -> TechnicalRoadmap:
        proposals = self._build_proposals(analysis, research, target_constraints)
        components = self._build_components(analysis, research)
        phases = self._build_phases(components)
        summary = (
            f"Roadmap synthesized from {len(analysis.findings)} analysis findings and "
            f"{len(research.sources)} external sources. Focus: {target_constraints.get('language', 'mixed')} "
            f"delivery with evidence-backed rewrite sequencing."
        )
        return TechnicalRoadmap(
            objective=target_constraints.get("objective", f"Advance {analysis.repo_path} with DARE"),
            summary=summary,
            constraints=dict(target_constraints),
            proposals=proposals,
            components=components,
            phases=phases,
            evidence_topics=[analysis.repo_path, research.topic],
            confidence="medium" if research.sources else "low",
            metadata={
                "analysis_summary": analysis.summary,
                "research_summary": research.summary,
            },
        )

    def refine_roadmap(
        self,
        roadmap: TechnicalRoadmap,
        experimental_results: Dict[str, ExperimentResult],
    ) -> TechnicalRoadmap:
        confidence = "high" if experimental_results else roadmap.confidence
        refined_summary = roadmap.summary
        if experimental_results:
            refined_summary += f" Refinement pass incorporated {len(experimental_results)} validation experiments."
        roadmap.summary = refined_summary
        roadmap.confidence = confidence
        roadmap.metadata["experimental_results"] = {
            key: value.to_dict() for key, value in experimental_results.items()
        }
        return roadmap

    def run_validation_experiments(
        self,
        proposals: List[InnovationProposal],
        scratch_dir: str,
    ) -> Dict[str, ExperimentResult]:
        del scratch_dir
        results: Dict[str, ExperimentResult] = {}
        for proposal in proposals:
            results[proposal.name] = ExperimentResult(
                name=proposal.name,
                hypothesis=proposal.description,
                verdict="planned",
                notes="DARE generated an experiment plan; execution is delegated to downstream campaign phases.",
                metrics={"steps": len(proposal.experiment_plan)},
            )
        return results

    @staticmethod
    def _build_proposals(
        analysis: AnalysisReport,
        research: ResearchReport,
        constraints: Dict,
    ) -> List[InnovationProposal]:
        proposals: List[InnovationProposal] = []
        if analysis.layer_analyses:
            proposals.append(
                InnovationProposal(
                    name="Layered rewrite plan",
                    description="Decompose rewrite work per subsystem layer instead of a single framework-wide migration.",
                    rationale="Deep analysis identified concrete subsystem clusters that can be validated independently.",
                    evidence_topics=[analysis.repo_path],
                    expected_impact="Reduces blast radius and improves gate determinism.",
                    confidence="high",
                    experiment_plan=[
                        "Generate per-layer parity fixtures.",
                        "Benchmark rewritten components against baseline behavior.",
                    ],
                )
            )
        if research.gaps:
            proposals.append(
                InnovationProposal(
                    name="Gap-driven feature differentiation",
                    description="Prioritize features and experiments around externally observed gaps instead of matching existing implementations feature-for-feature.",
                    rationale="Research surfaced recurring unmet needs across external sources.",
                    evidence_topics=[research.topic],
                    expected_impact="Improves novelty and avoids commodity reimplementation work.",
                    confidence="medium",
                    experiment_plan=["Attach each selected gap to a falsifiable benchmark or acceptance test."],
                )
            )
        if constraints.get("optimization"):
            proposals.append(
                InnovationProposal(
                    name="Constraint-aligned optimization track",
                    description="Schedule optimization only after correctness and integration phases for the targeted runtime constraints.",
                    rationale=f"Constraints include {constraints.get('optimization')}, which should not displace correctness gates.",
                    evidence_topics=[analysis.repo_path, research.topic],
                    expected_impact="Prevents premature micro-optimization while keeping performance work explicit.",
                    confidence="medium",
                    experiment_plan=["Capture pre-optimization baseline telemetry.", "Run micro-benchmarks after each optimized component lands."],
                )
            )
        return proposals

    @staticmethod
    def _build_components(analysis: AnalysisReport, research: ResearchReport) -> List[ComponentSpec]:
        components: List[ComponentSpec] = []
        for layer in analysis.layer_analyses:
            components.append(
                ComponentSpec(
                    name=layer.layer_type,
                    objective=f"Stabilize and improve the {layer.layer_type} subsystem.",
                    files=layer.files,
                    acceptance_criteria=[
                        "Behavioral parity tests exist.",
                        "Performance claims are backed by benchmark artifacts.",
                    ],
                )
            )
        if not components:
            components.append(
                ComponentSpec(
                    name="core-platform",
                    objective=f"Translate research findings for '{research.topic}' into an implementation-ready campaign scaffold.",
                    acceptance_criteria=["Evidence bundle recorded in campaign ledger."],
                )
            )
        return components

    @staticmethod
    def _build_phases(components: List[ComponentSpec]) -> List[PhaseSpec]:
        phases: List[PhaseSpec] = [
            PhaseSpec(
                phase_id="hydrate_knowledge",
                name="Knowledge Hydration",
                objective="Load DARE knowledge artifacts into campaign context.",
                acceptance_criteria=["At least one knowledge artifact is available to downstream phases."],
            )
        ]
        previous = ["hydrate_knowledge"]
        for index, component in enumerate(components, start=1):
            phase_id = f"phase_{index}_{component.name.replace(' ', '_').lower()}"
            phases.append(
                PhaseSpec(
                    phase_id=phase_id,
                    name=f"{component.name.title()} Delivery",
                    objective=component.objective,
                    files=component.files,
                    depends_on=list(previous[-1:]),
                    acceptance_criteria=component.acceptance_criteria,
                )
            )
            previous = [phase_id]
        phases.append(
            PhaseSpec(
                phase_id="phase_final_verification",
                name="Final Verification",
                objective="Run AES and evidence closure gates across all generated outputs.",
                depends_on=[previous[-1]],
                acceptance_criteria=["Verification output recorded.", "Final report emitted."],
                estimated_duration="1 phase",
            )
        )
        return phases
