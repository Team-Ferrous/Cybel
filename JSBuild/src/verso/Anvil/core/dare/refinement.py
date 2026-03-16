"""Two-pass refinement support for DARE pipelines."""

from __future__ import annotations

from core.dare.models import TechnicalRoadmap


class RefinementProtocol:
    """Run DARE in an explicit refinement pass."""

    def run_with_refinement(self, dare_pipeline) -> TechnicalRoadmap:
        initial_roadmap = dare_pipeline.run()
        refinement_context = (
            "REFINEMENT MODE\n"
            "Review the initial roadmap for weak assumptions, low-confidence claims, and missing experiments."
        )
        return dare_pipeline.run_refinement(refinement_context, initial_roadmap)
