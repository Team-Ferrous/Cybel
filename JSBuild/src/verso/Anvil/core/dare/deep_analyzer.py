"""Deep repository analysis and subsystem-oriented heuristics for DARE."""

from __future__ import annotations

import os
from typing import List

from core.dare.knowledge_base import DareKnowledgeBase
from core.dare.models import AnalysisFinding, AnalysisReport, LayerAnalysis, RepoProfile


class DeepAnalyzer:
    """Multi-subsystem deep analysis of repository functionality."""

    def __init__(self, knowledge_base: DareKnowledgeBase):
        self.kb = knowledge_base

    def analyze(self, repo_profile: RepoProfile) -> AnalysisReport:
        findings: List[AnalysisFinding] = []
        opportunities: List[str] = []

        architecture = self._classify_architecture(repo_profile)
        findings.append(
            AnalysisFinding(
                category="architecture",
                title="Detected architecture profile",
                detail=f"{repo_profile.repo_name} appears to follow a {architecture} layout based on build files, module layout, and entry points.",
                confidence="medium",
                severity="low",
                evidence=repo_profile.build_files[:3] + repo_profile.entry_points[:3],
                tags=["architecture"],
            )
        )

        if not repo_profile.test_files:
            findings.append(
                AnalysisFinding(
                    category="testing",
                    title="No conventional tests detected",
                    detail="The repo ingestion pass did not find tests under standard locations. DARE should treat downstream recommendations as lower confidence until validation coverage is improved.",
                    confidence="high",
                    severity="high",
                    tags=["testing", "coverage"],
                )
            )
            opportunities.append("Create subsystem-level tests before large rewrites.")

        if repo_profile.detected_patterns.get("bare_except", 0):
            findings.append(
                AnalysisFinding(
                    category="resilience",
                    title="Bare exception handlers present",
                    detail="Broad exception handling was detected. This can obscure failure modes in autonomous execution loops and should be narrowed.",
                    confidence="high",
                    severity="medium",
                    tags=["error-handling"],
                )
            )
            opportunities.append("Replace bare exceptions with typed failure handling on critical paths.")

        layer_analyses = self.analyze_model_layers(repo_profile)
        if layer_analyses:
            opportunities.append("Use per-layer rewrite planning instead of monolithic refactors.")

        summary = self._make_summary(repo_profile, architecture, findings, opportunities, layer_analyses)
        report = AnalysisReport(
            repo_path=repo_profile.repo_path,
            summary=summary,
            architecture=architecture,
            findings=findings,
            layer_analyses=layer_analyses,
            opportunities=opportunities,
            metadata=repo_profile.to_dict(),
        )
        self.kb.store(
            category="analysis",
            topic=f"{repo_profile.repo_name}_deep_analysis",
            content=self._report_to_markdown(report),
            source="deep-analyzer",
            confidence="medium",
            tags=["analysis", architecture],
            dependencies=[
                os.path.join(self.kb.kb_dir, "analysis", f"{self.kb._slugify(repo_profile.repo_name + '_ingestion')}.md")
            ],
            metadata=report.to_dict(),
        )
        return report

    def analyze_model_layers(self, repo_profile: RepoProfile) -> List[LayerAnalysis]:
        file_names = list(repo_profile.dependency_graph.keys()) + repo_profile.build_files + repo_profile.entry_points
        rules = {
            "attention": ("attention", "attn"),
            "ffn": ("ffn", "feed_forward", "mlp"),
            "normalization": ("norm", "rmsnorm", "layernorm"),
            "embedding": ("embed", "tokenizer"),
            "state-space": ("ssm", "mamba", "state_space"),
        }
        analyses: List[LayerAnalysis] = []
        for layer_type, markers in rules.items():
            matches = [path for path in file_names if any(marker in path.lower() for marker in markers)]
            if not matches:
                continue
            opportunities: List[str] = []
            if layer_type in {"attention", "ffn"}:
                opportunities.append("Benchmark fusion and vectorization opportunities before rewrite.")
            if layer_type == "embedding":
                opportunities.append("Validate tokenizer and embedding parity with corpus-based fixtures.")
            analyses.append(
                LayerAnalysis(
                    layer_type=layer_type,
                    files=sorted(set(matches))[:10],
                    opportunities=opportunities,
                    notes=[f"Detected {len(matches)} files touching {layer_type} concerns."],
                    evidence=sorted(set(matches))[:5],
                )
            )
        return analyses

    @staticmethod
    def _classify_architecture(repo_profile: RepoProfile) -> str:
        if "cmake" in repo_profile.tech_stack and "python" in repo_profile.tech_stack:
            return "hybrid-python-native"
        if "python" in repo_profile.tech_stack:
            return "python-application"
        if "cmake" in repo_profile.tech_stack:
            return "native-library"
        return "mixed-repository"

    @staticmethod
    def _make_summary(
        repo_profile: RepoProfile,
        architecture: str,
        findings: List[AnalysisFinding],
        opportunities: List[str],
        layer_analyses: List[LayerAnalysis],
    ) -> str:
        return (
            f"{repo_profile.repo_name} is a {architecture} repo with {repo_profile.file_count} files and "
            f"{repo_profile.loc} non-empty lines. The deep analysis pass produced {len(findings)} findings, "
            f"{len(opportunities)} rewrite opportunities, and {len(layer_analyses)} model-layer slices."
        )

    @staticmethod
    def _report_to_markdown(report: AnalysisReport) -> str:
        lines = [
            f"# Deep Analysis",
            "",
            report.summary,
            "",
            f"- Architecture: {report.architecture}",
            "",
            "## Findings",
        ]
        for finding in report.findings:
            lines.append(f"- **{finding.title}** ({finding.category}, {finding.confidence}): {finding.detail}")
        lines.extend(["", "## Opportunities"])
        lines.extend([f"- {item}" for item in report.opportunities] or ["- none"])
        if report.layer_analyses:
            lines.extend(["", "## Layer Analysis"])
            for item in report.layer_analyses:
                lines.append(f"### {item.layer_type}")
                lines.append(f"- Files: {', '.join(item.files)}")
                lines.extend([f"- Opportunity: {opportunity}" for opportunity in item.opportunities] or ["- Opportunity: none"])
        return "\n".join(lines).rstrip() + "\n"
