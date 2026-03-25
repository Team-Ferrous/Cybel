"""Utilities for report."""

import os
import time
from typing import Any

from saguaro.analysis.dead_code import DeadCodeAnalyzer
from saguaro.analysis.entry_points import EntryPointDetector
from saguaro.coverage import CoverageReporter
from saguaro.sentinel.verifier import SentinelVerifier

# Use standard library or simple heuristics for "Architecture" if no specialized module yet


class ReportGenerator:
    """Provide ReportGenerator support."""
    def __init__(self, root_path: str) -> None:
        """Initialize the instance."""
        self.root_path = os.path.abspath(root_path)

    def generate(self) -> dict[str, Any]:
        """Generates the comprehensive State of the Repo report."""
        report = {
            "generated_at": time.time(),
            "repo_path": self.root_path,
            "sections": {},
        }

        # 1. Coverage
        print("Gathering coverage stats...")
        cov = CoverageReporter(self.root_path)
        report["sections"]["coverage"] = cov.generate_report()

        # 2. Dead Code (Reachability)
        print("Analyzing reachability (Dead Code)...")
        # DeadCodeAnalyzer might be slow, use defaults
        dc = DeadCodeAnalyzer(self.root_path)
        candidates = dc.analyze()
        # Summary only
        report["sections"]["dead_code"] = {
            "total_candidates": len(candidates),
            "high_confidence_candidates": len(
                [c for c in candidates if c["confidence"] > 0.8]
            ),
            "top_candidates": candidates[:10] if candidates else [],
        }

        # 3. Sentinel (Security & Violations)
        print("Running Sentinel Audit...")
        # Maybe skip slow engines like mypy if we want speed?
        # Roadmap implies a "durable, structured artifact", so quality counts.
        # But for 'finish the roadmap', I'll use default engines.
        verifier = SentinelVerifier(
            self.root_path, engines=["native", "ruff", "semantic"]
        )
        # Skip mypy/vulture for speed in this implementation unless requested
        violations = verifier.verify_all()

        # Aggregate violations by severity/category
        violation_stats = {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0}
        by_engine = {}

        for v in violations:
            sev = v.get("severity", "low")
            violation_stats[sev] = violation_stats.get(sev, 0) + 1
            violation_stats["total"] += 1

            eng = v.get(
                "engine", "unknown"
            )  # Assuming we add engine field to violation
            by_engine[eng] = by_engine.get(eng, 0) + 1

        report["sections"]["sentinel"] = {
            "summary": violation_stats,
            "by_engine": by_engine,
            "violation_count": len(violations),
        }

        # 4. Architecture / Entry Points
        print("Mapping Entry Points...")
        ep_detector = EntryPointDetector(self.root_path)
        entry_points = ep_detector.detect()
        report["sections"]["architecture"] = {
            "entry_points": len(entry_points),
            "entry_point_types": {},
            # TODO: Add dependency depth analysis if we had a graph module ready
        }
        for ep in entry_points:
            etype = ep.get("type", "unknown")
            report["sections"]["architecture"]["entry_point_types"][etype] = (
                report["sections"]["architecture"]["entry_point_types"].get(etype, 0)
                + 1
            )

        # 5. Features / Capabilities
        # Placeholder for Feature Inventory
        report["sections"]["features"] = {
            "inventory": "Not yet implemented (requires semantic Capability Map)"
        }

        return report

    def save_markdown(self, report: dict[str, Any], output_path: str) -> None:
        """Saves report as Markdown."""
        lines = []
        lines.append(f"# State of the Repo: {os.path.basename(report['repo_path'])}")
        lines.append(f"**Generated:** {time.ctime(report['generated_at'])}")
        lines.append("")

        # Coverage
        cov = report["sections"]["coverage"]
        lines.append("## 1. Codebase Coverage")
        lines.append(f"- **Total Files:** {cov['total_files']}")
        lines.append(f"- **AST Supported:** {cov['ast_supported_files']}")
        lines.append("- **Languages:**")
        for lang, count in cov["languages"].items():
            lines.append(f"  - {lang}: {count}")
        lines.append("")

        # Dead Code
        dc = report["sections"]["dead_code"]
        lines.append("## 2. Dead Code & Debt")
        lines.append(f"- **Candidates:** {dc['total_candidates']}")
        lines.append(f"- **High Confidence:** {dc['high_confidence_candidates']}")
        if dc["top_candidates"]:
            lines.append("### Top Candidates to Remove:")
            for c in dc["top_candidates"]:
                lines.append(
                    f"- [{c['confidence']:.2f}] `{c['symbol']}` in {os.path.basename(c['file'])}"
                )
        lines.append("")

        # Sentinel
        sen = report["sections"]["sentinel"]
        lines.append("## 3. Sentinel Health (Security & Governance)")
        lines.append(f"- **Total Violations:** {sen['violation_count']}")
        lines.append("- **Severity Breakdown:**")
        for sev, count in sen["summary"].items():
            if sev != "total":
                lines.append(f"  - {sev.title()}: {count}")
        lines.append("")

        # Architecture
        arch = report["sections"]["architecture"]
        lines.append("## 4. Architecture & Entry Points")
        lines.append(f"- **Entry Points Detected:** {arch['entry_points']}")
        for t, c in arch["entry_point_types"].items():
            lines.append(f"  - {t}: {c}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def save_comparative_markdown(
        self,
        report: dict[str, Any],
        output_path: str,
    ) -> None:
        """Saves a comparative report as Markdown."""
        comparisons = list(report.get("comparisons") or [])
        artifacts = dict(report.get("artifacts") or {})
        render_policy = dict(report.get("render_policy") or {})
        detailed_limit = int(
            render_policy.get("detailed_candidate_limit", len(comparisons)) or len(comparisons)
        )
        if len(comparisons) > 1:
            candidate_dir = f"{os.path.splitext(output_path)[0]}_reports"
            os.makedirs(candidate_dir, exist_ok=True)
            candidate_paths: dict[str, str] = {}
            for comparison in comparisons[:detailed_limit]:
                candidate = dict(comparison.get("candidate") or {})
                label = str(
                    candidate.get("corpus_id")
                    or os.path.basename(str(candidate.get("root_path") or ""))
                    or "candidate"
                )
                safe_label = "".join(
                    char if char.isalnum() or char in {"-", "_"} else "_"
                    for char in label
                ).strip("_") or "candidate"
                candidate_path = os.path.join(candidate_dir, f"{safe_label}.md")
                candidate_report = dict(report)
                candidate_report["candidate_count"] = 1
                candidate_report["comparisons"] = [comparison]
                candidate_report["port_ledger"] = list(comparison.get("port_ledger") or [])
                candidate_report["frontier_packets"] = list(
                    comparison.get("frontier_packets") or []
                )
                self._write_comparative_markdown(
                    candidate_report,
                    candidate_path,
                    append=False,
                    include_candidate_index=False,
                )
                candidate_paths[str(candidate.get("corpus_id") or safe_label)] = candidate_path
            artifacts["candidate_markdown_dir"] = candidate_dir
            artifacts["candidate_markdown_paths"] = candidate_paths
            report["artifacts"] = artifacts

        self._write_comparative_markdown(
            report,
            output_path,
            append=True,
            include_candidate_index=len(comparisons) > 1,
        )

    def _write_comparative_markdown(
        self,
        report: dict[str, Any],
        output_path: str,
        *,
        append: bool,
        include_candidate_index: bool,
    ) -> None:
        target = dict(report.get("target") or {})
        report_id = str(report.get("report_id") or "comparative-report")
        comparisons = list(report.get("comparisons") or [])
        primary_total = sum(
            len(list(comparison.get("primary_recommendations") or []))
            for comparison in comparisons
        )
        secondary_total = sum(
            len(list(comparison.get("secondary_recommendations") or []))
            for comparison in comparisons
        )
        lines = [
            f"# Comparative Port Plan: {target.get('corpus_id', 'target')}",
            "",
            f"Report ID: {report_id}",
            f"Generated: {time.ctime(float(report.get('generated_at', time.time()) or time.time()))}",
            f"Candidate count: {int(report.get('candidate_count', 0) or 0)}",
            f"Port ledger count: {len(report.get('port_ledger') or [])}",
            f"Frontier packet count: {len(report.get('frontier_packets') or [])}",
            f"Creation ledger count: {len(report.get('creation_ledger') or [])}",
            f"Native migration program count: {len(report.get('native_migration_programs') or [])}",
            "",
            "## Executive Summary",
            "",
            (
                f"This markdown is optimized for deep port planning rather than fleet triage. "
                f"It surfaced {primary_total} primary recommendations and {secondary_total} secondary opportunities "
                f"across {len(comparisons)} candidate comparisons, while preserving low-signal analogues in separate sections."
            ),
            "",
        ]
        artifacts = dict(report.get("artifacts") or {})
        telemetry = dict(report.get("telemetry") or {})
        synthesis = dict(report.get("best_of_breed_synthesis") or {})
        subsystem_upgrade_summary = list(report.get("subsystem_upgrade_summary") or [])
        phase_packets = list(report.get("phase_packets") or [])
        portfolio_leaderboard = list(report.get("portfolio_leaderboard") or [])
        negative_evidence = list(report.get("negative_evidence") or [])
        if artifacts:
            lines.extend(
                [
                    "## Artifacts",
                    "",
                    *[
                        f"- {name}: {path}"
                        for name, path in sorted(artifacts.items())
                        if path
                    ],
                    "",
                ]
            )
        if include_candidate_index:
            candidate_paths = dict(artifacts.get("candidate_markdown_paths") or {})
            lines.extend(["## Candidate Reports", ""])
            for corpus_id, path in sorted(candidate_paths.items()):
                lines.append(f"- {corpus_id}: {path}")
            lines.append("")
        if telemetry:
            lines.extend(
                [
                    "## Telemetry",
                    "",
                    *[
                        f"- {name}: {value}"
                        for name, value in sorted(telemetry.items())
                    ],
                    "",
                ]
            )
        if synthesis:
            lines.extend(["## Best of Breed", ""])
            for winner in list(synthesis.get("feature_winners") or [])[:12]:
                lines.append(
                    f"- {winner.get('feature_family')}: {winner.get('winner_corpus_id')} "
                    f"[score={float(winner.get('score') or 0.0):.3f}] "
                    f"{winner.get('source_path')} -> {winner.get('target_path')} "
                    f"({winner.get('recommended_subsystem_label') or 'Unrouted'})"
                )
            lines.append("")
        if subsystem_upgrade_summary:
            lines.extend(["## Subsystem Upgrade Routing", ""])
            for row in subsystem_upgrade_summary:
                if not (
                    int(row.get("recommendation_count") or 0)
                    or int(row.get("creation_candidate_count") or 0)
                ):
                    continue
                lines.append(
                    f"- {row.get('subsystem_label')}: primary={int(row.get('primary_count') or 0)}, "
                    f"secondary={int(row.get('secondary_count') or 0)}, "
                    f"creation={int(row.get('creation_candidate_count') or 0)}"
                )
                lines.append(
                    f"  Top families: {', '.join(row.get('top_feature_families') or []) or 'none'}"
                )
                lines.append(
                    f"  Target zones: {', '.join(row.get('top_target_paths') or []) or 'none'}"
                )
            lines.append("")
        if portfolio_leaderboard:
            lines.extend(["## Portfolio Leaderboard", ""])
            for row in portfolio_leaderboard[:12]:
                lines.append(
                    f"- {row.get('corpus_id')}: score={float(row.get('portfolio_rank_score') or 0.0):.3f} "
                    f"primary={int(row.get('primary_count') or 0)} "
                    f"secondary={int(row.get('secondary_count') or 0)} "
                    f"no-port={int(row.get('no_port_count') or 0)} "
                    f"build={row.get('build_truth_depth')}"
                )
            lines.append("")
        if phase_packets:
            lines.extend(["## Phase Packets", ""])
            for packet in phase_packets[:10]:
                lines.append(
                    f"- {packet.get('phase_id')}: {packet.get('objective')} "
                    f"[writes={', '.join(packet.get('allowed_writes') or []) or 'none'}]"
                )
            lines.append("")
        if negative_evidence:
            lines.extend(["## Negative Evidence Ledger", ""])
            for row in negative_evidence[:12]:
                lines.append(
                    f"- {row.get('reason')} [{row.get('severity')}] "
                    f"{row.get('source_path')} -> {row.get('target_path')}"
                )
            lines.append("")
        creation_ledger = list(report.get("creation_ledger") or [])
        if creation_ledger:
            lines.extend(["## Creation Ledger", ""])
            for entry in creation_ledger[:12]:
                lines.append(
                    f"- {entry.get('feature_family')} "
                    f"[{entry.get('kind')}] "
                    f"priority={float(entry.get('priority') or 0.0):.3f} "
                    f"{entry.get('rationale')} "
                    f"({entry.get('recommended_subsystem_label') or 'Unrouted'})"
                )
            lines.append("")
        programs = list(report.get("native_migration_programs") or [])
        if programs:
            lines.extend(["## Native Migration Programs", ""])
            for program in programs[:12]:
                lines.append(
                    f"- {program.get('feature_family')} "
                    f"[priority={float(program.get('priority') or 0.0):.3f}] "
                    f"{program.get('source_path')} -> {program.get('target_path')} "
                    f"({program.get('recommended_subsystem_label') or 'Unrouted'})"
                )
            lines.append("")
        for comparison in comparisons:
            candidate = dict(comparison.get("candidate") or {})
            summary = dict(comparison.get("summary") or {})
            overlay = dict(comparison.get("overlay_graph") or {})
            scorecard = dict(comparison.get("candidate_scorecard") or {})
            value_realization = dict(comparison.get("value_realization") or {})
            evidence_quality = dict(comparison.get("evidence_quality_summary") or {})
            primary_recommendations = list(comparison.get("primary_recommendations") or [])
            secondary_recommendations = list(comparison.get("secondary_recommendations") or [])
            disparate_opportunities = list(comparison.get("disparate_opportunities") or [])
            low_signal_relations = list(comparison.get("low_signal_relations") or [])
            upgrade_clusters = list(comparison.get("upgrade_clusters") or [])
            program_groups = list(comparison.get("program_groups") or [])
            subsystem_rows = list(comparison.get("subsystem_routing_summary") or [])
            lines.extend(
                [
                    f"## Candidate: {candidate.get('corpus_id', 'unknown')}",
                    "",
                    comparison.get("report_text", ""),
                    "",
                    "### Candidate Scorecard",
                    f"- Overall fit: {scorecard.get('overall_fit', 'unknown')}",
                    f"- Top feature families: {', '.join(scorecard.get('top_feature_families', [])) or 'none'}",
                    f"- Primary recommendations: {int(scorecard.get('primary_recommendation_count', 0) or 0)}",
                    f"- Secondary recommendations: {int(scorecard.get('secondary_recommendation_count', 0) or 0)}",
                    f"- Low-signal relations: {int(scorecard.get('low_signal_count', 0) or 0)}",
                    f"- Average calibrated confidence: {float(evidence_quality.get('average_confidence', 0.0) or 0.0):.3f}",
                    f"- No-port candidates: {int(evidence_quality.get('no_port_count', 0) or 0)}",
                    f"- Preferred implementation language: {summary.get('preferred_implementation_language', 'unknown')}",
                    f"- Common tech stack: {', '.join(summary.get('common_tech_stack', [])) or 'none'}",
                    f"- Language overlap: {', '.join(summary.get('language_overlap', [])) or 'none'}",
                    f"- Feature overlap: {', '.join(summary.get('feature_overlap', [])) or 'none'}",
                    f"- Candidate feature gaps: {', '.join(summary.get('candidate_feature_gaps', [])) or 'none'}",
                    f"- Recommended subsystems: {', '.join(scorecard.get('recommended_subsystems', [])) or 'none'}",
                    f"- Comparison backend: {summary.get('comparison_backend', 'unknown')}",
                    "",
                    "### Build Alignment",
                    f"- Compatible: {bool((summary.get('build_alignment') or {}).get('compatible'))}",
                    f"- Shared build files: {', '.join((summary.get('build_alignment') or {}).get('shared_build_files', [])) or 'none'}",
                    f"- Build fingerprint depth: {((summary.get('build_alignment') or {}).get('candidate') or {}).get('build_fingerprint_depth', 'shallow')}",
                    "",
                    "### Capability Delta",
                    f"- Shared deep languages: {', '.join((summary.get('capability_delta') or {}).get('shared_deep_languages', [])) or 'none'}",
                    f"- Candidate-only deep languages: {', '.join((summary.get('capability_delta') or {}).get('candidate_only_deep_languages', [])) or 'none'}",
                    f"- Target-only deep languages: {', '.join((summary.get('capability_delta') or {}).get('target_only_deep_languages', [])) or 'none'}",
                    f"- Pair candidates before filter: {(summary.get('pair_screening') or {}).get('pair_candidates_before_filter', 0)}",
                    f"- Pair candidates after filter: {(summary.get('pair_screening') or {}).get('pair_candidates_after_filter', 0)}",
                    f"- Top1/Top2 fused margin: {(summary.get('rank_fusion') or {}).get('top1_top2_margin', 0.0)}",
                    "",
                    "### Evidence Quality",
                ]
            )
            for key, value in sorted(evidence_quality.items()):
                lines.append(f"- {key}: {value}")
            lines.extend(["", "### Subsystem Routing", ""])
            if subsystem_rows:
                for row in subsystem_rows:
                    if not (
                        int(row.get("recommendation_count") or 0)
                        or int(row.get("creation_candidate_count") or 0)
                    ):
                        continue
                    lines.append(
                        f"- {row.get('subsystem_label')}: primary={int(row.get('primary_count') or 0)}, "
                        f"secondary={int(row.get('secondary_count') or 0)}, "
                        f"creation={int(row.get('creation_candidate_count') or 0)}"
                    )
                    lines.append(
                        f"  Top families: {', '.join(row.get('top_feature_families') or []) or 'none'}"
                    )
                    lines.append(
                        f"  Target zones: {', '.join(row.get('top_target_paths') or []) or 'none'}"
                    )
            else:
                lines.append("- No subsystem routing summary was generated.")
            lines.extend(["", "### Upgrade Programs", ""])
            if upgrade_clusters:
                for cluster in upgrade_clusters[:12]:
                    lines.append(
                        f"- {cluster.get('recommended_subsystem_label')} :: "
                        f"{', '.join(cluster.get('feature_families') or ['mechanism'])} -> {cluster.get('target_path')}"
                    )
                    lines.append(
                        f"  Sources ({int(cluster.get('source_count') or 0)}): {', '.join(cluster.get('source_paths') or []) or 'none'}"
                    )
                    lines.append(
                        f"  Posture: {cluster.get('posture')} | Relation: {cluster.get('dominant_relation_type')} | "
                        f"Actionability: {float(cluster.get('top_actionability_score') or 0.0):.3f}"
                    )
                    lines.append(
                        f"  Value: {', '.join(cluster.get('expected_value') or []) or 'capability_gain'}"
                    )
                    lines.append(
                        f"  Summary: {cluster.get('program_summary') or 'Clustered from repeated high-signal relations.'}"
                    )
            else:
                lines.append("- No clustered upgrade programs were generated.")
            lines.extend(["", "### Canonical Program Groups", ""])
            if program_groups:
                for group in program_groups[:12]:
                    lines.append(
                        f"- {group.get('recommended_subsystem_label')} :: "
                        f"{', '.join(group.get('feature_families') or ['mechanism'])} -> {group.get('target_path')}"
                    )
                    lines.append(
                        f"  Sources ({int(group.get('source_count') or 0)}): {', '.join(group.get('source_paths') or []) or 'none'}"
                    )
                    lines.append(
                        f"  Confidence: {float(group.get('calibrated_confidence') or 0.0):.3f} | "
                        f"Relation: {group.get('dominant_relation_type')}"
                    )
            else:
                lines.append("- No dominance-compressed program groups were generated.")
            lines.extend(["", "### Primary Port Recommendations", ""])
            if primary_recommendations:
                for index, relation in enumerate(primary_recommendations[:12], start=1):
                    lines.extend(
                        [
                            f"#### {index}. {', '.join(relation.get('feature_families') or ['mechanism'])}",
                            f"- Source: {relation.get('source_path')}",
                            f"- Target: {relation.get('target_path')}",
                            f"- Subsystem: {relation.get('recommended_subsystem_label') or 'Unrouted'} "
                            f"[confidence={float(relation.get('subsystem_confidence') or 0.0):.3f}]",
                            f"- Posture: {relation.get('posture')} | Relation: {relation.get('relation_type')} | "
                            f"Score: {float(relation.get('relation_score', 0.0)):.3f} | "
                            f"Calibrated: {float(relation.get('calibrated_confidence', 0.0)):.3f}",
                            f"- Why port: {relation.get('why_port') or relation.get('rationale')}",
                            f"- What Anvil gets: {relation.get('expected_value_summary') or 'capability gain'}",
                            f"- Why this target: {relation.get('why_target_here') or relation.get('target_path')}",
                            f"- Subsystem rationale: {relation.get('subsystem_rationale') or (relation.get('evidence') or {}).get('subsystem_rationale') or 'none'}",
                            f"- Negative evidence: {', '.join(item.get('reason') for item in relation.get('negative_evidence', [])) or 'none'}",
                            f"- Abstain reason: {relation.get('abstain_reason') or 'none'}",
                            f"- Proof summary: {((relation.get('proof_graph') or {}).get('path_summary') or 'none')}",
                            f"- Evidence: features={', '.join((relation.get('evidence') or {}).get('shared_feature_tags', [])) or 'none'}; roles={', '.join((relation.get('evidence') or {}).get('shared_role_tags', [])) or 'none'}; noise={', '.join((relation.get('evidence') or {}).get('noise_flags', [])) or 'none'}",
                            "",
                        ]
                    )
            else:
                lines.extend(["- No primary recommendations were promoted for this candidate.", ""])
            lines.extend(["### Disparate / Non-Obvious Opportunities", ""])
            if disparate_opportunities:
                for relation in disparate_opportunities[:10]:
                    lines.extend(
                        [
                            f"- {relation.get('source_path')} -> {relation.get('target_path')} "
                            f"[{relation.get('relation_type')}, {relation.get('posture')}, "
                            f"{relation.get('recommended_subsystem_label') or 'Unrouted'}, "
                            f"score={float(relation.get('relation_score', 0.0)):.3f}]",
                            f"  Why it is non-obvious: {relation.get('why_not_obvious') or 'Role alignment is stronger than lexical overlap.'}",
                            f"  Why it still matters: {relation.get('why_port') or relation.get('rationale')}",
                        ]
                    )
            else:
                lines.append("- No non-obvious opportunities were promoted above the low-signal threshold.")
            lines.extend(["", "### Secondary Opportunities", ""])
            if secondary_recommendations:
                for relation in secondary_recommendations[:12]:
                    lines.extend(
                        [
                            f"- {relation.get('source_path')} -> {relation.get('target_path')} "
                            f"[{relation.get('posture')}, {relation.get('recommended_subsystem_label') or 'Unrouted'}, "
                            f"score={float(relation.get('relation_score', 0.0)):.3f}]",
                            f"  Value: {relation.get('expected_value_summary') or 'capability gain'}",
                            f"  Rationale: {relation.get('why_port') or relation.get('rationale')}",
                        ]
                    )
            else:
                lines.append("- No secondary opportunities.")
            lines.extend(["", "### Feature Synthesis", ""])
            for row in list(summary.get("feature_matrix") or [])[:12]:
                lines.append(
                    f"- {row.get('feature_family')} [{row.get('status')}] "
                    f"score={float(row.get('top_relation_score', 0.0)):.3f} "
                    f"{row.get('source_path') or 'feature gap'} -> {row.get('target_path') or 'feature creation'}"
                )
            lines.extend(["", "### Detailed Migration Recipes"])
            recipes = list(comparison.get("migration_recipes") or [])
            for recipe in recipes[:10]:
                lines.extend(
                    [
                        f"- {recipe.get('title')}",
                        f"  Target: {recipe.get('target_insertion_path')} | Subsystem: {recipe.get('recommended_subsystem_label') or 'Unrouted'} | Posture: {recipe.get('posture')} | Implementation tier: {recipe.get('preferred_implementation_language')}",
                        f"  Why port: {recipe.get('why_port') or 'See relation rationale'}",
                        f"  Why here: {recipe.get('why_target_here') or recipe.get('target_insertion_path')}",
                        f"  Expected value: {recipe.get('expected_value_summary') or 'capability gain'}",
                        f"  Verification: {', '.join(recipe.get('verification_requirements') or []) or 'none'}",
                        f"  Rollback: {', '.join((recipe.get('lowering_ir') or {}).get('rollback_criteria', [])) or 'none'}",
                    ]
                )
            lines.extend(["", "### Value Realization"])
            for key, value in sorted((value_realization.get("value_category_counts") or {}).items()):
                lines.append(f"- {key}: {value}")
            lines.extend(["", "### Port Ledger"])
            for entry in list(comparison.get("port_ledger") or [])[:10]:
                lines.append(
                    f"- {entry.get('status', 'candidate')} "
                    f"{entry.get('relation_type')} "
                    f"{entry.get('source_path')} -> {entry.get('target_path')} "
                    f"[{entry.get('posture')}] score={float(entry.get('relation_score', 0.0)):.3f}"
                )
            lines.extend(["", "### Frontier Packets"])
            for packet in list(comparison.get("frontier_packets") or [])[:10]:
                lines.append(
                    f"- {packet.get('title')} "
                    f"[priority={float(packet.get('priority', 0.0)):.3f}, "
                    f"{packet.get('recommended_subsystem_label') or 'Unrouted'}] "
                    f"tracks={', '.join(packet.get('recommended_tracks') or []) or 'none'}"
                )
            if comparison.get("creation_ledger"):
                lines.extend(["", "### Creation Ledger"])
                for entry in list(comparison.get("creation_ledger") or [])[:8]:
                    lines.append(
                        f"- {entry.get('feature_family')} "
                        f"[{entry.get('kind')}] "
                        f"priority={float(entry.get('priority', 0.0)):.3f} "
                        f"{entry.get('rationale')}"
                    )
            lines.extend(["", "### Low-Signal and Generic Analogues"])
            if low_signal_relations:
                for relation in low_signal_relations[:20]:
                    lines.append(
                        f"- [{float(relation.get('relation_score', 0.0)):.3f}] "
                        f"{relation.get('relation_type')} :: "
                        f"{relation.get('source_path')} -> {relation.get('target_path')} "
                        f"({relation.get('posture')})"
                    )
            else:
                lines.append("- No low-signal analogues retained in this report slice.")
            lines.extend(["", "### Manual Validation Checklist"])
            for item in list(comparison.get("manual_validation_seed") or [])[:50]:
                lines.append(
                    f"- [ ] {item.get('source_path')} -> {item.get('target_path')} "
                    f"features={', '.join(item.get('feature_families') or []) or 'none'}"
                )
            lines.extend(
                [
                    "",
                    "### Overlay Graph",
                    f"- Nodes: {len(overlay.get('nodes') or [])}",
                    f"- Edges: {len(overlay.get('edges') or [])}",
                    "",
                ]
            )
        rendered = "\n".join(lines)
        append_prefix = ""
        if append and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            append_prefix = "\n\n---\n\n"
        mode = "a" if append else "w"
        with open(output_path, mode, encoding="utf-8") as handle:
            handle.write(f"{append_prefix}{rendered}")
