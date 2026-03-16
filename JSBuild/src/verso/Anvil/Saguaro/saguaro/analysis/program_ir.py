from __future__ import annotations

from typing import Any


class ProgramIRBuilder:
    """Compile comparative recommendations into executable migration programs."""

    def compile(
        self,
        *,
        relation: dict[str, Any],
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
        impact_assessment: dict[str, Any],
    ) -> dict[str, Any]:
        source_path = str(relation.get("source_path") or "")
        target_path = str(relation.get("target_path") or "")
        preferred_language = str(
            relation.get("preferred_implementation_language")
            or relation.get("target_language")
            or "python"
        )
        test_candidates = self._targeted_tests(target_path=target_path, target_pack=target_pack)
        verification = self._verification_plan(
            target_path=target_path,
            test_candidates=test_candidates,
        )
        steps = [
            {
                "step_id": "capture_mechanism",
                "title": f"Capture mechanism from {source_path}",
                "writes": [target_path],
                "evidence_refs": [source_path],
            },
            {
                "step_id": "land_in_target",
                "title": f"Integrate into {target_path}",
                "writes": [target_path],
                "evidence_refs": [target_path],
            },
        ]
        if preferred_language == "cpp":
            steps.append(
                {
                    "step_id": "expose_wrapper_surface",
                    "title": "Expose a thin orchestration/wrapper surface only if needed",
                    "writes": [target_path],
                    "evidence_refs": ["wrapper_policy"],
                }
            )
        return {
            "ir_version": "migration_program_ir.v2",
            "implementation_tier": preferred_language,
            "source_scope": source_path,
            "target_scope": target_path,
            "posture": str(relation.get("posture") or "pattern_only_adoption"),
            "dependencies": sorted(
                set(
                    [target_path]
                    + list(test_candidates)
                    + list((relation.get("feature_families") or [])[:2])
                )
            ),
            "steps": steps,
            "invariants": [
                "preserve public API contracts",
                "keep target build graph authoritative",
                "prefer a native implementation over wrapper accretion where feasible",
            ],
            "verification_plan": verification,
            "rollback_criteria": [
                "landing-zone proof confidence drops below 0.45 after review",
                "targeted tests fail",
                "build verification introduces unresolved breakage",
            ],
            "impact_assessment": impact_assessment,
            "affected_tests": test_candidates,
            "program_invariant_count": 3,
            "verification_step_count": len(verification),
        }

    def _targeted_tests(
        self,
        *,
        target_path: str,
        target_pack: dict[str, Any],
    ) -> list[str]:
        test_files = list(target_pack.get("test_files") or [])
        if not test_files:
            return []
        target_stem = target_path.rsplit("/", 1)[-1].split(".", 1)[0]
        targeted = [
            path
            for path in test_files
            if target_stem and target_stem in path
        ]
        return sorted(targeted or test_files[:3])

    def _verification_plan(
        self,
        *,
        target_path: str,
        test_candidates: list[str],
    ) -> list[dict[str, Any]]:
        commands = [
            {
                "kind": "governance",
                "command": "./venv/bin/saguaro verify . --engines native,ruff,semantic --format json",
            }
        ]
        for path in test_candidates[:3]:
            commands.append(
                {
                    "kind": "targeted_test",
                    "command": f"pytest {path}",
                }
            )
        if not test_candidates:
            commands.append(
                {
                    "kind": "fallback_test",
                    "command": f"pytest -k {target_path.rsplit('/', 1)[-1].split('.', 1)[0]}",
                }
            )
        return commands
