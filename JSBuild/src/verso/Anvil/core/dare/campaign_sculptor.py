"""Generate DARE-informed campaign code."""

from __future__ import annotations

import os

from core.dare.knowledge_base import DareKnowledgeBase
from core.dare.models import PhaseSpec, TechnicalRoadmap
from core.loops.loop_builder import LoopValidator


class CampaignSculptor:
    """Generate rewrite campaigns from DARE roadmaps."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)
        self.validator = LoopValidator()

    def sculpt_campaign(
        self,
        roadmap: TechnicalRoadmap,
        kb: DareKnowledgeBase,
        output_dir: str,
    ) -> str:
        del kb
        os.makedirs(output_dir, exist_ok=True)
        class_name = self._class_name(roadmap.objective)
        code = self._render_code(class_name, roadmap)
        valid, errors = self.validator.validate_all(code)
        if not valid:
            raise ValueError(f"Generated DARE campaign failed validation: {errors}")
        filename = f"{self._slugify(roadmap.objective)}_campaign.py"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
        return path

    def _render_code(self, class_name: str, roadmap: TechnicalRoadmap) -> str:
        lines = [
            "from core.campaign.base_campaign import BaseCampaignLoop, gate, phase",
            "from core.dare.knowledge_base import DareKnowledgeBase",
            "",
            "",
            f"class {class_name}(BaseCampaignLoop):",
            f"    campaign_name = {roadmap.objective!r}",
            '    campaign_version = "1.0"',
            "    max_retries_per_phase = 2",
            "",
        ]
        for index, phase in enumerate(roadmap.phases):
            lines.extend(self._render_phase(index, phase))
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def _render_phase(self, order: int, phase: PhaseSpec) -> list[str]:
        accepts = "\\n".join(f"- {item}" for item in phase.acceptance_criteria)
        objective_suffix = "\n\nAcceptance Criteria:\n" + accepts
        if phase.phase_id == "hydrate_knowledge":
            body_lines = [
                "        kb = DareKnowledgeBase(self.root_dir, campaign_id=self.campaign_id)",
                "        entries = kb.list_entries(limit=25)",
                '        self.ledger.record_metric("kb_entries_loaded", len(entries))',
                "        for entry in entries:",
                '            self.ledger.record_artifact(f"kb_{entry.category}_{entry.topic}", entry.content[:4000])',
                '        return {"kb_entries_loaded": len(entries), "ok": len(entries) > 0}',
            ]
            gate_line = '        assert result.get("kb_entries_loaded", 0) > 0, "No DARE knowledge was loaded"'
        elif phase.phase_id == "phase_final_verification":
            body_lines = [
                '        verify = self.run_shell("venv/bin/saguaro verify . --engines native,ruff,semantic", timeout=300)',
                '        self.ledger.record_artifact("dare_verify_output", verify.stdout or verify.stderr)',
                "        self.record_evidence(",
                '            name="final_verification",',
                '            summary="Captured verification output for the DARE campaign.",',
                '            evidence_type="verification",',
                '            confidence="medium",',
                '            payload={"returncode": verify.returncode},',
                '            source_phase="phase_final_verification",',
                "        )",
                "        report_path = self.save_report(self.ledger.get_context_summary())",
                '        return {"ok": verify.returncode == 0, "report_path": report_path}',
            ]
            gate_line = '        assert result.get("report_path"), "Final report was not written"'
        else:
            body_lines = [
                "        self.record_evidence(",
                f"            name={phase.phase_id!r},",
                f"            summary={phase.objective!r},",
                '            evidence_type="phase_objective",',
                '            confidence="medium",',
                f"            payload={{'acceptance_criteria': {phase.acceptance_criteria!r}}},",
                f"            source_phase={phase.phase_id!r},",
                "        )",
                "        response = self.spawn_agent(",
                f"            objective={phase.objective!r} + {objective_suffix!r},",
                f"            files={phase.files!r},",
                "            context_from_ledger=True,",
                f"            phase_id={phase.phase_id!r},",
                "        )",
                f"        self.ledger.record_artifact({phase.phase_id!r} + '_summary', response.summary)",
                '        return {"ok": True, "summary": response.summary}',
            ]
            gate_line = '        assert result.get("ok") is True, "Phase execution did not complete successfully"'
        decorator = f"    @phase(order={order}, name={phase.name!r}"
        if phase.depends_on:
            decorator += f", depends_on={phase.depends_on!r}"
        if phase.files:
            decorator += f", files={phase.files!r}"
        decorator += ")"
        return [
            decorator,
            f"    def {phase.phase_id}(self):",
            *body_lines,
            "",
            f"    @gate(phase={phase.phase_id!r})",
            f"    def gate_{phase.phase_id}(self, result):",
            gate_line,
        ]

    @staticmethod
    def _slugify(value: str) -> str:
        return "_".join(part for part in "".join(ch if ch.isalnum() else " " for ch in value).lower().split()) or "dare"

    @classmethod
    def _class_name(cls, value: str) -> str:
        slug = cls._slugify(value)
        return "".join(part.capitalize() for part in slug.split("_")) + "Campaign"
