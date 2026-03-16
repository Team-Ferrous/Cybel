"""Campaign code generation from natural language or YAML specs."""

from __future__ import annotations

import os
import re
import textwrap
import time
from typing import Any, Dict, Optional

from core.campaign.manifest import CampaignManifest, ManifestLoader
from core.loops.loop_builder import LoopBuilder, LoopValidator

try:
    from config.settings import CAMPAIGN_CONFIG as DEFAULT_CAMPAIGN_CONFIG
except Exception:
    DEFAULT_CAMPAIGN_CONFIG = {}


class CampaignGenerator:
    """Generate executable `BaseCampaignLoop` subclasses."""

    def __init__(self, agent: Optional[Any] = None, loop_builder: Optional[LoopBuilder] = None):
        self.agent = agent
        self.loop_builder = loop_builder or (LoopBuilder(agent) if agent is not None else None)
        self.validator = LoopValidator()
        self.manifest_loader = ManifestLoader()
        self.config = dict(DEFAULT_CAMPAIGN_CONFIG)

    def generate_from_description(self, description: str, target_repo: str = ".") -> str:
        repo_analysis = self._analyze_repo(target_repo)
        code = self._generate_code(description, repo_analysis)

        valid, errors = self.validator.validate_all(code)
        if not valid:
            code = self._fix_validation_errors(code, errors, description)

        return self._save_campaign(code, self._campaign_name_from_description(description))

    def generate_from_yaml(self, yaml_path: str) -> str:
        manifest = self.manifest_loader.load_yaml(yaml_path)
        code = self._manifest_to_code(manifest)

        valid, errors = self.validator.validate_all(code)
        if not valid:
            code = self._fix_validation_errors(code, errors, manifest.name)

        return self._save_campaign(code, manifest.name)

    def generate_from_dare_roadmap(
        self,
        roadmap,
        knowledge_base,
        output_dir: Optional[str] = None,
    ) -> str:
        from core.dare.campaign_sculptor import CampaignSculptor

        sculptor = CampaignSculptor(root_dir=".")
        return sculptor.sculpt_campaign(
            roadmap=roadmap,
            kb=knowledge_base,
            output_dir=output_dir or self.config.get("generated_dir", ".anvil/campaigns/generated"),
        )

    def _generate_code(self, description: str, repo_analysis: Dict[str, Any]) -> str:
        if self.agent is not None and getattr(self.agent, "brain", None) is not None:
            prompt = textwrap.dedent(
                f"""
                Generate Python code for a campaign class that inherits from BaseCampaignLoop.

                User request: {description}
                Repository analysis: {repo_analysis}

                Requirements:
                - Use @phase(order=...) methods.
                - Use @gate(phase="phase_method") with deterministic assertions.
                - Use discover_files/discover_entry_points for iteration.
                - Use spawn_agent only for subjective synthesis.
                - Return only Python code.
                """
            ).strip()
            response = self.agent.brain.generate(prompt)
            extracted = self._extract_code(response)
            if extracted:
                return extracted

        name = self._campaign_name_from_description(description)
        return self._default_campaign_code(name, description)

    def _manifest_to_code(self, manifest: CampaignManifest) -> str:
        class_name = self._sanitize_class_name(manifest.name)
        lines = [
            "from core.campaign.base_campaign import BaseCampaignLoop, gate, phase",
            "",
            "",
            f"class {class_name}(BaseCampaignLoop):",
            f"    campaign_name = {manifest.name!r}",
            "    campaign_version = '1.0'",
            "",
        ]

        for order, phase_spec in enumerate(manifest.phases):
            method_name = f"phase_{self._sanitize_identifier(phase_spec.id)}"
            depends_repr = (
                f", depends_on={repr([f'phase_{self._sanitize_identifier(dep)}' for dep in phase_spec.depends_on])}"
                if phase_spec.depends_on
                else ""
            )
            lines.append(
                f"    @phase(order={order}, name={phase_spec.name!r}{depends_repr})"
            )
            lines.append(f"    def {method_name}(self):")
            lines.extend(self._build_phase_body(phase_spec))
            lines.append("")

            lines.append(f"    @gate(phase={method_name!r})")
            lines.append(f"    def gate_{self._sanitize_identifier(phase_spec.id)}(self, result):")
            if phase_spec.gate.assertions:
                lines.append("        context = dict(result or {})")
                lines.append("        for key, value in self.ledger.get_all_metrics().items():")
                lines.append("            context.setdefault(key, value)")
                for assertion in phase_spec.gate.assertions:
                    compiled = self._compile_assertion(assertion)
                    lines.append(
                        f"        assert {compiled}, {f'Gate assertion failed: {assertion}'!r}"
                    )
            else:
                lines.append("        return")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _build_phase_body(self, phase_spec) -> list[str]:
        body: list[str] = ["        metrics = {}", "        artifacts = []"]

        if phase_spec.iteration == "entry_points":
            body.extend(
                [
                    "        targets = self.discover_entry_points()",
                    "        metrics['entry_points'] = len(targets)",
                ]
            )
        elif phase_spec.iteration == "all_files":
            body.extend(
                [
                    "        targets = self.discover_files(extensions=['.py'])",
                    "        metrics['total_files'] = len(targets)",
                ]
            )
        else:
            body.append("        targets = []")

        if phase_spec.agent_tasks:
            body.append("        for task in %r:" % phase_spec.agent_tasks)
            body.append(
                "            response = self.spawn_agent(task, context_from_ledger=True)"
            )
            body.append("            artifacts.append(response.summary)")

        if phase_spec.per_file_agent_task:
            task_literal = repr(str(phase_spec.per_file_agent_task).strip())
            body.extend(
                [
                    "        for file_path in targets:",
                    f"            objective = {task_literal} + '\\nFile: ' + file_path",
                    "            response = self.spawn_agent(",
                    "                objective,",
                    "                files=[file_path],",
                    "                context_from_ledger=True,",
                    "            )",
                    "            artifacts.append(response.summary)",
                ]
            )

        body.extend(
            [
                "        metrics['processed'] = len(targets)",
                "        for key, value in metrics.items():",
                "            self.ledger.record_metric(key, value)",
                "        if artifacts:",
                "            self.ledger.record_artifact('phase_notes', '\\n'.join(artifacts))",
                "        return metrics",
            ]
        )
        return body

    def _fix_validation_errors(self, code: str, errors: list[str], name_hint: str) -> str:
        del code
        fallback = self._default_campaign_code(name_hint, name_hint)
        valid, _ = self.validator.validate_all(fallback)
        if valid:
            return fallback
        raise ValueError(f"Unable to generate valid campaign code: {errors}")

    def _save_campaign(self, code: str, name: str) -> str:
        generated_dir = self.config.get("generated_dir", ".anvil/campaigns/generated")
        os.makedirs(generated_dir, exist_ok=True)

        slug = self._sanitize_identifier(name)
        path = os.path.join(generated_dir, f"{slug}_campaign.py")

        # Keep deterministic output names while avoiding accidental overwrite collisions.
        if os.path.exists(path):
            path = os.path.join(generated_dir, f"{slug}_campaign_{int(time.time())}.py")

        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
        return path

    def _analyze_repo(self, repo_path: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_files": 0,
            "python_files": 0,
            "directories": 0,
            "entry_points": [],
        }

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [entry for entry in dirs if entry not in {".git", ".venv", "__pycache__"}]
            summary["directories"] += len(dirs)
            summary["total_files"] += len(files)
            for filename in files:
                if filename.endswith(".py"):
                    summary["python_files"] += 1
                if filename in {"main.py", "setup.py"}:
                    rel = os.path.relpath(os.path.join(root, filename), repo_path)
                    summary["entry_points"].append(rel)

        summary["entry_points"] = sorted(set(summary["entry_points"]))
        return summary

    @staticmethod
    def _campaign_name_from_description(description: str) -> str:
        cleaned = " ".join(description.strip().split())
        if not cleaned:
            return "Generated Campaign"
        return cleaned[:80]

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
        if not cleaned:
            return "campaign"
        if cleaned[0].isdigit():
            cleaned = f"campaign_{cleaned}"
        return cleaned

    @staticmethod
    def _sanitize_class_name(value: str) -> str:
        identifier = re.sub(r"[^a-zA-Z0-9]+", " ", value).title().replace(" ", "")
        if not identifier:
            identifier = "GeneratedCampaign"
        if identifier[0].isdigit():
            identifier = f"Campaign{identifier}"
        if not identifier.endswith("Campaign"):
            identifier = f"{identifier}Campaign"
        return identifier

    @staticmethod
    def _extract_code(text: str) -> str:
        match = re.search(r"```python(.*?)```", text or "", flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if "class" in (text or "") and "BaseCampaignLoop" in (text or ""):
            return text.strip()
        return ""

    def _default_campaign_code(self, name: str, description: str) -> str:
        class_name = self._sanitize_class_name(name)
        return textwrap.dedent(
            f"""
            from core.campaign.base_campaign import BaseCampaignLoop, gate, phase


            class {class_name}(BaseCampaignLoop):
                campaign_name = {name!r}
                campaign_version = "1.0"

                @phase(order=0, name="Baseline Snapshot")
                def phase_baseline(self):
                    files = self.discover_files(extensions=['.py'])
                    self.ledger.record_metric('total_files', len(files))
                    self.ledger.record_artifact('objective', {description!r})
                    return {{'total_files': len(files)}}

                @gate(phase="phase_baseline")
                def gate_baseline(self, result):
                    assert result.get('total_files', 0) > 0, 'No source files discovered'

                @phase(order=1, name="Synthesis", depends_on=["phase_baseline"])
                def phase_synthesis(self):
                    summary = self.spawn_agent(
                        objective="Summarize baseline findings and propose next steps",
                        context_from_ledger=True,
                    )
                    self.ledger.record_artifact('synthesis', summary.summary)
                    return {{'summary': summary.summary}}

                @gate(phase="phase_synthesis")
                def gate_synthesis(self, result):
                    assert bool(result.get('summary')), 'Synthesis output is empty'
            """
        ).strip() + "\n"

    @staticmethod
    def _compile_assertion(expression: str) -> str:
        expr = expression.strip()
        len_match = re.fullmatch(r"len\((\w+)\)\s*(==|!=|>=|<=|>|<)\s*(\d+)", expr)
        if len_match:
            var_name, operator, rhs = len_match.groups()
            return f"len(context.get('{var_name}', [])) {operator} {rhs}"

        direct_match = re.fullmatch(r"(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)", expr)
        if direct_match:
            lhs, operator, rhs = direct_match.groups()
            rhs = rhs.strip()
            if re.fullmatch(r"\w+", rhs):
                rhs_compiled = f"context.get('{rhs}')"
            else:
                rhs_compiled = rhs
            return f"context.get('{lhs}') {operator} {rhs_compiled}"

        # Unknown assertion grammar: preserve traceability while forcing explicit implementation.
        return "False"
