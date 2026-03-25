import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from config.settings import SUB_MODEL
from core.aes import AALClassifier, AESRuleRegistry, DomainDetector, RedTeamProtocol
from core.ollama_client import DeterministicOllama
from core.prompts.aes_prompt_builder import AESPromptBuilder
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate


class SubAgent:
    """
    Transient worker (Anvil Tiny).
    Uses AES-aware prompting and self-verification before returning output.
    """

    _CODE_FENCE_RE = re.compile(r"```(?:[^\n]*)\n(.*?)```", re.DOTALL)
    _CITATION_RE = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|cc|cpp|h|hpp|md|json))(?::L\d+)?")

    def __init__(
        self,
        role: str,
        task: str,
        substrate: SaguaroSubstrate,
        context: list,
        aal: str = "AAL-2",
        domains: Optional[Iterable[str]] = None,
        compliance_context: Optional[Dict[str, Any]] = None,
        max_self_verify_retries: int = 1,
    ):
        self.brain = DeterministicOllama(SUB_MODEL)
        self.role = role
        self.task = task
        self.saguaro = substrate
        self.context = context
        self.max_self_verify_retries = max(0, int(max_self_verify_retries))

        self.aal_classifier = AALClassifier()
        self.domain_detector = DomainDetector()
        self.rule_registry = AESRuleRegistry()
        self.red_team_protocol = RedTeamProtocol()
        rules_path = Path("standards/AES_RULES.json")
        if rules_path.exists():
            self.rule_registry.load(str(rules_path))

        classified_aal = self.aal_classifier.classify_from_description(task)
        self.aal = str(aal or classified_aal)

        detected_domains = set(self.domain_detector.detect_from_description(task))
        self.domains: Set[str] = set(domains or detected_domains)

        self.compliance_context = dict(compliance_context or {})
        self.compliance_context.setdefault("trace_id", None)
        self.compliance_context.setdefault("evidence_bundle_id", None)
        self.compliance_context.setdefault("red_team_required", self.aal in {"AAL-0", "AAL-1"})
        self.compliance_context.setdefault("waiver_ids", [])

        prompt_builder = AESPromptBuilder()
        prompt_payload, prompt_contract = prompt_builder.build_subagent_prompt(
            role=role,
            task_files=[],
            task_text=task,
        )
        contract_block = prompt_builder.render_contract_block(prompt_contract)
        self.system_prompt = (
            f"{contract_block}\n\n"
            f"{prompt_payload}\n\n"
            f"Assigned AAL: {self.aal}\n"
            f"Assigned Domains: {', '.join(sorted(self.domains)) if self.domains else 'none'}\n"
            "You MUST return concrete, evidence-backed output with citations."
        )

    def execute(self) -> str:
        context_str = json.dumps(self.context[-2:], default=str) if self.context else "No prior context."

        if "research" in self.role or "find" in self.task.lower():
            queries = [self.task]
            role_query = f"{self.role} {self.task}".strip()
            if role_query not in queries:
                queries.append(role_query)
            tool_context = self.saguaro.agent_query_bundle(
                list(dict.fromkeys(queries)),
                k=5,
            )
        else:
            tool_context = self.saguaro.execute_command("skeleton main.py")

        action_prompt = f"""
TASK: {self.task}
AAL: {self.aal}
DOMAINS: {', '.join(sorted(self.domains)) if self.domains else 'none'}
COMPLIANCE_CONTEXT: {json.dumps(self.compliance_context, default=str)}
PREVIOUS CONTEXT: {context_str}
SAGUARO CONTEXT:
{tool_context}

Execute the task with concrete citations.
"""

        result = self.brain.generate(action_prompt, system_prompt=self.system_prompt)
        verification = self._self_verify(result)

        attempts = 0
        while self._should_retry(verification) and attempts < self.max_self_verify_retries:
            attempts += 1
            correction_prompt = f"""
Your previous output failed mandatory AES self-verification.
Violations:
{json.dumps(verification.get('violations', [])[:20], indent=2, default=str)}

Revise the output to resolve blocking findings while preserving task intent.
"""
            result = self.brain.generate(correction_prompt, system_prompt=self.system_prompt)
            verification = self._self_verify(result)

        return self._package_result(result, verification)

    def _should_retry(self, verification: Dict[str, Any]) -> bool:
        return self.aal in {"AAL-0", "AAL-1"} and not verification.get("passed", False)

    def _self_verify(self, result: str) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        code_blocks = self._extract_code(result)

        if code_blocks and self.rule_registry.rules:
            for rule in self.rule_registry.get_rules_for_aal(self.aal):
                check_fn = self.rule_registry.get_check_function(rule.id)
                if not check_fn:
                    continue
                for code in code_blocks:
                    try:
                        findings = check_fn(code, "<subagent_output>")
                    except Exception as exc:
                        findings = [
                            {
                                "rule_id": rule.id,
                                "filepath": "<subagent_output>",
                                "line": 1,
                                "message": f"checker_failed: {exc}",
                            }
                        ]
                    for finding in findings or []:
                        if not isinstance(finding, dict):
                            finding = {
                                "rule_id": rule.id,
                                "filepath": "<subagent_output>",
                                "line": 1,
                                "message": str(finding),
                            }
                        normalized = dict(finding)
                        normalized.setdefault("rule_id", rule.id)
                        normalized.setdefault("filepath", "<subagent_output>")
                        normalized.setdefault("line", 1)
                        normalized.setdefault("message", "rule violation")
                        normalized["severity"] = rule.severity
                        violations.append(normalized)

        blocking_count = sum(
            1
            for item in violations
            if str(item.get("severity", "")).upper() in {"AAL-0", "AAL-1", "P0", "P1"}
        )

        if self.aal in {"AAL-0", "AAL-1"}:
            passed = blocking_count == 0
        else:
            passed = True

        return {
            "aal": self.aal,
            "domains": sorted(self.domains),
            "violations": violations,
            "blocking_count": blocking_count,
            "passed": passed,
        }

    def _extract_code(self, text: str) -> List[str]:
        blocks = [block.strip() for block in self._CODE_FENCE_RE.findall(text or "") if block.strip()]
        if blocks:
            return blocks
        return [text.strip()] if text and text.strip() else []

    def _extract_citations(self, text: str) -> List[str]:
        citations: List[str] = []
        for match in self._CITATION_RE.findall(text or ""):
            if match not in citations:
                citations.append(match)
        return citations

    def _package_result(self, result: str, verification: Dict[str, Any]) -> str:
        artifacts: Dict[str, Any] = {
            "verification_summary": {
                "status": "pass" if verification.get("passed", False) else "fail",
                "blocking_count": int(verification.get("blocking_count", 0)),
                "violations": len(verification.get("violations", []) or []),
            }
        }
        artifacts.update(
            self.red_team_protocol.build_placeholder_bundle(
                self.aal,
                bool(self.compliance_context.get("red_team_required", False)),
            )
        )
        payload = {
            "role": self.role,
            "task": self.task,
            "aal": self.aal,
            "domains": sorted(self.domains),
            "subagent_analysis": result,
            "codebase_files": self._extract_citations(result),
            "verification": verification,
            "compliance": self.compliance_context,
            "artifacts": artifacts,
        }
        return json.dumps(payload, default=str)
