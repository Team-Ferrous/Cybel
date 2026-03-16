import os
from typing import Dict, Any, Iterable, Optional

from core.aes import AALClassifier, DomainDetector
from core.prompts.aes_prompt_builder import AESPromptBuilder
from core.prompts.system_prompt_builder import SystemPromptBuilder


class PromptManager:
    """
    Manages loading and formatting system prompts from external Markdown templates.
    """

    def __init__(self, template_dir: str = None):
        if template_dir is None:
            # Default to core/prompts/templates relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.template_dir = os.path.join(base_dir, "templates")
        else:
            self.template_dir = template_dir

        self._cache: Dict[str, str] = {}
        self.system_prompt_builder = SystemPromptBuilder()
        self.aes_builder = AESPromptBuilder()
        self.aal_classifier = AALClassifier()
        self.domain_detector = DomainDetector()

    def get_template(self, name: str) -> str:
        """Load a template from the template directory."""
        if name in self._cache:
            return self._cache[name]

        file_path = os.path.join(self.template_dir, f"{name}.md")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prompt template not found: {file_path}")

        with open(file_path, "r") as f:
            content = f.read()

        self._cache[name] = content
        return content

    def format_prompt(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Load and format a template with provided variables."""
        template = self.get_template(template_name)

        # Simple string replacement for {{variable}}
        formatted = template
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            formatted = formatted.replace(placeholder, str(value))

        return formatted

    def get_master_prompt(
        self,
        agent_name: str,
        context_type: str = "general",
        task_text: str = "",
        workset_files: Optional[Iterable[str]] = None,
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Assemble the master system prompt by combining layers.
        """
        variables = {"agent_name": agent_name}

        identity = self.format_prompt("identity", variables)
        cognitive = self.get_template("cognitive")
        protection = self.get_template("protection")

        # Select context-specific layer
        if context_type not in ["conversational", "synthesis", "action"]:
            context_type = "general"

        context_layer = self.get_template(context_type)
        tool_mandate = self.get_template("tool_mandate")
        aes_runtime, prompt_contract = self.aes_builder.build_master_prompt(
            task_text=task_text,
            task_files=workset_files,
            contract_context=prompt_context,
        )
        prompt_contract_block = self.format_prompt_contract(prompt_contract)

        # Assemble
        prompt = f"""{identity}

{cognitive}

{protection}

{context_layer}

{tool_mandate}

## AES Prompt Contract
{prompt_contract_block}

## AES Runtime Contract
{aes_runtime}

# OPERATIONAL EXCELLENCE
You represent the pinnacle of autonomous software engineering intelligence. Every response must exemplify world-class technical quality, architectural insight, and professional excellence.
"""
        return prompt

    def build_prompt_contract(
        self,
        task_text: str = "",
        workset_files: Optional[Iterable[str]] = None,
        role: str = "master",
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        files = list(workset_files) if workset_files else None
        if role == "master":
            _, contract = self.aes_builder.build_master_prompt(
                task_text=task_text,
                task_files=files,
                contract_context=prompt_context,
            )
            return contract
        if role == "verification":
            _, contract = self.aes_builder.build_verification_prompt(
                aal=self.aal_classifier.classify_text(task_text),
                violations=[],
                contract_context=prompt_context,
            )
            return contract
        _, contract = self.aes_builder.build_subagent_prompt(
            role=role,
            task_files=files,
            task_text=task_text,
            contract_context=prompt_context,
        )
        return contract

    def validate_prompt_contract(self, contract: Dict[str, Any]) -> list[str]:
        return self.aes_builder.validate_contract(contract)

    def format_prompt_contract(self, contract: Dict[str, Any]) -> str:
        return self.aes_builder.render_contract_block(contract)

    def get_system_prompt(
        self,
        workset_files: Optional[Iterable[str]] = None,
        task_text: str = "",
        role: str = "master",
        prompt_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        files = list(workset_files) if workset_files else None
        if role == "master":
            payload, contract = self.aes_builder.build_master_prompt(
                task_text=task_text,
                task_files=files,
                contract_context=prompt_context,
            )
        elif role == "verification":
            aal = self.aal_classifier.classify_text(task_text)
            payload, contract = self.aes_builder.build_verification_prompt(
                aal=aal,
                violations=[],
                contract_context=prompt_context,
            )
        else:
            payload, contract = self.aes_builder.build_subagent_prompt(
                role=role,
                task_files=files,
                task_text=task_text,
                contract_context=prompt_context,
            )
        return f"{self.format_prompt_contract(contract)}\n\n{payload}".strip()

    def get_model_family_compression_guidance(self, model_name: str) -> str:
        """Return model-family guidance for context compression behavior."""
        model = (model_name or "").lower()
        if "qwen" in model:
            return "Qwen guidance: keep `_context_updates` terse and deterministic."
        if "llama" in model:
            return "Llama guidance: prefer one-line tool-result summaries after 70% context fill."
        if "deepseek" in model:
            return "DeepSeek guidance: summarize only stale [tcN] results and keep active evidence verbatim."
        if "granite" in model:
            return "Granite guidance: always include `_context_updates` in every tool call, even when []."
        return "Generic guidance: summarize stale [tcN] tool results to preserve context budget."
