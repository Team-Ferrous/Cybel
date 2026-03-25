"""Shared base scaffolding for domain specialist subagents."""

from __future__ import annotations

from core.agents.subagent import SubAgent


class DomainSpecialistSubagent(SubAgent):
    """Lightweight base for specialist portfolio scaffolding."""

    SAGUARO_TOOLS = ["saguaro_query", "skeleton", "slice", "impact"]
    RESEARCH_BASELINE_TOOLS = [
        "web_search",
        "web_fetch",
        "search_scholar",
        "search_arxiv",
        "fetch_arxiv_paper",
    ]
    prompt_profile = "sovereign_build"
    sovereign_build_policy_enabled = True

    def __init__(self, *args, **kwargs):
        if "specialist_prompt_key" not in kwargs:
            module_name = str(self.__class__.__module__ or "")
            token = module_name.rsplit(".", maxsplit=1)[-1]
            base_name = token.removesuffix("_subagent")
            family = ""
            marker = "core.agents.domain."
            if marker in module_name:
                suffix = module_name.split(marker, maxsplit=1)[1]
                family = suffix.split(".", maxsplit=1)[0]
            if family:
                kwargs["specialist_prompt_key"] = f"{family}/{base_name}"
        super().__init__(*args, **kwargs)

    @classmethod
    def default_tools(cls) -> list[str]:
        """Default specialist toolset favoring semantic code and research."""
        return [*cls.SAGUARO_TOOLS, "read_file", *cls.RESEARCH_BASELINE_TOOLS]

    @classmethod
    def governance_tools(cls) -> list[str]:
        """Governance-leaning toolset with validation hooks."""
        return [*cls.default_tools(), "verify", "run_tests"]
