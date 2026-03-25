from __future__ import annotations

from pathlib import Path
from typing import Optional


class SpecialistPromptLoader:
    """Compose specialist prompts from shared prompt assets."""

    def __init__(self, prompts_root: Optional[Path] = None) -> None:
        self.prompts_root = prompts_root or (
            Path(__file__).resolve().parents[2] / "prompts" / "subagents"
        )

    def _read_prompt(self, filename: str) -> str:
        path = self.prompts_root / filename
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def load_base_prompt(self) -> str:
        return self._read_prompt("base_specialist.md")

    def load_sovereign_policy_prompt(self) -> str:
        return self._read_prompt("sovereign_build.md")

    def load_role_addendum(self, specialist_prompt_key: Optional[str]) -> str:
        key = str(specialist_prompt_key or "").strip()
        if not key:
            return ""
        if "/" in key:
            return self._read_prompt(f"{key}.md")
        direct = self._read_prompt(f"{key}.md")
        if direct:
            return direct
        candidates = sorted(self.prompts_root.glob(f"**/{key}.md"))
        if not candidates:
            return ""
        try:
            return candidates[0].read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def compose(
        self,
        *,
        role_addendum: str = "",
        prompt_profile: Optional[str] = None,
        specialist_prompt_key: Optional[str] = None,
        include_sovereign_policy: bool = False,
        sovereign_policy_block: str = "",
    ) -> str:
        sections = []

        base_prompt = self.load_base_prompt()
        if base_prompt:
            sections.append(base_prompt)

        profile = str(prompt_profile or "").strip().lower()
        wants_sovereign = include_sovereign_policy or profile in {
            "sovereign",
            "sovereign_build",
        }
        if wants_sovereign:
            sovereign_prompt = self.load_sovereign_policy_prompt()
            if sovereign_prompt:
                sections.append(sovereign_prompt)

        policy_block = str(sovereign_policy_block or "").strip()
        if policy_block:
            sections.append(f"## Sovereign Policy Override\n{policy_block}")

        keyed_addendum = self.load_role_addendum(specialist_prompt_key)
        if keyed_addendum:
            sections.append(keyed_addendum)

        inline_addendum = str(role_addendum or "").strip()
        if inline_addendum:
            sections.append(inline_addendum)

        return "\n\n".join(part for part in sections if part).strip()
