import re
from typing import Any, Dict, Iterable, List, Optional

from rich.console import Console

from agents.subagent import SubAgent as AESSubAgent
from cli.history import ConversationHistory
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate


class SubAgent:
    """Compatibility facade for legacy delegation imports."""

    _THINKING_RE = re.compile(r"<thinking(?:[^>]*)>(.*?)</thinking>", re.DOTALL)

    def __init__(
        self,
        task: str,
        role: str = "researcher",
        context: Optional[List[Dict[str, Any]]] = None,
        substrate: Optional[SaguaroSubstrate] = None,
        quiet: bool = False,
        console: Optional[Console] = None,
        aal: str = "AAL-2",
        domains: Optional[Iterable[str]] = None,
        compliance_context: Optional[Dict[str, Any]] = None,
        max_self_verify_retries: int = 1,
    ) -> None:
        self.task = str(task)
        self.role = str(role)
        self.quiet = bool(quiet)
        self.console = console or Console()
        self._original_quiet = bool(getattr(self.console, "quiet", False))
        self.history = ConversationHistory(history_file=None, persist_db=False)
        self._delegate = AESSubAgent(
            role=self.role,
            task=self.task,
            substrate=substrate or SaguaroSubstrate(),
            context=list(context or []),
            aal=aal,
            domains=domains,
            compliance_context=compliance_context,
            max_self_verify_retries=max_self_verify_retries,
        )

    def execute(self) -> str:
        if self.quiet:
            self.console.quiet = True
        try:
            seeded_response = self._last_assistant_message()
            if seeded_response:
                return self._extract_summary(seeded_response)
            result = self._delegate.execute()
            self.history.add_message("assistant", result)
            return result
        finally:
            if self.quiet:
                self.console.quiet = self._original_quiet

    def _last_assistant_message(self) -> str:
        for message in reversed(self.history.get_messages()):
            if message.get("role") == "assistant":
                return str(message.get("content") or "")
        return ""

    def _extract_summary(self, content: str) -> str:
        cleaned = self._THINKING_RE.sub("", content or "").strip()
        if cleaned:
            return cleaned
        thinking_match = self._THINKING_RE.search(content or "")
        if thinking_match:
            return thinking_match.group(1).strip()
        return str(content or "").strip()
