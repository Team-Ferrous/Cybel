"""Safe context sharing utilities for cross-instance collaboration."""

from __future__ import annotations

import re
from typing import Dict, List


class ContextShareProtocol:
    """Prepares and receives shareable task context with redaction safeguards."""

    SECRET_PATTERNS = [
        re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([\w-]{8,})"),
        re.compile(r"(?i)(token\s*[:=]\s*)([\w.-]{8,})"),
        re.compile(r"(?i)(secret\s*[:=]\s*)([\w-]{8,})"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ]

    def _redact(self, text: str) -> str:
        redacted = text
        for pattern in self.SECRET_PATTERNS:
            redacted = pattern.sub(lambda match: f"{match.group(1) if match.lastindex and match.lastindex > 1 else ''}[REDACTED]", redacted)
        return redacted

    def _relevant_history(self, task_text: str, chat_history: List[dict]) -> List[dict]:
        keywords = set(re.findall(r"[a-zA-Z0-9_]+", task_text.lower()))
        if not keywords:
            return chat_history[-8:]

        relevant = []
        for item in chat_history:
            content = str(item.get("content", ""))
            terms = set(re.findall(r"[a-zA-Z0-9_]+", content.lower()))
            if keywords & terms:
                relevant.append(item)
        return relevant[-8:] if relevant else chat_history[-6:]

    def prepare_shareable_context(
        self,
        task,
        chat_history: List[dict],
        sharing_level: str = "summary",
    ) -> dict:
        task_id = ""
        instruction = ""
        context_files: List[str] = []
        if isinstance(task, dict):
            task_id = str(task.get("id") or "")
            instruction = str(task.get("instruction") or task.get("task") or "")
            context_files = list(task.get("context_files") or [])
        else:
            task_id = str(getattr(task, "id", "") or "")
            instruction = str(getattr(task, "instruction", "") or "")
            context_files = list(getattr(task, "context_files", []) or [])

        if sharing_level == "none":
            return {"task_id": task_id, "sharing_level": "none"}

        relevant = self._relevant_history(instruction, chat_history)
        redacted_messages = [
            {
                "role": entry.get("role"),
                "content": self._redact(str(entry.get("content", ""))),
            }
            for entry in relevant
        ]

        if sharing_level == "metadata_only":
            return {
                "task_id": task_id,
                "instruction": instruction,
                "context_files": context_files,
                "sharing_level": "metadata_only",
            }

        if sharing_level == "full":
            return {
                "task_id": task_id,
                "instruction": instruction,
                "context_files": context_files,
                "messages": redacted_messages,
                "sharing_level": "full",
            }

        # Default: summary
        summary_lines = [
            f"Task: {instruction}",
            f"Files: {', '.join(context_files) if context_files else 'none'}",
            f"Context messages: {len(redacted_messages)} relevant entries",
        ]
        return {
            "task_id": task_id,
            "instruction": instruction,
            "context_files": context_files,
            "summary": "\n".join(summary_lines),
            "messages": redacted_messages[:3],
            "sharing_level": "summary",
        }

    def receive_context(self, peer_context: dict) -> dict:
        sanitized = dict(peer_context or {})
        if "messages" in sanitized:
            sanitized["messages"] = [
                {
                    "role": item.get("role"),
                    "content": self._redact(str(item.get("content", ""))),
                }
                for item in (sanitized.get("messages") or [])
            ]
        if "summary" in sanitized:
            sanitized["summary"] = self._redact(str(sanitized.get("summary", "")))
        return sanitized
