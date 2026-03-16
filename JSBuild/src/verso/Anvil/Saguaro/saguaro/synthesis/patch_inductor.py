from __future__ import annotations

import difflib
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticPatchRule:
    rule_id: str
    before_fragment: str
    after_fragment: str
    anchors: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PatchInductor:
    """Infer and replay a bounded structural rewrite from example edits."""

    def induce_rule(
        self,
        before_text: str,
        after_text: str,
        *,
        rule_name: str = "semantic_patch",
    ) -> SemanticPatchRule:
        matcher = difflib.SequenceMatcher(None, before_text, after_text)
        before_chunks: list[str] = []
        after_chunks: list[str] = []
        anchors: list[str] = []
        for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
            if opcode == "equal":
                fragment = before_text[a0:a1].strip()
                if fragment:
                    anchors.append(fragment.splitlines()[0][:80])
                continue
            before_chunks.append(before_text[a0:a1])
            after_chunks.append(after_text[b0:b1])
        before_fragment = "".join(before_chunks)
        after_fragment = "".join(after_chunks)
        if not before_fragment or before_fragment not in before_text:
            before_fragment = before_text
            after_fragment = after_text
        digest = hashlib.sha1(
            f"{rule_name}|{before_fragment}|{after_fragment}".encode("utf-8")
        ).hexdigest()[:12]
        return SemanticPatchRule(
            rule_id=f"{rule_name}:{digest}",
            before_fragment=before_fragment,
            after_fragment=after_fragment,
            anchors=[item for item in anchors[:3] if item],
            evidence=[f"changed_lines={len(before_fragment.splitlines())}"],
        )

    def replay(self, rule: SemanticPatchRule, text: str) -> dict[str, Any]:
        if not rule.before_fragment:
            return {
                "applied": False,
                "result": text,
                "rule_match_precision": 0.0,
                "rule_replay_success_rate": 0.0,
                "unsafe_rewrite_block_count": 1,
            }
        if rule.before_fragment not in text:
            return {
                "applied": False,
                "result": text,
                "rule_match_precision": 0.0,
                "rule_replay_success_rate": 0.0,
                "unsafe_rewrite_block_count": 0,
            }
        updated = text.replace(rule.before_fragment, rule.after_fragment, 1)
        return {
            "applied": True,
            "result": updated,
            "rule_match_precision": 1.0,
            "rule_replay_success_rate": 1.0,
            "unsafe_rewrite_block_count": 0,
        }

    def validate_replay(
        self,
        rule: SemanticPatchRule,
        before_text: str,
        expected_after: str,
    ) -> bool:
        replay = self.replay(rule, before_text)
        return bool(replay["applied"] and replay["result"] == expected_after)
