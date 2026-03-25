import json
from pathlib import Path
from typing import Iterable, Optional

from core.aes import AALClassifier, DomainDetector, AESRuleRegistry


class SystemPromptBuilder:
    """Assembles AES prompt fragments with deterministic domain and AAL context."""

    VISUAL_PACK_VERSIONS = ("v1", "v2")

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.standards_dir = self.repo_root / "standards"
        self.prompts_dir = self.repo_root / "prompts"
        self.visuals_dir = self.repo_root / "aes_visuals"
        self.legacy_visuals_dir = self.prompts_dir / "aes_visuals"
        self._aal_classifier = AALClassifier()
        self._domain_detector = DomainDetector()
        self._rule_registry = AESRuleRegistry()
        self._rule_registry.load(str(self.standards_dir / "AES_RULES.json"))

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    def _resolve_visual_pack_source(self, version: str) -> tuple[Optional[Path], str]:
        candidates = (
            (
                self.visuals_dir / version / "PROMPT_GUIDANCE.md",
                "canonical_prompt_guidance",
            ),
            (
                self.visuals_dir / version / "directives.json",
                "canonical_directives_json",
            ),
            (
                self.legacy_visuals_dir / version / "PROMPT_GUIDANCE.md",
                "legacy_prompt_guidance",
            ),
            (
                self.legacy_visuals_dir / version / "directives.json",
                "legacy_directives_json",
            ),
        )
        for candidate, source_kind in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate, source_kind
        return None, "missing"

    @staticmethod
    def _relative_to_repo(path: Path, repo_root: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    @staticmethod
    def _truncate_summary(text: str, max_chars: int = 220) -> str:
        cleaned = " ".join(text.split()).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[: max_chars - 3].rstrip()}..."

    @staticmethod
    def _summarize_visual_text_guidance(text: str, *, max_lines: int = 2) -> str:
        normalized_lines: list[str] = []
        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line or line.startswith("#"):
                continue
            if line[0] in {"-", "*"}:
                line = line[1:].strip()
            if len(line) >= 2 and line[0].isdigit() and line[1] in {".", ")"}:
                line = line[2:].strip()
            if line:
                normalized_lines.append(line)
            if len(normalized_lines) >= max_lines:
                break
        if not normalized_lines:
            return "present but empty; fallback to AES condensed policy"
        summary = " | ".join(normalized_lines)
        return SystemPromptBuilder._truncate_summary(summary)

    def _summarize_visual_json_guidance(self, path: Path, version: str) -> str:
        raw_text = self._read(path)
        if not raw_text:
            return "JSON directives pack present but empty; fallback to AES condensed policy"
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return "JSON directives pack unreadable; fallback to AES condensed policy"

        artifact = ""
        profile = ""
        directives_count = 0
        sample_ids: list[str] = []
        if isinstance(payload, dict):
            artifact = str(payload.get("artifact", "")).strip()
            profile = str(payload.get("profile", "")).strip()
            directives = payload.get("directives")
            if isinstance(directives, list):
                directives_count = len(directives)
                for directive in directives[:3]:
                    if isinstance(directive, dict):
                        directive_id = str(
                            directive.get("directive_id") or directive.get("id", "")
                        ).strip()
                        if directive_id:
                            sample_ids.append(directive_id)
        elif isinstance(payload, list):
            directives_count = len(payload)

        identity = ":".join(part for part in (artifact, profile or version) if part)
        if not identity:
            identity = f"aes_visuals:{version}"
        summary = f"JSON directives pack ({identity}) with {directives_count} directives"
        if sample_ids:
            summary = f"{summary}; sample ids: {', '.join(sample_ids)}"
        return self._truncate_summary(summary)

    def _aes_visual_prompt_section(self) -> str:
        lines = ["AES Visual Guidance (bounded summaries):"]
        for version in self.VISUAL_PACK_VERSIONS:
            reference = f"aes_visuals/{version}"
            expected_path = f"aes_visuals/{version}/PROMPT_GUIDANCE.md"
            visual_path, source_kind = self._resolve_visual_pack_source(version)
            if visual_path:
                source = self._relative_to_repo(visual_path, self.repo_root)
                if visual_path.suffix.lower() == ".json":
                    summary = self._summarize_visual_json_guidance(visual_path, version)
                else:
                    summary = self._summarize_visual_text_guidance(
                        self._read(visual_path)
                    )
                status = (
                    "present_legacy_fallback"
                    if source_kind.startswith("legacy")
                    else "present"
                )
            else:
                source = expected_path
                summary = "missing; fallback to AES condensed and domain rules only"
                status = "missing"
            lines.append(f"- {reference} ({status} @ {source}): {summary}")
        return "\n".join(lines)

    def _existing_files(self, files: Optional[Iterable[str]]) -> list[str]:
        if not files:
            return []
        resolved: list[str] = []
        for item in files:
            path = Path(item)
            if path.exists():
                resolved.append(str(path))
        return resolved

    @staticmethod
    def _connectivity_section(connectivity_context: Optional[dict[str, object]]) -> str:
        if not connectivity_context:
            return ""
        peers = int(connectivity_context.get("peer_count", 0) or 0)
        promotable = int(connectivity_context.get("promotable_peer_count", 0) or 0)
        return (
            "Connectivity Context: "
            f"campaign={connectivity_context.get('local_campaign_id', '') or '-'}; "
            f"phase={connectivity_context.get('local_phase_id', '') or '-'}; "
            f"claims={connectivity_context.get('local_claim_count', 0) or 0}; "
            f"peers={peers}; promotable_peers={promotable}; "
            f"transport={connectivity_context.get('transport_provider', 'none')}; "
            f"trust_zone={connectivity_context.get('trust_zone', 'internal')}"
        )

    def build(
        self,
        task_text: str = "",
        files: Optional[Iterable[str]] = None,
        connectivity_context: Optional[dict[str, object]] = None,
    ) -> str:
        parts = [
            self._read(self.standards_dir / "AES_CONDENSED.md"),
            self._aes_visual_prompt_section(),
        ]
        connectivity = self._connectivity_section(connectivity_context)
        if connectivity:
            parts.append(connectivity)
        tracked_files = self._existing_files(files)
        if tracked_files:
            aal = self._aal_classifier.classify_changeset(tracked_files)
            domains = sorted(self._domain_detector.detect_domains(tracked_files))
            parts.append(f"AAL Context: {aal}")
            applicable_rules = self._rule_registry.get_rules_for_aal(aal)
            if domains:
                domain_rules = []
                for domain in domains:
                    domain_rules.extend(self._rule_registry.get_rules_for_domain(domain))
                merged = {rule.id: rule for rule in [*applicable_rules, *domain_rules]}
                applicable_rules = list(merged.values())
            if applicable_rules:
                parts.append(
                    "Applicable Rules: " + ", ".join(sorted(rule.id for rule in applicable_rules))
                )
            if aal in {"AAL-0", "AAL-1"}:
                parts.append(
                    "Required Artifacts: traceability record, evidence bundle, review signoff, valid waiver if applicable"
                )
            if domains:
                parts.append("Active Domains: " + ", ".join(domains))
                for domain in domains:
                    rule_path = self.standards_dir / "domain_rules" / f"{domain}.md"
                    if rule_path.exists():
                        parts.append(self._read(rule_path))
        elif task_text:
            default_aal = "AAL-2"
            parts.append(f"AAL Context: {default_aal}")
            applicable_rules = self._rule_registry.get_rules_for_aal(default_aal)
            if applicable_rules:
                parts.append(
                    "Applicable Rules: "
                    + ", ".join(sorted(rule.id for rule in applicable_rules))
                )
            parts.append("Required Artifacts: evidence-backed answer and cited file grounding")
        return "\n\n".join(part for part in parts if part)
