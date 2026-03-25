"""Typed requirement and traceability models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import hashlib
import re
from typing import Any

_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_SPACE_RE = re.compile(r"\s+")


class RequirementModality(StrEnum):
    """Normalized modality tags."""

    MUST = "must"
    SHALL = "shall"
    SHOULD = "should"
    MAY = "may"
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    WILL = "will"
    IMPLICIT = "implicit"
    UNKNOWN = "unknown"


class RequirementStrength(StrEnum):
    """Normalized strength tags."""

    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    UNSPECIFIED = "unspecified"
    INFORMATIONAL = "informational"


class RequirementPolarity(StrEnum):
    """Polarity of the normative statement."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True, slots=True)
class RequirementClassification:
    """Store modality metadata for a requirement."""

    modality: RequirementModality
    strength: RequirementStrength
    polarity: RequirementPolarity = RequirementPolarity.POSITIVE
    keyword: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to a JSON-friendly mapping."""
        return {
            "modality": self.modality.value,
            "strength": self.strength.value,
            "polarity": self.polarity.value,
            "keyword": self.keyword,
        }


@dataclass(frozen=True, slots=True)
class RequirementRecord:
    """Represent one extracted requirement."""

    requirement_id: str
    source_path: str
    section_path: tuple[str, ...]
    section_anchor: str
    line_start: int
    line_end: int
    statement: str
    normalized_statement: str
    classification: RequirementClassification
    block_kind: str
    ordinal: int = 1
    code_refs: tuple[str, ...] = ()
    test_refs: tuple[str, ...] = ()
    verification_refs: tuple[str, ...] = ()
    graph_refs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    equation_ids: tuple[str, ...] = ()
    status: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_hash(self) -> str:
        """Return a stable content hash for dedupe and traceability."""
        payload = "||".join(
            [
                self.source_path,
                self.section_anchor,
                self.normalized_statement,
                self.classification.modality.value,
                self.classification.strength.value,
                str(self.ordinal),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def design_ref(self) -> str:
        """Return a markdown anchor-style design reference."""
        return (
            f"{self.source_path}#{self.section_anchor}"
            if self.section_anchor
            else self.source_path
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-friendly mapping."""
        payload = asdict(self)
        payload["section_path"] = list(self.section_path)
        payload["classification"] = self.classification.to_dict()
        payload["code_refs"] = list(self.code_refs)
        payload["test_refs"] = list(self.test_refs)
        payload["verification_refs"] = list(self.verification_refs)
        payload["graph_refs"] = list(self.graph_refs)
        payload["tags"] = list(self.tags)
        payload["equation_ids"] = list(self.equation_ids)
        payload["source_hash"] = self.source_hash
        payload["design_ref"] = self.design_ref()
        return payload


def normalize_requirement_text(text: str) -> str:
    """Normalize requirement text for stable identity."""
    collapsed = _SPACE_RE.sub(" ", text.strip())
    return collapsed.strip(" .;:").lower()


def slugify_fragment(value: str, *, fallback: str = "root", max_length: int = 40) -> str:
    """Return a stable slug fragment."""
    normalized = _NON_ALNUM_RE.sub("-", value.strip().lower()).strip("-")
    if not normalized:
        normalized = fallback
    return normalized[:max_length]


def build_section_anchor(section_path: tuple[str, ...]) -> str:
    """Build a markdown anchor from a section path."""
    if not section_path:
        return "document"
    return "/".join(slugify_fragment(part, fallback="section") for part in section_path)


def build_requirement_id(
    *,
    source_path: str,
    section_path: tuple[str, ...],
    normalized_statement: str,
    ordinal: int,
) -> str:
    """Build a stable requirement identifier."""
    path_slug = slugify_fragment(source_path.replace("/", "-"), fallback="doc", max_length=36)
    section_anchor = build_section_anchor(section_path)
    digest_seed = "||".join(
        [source_path, section_anchor, normalized_statement, str(max(1, ordinal))]
    )
    digest = hashlib.sha1(digest_seed.encode("utf-8")).hexdigest()[:12]
    return f"REQ-{path_slug}-{digest}".upper()


@dataclass(frozen=True, slots=True)
class WitnessRecord:
    """A normalized witness attached to a requirement."""

    id: str
    requirement_id: str
    witness_type: str
    artifact_id: str
    result: str
    observed_at: str
    generation_id: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CounterexampleRecord:
    """A minimal counterexample contract for future validation work."""

    id: str
    target_relation_id: str
    counterexample_type: str
    input_or_state: str
    observed_failure: str
    severity: str
    repro_steps: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-friendly mapping."""
        payload = asdict(self)
        payload["repro_steps"] = list(self.repro_steps)
        return payload
