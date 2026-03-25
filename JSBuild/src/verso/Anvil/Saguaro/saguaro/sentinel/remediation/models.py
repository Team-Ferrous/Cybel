"""Utilities for models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FIX_CLASS_STOCK_LINTER = "stock_linter_autofix"
FIX_CLASS_FORMATTER = "formatter_fix"
FIX_CLASS_AST_CODEMOD = "ast_codemod"
FIX_CLASS_AES_ARTIFACT = "aes_artifact_or_metadata_fix"
FIX_CLASS_NATIVE_SEMANTIC = "native_semantic_patch"
FIX_CLASS_RULE_TUNE = "rule_tune"
FIX_CLASS_MANUAL = "manual_only"

SAFETY_SAFE_FORMAT = "safe-format"
SAFETY_SAFE_LINT = "safe-lint"
SAFETY_SAFE_STRUCTURED = "safe-structured"
SAFETY_GUARDED_STRUCTURED = "guarded-structured"
SAFETY_SEMANTIC = "semantic"
SAFETY_ARTIFACT_ONLY = "artifact-only"


@dataclass(slots=True)
class FindingRecord:
    finding_id: str
    file: str
    abs_path: str | None
    file_id: str
    line: int
    rule_id: str
    message: str
    severity: str
    aal: str
    domain: list[str]
    closure_level: str
    context: str
    engine: str
    language: str
    fix_class: str = FIX_CLASS_MANUAL
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw", None)
        return payload


@dataclass(slots=True)
class LanguageCapability:
    language: str
    formatter_adapter: str | None = None
    fix_adapter: str | None = None
    codemod_adapter: str | None = None
    semantic_patch_adapter: str | None = None
    required_tools: list[str] = field(default_factory=list)
    installed_tools: dict[str, bool] = field(default_factory=dict)
    resolved_tools: dict[str, str | None] = field(default_factory=dict)
    install_gated: bool = False
    managed_by_anvil: bool = False
    toolchain_profile: str | None = None
    toolchain_state: str = "unmanaged"
    toolchain_source: str = "unmanaged"
    toolchain_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FixAction:
    action_id: str
    adapter_key: str
    tool: str
    language: str
    fix_class: str
    safety_tier: str
    finding_ids: list[str]
    rule_ids: list[str]
    files: list[str]
    confidence: float
    supported: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FixBatch:
    batch_id: str
    adapter_key: str
    tool: str
    language: str
    fix_class: str
    safety_tier: str
    finding_ids: list[str]
    rule_ids: list[str]
    files: list[str]
    confidence: float
    supported: bool
    reason: str = ""
    toolchain_profile: str | None = None
    toolchain_state: str = "unmanaged"
    toolchain_source: str = "unmanaged"
    toolchain_tools: dict[str, str | None] = field(default_factory=dict)
    toolchain_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FixPlan:
    run_id: str
    target_path: str
    fix_mode: str
    dry_run: bool
    batch_count: int
    finding_count: int
    batches: list[FixBatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "target_path": self.target_path,
            "fix_mode": self.fix_mode,
            "dry_run": self.dry_run,
            "batch_count": self.batch_count,
            "finding_count": self.finding_count,
            "batches": [batch.to_dict() for batch in self.batches],
        }


@dataclass(slots=True)
class FixReceipt:
    receipt_id: str
    run_id: str
    batch_id: str
    adapter_key: str
    tool: str
    language: str
    safety_tier: str
    status: str
    dry_run: bool
    finding_ids: list[str]
    rule_ids: list[str]
    files: list[str]
    changed_files: list[str] = field(default_factory=list)
    fixed_count: int = 0
    confidence: float = 0.0
    diff_path: str | None = None
    rollback_bundle_path: str | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    command_log: list[str] = field(default_factory=list)
    toolchain_profile: str | None = None
    toolchain_state: str = "unmanaged"
    toolchain_source: str = "unmanaged"
    toolchain_tools: dict[str, str | None] = field(default_factory=dict)
    toolchain_message: str = ""
    verification_envelope: dict[str, Any] = field(default_factory=dict)
    rollback_coordinates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RollbackReceipt:
    receipt_id: str
    restored_files: list[str]
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
