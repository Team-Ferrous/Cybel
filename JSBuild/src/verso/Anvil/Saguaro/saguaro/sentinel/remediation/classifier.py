"""Utilities for classifier."""

from __future__ import annotations

from .models import (
    FIX_CLASS_AES_ARTIFACT,
    FIX_CLASS_AST_CODEMOD,
    FIX_CLASS_MANUAL,
    FIX_CLASS_NATIVE_SEMANTIC,
    FIX_CLASS_RULE_TUNE,
    FIX_CLASS_STOCK_LINTER,
    SAFETY_ARTIFACT_ONLY,
    SAFETY_GUARDED_STRUCTURED,
    SAFETY_SAFE_LINT,
    SAFETY_SAFE_STRUCTURED,
    FindingRecord,
    LanguageCapability,
)

_RULE_TUNE_RULE_IDS = {
    "AES-HPC-1",
    "AES-HPC-2",
    "AES-HPC-3",
    "AES-ERR-1",
    "AES-PY-1",
    "AES-CPLX-1",
    "AES-CR-1",
}
_AES_ARTIFACT_RULE_IDS = {
    "AES-AG-1",
    "AES-VIS-1",
    "AES-VIS-2",
    "AES-TR-1",
    "AES-TR-2",
    "AES-TR-3",
    "AES-SUP-1",
    "AES-SUP-2",
}
_PYTHON_CODEMOD_RULE_IDS = {
    "AES-PY-6",
    "AES-ERR-2",
    "AES-PY-3",
}
_PYTHON_STRUCTURED_PREFIXES = ("ANN", "D")
_RUFF_STOCK_SAFE_PREFIXES = ("I", "UP", "SIM", "RET", "F")


class FindingClassifier:
    """Provide FindingClassifier support."""
    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        self._repo_path = repo_path

    def classify(
        self,
        finding: FindingRecord,
        capability: LanguageCapability,
    ) -> tuple[str, str]:
        """Handle classify."""
        rule_id = finding.rule_id
        if rule_id in _RULE_TUNE_RULE_IDS:
            return FIX_CLASS_RULE_TUNE, SAFETY_GUARDED_STRUCTURED
        if rule_id in _AES_ARTIFACT_RULE_IDS:
            return FIX_CLASS_AES_ARTIFACT, SAFETY_ARTIFACT_ONLY
        if capability.language == "python":
            if rule_id == "AES-CR-2":
                return FIX_CLASS_AST_CODEMOD, SAFETY_GUARDED_STRUCTURED
            if rule_id == "AES-PY-3":
                return FIX_CLASS_AST_CODEMOD, SAFETY_GUARDED_STRUCTURED
            if rule_id in _PYTHON_CODEMOD_RULE_IDS or rule_id.startswith(
                _PYTHON_STRUCTURED_PREFIXES
            ):
                return FIX_CLASS_AST_CODEMOD, SAFETY_SAFE_STRUCTURED
            if finding.engine == "ruff" and rule_id.startswith(_RUFF_STOCK_SAFE_PREFIXES):
                return FIX_CLASS_STOCK_LINTER, SAFETY_SAFE_LINT
        if capability.language in {"c", "c++"} and rule_id.startswith("AES-CPP-"):
            return FIX_CLASS_NATIVE_SEMANTIC, SAFETY_GUARDED_STRUCTURED
        return FIX_CLASS_MANUAL, SAFETY_GUARDED_STRUCTURED
