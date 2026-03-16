import re
from pathlib import Path
from typing import Iterable, Mapping

from core.aes.security_verification import SecurityVerificationLevel, svl_for_aal


class AALClassifier:
    """Deterministic classifier for AES assurance levels."""

    PATTERNS = {
        "AAL-0": [
            r"_mm256_|_mm512_|__m256|__m128",
            r"#pragma\s+omp\s+parallel",
            r"alignas\(\d+\)",
            r"EVP_|AES_|SHA256_|HMAC_",
        ],
        "AAL-1": [
            r"optimizer\.step|backward\(|loss\.backward",
            r"QuantumCircuit|cirq\.Circuit|qiskit",
            r"torch\.autograd|tf\.GradientTape",
            r"symplectic|integrator|conservation",
        ],
        "AAL-2": [
            r"argparse|click\.|typer\.",
            r"config\.|settings\.|\.toml|\.yaml",
            r"logging\.|logger\.",
        ],
    }
    ORDER = ("AAL-0", "AAL-1", "AAL-2", "AAL-3")
    DESCRIPTION_KEYWORDS: Mapping[str, tuple[str, ...]] = {
        "AAL-0": (
            "simd",
            "openmp",
            "kernel",
            "crypto",
            "secret",
            "token",
            "password",
        ),
        "AAL-1": (
            "optimizer",
            "gradient",
            "training loop",
            "quantum",
            "hamiltonian",
            "symplectic",
            "integrator",
        ),
        "AAL-2": (
            "config",
            "settings",
            "logging",
            "cli",
            "argparse",
            "typer",
        ),
    }
    PATH_PATTERNS: Mapping[str, tuple[str, ...]] = {
        "AAL-0": ("simd", "crypto", "secret", "security", "kernel"),
        "AAL-1": ("quantum", "physics", "ml", "training", "optimizer"),
        "AAL-2": ("config", "settings", "cli", "logging"),
    }

    def __init__(self) -> None:
        self._compiled = {
            level: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for level, patterns in self.PATTERNS.items()
        }

    def classify_text(self, content: str) -> str:
        return self._classify_with_patterns(content or "")

    def classify_text_with_svl(self, content: str) -> tuple[str, SecurityVerificationLevel]:
        aal = self.classify_text(content)
        return aal, svl_for_aal(aal)

    def classify_text_with_security_level(
        self, content: str
    ) -> tuple[str, SecurityVerificationLevel]:
        aal = self.classify_text(content)
        return aal, svl_for_aal(aal)

    def classify_from_description(self, description: str) -> str:
        """Alias used by orchestration code for textual task classification."""
        text = (description or "").lower()
        for level in self.ORDER[:-1]:
            if any(keyword in text for keyword in self.DESCRIPTION_KEYWORDS[level]):
                return level
        return "AAL-3"

    def classify_file(self, filepath: str) -> str:
        path = Path(filepath)
        path_text = str(path).lower()
        path_level = "AAL-3"
        for level in self.ORDER[:-1]:
            if any(marker in path_text for marker in self.PATH_PATTERNS[level]):
                path_level = level
                break
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return path_level
        content_level = self.classify_text(content)
        return self._strictest(path_level, content_level)

    def classify_file_with_svl(self, filepath: str) -> tuple[str, SecurityVerificationLevel]:
        aal = self.classify_file(filepath)
        return aal, svl_for_aal(aal)

    def classify_file_with_security_level(
        self, filepath: str
    ) -> tuple[str, SecurityVerificationLevel]:
        aal = self.classify_file(filepath)
        return aal, svl_for_aal(aal)

    def classify_changeset(self, changed_files: Iterable[str]) -> str:
        highest = "AAL-3"
        for filepath in changed_files:
            level = self.classify_file(filepath)
            if self.ORDER.index(level) < self.ORDER.index(highest):
                highest = level
        return highest

    def classify_changeset_with_security_level(
        self, changed_files: Iterable[str]
    ) -> tuple[str, SecurityVerificationLevel]:
        aal = self.classify_changeset(changed_files)
        return aal, svl_for_aal(aal)

    def classify_changeset_with_svl(
        self, changed_files: Iterable[str]
    ) -> tuple[str, SecurityVerificationLevel]:
        aal = self.classify_changeset(changed_files)
        return aal, svl_for_aal(aal)

    def _classify_with_patterns(self, text: str) -> str:
        for level in self.ORDER[:-1]:
            if any(pattern.search(text) for pattern in self._compiled[level]):
                return level
        return "AAL-3"

    def _strictest(self, first: str, second: str) -> str:
        return first if self.ORDER.index(first) < self.ORDER.index(second) else second

    @staticmethod
    def map_aal_to_svl(aal: str) -> SecurityVerificationLevel:
        return svl_for_aal(aal)
