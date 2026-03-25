from __future__ import annotations

import ast
import re
from enum import Enum
from pathlib import Path
from typing import Any


class SecurityVerificationLevel(str, Enum):
    """Deterministic security verification tiers for AES governance."""

    SVL_0 = "SVL-0"
    SVL_1 = "SVL-1"
    SVL_2 = "SVL-2"
    SVL_3 = "SVL-3"


_AAL_TO_SVL = {
    "AAL-0": SecurityVerificationLevel.SVL_3,
    "AAL-1": SecurityVerificationLevel.SVL_2,
    "AAL-2": SecurityVerificationLevel.SVL_1,
    "AAL-3": SecurityVerificationLevel.SVL_0,
}

_DANGEROUS_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\beval\(", "CWE-95", "Dynamic evaluation requires hardened sandbox controls."),
    (r"\bexec\(", "CWE-94", "Dynamic code execution requires hardened sandbox controls."),
    (
        r"subprocess\.(Popen|run)\([^\n]*shell\s*=\s*True",
        "CWE-78",
        "Command execution with shell=True requires strict sanitization controls.",
    ),
    (
        r"pickle\.loads\(",
        "CWE-502",
        "Deserialization of untrusted data requires explicit trust boundary checks.",
    ),
)

_EXPOSED_SURFACE_MARKERS: tuple[str, ...] = (
    "fastapi",
    "flask",
    "django",
    "socket",
    "http.server",
    "grpc",
    "aiohttp",
)

_THREAT_MODEL_MARKERS: tuple[str, ...] = (
    "threat model",
    "misuse case",
    "abuse case",
    "security review",
)


def svl_for_aal(aal: str) -> SecurityVerificationLevel:
    return _AAL_TO_SVL.get(str(aal or "").upper(), SecurityVerificationLevel.SVL_0)


def map_aal_to_svl(aal: str) -> SecurityVerificationLevel:
    return svl_for_aal(aal)


def classify_svl_from_text(source: str) -> SecurityVerificationLevel:
    lowered = (source or "").lower()
    if any(token in lowered for token in ("_mm256_", "_mm512_", "secret", "token", "password")):
        return SecurityVerificationLevel.SVL_3
    if any(token in lowered for token in ("optimizer.step", "quantum", "backward(", "loss.backward")):
        return SecurityVerificationLevel.SVL_2
    if any(token in lowered for token in ("argparse", "settings", "logging")):
        return SecurityVerificationLevel.SVL_1
    return SecurityVerificationLevel.SVL_0


def _violation(
    rule_id: str,
    filepath: str,
    line: int,
    message: str,
    *,
    aal: str,
    svl: SecurityVerificationLevel,
    cwe: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
        "aal": aal,
        "svl": svl.value,
    }
    if cwe:
        payload["cwe"] = cwe
    return payload


def check_svl_compliance(source: str, filepath: str) -> list[dict[str, Any]]:
    """Validate SVL-specific security expectations for the target source."""

    lowered = (source or "").lower()
    aal = _infer_aal_from_source(lowered)
    svl = svl_for_aal(aal)
    violations: list[dict[str, Any]] = []

    if any(marker in lowered for marker in _EXPOSED_SURFACE_MARKERS):
        if not any(marker in lowered for marker in _THREAT_MODEL_MARKERS):
            violations.append(
                _violation(
                    "AES-SEC-3",
                    filepath,
                    1,
                    "Exposed service surface found without threat-model or misuse-case evidence markers.",
                    aal=aal,
                    svl=svl,
                    cwe="CWE-693",
                )
            )

    for line, cwe, message in _dangerous_call_sites(source):
        violations.append(
            _violation(
                "AES-SEC-2",
                filepath,
                line,
                message,
                aal=aal,
                svl=svl,
                cwe=cwe,
            )
        )

    path = Path(filepath)
    if path.name.startswith("requirements"):
        for line_no, raw in enumerate(source.splitlines(), start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "==" in stripped and "--hash=" not in stripped:
                violations.append(
                    _violation(
                        "AES-SEC-3",
                        filepath,
                        line_no,
                        "Pinned dependency entry is missing hash provenance metadata (--hash=...).",
                        aal=aal,
                        svl=svl,
                        cwe="CWE-353",
                    )
                )

    return violations


def check_hardcoded_secrets(source: str, filepath: str) -> list[dict[str, Any]]:
    aal = _infer_aal_from_source((source or "").lower())
    svl = svl_for_aal(aal)
    violations: list[dict[str, Any]] = []
    secret_re = re.compile(
        r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{8,}['\"]"
    )
    for match in secret_re.finditer(source):
        line = source.count("\n", 0, match.start()) + 1
        violations.append(
            _violation(
                "AES-SEC-1",
                filepath,
                line,
                "Hardcoded credential-like value detected.",
                aal=aal,
                svl=svl,
                cwe="CWE-798",
            )
        )
    return violations


def _dangerous_call_sites(source: str) -> list[tuple[int, str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "eval":
            violations.append(
                (
                    node.lineno,
                    "CWE-95",
                    "Dynamic evaluation requires hardened sandbox controls.",
                )
            )
        elif isinstance(node.func, ast.Name) and node.func.id == "exec":
            violations.append(
                (
                    node.lineno,
                    "CWE-94",
                    "Dynamic code execution requires hardened sandbox controls.",
                )
            )
        elif isinstance(node.func, ast.Attribute):
            owner = _attribute_owner(node.func)
            if owner == "pickle" and node.func.attr == "loads":
                violations.append(
                    (
                        node.lineno,
                        "CWE-502",
                        "Deserialization of untrusted data requires explicit trust boundary checks.",
                    )
                )
            elif owner == "subprocess" and node.func.attr in {"Popen", "run"}:
                if any(
                    keyword.arg == "shell"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in node.keywords
                ):
                    violations.append(
                        (
                            node.lineno,
                            "CWE-78",
                            "Command execution with shell=True requires strict sanitization controls.",
                        )
                    )
    return violations


def _attribute_owner(node: ast.Attribute) -> str:
    if isinstance(node.value, ast.Name):
        return node.value.id
    if isinstance(node.value, ast.Attribute):
        base = _attribute_owner(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _infer_aal_from_source(lowered: str) -> str:
    if any(token in lowered for token in ("_mm256_", "_mm512_", "secret", "token", "password")):
        return "AAL-0"
    if any(
        token in lowered
        for token in (
            "optimizer.step",
            "quantum",
            "backward(",
            "loss.backward",
            "eval(",
            "exec(",
            "fastapi",
            "flask",
            "grpc",
        )
    ):
        return "AAL-1"
    if any(token in lowered for token in ("settings", "argparse", "logging")):
        return "AAL-2"
    return "AAL-3"
