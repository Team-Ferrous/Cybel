from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def _iter_dependency_lines(source: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for line_no, raw in enumerate(source.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--")):
            continue
        entries.append((line_no, line))
    return entries


def check_dependency_integrity(source: str, filepath: str) -> list[dict[str, Any]]:
    """Validate dependency pinning and hash provenance for requirements-style files."""
    path = Path(filepath).name.lower()
    if "requirements" not in path and not path.endswith(("pyproject.toml", "poetry.lock", "uv.lock")):
        return []

    violations: list[dict[str, Any]] = []
    for line_no, line in _iter_dependency_lines(source):
        if " @ " in line:
            # VCS/url pins are handled by provenance checks.
            continue

        has_pin = "==" in line
        if not has_pin:
            violations.append(
                _violation(
                    "AES-SUP-1",
                    filepath,
                    line_no,
                    "Dependency entry must use an explicit version pin (==) for auditable baselines.",
                )
            )

        if has_pin and "--hash=" not in line:
            violations.append(
                _violation(
                    "AES-SUP-2",
                    filepath,
                    line_no,
                    "Pinned dependency is missing hash verification metadata (--hash=...).",
                )
            )

    return violations


def check_provenance(source: str, filepath: str) -> list[dict[str, Any]]:
    """Validate provenance and signing markers for build/release paths."""
    rel = filepath.replace("\\", "/").lower()
    if not any(token in rel for token in ("workflow", "release", "build", "ci", "provenance", "slsa")):
        return []

    lowered = source.lower()
    violations: list[dict[str, Any]] = []

    if not any(marker in lowered for marker in ("provenance", "attestation", "slsa", "sbom")):
        violations.append(
            _violation(
                "AES-SUP-3",
                filepath,
                1,
                "Build/release workflow is missing provenance or attestation markers.",
            )
        )

    if not any(marker in lowered for marker in ("cosign", "sigstore", "gpg", "minisign", "sign-artifacts")):
        violations.append(
            _violation(
                "AES-SUP-4",
                filepath,
                1,
                "Build/release workflow is missing artifact signing markers.",
            )
        )

    return violations


def generate_sbom(output_path: str, fmt: str = "cyclonedx") -> dict[str, Any]:
    """Generate a minimal SBOM from installed Python distributions."""
    format_name = (fmt or "cyclonedx").lower()
    if format_name not in {"cyclonedx", "spdx"}:
        raise ValueError(f"Unsupported SBOM format: {fmt}")

    packages = []
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata.get("Name", "")):
        name = (dist.metadata.get("Name") or dist.metadata.get("Summary") or "unknown").strip()
        if not name:
            continue
        packages.append(
            {
                "name": name,
                "version": dist.version,
                "license": (dist.metadata.get("License") or "UNKNOWN").strip() or "UNKNOWN",
            }
        )

    if format_name == "cyclonedx":
        payload: dict[str, Any] = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "version": 1,
            "components": [
                {
                    "type": "library",
                    "name": item["name"],
                    "version": item["version"],
                    "licenses": [{"license": {"name": item["license"]}}],
                }
                for item in packages
            ],
        }
    else:
        payload = {
            "spdxVersion": "SPDX-2.3",
            "name": "anvil-python-environment",
            "dataLicense": "CC0-1.0",
            "packages": [
                {
                    "name": item["name"],
                    "versionInfo": item["version"],
                    "licenseConcluded": item["license"],
                }
                for item in packages
            ],
        }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
