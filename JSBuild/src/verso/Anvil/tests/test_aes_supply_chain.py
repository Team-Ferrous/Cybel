from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from core.aes.supply_chain import (
    check_dependency_integrity,
    check_provenance,
    generate_sbom,
)


def test_dependency_integrity_flags_unpinned_and_unhashed_entries() -> None:
    source = "requests>=2.0\nnumpy==1.26.4\n"
    violations = check_dependency_integrity(source, "requirements.txt")
    rule_ids = {item["rule_id"] for item in violations}
    assert "AES-SUP-1" in rule_ids
    assert "AES-SUP-2" in rule_ids


def test_dependency_integrity_accepts_hashed_pins() -> None:
    source = "requests==2.32.0 --hash=sha256:abcdef1234567890\n"
    assert check_dependency_integrity(source, "requirements.txt") == []


def test_provenance_flags_missing_attestation_and_signing_markers() -> None:
    source = "name: release\njobs:\n  build:\n    runs-on: ubuntu-latest\n"
    violations = check_provenance(source, ".github/workflows/release.yml")
    rule_ids = {item["rule_id"] for item in violations}
    assert "AES-SUP-3" in rule_ids
    assert "AES-SUP-4" in rule_ids


def test_generate_sbom_writes_cyclonedx_payload(tmp_path: Path) -> None:
    output = tmp_path / "sbom.json"
    payload = generate_sbom(str(output), fmt="cyclonedx")

    assert output.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["bomFormat"] == "CycloneDX"
    assert payload["bomFormat"] == "CycloneDX"


def test_sbom_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "cli-sbom.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/aes_generate_sbom.py",
            "--output",
            str(output),
            "--format",
            "spdx",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["spdxVersion"] == "SPDX-2.3"
