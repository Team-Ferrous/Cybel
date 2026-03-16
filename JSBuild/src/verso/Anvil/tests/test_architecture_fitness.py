from __future__ import annotations

import json
from pathlib import Path

from core.aes.fitness_functions import (
    check_abi_boundary_stability,
    check_forbidden_imports,
    check_no_cyclic_dependencies,
    run_all_fitness_checks,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_check_no_cyclic_dependencies_detects_cycle(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "a.py", "import pkg.b\n")
    _write(tmp_path / "pkg" / "b.py", "import pkg.a\n")

    violations = check_no_cyclic_dependencies(str(tmp_path))
    assert violations
    assert any("Import cycle detected" in item["message"] for item in violations)


def test_check_forbidden_imports_flags_core_aes_to_agents(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "aes" / "module.py", "import agents.master\n")

    violations = check_forbidden_imports(str(tmp_path))
    assert violations
    assert violations[0]["rule_id"] == "AES-ARCH-3"


def test_check_abi_boundary_stability_flags_signature_drift(tmp_path: Path) -> None:
    _write(tmp_path / "core" / "api.py", "def public_fn(x, y):\n    return x + y\n")
    baseline = {"core.api.public_fn": "(x)"}
    _write(tmp_path / ".anvil" / "aes_abi_baseline.json", json.dumps(baseline))

    violations = check_abi_boundary_stability(str(tmp_path))
    assert violations
    assert "signature changed" in violations[0]["message"]


def test_run_all_fitness_checks_passes_on_clean_layout(tmp_path: Path) -> None:
    _write(tmp_path / "domains" / "feature.py", "def run(x):\n    return x\n")
    _write(tmp_path / "core" / "aes" / "policy.py", "def guard(v):\n    return v\n")

    result = run_all_fitness_checks(str(tmp_path))
    assert result["passed"] is True
    assert result["violations"] == []
