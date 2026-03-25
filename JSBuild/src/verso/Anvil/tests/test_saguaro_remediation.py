from __future__ import annotations

import json
import subprocess
from pathlib import Path

from saguaro.sentinel.engines.base import BaseEngine
from saguaro.sentinel.remediation import (
    FixOrchestrator,
    ToolchainManager,
    ToolchainResolution,
)
from saguaro.sentinel.remediation.capabilities import LanguageCapabilityRegistry
from saguaro.sentinel.remediation.models import LanguageCapability


class RuffEngine(BaseEngine):
    def run(self, path_arg: str = ".") -> list[dict]:
        return []

    def fix(self, violation: dict[str, object]) -> bool:
        file_path = Path(self.repo_path) / str(violation["file"])
        before = file_path.read_text(encoding="utf-8")
        after = before.replace("import os,sys\n", "import os\nimport sys\n")
        if after == before:
            return False
        file_path.write_text(after, encoding="utf-8")
        return True


class FakeVerifier:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self.engines = [RuffEngine(repo_path)]

    def verify_all(self, path_arg: str = ".", **_: object) -> list[dict]:
        file_path = Path(path_arg)
        if file_path.is_file():
            text = file_path.read_text(encoding="utf-8")
            if "import os,sys" in text:
                return [
                    {
                        "file": str(file_path),
                        "line": 1,
                        "rule_id": "I001",
                        "message": "Imports are unsorted",
                        "severity": "P2",
                        "aal": "AAL-2",
                        "domain": ["universal"],
                        "closure_level": "guarded",
                    }
                ]
            return []
        return []


def test_orchestrator_normalizes_paths_and_builds_plan(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("import os,sys\n", encoding="utf-8")

    orchestrator = FixOrchestrator(str(tmp_path), FakeVerifier(str(tmp_path)))
    findings = [
        {
            "file": str(target),
            "line": 1,
            "rule_id": "I001",
            "message": "Imports are unsorted",
            "severity": "P2",
            "aal": "AAL-2",
            "domain": ["universal"],
            "closure_level": "guarded",
        }
    ]

    normalized = orchestrator.normalize_findings(findings)

    assert normalized[0].file == "pkg/module.py"
    assert normalized[0].language == "python"
    assert normalized[0].engine == "ruff"

    plan = orchestrator.plan(
        normalized,
        target_path=str(target),
        fix_mode="safe",
        dry_run=True,
    )

    assert plan.batch_count == 1
    assert plan.batches[0].adapter_key == "legacy_ruff"
    assert plan.batches[0].supported is True


def test_orchestrator_apply_and_rollback(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("import os,sys\n", encoding="utf-8")

    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)
    initial_findings = verifier.verify_all(path_arg=str(target))

    result = orchestrator.execute(
        findings=initial_findings,
        target_path=str(target),
        verification_kwargs={},
        fix_mode="safe",
        dry_run=False,
    )

    assert result["fixed"] == 1
    assert Path(target).read_text(encoding="utf-8") == "import os\nimport sys\n"
    assert result["fix_receipts"][0]["status"] == "applied"

    rollback = orchestrator.rollback(result["receipt_dir"])

    assert rollback[0].status == "restored"
    assert Path(target).read_text(encoding="utf-8") == "import os,sys\n"


def test_orchestrator_writes_receipts_json(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("import os,sys\n", encoding="utf-8")

    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)
    result = orchestrator.execute(
        findings=verifier.verify_all(path_arg=str(target)),
        target_path=str(target),
        verification_kwargs={},
        fix_mode="safe",
        dry_run=True,
    )

    receipts_path = Path(result["receipts_path"])
    payload = json.loads(receipts_path.read_text(encoding="utf-8"))

    assert receipts_path.exists()
    assert payload[0]["status"] == "planned"


def test_orchestrator_splits_python_codemod_batches_by_rule_family(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("def f(value):\n    return value\n", encoding="utf-8")

    orchestrator = FixOrchestrator(str(tmp_path), FakeVerifier(str(tmp_path)))
    normalized = orchestrator.normalize_findings(
        [
            {
                "file": str(target),
                "line": 1,
                "rule_id": "D103",
                "message": "Missing docstring in public function",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            },
            {
                "file": str(target),
                "line": 1,
                "rule_id": "ANN001",
                "message": "Missing type annotation for function argument `value`",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            },
        ]
    )

    plan = orchestrator.plan(
        normalized,
        target_path=str(target),
        fix_mode="safe",
        dry_run=True,
    )

    assert plan.batch_count == 2
    supported_batches = {tuple(batch.rule_ids): batch.supported for batch in plan.batches}
    assert supported_batches[("D103",)] is True
    assert supported_batches[("ANN001",)] is False


def test_python_codemod_requires_guarded_mode(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def f():\n    try:\n        return 1\n    except:\n        return 0\n",
        encoding="utf-8",
    )

    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)
    result = orchestrator.execute(
        findings=[
            {
                "file": str(target),
                "line": 4,
                "rule_id": "AES-CR-2",
                "message": "Bare except is not allowed",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            }
        ],
        target_path=str(target),
        verification_kwargs={},
        fix_mode="safe",
        dry_run=False,
    )

    assert result["fixed"] == 0
    assert "except:\n" in target.read_text(encoding="utf-8")
    assert result["fix_receipts"][0]["status"] == "skipped"


def test_python_codemod_rewrites_bare_except_in_guarded_mode(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "def f():\n    try:\n        return 1\n    except:\n        return 0\n",
        encoding="utf-8",
    )

    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)
    result = orchestrator.execute(
        findings=[
            {
                "file": str(target),
                "line": 4,
                "rule_id": "AES-CR-2",
                "message": "Bare except is not allowed",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            }
        ],
        target_path=str(target),
        verification_kwargs={},
        fix_mode="guarded",
        dry_run=False,
    )

    assert result["fixed"] == 1
    assert "except Exception:\n" in target.read_text(encoding="utf-8")
    assert result["fix_receipts"][0]["status"] == "applied"


def test_python_codemod_rewrites_mutable_defaults_in_guarded_mode(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        'def f(items=[], *, config={"a": 1}):\n'
        '    """demo"""\n'
        "    return items, config\n",
        encoding="utf-8",
    )

    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)
    result = orchestrator.execute(
        findings=[
            {
                "file": str(target),
                "line": 1,
                "rule_id": "AES-PY-3",
                "message": "Mutable default argument",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            }
        ],
        target_path=str(target),
        verification_kwargs={},
        fix_mode="guarded",
        dry_run=False,
    )

    updated = target.read_text(encoding="utf-8")
    assert result["fixed"] == 2
    assert "def f(items=None, *, config=None):\n" in updated
    assert '    if items is None:\n        items = []\n' in updated
    assert '    if config is None:\n        config = {"a": 1}\n' in updated
    assert result["fix_receipts"][0]["status"] == "applied"


def test_python_codemod_adds_missing_docstrings_in_safe_mode(tmp_path: Path) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "class Greeter:\n"
        "    def __init__(self):\n"
        "        self.name = 'hi'\n"
        "\n"
        "    def greet(self):\n"
        "        return self.name\n"
        "\n"
        "def helper():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)
    result = orchestrator.execute(
        findings=[
            {
                "file": str(target),
                "line": 1,
                "rule_id": "D100",
                "message": "Missing docstring in public module",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            },
            {
                "file": str(target),
                "line": 1,
                "rule_id": "D101",
                "message": "Missing docstring in public class",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            },
            {
                "file": str(target),
                "line": 2,
                "rule_id": "D107",
                "message": "Missing docstring in __init__",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            },
            {
                "file": str(target),
                "line": 5,
                "rule_id": "D102",
                "message": "Missing docstring in public method",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            },
            {
                "file": str(target),
                "line": 8,
                "rule_id": "D103",
                "message": "Missing docstring in public function",
                "severity": "P2",
                "aal": "AAL-2",
                "domain": ["universal"],
                "closure_level": "guarded",
            },
        ],
        target_path=str(target),
        verification_kwargs={},
        fix_mode="safe",
        dry_run=False,
    )

    updated = target.read_text(encoding="utf-8")
    assert result["fixed"] == 5
    assert '"""Utilities for module."""' in updated
    assert '"""Provide Greeter support."""' in updated
    assert '"""Initialize the instance."""' in updated
    assert '"""Handle greet."""' in updated
    assert '"""Handle helper."""' in updated
    assert result["fix_receipts"][0]["status"] == "applied"


def test_artifact_template_adapter_scaffolds_and_rolls_back(tmp_path: Path) -> None:
    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)

    result = orchestrator.execute(
        findings=[
            {
                "file": "standards",
                "line": 1,
                "rule_id": "AES-TR-1",
                "message": "Missing traceability records",
                "severity": "P1",
                "aal": "AAL-1",
                "domain": ["universal"],
                "closure_level": "blocking",
            },
            {
                "file": "standards",
                "line": 1,
                "rule_id": "AES-TR-2",
                "message": "Missing evidence bundle",
                "severity": "P1",
                "aal": "AAL-1",
                "domain": ["universal"],
                "closure_level": "blocking",
            },
        ],
        target_path=str(tmp_path),
        verification_kwargs={},
        fix_mode="safe",
        dry_run=False,
    )

    traceability = tmp_path / "standards" / "traceability" / "TRACEABILITY.jsonl"
    evidence = tmp_path / "standards" / "evidence_bundle.json"

    assert result["fixed"] == 2
    assert traceability.exists()
    assert evidence.exists()
    assert result["fix_receipts"][0]["status"] == "applied"

    rollback = orchestrator.rollback(result["receipt_dir"])

    assert rollback[0].status == "restored"
    assert not traceability.exists()
    assert not evidence.exists()


def test_toolchain_manager_uses_system_tools_without_bootstrap(
    tmp_path: Path, monkeypatch
) -> None:
    manager = ToolchainManager(str(tmp_path), root_dir=str(tmp_path / ".anvil" / "toolchains"))

    def fake_which(tool: str) -> str | None:
        if tool in {"cargo", "rustup"}:
            return f"/usr/bin/{tool}"
        return None

    monkeypatch.setattr(
        "saguaro.sentinel.remediation.toolchains.shutil.which",
        fake_which,
    )

    resolution = manager.resolve("rust-toolchain")

    assert resolution.installed is True
    assert resolution.state == "system"
    assert resolution.bootstrap_attempted is False
    assert resolution.tool_paths["cargo"] == "/usr/bin/cargo"


def test_toolchain_manager_supports_hybrid_system_and_managed_tools(
    tmp_path: Path, monkeypatch
) -> None:
    root_dir = tmp_path / ".anvil" / "toolchains"
    install_root = root_dir / "node-web" / "node_modules" / ".bin"
    install_root.mkdir(parents=True)
    for tool in ("eslint", "prettier", "stylelint"):
        (install_root / tool).write_text("", encoding="utf-8")

    manager = ToolchainManager(str(tmp_path), root_dir=str(root_dir))

    def fake_which(tool: str) -> str | None:
        if tool in {"node", "npm"}:
            return f"/usr/bin/{tool}"
        return None

    monkeypatch.setattr(
        "saguaro.sentinel.remediation.toolchains.shutil.which",
        fake_which,
    )

    resolution = manager.resolve("node-web")

    assert resolution.installed is True
    assert resolution.state == "hybrid"
    assert resolution.tool_paths["node"] == "/usr/bin/node"
    assert resolution.tool_paths["eslint"].endswith("node_modules/.bin/eslint")

    state_vector = manager.state_vector("node-web")
    assert state_vector.qualification_state == "ready"
    assert "eslint" in state_vector.available_tools
    assert "stylelint" in state_vector.available_tools


def test_llvm_native_profile_covers_repo_build_tools_and_managed_clang(
    tmp_path: Path, monkeypatch
) -> None:
    root_dir = tmp_path / ".anvil" / "toolchains"
    install_root = root_dir / "llvm-native" / "bin"
    install_root.mkdir(parents=True)
    for tool in ("clang-format", "clang-tidy", "clang-apply-replacements"):
        (install_root / tool).write_text("", encoding="utf-8")

    manager = ToolchainManager(str(tmp_path), root_dir=str(root_dir))

    def fake_which(tool: str) -> str | None:
        if tool in {"cmake", "c++", "cc", "g++"}:
            return f"/usr/bin/{tool}"
        return None

    monkeypatch.setattr(
        "saguaro.sentinel.remediation.toolchains.shutil.which",
        fake_which,
    )

    resolution = manager.resolve("llvm-native")

    assert resolution.installed is True
    assert resolution.state == "hybrid"
    assert resolution.tool_paths["g++"] == "/usr/bin/g++"
    assert resolution.tool_paths["clang-tidy"].endswith("llvm-native/bin/clang-tidy")


def test_toolchain_manager_can_link_llvm_from_brew(tmp_path: Path, monkeypatch) -> None:
    root_dir = tmp_path / ".anvil" / "toolchains"
    brew_prefix = tmp_path / "brew" / "opt" / "llvm" / "bin"
    brew_prefix.mkdir(parents=True)
    for tool in ("clang-format", "clang-tidy", "clang-apply-replacements"):
        (brew_prefix / tool).write_text("", encoding="utf-8")

    manager = ToolchainManager(str(tmp_path), root_dir=str(root_dir))

    monkeypatch.setattr(
        "saguaro.sentinel.remediation.toolchains.shutil.which",
        lambda tool: "/home/linuxbrew/.linuxbrew/bin/brew" if tool == "brew" else f"/usr/bin/{tool}" if tool in {"cmake", "c++", "cc", "g++"} else None,
    )

    def fake_run(cmd, **kwargs):
        if cmd[:3] == ["/home/linuxbrew/.linuxbrew/bin/brew", "--prefix", "llvm"]:
            return subprocess.CompletedProcess(cmd, 0, stdout=str(brew_prefix.parent) + "\n", stderr="")
        if cmd[:3] == ["/home/linuxbrew/.linuxbrew/bin/brew", "install", "llvm"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "saguaro.sentinel.remediation.toolchains.subprocess.run",
        fake_run,
    )

    resolution = manager.ensure("llvm-native")

    assert resolution.installed is True
    assert resolution.state == "hybrid"
    assert (root_dir / "llvm-native" / "bin" / "clang-tidy").is_symlink()


def test_language_capability_registry_uses_managed_toolchain_resolution(tmp_path: Path) -> None:
    class _ToolchainStub:
        def resolve(
            self,
            profile_name: str,
            *,
            auto_bootstrap: bool = False,
        ) -> ToolchainResolution:
            assert profile_name == "node-web"
            assert auto_bootstrap is False
            return ToolchainResolution(
                profile="node-web",
                state="system",
                tool_paths={
                    "node": "/usr/bin/node",
                    "npm": "/usr/bin/npm",
                    "eslint": "/usr/bin/eslint",
                    "prettier": "/usr/bin/prettier",
                    "stylelint": "/usr/bin/stylelint",
                },
                source="system",
                installed=True,
                bootstrap_skipped=True,
                message="all tools available on PATH",
            )

    registry = LanguageCapabilityRegistry(str(tmp_path), toolchains=_ToolchainStub())
    capability = registry.detect("web/App.tsx")

    assert capability.language == "typescript"
    assert capability.managed_by_anvil is True
    assert capability.install_gated is False
    assert capability.toolchain_profile == "node-web"
    assert capability.toolchain_state == "system"
    assert capability.resolved_tools["eslint"] == "/usr/bin/eslint"


def test_orchestrator_executes_with_managed_toolchain_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("import os,sys\n", encoding="utf-8")

    verifier = FakeVerifier(str(tmp_path))
    orchestrator = FixOrchestrator(str(tmp_path), verifier)

    capability = LanguageCapability(
        language="python",
        formatter_adapter="legacy_ruff",
        fix_adapter="legacy_ruff",
        required_tools=["ruff"],
        installed_tools={"ruff": False},
        managed_by_anvil=True,
        toolchain_profile="python-managed",
        toolchain_state="missing",
    )
    monkeypatch.setattr(orchestrator.registry, "detect", lambda _rel_path: capability)

    calls: list[tuple[str, bool]] = []

    class _ToolchainStub:
        def resolve(
            self,
            profile_name: str,
            *,
            auto_bootstrap: bool = False,
        ) -> ToolchainResolution:
            calls.append((profile_name, auto_bootstrap))
            return ToolchainResolution(
                profile=profile_name,
                state="system",
                tool_paths={"ruff": "/usr/bin/ruff"},
                source="system",
                installed=True,
                bootstrap_attempted=auto_bootstrap,
                message="all tools available on PATH",
            )

    orchestrator.toolchains = _ToolchainStub()

    result = orchestrator.execute(
        findings=verifier.verify_all(path_arg=str(target)),
        target_path=str(target),
        verification_kwargs={},
        fix_mode="safe",
        dry_run=False,
    )

    assert calls == [("python-managed", True)]
    assert result["fix_receipts"][0]["status"] == "applied"
    assert result["fix_receipts"][0]["toolchain_profile"] == "python-managed"
    assert result["fix_receipts"][0]["toolchain_state"] == "system"
    assert result["fix_receipts"][0]["toolchain_tools"]["ruff"] == "/usr/bin/ruff"
    assert result["receipt_summary"][0]["rollback_bundle_path"]
    assert result["verification_envelope"]["pre_mutation"]["finding_count"] == 1
    assert result["toolchain_state_vector"][0]["profile"] == "python-managed"
