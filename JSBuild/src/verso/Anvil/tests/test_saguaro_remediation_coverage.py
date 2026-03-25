from __future__ import annotations

from saguaro.sentinel.remediation.capabilities import LanguageCapabilityRegistry
from saguaro.sentinel.remediation.orchestrator import FixOrchestrator


class _VerifierStub:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path
        self.engines = []

    def verify_all(self, path_arg: str = ".", **_: object) -> list[dict]:
        return []


def test_language_capability_registry_covers_plan_languages(tmp_path) -> None:
    registry = LanguageCapabilityRegistry(str(tmp_path))
    cases = {
        "main.py": ("python", "legacy_ruff", False, None),
        "native/kernel.c": ("c", "clang_native", True, "llvm-native"),
        "native/kernel.cpp": ("c++", "clang_native", True, "llvm-native"),
        "native/header.h": ("c++", "clang_native", True, "llvm-native"),
        "crate/lib.rs": ("rust", "rust_tooling", True, "rust-toolchain"),
        "go/main.go": ("go", "go_tooling", True, "go-toolchain"),
        "src/Main.java": ("java", "java_tooling", True, "java-toolchain"),
        "src/App.kt": ("kotlin", "kotlin_tooling", True, "java-toolchain"),
        "web/app.js": ("javascript", "web_linter", True, "node-web"),
        "web/App.jsx": ("javascript", "web_linter", True, "node-web"),
        "web/app.ts": ("typescript", "web_linter", True, "node-web"),
        "web/App.tsx": ("typescript", "web_linter", True, "node-web"),
        "web/index.html": ("html", "web_markup", True, "node-web"),
        "web/site.css": ("css", "web_styles", True, "node-web"),
        "web/site.scss": ("css", "web_styles", True, "node-web"),
        "config/settings.json": ("json", "config_formatter", False, None),
        "config/settings.yaml": ("yaml", "config_formatter", False, None),
        "config/settings.toml": ("toml", "config_formatter", True, "config-formatters"),
        "docs/guide.md": ("md", "docs_formatter", True, "node-web"),
        "scripts/run.sh": ("shell", "shell_formatter", True, "shell-tooling"),
    }

    for rel_path, (language, fix_adapter, managed_by_anvil, toolchain_profile) in cases.items():
        capability = registry.detect(rel_path)
        assert capability.language == language
        assert capability.fix_adapter == fix_adapter
        assert capability.managed_by_anvil is managed_by_anvil
        assert capability.toolchain_profile == toolchain_profile
        if managed_by_anvil:
            assert capability.install_gated is False


def test_orchestrator_registers_all_plan_adapter_keys(tmp_path) -> None:
    orchestrator = FixOrchestrator(str(tmp_path), _VerifierStub(str(tmp_path)))
    expected = {
        "legacy_ruff",
        "python_codemod",
        "clang_native",
        "cpp_semantic",
        "rust_tooling",
        "rust_semantic",
        "go_tooling",
        "go_semantic",
        "java_tooling",
        "java_semantic",
        "kotlin_tooling",
        "kotlin_semantic",
        "web_formatter",
        "web_linter",
        "web_codemod",
        "web_markup",
        "web_styles",
        "web_semantic",
        "config_formatter",
        "config_schema",
        "docs_formatter",
        "docs_semantic",
        "shell_formatter",
        "artifact_templates",
    }

    assert expected.issubset(orchestrator.adapters)
