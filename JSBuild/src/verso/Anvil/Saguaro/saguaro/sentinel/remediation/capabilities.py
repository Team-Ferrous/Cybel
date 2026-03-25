"""Utilities for capabilities."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from .models import LanguageCapability
from .toolchains import ToolchainManager


@dataclass(frozen=True, slots=True)
class _LanguageDefinition:
    language: str
    extensions: tuple[str, ...]
    formatter_adapter: str | None = None
    fix_adapter: str | None = None
    codemod_adapter: str | None = None
    semantic_patch_adapter: str | None = None
    required_tools: tuple[str, ...] = ()
    install_gated: bool = False
    managed_by_anvil: bool = False
    toolchain_profile: str | None = None


_LANGUAGE_DEFINITIONS = (
    _LanguageDefinition(
        language="python",
        extensions=(".py", ".pyi"),
        formatter_adapter="legacy_ruff",
        fix_adapter="legacy_ruff",
        codemod_adapter="python_codemod",
        required_tools=("ruff",),
    ),
    _LanguageDefinition(
        language="c",
        extensions=(".c",),
        formatter_adapter="clang_native",
        fix_adapter="clang_native",
        semantic_patch_adapter="cpp_semantic",
        required_tools=("clang-format", "clang-tidy"),
        managed_by_anvil=True,
        toolchain_profile="llvm-native",
    ),
    _LanguageDefinition(
        language="c++",
        extensions=(".cc", ".cpp", ".cxx", ".h", ".hpp"),
        formatter_adapter="clang_native",
        fix_adapter="clang_native",
        semantic_patch_adapter="cpp_semantic",
        required_tools=("clang-format", "clang-tidy"),
        managed_by_anvil=True,
        toolchain_profile="llvm-native",
    ),
    _LanguageDefinition(
        language="rust",
        extensions=(".rs",),
        formatter_adapter="rust_tooling",
        fix_adapter="rust_tooling",
        semantic_patch_adapter="rust_semantic",
        required_tools=("cargo",),
        managed_by_anvil=True,
        toolchain_profile="rust-toolchain",
    ),
    _LanguageDefinition(
        language="go",
        extensions=(".go",),
        formatter_adapter="go_tooling",
        fix_adapter="go_tooling",
        semantic_patch_adapter="go_semantic",
        required_tools=("gofmt", "gopls"),
        managed_by_anvil=True,
        toolchain_profile="go-toolchain",
    ),
    _LanguageDefinition(
        language="java",
        extensions=(".java",),
        formatter_adapter="java_tooling",
        fix_adapter="java_tooling",
        semantic_patch_adapter="java_semantic",
        required_tools=("google-java-format",),
        managed_by_anvil=True,
        toolchain_profile="java-toolchain",
    ),
    _LanguageDefinition(
        language="kotlin",
        extensions=(".kt", ".kts"),
        formatter_adapter="kotlin_tooling",
        fix_adapter="kotlin_tooling",
        semantic_patch_adapter="kotlin_semantic",
        required_tools=("ktlint",),
        managed_by_anvil=True,
        toolchain_profile="java-toolchain",
    ),
    _LanguageDefinition(
        language="javascript",
        extensions=(".js", ".jsx"),
        formatter_adapter="web_formatter",
        fix_adapter="web_linter",
        codemod_adapter="web_codemod",
        required_tools=("eslint", "prettier"),
        managed_by_anvil=True,
        toolchain_profile="node-web",
    ),
    _LanguageDefinition(
        language="typescript",
        extensions=(".ts", ".tsx"),
        formatter_adapter="web_formatter",
        fix_adapter="web_linter",
        codemod_adapter="web_codemod",
        required_tools=("eslint", "prettier"),
        managed_by_anvil=True,
        toolchain_profile="node-web",
    ),
    _LanguageDefinition(
        language="html",
        extensions=(".html", ".htm"),
        formatter_adapter="web_formatter",
        fix_adapter="web_markup",
        semantic_patch_adapter="web_semantic",
        required_tools=("prettier",),
        managed_by_anvil=True,
        toolchain_profile="node-web",
    ),
    _LanguageDefinition(
        language="css",
        extensions=(".css", ".scss"),
        formatter_adapter="web_formatter",
        fix_adapter="web_styles",
        semantic_patch_adapter="web_semantic",
        required_tools=("stylelint", "prettier"),
        managed_by_anvil=True,
        toolchain_profile="node-web",
    ),
    _LanguageDefinition(
        language="json",
        extensions=(".json",),
        formatter_adapter="config_formatter",
        fix_adapter="config_formatter",
        semantic_patch_adapter="config_schema",
        required_tools=("prettier",),
    ),
    _LanguageDefinition(
        language="yaml",
        extensions=(".yaml", ".yml"),
        formatter_adapter="config_formatter",
        fix_adapter="config_formatter",
        semantic_patch_adapter="config_schema",
        required_tools=("prettier",),
    ),
    _LanguageDefinition(
        language="toml",
        extensions=(".toml",),
        formatter_adapter="config_formatter",
        fix_adapter="config_formatter",
        semantic_patch_adapter="config_schema",
        required_tools=("taplo",),
        managed_by_anvil=True,
        toolchain_profile="config-formatters",
    ),
    _LanguageDefinition(
        language="md",
        extensions=(".md",),
        formatter_adapter="docs_formatter",
        fix_adapter="docs_formatter",
        semantic_patch_adapter="docs_semantic",
        required_tools=("prettier",),
        managed_by_anvil=True,
        toolchain_profile="node-web",
    ),
    _LanguageDefinition(
        language="shell",
        extensions=(".sh",),
        formatter_adapter="shell_formatter",
        fix_adapter="shell_formatter",
        required_tools=("shfmt", "shellcheck"),
        managed_by_anvil=True,
        toolchain_profile="shell-tooling",
    ),
    _LanguageDefinition(
        language="txt",
        extensions=(".txt", ".env"),
        formatter_adapter="docs_formatter",
        fix_adapter="artifact_templates",
    ),
)


class LanguageCapabilityRegistry:
    """Provide LanguageCapabilityRegistry support."""
    def __init__(
        self,
        repo_path: str,
        *,
        toolchains: ToolchainManager | None = None,
        auto_bootstrap: bool = False,
    ) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.toolchains = toolchains or ToolchainManager(self.repo_path)
        self.auto_bootstrap = auto_bootstrap
        self._by_extension: dict[str, _LanguageDefinition] = {}
        for definition in _LANGUAGE_DEFINITIONS:
            for ext in definition.extensions:
                self._by_extension[ext] = definition

    def detect(self, rel_path: str, *, auto_bootstrap: bool | None = None) -> LanguageCapability:
        """Handle detect."""
        _, ext = os.path.splitext(rel_path.lower())
        definition = self._by_extension.get(ext)
        if definition is None:
            return LanguageCapability(language="unknown")

        resolved_tools: dict[str, str | None]
        toolchain_state = "unmanaged"
        toolchain_source = "unmanaged"
        toolchain_message = ""
        if definition.toolchain_profile:
            resolution = self.toolchains.resolve(
                definition.toolchain_profile,
                auto_bootstrap=self.auto_bootstrap if auto_bootstrap is None else auto_bootstrap,
            )
            resolved_tools = dict(resolution.tool_paths)
            installed = {tool: path is not None for tool, path in resolved_tools.items()}
            toolchain_state = resolution.state
            toolchain_source = resolution.source
            toolchain_message = resolution.message
        else:
            resolved_tools = {tool: shutil.which(tool) for tool in definition.required_tools}
            installed = {tool: path is not None for tool, path in resolved_tools.items()}
            if resolved_tools:
                toolchain_state = "system"
                toolchain_source = "system"

        return LanguageCapability(
            language=definition.language,
            formatter_adapter=definition.formatter_adapter,
            fix_adapter=definition.fix_adapter,
            codemod_adapter=definition.codemod_adapter,
            semantic_patch_adapter=definition.semantic_patch_adapter,
            required_tools=list(definition.required_tools),
            installed_tools=installed,
            resolved_tools=resolved_tools,
            install_gated=definition.install_gated,
            managed_by_anvil=definition.managed_by_anvil,
            toolchain_profile=definition.toolchain_profile,
            toolchain_state=toolchain_state,
            toolchain_source=toolchain_source,
            toolchain_message=toolchain_message,
        )

    def known_languages(self) -> list[str]:
        """Handle known languages."""
        return sorted({definition.language for definition in _LANGUAGE_DEFINITIONS})
