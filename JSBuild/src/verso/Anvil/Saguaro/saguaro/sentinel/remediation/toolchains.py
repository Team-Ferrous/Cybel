"""Utilities for toolchains."""

from __future__ import annotations

import gzip
import json
import os
import platform
import shutil
import stat
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class ToolchainProfile:
    """Provide ToolchainProfile support."""
    name: str
    tools: tuple[str, ...]
    install_root_name: str
    bootstrap_commands: tuple[tuple[str, ...], ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolchainResolution:
    """Provide ToolchainResolution support."""
    profile: str
    state: str
    tool_paths: dict[str, str | None]
    source: str
    installed: bool
    bootstrap_attempted: bool = False
    bootstrap_skipped: bool = False
    message: str = ""


@dataclass(frozen=True, slots=True)
class ToolchainStateVector:
    """Compact qualification snapshot exposed to planners and fix flows."""

    profile: str
    qualification_state: str
    state: str
    source: str
    installed: bool
    available_tools: tuple[str, ...]
    missing_tools: tuple[str, ...]
    bootstrap_attempted: bool = False
    bootstrap_skipped: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "qualification_state": self.qualification_state,
            "state": self.state,
            "source": self.source,
            "installed": self.installed,
            "available_tools": list(self.available_tools),
            "missing_tools": list(self.missing_tools),
            "bootstrap_attempted": self.bootstrap_attempted,
            "bootstrap_skipped": self.bootstrap_skipped,
            "message": self.message,
        }


_PROFILE_DEFINITIONS: dict[str, ToolchainProfile] = {
    "llvm-native": ToolchainProfile(
        name="llvm-native",
        tools=("cmake", "c++", "cc", "g++", "clang-format", "clang-tidy", "clang-apply-replacements"),
        install_root_name="llvm-native",
    ),
    "rust-toolchain": ToolchainProfile(
        name="rust-toolchain",
        tools=("cargo", "rustup"),
        install_root_name="rust-toolchain",
        bootstrap_commands=(
            (
                "sh",
                "-c",
                "curl https://sh.rustup.rs -sSf | sh -s -- -y --no-modify-path --profile minimal",
            ),
        ),
    ),
    "go-toolchain": ToolchainProfile(
        name="go-toolchain",
        tools=("go", "gofmt", "gopls"),
        install_root_name="go-toolchain",
    ),
    "java-toolchain": ToolchainProfile(
        name="java-toolchain",
        tools=("java", "javac", "google-java-format", "ktlint"),
        install_root_name="java-toolchain",
    ),
    "node-web": ToolchainProfile(
        name="node-web",
        tools=("node", "npm", "eslint", "prettier", "stylelint"),
        install_root_name="node-web",
        bootstrap_commands=(
            (
                "npm",
                "install",
                "--no-save",
                "--prefix",
                "{install_root}",
                "eslint@9.22.0",
                "prettier@3.5.3",
                "stylelint@16.16.0",
                "jscodeshift@17.3.0",
                "ts-morph@25.0.1",
                "typescript@5.8.2",
                "@typescript-eslint/parser@8.26.1",
                "@typescript-eslint/eslint-plugin@8.26.1",
            ),
        ),
    ),
    "config-formatters": ToolchainProfile(
        name="config-formatters",
        tools=("taplo",),
        install_root_name="config-formatters",
    ),
    "shell-tooling": ToolchainProfile(
        name="shell-tooling",
        tools=("shfmt", "shellcheck"),
        install_root_name="shell-tooling",
    ),
}


class ToolchainBootstrapError(RuntimeError):
    """Provide ToolchainBootstrapError support."""
    pass


class ToolchainManager:
    """Provide ToolchainManager support."""
    def __init__(self, repo_path: str, root_dir: str | None = None) -> None:
        """Initialize the instance."""
        self.repo_path = os.path.abspath(repo_path)
        self.root_dir = os.path.abspath(
            root_dir or os.path.join(self.repo_path, ".anvil", "toolchains")
        )
        self.manifest_path = os.path.join(self.root_dir, "manifest.json")
        os.makedirs(self.root_dir, exist_ok=True)

    def resolve(
        self,
        profile_name: str,
        *,
        auto_bootstrap: bool = False,
    ) -> ToolchainResolution:
        """Handle resolve."""
        profile = _PROFILE_DEFINITIONS[profile_name]
        system_paths = self._resolve_system_tools(profile.tools)
        managed_paths = self._resolve_managed_tools(profile)
        combined_paths = self._merge_tool_paths(system_paths, managed_paths)
        state, source = self._resolution_origin(system_paths, managed_paths, combined_paths)

        if all(combined_paths.values()):
            return ToolchainResolution(
                profile=profile.name,
                state=state,
                tool_paths=combined_paths,
                source=source,
                installed=True,
                message=self._resolution_message(source),
            )

        if not auto_bootstrap:
            return ToolchainResolution(
                profile=profile.name,
                state="missing",
                tool_paths=combined_paths,
                source="missing",
                installed=False,
                bootstrap_skipped=True,
                message="toolchain missing and auto bootstrap disabled",
            )

        try:
            return self.ensure(profile.name)
        except ToolchainBootstrapError as exc:
            system_paths = self._resolve_system_tools(profile.tools)
            managed_paths = self._resolve_managed_tools(profile)
            combined_paths = self._merge_tool_paths(system_paths, managed_paths)
            return ToolchainResolution(
                profile=profile.name,
                state="missing",
                tool_paths=combined_paths,
                source="missing",
                installed=False,
                bootstrap_attempted=True,
                message=str(exc),
            )

    def ensure(self, profile_name: str) -> ToolchainResolution:
        """Handle ensure."""
        profile = _PROFILE_DEFINITIONS[profile_name]
        system_paths = self._resolve_system_tools(profile.tools)
        managed_paths = self._resolve_managed_tools(profile)
        combined_paths = self._merge_tool_paths(system_paths, managed_paths)
        state, source = self._resolution_origin(system_paths, managed_paths, combined_paths)
        if all(combined_paths.values()):
            resolution = ToolchainResolution(
                profile=profile.name,
                state=state,
                tool_paths=combined_paths,
                source=source,
                installed=True,
                bootstrap_skipped=True,
                message=self._resolution_message(source),
            )
            self._record_manifest(resolution)
            return resolution

        install_root = self._install_root(profile)
        os.makedirs(install_root, exist_ok=True)
        if profile.bootstrap_commands:
            for command in profile.bootstrap_commands:
                rendered = [part.format(install_root=install_root) for part in command]
                self._run_bootstrap_command(rendered, install_root, profile)
        else:
            self._bootstrap_profile(profile)

        system_paths = self._resolve_system_tools(profile.tools)
        managed_paths = self._resolve_managed_tools(profile)
        combined_paths = self._merge_tool_paths(system_paths, managed_paths)
        installed = all(combined_paths.values())
        state, source = self._resolution_origin(system_paths, managed_paths, combined_paths)
        if not installed:
            raise ToolchainBootstrapError(
                f"profile '{profile_name}' bootstrap completed but required tools are still missing"
            )
        resolution = ToolchainResolution(
            profile=profile.name,
            state=state,
            tool_paths=combined_paths,
            source=source,
            installed=installed,
            bootstrap_attempted=True,
            message="bootstrapped Anvil-managed toolchain",
        )
        self._record_manifest(resolution)
        return resolution

    def bootstrap_plan(self, profile_name: str) -> dict[str, object]:
        """Handle bootstrap plan."""
        profile = _PROFILE_DEFINITIONS[profile_name]
        return {
            "profile": profile.name,
            "tools": list(profile.tools),
            "install_root": self._install_root(profile),
            "bootstrap_handler": profile.name if not profile.bootstrap_commands else "command-sequence",
            "bootstrap_commands": [
                [part.format(install_root=self._install_root(profile)) for part in command]
                for command in profile.bootstrap_commands
            ],
            "platform": platform.system().lower(),
        }

    def state_vector(
        self,
        profile_name: str,
        *,
        auto_bootstrap: bool = False,
    ) -> ToolchainStateVector:
        """Return a compact planner-safe qualification snapshot."""

        resolution = self.resolve(profile_name, auto_bootstrap=auto_bootstrap)
        available_tools = tuple(
            sorted(tool for tool, path in resolution.tool_paths.items() if path)
        )
        missing_tools = tuple(
            sorted(tool for tool, path in resolution.tool_paths.items() if not path)
        )
        qualification_state = "ready"
        if not resolution.installed:
            qualification_state = "degraded" if available_tools else "missing"
        elif resolution.state == "hybrid":
            qualification_state = "ready"

        return ToolchainStateVector(
            profile=resolution.profile,
            qualification_state=qualification_state,
            state=resolution.state,
            source=resolution.source,
            installed=resolution.installed,
            available_tools=available_tools,
            missing_tools=missing_tools,
            bootstrap_attempted=resolution.bootstrap_attempted,
            bootstrap_skipped=resolution.bootstrap_skipped,
            message=resolution.message,
        )

    def state_vectors(
        self,
        profile_names: list[str] | tuple[str, ...] | None = None,
        *,
        auto_bootstrap: bool = False,
    ) -> list[ToolchainStateVector]:
        profiles = (
            list(profile_names)
            if profile_names
            else sorted(_PROFILE_DEFINITIONS.keys())
        )
        vectors: list[ToolchainStateVector] = []
        for profile_name in profiles:
            if profile_name not in _PROFILE_DEFINITIONS:
                continue
            vectors.append(
                self.state_vector(profile_name, auto_bootstrap=auto_bootstrap)
            )
        return vectors

    def _resolve_system_tools(self, tools: tuple[str, ...]) -> dict[str, str | None]:
        return {tool: shutil.which(tool) for tool in tools}

    def _resolve_managed_tools(self, profile: ToolchainProfile) -> dict[str, str | None]:
        install_root = Path(self._install_root(profile))
        tool_paths: dict[str, str | None] = {}
        for tool in profile.tools:
            path = self._managed_binary_path(install_root, tool)
            tool_paths[tool] = str(path) if path and path.exists() else None
        return tool_paths

    def _merge_tool_paths(
        self,
        system_paths: dict[str, str | None],
        managed_paths: dict[str, str | None],
    ) -> dict[str, str | None]:
        merged: dict[str, str | None] = {}
        for tool in {**system_paths, **managed_paths}:
            merged[tool] = system_paths.get(tool) or managed_paths.get(tool)
        return merged

    def _resolution_origin(
        self,
        system_paths: dict[str, str | None],
        managed_paths: dict[str, str | None],
        combined_paths: dict[str, str | None],
    ) -> tuple[str, str]:
        if not any(combined_paths.values()):
            return "missing", "missing"
        has_system = any(system_paths.get(tool) for tool in combined_paths)
        has_managed = any(managed_paths.get(tool) for tool in combined_paths)
        if has_system and has_managed:
            return "hybrid", "hybrid"
        if has_managed:
            return "managed", "managed"
        return "system", "system"

    def _resolution_message(self, source: str) -> str:
        if source == "system":
            return "all tools available on PATH"
        if source == "managed":
            return "using Anvil-managed cached toolchain"
        if source == "hybrid":
            return "using hybrid system and Anvil-managed toolchain"
        return ""

    def _managed_binary_path(self, install_root: Path, tool: str) -> Path | None:
        candidates = [
            install_root / "bin" / tool,
            install_root / tool / "bin" / tool,
            install_root / "node_modules" / ".bin" / tool,
        ]
        if platform.system().lower().startswith("win"):
            candidates.extend(
                [
                    install_root / "bin" / f"{tool}.exe",
                    install_root / "node_modules" / ".bin" / f"{tool}.cmd",
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _install_root(self, profile: ToolchainProfile) -> str:
        return os.path.join(self.root_dir, profile.install_root_name)

    def _run_bootstrap_command(
        self,
        command: list[str],
        install_root: str,
        profile: ToolchainProfile,
    ) -> None:
        env = os.environ.copy()
        env.update(profile.env)
        env.setdefault("CARGO_HOME", os.path.join(install_root, ".cargo"))
        env.setdefault("RUSTUP_HOME", os.path.join(install_root, ".rustup"))
        env.setdefault("GOBIN", os.path.join(install_root, "bin"))
        env.setdefault("npm_config_cache", os.path.join(install_root, ".npm-cache"))
        proc = subprocess.run(
            command,
            cwd=self.repo_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise ToolchainBootstrapError(
                f"bootstrap for profile '{profile.name}' failed: {' '.join(command)}\n"
                f"{proc.stdout}\n{proc.stderr}"
            )

    def _bootstrap_profile(self, profile: ToolchainProfile) -> None:
        if profile.name == "llvm-native":
            self._bootstrap_llvm_native(profile)
            return
        if profile.name == "go-toolchain":
            self._bootstrap_go_toolchain(profile)
            return
        if profile.name == "java-toolchain":
            self._bootstrap_java_toolchain(profile)
            return
        if profile.name == "shell-tooling":
            self._bootstrap_shell_tooling(profile)
            return
        if profile.name == "config-formatters":
            self._bootstrap_config_formatters(profile)
            return
        raise ToolchainBootstrapError(
            f"profile '{profile.name}' does not yet have a managed bootstrap implementation"
        )

    def _bootstrap_llvm_native(self, profile: ToolchainProfile) -> None:
        if self._bootstrap_llvm_via_brew(profile):
            return

        release = self._github_latest_release("llvm/llvm-project")
        tag = str(release.get("tag_name", ""))
        version = tag.removeprefix("llvmorg-")
        if not version:
            raise ToolchainBootstrapError("could not determine latest LLVM version")

        suffix_map = {
            ("linux", "x86_64"): "LLVM-{version}-Linux-X64.tar.xz",
            ("linux", "aarch64"): "LLVM-{version}-Linux-ARM64.tar.xz",
            ("darwin", "arm64"): "LLVM-{version}-macOS-ARM64.tar.xz",
        }
        asset_name = suffix_map.get(self._platform_key())
        if asset_name is None:
            raise ToolchainBootstrapError(
                f"llvm-native bootstrap is not implemented for platform {platform.system()} {platform.machine()}"
            )

        asset = self._find_release_asset(release, asset_name.format(version=version))
        install_root = Path(self._install_root(profile))
        archive_path = install_root / Path(asset["name"]).name
        self._download_file(asset["browser_download_url"], archive_path)
        extracted_root = self._extract_archive(archive_path, install_root)
        self._link_tools(
            extracted_root / "bin",
            install_root / "bin",
            profile.tools,
        )

    def _bootstrap_llvm_via_brew(self, profile: ToolchainProfile) -> bool:
        brew = shutil.which("brew")
        if brew is None:
            return False

        prefix_proc = subprocess.run(
            [brew, "--prefix", "llvm"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        prefix = Path(prefix_proc.stdout.strip()) if prefix_proc.returncode == 0 else None
        if prefix is not None and self._llvm_brew_tools_exist(prefix):
            self._link_tools(prefix / "bin", Path(self._install_root(profile)) / "bin", profile.tools)
            return True

        install_proc = subprocess.run(
            [brew, "install", "llvm"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if install_proc.returncode != 0:
            return False

        prefix_proc = subprocess.run(
            [brew, "--prefix", "llvm"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if prefix_proc.returncode != 0:
            return False
        prefix = Path(prefix_proc.stdout.strip())
        if not self._llvm_brew_tools_exist(prefix):
            return False
        self._link_tools(prefix / "bin", Path(self._install_root(profile)) / "bin", profile.tools)
        return True

    def _llvm_brew_tools_exist(self, prefix: Path) -> bool:
        return all(
            (prefix / "bin" / tool).exists()
            for tool in ("clang-format", "clang-tidy", "clang-apply-replacements")
        )

    def _bootstrap_go_toolchain(self, profile: ToolchainProfile) -> None:
        platform_map = {
            ("linux", "x86_64"): ("linux", "amd64"),
            ("linux", "aarch64"): ("linux", "arm64"),
            ("darwin", "arm64"): ("darwin", "arm64"),
            ("darwin", "x86_64"): ("darwin", "amd64"),
        }
        go_os_arch = platform_map.get(self._platform_key())
        if go_os_arch is None:
            raise ToolchainBootstrapError(
                f"go-toolchain bootstrap is not implemented for platform {platform.system()} {platform.machine()}"
            )

        release = self._download_json("https://go.dev/dl/?mode=json")[0]
        archive_name = None
        for entry in release.get("files", []):
            if (
                entry.get("os") == go_os_arch[0]
                and entry.get("arch") == go_os_arch[1]
                and entry.get("kind") == "archive"
            ):
                archive_name = str(entry["filename"])
                break
        if archive_name is None:
            raise ToolchainBootstrapError("could not determine Go archive for this platform")

        install_root = Path(self._install_root(profile))
        archive_path = install_root / archive_name
        self._download_file(f"https://go.dev/dl/{archive_name}", archive_path)
        extracted_root = self._extract_archive(archive_path, install_root)
        if extracted_root.name != "go" and (extracted_root / "go").exists():
            extracted_root = extracted_root / "go"
        if extracted_root.name != "go":
            raise ToolchainBootstrapError("unexpected Go archive layout")
        self._link_tools(extracted_root / "bin", install_root / "bin", ("go", "gofmt"))

        env = os.environ.copy()
        env["GOROOT"] = str(extracted_root)
        env["PATH"] = f"{install_root / 'bin'}:{env.get('PATH', '')}"
        env["GOBIN"] = str(install_root / "bin")
        self._run_bootstrap_command(
            [str(install_root / "bin" / "go"), "install", "golang.org/x/tools/gopls@latest"],
            str(install_root),
            ToolchainProfile(
                name=profile.name,
                tools=profile.tools,
                install_root_name=profile.install_root_name,
                env={k: v for k, v in env.items() if k in {"GOROOT", "PATH", "GOBIN"}},
            ),
        )

    def _bootstrap_java_toolchain(self, profile: ToolchainProfile) -> None:
        platform_map = {
            ("linux", "x86_64"): ("linux", "x64"),
            ("linux", "aarch64"): ("linux", "aarch64"),
            ("darwin", "arm64"): ("mac", "aarch64"),
            ("darwin", "x86_64"): ("mac", "x64"),
        }
        java_os_arch = platform_map.get(self._platform_key())
        if java_os_arch is None:
            raise ToolchainBootstrapError(
                f"java-toolchain bootstrap is not implemented for platform {platform.system()} {platform.machine()}"
            )

        install_root = Path(self._install_root(profile))
        jdk_url = (
            "https://api.adoptium.net/v3/binary/latest/21/ga/"
            f"{java_os_arch[0]}/{java_os_arch[1]}/jdk/hotspot/normal/eclipse"
        )
        jdk_archive = install_root / f"temurin-21-{java_os_arch[0]}-{java_os_arch[1]}.tar.gz"
        self._download_file(jdk_url, jdk_archive)
        extracted_root = self._extract_archive(jdk_archive, install_root)
        self._link_tools(extracted_root / "bin", install_root / "bin", ("java", "javac"))

        gjf_release = self._github_latest_release("google/google-java-format")
        gjf_asset = self._find_release_asset(gjf_release, lambda name: name.endswith("-all-deps.jar"))
        gjf_jar = install_root / "lib" / gjf_asset["name"]
        self._download_file(gjf_asset["browser_download_url"], gjf_jar)
        self._write_java_wrapper(
            install_root / "bin" / "google-java-format",
            gjf_jar,
            install_root / "bin" / "java",
        )

        ktlint_release = self._github_latest_release("pinterest/ktlint")
        ktlint_asset = self._find_release_asset(ktlint_release, "ktlint")
        ktlint_bin = install_root / "bin" / "ktlint"
        self._download_file(ktlint_asset["browser_download_url"], ktlint_bin)
        self._ensure_executable(ktlint_bin)

    def _bootstrap_shell_tooling(self, profile: ToolchainProfile) -> None:
        install_root = Path(self._install_root(profile))
        shellcheck_release = self._github_latest_release("koalaman/shellcheck")
        shellcheck_asset_name_map = {
            ("linux", "x86_64"): "shellcheck-v{version}.linux.x86_64.tar.xz",
            ("linux", "aarch64"): "shellcheck-v{version}.linux.aarch64.tar.xz",
            ("darwin", "arm64"): "shellcheck-v{version}.darwin.aarch64.tar.xz",
            ("darwin", "x86_64"): "shellcheck-v{version}.darwin.x86_64.tar.xz",
        }
        version = str(shellcheck_release.get("tag_name", "")).removeprefix("v")
        asset_name = shellcheck_asset_name_map.get(self._platform_key())
        if asset_name is None:
            raise ToolchainBootstrapError(
                f"shell-tooling bootstrap is not implemented for platform {platform.system()} {platform.machine()}"
            )
        shellcheck_asset = self._find_release_asset(
            shellcheck_release,
            asset_name.format(version=version),
        )
        shellcheck_archive = install_root / Path(shellcheck_asset["name"]).name
        self._download_file(shellcheck_asset["browser_download_url"], shellcheck_archive)
        shellcheck_root = self._extract_archive(shellcheck_archive, install_root)
        self._link_tools(shellcheck_root, install_root / "bin", ("shellcheck",))

        shfmt_release = self._github_latest_release("mvdan/sh")
        shfmt_asset_name_map = {
            ("linux", "x86_64"): "shfmt_{version}_linux_amd64",
            ("linux", "aarch64"): "shfmt_{version}_linux_arm64",
            ("darwin", "arm64"): "shfmt_{version}_darwin_arm64",
            ("darwin", "x86_64"): "shfmt_{version}_darwin_amd64",
        }
        shfmt_version = str(shfmt_release.get("tag_name", ""))
        shfmt_asset_name = shfmt_asset_name_map.get(self._platform_key())
        if shfmt_asset_name is None:
            raise ToolchainBootstrapError(
                f"shfmt bootstrap is not implemented for platform {platform.system()} {platform.machine()}"
            )
        shfmt_asset = self._find_release_asset(
            shfmt_release,
            shfmt_asset_name.format(version=shfmt_version),
        )
        shfmt_bin = install_root / "bin" / "shfmt"
        self._download_file(shfmt_asset["browser_download_url"], shfmt_bin)
        self._ensure_executable(shfmt_bin)

    def _bootstrap_config_formatters(self, profile: ToolchainProfile) -> None:
        release = self._github_latest_release("tamasfe/taplo")
        asset_name_map = {
            ("linux", "x86_64"): "taplo-linux-x86_64.gz",
            ("linux", "aarch64"): "taplo-linux-aarch64.gz",
            ("darwin", "x86_64"): "taplo-darwin-x86_64.gz",
            ("darwin", "arm64"): "taplo-darwin-aarch64.gz",
        }
        asset_name = asset_name_map.get(self._platform_key())
        if asset_name is None:
            raise ToolchainBootstrapError(
                f"config-formatters bootstrap is not implemented for platform {platform.system()} {platform.machine()}"
            )
        asset = self._find_release_asset(release, asset_name)
        install_root = Path(self._install_root(profile))
        archive_path = install_root / asset_name
        self._download_file(asset["browser_download_url"], archive_path)
        taplo_path = install_root / "bin" / "taplo"
        self._decompress_gzip_binary(archive_path, taplo_path)
        self._ensure_executable(taplo_path)

    def _download_json(self, url: str) -> object:
        request = Request(url, headers={"User-Agent": "AnvilToolchainManager/1.0"})
        with urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))

    def _github_latest_release(self, repo: str) -> dict[str, object]:
        payload = self._download_json(f"https://api.github.com/repos/{repo}/releases/latest")
        if not isinstance(payload, dict):
            raise ToolchainBootstrapError(f"unexpected release payload for {repo}")
        return payload

    def _find_release_asset(
        self,
        release: dict[str, object],
        matcher: str | callable,
    ) -> dict[str, str]:
        assets = release.get("assets", [])
        if not isinstance(assets, list):
            raise ToolchainBootstrapError("release payload did not include assets")
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", ""))
            matched = matcher(name) if callable(matcher) else name == matcher
            if matched:
                return {
                    "name": name,
                    "browser_download_url": str(asset.get("browser_download_url", "")),
                }
        raise ToolchainBootstrapError(f"could not locate expected release asset: {matcher}")

    def _download_file(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = Request(url, headers={"User-Agent": "AnvilToolchainManager/1.0"})
        with urlopen(request) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    def _extract_archive(self, archive_path: Path, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        top_level = self._peek_tar_root(archive_path)
        self._run_extract_command(archive_path, destination)
        if top_level:
            candidate = destination / top_level
            if candidate.exists():
                return candidate
        stem = archive_path.name
        for suffix in (".tar.xz", ".tar.gz", ".tgz", ".tar"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        candidate = destination / stem
        if candidate.exists():
            return candidate
        return destination

    def _peek_tar_root(self, archive_path: Path) -> str | None:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive:
                name = member.name.split("/", 1)[0]
                if name:
                    return name
        return None

    def _run_extract_command(self, archive_path: Path, destination: Path) -> None:
        proc = subprocess.run(
            ["tar", "-xf", str(archive_path), "-C", str(destination)],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise ToolchainBootstrapError(
                f"archive extraction failed for {archive_path.name}\n{proc.stdout}\n{proc.stderr}"
            )

    def _link_tools(
        self,
        source_root: Path,
        bin_dir: Path,
        tools: tuple[str, ...],
    ) -> None:
        bin_dir.mkdir(parents=True, exist_ok=True)
        for tool in tools:
            source = source_root / tool
            if not source.exists():
                continue
            target = bin_dir / tool
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)

    def _write_java_wrapper(self, wrapper_path: Path, jar_path: Path, java_path: Path) -> None:
        wrapper_path.parent.mkdir(parents=True, exist_ok=True)
        wrapper_path.write_text(
            "#!/bin/sh\n"
            f'exec "{java_path}" -jar "{jar_path}" "$@"\n',
            encoding="utf-8",
        )
        self._ensure_executable(wrapper_path)

    def _ensure_executable(self, path: Path) -> None:
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def _decompress_gzip_binary(self, archive_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive_path, "rb") as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    def _platform_key(self) -> tuple[str, str]:
        system = platform.system().lower()
        machine = platform.machine().lower()
        aliases = {
            "amd64": "x86_64",
            "x64": "x86_64",
            "arm64": "aarch64" if system == "linux" else "arm64",
        }
        machine = aliases.get(machine, machine)
        return system, machine

    def _record_manifest(self, resolution: ToolchainResolution) -> None:
        manifest = self._load_manifest()
        manifest[resolution.profile] = {
            "state": resolution.state,
            "source": resolution.source,
            "installed": resolution.installed,
            "tool_paths": resolution.tool_paths,
            "message": resolution.message,
        }
        Path(self.manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_manifest(self) -> dict[str, object]:
        if not os.path.exists(self.manifest_path):
            return {}
        try:
            return json.loads(Path(self.manifest_path).read_text(encoding="utf-8"))
        except Exception:
            return {}
