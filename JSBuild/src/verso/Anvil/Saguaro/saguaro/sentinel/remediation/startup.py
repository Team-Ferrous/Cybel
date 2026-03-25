"""Interactive REPL startup checks for managed remediation toolchains."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .toolchains import ToolchainManager

_REPORT_SCHEMA_VERSION = "1.0"
_EAGER_PROFILES = (
    "node-web",
    "config-formatters",
    "shell-tooling",
    "go-toolchain",
    "java-toolchain",
)
_DEFERRED_PROFILES = ("llvm-native",)


def run_repl_startup_toolchain_check(
    repo_path: str,
    *,
    force: bool = False,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """Check managed remediation toolchains during the first interactive REPL start."""
    if os.environ.get("ANVIL_SKIP_TOOLCHAIN_CHECK") == "1":
        return {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "status": "skipped",
            "skipped": True,
            "reason": "ANVIL_SKIP_TOOLCHAIN_CHECK=1",
        }

    manager = ToolchainManager(repo_path)
    report_path = Path(manager.root_dir) / "repl_startup_check.json"
    if report_path.exists() and not force:
        cached = json.loads(report_path.read_text(encoding="utf-8"))
        cached["cached"] = True
        return cached

    bootstrap_all = os.environ.get("ANVIL_BOOTSTRAP_ALL_TOOLCHAINS") == "1"
    profile_results: dict[str, dict[str, Any]] = {}
    ready_profiles: list[str] = []
    missing_profiles: list[str] = []
    bootstrapped_profiles: list[str] = []
    deferred_profiles: list[str] = []

    for profile in (*_EAGER_PROFILES, *_DEFERRED_PROFILES):
        auto_bootstrap = bootstrap and (bootstrap_all or profile in _EAGER_PROFILES)
        resolution = manager.resolve(profile, auto_bootstrap=auto_bootstrap)
        if resolution.installed:
            ready_profiles.append(profile)
        else:
            missing_profiles.append(profile)
        if resolution.bootstrap_attempted and resolution.installed:
            bootstrapped_profiles.append(profile)
        if profile in _DEFERRED_PROFILES and not auto_bootstrap:
            deferred_profiles.append(profile)
        profile_results[profile] = {
            "state": resolution.state,
            "source": resolution.source,
            "installed": resolution.installed,
            "bootstrap_attempted": resolution.bootstrap_attempted,
            "bootstrap_skipped": resolution.bootstrap_skipped,
            "message": resolution.message,
            "tool_paths": resolution.tool_paths,
        }

    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "status": "ok" if not missing_profiles else "partial",
        "cached": False,
        "skipped": False,
        "bootstrap": bootstrap,
        "ready_profiles": ready_profiles,
        "missing_profiles": missing_profiles,
        "bootstrapped_profiles": bootstrapped_profiles,
        "deferred_profiles": deferred_profiles,
        "profiles": profile_results,
        "report_path": str(report_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_repl_startup_toolchain_summary(report: dict[str, Any]) -> list[str]:
    """Render a concise first-run startup summary for the interactive REPL."""
    if report.get("skipped"):
        return []

    ready = ", ".join(report.get("ready_profiles") or []) or "none"
    lines = [f"[anvil] Managed remediation toolchains checked. Ready: {ready}."]

    bootstrapped = report.get("bootstrapped_profiles") or []
    if bootstrapped:
        lines.append(
            "[anvil] Bootstrapped on startup: " + ", ".join(bootstrapped) + "."
        )

    deferred = report.get("deferred_profiles") or []
    if deferred:
        lines.append(
            "[anvil] Deferred heavy profiles until first use or explicit opt-in: "
            + ", ".join(deferred)
            + "."
        )

    missing = report.get("missing_profiles") or []
    if missing:
        lines.append(
            "[anvil] Missing profiles remain unavailable: " + ", ".join(missing) + "."
        )

    return lines
