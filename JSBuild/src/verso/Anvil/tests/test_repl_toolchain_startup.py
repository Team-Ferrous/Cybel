from __future__ import annotations

import builtins
import json

from cli import repl
from saguaro.sentinel.remediation.startup import (
    format_repl_startup_toolchain_summary,
    run_repl_startup_toolchain_check,
)
from saguaro.sentinel.remediation.toolchains import ToolchainResolution


class _ToolchainStub:
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.calls: list[tuple[str, bool]] = []

    def resolve(
        self,
        profile_name: str,
        *,
        auto_bootstrap: bool = False,
    ) -> ToolchainResolution:
        self.calls.append((profile_name, auto_bootstrap))
        installed = profile_name != "llvm-native"
        return ToolchainResolution(
            profile=profile_name,
            state="managed" if installed else "missing",
            tool_paths={},
            source="managed" if installed else "missing",
            installed=installed,
            bootstrap_attempted=auto_bootstrap,
            bootstrap_skipped=not auto_bootstrap,
            message="ok" if installed else "deferred",
        )


def test_run_repl_startup_toolchain_check_bootstraps_eager_profiles(
    tmp_path, monkeypatch
) -> None:
    stub = _ToolchainStub(str(tmp_path / ".anvil" / "toolchains"))
    monkeypatch.setattr(
        "saguaro.sentinel.remediation.startup.ToolchainManager",
        lambda _repo_path: stub,
    )

    report = run_repl_startup_toolchain_check(str(tmp_path), force=True)

    assert report["ready_profiles"] == [
        "node-web",
        "config-formatters",
        "shell-tooling",
        "go-toolchain",
        "java-toolchain",
    ]
    assert report["deferred_profiles"] == ["llvm-native"]
    assert ("llvm-native", False) in stub.calls
    assert ("node-web", True) in stub.calls
    assert json.loads(
        (tmp_path / ".anvil" / "toolchains" / "repl_startup_check.json").read_text(
            encoding="utf-8"
        )
    )["status"] == "partial"


def test_format_repl_startup_toolchain_summary_handles_cached_ready_report() -> None:
    lines = format_repl_startup_toolchain_summary(
        {
            "cached": True,
            "skipped": False,
            "ready_profiles": ["node-web"],
            "missing_profiles": [],
            "bootstrapped_profiles": [],
            "deferred_profiles": ["llvm-native"],
        }
    )
    assert "Ready: node-web." in lines[0]
    assert "Deferred heavy profiles" in lines[1]


def test_main_runs_startup_toolchain_check_before_interactive_repl(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        repl,
        "run_repl_startup_toolchain_check",
        lambda _repo_path: {
            "cached": False,
            "skipped": False,
            "ready_profiles": ["node-web"],
            "missing_profiles": [],
            "bootstrapped_profiles": [],
            "deferred_profiles": ["llvm-native"],
        },
    )

    class _DummyRepl:
        def run(self) -> None:
            calls.append("run")

    monkeypatch.setattr(repl, "AgentREPL", _DummyRepl)
    monkeypatch.setattr(repl, "setup_logging", lambda **_: None)
    monkeypatch.setattr(
        repl,
        "format_repl_startup_toolchain_summary",
        lambda report: ["startup checked"] if report else [],
    )
    monkeypatch.setattr(repl.sys, "argv", ["anvil"])

    class _DummyStdin:
        @staticmethod
        def isatty() -> bool:
            return True

        @staticmethod
        def read() -> str:
            return ""

    monkeypatch.setattr(repl.sys, "stdin", _DummyStdin())
    monkeypatch.setattr(
        builtins,
        "print",
        lambda *_args, **_kwargs: calls.append("print"),
    )

    repl.main()

    assert calls == ["print", "run"]
