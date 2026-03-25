"""Utilities for external."""

import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from typing import Any

from saguaro.utils.file_utils import get_code_files

from .base import BaseEngine

logger = logging.getLogger(__name__)


class EngineError(Exception):
    """Raised when an external engine fails catastrophically."""


class ExternalUtil:
    """Provide ExternalUtil support."""
    MAX_FILES_PER_CALL = 200

    @staticmethod
    def run_subproc(
        cmd: list[str],
        cwd: str,
        check: bool = False,
        return_result: bool = False,
    ) -> str | subprocess.CompletedProcess[str]:
        # Ensure tools installed in the active venv are discoverable.
        """Handle run subproc."""
        env = os.environ.copy()
        venv_bin = os.path.dirname(sys.executable)
        path = env.get("PATH", "")
        if venv_bin not in path:
            env["PATH"] = f"{venv_bin}:{path}"

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                env=env,
            )
            if check and result.returncode != 0:
                raise EngineError(
                    f"Command {' '.join(cmd)} failed with exit code {result.returncode}\n"
                    f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                )
            if return_result:
                return result
            return result.stdout + "\n" + result.stderr
        except EngineError:
            raise
        except Exception as e:
            logger.error("Failed to run command %s: %s", " ".join(cmd), e)
            if check:
                raise EngineError(f"Subprocess execution failed: {e}") from e
            if return_result:
                return subprocess.CompletedProcess(
                    cmd,
                    returncode=1,
                    stdout="",
                    stderr=str(e),
                )
            return ""

    @staticmethod
    def _normalize_rel_path(path: str) -> str:
        return path.replace("\\", "/")

    @staticmethod
    def _resolve_target(repo_path: str, path_arg: str) -> str:
        if not path_arg or path_arg in {".", "./"}:
            return os.path.abspath(repo_path)
        if os.path.isabs(path_arg):
            return os.path.abspath(path_arg)
        return os.path.abspath(os.path.join(repo_path, path_arg))

    @staticmethod
    def _git_list_python_files(repo_path: str, target_abs: str) -> list[str] | None:
        if not os.path.isdir(os.path.join(repo_path, ".git")):
            return None

        try:
            rel_target = os.path.relpath(target_abs, repo_path)
        except Exception:
            return []

        if rel_target.startswith(".."):
            return []

        pathspec = "." if rel_target in {"", "."} else rel_target
        cmd = ["git", "ls-files", "--", pathspec]
        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
        if result.returncode != 0:
            return None

        files = []
        for line in result.stdout.splitlines():
            rel = line.strip()
            if not rel.endswith(".py"):
                continue
            abs_candidate = os.path.join(repo_path, rel)
            # Skip tracked paths that were deleted from the working tree.
            if not os.path.isfile(abs_candidate):
                continue
            files.append(ExternalUtil._normalize_rel_path(rel))

        # Include untracked single-file target when explicitly requested.
        if (
            not files
            and os.path.isfile(target_abs)
            and target_abs.endswith(".py")
            and os.path.exists(target_abs)
        ):
            rel = os.path.relpath(target_abs, repo_path)
            if not rel.startswith(".."):
                files.append(ExternalUtil._normalize_rel_path(rel))

        return sorted(set(files))

    @staticmethod
    def _filesystem_list_python_files(repo_path: str, target_abs: str) -> list[str]:
        if os.path.isfile(target_abs):
            if target_abs.endswith(".py"):
                rel = os.path.relpath(target_abs, repo_path)
                if not rel.startswith(".."):
                    return [ExternalUtil._normalize_rel_path(rel)]
            return []

        if not os.path.isdir(target_abs):
            return []

        files = get_code_files(target_abs, exclusions=[])
        rel_files = []
        for abs_path in files:
            if not abs_path.endswith(".py"):
                continue
            rel = os.path.relpath(abs_path, repo_path)
            if rel.startswith(".."):
                continue
            rel_files.append(ExternalUtil._normalize_rel_path(rel))
        return sorted(set(rel_files))

    @staticmethod
    def resolve_python_targets(repo_path: str, path_arg: str = ".") -> list[str]:
        """Handle resolve python targets."""
        target_abs = ExternalUtil._resolve_target(repo_path, path_arg)
        if not target_abs.startswith(os.path.abspath(repo_path)):
            return []

        git_files = ExternalUtil._git_list_python_files(repo_path, target_abs)
        if git_files is not None:
            return git_files
        return ExternalUtil._filesystem_list_python_files(repo_path, target_abs)

    @staticmethod
    def chunked(items: list[str], size: int) -> Iterable[list[str]]:
        """Handle chunked."""
        for idx in range(0, len(items), size):
            yield items[idx : idx + size]


class RuffEngine(BaseEngine):
    """Provide RuffEngine support."""
    def run(self, path_arg: str = ".") -> list[dict[str, Any]]:
        """Handle run."""
        logger.info("Running RuffEngine...")
        targets = ExternalUtil.resolve_python_targets(self.repo_path, path_arg)
        targets = self._filter_targets(targets, path_arg)
        if not targets:
            return []

        violations = []
        for chunk in ExternalUtil.chunked(targets, ExternalUtil.MAX_FILES_PER_CALL):
            cmd = ["ruff", "check", "--output-format=json", *chunk]
            result = ExternalUtil.run_subproc(cmd, self.repo_path, return_result=True)
            assert isinstance(result, subprocess.CompletedProcess)

            payload = (result.stdout or "").strip()
            if not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                logger.warning("Ruff output not JSON: %s...", payload[:200])
                continue

            for item in data:
                violations.append(
                    {
                        "file": item.get("filename", ""),
                        "line": item.get("location", {}).get("row", 0),
                        "rule_id": item.get("code", "RUFF"),
                        "message": item.get("message", ""),
                        "severity": "error",
                        "aal": "AAL-2",
                        "domain": ["universal"],
                        "closure_level": "guarded",
                        "evidence_refs": [],
                        "context": "",
                    }
                )

        violations.extend(self._run_aes_ruff_checks(path_arg))
        return self._dedupe_violations(violations)

    def _run_aes_ruff_checks(self, path_arg: str) -> list[dict[str, Any]]:
        rules_path = os.path.join(self.repo_path, "standards", "AES_RULES.json")
        if not os.path.exists(rules_path):
            return []

        try:
            from core.aes import AESRuleRegistry
            from saguaro.sentinel.engines.aes import AESEngine

            registry = AESRuleRegistry()
            registry.load(rules_path)
            ruff_rule_ids = {
                rule.id
                for rule in registry.get_rules_for_engine("ruff")
                if str(getattr(rule, "execution_mode", "static") or "static")
                == "static"
            }
            if not ruff_rule_ids:
                return []

            engine = AESEngine(self.repo_path)
            policy = dict(self.policy_config)
            policy["allow_ruff_rules"] = True
            engine.set_policy(policy)
            return [
                violation
                for violation in engine.run(path_arg=path_arg)
                if str(violation.get("rule_id", "")) in ruff_rule_ids
            ]
        except Exception as exc:
            logger.warning("Failed to run Ruff-backed AES checks: %s", exc)
            return []

    @staticmethod
    def _dedupe_violations(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[str, int, str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for violation in sorted(
            violations,
            key=lambda item: (
                str(item.get("file", "")),
                int(item.get("line", 0) or 0),
                str(item.get("rule_id", "")),
                str(item.get("message", "")),
            ),
        ):
            key = (
                str(violation.get("file", "")),
                int(violation.get("line", 0) or 0),
                str(violation.get("rule_id", "")),
                str(violation.get("message", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(violation)
        return deduped

    def fix(self, violation: dict[str, Any]) -> bool:
        """Attempts to fix a Ruff violation using targeted --select."""
        try:
            filename = violation.get("file")
            rule_id = violation.get("rule_id")

            if not filename:
                return False

            cmd = ["ruff", "check", "--fix"]
            if rule_id and rule_id != "RUFF":
                cmd.extend(["--select", rule_id])
            cmd.append(filename)

            logger.info("Auto-fixing %s rule=%s with Ruff...", filename, rule_id)
            output = ExternalUtil.run_subproc(cmd, self.repo_path)
            lowered = output.lower()
            return (
                "fixed" in lowered
                or "all checks passed" in lowered
                or not lowered.strip()
            )
        except Exception as e:
            logger.error("Ruff fix failed: %s", e)
            return False

    def _filter_targets(self, targets: list[str], path_arg: str) -> list[str]:
        target_abs = ExternalUtil._resolve_target(self.repo_path, path_arg)
        repo_abs = os.path.abspath(self.repo_path)
        enforce_authoritative = target_abs == repo_abs
        authoritative_root = (
            str(
                self.policy_config.get("aes_authoritative_package_root", "saguaro")
                or ""
            )
            .strip()
            .strip("/")
        )
        excluded_roots = [
            str(item).replace("\\", "/").rstrip("/") + "/"
            for item in self.policy_config.get("excluded_paths", []) or []
        ]

        filtered: list[str] = []
        for rel_path in targets:
            normalized = rel_path.replace("\\", "/").lstrip("./")
            if any(
                normalized == root.rstrip("/") or normalized.startswith(root)
                for root in excluded_roots
            ):
                continue
            if (
                enforce_authoritative
                and authoritative_root
                and os.path.isdir(os.path.join(self.repo_path, authoritative_root))
                and normalized != authoritative_root
                and not normalized.startswith(authoritative_root + "/")
            ):
                continue
            filtered.append(rel_path)

        return filtered


class MypyEngine(BaseEngine):
    """Provide MypyEngine support."""
    _PATTERN = re.compile(r"^([^:]+):(\d+):\s*([a-z]+):\s*(.*)$")

    def run(self, path_arg: str = ".") -> list[dict[str, Any]]:
        """Handle run."""
        logger.info("Running MypyEngine...")
        targets = ExternalUtil.resolve_python_targets(self.repo_path, path_arg)
        if not targets:
            return []

        violations = []
        seen: set[tuple[str, int, str]] = set()
        for chunk in ExternalUtil.chunked(targets, ExternalUtil.MAX_FILES_PER_CALL):
            cmd = ["mypy", "--no-error-summary", "--no-pretty", *chunk]
            result = ExternalUtil.run_subproc(cmd, self.repo_path, return_result=True)
            assert isinstance(result, subprocess.CompletedProcess)
            output = (result.stdout or "") + "\n" + (result.stderr or "")

            for line in output.splitlines():
                match = self._PATTERN.match(line)
                if not match:
                    continue
                fpath, lineno, severity, msg = match.groups()
                key = (fpath.strip(), int(lineno), msg.strip())
                if key in seen:
                    continue
                seen.add(key)
                violations.append(
                    {
                        "file": fpath.strip(),
                        "line": int(lineno),
                        "rule_id": "MYPY",
                        "message": msg.strip(),
                        "severity": severity,
                        "aal": "AAL-2",
                        "domain": ["universal"],
                        "closure_level": "guarded",
                        "evidence_refs": [],
                        "context": "",
                    }
                )
        return violations


class VultureEngine(BaseEngine):
    """Provide VultureEngine support."""
    _PATTERN = re.compile(r"^([^:]+):(\d+):\s*(.*)$")

    def run(self, path_arg: str = ".") -> list[dict[str, Any]]:
        """Handle run."""
        logger.info("Running VultureEngine...")
        targets = ExternalUtil.resolve_python_targets(self.repo_path, path_arg)
        if not targets:
            return []

        violations = []
        seen: set[tuple[str, int, str]] = set()
        for chunk in ExternalUtil.chunked(targets, ExternalUtil.MAX_FILES_PER_CALL):
            cmd = ["vulture", *chunk]
            result = ExternalUtil.run_subproc(cmd, self.repo_path, return_result=True)
            assert isinstance(result, subprocess.CompletedProcess)
            output = (result.stdout or "") + "\n" + (result.stderr or "")

            for line in output.splitlines():
                match = self._PATTERN.match(line)
                if not match:
                    continue
                fpath, lineno, msg = match.groups()
                key = (fpath.strip(), int(lineno), msg.strip())
                if key in seen:
                    continue
                seen.add(key)
                violations.append(
                    {
                        "file": fpath.strip(),
                        "line": int(lineno),
                        "rule_id": "VULTURE",
                        "message": msg.strip(),
                        "severity": "warning",
                        "aal": "AAL-2",
                        "domain": ["universal"],
                        "closure_level": "advisory",
                        "evidence_refs": [],
                        "context": "",
                    }
                )
        return violations
