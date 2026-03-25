"""
Automatic Verification System

Validates code changes automatically:
1. Syntax checking
2. Test execution
3. Linting
4. Type checking
5. Self-correction on failures
"""

import ast
import hashlib
import os
import re
import subprocess
from typing import Any, Dict, List
from rich.table import Table

from saguaro.parsing import RuntimeSymbolResolver
from saguaro.requirements.model import CounterexampleRecord


class AutoVerifier:
    """
    Automatically verify code changes and trigger self-correction.
    """

    def __init__(self, registry, console):
        self.registry = registry
        self.console = console
        self.symbol_resolver = RuntimeSymbolResolver(".")
        self.last_results: Dict[str, Any] = {}

    def verify_changes(self, modified_files: List[str]) -> Dict[str, Any]:
        """
        Run all verification checks on modified files.

        Returns dict with results:
        {
            "syntax": {"passed": True/False, "errors": [...]},
            "tests": {"passed": True/False, "output": "..."},
            "lint": {"passed": True/False, "warnings": [...]},
            "sentinel": {"passed": True/False, "violations": [...]},
            "all_passed": True/False
        }
        """
        self.console.print("\n[cyan]═══ Running Verification ═══[/cyan]")

        results = {
            "syntax": self._check_syntax(modified_files),
            "tests": self._run_tests(modified_files),
            "lint": self._check_lint(modified_files),
            "qsg_runtime": self._check_qsg_runtime(modified_files),
            "sentinel": self._check_sentinel(modified_files),
            "all_passed": True,
        }

        # Determine overall pass/fail with strict fail-closed semantics.
        required = [results["syntax"], results["qsg_runtime"], results["sentinel"]]
        optional = [results["tests"], results["lint"]]
        optional_required = [check for check in optional if not check.get("skipped", False)]
        results["all_passed"] = all(
            check.get("passed", False) for check in required + optional_required
        )
        results["runtime_symbols"] = sorted(
            {
                *results["sentinel"].get("runtime_symbols", []),
                *[
                    symbol
                    for check in (results["syntax"], results["tests"], results["lint"])
                    for symbol in check.get("runtime_symbols", [])
                ],
            }
        )
        results["counterexamples"] = [
            *self._syntax_counterexamples(results["syntax"]),
            *self._test_counterexamples(results["tests"]),
            *list(results["sentinel"].get("counterexamples", [])),
        ]
        recovered = self.symbol_resolver.resolve_output(
            "\n".join(
                [
                    *[
                        f'File "{item.get("file")}", line {item.get("line", 1)}, in <syntax>\n{item.get("message", "")}'
                        for item in results["syntax"].get("errors", [])
                    ],
                    str(results["tests"].get("output", "") or ""),
                ]
            )
        )
        results["runtime_symbols"] = sorted(
            set(results["runtime_symbols"])
            | {
                str(item.get("symbol"))
                for item in recovered
                if str(item.get("symbol") or "").strip()
            }
        )

        # Display results
        self._display_results(results)
        self.last_results = results

        return results

    def _syntax_counterexamples(self, syntax_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        counterexamples: List[Dict[str, Any]] = []
        for error in syntax_results.get("errors", []):
            payload = f"{error.get('file')}|{error.get('line')}|{error.get('message')}"
            digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
            counterexamples.append(
                CounterexampleRecord(
                    id=f"CX-{digest}".upper(),
                    target_relation_id=str(error.get("file") or "<syntax>"),
                    counterexample_type="syntax_error",
                    input_or_state=f"{error.get('file')}:{error.get('line')}",
                    observed_failure=str(error.get("message") or "syntax error"),
                    severity="high",
                    repro_steps=(f"python -m py_compile {error.get('file')}",),
                ).to_dict()
            )
        return counterexamples

    def _test_counterexamples(self, test_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        if test_results.get("passed", True) or test_results.get("skipped"):
            return []
        output = str(test_results.get("output", "") or "")
        digest = hashlib.sha1(output.encode("utf-8")).hexdigest()[:12]
        return [
            CounterexampleRecord(
                id=f"CX-{digest}".upper(),
                target_relation_id="pytest",
                counterexample_type="test_failure",
                input_or_state="pytest",
                observed_failure=output[-400:],
                severity="high",
                repro_steps=("pytest -xvs",),
            ).to_dict()
        ]

    def _check_syntax(self, files: List[str]) -> Dict[str, Any]:
        """
        Check Python syntax using AST parser.
        """
        self.console.print("  [dim]→ Checking syntax...[/dim]")

        errors = []

        for file_path in files:
            if not file_path.endswith(".py"):
                continue

            try:
                content = self.registry.dispatch("read_file", {"file_path": file_path})
                if content and not content.startswith("Error"):
                    ast.parse(content)
                    self.console.print(f"  [green]✓ {file_path} - syntax OK[/green]")
            except SyntaxError as e:
                errors.append(
                    {"file": file_path, "line": e.lineno, "message": str(e.msg)}
                )
                self.console.print(f"  [red]✗ {file_path}:{e.lineno} - {e.msg}[/red]")

        return {"passed": len(errors) == 0, "errors": errors}

    def _run_tests(self, files: List[str]) -> Dict[str, Any]:
        """
        Run tests related to modified files.
        """
        self.console.print("  [dim]→ Running tests...[/dim]")

        # Find related test files
        test_files = self._find_test_files(files)

        if not test_files:
            self.console.print("  [yellow]⚠ No tests found[/yellow]")
            return {"passed": True, "output": "No tests found", "skipped": True}

        # Run pytest
        try:
            result = subprocess.run(
                ["pytest", "-xvs"] + test_files,
                capture_output=True,
                text=True,
                timeout=60,
            )

            passed = result.returncode == 0

            if passed:
                self.console.print("  [green]✓ All tests passed[/green]")
            else:
                self.console.print("  [red]✗ Tests failed[/red]")

            return {
                "passed": passed,
                "output": result.stdout + result.stderr,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            self.console.print("  [red]✗ Tests timed out[/red]")
            return {
                "passed": False,
                "output": "Tests timed out after 60 seconds",
                "return_code": -1,
            }
        except FileNotFoundError:
            self.console.print("  [yellow]⚠ pytest not found[/yellow]")
            return {
                "passed": False,
                "output": "pytest not installed",
                "skipped": False,
            }

    def _check_lint(self, files: List[str]) -> Dict[str, Any]:
        """
        Run linting checks (flake8, pylint, etc.)
        """
        self.console.print("  [dim]→ Running linter...[/dim]")

        warnings = []
        python_files = [f for f in files if f.endswith(".py")]

        if not python_files:
            return {"passed": True, "warnings": [], "skipped": True}

        # Try flake8 first
        try:
            result = subprocess.run(
                ["flake8", "--max-line-length=120"] + python_files,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.stdout:
                warnings = result.stdout.strip().split("\n")
                self.console.print(
                    f"  [yellow]⚠ {len(warnings)} linting warnings[/yellow]"
                )
            else:
                self.console.print("  [green]✓ No linting issues[/green]")

            return {"passed": len(warnings) == 0, "warnings": warnings}

        except FileNotFoundError:
            self.console.print("  [yellow]⚠ flake8 not found[/yellow]")
            return {
                "passed": False,
                "warnings": ["flake8 not installed"],
                "skipped": False,
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "warnings": ["Linting timed out"]}

    def _check_qsg_runtime(self, files: List[str]) -> Dict[str, Any]:
        del files
        self.console.print("  [dim]→ Checking QSG runtime invariants...[/dim]")

        try:
            from tools.verify import verify_qsg_runtime

            result = verify_qsg_runtime(".")
            details = list(result.details or [])
            if result.passed:
                self.console.print("  [green]✓ QSG runtime invariants passed[/green]")
            else:
                self.console.print("  [red]✗ QSG runtime invariants failed[/red]")
                for detail in details[:3]:
                    self.console.print(f"    - {detail}")
            return {
                "passed": bool(result.passed),
                "violations": details,
                "message": result.message,
            }
        except Exception as exc:
            self.console.print(f"  [yellow]⚠ QSG runtime check failed: {exc}[/yellow]")
            return {
                "passed": False,
                "violations": [str(exc)],
                "message": "QSG runtime invariant verification crashed",
                "skipped": False,
            }

    def _check_sentinel(self, files: List[str]) -> Dict[str, Any]:
        """
        Run Saguaro Sentinel verification (Code Graph & Policy).
        """
        self.console.print("  [dim]→ Running Sentinel...[/dim]")

        try:
            # Lazy import to avoid circular dependency
            from saguaro.sentinel.verifier import SentinelVerifier

            # Initialize strict AES verification stack.
            verifier = SentinelVerifier(
                repo_path=".", engines=["native", "ruff", "semantic", "aes"]
            )

            # Scope verification to touched files, de-duplicating across paths.
            targets = [f.strip() for f in files if isinstance(f, str) and f.strip()]
            if not targets:
                targets = ["."]
            targets = list(dict.fromkeys(targets))

            violations: List[Any] = []
            seen: set[tuple[str, str, str, str]] = set()
            for path_arg in targets:
                scoped = verifier.verify_all(path_arg=path_arg)
                if not isinstance(scoped, list):
                    raise RuntimeError("Sentinel returned malformed violation payload")
                for violation in scoped:
                    if isinstance(violation, dict):
                        key = (
                            str(violation.get("rule_id", "")),
                            str(violation.get("file", "")),
                            str(violation.get("line", "")),
                            str(violation.get("message", "")),
                        )
                    else:
                        key = ("SENTINEL-MALFORMED", str(type(violation)), "", "")
                    if key in seen:
                        continue
                    seen.add(key)
                    if isinstance(violation, dict):
                        violations.append(violation)
                    else:
                        violations.append(
                            {
                                "rule_id": "SENTINEL-MALFORMED-VIOLATION",
                                "file": "<sentinel>",
                                "line": 0,
                                "severity": "P0",
                                "closure_level": "blocking",
                                "message": f"Malformed Sentinel violation: {violation!r}",
                            }
                        )

            # Filter violations relevant to modified files if possible
            # Or just report all structural issues (since graph checks are global)
            relevant_violations = [
                v for v in violations if self._is_blocking_violation(v)
            ]

            if relevant_violations:
                self.console.print(
                    f"  [red]✗ Sentinel found {len(relevant_violations)} violations[/red]"
                )
                for v in relevant_violations[:3]:
                    self.console.print(f"    - {v.get('message')}")
            else:
                self.console.print("  [green]✓ Sentinel checks passed[/green]")

            runtime_symbols = self.recover_runtime_symbols(relevant_violations)
            counterexamples = self.build_counterexamples(relevant_violations)

            return {
                "passed": len(relevant_violations) == 0,
                "violations": relevant_violations,
                "runtime_symbols": runtime_symbols,
                "counterexamples": counterexamples,
            }

        except Exception as e:
            self.console.print(f"  [yellow]⚠ Sentinel check failed: {e}[/yellow]")
            return {
                "passed": False,
                "violations": [],
                "runtime_symbols": [],
                "counterexamples": [],
                "skipped": False,
                "error": str(e),
            }

    @staticmethod
    def recover_runtime_symbols(violations: List[Any]) -> List[str]:
        """Best-effort runtime symbol recovery from verifier outputs."""
        symbols: set[str] = set()
        for violation in violations:
            if not isinstance(violation, dict):
                continue
            for key in ("symbol", "entity", "frame", "target"):
                value = violation.get(key)
                if isinstance(value, str) and value.strip():
                    symbols.add(value.strip())
            message = str(violation.get("message", "") or "")
            for token in re.findall(r"`([^`\n]+)`", message):
                cleaned = token.strip()
                if cleaned and any(ch.isalpha() for ch in cleaned):
                    symbols.add(cleaned)
        return sorted(symbols)

    @staticmethod
    def build_counterexamples(violations: List[Any]) -> List[Dict[str, Any]]:
        """Normalize blocking violations into replayable counterexamples."""
        counterexamples: List[Dict[str, Any]] = []
        for index, violation in enumerate(violations, start=1):
            if not isinstance(violation, dict):
                continue
            file_path = str(violation.get("file") or "<unknown>")
            line = int(violation.get("line", 0) or 0)
            rule_id = str(violation.get("rule_id") or f"VIOLATION-{index}")
            counterexamples.append(
                {
                    "id": f"counterexample::{rule_id.lower()}::{index}",
                    "target_relation_id": rule_id,
                    "counterexample_type": "verification_violation",
                    "input_or_state": f"{file_path}:{line}",
                    "observed_failure": str(violation.get("message") or rule_id),
                    "severity": str(violation.get("severity") or "unknown"),
                    "repro_steps": [
                        f"Run verification for {file_path}",
                        f"Observe rule {rule_id} at line {line}",
                    ],
                }
            )
        return counterexamples

    def _is_blocking_violation(self, violation: Any) -> bool:
        """Fail closed for malformed violations and block on policy-critical severities."""
        if not isinstance(violation, dict):
            return True

        closure_level = str(violation.get("closure_level", "")).lower()
        severity = str(violation.get("severity", "")).upper()

        if closure_level in {"blocking", "guarded"}:
            return True

        if severity in {"P0", "P1", "ERROR", "CRITICAL"}:
            return True

        if severity in {"P2", "P3", "WARN", "WARNING", "INFO"}:
            return False

        # Unknown severity/closure should not silently pass.
        return True

    def _find_test_files(self, files: List[str]) -> List[str]:
        """
        Find test files related to the modified files.

        Heuristics:
        - test_<filename>.py
        - <filename>_test.py
        - tests/test_<filename>.py
        """
        test_files = []

        for file_path in files:
            if file_path.endswith(".py"):
                # Get base name
                base_name = os.path.basename(file_path).replace(".py", "")

                # Try common patterns
                candidates = [
                    f"test_{base_name}.py",
                    f"{base_name}_test.py",
                    f"tests/test_{base_name}.py",
                    f"tests/{base_name}_test.py",
                ]

                for candidate in candidates:
                    if os.path.exists(candidate):
                        test_files.append(candidate)
                        break

        return test_files

    def _display_results(self, results: Dict[str, Any]):
        """
        Display verification results in a nice table.
        """
        table = Table(title="Verification Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")

        # Syntax
        syntax = results["syntax"]
        status = "[green]✓ PASS[/green]" if syntax["passed"] else "[red]✗ FAIL[/red]"
        details = f"{len(syntax['errors'])} errors" if syntax.get("errors") else "OK"
        table.add_row("Syntax", status, details)

        # Tests
        tests = results["tests"]
        if tests.get("skipped"):
            status = "[yellow]⊘ SKIP[/yellow]"
        else:
            status = "[green]✓ PASS[/green]" if tests["passed"] else "[red]✗ FAIL[/red]"
        details = tests.get("output", "")[:50]
        table.add_row("Tests", status, details)

        # Lint
        lint = results["lint"]
        if lint.get("skipped"):
            status = "[yellow]⊘ SKIP[/yellow]"
        else:
            status = (
                "[green]✓ PASS[/green]" if lint["passed"] else "[yellow]⚠ WARN[/yellow]"
            )
        details = f"{len(lint.get('warnings', []))} warnings"
        table.add_row("Lint", status, details)

        # Sentinel
        sentinel = results["sentinel"]
        if sentinel.get("skipped"):
            status = "[yellow]⊘ SKIP[/yellow]"
        else:
            status = (
                "[green]✓ PASS[/green]" if sentinel["passed"] else "[red]✗ FAIL[/red]"
            )
        details = f"{len(sentinel.get('violations', []))} violations"
        table.add_row("Sentinel", status, details)

        qsg_runtime = results["qsg_runtime"]
        if qsg_runtime.get("skipped"):
            status = "[yellow]⊘ SKIP[/yellow]"
        else:
            status = (
                "[green]✓ PASS[/green]"
                if qsg_runtime["passed"]
                else "[red]✗ FAIL[/red]"
            )
        details = (
            qsg_runtime.get("message")
            or f"{len(qsg_runtime.get('violations', []))} violations"
        )
        table.add_row("QSG Runtime", status, details)

        self.console.print(table)

        # Overall
        if results["all_passed"]:
            self.console.print("\n[green]✓ All checks passed![/green]")
        else:
            self.console.print("\n[red]✗ Some checks failed[/red]")

    def attempt_fix(self, results: Dict[str, Any], modified_files: List[str]) -> bool:
        """
        Attempt to automatically fix verification failures.

        Returns True if fixes were successful.
        """
        self.console.print("\n[cyan]Attempting automatic fixes...[/cyan]")

        # Fix syntax errors
        if not results["syntax"]["passed"]:
            for error in results["syntax"]["errors"]:
                self.console.print(
                    f"  [yellow]Attempting to fix syntax error in {error['file']}...[/yellow]"
                )
                # TODO: Use model to fix syntax error

        # Fix test failures
        if not results["tests"]["passed"] and not results["tests"].get("skipped"):
            self.console.print("  [yellow]Attempting to fix test failures...[/yellow]")
            # TODO: Analyze test output and fix code

        # Fix lint warnings (lower priority)
        if not results["lint"]["passed"] and not results["lint"].get("skipped"):
            for warning in results["lint"]["warnings"][:5]:  # Limit to first 5
                self.console.print(f"  [yellow]Linting: {warning}[/yellow]")

        return False  # Not implemented yet


class VerificationLoop:
    """
    Continuous verification with self-correction.
    """

    def __init__(self, verifier: AutoVerifier, smart_editor):
        self.verifier = verifier
        self.smart_editor = smart_editor
        self.console = verifier.console
        self.last_results: Dict[str, Any] = {}

    def verify_with_retry(
        self, modified_files: List[str], max_attempts: int = 3
    ) -> bool:
        """
        Verify changes with automatic retry/fix.
        """
        for attempt in range(max_attempts):
            self.console.print(
                f"\n[cyan]Verification attempt {attempt + 1}/{max_attempts}[/cyan]"
            )

            results = self.verifier.verify_changes(modified_files)
            self.last_results = results

            if results["all_passed"]:
                return True

            if attempt < max_attempts - 1:
                # Try to fix
                fixed = self.verifier.attempt_fix(results, modified_files)
                if not fixed:
                    self.console.print(
                        "[yellow]Automatic fix not available, continuing...[/yellow]"
                    )

        self.last_results = results
        return False
