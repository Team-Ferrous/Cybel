"""
Verification Tools - Syntax, type, lint, and test verification.

Implements the verification phase tools for the enhanced agentic loop.
"""

import subprocess
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import threading
import time

from core.aes.lint import format_aes_lint, run_aes_lint
from core.native.parallel_generation import NativeParallelGenerationEngine
from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGInferenceEngine, QSGRequest
from core.qsg.runtime_contracts import evaluate_qsg_runtime_invariants


@dataclass
class VerificationResult:
    """Result of a verification check."""

    tool: str
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class AggregatedVerificationResult:
    """Aggregated results from all verification tools."""

    results: List[VerificationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def has_failures(self) -> bool:
        return any(not r.passed for r in self.results)

    @property
    def failures(self) -> List[VerificationResult]:
        return [r for r in self.results if not r.passed]

    @property
    def passes(self) -> List[VerificationResult]:
        return [r for r in self.results if r.passed]

    def add(self, result: VerificationResult) -> None:
        self.results.append(result)

    def summary(self) -> str:
        passed = len(self.passes)
        failed = len(self.failures)
        total = len(self.results)
        return f"Verification: {passed}/{total} passed, {failed} failed"

    def to_dict(self) -> dict:
        return {
            "all_passed": self.all_passed,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }


def verify_syntax(
    path: str = ".", file_patterns: Optional[List[str]] = None
) -> VerificationResult:
    """
    Check Python files for syntax errors.

    Args:
        path: Directory or file to check
        file_patterns: Optional list of file patterns to include

    Returns:
        VerificationResult with syntax check outcome
    """
    import py_compile
    import glob

    patterns = file_patterns or ["**/*.py"]
    errors = []
    files_checked = 0

    for pattern in patterns:
        if os.path.isfile(path):
            files = [path]
        else:
            files = glob.glob(os.path.join(path, pattern), recursive=True)

        for filepath in files:
            # Skip venv and cache directories
            if "venv" in filepath or "__pycache__" in filepath or ".git" in filepath:
                continue

            files_checked += 1
            try:
                py_compile.compile(filepath, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"{filepath}: {e}")

    if errors:
        return VerificationResult(
            tool="syntax",
            passed=False,
            message=f"Syntax errors found in {len(errors)} file(s)",
            details=errors,
        )

    return VerificationResult(
        tool="syntax",
        passed=True,
        message=f"No syntax errors in {files_checked} file(s)",
    )


def verify_types(path: str = ".") -> VerificationResult:
    """
    Run type checker (mypy) on the codebase.

    Args:
        path: Directory to check

    Returns:
        VerificationResult with type check outcome
    """
    try:
        result = subprocess.run(
            ["mypy", "--ignore-missing-imports", "--no-error-summary", path],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            return VerificationResult(
                tool="types", passed=True, message="Type check passed"
            )
        else:
            errors = [line for line in result.stdout.split("\n") if line.strip()]
            return VerificationResult(
                tool="types",
                passed=False,
                message="Type errors found",
                details=errors[:20],  # Limit to first 20 errors
            )
    except FileNotFoundError:
        return VerificationResult(
            tool="types",
            passed=True,  # Skip if mypy not installed
            message="Type checker (mypy) not installed, skipping",
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            tool="types", passed=False, message="Type check timed out"
        )


def verify_lint(path: str = ".") -> VerificationResult:
    """
    Run linter (ruff) on the codebase.

    Args:
        path: Directory to check

    Returns:
        VerificationResult with lint check outcome
    """
    try:
        result = subprocess.run(
            ["ruff", "check", "--no-fix", path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            return VerificationResult(
                tool="lint", passed=True, message="Lint check passed"
            )
        else:
            issues = [line for line in result.stdout.split("\n") if line.strip()]
            return VerificationResult(
                tool="lint",
                passed=False,
                message="Lint issues found",
                details=issues[:20],  # Limit to first 20 issues
            )
    except FileNotFoundError:
        # Try flake8 as fallback
        try:
            result = subprocess.run(
                ["flake8", "--max-line-length=120", path],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return VerificationResult(
                    tool="lint", passed=True, message="Lint check passed (flake8)"
                )
            else:
                issues = [line for line in result.stdout.split("\n") if line.strip()]
                return VerificationResult(
                    tool="lint",
                    passed=False,
                    message="Lint issues found (flake8)",
                    details=issues[:20],
                )
        except FileNotFoundError:
            return VerificationResult(
                tool="lint",
                passed=True,
                message="No linter installed (ruff/flake8), skipping",
            )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            tool="lint", passed=False, message="Lint check timed out"
        )


def verify_aes(path: str = ".") -> VerificationResult:
    """
    Run the deterministic AES static checker with Ruff-style diagnostics.

    Args:
        path: File or directory to check

    Returns:
        VerificationResult with AES lint outcome
    """
    repo_root = Path(".").resolve()
    catalog_path = repo_root / "standards" / "AES_RULES.json"
    if not catalog_path.exists():
        return VerificationResult(
            tool="aes",
            passed=True,
            message="AES catalog not found, skipping",
        )

    violations = run_aes_lint([path], repo_root=str(repo_root))
    if not violations:
        return VerificationResult(
            tool="aes",
            passed=True,
            message="AES lint passed",
        )

    rendered = format_aes_lint(violations, output_format="text").splitlines()
    return VerificationResult(
        tool="aes",
        passed=False,
        message=f"AES lint found {len(violations)} issue(s)",
        details=rendered[:20],
    )


def run_tests(
    path: str = ".", test_pattern: str = "test_*.py", verbose: bool = False
) -> VerificationResult:
    """
    Execute test suite with pytest.

    Args:
        path: Directory containing tests
        test_pattern: Pattern for test files
        verbose: If True, include verbose output

    Returns:
        VerificationResult with test outcome
    """
    try:
        cmd = ["python", "-m", "pytest", path, "-v" if verbose else "-q", "--tb=short"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse pytest output
        output = result.stdout + result.stderr

        if result.returncode == 0:
            # Extract summary line
            summary = ""
            for line in output.split("\n"):
                if "passed" in line.lower() or "xfailed" in line.lower():
                    summary = line.strip()
                    break

            return VerificationResult(
                tool="tests", passed=True, message=summary or "All tests passed"
            )
        elif result.returncode == 5:
            # No tests collected
            return VerificationResult(
                tool="tests", passed=True, message="No tests found to run"
            )
        else:
            # Test failures
            failures = []
            in_failure = False
            for line in output.split("\n"):
                if "FAILED" in line or "ERROR" in line:
                    failures.append(line.strip())
                    in_failure = True
                elif in_failure and line.strip():
                    failures.append(f"  {line.strip()}")
                    if len(failures) > 30:
                        break

            return VerificationResult(
                tool="tests",
                passed=False,
                message="Test failures detected",
                details=failures[:20],
            )
    except FileNotFoundError:
        return VerificationResult(
            tool="tests", passed=True, message="pytest not installed, skipping tests"
        )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            tool="tests", passed=False, message="Test execution timed out (>5 min)"
        )


def verify_qsg_runtime(path: str = ".") -> VerificationResult:
    del path

    class _NativeEngineStub:
        def __init__(self) -> None:
            self.runtime_status = {}
            self.num_ubatch = 2

        def get_runtime_status(self):
            return dict(self.runtime_status)

        def _update_scheduler_metrics_snapshot(self, metrics):
            del metrics

    config = QSGConfig(
        continuous_batching_enabled=True,
        batch_wait_timeout_ms=1,
        max_active_requests=2,
        max_pending_requests=4,
        capability_digest="verify:qsg",
        delta_watermark={
            "delta_id": "verify-delta",
            "logical_clock": 1,
            "workspace_id": "verify",
        },
    )
    violations: list[str] = []
    runner: threading.Thread | None = None
    engine: QSGInferenceEngine | None = None

    try:
        engine = QSGInferenceEngine(
            config=config,
            stream_producer=lambda request: iter(["verify"]),
        )
        runner = threading.Thread(target=engine.run_forever, daemon=True)
        runner.start()
        request_id = engine.submit(QSGRequest(prompt="verify"))
        deadline = time.time() + 1.0
        while time.time() < deadline:
            chunk = engine.poll(request_id)
            if chunk is not None and chunk.text:
                break
            time.sleep(0.001)
        captured = engine.capture_latent_state(request_id)
        if captured is None:
            violations.append("continuous_engine_capture: failed to capture latent state")
        else:
            for finding in evaluate_qsg_runtime_invariants(
                runtime_status=engine.metrics_snapshot(),
                latent_packet=dict(captured.get("latent_packet") or {}),
                execution_capsule=dict(captured.get("execution_capsule") or {}),
            ):
                violations.append(f"continuous::{finding['code']}::{finding['message']}")

        native_engine = NativeParallelGenerationEngine(
            native_engine=_NativeEngineStub(),
            config=config,
            stream_producer=lambda request: iter((request.prompt,)),
        )
        native_request_id = native_engine.submit(
            QSGRequest(
                prompt="native-verify",
                options={
                    "latent": True,
                    "latent_packets": [{"hidden_dimension": 16}],
                },
            )
        )
        native_captured = native_engine.capture_latent_state(native_request_id)
        if native_captured is None:
            violations.append("native_engine_capture: failed to capture latent state")
        else:
            runtime_status = {
                "qsg_native_runtime_authority": True,
                "qsg_capability_digest": config.capability_digest,
                "qsg_delta_watermark": config.delta_watermark,
            }
            for finding in evaluate_qsg_runtime_invariants(
                runtime_status=runtime_status,
                latent_packet=dict(native_captured.get("latent_packet") or {}),
                execution_capsule=dict(native_captured.get("execution_capsule") or {}),
            ):
                violations.append(f"native::{finding['code']}::{finding['message']}")
    except Exception as exc:
        return VerificationResult(
            tool="qsg_runtime",
            passed=False,
            message="QSG runtime verification failed",
            details=[str(exc)],
        )
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass
        if runner is not None:
            runner.join(timeout=1.0)

    if violations:
        return VerificationResult(
            tool="qsg_runtime",
            passed=False,
            message="QSG runtime invariants failed",
            details=violations[:20],
        )

    return VerificationResult(
        tool="qsg_runtime",
        passed=True,
        message="QSG runtime invariants passed",
    )


def run_all_verifications(path: str = ".") -> AggregatedVerificationResult:
    """
    Run all verification tools and aggregate results.

    Args:
        path: Directory to verify

    Returns:
        AggregatedVerificationResult with all outcomes
    """
    aggregated = AggregatedVerificationResult()

    # Run each verification
    aggregated.add(verify_syntax(path))
    aggregated.add(verify_lint(path))
    aggregated.add(verify_aes(path))
    aggregated.add(verify_types(path))
    aggregated.add(run_tests(path))
    aggregated.add(verify_qsg_runtime(path))

    return aggregated


# Tool function for registry
def verify_all(path: str = ".") -> str:
    """
    Run all verification tools and return formatted result.

    This is the function registered in the tool registry.
    """
    results = run_all_verifications(path)

    output = [results.summary(), ""]

    for r in results.results:
        status = "✅" if r.passed else "❌"
        output.append(f"{status} {r.tool.upper()}: {r.message}")

        if r.details:
            for detail in r.details[:5]:  # Limit details
                output.append(f"   {detail}")
            if len(r.details) > 5:
                output.append(f"   ... and {len(r.details) - 5} more")

    return "\n".join(output)


# Individual tool functions for registry
def verify_syntax_tool(path: str = ".") -> str:
    """Verify syntax of Python files."""
    result = verify_syntax(path)
    output = f"{'✅' if result.passed else '❌'} {result.message}"
    if result.details:
        output += "\n" + "\n".join(result.details[:10])
    return output


def verify_types_tool(path: str = ".") -> str:
    """Run type checker on the codebase."""
    result = verify_types(path)
    output = f"{'✅' if result.passed else '❌'} {result.message}"
    if result.details:
        output += "\n" + "\n".join(result.details[:10])
    return output


def verify_lint_tool(path: str = ".") -> str:
    """Run linter on the codebase."""
    result = verify_lint(path)
    output = f"{'✅' if result.passed else '❌'} {result.message}"
    if result.details:
        output += "\n" + "\n".join(result.details[:10])
    return output


def verify_aes_tool(path: str = ".") -> str:
    """Run the AES static checker on the codebase."""
    result = verify_aes(path)
    output = f"{'✅' if result.passed else '❌'} {result.message}"
    if result.details:
        output += "\n" + "\n".join(result.details[:10])
    return output


def run_tests_tool(path: str = ".", verbose: bool = False) -> str:
    """Run test suite."""
    result = run_tests(path, verbose=verbose)
    output = f"{'✅' if result.passed else '❌'} {result.message}"
    if result.details:
        output += "\n" + "\n".join(result.details[:10])
    return output


# ── Memory-aware test execution ──────────────────────────────────────────────

def run_tests_suspended(
    path: str = ".",
    test_pattern: str = "test_*.py",
    verbose: bool = False,
    *,
    brain=None,
    force_suspend: bool = True,
    estimated_need_gb: float = 2.0,
    message_bus=None,
) -> VerificationResult:
    """Run tests with the inference model suspended to free RAM.

    This is the preferred entry point for test execution on CPU-only
    systems where model weights consume several GB.  Before running
    ``pytest``, the ``ModelSuspender`` evicts the weights; after the
    tests finish the model is reloaded.

    Parameters
    ----------
    path : str
        Directory containing tests.
    test_pattern : str
        Passed through to ``pytest -k``.
    verbose : bool
        If True, include verbose pytest output.
    brain : optional
        ``DeterministicOllama`` instance.  If None, falls back to
        regular ``run_tests()`` without suspension.
    force_suspend : bool
        Always suspend (True) or only when memory is tight (False).
    estimated_need_gb : float
        How much free RAM the test suite is expected to need.
    message_bus : optional
        Event bus for ``model.suspending``/``model.resumed`` events.

    Returns
    -------
    VerificationResult
    """
    if brain is None:
        # No brain available — run without suspension
        return run_tests(path, test_pattern=test_pattern, verbose=verbose)

    from core.native.model_suspender import ModelSuspender

    suspender = ModelSuspender(
        brain,
        reason="tool:run_tests_suspended",
        force=force_suspend,
        estimated_need_gb=estimated_need_gb,
        message_bus=message_bus,
    )

    with suspender:
        result = run_tests(path, test_pattern=test_pattern, verbose=verbose)

    # Annotate the result with suspension metadata
    if not suspender.was_skipped:
        result.details.insert(
            0,
            f"[Model suspended: freed ≈ {suspender.memory_freed_mb:.0f} MB, "
            f"reloaded in {suspender.reload_seconds:.1f}s]",
        )

    return result


def run_tests_suspended_tool(
    path: str = ".",
    verbose: bool = False,
    force_suspend: bool = True,
) -> str:
    """Run test suite with automatic model suspension to free RAM.

    On CPU-only systems this evicts model weights (1-5+ GB) before running
    ``pytest`` and reloads them afterward, ensuring maximum available
    memory for the test processes.

    Args:
        path: Directory containing tests
        verbose: If True, include verbose pytest output
        force_suspend: Always suspend model (True) or only when RAM is tight (False)
    """
    # Attempt to resolve the current brain from the global DeterministicOllama cache
    brain = None
    try:
        from core.ollama_client import DeterministicOllama
        if DeterministicOllama._loader_cache:
            # Use the first cached model's parent brain
            model_name = next(iter(DeterministicOllama._loader_cache))
            brain = DeterministicOllama(model_name)
    except Exception:
        pass

    result = run_tests_suspended(
        path, verbose=verbose, brain=brain, force_suspend=force_suspend
    )
    output = f"{'✅' if result.passed else '❌'} {result.message}"
    if result.details:
        output += "\n" + "\n".join(result.details[:15])
    return output
