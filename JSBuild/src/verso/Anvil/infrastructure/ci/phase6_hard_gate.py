"""Phase 6 AES hard-gate collection and enforcement helpers for CI."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT_DIR = Path(".anvil/artifacts/phase6")

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict = dict[str, JsonValue]


def extract_trailing_json(raw: str) -> JsonValue:
    """Return the final JSON payload from mixed stdout/stderr text."""
    decoder = json.JSONDecoder()
    starts = [index for index, ch in enumerate(raw) if ch in "{["]
    for start in reversed(starts):
        try:
            payload, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            continue
        if raw[end:].strip() == "":
            return payload
    raise ValueError("No trailing JSON payload found")


def load_json_artifact(path: Path) -> JsonValue:
    """Load and parse a possibly noisy JSON artifact file."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return extract_trailing_json(raw)


def _as_dict(payload: JsonValue, label: str) -> JsonDict:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} payload is not a JSON object")
    return payload


def _find_keyword_matches(
    violations: list[JsonDict], terms: list[str]
) -> list[JsonDict]:
    lowered_terms = [term.lower() for term in terms]
    matches: list[JsonDict] = []
    for violation in violations:
        rule_id = str(violation.get("rule_id", "")).lower()
        message = str(violation.get("message", "")).lower()
        if any(term in rule_id or term in message for term in lowered_terms):
            matches.append(violation)
    return matches


def evaluate_policy_errors(
    verify: JsonDict, deadcode: JsonDict, audit: JsonDict
) -> list[str]:
    """Evaluate policy violations from parsed verification artifacts."""
    raw_violations = verify.get("violations") or []
    violations = [item for item in raw_violations if isinstance(item, dict)]

    missing_trace = _find_keyword_matches(
        violations,
        ["aes-tr", "traceability", "trace id", "trace"],
    )
    missing_evidence = _find_keyword_matches(violations, ["evidence", "bundle"])
    invalid_waiver = _find_keyword_matches(
        violations,
        ["waiver", "expired", "invalid waiver"],
    )
    missing_review = _find_keyword_matches(
        violations,
        ["independent review", "review gate", "aes-rev"],
    )
    critical_red_team = _find_keyword_matches(
        violations,
        ["red-team", "fmea", "critical finding"],
    )

    errors: list[str] = []
    if missing_trace:
        errors.append(f"Missing traceability closure: {len(missing_trace)} findings")
    if missing_evidence:
        errors.append(
            f"Missing evidence-bundle closure: {len(missing_evidence)} findings"
        )
    if invalid_waiver:
        errors.append(f"Expired/invalid waivers: {len(invalid_waiver)} findings")
    if missing_review:
        errors.append(
            f"Missing independent-review proof: {len(missing_review)} findings"
        )
    if critical_red_team:
        errors.append(
            f"Unresolved critical red-team findings: {len(critical_red_team)} findings"
        )

    violation_count = int(verify.get("count", len(violations)))
    if verify.get("status") != "pass":
        errors.append(f"Verify failed ({violation_count} violations)")
    if int(deadcode.get("count", 0)) > 0:
        errors.append(f"Dead code candidates present ({deadcode.get('count', 0)})")
    if str(audit.get("status", "fail")).lower() != "pass":
        errors.append("Governance audit failed")
    return errors


def evaluate_execution_errors(status: JsonDict) -> list[str]:
    """Validate command execution status metadata from collection step."""
    def status_exit(command: str) -> int:
        details = status.get(command, {})
        if isinstance(details, dict):
            return int(details.get("exit_code", 99))
        return 99

    errors: list[str] = []
    verify_exit = status_exit("verify")
    aes_report_exit = status_exit("aes_report")
    deadcode_exit = status_exit("deadcode")
    audit_exit = status_exit("audit")

    if verify_exit not in {0, 1}:
        errors.append(f"Verify command execution failed (exit={verify_exit})")
    if aes_report_exit not in {0, 1}:
        errors.append(f"AES report command execution failed (exit={aes_report_exit})")
    if deadcode_exit not in {0, 1}:
        errors.append(f"Deadcode command execution failed (exit={deadcode_exit})")
    if audit_exit not in {0, 1}:
        errors.append(f"Audit command execution failed (exit={audit_exit})")

    impact_status = status.get("impact", {})
    attempted = 0
    failed = 0
    if isinstance(impact_status, dict):
        attempted = int(impact_status.get("attempted", 0))
        failed = int(impact_status.get("failed", 0))
    if attempted == 0:
        errors.append("Impact stage did not run on any file target")
    if failed > 0:
        errors.append(f"Impact stage had {failed} failed target(s)")
    return errors


@dataclass(frozen=True)
class CommandSpec:
    """Single command execution descriptor with output destinations."""

    name: str
    command: list[str]
    stdout_path: Path
    stderr_path: Path


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _resolve_executable(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise FileNotFoundError(f"Required executable not found in PATH: {name}")
    return resolved


def _run_and_capture(spec: CommandSpec) -> int:
    process = subprocess.run(  # noqa: S603
        spec.command,
        capture_output=True,
        text=True,
        check=False,
    )
    _write_text(spec.stdout_path, process.stdout)
    _write_text(spec.stderr_path, process.stderr)
    return process.returncode


def _changed_files_from_input(changed_files: str) -> list[str]:
    files = [entry for entry in shlex.split(changed_files) if entry]
    if files:
        return files
    git_bin = _resolve_executable("git")
    process = subprocess.run(  # noqa: S603
        [git_bin, "ls-files"],
        capture_output=True,
        text=True,
        check=False,
    )
    repo_files = [line.strip() for line in process.stdout.splitlines() if line.strip()]
    python_files = [path for path in repo_files if path.endswith(".py")]
    if python_files:
        return python_files
    return ["core/unified_chat_loop.py"]


def _base_verify_command(saguaro_bin: str) -> list[str]:
    return [
        saguaro_bin,
        "verify",
        ".",
        "--engines",
        "native,ruff,semantic,aes",
        "--aal",
        "0,1",
        "--require-trace",
        "--require-evidence",
        "--require-valid-waivers",
        "--format",
        "json",
    ]


def _collect_core_command_specs(saguaro_bin: str) -> list[CommandSpec]:
    verify_cmd = _base_verify_command(saguaro_bin)
    aes_report_cmd = verify_cmd[:-2] + ["--format", "json", "--aes-report"]
    deadcode_cmd = [saguaro_bin, "deadcode", "--format", "json"]
    audit_cmd = [saguaro_bin, "audit", "--format", "json"]
    return [
        CommandSpec(
            name="verify",
            command=verify_cmd,
            stdout_path=ARTIFACT_DIR / "verify.json",
            stderr_path=ARTIFACT_DIR / "verify.stderr",
        ),
        CommandSpec(
            name="aes_report",
            command=aes_report_cmd,
            stdout_path=ARTIFACT_DIR / "aes_report.json",
            stderr_path=ARTIFACT_DIR / "aes_report.stderr",
        ),
        CommandSpec(
            name="deadcode",
            command=deadcode_cmd,
            stdout_path=ARTIFACT_DIR / "deadcode.json",
            stderr_path=ARTIFACT_DIR / "deadcode.stderr",
        ),
        CommandSpec(
            name="audit",
            command=audit_cmd,
            stdout_path=ARTIFACT_DIR / "audit.json",
            stderr_path=ARTIFACT_DIR / "audit.stderr",
        ),
    ]


def collect_phase6_artifacts(changed_files: str) -> int:
    """Run Phase 6 command flow and save artifacts + execution metadata."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    impact_dir = ARTIFACT_DIR / "impact"
    impact_dir.mkdir(parents=True, exist_ok=True)

    saguaro_bin = _resolve_executable("saguaro")
    status: JsonDict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verify": {},
        "aes_report": {},
        "deadcode": {},
        "audit": {},
        "impact": {"attempted": 0, "succeeded": 0, "failed": 0, "targets": []},
    }

    for spec in _collect_core_command_specs(saguaro_bin):
        exit_code = _run_and_capture(spec)
        status[spec.name] = {
            "command": " ".join(spec.command),
            "exit_code": exit_code,
            "stdout": str(spec.stdout_path),
            "stderr": str(spec.stderr_path),
        }

    targets = _changed_files_from_input(changed_files)
    for target in targets:
        target_path = Path(target)
        if not target_path.exists() or not target_path.is_file():
            continue
        safe_name = target.replace("/", "__").replace(" ", "__")
        impact_spec = CommandSpec(
            name="impact",
            command=[saguaro_bin, "impact", "--path", target],
            stdout_path=impact_dir / f"{safe_name}.json",
            stderr_path=impact_dir / f"{safe_name}.stderr",
        )
        exit_code = _run_and_capture(impact_spec)
        impact = _as_dict(status["impact"], "impact")
        impact["attempted"] = int(impact.get("attempted", 0)) + 1
        if exit_code == 0:
            impact["succeeded"] = int(impact.get("succeeded", 0)) + 1
        else:
            impact["failed"] = int(impact.get("failed", 0)) + 1
        targets_list = impact.get("targets")
        if not isinstance(targets_list, list):
            targets_list = []
        targets_list.append(
            {
                "path": target,
                "exit_code": exit_code,
                "stdout": str(impact_spec.stdout_path),
                "stderr": str(impact_spec.stderr_path),
            }
        )
        impact["targets"] = targets_list

    (ARTIFACT_DIR / "command_status.json").write_text(
        json.dumps(status, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Phase 6 artifacts collected under {ARTIFACT_DIR}")
    return 0


def _load_status_errors(status_path: Path) -> tuple[list[str], JsonDict | None]:
    if not status_path.exists():
        return [f"Missing status artifact: {status_path}"], None
    status = _as_dict(
        json.loads(status_path.read_text(encoding="utf-8")),
        "command_status",
    )
    return evaluate_execution_errors(status), status


def _load_required_policy_inputs() -> tuple[
    list[str], JsonDict | None, JsonDict | None, JsonDict | None
]:
    errors: list[str] = []
    payloads: dict[str, JsonDict] = {}
    required = {
        "verify": ARTIFACT_DIR / "verify.json",
        "deadcode": ARTIFACT_DIR / "deadcode.json",
        "audit": ARTIFACT_DIR / "audit.json",
    }

    for label, path in required.items():
        if not path.exists():
            errors.append(f"Missing required artifact: {path}")
            continue
        try:
            parsed = load_json_artifact(path)
            payloads[label] = _as_dict(parsed, label)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            errors.append(f"Failed to parse {label} artifact: {exc}")

    return (
        errors,
        payloads.get("verify"),
        payloads.get("deadcode"),
        payloads.get("audit"),
    )


def enforce_phase6_policy() -> int:
    """Evaluate collected artifacts against Phase 6 hard-gate policy."""
    status_path = ARTIFACT_DIR / "command_status.json"
    errors, _status = _load_status_errors(status_path)

    policy_load_errors, verify, deadcode, audit = _load_required_policy_inputs()
    errors.extend(policy_load_errors)
    if verify is not None and deadcode is not None and audit is not None:
        errors.extend(evaluate_policy_errors(verify, deadcode, audit))

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "artifacts": {
            "verify": str(ARTIFACT_DIR / "verify.json"),
            "deadcode": str(ARTIFACT_DIR / "deadcode.json"),
            "audit": str(ARTIFACT_DIR / "audit.json"),
            "status": str(status_path),
        },
    }
    (ARTIFACT_DIR / "hard_gate_result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    if errors:
        print("AES hard gate failed:")
        for error in errors:
            print(f" - {error}")
        return 1

    print("AES hard gate passed.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6 AES hard-gate runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run command flow and collect artifacts",
    )
    collect_parser.add_argument(
        "--changed-files",
        default="",
        help="Space-separated changed file paths from CI context",
    )
    subparsers.add_parser(
        "enforce",
        help="Evaluate artifacts against hard-gate policy",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Phase 6 hard-gate collection and enforcement."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "collect":
        return collect_phase6_artifacts(changed_files=args.changed_files)
    if args.command == "enforce":
        return enforce_phase6_policy()
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
