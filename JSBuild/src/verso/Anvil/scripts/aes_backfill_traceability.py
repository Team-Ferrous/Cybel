#!/usr/bin/env python3
"""Backfill AES traceability records for legacy high-AAL paths."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_TRACEABILITY_JSONL = Path("standards/traceability/TRACEABILITY.jsonl")
DEFAULT_REPORT_PATH = Path(".anvil/artifacts/phase5/traceability_backfill_report.json")
DEFAULT_HIGH_AAL_GLOBS = ("core/agent.py", "core/unified_chat_loop.py")
DEFAULT_DESIGN_REF = "specs/phase5_codebase_remediation.md"
DEFAULT_VERIFICATION_REF = "ROADMAP_AES.md#phase-5-codebase-remediation"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _discover_targets(repo_root: Path, globs: list[str]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in globs:
        for candidate in repo_root.glob(pattern):
            if candidate.is_file():
                matches.add(candidate.resolve())
    return sorted(matches)


def _infer_test_refs(repo_root: Path, rel_path: str) -> list[str]:
    source = Path(rel_path)
    candidates = [
        Path("tests") / f"test_{source.stem}.py",
        Path("tests") / source.parent / f"test_{source.stem}.py",
        Path("tests") / f"{source.stem}_test.py",
    ]
    discovered: list[str] = []
    for candidate in candidates:
        resolved = repo_root / candidate
        if resolved.exists():
            discovered.append(candidate.as_posix())
    return discovered


def _build_trace_record(
    *,
    rel_path: str,
    owner: str,
    requirement_idx: int,
    design_ref: str,
    verification_ref: str,
    repo_root: Path,
) -> dict[str, Any]:
    digest = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:12]
    trace_id = f"trace::phase5::{digest}"
    test_refs = _infer_test_refs(repo_root, rel_path)
    if not test_refs:
        test_refs = ["tests/test_phase5_remediation_scripts.py"]

    now = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    return {
        "trace_id": trace_id,
        "requirement_id": f"AES-P5-LEGACY-{requirement_idx:03d}",
        "design_ref": design_ref,
        "code_refs": [rel_path],
        "test_refs": test_refs,
        "verification_refs": [
            verification_ref,
            "saguaro verify . --engines native,ruff,semantic,aes --format json",
        ],
        "aal": "AAL-1",
        "owner": owner,
        "timestamp": now,
    }


def run(
    repo_root: Path,
    *,
    traceability_jsonl: Path = DEFAULT_TRACEABILITY_JSONL,
    high_aal_globs: list[str] | None = None,
    owner: str | None = None,
    design_ref: str = DEFAULT_DESIGN_REF,
    verification_ref: str = DEFAULT_VERIFICATION_REF,
    report_path: Path = DEFAULT_REPORT_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    owner_value = owner or os.getenv("USER", "phase5-remediation")
    glob_patterns = list(high_aal_globs or DEFAULT_HIGH_AAL_GLOBS)

    absolute_traceability = (repo_root / traceability_jsonl).resolve()
    existing_records = _read_jsonl(absolute_traceability)

    indexed_files: set[str] = set()
    for record in existing_records:
        for code_ref in record.get("code_refs", []):
            if isinstance(code_ref, str) and code_ref.strip():
                indexed_files.add(code_ref)

    targets = _discover_targets(repo_root, glob_patterns)
    missing_targets: list[str] = []
    for target in targets:
        rel_path = target.relative_to(repo_root).as_posix()
        if rel_path not in indexed_files:
            missing_targets.append(rel_path)

    created_records: list[dict[str, Any]] = []
    for idx, rel_path in enumerate(missing_targets, start=1):
        created_records.append(
            _build_trace_record(
                rel_path=rel_path,
                owner=owner_value,
                requirement_idx=idx,
                design_ref=design_ref,
                verification_ref=verification_ref,
                repo_root=repo_root,
            )
        )

    if not dry_run:
        _append_jsonl(absolute_traceability, created_records)

    report = {
        "ok": True,
        "dry_run": dry_run,
        "traceability_jsonl": str(traceability_jsonl),
        "targets_scanned": [path.relative_to(repo_root).as_posix() for path in targets],
        "existing_records": len(existing_records),
        "missing_targets": missing_targets,
        "created_count": len(created_records),
        "created_trace_ids": [record["trace_id"] for record in created_records],
    }

    absolute_report = (repo_root / report_path).resolve()
    if not dry_run:
        absolute_report.parent.mkdir(parents=True, exist_ok=True)
        absolute_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill AES traceability records for legacy high-AAL files."
    )
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument(
        "--traceability-jsonl",
        default=str(DEFAULT_TRACEABILITY_JSONL),
        help="Path to traceability JSONL relative to repo root",
    )
    parser.add_argument(
        "--high-aal-glob",
        action="append",
        dest="high_aal_globs",
        help="Glob to include high-AAL files (repeatable)",
    )
    parser.add_argument("--owner", default=None, help="Owner for backfilled records")
    parser.add_argument(
        "--design-ref",
        default=DEFAULT_DESIGN_REF,
        help="Design reference for generated records",
    )
    parser.add_argument(
        "--verification-ref",
        default=DEFAULT_VERIFICATION_REF,
        help="Verification reference for generated records",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Path to write JSON report relative to repo root",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write changes")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")

    args = parser.parse_args()
    repo_root = Path(args.repo).resolve()

    result = run(
        repo_root,
        traceability_jsonl=Path(args.traceability_jsonl),
        high_aal_globs=args.high_aal_globs,
        owner=args.owner,
        design_ref=args.design_ref,
        verification_ref=args.verification_ref,
        report_path=Path(args.report),
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        missing_count = len(result["missing_targets"])
        print(
            "Traceability backfill: "
            f"created={result['created_count']} missing={missing_count}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
