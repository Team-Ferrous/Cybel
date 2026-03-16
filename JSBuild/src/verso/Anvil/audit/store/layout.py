from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    run_id: str
    root: Path
    manifest_json: Path
    attempts_ndjson: Path
    phases_ndjson: Path
    summary_json: Path
    failures_ndjson: Path
    checkpoint_json: Path


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_layout(base_dir: Path, run_id: str | None = None) -> RunLayout:
    resolved_id = str(run_id or default_run_id())
    root = base_dir / resolved_id
    return RunLayout(
        run_id=resolved_id,
        root=root,
        manifest_json=root / "manifest.json",
        attempts_ndjson=root / "attempts.ndjson",
        phases_ndjson=root / "phases.ndjson",
        summary_json=root / "summary.json",
        failures_ndjson=root / "failures.ndjson",
        checkpoint_json=root / "checkpoint.json",
    )


def ensure_layout(layout: RunLayout) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
