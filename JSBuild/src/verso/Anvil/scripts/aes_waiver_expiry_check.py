#!/usr/bin/env python3
"""Scan AES waiver artifacts for expiry and emit remediation findings."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

DEFAULT_WAIVER_PATHS = (
    "standards/waivers.jsonl",
    "standards/waivers.json",
    "standards/traceability/waivers.jsonl",
)
DEFAULT_REPORT_PATH = Path(".anvil/artifacts/phase5/waiver_expiry_report.json")
DEFAULT_TICKET_LOG = Path("standards/remediation/expired_waiver_tickets.md")

REQUIRED_KEYS = {
    "waiver_id",
    "rule_id",
    "change_scope",
    "compensating_control",
    "risk_owner",
    "expiry",
    "remediation_ticket",
}


def _parse_expiry(value: str) -> dt.date | None:
    text = str(value).strip()
    if not text:
        return None

    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")

    for candidate in candidates:
        try:
            return dt.date.fromisoformat(candidate)
        except ValueError:
            pass
        try:
            return dt.datetime.fromisoformat(candidate).date()
        except ValueError:
            pass
    return None


def _load_waivers(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    if path.suffix == ".jsonl":
        waivers: list[dict[str, Any]] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                waivers.append(payload)
        return waivers

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("waivers"), list):
        return [item for item in payload["waivers"] if isinstance(item, dict)]
    return []


def _write_ticket_log(
    path: Path, expired: list[dict[str, Any]], as_of: dt.date
) -> None:
    lines = [
        "# Expired Waiver Remediation Tickets",
        "",
        f"Generated: {as_of.isoformat()}",
        "",
    ]
    if not expired:
        lines.append("No expired waivers detected.")
    else:
        lines.append("| Waiver | Rule | Expiry | Ticket | Days Overdue |")
        lines.append("|---|---|---|---|---:|")
        for item in expired:
            lines.append(
                "| "
                + f"{item['waiver_id']} | {item['rule_id']} | {item['expiry']} | "
                + f"{item['remediation_ticket']} | {item['days_overdue']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    repo_root: Path,
    *,
    waiver_paths: list[str] | None = None,
    due_soon_days: int = 14,
    today: dt.date | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    ticket_log: Path | None = DEFAULT_TICKET_LOG,
    write_ticket_log: bool = True,
) -> dict[str, Any]:
    as_of = today or dt.date.today()
    configured_paths = list(waiver_paths or DEFAULT_WAIVER_PATHS)

    all_waivers: list[dict[str, Any]] = []
    discovered_paths: list[str] = []
    for relative_path in configured_paths:
        absolute = (repo_root / relative_path).resolve()
        if not absolute.exists():
            continue
        discovered_paths.append(relative_path)
        all_waivers.extend(_load_waivers(absolute))

    expired: list[dict[str, Any]] = []
    due_soon: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for waiver in all_waivers:
        missing = sorted(REQUIRED_KEYS.difference(waiver.keys()))
        if missing:
            invalid.append(
                {
                    "waiver": waiver,
                    "error": f"missing required keys: {', '.join(missing)}",
                }
            )
            continue

        expiry_date = _parse_expiry(str(waiver.get("expiry", "")))
        if expiry_date is None:
            invalid.append(
                {
                    "waiver": waiver,
                    "error": "invalid expiry date format",
                }
            )
            continue

        days_remaining = (expiry_date - as_of).days
        normalized = {
            "waiver_id": str(waiver["waiver_id"]),
            "rule_id": str(waiver["rule_id"]),
            "expiry": expiry_date.isoformat(),
            "remediation_ticket": str(waiver["remediation_ticket"]),
            "risk_owner": str(waiver["risk_owner"]),
            "days_remaining": days_remaining,
        }

        if days_remaining < 0:
            normalized["days_overdue"] = abs(days_remaining)
            expired.append(normalized)
        elif days_remaining <= due_soon_days:
            due_soon.append(normalized)
        else:
            active.append(normalized)

    expired.sort(key=lambda item: item["days_overdue"], reverse=True)
    due_soon.sort(key=lambda item: item["days_remaining"])

    report = {
        "ok": len(expired) == 0 and len(invalid) == 0,
        "as_of": as_of.isoformat(),
        "waiver_paths_scanned": configured_paths,
        "waiver_paths_found": discovered_paths,
        "total_waivers": len(all_waivers),
        "expired_count": len(expired),
        "due_soon_count": len(due_soon),
        "active_count": len(active),
        "invalid_count": len(invalid),
        "expired": expired,
        "due_soon": due_soon,
        "invalid": invalid,
        "ticket_recommendations": [
            {
                "waiver_id": item["waiver_id"],
                "ticket": item["remediation_ticket"],
                "priority": "P0",
            }
            for item in expired
        ],
    }

    absolute_report = (repo_root / report_path).resolve()
    absolute_report.parent.mkdir(parents=True, exist_ok=True)
    absolute_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if write_ticket_log and ticket_log is not None:
        _write_ticket_log((repo_root / ticket_log).resolve(), expired, as_of)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan AES waiver artifacts for expiry."
    )
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument(
        "--waiver-path",
        action="append",
        dest="waiver_paths",
        help="Relative waiver artifact path (repeatable)",
    )
    parser.add_argument(
        "--due-soon-days",
        type=int,
        default=14,
        help="Number of days to classify waivers as due soon",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Report output path relative to repo root",
    )
    parser.add_argument(
        "--ticket-log",
        default=str(DEFAULT_TICKET_LOG),
        help="Ticket markdown output path relative to repo root",
    )
    parser.add_argument(
        "--no-ticket-log",
        action="store_true",
        help="Do not write ticket markdown output",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")

    args = parser.parse_args()
    repo_root = Path(args.repo).resolve()

    result = run(
        repo_root,
        waiver_paths=args.waiver_paths,
        due_soon_days=args.due_soon_days,
        report_path=Path(args.report),
        ticket_log=Path(args.ticket_log),
        write_ticket_log=not args.no_ticket_log,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            "Waiver expiry scan: "
            f"expired={result['expired_count']} due_soon={result['due_soon_count']} "
            f"invalid={result['invalid_count']}"
        )

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
