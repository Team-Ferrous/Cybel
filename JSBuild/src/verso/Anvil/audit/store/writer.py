from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_ndjson(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True))
        handle.write("\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_ndjson_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    content = "".join(
        f"{json.dumps(row, sort_keys=True, ensure_ascii=True)}\n" for row in rows
    )
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def read_ndjson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Crash-safe behavior: salvage whatever valid UTF-8 prefix exists.
        text = path.read_bytes().decode("utf-8", errors="ignore")
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            # Crash-safe behavior: tolerate a torn final line from interrupted writes.
            if index == len(lines):
                break
            raise
    return rows
