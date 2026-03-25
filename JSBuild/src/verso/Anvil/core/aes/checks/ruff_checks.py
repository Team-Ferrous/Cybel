from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def _run_ruff_command(path: str, select: tuple[str, ...], timeout_seconds: int = 15) -> list[dict[str, Any]]:
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).resolve().parent)
    if venv_bin not in env.get("PATH", ""):
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

    command = [sys.executable, "-m", "ruff", "check", "--output-format=json"]
    for code in select:
        command.extend(["--select", code])
    command.append(path)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except Exception:
        return []

    if not result.stdout:
        return []

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []

    diagnostics: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).upper()
        if not any(code.startswith(prefix) for prefix in select):
            continue

        location = item.get("location") or {}
        diagnostics.append(
            {
                "code": code,
                "filename": str(item.get("filename", path)),
                "line": int(location.get("row", 1) or 1),
                "message": str(item.get("message", "Ruff rule violation.")),
            }
        )

    diagnostics.sort(
        key=lambda item: (
            str(item.get("filename", "")),
            int(item.get("line", 1)),
            str(item.get("code", "")),
        )
    )
    return diagnostics


def check_ruff_import_order(source: str, filepath: str) -> list[dict[str, Any]]:
    del source

    rel_path = Path(filepath).as_posix()
    if not rel_path.endswith(".py"):
        return []

    diagnostics = _run_ruff_command(str(Path(filepath).resolve()), ("I",))
    return [
        _violation(
            "AES-PY-2",
            rel_path,
            item.get("line", 1),
            f"Ruff {item.get('code', 'I')}: {item.get('message', 'import ordering violation')}",
        )
        for item in diagnostics
    ]
