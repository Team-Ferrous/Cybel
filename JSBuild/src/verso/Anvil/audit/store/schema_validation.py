from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def _schema_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas"


@lru_cache(maxsize=16)
def load_schema(name: str) -> dict[str, Any]:
    path = _schema_dir() / name
    if not path.exists():
        raise RuntimeError(f"Schema file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_payload(name: str, payload: dict[str, Any]) -> None:
    try:
        import jsonschema
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "jsonschema is required for audit schema-first validation."
        ) from exc
    jsonschema.validate(payload, load_schema(name))
