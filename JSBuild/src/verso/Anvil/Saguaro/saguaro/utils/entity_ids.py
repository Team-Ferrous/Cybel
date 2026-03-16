"""Utilities for entity ids."""

from __future__ import annotations

import os
from typing import Any


def entity_identity(
    repo_path: str,
    file_path: str,
    name: str,
    entity_type: str,
    start_line: int,
) -> dict[str, Any]:
    """Build stable entity metadata shared across graph and vector stores."""
    rel_file = os.path.relpath(os.path.abspath(file_path), os.path.abspath(repo_path))
    display_name = name
    qualified_name = None
    if entity_type == "method" and "." in name:
        qualified_name = name
        display_name = name.split(".")[-1]
    entity_key = qualified_name or display_name
    return {
        "rel_file": rel_file,
        "display_name": display_name,
        "qualified_name": qualified_name,
        "entity_id": f"{rel_file}:{entity_key}:{int(start_line)}",
    }
