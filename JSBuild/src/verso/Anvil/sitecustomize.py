"""Prefer the authoritative `Saguaro/saguaro` tree for in-repo execution."""

from __future__ import annotations

import os
import sys


def _prefer_authoritative_saguaro() -> None:
    repo_root = os.path.dirname(__file__)
    authority_root = os.path.join(repo_root, "Saguaro")
    package_root = os.path.join(authority_root, "saguaro")
    if not os.path.isdir(package_root):
        return
    if authority_root in sys.path:
        sys.path.remove(authority_root)
    sys.path.insert(0, authority_root)


_prefer_authoritative_saguaro()
