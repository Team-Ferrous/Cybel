#!/usr/bin/env python3
"""Install Anvil CLI shims into a user bin directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.exists():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Anvil CLI shims")
    parser.add_argument(
        "--bin-dir",
        default="~/.local/bin",
        help="Directory where shim symlinks are created (default: ~/.local/bin)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing shim files if present",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "bin" / "saguaro"
    if not source.exists():
        raise FileNotFoundError(f"Missing launcher: {source}")

    bin_dir = Path(args.bin_dir).expanduser().resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)

    target = bin_dir / "saguaro"
    if target.exists() or target.is_symlink():
        if not args.force:
            raise FileExistsError(
                f"{target} already exists. Re-run with --force to replace it."
            )
        _safe_unlink(target)

    target.symlink_to(source)

    print(f"Installed: {target} -> {source}")
    if str(bin_dir) not in os.environ.get("PATH", ""):
        print(
            f"Note: {bin_dir} is not currently in PATH. "
            "Add it to your shell profile to invoke `saguaro` globally."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
