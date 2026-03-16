#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.aes.supply_chain import generate_sbom


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate AES SBOM artifact")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--format",
        default="cyclonedx",
        choices=("cyclonedx", "spdx"),
        help="SBOM format",
    )
    args = parser.parse_args()

    payload = generate_sbom(output_path=args.output, fmt=args.format)
    summary = {
        "format": args.format,
        "output": str(Path(args.output).resolve()),
        "component_count": len(payload.get("components", payload.get("packages", []))),
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
