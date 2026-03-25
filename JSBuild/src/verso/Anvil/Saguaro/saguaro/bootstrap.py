"""Low-overhead Saguaro CLI bootstrap for hot command paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from saguaro import __version__


def _hot_command_requested(argv: list[str]) -> bool:
    if not argv:
        return False
    if argv[0] in {"--version", "-h", "--help"}:
        return False
    command = argv[0]
    if command in {"health", "doctor"}:
        return True
    if command == "query":
        return True
    if command == "abi":
        return len(argv) == 1 or argv[1] == "verify" or argv[1].startswith("--")
    if command == "ffi":
        return len(argv) == 1 or argv[1] == "audit" or argv[1].startswith("--")
    if command == "math":
        return len(argv) > 1 and argv[1] == "parse"
    if command == "cpu":
        return len(argv) > 1 and argv[1] == "scan"
    return False


def _hot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"SAGUARO v{__version__} - Quantum Codebase OS"
    )
    parser.add_argument(
        "--version", action="version", version=f"SAGUARO v{__version__}"
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Repository root to operate on. Defaults to current working directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Index Health Dashboard")
    subparsers.add_parser(
        "doctor", help="One-shot diagnostics for freshness and ABI/parser health"
    )

    abi_parser = subparsers.add_parser("abi", help="Roadmap ABI compatibility surface")
    abi_parser.add_argument(
        "abi_op",
        nargs="?",
        choices=["verify"],
        default="verify",
    )
    abi_parser.add_argument("--format", choices=["text", "json"], default="text")

    ffi_parser = subparsers.add_parser("ffi", help="Roadmap FFI compatibility surface")
    ffi_parser.add_argument(
        "ffi_op",
        nargs="?",
        choices=["audit"],
        default="audit",
    )
    ffi_parser.add_argument("--path", default=".")
    ffi_parser.add_argument("--limit", type=int, default=200)
    ffi_parser.add_argument("--format", choices=["text", "json"], default="text")

    query_parser = subparsers.add_parser("query", help="Query the index")
    query_parser.add_argument("text")
    query_parser.add_argument("--k", type=int, default=5)
    query_parser.add_argument("--file")
    query_parser.add_argument("--level", type=int, default=3, choices=[0, 1, 2, 3])
    query_parser.add_argument("--json", action="store_true")
    query_parser.add_argument(
        "--strategy",
        choices=[
            "lexical",
            "semantic",
            "hybrid",
            "graph",
            "symbol",
            "search-by-symbol",
            "concept",
            "search-by-concept",
            "impact",
            "search-by-impact",
            "drift",
            "search-by-drift",
            "test-failure",
            "search-by-test-failure",
            "policy",
            "search-by-policy",
            "roadmap",
            "search-by-roadmap",
        ],
        default="hybrid",
    )
    query_parser.add_argument("--explain", action="store_true")
    query_parser.add_argument(
        "--scope",
        choices=["local", "workspace", "peer", "global"],
        default="global",
    )
    query_parser.add_argument(
        "--dedupe-by",
        choices=["entity", "path", "symbol"],
        default="entity",
    )

    math_parser = subparsers.add_parser("math", help="Repo math extraction and mapping")
    math_sub = math_parser.add_subparsers(dest="math_op", required=True)
    math_parse = math_sub.add_parser("parse")
    math_parse.add_argument("--path", default=".")
    math_parse.add_argument("--format", choices=["json", "text"], default="json")

    cpu_parser = subparsers.add_parser("cpu", help="CPU advisory and hotspot analysis")
    cpu_sub = cpu_parser.add_subparsers(dest="cpu_op", required=True)
    cpu_scan = cpu_sub.add_parser("scan")
    cpu_scan.add_argument("--path", default=".")
    cpu_scan.add_argument("--arch", default="x86_64-avx2")
    cpu_scan.add_argument("--limit", type=int, default=20)
    cpu_scan.add_argument("--format", choices=["json", "text"], default="json")

    return parser


def _emit_text_query(result: dict[str, Any], *, explain: bool) -> None:
    print(f"Query: '{result.get('query', '')}'")
    for row in result.get("results", []):
        print(
            f"[{row.get('rank', '?')}] [{row.get('score', 0.0):.4f}] "
            f"{row.get('name', 'unknown')} ({row.get('type', 'symbol')})"
        )
        print(f"    Path: {row.get('file', '?')}:{row.get('line', '?')}")
        if row.get("reason"):
            print(f"    Why:  {row['reason']}")
        if explain and row.get("explanation"):
            print(f"    Explain: {json.dumps(row['explanation'], sort_keys=True)}")
            print("")


def _run_hot_command(args: argparse.Namespace) -> int:
    from saguaro.fastpath import FastCommandAPI

    repo_root = os.path.abspath(args.repo or os.getcwd())
    fast_api = FastCommandAPI(repo_root)

    if args.command == "health":
        print(json.dumps(fast_api.health(), indent=2))
        return 0
    if args.command == "doctor":
        print(json.dumps(fast_api.doctor(), indent=2))
        return 0
    if args.command == "abi":
        report = fast_api.abi(action=getattr(args, "abi_op", "verify"))
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, indent=2))
        else:
            native_abi = dict(report.get("native_abi") or {})
            status = "pass" if native_abi.get("ok") else "warning"
            print(f"ABI verify: {status}")
            if native_abi.get("reason"):
                print(str(native_abi.get("reason")))
        return 0
    if args.command == "ffi":
        report = fast_api.ffi_audit(
            path=getattr(args, "path", "."),
            limit=int(getattr(args, "limit", 200) or 200),
        )
        if getattr(args, "format", "text") == "json":
            print(json.dumps(report, indent=2))
        else:
            print(f"FFI audit boundaries: {report.get('count', 0)}")
        return 0
    if args.command == "math":
        result = fast_api.math_parse(path=getattr(args, "path", "."))
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "cpu":
        result = fast_api.cpu_scan(
            path=getattr(args, "path", "."),
            arch=getattr(args, "arch", "x86_64-avx2"),
            limit=int(getattr(args, "limit", 20) or 20),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "query":
        result = fast_api.query(
            text=args.text,
            k=args.k,
            file=getattr(args, "file", None),
            level=getattr(args, "level", 3),
            strategy=getattr(args, "strategy", "hybrid"),
            explain=bool(getattr(args, "explain", False)),
            scope=getattr(args, "scope", "global"),
            dedupe_by=getattr(args, "dedupe_by", "entity"),
        )
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            _emit_text_query(result, explain=bool(getattr(args, "explain", False)))
        return 0
    return 1


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if _hot_command_requested(args_list):
        parser = _hot_parser()
        return _run_hot_command(parser.parse_args(args_list))

    from saguaro.cli import main as cli_main

    cli_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
