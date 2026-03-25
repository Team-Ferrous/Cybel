#!/usr/bin/env python3
import sys


def _load_repl_main():
    from cli.repl import main as repl_main

    return repl_main


def main(argv: list[str] | None = None):
    print(
        "Deprecated launcher: `python anvil.py` now routes through `anvil` / `cli.repl:main`.",
        file=sys.stderr,
    )
    return _load_repl_main()(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
