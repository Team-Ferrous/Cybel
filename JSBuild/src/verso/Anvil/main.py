import sys

from anvil import main as anvil_main


def main(argv: list[str] | None = None):
    print(
        "Deprecated launcher: `python main.py` now routes through `python anvil.py`.",
        file=sys.stderr,
    )
    return anvil_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
