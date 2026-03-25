"""Allow `python -m saguaro` to execute the low-overhead CLI bootstrap."""

from __future__ import annotations

from saguaro.bootstrap import main


if __name__ == "__main__":
    raise SystemExit(main())
