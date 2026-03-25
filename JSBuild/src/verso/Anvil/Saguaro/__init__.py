"""In-repo compatibility package for the authoritative Saguaro runtime."""

from __future__ import annotations

from pathlib import Path

_AUTHORITATIVE_PACKAGE = (
    Path(__file__).resolve().parent.parent / "Saguaro" / "saguaro"
)
_AUTHORITATIVE_INIT = _AUTHORITATIVE_PACKAGE / "__init__.py"

if not _AUTHORITATIVE_INIT.exists():
    raise ModuleNotFoundError(
        "Authoritative Saguaro package is missing at "
        f"{_AUTHORITATIVE_INIT}"
    )

__path__ = [str(_AUTHORITATIVE_PACKAGE)]
__file__ = str(_AUTHORITATIVE_INIT)

exec(compile(_AUTHORITATIVE_INIT.read_text(encoding="utf-8"), __file__, "exec"))
