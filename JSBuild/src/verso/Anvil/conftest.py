from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
AUTHORITY_ROOT = REPO_ROOT / "Saguaro"
RUNTIME_SAGUARO_ROOT = AUTHORITY_ROOT / "saguaro"


def _pin_runtime_saguaro() -> None:
    authority_root = str(AUTHORITY_ROOT)
    repo_root = str(REPO_ROOT)
    for path in (authority_root, repo_root):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, authority_root)
    sys.path.insert(1, repo_root)

    for name in list(sys.modules):
        if name == "saguaro" or name.startswith("saguaro."):
            del sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        "saguaro",
        RUNTIME_SAGUARO_ROOT / "__init__.py",
        submodule_search_locations=[str(RUNTIME_SAGUARO_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to pin runtime saguaro package for pytest.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["saguaro"] = module
    spec.loader.exec_module(module)


_pin_runtime_saguaro()
