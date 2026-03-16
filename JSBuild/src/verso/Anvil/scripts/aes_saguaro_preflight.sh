#!/usr/bin/env bash
set -euo pipefail

MODE="block"
if [[ "${1:-}" == "--mode" ]]; then
  MODE="${2:-block}"
fi

warn_count=0

report_issue() {
  local message="$1"
  if [[ "$MODE" == "warn" ]]; then
    echo "WARN: ${message}"
    warn_count=$((warn_count + 1))
  else
    echo "ERROR: ${message}" >&2
    exit 1
  fi
}

if [[ "$MODE" != "warn" && "$MODE" != "block" ]]; then
  echo "ERROR: --mode must be 'warn' or 'block'." >&2
  exit 2
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  report_issue "No active virtual environment. Activate ./venv before strict verification."
fi

python_bin=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  python_bin="${VIRTUAL_ENV}/bin/python"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  report_issue "Python executable not found."
fi

if [[ -z "$python_bin" ]]; then
  if [[ "$MODE" == "warn" ]]; then
    echo "Preflight completed with ${warn_count} warning(s)."
    exit 0
  fi
  exit 1
fi

if ! "$python_bin" - <<'PY'
import importlib
import os
import sys

errors = []

venv = os.environ.get("VIRTUAL_ENV")
if venv and not sys.executable.startswith(venv):
    errors.append(
        f"python executable {sys.executable} is outside active venv {venv}"
    )

try:
    import tensorflow  # noqa: F401
except Exception as exc:
    errors.append(f"tensorflow unavailable: {exc}")

for module in (
    "saguaro.sentinel.engines.native",
    "saguaro.sentinel.engines.external",
    "saguaro.sentinel.engines.semantic",
    "saguaro.sentinel.engines.aes",
):
    try:
        importlib.import_module(module)
    except Exception as exc:
        errors.append(f"{module} import failed: {exc}")

if errors:
    print("\n".join(errors))
    raise SystemExit(1)

print("AES strict preflight checks passed.")
PY
then
  report_issue "Sentinel strict-mode prerequisites are unavailable."
fi

if [[ "$MODE" == "warn" && "$warn_count" -gt 0 ]]; then
  echo "Preflight completed with ${warn_count} warning(s)."
  exit 0
fi

echo "Preflight complete: strict-mode prerequisites satisfied."
