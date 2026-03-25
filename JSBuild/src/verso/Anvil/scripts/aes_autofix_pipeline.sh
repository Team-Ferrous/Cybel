#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ -d venv ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "[aes-autofix] repo: $REPO_ROOT"
echo "[aes-autofix] dry_run: $DRY_RUN"

mkdir -p .anvil/artifacts/phase5
PRE_CHANGE_FILE=".anvil/artifacts/phase5/pre_autofix_changed.txt"
POST_CHANGE_FILE=".anvil/artifacts/phase5/post_autofix_changed.txt"
NEW_CHANGE_FILE=".anvil/artifacts/phase5/new_autofix_changed.txt"
git diff --name-only | sort -u > "$PRE_CHANGE_FILE"

# Guardrail: never touch known high-AAL monolith files in the low-risk autofix pass.
HIGH_AAL_PATTERNS='^(core/agent\.py|core/unified_chat_loop\.py)$'

# Target only low-risk locations for formatting/lint autofix.
SAFE_PATHS=(
  scripts
  tests
  prompts
  standards
)

if (( DRY_RUN == 1 )); then
  ruff check "${SAFE_PATHS[@]}" --select I,UP,SIM,RET --fix --diff || true
  ruff format "${SAFE_PATHS[@]}" --check || true
else
  ruff check "${SAFE_PATHS[@]}" --select I,UP,SIM,RET --fix
  ruff format "${SAFE_PATHS[@]}"
fi

# Obvious dead artifact class: backup files.
BACKUP_LIST=".anvil/artifacts/phase5/backup_artifacts.txt"
find . -type f -name "*.py.bak" | sort > "$BACKUP_LIST"

if (( DRY_RUN == 0 )) && [[ -s "$BACKUP_LIST" ]]; then
  xargs -r rm -f < "$BACKUP_LIST"
fi

# Prompt contract lint auto-fix validation gate.
if (( DRY_RUN == 1 )); then
  python scripts/validate_prompt_contracts.py --json || true
else
  python scripts/validate_prompt_contracts.py --json
fi

# Guardrail assertion: no high-AAL files should be changed by this pipeline.
git diff --name-only | sort -u > "$POST_CHANGE_FILE"
comm -13 "$PRE_CHANGE_FILE" "$POST_CHANGE_FILE" > "$NEW_CHANGE_FILE"
if [[ -s "$NEW_CHANGE_FILE" ]] && grep -Eq "$HIGH_AAL_PATTERNS" "$NEW_CHANGE_FILE"; then
  echo "[aes-autofix] ERROR: high-AAL file touched by low-risk autofix pipeline."
  exit 2
fi

echo "[aes-autofix] complete"
