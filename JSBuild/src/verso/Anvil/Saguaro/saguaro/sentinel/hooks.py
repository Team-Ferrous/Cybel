# saguaro/sentinel/hooks.py
"""Utilities for hooks."""

import os
import stat


def install_pre_commit_hook(repo_path: str) -> bool:
    """Installs a git pre-commit hook that runs Saguaro verification."""
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.exists(git_dir):
        print(f"Error: {repo_path} is not a git repository.")
        return False

    hooks_dir = os.path.join(git_dir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)

    hook_path = os.path.join(hooks_dir, "pre-commit")

    hook_content = """#!/bin/bash
# SAGUARO Pre-commit Hook
# Verifies codebase compliance before allow commit.

# Resolve saguaro binary: PATH first, then repo-local venv fallback
if command -v saguaro &> /dev/null; then
  SAGUARO_BIN="saguaro"
elif [ -x "./venv/bin/saguaro" ]; then
  SAGUARO_BIN="./venv/bin/saguaro"
elif [ -x "./.venv/bin/saguaro" ]; then
  SAGUARO_BIN="./.venv/bin/saguaro"
else
  echo "⚠️  saguaro not found on PATH or in repo venv. Skipping verification."
  exit 0
fi

./scripts/aes_saguaro_preflight.sh --mode warn
$SAGUARO_BIN verify . --engines native,ruff,semantic,aes --format json
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  echo "❌ SAGUARO Verification Failed. Commit blocked."
  exit 1
fi

echo "✅ SAGUARO Verification Passed."
exit 0
"""

    with open(hook_path, "w") as f:
        f.write(hook_content)

    # Make executable
    st = os.stat(hook_path)
    os.chmod(hook_path, st.st_mode | stat.S_IEXEC)

    print(f"Saguaro pre-commit hook installed at {hook_path}")
    return True
