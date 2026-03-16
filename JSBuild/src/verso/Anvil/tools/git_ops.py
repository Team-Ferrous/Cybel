from typing import List
import subprocess


class GitOperationsManager:
    """Comprehensive git workflow automation."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path

    def create_feature_branch(self, feature_name: str) -> str:
        """Create and checkout feature branch."""
        branch_name = f"feature/{feature_name}"
        self._run_git(["checkout", "-b", branch_name])
        return branch_name

    def smart_commit(self, files: List[str], message: str) -> str:
        """Commit specific files with message."""
        for f in files:
            self._run_git(["add", f])

        self._run_git(["commit", "-m", message])
        # Return hash
        return self._run_git(["rev-parse", "HEAD"]).strip()

    def create_pull_request(self, title: str, body: str, base: str = "main") -> str:
        """Create pull request via gh CLI."""
        # Check if gh installed
        if not self._check_gh_installed():
            return "Error: GitHub CLI (gh) not installed."

        # Push
        try:
            curr = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
            self._run_git(["push", "--set-upstream", "origin", curr])
        except Exception as e:
            return f"Error pushing branch: {e}"

        # PR
        try:
            res = subprocess.run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--title",
                    title,
                    "--body",
                    body,
                    "--base",
                    base,
                ],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            if res.returncode != 0:
                return f"Error creating PR: {res.stderr}"
            return res.stdout.strip()
        except Exception as e:
            return f"Error running gh: {e}"

    def _run_git(self, args: List[str]) -> str:
        res = subprocess.run(
            ["git"] + args,
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout

    def _check_gh_installed(self) -> bool:
        from shutil import which

        return which("gh") is not None
