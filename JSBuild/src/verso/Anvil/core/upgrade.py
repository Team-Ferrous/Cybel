import subprocess
import os


class UpgradeManager:
    """
    Handles self-upgrading of the agent via Git.
    """

    def __init__(self, root_dir="."):
        self.root_dir = os.path.abspath(root_dir)

    def check_updates(self):
        """Checks for git updates."""
        try:
            # Fetch origin
            subprocess.run(
                ["git", "fetch"],
                cwd=self.root_dir,
                check=True,
                timeout=30,
                capture_output=True,
            )
            # Check status
            result = subprocess.run(
                ["git", "status", "-uno"],
                cwd=self.root_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            if "behind" in result.stdout:
                return True, "Update available."
            return False, "Already up to date."
        except Exception as e:
            return False, f"Error checking updates: {e}"

    def update(self):
        """
        Performs the update: git pull + pip install.
        """
        try:
            # 1. Pull
            res = subprocess.run(
                ["git", "pull"],
                cwd=self.root_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            # 2. Update dependencies
            subprocess.run(
                ["./venv/bin/pip", "install", "-r", "requirements.txt"],
                cwd=self.root_dir,
                check=True,
                capture_output=True,
            )

            return f"Update successful:\n{res.stdout}"
        except subprocess.CalledProcessError as e:
            return f"Update failed: {e.stderr}"
        except Exception as e:
            return f"Update error: {str(e)}"
