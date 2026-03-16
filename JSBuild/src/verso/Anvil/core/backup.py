import shutil
import os
from glob import glob
from datetime import datetime


class BackupManager:
    """
    Creates backups of files before modification.
    """

    def __init__(self, root_dir="."):
        self.root_dir = os.path.abspath(root_dir)
        self.backup_dir = os.path.join(self.root_dir, ".anvil_backups")
        os.makedirs(self.backup_dir, exist_ok=True)

    def backup(self, file_path):
        """
        Backs up a single file. Returns path to backup.
        """
        try:
            full_path = os.path.abspath(file_path)
            if not os.path.exists(full_path):
                return None

            rel_path = os.path.relpath(full_path, self.root_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Preserve directory structure in backup
            backup_path = os.path.join(
                self.backup_dir,
                os.path.dirname(rel_path),
                f"{os.path.basename(rel_path)}.{timestamp}.bak",
            )

            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            shutil.copy2(full_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Backup failed for {file_path}: {e}")
            return None

    def list_backups(self, file_path, max_items: int = 20):
        """List backups for a file path, newest first."""
        try:
            full_path = os.path.abspath(file_path)
            rel_path = os.path.relpath(full_path, self.root_dir)
            backup_dir = os.path.join(self.backup_dir, os.path.dirname(rel_path))
            base = os.path.basename(rel_path)
            pattern = os.path.join(backup_dir, f"{base}.*.bak")
            matches = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
            return matches[: max(1, int(max_items))]
        except Exception:
            return []

    def restore(self, file_path, backup_path: str = None):
        """Restore a file from a specific backup or latest backup."""
        full_path = os.path.abspath(file_path)
        selected = backup_path
        if not selected:
            candidates = self.list_backups(file_path, max_items=1)
            if not candidates:
                return None
            selected = candidates[0]

        selected_abs = os.path.abspath(selected)
        if not os.path.exists(selected_abs):
            return None
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        shutil.copy2(selected_abs, full_path)
        return selected_abs
