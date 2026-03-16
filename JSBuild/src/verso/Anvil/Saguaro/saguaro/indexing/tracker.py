"""Utilities for tracker."""

import hashlib
import json
import logging
import os
from typing import Any

from saguaro.errors import SaguaroStateCorruptionError
from saguaro.storage.atomic_fs import atomic_write_json

logger = logging.getLogger(__name__)


class IndexTracker:
    """Tracks file modification times and verification states to support
    incremental indexing and robust governance.
    """

    def __init__(self, tracking_file: str) -> None:
        """Initialize the instance."""
        self.tracking_file = tracking_file
        self.watermark_file = f"{tracking_file}.watermark.json"
        self.state: dict[str, dict[str, Any]] = {}  # path -> {mtime, hash, verified}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file) as f:
                    raw_state = json.load(f)

                    # Handle legacy format (flat dict of mtimes)
                    if raw_state and not isinstance(
                        next(iter(raw_state.values())), dict
                    ):
                        logger.info("Upgrading IndexTracker state to new format.")
                        self.state = {
                            path: {
                                "mtime": mtime,
                                "size": -1,
                                "hash": "",
                                "verified": False,
                            }
                            for path, mtime in raw_state.items()
                        }
                    else:
                        normalized: dict[str, dict[str, Any]] = {}
                        for path, entry in (raw_state or {}).items():
                            if not isinstance(entry, dict):
                                normalized[path] = {
                                    "mtime": float(entry or 0.0),
                                    "size": -1,
                                    "hash": "",
                                    "verified": False,
                                }
                                continue
                            try:
                                size = int(entry.get("size", -1))
                            except Exception:
                                size = -1
                            normalized[path] = {
                                "mtime": float(entry.get("mtime", 0.0) or 0.0),
                                "size": size,
                                "hash": str(entry.get("hash", "") or ""),
                                "verified": bool(entry.get("verified", False)),
                            }
                        self.state = normalized
            except Exception as e:
                raise SaguaroStateCorruptionError(
                    f"Failed to load tracking file {self.tracking_file}: {e}"
                ) from e
        else:
            self.state = {}

    def save(self) -> None:
        """Handle save."""
        self._validate_state(self.state)
        atomic_write_json(self.tracking_file, self.state, indent=2, sort_keys=True)

    def filter_needs_indexing(self, file_paths: list[str]) -> list[str]:
        """Returns list of files that have changed or are new."""
        needs_update: list[str] = []
        state_changed = False
        for path in file_paths:
            if not os.path.exists(path):
                continue

            current_mtime = os.path.getmtime(path)
            current_size = os.path.getsize(path)
            entry = self.state.get(path, {})
            if not entry:
                needs_update.append(path)
                continue

            last_mtime = float(entry.get("mtime", 0.0) or 0.0)
            last_size = int(entry.get("size", -1) or -1)

            # Size drift is a guaranteed content drift.
            if current_size != last_size:
                needs_update.append(path)
                continue

            # mtime drift can be metadata-only; validate via hash.
            if current_mtime != last_mtime:
                known_hash = str(entry.get("hash", "") or "")
                if not known_hash:
                    needs_update.append(path)
                    continue
                current_hash = self._compute_hash(path)
                if current_hash != known_hash:
                    needs_update.append(path)
                else:
                    entry["mtime"] = current_mtime
                    entry["size"] = current_size
                    self.state[path] = entry
                    state_changed = True

        if state_changed:
            self.save()

        return needs_update

    def prune_missing(
        self,
        existing_files: list[str] | None = None,
        *,
        persist: bool = True,
    ) -> list[str]:
        """Drop tracker entries for files that no longer exist."""
        if not self.state:
            return []

        existing: set[str] | None = (
            set(existing_files) if existing_files is not None else None
        )
        removed: list[str] = []
        for path in list(self.state.keys()):
            if existing is not None and path not in existing:
                removed.append(path)
                self.state.pop(path, None)
                continue
            if not os.path.exists(path):
                removed.append(path)
                self.state.pop(path, None)

        if removed and persist:
            self.save()
        return removed

    def _compute_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        if not os.path.exists(file_path):
            return ""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {file_path}: {e}")
            return ""

    def update(self, file_paths: list[str], *, compute_hash: bool = True) -> None:
        """Updates the indexing state for the given files."""
        for path in file_paths:
            if os.path.exists(path):
                mtime = os.path.getmtime(path)
                size = os.path.getsize(path)
                file_hash = (
                    self._compute_hash(path)
                    if compute_hash
                    else str(self.state.get(path, {}).get("hash", "") or "")
                )

                # If mtime or hash changed, it's no longer verified
                existing = self.state.get(path, {})
                was_verified = existing.get("verified", False)
                if compute_hash and existing.get("hash") != file_hash:
                    was_verified = False

                self.state[path] = {
                    "mtime": mtime,
                    "size": size,
                    "hash": file_hash,
                    "verified": was_verified,
                }
        self.save()

    def update_verification(self, file_path: str, verified: bool = True) -> None:
        """Marks a file as verified and stores its current hash."""
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            size = os.path.getsize(file_path)
            file_hash = self._compute_hash(file_path)
            self.state[file_path] = {
                "mtime": mtime,
                "size": size,
                "hash": file_hash,
                "verified": verified,
            }
            self.save()

    def is_verified(self, file_path: str) -> bool:
        """Check if a file is verified based on its current content hash."""
        if not os.path.exists(file_path):
            return False

        entry = self.state.get(file_path)
        if not entry or not entry.get("verified"):
            return False

        # Verify hash match to ensure content hasn't changed since verification
        current_hash = self._compute_hash(file_path)
        return entry.get("hash") == current_hash

    def clear(self) -> None:
        """Handle clear."""
        self.state = {}
        self.save()

    def load_watermark(self) -> dict[str, Any]:
        if not os.path.exists(self.watermark_file):
            return {}
        try:
            with open(self.watermark_file) as f:
                payload = json.load(f)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def save_watermark(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.watermark_file, payload or {}, indent=2, sort_keys=True)

    @staticmethod
    def _validate_state(payload: dict[str, dict[str, Any]]) -> None:
        if not isinstance(payload, dict):
            raise SaguaroStateCorruptionError("Tracking payload must be a dictionary.")
        for path, entry in payload.items():
            if not isinstance(path, str):
                raise SaguaroStateCorruptionError(
                    "Tracking file contains a non-string file path."
                )
            if not isinstance(entry, dict):
                raise SaguaroStateCorruptionError(
                    f"Tracking entry for {path} must be a dictionary."
                )
            required = {"mtime", "size", "hash", "verified"}
            if not required.issubset(entry):
                raise SaguaroStateCorruptionError(
                    f"Tracking entry for {path} is missing required keys."
                )
