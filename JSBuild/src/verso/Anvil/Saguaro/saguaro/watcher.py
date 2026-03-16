"""Utilities for watcher."""

import logging
import time

from saguaro.indexing.coordinator import IndexCoordinator
from saguaro.indexing.engine import IndexEngine
from saguaro.utils.file_utils import get_code_files

logger = logging.getLogger(__name__)


class Watcher:
    """Provide Watcher support."""

    def __init__(
        self,
        engine: IndexEngine | None,
        target_path: str,
        interval: int = 5,
        coordinator: IndexCoordinator | None = None,
    ) -> None:
        """Initialize the instance."""
        self.engine = engine
        self.target_path = target_path
        self.interval = interval
        self.running = False
        self.coordinator = coordinator

    def scan_files(self) -> list[str]:
        """Handle scan files."""
        return get_code_files(self.target_path)

    def start(self) -> None:
        """Handle start."""
        self.running = True
        logger.info(
            f"Starting SAGUARO Watcher on {self.target_path} (Interval: {self.interval}s)"
        )

        while self.running:
            try:
                if self.coordinator is not None:
                    changes = self.coordinator.discover_changes(path=self.target_path)
                    changed = changes.get("changed_files", [])
                    deleted = changes.get("deleted_files", [])
                    if changed or deleted:
                        logger.info(
                            "Detected changes (changed=%d deleted=%d). Syncing via coordinator.",
                            len(changed),
                            len(deleted),
                        )
                        self.coordinator.sync(
                            path=self.target_path,
                            changed_files=changed,
                            deleted_files=deleted,
                            reason="watcher",
                            prune_deleted=True,
                        )
                else:
                    # Legacy engine-based mode.
                    if self.engine is None:
                        logger.warning(
                            "Watcher has neither coordinator nor engine; sleeping."
                        )
                        time.sleep(self.interval)
                        continue

                    all_files = self.scan_files()
                    stale = self.engine.tracker.prune_missing(all_files)
                    stale_removed = 0
                    if stale and hasattr(self.engine.store, "remove_file"):
                        for stale_path in stale:
                            stale_removed += int(
                                self.engine.store.remove_file(stale_path)
                            )

                    needed = self.engine.tracker.filter_needs_indexing(all_files)

                    if needed:
                        logger.info(
                            f"Detected changes in {len(needed)} files. Indexing..."
                        )
                        f_count, _ = self.engine.index_batch(needed, force=True)
                        if stale_removed:
                            logger.info(
                                "Pruned %d stale vectors from %d removed files.",
                                stale_removed,
                                len(stale),
                            )
                        self.engine.commit()
                        logger.info(f"Update complete. {f_count} files processed.")
                    elif stale_removed:
                        logger.info(
                            "Pruned %d stale vectors from %d removed files.",
                            stale_removed,
                            len(stale),
                        )
                        self.engine.commit()

            except Exception as e:
                logger.error(f"Watcher error: {e}")
                # Resilience: Don't crash

            time.sleep(self.interval)

    def stop(self) -> None:
        """Handle stop."""
        self.running = False
