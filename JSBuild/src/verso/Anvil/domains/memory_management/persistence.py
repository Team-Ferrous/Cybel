import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from core.memory.fabric import MemoryFabricSnapshotter, MemoryFabricStore
from domains.memory_management.memory_types import MemoryTier


class MemoryPersistence:
    """
    Handles save/load of UnifiedMemory to disk.

    Structure:
    .anvil/memory/
      ├── working.json
      ├── semantic.json
      ├── episodic_index.npy  (HD Vectors)
      └── episodic_meta.json  (Descriptions/Timestamps)
    """

    def __init__(self, base_dir: str = ".anvil/memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        memory_storage: Dict[MemoryTier, Dict],
        episodic_timeline: List[np.ndarray],
    ):
        """Save memory state to disk."""

        # Save Tiers (Working, Semantic, Preference, etc)
        for tier, data in memory_storage.items():
            if tier == MemoryTier.EPISODIC:
                continue  # Handled separately

            # Convert non-serializable if needed
            safe_data = self._make_serializable(data)

            path = self.base_dir / f"{tier.value.lower()}.json"
            path.write_text(json.dumps(safe_data, indent=2))

        # Save Episodic (HD Vectors)
        if episodic_timeline:
            # Stack into one numpy array (N, D)
            matrix = np.vstack(episodic_timeline)
            np.save(self.base_dir / "episodic_index.npy", matrix)

    def load(self, memory_storage: Dict[MemoryTier, Dict]) -> List[np.ndarray]:
        """
        Load memory state from disk.
        Updates memory_storage in-place.
        Returns episodic_timeline list.
        """

        # Load Tiers
        for tier in memory_storage.keys():
            if tier == MemoryTier.EPISODIC:
                continue

            path = self.base_dir / f"{tier.value.lower()}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    memory_storage[tier].update(data)
                except Exception as e:
                    print(f"Error loading {tier}: {e}")

        # Load Episodic
        timeline = []
        ep_path = self.base_dir / "episodic_index.npy"
        if ep_path.exists():
            try:
                matrix = np.load(ep_path)
                # Split back into list of arrays
                timeline = [row for row in matrix]
            except Exception as e:
                print(f"Error loading episodic index: {e}")

        return timeline

    def _make_serializable(self, data: Any) -> Any:
        # Simple recursive sanitizer
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(i) for i in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)

    def save_almf_snapshot(
        self,
        *,
        campaign_id: str,
        db_path: str,
        storage_root: str | None = None,
        snapshot_id: str | None = None,
    ) -> Dict[str, Any]:
        store = MemoryFabricStore.from_db_path(db_path, storage_root=storage_root)
        snapshotter = MemoryFabricSnapshotter(store)
        return snapshotter.snapshot_campaign(
            campaign_id,
            snapshot_id=snapshot_id,
        )

    def load_almf_snapshot(
        self,
        *,
        snapshot_dir: str,
        db_path: str,
        storage_root: str | None = None,
        target_campaign_id: str | None = None,
    ) -> Dict[str, Any]:
        store = MemoryFabricStore.from_db_path(db_path, storage_root=storage_root)
        snapshotter = MemoryFabricSnapshotter(store)
        return snapshotter.restore_campaign(
            snapshot_dir,
            target_campaign_id=target_campaign_id,
        )
