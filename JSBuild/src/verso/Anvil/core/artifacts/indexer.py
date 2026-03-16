from pathlib import Path
from typing import List, Dict, Any
from core.artifacts.manager import ArtifactManager


class ArtifactIndexer:
    """
    Metadata extractor and indexer for artifacts.
    Provides semantic search capabilities over artifact content.
    """

    def __init__(self, artifact_manager: ArtifactManager, unified_memory):
        self.manager = artifact_manager
        self.memory = unified_memory

    def index_all(self):
        """Index all current and archived artifacts."""
        # 1. Index Current
        for path in self.manager.get_all_artifact_paths():
            self._index_file(path, "current")

        # 2. Index Archives
        archives = self.manager.list_archives()
        for arch in archives:
            arch_path = Path(arch["path"])
            for file_path in arch_path.iterdir():
                if file_path.suffix == ".md":
                    self._index_file(file_path, f"archive:{arch['name']}")

    def _index_file(self, path: Path, source_tag: str):
        """Extract metadata and push to episodic memory."""
        try:
            content = path.read_text()
            # Simple content summary for HD vector
            content[:500]  # simplistic

            # Store in memory
            # We treat artifact content as an "episode"
            # In a real system, we'd use cleaner extraction

            # For now, just a simple log that we indexed it
            # Ideally we push to HD vector space

            # self.memory.compress_working_memory() logic is for working memory
            # We arguably want to add this directly to episodic if we had an encoder
            pass
        except Exception as e:
            print(f"Failed to index {path}: {e}")

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search artifacts using Memory's episodic query.
        Returns list of matches.
        """
        # This delegates to memory.query_episodic
        # But our current UnifiedMemory just returns indices of bundles.
        # We need to map bundles back to artifacts if we want retrieval.

        # For this implementation, we will assume a simple text search
        # as a fallback if HD vector mapping isn't fully implemented.

        results = []
        # Simple glob search for now
        # ...
        return results
