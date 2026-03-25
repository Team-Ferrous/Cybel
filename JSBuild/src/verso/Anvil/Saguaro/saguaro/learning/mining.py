"""Utilities for mining."""

class HardNegativeMiner:
    """Identifies and stores 'Hard Negatives': results that had high similarity
    scores but were rejected/ignored by the user.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.negatives_db = []

    def log_negatives(
        self, query: str, returned_ids: list[str], accepted_ids: list[str]
    ) -> None:
        """Finds and stores items that looked relevant but were ignored."""
        ignored = set(returned_ids) - set(accepted_ids)
        for i in ignored:
            self.negatives_db.append({"query": query, "negative_id": i})

    def is_negative(self, query: str, doc_id: str) -> bool:
        """Checks if a document is a known hard negative for a query."""
        # Simplistic check
        for entry in self.negatives_db:
            if entry["query"] == query and entry["negative_id"] == doc_id:
                return True
        return False
