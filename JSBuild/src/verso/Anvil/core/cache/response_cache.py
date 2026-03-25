import hashlib
import json
import os
from typing import Optional
from datetime import datetime, timedelta


class ResponseCache:
    def __init__(self, cache_dir: str = ".anvil/cache", ttl_seconds: int = 3600):
        self._cache_dir = cache_dir
        self._ttl = timedelta(seconds=ttl_seconds)
        os.makedirs(cache_dir, exist_ok=True)

    def _hash_key(self, query: str, context_hash: str) -> str:
        combined = f"{query}:{context_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get_cached_response(self, query: str, context_hash: str) -> Optional[str]:
        """Return cached response if query + context unchanged and not expired."""
        key = self._hash_key(query, context_hash)
        cache_path = os.path.join(self._cache_dir, f"{key}.json")

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            cached_at = datetime.fromisoformat(data["timestamp"])
            if datetime.now() - cached_at > self._ttl:
                os.remove(cache_path)
                return None

            return data["response"]
        except Exception:
            return None

    def cache_response(self, query: str, context_hash: str, response: str):
        """Store response in cache."""
        key = self._hash_key(query, context_hash)
        cache_path = os.path.join(self._cache_dir, f"{key}.json")

        try:
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "query": query,
                        "context_hash": context_hash,
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                )
        except Exception:
            pass
