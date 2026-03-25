"""Browser-capable research runtime facade."""

from __future__ import annotations

import hashlib
import time
from typing import Any


class BrowserResearchRuntime:
    """Replayable browser research facade with deterministic fetch records."""

    def fetch(self, url: str) -> dict[str, Any]:
        return {
            "url": url,
            "status": "fetched",
            "fetched_at": time.time(),
            "mode": "headless_stub",
            "content_digest": hashlib.sha256(url.encode("utf-8")).hexdigest(),
            "replay_token": f"browser:{hashlib.sha1(url.encode('utf-8')).hexdigest()[:12]}",
        }

    def evaluate(self, urls: list[str] | Any) -> dict[str, Any]:
        url_list = list(urls)
        return {
            "visited_urls": url_list,
            "count": len(url_list),
            "replayable": True,
            "unique_domains": len({item.split('/')[2] for item in url_list if '://' in item}),
        }
