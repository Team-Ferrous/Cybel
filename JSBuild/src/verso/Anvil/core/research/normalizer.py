"""Normalization helpers for research documents and evidence."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict


class ResearchNormalizer:
    """Converts raw fetched content into deterministic normalized records."""

    def normalize(self, title: str, content: str, origin_url: str, topic: str = "general") -> Dict[str, Any]:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return {
            "title": title,
            "content": content,
            "origin_url": origin_url,
            "topic": topic,
            "digest": digest,
            "fetched_at": time.time(),
        }
