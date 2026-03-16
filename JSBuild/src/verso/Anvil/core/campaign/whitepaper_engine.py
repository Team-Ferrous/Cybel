"""Whitepaper generation helpers for innovation and closure artifacts."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from core.campaign.state_store import CampaignStateStore


class WhitepaperEngine:
    """Persists lightweight whitepaper summaries for campaign outputs."""

    def __init__(self, campaign_id: str, state_store: CampaignStateStore):
        self.campaign_id = campaign_id
        self.state_store = state_store

    def write(self, title: str, summary: str, findings: List[str]) -> Dict[str, Any]:
        whitepaper_id = str(uuid.uuid4())
        payload = {
            "whitepaper_id": whitepaper_id,
            "title": title,
            "summary": summary,
            "findings": list(findings),
            "created_at": time.time(),
        }
        self.state_store.insert_json_row(
            "whitepapers",
            campaign_id=self.campaign_id,
            payload=payload,
            extra={"whitepaper_id": whitepaper_id},
        )
        return payload
