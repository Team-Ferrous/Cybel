"""Generates measurement tooling tasks when instrumentation is missing."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List

from core.campaign.state_store import CampaignStateStore


class ToolingTaskFactory:
    """Creates tooling tasks instead of allowing unmeasured claims."""

    def __init__(self, campaign_id: str, state_store: CampaignStateStore):
        self.campaign_id = campaign_id
        self.state_store = state_store

    def create_measurement_task(
        self,
        title: str,
        objective: str,
        missing_metrics: List[str],
    ) -> Dict[str, Any]:
        tooling_task_id = str(uuid.uuid4())
        payload = {
            "tooling_task_id": tooling_task_id,
            "title": title,
            "objective": objective,
            "missing_metrics": list(missing_metrics),
            "created_at": time.time(),
        }
        self.state_store.insert_json_row(
            "tooling_tasks",
            campaign_id=self.campaign_id,
            payload=payload,
            id_field="tooling_task_id",
            id_value=tooling_task_id,
        )
        return payload
