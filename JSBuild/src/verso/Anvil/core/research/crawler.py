"""Deterministic research frontier management."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from core.campaign.state_store import CampaignStateStore


class ResearchCrawler:
    """Maintains a persistent crawl frontier with explicit stop criteria."""

    def __init__(self, campaign_id: str, state_store: CampaignStateStore):
        self.campaign_id = campaign_id
        self.state_store = state_store

    def enqueue(
        self,
        url: str,
        topic: str,
        priority: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        now = time.time()
        frontier_id = str(uuid.uuid4())
        normalized_metadata = self._normalize_frontier_metadata(
            topic=topic,
            priority=priority,
            metadata=metadata,
        )
        self.state_store.execute(
            """
            INSERT INTO crawl_queue (
                campaign_id, frontier_id, url, topic, status, priority,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, url, topic) DO UPDATE SET
                status = 'queued',
                priority = CASE
                    WHEN excluded.priority > crawl_queue.priority
                    THEN excluded.priority
                    ELSE crawl_queue.priority
                END,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                self.campaign_id,
                frontier_id,
                url,
                topic,
                "queued",
                float(normalized_metadata["frontier_priority"]),
                json.dumps(normalized_metadata, default=str),
                now,
                now,
            ),
        )
        row = self.state_store.fetchone(
            """
            SELECT frontier_id
            FROM crawl_queue
            WHERE campaign_id = ? AND url = ? AND topic = ?
            LIMIT 1
            """,
            (self.campaign_id, url, topic),
        )
        return str((row or {})["frontier_id"])

    def ranked_frontier(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self.queued()
        ranked: list[dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row.get("metadata_json") or "{}")
            ranked.append(
                {
                    **row,
                    "metadata": metadata,
                    "frontier_priority": float(
                        metadata.get("frontier_priority", row.get("priority") or 0.0)
                    ),
                    "expected_information_gain": float(
                        metadata.get("expected_information_gain", 0.0)
                    ),
                    "expected_runtime_cost": float(
                        metadata.get("expected_runtime_cost", 0.0)
                    ),
                    "route_class": str(metadata.get("route_class") or ""),
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item["frontier_priority"]),
                -float(item["expected_information_gain"]),
                str(item.get("url") or ""),
            )
        )
        if limit is not None:
            return ranked[: max(0, int(limit))]
        return ranked

    def dequeue_batch(self, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.state_store.fetchall(
            """
            SELECT *
            FROM crawl_queue
            WHERE campaign_id = ? AND status = 'queued'
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
            """,
            (self.campaign_id, limit),
        )
        if rows:
            self.state_store.execute(
                """
                UPDATE crawl_queue
                SET status = 'in_progress', updated_at = ?
                WHERE frontier_id IN ({})
                """.format(", ".join(["?"] * len(rows))),
                (time.time(), *[row["frontier_id"] for row in rows]),
            )
        return rows

    def queued(self) -> list[dict[str, Any]]:
        return self.state_store.fetchall(
            """
            SELECT *
            FROM crawl_queue
            WHERE campaign_id = ? AND status = 'queued'
            ORDER BY priority DESC, created_at ASC
            """,
            (self.campaign_id,),
        )

    def queued_count(self) -> int:
        row = self.state_store.fetchone(
            "SELECT COUNT(*) AS count FROM crawl_queue WHERE campaign_id = ? AND status = 'queued'",
            (self.campaign_id,),
        )
        return int((row or {}).get("count", 0))

    def mark(
        self,
        frontier_id: str,
        status: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not metadata:
            self.state_store.execute(
                "UPDATE crawl_queue SET status = ?, updated_at = ? WHERE frontier_id = ?",
                (status, time.time(), frontier_id),
            )
            return
        current = self.state_store.fetchone(
            "SELECT metadata_json FROM crawl_queue WHERE frontier_id = ?",
            (frontier_id,),
        )
        payload = dict(metadata or {})
        if current is not None and current.get("metadata_json"):
            payload = {**json.loads(current["metadata_json"]), **payload}
        self.state_store.execute(
            "UPDATE crawl_queue SET status = ?, metadata_json = ?, updated_at = ? WHERE frontier_id = ?",
            (status, json.dumps(payload, default=str), time.time(), frontier_id),
        )

    def link(
        self,
        from_frontier_id: str,
        to_frontier_id: str,
        *,
        edge_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.state_store.execute(
            """
            INSERT INTO crawl_edges (
                campaign_id, from_frontier_id, to_frontier_id, edge_type, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.campaign_id,
                from_frontier_id,
                to_frontier_id,
                edge_type,
                json.dumps(metadata or {}, default=str),
                time.time(),
            ),
        )

    def stop_conditions(
        self,
        high_value_yield: list[float],
        coverage_complete: bool,
        unknowns_remaining: int,
        *,
        impact_deltas: list[float] | None = None,
        impact_threshold: float = 0.15,
    ) -> dict[str, Any]:
        low_yield = bool(high_value_yield[-3:]) and max(high_value_yield[-3:]) < 0.2
        impact_window = list(impact_deltas or [])
        impact_below_threshold = (
            bool(impact_window[-3:]) and max(impact_window[-3:]) < impact_threshold
        )
        frontier = self.ranked_frontier()
        remaining_frontier = len(frontier)
        route_mix: dict[str, int] = {}
        for item in frontier:
            route_class = str(item.get("route_class") or "unclassified")
            route_mix[route_class] = route_mix.get(route_class, 0) + 1
        frontier_priority_mean = (
            sum(float(item["frontier_priority"]) for item in frontier) / len(frontier)
            if frontier
            else 0.0
        )
        marginal_information_gain = (
            max(float(item["expected_information_gain"]) for item in frontier)
            if frontier
            else 0.0
        )
        frontier_quality_decay = (
            round(high_value_yield[-1] - high_value_yield[-3], 3)
            if len(high_value_yield) >= 3
            else 0.0
        )
        return {
            "frontier_below_threshold": remaining_frontier == 0,
            "yield_below_threshold": low_yield,
            "coverage_complete": coverage_complete,
            "unknowns_blocking": unknowns_remaining > 0,
            "impact_below_threshold": impact_below_threshold or not impact_window,
            "remaining_frontier": remaining_frontier,
            "frontier_priority_mean": round(frontier_priority_mean, 3),
            "marginal_information_gain": round(marginal_information_gain, 3),
            "route_mix": route_mix,
            "frontier_quality_decay": frontier_quality_decay,
            "stop_allowed": (
                remaining_frontier == 0
                and coverage_complete
                and unknowns_remaining == 0
                and (low_yield or not high_value_yield)
                and (impact_below_threshold or not impact_window)
            ),
        }

    @staticmethod
    def _normalize_frontier_metadata(
        *,
        topic: str,
        priority: float,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = dict(metadata or {})
        route_class = str(payload.get("route_class") or "browser_fetch")
        expected_information_gain = float(
            payload.get(
                "expected_information_gain",
                0.55 + min(0.2, len(topic.split()) * 0.03),
            )
        )
        expected_runtime_cost = float(payload.get("expected_runtime_cost", 1.0))
        reuse_bias = float(payload.get("reuse_bias", 0.0))
        frontier_priority = float(
            payload.get(
                "frontier_priority",
                max(
                    0.0,
                    float(priority) * 0.4
                    + expected_information_gain * 0.45
                    + reuse_bias * 0.15
                    - min(0.2, expected_runtime_cost * 0.05),
                ),
            )
        )
        payload.update(
            {
                "route_class": route_class,
                "expected_information_gain": round(expected_information_gain, 3),
                "expected_runtime_cost": round(expected_runtime_cost, 3),
                "frontier_priority": round(frontier_priority, 3),
                "reuse_bias": round(reuse_bias, 3),
            }
        )
        return payload
