"""Task announcement and overlap detection for collaborative planning."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from config.settings import COLLABORATION_CONFIG


@dataclass
class OverlapResult:
    local_task_id: str
    remote_task_id: str
    similarity_score: float
    overlap_type: str
    local_files: List[str]
    remote_files: List[str]


class TaskAnnouncer:
    """Announces tasks to peers and detects overlap with local workload."""

    def __init__(self, transport=None, overlap_threshold: Optional[float] = None):
        self.transport = transport
        self.local_tasks: List[Dict[str, object]] = []
        self.peer_announcements: Dict[str, List[Dict[str, object]]] = {}
        self.overlap_threshold = float(
            overlap_threshold
            if overlap_threshold is not None
            else COLLABORATION_CONFIG.get("overlap_threshold", 0.75)
        )

    @staticmethod
    def _run_async(coro) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            asyncio.run(coro)

    @staticmethod
    def _task_to_dict(task) -> Dict[str, object]:
        if isinstance(task, dict):
            instruction = str(task.get("instruction") or task.get("task") or "")
            task_id = str(task.get("id") or hash(instruction))
            context_files = list(task.get("context_files") or [])
            return {
                "id": task_id,
                "instruction": instruction,
                "context_files": context_files,
                "phase_id": str(task.get("phase_id") or ""),
                "campaign_id": str(task.get("campaign_id") or ""),
                "context_symbols": list(task.get("context_symbols") or []),
                "verification_targets": list(task.get("verification_targets") or []),
            }

        instruction = str(getattr(task, "instruction", ""))
        task_id = str(getattr(task, "id", hash(instruction)))
        context_files = list(getattr(task, "context_files", []) or [])
        return {
            "id": task_id,
            "instruction": instruction,
            "context_files": context_files,
            "phase_id": str(getattr(task, "phase_id", "") or ""),
            "campaign_id": str(getattr(task, "campaign_id", "") or ""),
            "context_symbols": list(getattr(task, "context_symbols", []) or []),
            "verification_targets": list(
                getattr(task, "verification_targets", []) or []
            ),
        }

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    @classmethod
    def _similarity(cls, left: str, right: str) -> float:
        a = cls._tokenize(left)
        b = cls._tokenize(right)
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return float(intersection / union) if union else 0.0

    def announce_tasks(self, tasks: List) -> List[Dict[str, object]]:
        self.local_tasks = [self._task_to_dict(task) for task in tasks]
        if self.transport is None:
            return self.local_tasks

        payload = {
            "tasks": self.local_tasks,
            "timestamp": time.time(),
        }
        self._run_async(
            self.transport.broadcast(
                {
                    "type": "task_announce",
                    "sender_id": self.transport.instance.instance_id,
                    "payload": payload,
                }
            )
        )
        return self.local_tasks

    def on_peer_announcement(self, peer_id: str, tasks: List[dict]) -> List[OverlapResult]:
        self.peer_announcements[peer_id] = [self._task_to_dict(task) for task in tasks]
        return self.detect_overlap(self.local_tasks, self.peer_announcements[peer_id])

    def detect_overlap(self, local_tasks: Iterable[dict], remote_tasks: Iterable[dict]) -> List[OverlapResult]:
        overlaps: List[OverlapResult] = []

        for local in local_tasks:
            local_instruction = str(local.get("instruction", ""))
            local_files = list(local.get("context_files") or [])
            local_file_set = set(local_files)

            for remote in remote_tasks:
                remote_instruction = str(remote.get("instruction", ""))
                remote_files = list(remote.get("context_files") or [])
                remote_file_set = set(remote_files)

                score = self._similarity(local_instruction, remote_instruction)
                file_overlap = len(local_file_set & remote_file_set)

                if score < 0.35 and file_overlap == 0:
                    continue

                if score >= self.overlap_threshold and file_overlap > 0:
                    overlap_type = "duplicate"
                elif file_overlap > 0 and score < self.overlap_threshold:
                    overlap_type = "conflicting"
                elif score >= self.overlap_threshold:
                    overlap_type = "complementary"
                else:
                    overlap_type = "dependent"

                overlaps.append(
                    OverlapResult(
                        local_task_id=str(local.get("id")),
                        remote_task_id=str(remote.get("id")),
                        similarity_score=round(score, 4),
                        overlap_type=overlap_type,
                        local_files=local_files,
                        remote_files=remote_files,
                    )
                )

        overlaps.sort(key=lambda item: item.similarity_score, reverse=True)
        return overlaps
