"""Task packet execution and structured output enforcement."""

from __future__ import annotations

from typing import Any, Callable


class TaskPacketExecutor:
    """Execute task packets under explicit payload and result obligations."""

    RESULT_SCHEMAS = {
        "research": ("summary", "evidence"),
        "implementation": ("summary", "changed_files", "verification"),
        "verification": ("summary", "checks"),
        "architecture_review": ("summary", "risks", "recommendations"),
    }

    def __init__(self, state_store=None) -> None:
        self.state_store = state_store

    def execute(
        self,
        packet: dict[str, Any],
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        missing_packet_fields = [
            field
            for field in ("task_packet_id", "campaign_id", "objective")
            if not packet.get(field)
        ]
        packet_kind = str(packet.get("packet_kind") or packet.get("metadata", {}).get("packet_kind") or "implementation")
        required_result_fields = list(self.RESULT_SCHEMAS.get(packet_kind, self.RESULT_SCHEMAS["implementation"]))
        if missing_packet_fields:
            result = {
                "accepted": False,
                "status": "packet_invalid",
                "missing_packet_fields": missing_packet_fields,
                "missing_result_fields": required_result_fields,
                "packet_kind": packet_kind,
            }
            self._record(packet, "failed", result)
            return result

        handler_result = handler(packet) or {}
        missing_result_fields = [
            field for field in required_result_fields if field not in handler_result
        ]
        accepted = not missing_result_fields
        result = {
            "accepted": accepted,
            "status": "completed" if accepted else "obligations_missing",
            "packet_kind": packet_kind,
            "missing_packet_fields": [],
            "missing_result_fields": missing_result_fields,
            "result": handler_result,
            "required_result_fields": required_result_fields,
        }
        self._record(packet, "completed" if accepted else "failed", result)
        return result

    def _record(self, packet: dict[str, Any], status: str, result: dict[str, Any]) -> None:
        if self.state_store is None:
            return
        self.state_store.record_task_run(
            str(packet.get("task_packet_id") or "task_packet"),
            status=status,
            result={
                "campaign_id": packet.get("campaign_id"),
                "task_packet_id": packet.get("task_packet_id"),
                **result,
            },
        )
